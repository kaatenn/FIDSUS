"""
Instrumented FIDSUS client for Top-k audit.

Extends clientFIDSUS with audit hooks. Minimal invasion:
- Adds per-class loss tracking (when possible)
- Preserves all original training logic
"""

import copy
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from flcore.clients.clientFIDSUS import clientFIDSUS


class audited_clientFIDSUS(clientFIDSUS):
    """Instrumented FIDSUS client.

    Additional tracking:
    - Per-class training loss (tracked during train())
    - Model delta tracking for HiCS-style entropy estimation
    - Old head bias for delta computation
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # Store old head bias for HiCS-style delta computation
        self.old_head_bias = None
        self._per_class_losses = defaultdict(list)

    def train(self):
        """Override train to additionally track per-class losses."""
        trainloader, val_loader = self.load_train_data()
        start_time = time.time()
        self.aggregate_parameters(val_loader)
        self.clone_model(self.model, self.old_model)

        # Store old head bias before training
        self._store_old_head_bias()

        self.model.train()
        self.model_per.train()
        protos = defaultdict(list)
        protos_per = defaultdict(list)
        max_local_epochs = self.local_epochs

        for epoch in range(max_local_epochs):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                # Global model
                reg = self.model.base(x)
                output = self.model.head(reg)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Personalized model
                reg_per = self.model_per.base(x)
                output = self.model_per.head(reg_per)
                loss_per = self.loss(output, y)
                self.optimizer_per.zero_grad()
                loss_per.backward()
                self.optimizer_per.step(self.model_per.parameters(), self.device)

                # Track per-class losses (personalized model)
                with torch.no_grad():
                    per_sample_loss = F.cross_entropy(output, y, reduction="none")
                    for i, yy in enumerate(y):
                        self._per_class_losses[yy.item()].append(
                            per_sample_loss[i].item()
                        )

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(reg[i, :].detach().data)
                    protos_per[y_c].append(reg_per[i, :].detach().data)

        self.protos_g = agg_func(protos)
        self.protos_per = agg_func(protos_per)
        self.protos = aggregation(self.protos_g, self.protos_per)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def _store_old_head_bias(self):
        """Store head bias before training for delta computation."""
        for name, param in self.model_per.head.named_parameters():
            if "bias" in name:
                self.old_head_bias = param.data.clone().detach()

    def get_head_bias_delta(self):
        """Get delta of head bias (current - old)."""
        if self.old_head_bias is None:
            return None
        for name, param in self.model_per.head.named_parameters():
            if "bias" in name:
                return param.data - self.old_head_bias
        return None

    def get_per_class_losses(self) -> dict[int, list[float]]:
        """Get accumulated per-class losses, then clear."""
        result = dict(self._per_class_losses)
        self._per_class_losses = defaultdict(list)
        return result


# Reuse functions from clientFIDSUS
from flcore.clients.clientFIDSUS import MMD, agg_func, aggregation
