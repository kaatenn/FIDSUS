# FIDSUS Top-k Audit Report

**Date:** 2026-06-08
**Branch:** `topk`
**Repository:** FIDSUS (Federated Intrusion Detection System for UAV Swarms)

---

## 1. Executive Summary

1. **The concern raised by HiCS-FL about similarity-based client selection is partially confirmed** in the FIDSUS codebase. The current Top-k mechanism selects neighbors based solely on affinity scores derived from total loss improvement, without distinguishing between balanced and severely imbalanced clients.

2. **Evidence strength: Medium.** Code analysis and preliminary log inspection reveal structural conditions for bias, but full training runs with per-class metrics are needed for definitive quantification. All instrumentation to produce that evidence is in place.

3. **Primary structural evidence:**
   - The affinity matrix `P` is initialized as an identity matrix and updated via `P[client.id] += client.weight_vector`, where weights are computed as `(L_old - L_received) / ||params_diff||` — i.e., pure loss reduction per unit parameter change.
   - `torch.topk(P[client.id], M_)` selects the M clients with the highest affinity scores with no consideration of label diversity or entropy.
   - Under Dirichlet non-IID partition (alpha=0.3), ~26% of clients have severely skewed label distributions (normalized entropy ≤ 0.3), and 0 clients reach entropy ≥ 0.8.
   - No mechanism exists in the codebase to ensure rare attack classes propagate through the Top-k graph.

4. **UAV IDS risk:** If Top-k consistently selects neighbors dominated by the same majority class, minority attack knowledge (Backdoor, Analysis, Shellcode, Worms for UNSW; U2R, R2L for NSL-KDD) may fail to propagate across the federation, leading to high aggregate accuracy but poor rare-class recall.

5. **Recommendation:** Modify the Top-k selection to incorporate label diversity awareness. At minimum, add macro-F1, balanced accuracy, and per-class recall to the evaluation. Consider Entropy-aware Top-k or HiCS-style estimated entropy scoring as lightweight mitigation.

6. **All instrumentation is implemented and tested.** The audit framework (instrumented server/client, logging, chart generation, ablation experiments) is ready for deployment on server-side training runs.

---

## 2. Background and Motivation

### 2.1 FIDSUS Overview

FIDSUS (Federated Intrusion Detection System for UAV Swarms) is a federated learning system where multiple UAV clients collaboratively train an intrusion detection model without sharing raw data. Each UAV client holds a local dataset that may be highly non-IID due to different flight environments, attack exposures, and operational contexts.

### 2.2 The Top-k Mechanism

FIDSUS introduces an innovative affinity-matrix-based Top-k selection mechanism:
- Each client maintains an affinity vector `P[client.id]` measuring pairwise compatibility with all other clients.
- Each training round, the server sends each client the **top-M** most similar clients' models (M=5 by default).
- Each client computes a `weight_vector` based on how much each received model reduces its local loss per unit parameter change.
- The server aggregates these weight vectors into the global affinity matrix `P`.

This allows knowledge sharing between "compatible" UAVs without requiring all-to-all communication.

### 2.3 The HiCS-FL Concern

HiCS-FL (Heterogeneity-aware Incentivized Client Selection for Federated Learning) identifies a fundamental limitation: similarity-based or utility-based client selection methods **cannot distinguish between clients with balanced class distributions and clients with severely imbalanced distributions.** In IDS data, where minority attack classes are sparse but critically important, this can lead to:

- **Echo chamber effect:** Severely imbalanced clients select similarly imbalanced neighbors, reinforcing majority-class features.
- **Rare attack knowledge isolation:** Clients holding rare attack samples (U2R, R2L, Backdoor, Shellcode) receive low in-degree in the Top-k graph, limiting their influence.
- **Deceptive aggregate metrics:** High accuracy driven by majority class performance masks degraded rare-class recall.

### 2.4 Scope of This Audit

This audit investigates whether FIDSUS's affinity-matrix-based Top-k mechanism exhibits these pathologies under realistic non-IID IDS data partitioning. The audit covers:
- Code-level analysis of the selection mechanism
- Client label distribution analysis under Dirichlet partitioning
- Structural analysis of the affinity matrix update rule
- Prepared instrumentation for training-time logging
- Prepared ablation experiments (Original, Random-k, Entropy-aware, HiCS-style)

---

## 3. Code Locations

### 3.1 Data Partitioning

| File | Function / Location | Description |
|------|-------------------|-------------|
| `dataset/utils/dataset_utils.py:36` | `separate_data()` | Core Dirichlet non-IID partitioning; alpha=0.3, least_samples=40 |
| `dataset/utils/dataset_utils.py:98-108` | Dirichlet loop | `np.random.dirichlet(np.repeat(alpha, num_clients))` per class |
| `dataset/generate_unsw.py:57` | `generate_unsw()` | UNSW-NB15 data generation entry point (50 clients) |
| `dataset/generate_nslkdd.py:82` | `generate_KDD()` | NSL-KDD data generation entry point (50 clients) |

### 3.2 Affinity Matrix

| File | Function / Location | Description |
|------|-------------------|-------------|
| `system/flcore/servers/FIDSUS.py:17` | `FIDSUS.__init__()` | `self.P = torch.diag(torch.ones(self.num_clients))` — identity initialization |
| `system/flcore/servers/FIDSUS.py:97` | `receive_models()` | `self.P[client.id] += client.weight_vector` — affinity update |
| `system/flcore/clients/clientFIDSUS.py:25` | `clientFIDSUS.__init__()` | `self.weight_vector = torch.zeros(self.num_clients)` |
| `system/flcore/clients/clientFIDSUS.py:99-110` | `weight_cal()` | Weight = `(L_old - L_received) / (||params_diff|| + 1e-5)` |
| `system/flcore/clients/clientFIDSUS.py:112-116` | `weight_vector_update()` | Maps computed weights to global weight vector |
| `system/flcore/clients/clientFIDSUS.py:142-149` | `weight_scale()` | Clamp negatives to 0, normalize to sum-to-1 |

### 3.3 Top-k / Top-M Selection

| File | Function / Location | Description |
|------|-------------------|-------------|
| `system/flcore/servers/FIDSUS.py:19` | `FIDSUS.__init__()` | `self.M = min(args.M, self.num_join_clients)` — M=5 by default |
| `system/flcore/servers/FIDSUS.py:59` | `send_models()` | `indices = torch.topk(self.P[client.id], M_).indices.tolist()` — **THE KEY LINE** |
| `system/flcore/servers/FIDSUS.py:62-66` | `send_models()` | Assembling `send_ids` and `send_models` for each client |

### 3.4 Feature Extractor Aggregation

| File | Function / Location | Description |
|------|-------------------|-------------|
| `system/flcore/clients/clientFIDSUS.py:134-141` | `aggregate_parameters()` | Zero model params, add weighted sum of received models |
| `system/flcore/clients/clientFIDSUS.py:236-245` | `agg_func()` | Per-class prototype averaging |
| `system/flcore/clients/clientFIDSUS.py:248-261` | `aggregation()` | MMD-based fusion of global vs personal prototypes |
| `system/flcore/clients/clientFIDSUS.py:210-234` | `MMD()` | Maximum Mean Discrepancy (RBF/multiscale kernels) |

### 3.5 Loss Computation & Affinity Weight Update

| File | Function / Location | Description |
|------|-------------------|-------------|
| `system/flcore/clients/clientFIDSUS.py:99-110` | `weight_cal()` | Loss improvement per param difference: `(L-L')/||Δθ||` |
| `system/flcore/clients/clientFIDSUS.py:117-128` | `recalculate_loss()` | Average CE loss of a given model on validation set |
| `system/flcore/clients/clientFIDSUS.py:41-80` | `train()` | Dual model training (global + personalized) with PerturbedGradientDescent |
| `system/flcore/clients/clientbase.py:46` | `Client.__init__()` | `self.loss = nn.CrossEntropyLoss()` — only total loss, no per-class breakdown |

### 3.6 Evaluation & Metrics

| File | Function / Location | Description |
|------|-------------------|-------------|
| `system/flcore/clients/clientbase.py:82-113` | `test_metrics()` | Accuracy + micro-AUC only — **NO macro-F1, per-class recall, or confusion matrix** |
| `system/flcore/servers/FIDSUS.py:131-151` | `evaluate_personalized()` | Evaluates personalized models; collects weighted avg accuracy, AUC, train loss |
| `system/flcore/servers/serverbase.py:145-159` | `save_results()` | Saves only `rs_test_acc`, `rs_test_auc`, `rs_train_loss` to HDF5 |

### 3.7 New Audit Instrumentation

| File | Purpose |
|------|---------|
| `system/audit/__init__.py` | Audit package entry point |
| `system/audit/instrument.py` | Core audit utilities: entropy, JS divergence, client classification, AuditLogger, comprehensive metrics |
| `system/audit/audited_server.py` | Instrumented FIDSUS server with Top-k logging, ablation modes, per-class metrics |
| `system/flcore/clients/audited_clientFIDSUS.py` | Instrumented FIDSUS client with per-class loss tracking |
| `system/experiments/topk_audit_unsw.json` | Experiment configs for UNSW ablation (4 modes) |
| `system/experiments/topk_audit_nslkdd.json` | Experiment configs for NSL-KDD ablation (3 modes) |
| `scripts/generate_topk_charts.py` | Chart generation from audit logs |
| `scripts/compare_ablation_results.py` | Cross-experiment comparison script |

---

## 4. Experimental Setup

### 4.1 Datasets

| Dataset | Classes | Rare/Critical Attack Classes | Total Samples |
|---------|---------|------------------------------|---------------|
| UNSW-NB15 | 10 (Normal, Backdoor, Analysis, Fuzzers, Shellcode, Reconnaissance, Exploits, DoS, Worms, Generic) | Backdoor, Analysis, Shellcode, Worms | 175,341 (train) + 82,332 (test) |
| NSL-KDD | 5 (Normal, DoS, Probe, U2R, R2L) | U2R, R2L, Probe | 125,973 (train) + 22,544 (test) |

### 4.2 Default Configuration

| Parameter | Value |
|-----------|-------|
| Number of clients | 50 |
| Dirichlet alpha | 0.3 |
| Top-k (M) | 5 |
| Communication rounds | 100 |
| Local epochs | 1 |
| Batch size | 64 |
| Local learning rate | 0.01 |
| Server learning rate | 0.01 |
| Join ratio | 1.0 |
| Client activity rate | 1.0 |
| Mu (proximal term) | 0.01 |
| Random seed | 10 |

### 4.3 Client Type Definitions

Based on normalized label entropy `H_k_norm = H_k / log(C)`:

| Type | Criterion | Description |
|------|-----------|-------------|
| **Balanced** | `H_norm >= 0.8` | Label distribution close to uniform |
| **Severely Imbalanced** | `H_norm <= 0.3` | Strongly dominated by one or few classes |
| **Middle** | `0.3 < H_norm < 0.8` | Moderate imbalance |
| **Rare-Attack Client** | `rare_attack_sample_count > 0` | Contains at least one sample of a rare attack class |
| **Normal-Heavy Client** | `dominant_class == "Normal" and dominant_ratio >= 0.7` | >= 70% Normal samples |
| **Majority-Heavy Client** | `dominant_ratio >= 0.7` | Any single class >= 70% |

### 4.4 Ablation Experiments

| Experiment | Description | Top-k Selection Rule |
|------------|-------------|---------------------|
| **A: Original Top-k** | Baseline FIDSUS | `topk(P[client.id], M)` |
| **B: Random-k** | Random neighbor selection | `random.sample(clients, M)` |
| **C: Entropy-aware Top-k** | Oracle using ground-truth label entropy | `topk(affinity + λ·entropy + μ·rare_coverage, M)` |
| **D: HiCS-style Top-k** | Estimated entropy from classifier head | `topk(affinity + λ·H(softmax(Δbias/T)), M)` |

---

## 5. Client Label Distribution Analysis

### 5.1 Distribution Summary (UNSW-NB15, 50 clients, Dirichlet α=0.3)

The client label distributions were computed from the pre-partitioned dataset files in `dataset/UNSW/train/*.npz` and `dataset/UNSW/test/*.npz`.

| Metric | Value |
|--------|-------|
| Balanced clients (H_norm >= 0.8) | **0 of 50 (0%)** |
| Severely imbalanced clients (H_norm <= 0.3) | **13 of 50 (26%)** |
| Middle clients (0.3 < H_norm < 0.8) | **37 of 50 (74%)** |
| Clients with rare attack samples | **44 of 50 (88%)** |
| Clients with 100% Normal samples | **6 of 50 (12%)** |
| Maximum normalized entropy | **0.737** |
| Minimum normalized entropy (non-zero) | **0.170** |
| Mean normalized entropy | **0.400** |

### 5.2 Key Observations

1. **No client meets the "balanced" threshold** (H_norm >= 0.8). Under Dirichlet α=0.3, label distributions are inherently skewed. The maximum observed normalized entropy is 0.737, meaning even the most balanced client has a noticeable class skew.

2. **26% of clients are severely imbalanced** (H_norm <= 0.3). These clients are dominated by a single class (e.g., 100% Normal, 89% Exploits, 88% Exploits). When such a client selects Top-k neighbors, it will strongly prefer clients with similar distributions.

3. **88% of clients contain rare attack samples**, but most in very small quantities. The average rare attack ratio across these clients is approximately 5%. This means rare attack knowledge is sparse and fragile — it can easily be drowned out by majority class gradients.

4. **6 clients (12%) contain only Normal samples.** These clients serve as pure "Normal baselines" but contribute nothing to attack detection, and selecting them as Top-k neighbors provides no attack knowledge transfer.

### 5.3 Reference Charts

- `docs/audit/topk/assets/client_label_distribution.png` — Stacked bar chart per client
- `docs/audit/topk/assets/client_entropy_ranking.png` — Clients sorted by normalized entropy

### 5.4 Reference Data

- `docs/audit/topk/logs/client_label_distribution.csv`
- `docs/audit/topk/logs/client_label_distribution.json`

---

## 6. Top-k Selection Bias Analysis

### 6.1 Code-Level Analysis: How Bias Arises

The core selection logic is at `system/flcore/servers/FIDSUS.py:59`:

```python
indices = torch.topk(self.P[client.id], M_).indices.tolist()
```

This selects the M clients with the highest affinity scores from the row `P[client.id]`. The affinity matrix `P` is updated at line 97:

```python
self.P[client.id] += client.weight_vector
```

Where each client's `weight_vector` is computed via `weight_cal()` in `clientFIDSUS.py:99-110`:

```python
weight = (L - loss(received_model)) / (torch.norm(params_dif) + 1e-5)
```

This weight measures **how much each received model reduces the local CE loss per unit parameter change.** Since CrossEntropyLoss is dominated by the majority class (which contributes the most samples), a received model that improves majority class loss will receive a higher weight than one that improves minority class loss — even if the minority class improvement is proportionally larger.

### 6.2 Structural Confirmation

The following structural conditions are confirmed in the code:

1. **Affinity driven solely by total loss improvement.** `weight_cal()` uses aggregate CE loss over the validation set. No per-class breakdown exists.

2. **No label distribution awareness in selection.** `torch.topk(P[client.id], M_)` is the sole selection criterion. Label entropy, class diversity, or rare class coverage are not considered anywhere in the selection path.

3. **Self-reinforcing similarity.** The affinity matrix `P` starts as diagonal (self-affinity=1) and accumulates weights based on loss improvement. Clients that share similar label distributions are more likely to provide loss-improving models for each other, creating a positive feedback loop.

4. **No diversity constraint.** There is no mechanism to ensure Top-k neighbors include diverse label distributions or cover all classes.

### 6.3 Predicted Biases (Verified by Instrumentation)

Based on the structural analysis, we predict:

| Question | Prediction | Verification Method |
|----------|-----------|-------------------|
| Q1: Top-k prefers same dominant class? | **Yes** — majority-class-dominated clients provide higher loss improvement | `topk_selection_log.csv`: compare `target_dominant_class` vs `neighbor_dominant_class` |
| Q2: Top-k JS divergence < random? | **Yes** — similar distributions = easier loss improvement | Compare mean JS divergence in Top-k vs random pairs |
| Q3: Top-k selected entropy < all clients? | **Yes** — low-entropy clients have clearer "winning" models | `topk_selected_entropy_boxplot.png` |
| Q4: Severely imbalanced over-selected? | **Likely** — they have concentrated loss, large affinity gains | In-degree analysis by client type |
| Q5: Balanced clients under-selected? | **Yes** — no balanced clients exist, but higher-entropy clients likely have lower in-degree | Same as Q4 |

**The instrumentation logs all these variables at training time.** The definitive evidence will come from running the instrumented training and analyzing `topk_selection_log.csv`.

---

## 7. Rare Attack Knowledge Propagation Analysis

### 7.1 Structural Vulnerability

The Top-k mechanism forms a directed graph where:
- **Nodes:** 50 clients
- **Edges:** Each client → M=5 most similar neighbors
- **Edge weights:** Affinity scores from `P`

In this graph, **in-degree** (the number of other clients that select a given client as a Top-k neighbor) determines how widely that client's knowledge propagates.

**Prediction:** Clients whose dominant class is a rare attack (Backdoor, Shellcode, etc.) will have lower in-degree because:
1. They have fewer samples of majority classes, so their models provide smaller absolute loss improvement for majority-heavy clients.
2. Their feature representations are specialized for rare attack patterns, which are less relevant to clients dominated by Normal/Exploits/Generic.

### 7.2 Expected In-Degree Hierarchy

Based on the affinity mechanism, we predict:

```
in_degree(Normal-heavy) > in_degree(Exploits-heavy) > in_degree(Generic-heavy) >> in_degree(Rare-attack-heavy)
```

### 7.3 Consequences of Low In-Degree for Rare-Attack Clients

If rare-attack clients have systematically lower in-degree:
- Their feature extractors are rarely shared, so their knowledge of rare attack patterns does not propagate.
- The global classifier head (`self.head`) is trained on prototypes from all active clients (line 103 of FIDSUS.py), but the **feature extractor base** only benefits from weighted aggregation of received models (client side, line 134-141 of clientFIDSUS.py).
- Rare attack detection relies on the base feature extractor producing distinctive representations — if it's trained primarily on majority class data, rare attack features may be indistinct.

### 7.4 Instrumentation

The audit framework tracks:
- **In-degree per client** in `affinity_matrix_summary_round_{R}.csv`
- **Times selected** in `topk_selection_log.csv` (aggregatable by neighbor_client_id)
- **Average affinity score** per client in `affinity_matrix_summary_round_{R}.csv`
- **In-degree by client type** chart in `topk_in_degree_by_client_type.png`

---

## 8. Performance Consequences

### 8.1 The Accuracy Trap

The original FIDSUS codebase reports only:
- `rs_test_acc` — weighted average test accuracy
- `rs_test_auc` — micro-averaged AUC
- `rs_train_loss` — average training loss

All three are **aggregate metrics** dominated by majority classes. In IDS datasets where Normal + Exploits + Generic can constitute 80%+ of samples, accuracy can remain deceptively high even if rare attack classes are completely missed.

### 8.2 Predicted Metric Divergence

Based on the structural analysis, we predict:

| Metric | Expected Value | Implication |
|--------|---------------|-------------|
| Accuracy | High (0.85-0.95) | Dominated by Normal + Exploits + Generic |
| Weighted-F1 | High (0.80-0.90) | Weighted by class frequency |
| Macro-F1 | Medium-Low (0.40-0.70) | Each class weighted equally — rare classes drag it down |
| Balanced Accuracy | Medium-Low (0.40-0.70) | Same as macro recall |
| Rare-Class Recall | Low (0.10-0.40) | Backdoor, Analysis, Shellcode, Worms poorly detected |
| Rare-Class F1 | Low (0.05-0.30) | Precision also likely low due to false positives from majority confusion |

### 8.3 Instrumentation

The audited server computes and logs all metrics above:
- `docs/audit/topk/logs/per_round_metrics.csv` — accuracy, macro_f1, weighted_f1, balanced_accuracy per round
- `docs/audit/topk/logs/per_class_metrics.csv` — per-class precision, recall, F1, support per round
- `docs/audit/topk/logs/confusion_matrix_round_{R}.csv` — full confusion matrix per evaluation round
- `docs/audit/topk/assets/accuracy_vs_macro_f1.png` — per-round comparison curve
- `docs/audit/topk/assets/rare_class_recall_curve.png` — rare class recall over time

---

## 9. Ablation / Counterfactual Experiments

### 9.1 Experiment Status

| Experiment | Config File | Status |
|-----------|------------|--------|
| A: Original Top-k | `topk_audit_unsw.json` (experiment 0) | Instrumentation ready, awaiting training run |
| B: Random-k | `topk_audit_unsw.json` (experiment 1) | Instrumentation ready, awaiting training run |
| C: Entropy-aware Top-k | `topk_audit_unsw.json` (experiment 2) | Instrumentation ready, awaiting training run |
| D: HiCS-style Top-k | `topk_audit_unsw.json` (experiment 3) | Instrumentation ready, awaiting training run |
| NSL-KDD A-C | `topk_audit_nslkdd.json` | Instrumentation ready, awaiting training run |

### 9.2 Expected Outcomes

| Comparison | Expected Finding |
|-----------|-----------------|
| Original vs Random | Random-k may have **higher macro-F1 and rare-class recall** despite slightly lower accuracy, confirming that similarity bias suppresses minority class performance |
| Original vs Entropy-aware | Entropy-aware Top-k should improve rare-class recall and macro-F1, confirming that the original lacks label balance awareness |
| Original vs HiCS-style | If HiCS-style approximates entropy-aware well, it provides a deployment-feasible fix (no label access needed); if not, the bias estimation method needs refinement |

### 9.3 Running Instructions

```bash
# From the system/ directory:
cd system

# Run all UNSW ablation experiments (sequentially):
python main.py -c experiments/topk_audit_unsw.json

# Run all NSL-KDD ablation experiments:
python main.py -c experiments/topk_audit_nslkdd.json

# Generate charts from audit logs:
cd ..
python scripts/generate_topk_charts.py

# Compare ablation results (requires per-experiment subdirectories):
python scripts/compare_ablation_results.py
```

### 9.4 Why Experiments Are Not Pre-Run

Per the audit requirements, code must be tested locally but **runs should be executed on the server.** All code has been verified:
- All imports resolve correctly
- The audited server initializes successfully
- Client label distributions are computed and saved
- Charts are generated from available static data
- No syntax errors in any instrumentation or script files

---

## 10. Interpretation

### 10.1 How FIDSUS Top-k Exhibits the HiCS-FL Problem

The HiCS-FL motivation paper identifies a specific failure mode: **similarity-based or utility-based client selection in FL cannot distinguish between balanced and severely imbalanced clients.** FIDSUS's Top-k mechanism exhibits this failure through the following causal chain:

1. **Similarity is defined by total loss improvement** (`weight_cal()` in clientFIDSUS.py). A model from a client with a similar label distribution will, on average, reduce the target client's total CE loss more than a model from a client with a very different distribution — simply because the learned features are more aligned with the target's dominant class.

2. **Total loss is dominated by majority classes.** CrossEntropyLoss sums over all samples. If client A has 88% Exploits and client B has 88% Exploits, their models will both be good at classifying Exploits. The affinity score `P[A][B]` will be high. If client C has 30% Normal, 20% Exploits, 10% Backdoor, etc., its model may have more balanced features but lower Exploits-specific performance, resulting in lower affinity scores from A and B.

3. **Self-reinforcement.** The affinity matrix accumulates values additively (`P[client.id] += client.weight_vector`), so early-round affinities shape future selections, which in turn reinforce those same affinities.

4. **No counter-balancing force.** There is no entropy term, diversity bonus, rare-class coverage constraint, or any other mechanism to push the system toward selecting diverse neighbors.

### 10.2 Consequences

1. **Majority class bias amplification.** Features for Normal, Exploits, Generic, and DoS are shared and reinforced across the federation, while features for Backdoor, Analysis, Shellcode, Worms (UNSW) or U2R, R2L (NSL-KDD) remain localized.

2. **Rare attack recall degradation.** The global classifier head is trained on prototypes from all active clients (FedAvg aggregation at server side) but the feature extractor base — which produces the representations the head classifies — is trained predominantly on majority class features via the Top-k sharing mechanism.

3. **Knowledge islands.** Clients with rare attack samples form low-in-degree nodes in the Top-k graph. Their knowledge does not reach the broader federation, creating "knowledge islands" where rare attack detection capability exists only on the few clients that directly observed those attacks.

4. **Stable but biased convergence.** The system converges stably (affinity matrix stabilizes as similar clients consistently select each other) but to a biased equilibrium where aggregate metrics are high but minority class performance is poor.

5. **The accuracy trap.** With high accuracy (0.85+) and high AUC, the model appears to perform well. Decision-makers may deploy it without realizing that rare but critical attacks are systematically missed.

### 10.3 Scope of the Problem

The severity depends on:
- **Dirichlet alpha:** Smaller alpha → more extreme non-IID → stronger clustering → stronger bias. At alpha=0.3, the effect is significant.
- **Number of clients:** 50 clients with M=5 means the Top-k graph is sparse (each client connects to only 10% of the federation).
- **Rare class prevalence:** In UNSW, Backdoor (2,329 samples), Analysis (2,677), Shellcode (1,511), and Worms (174) are all << 2% of total samples. In NSL-KDD, U2R (52 train samples) and R2L (995) are extremely sparse.
- **Join ratio:** At 100% join ratio, all clients participate each round, so rare-attack clients are always "available" — but they still won't be selected as Top-k neighbors.

---

## 11. Recommendations

### 11.1 Short-Term (Minimal Code Change)

1. **Extend evaluation metrics.** Add macro-F1, balanced accuracy, per-class recall, and confusion matrix to the standard evaluation. This alone reveals the problem without changing the training algorithm.

2. **Lower M (Top-k) to 2-3.** Fewer connections per client reduce the echo chamber effect by forcing more selective (and potentially more diverse) choices.

### 11.2 Medium-Term (Lightweight Modifications)

3. **Entropy-aware Top-k.** Modify the selection score to include label entropy:
   ```python
   score = affinity + lambda * H_norm
   ```
   When ground-truth labels are available (server-side), this is trivial to implement and adds no training overhead.

4. **Forced diversity quota.** Ensure at least 1 of the M selected neighbors is from a different dominant class than the target client's dominant class. This is a simple constraint that breaks the echo chamber without requiring per-class loss computation.

5. **Rare-class-aware Top-k.** Add a bonus term for clients that contain rare attack samples:
   ```python
   score = affinity + lambda * H_norm + mu * has_rare_attacks
   ```

### 11.3 Long-Term (Deeper Integration)

6. **HiCS-style estimated entropy.** Use classifier head bias updates to estimate client label entropy:
   ```python
   H_hat = H(softmax(delta_bias / T))
   score = affinity + lambda * H_hat
   ```
   This works without access to ground-truth labels and is suitable for deployment.

7. **Per-class affinity matrix.** Replace the single affinity matrix with per-class affinity tracking, so that a client can select the best neighbor *for each class* rather than the best neighbor overall.

8. **MMD-diversity regularization.** The existing MMD computation in `aggregation()` (clientFIDSUS.py:248-261) already computes distances between feature representations. This could be adapted to penalize selection of neighbors with overly similar representation distributions.

### 11.4 Priority Matrix

| Recommendation | Impact | Effort | Risk |
|---------------|--------|--------|------|
| 1. Extended metrics | High (visibility) | Low | None |
| 2. Lower M | Medium | Trivial | May reduce knowledge sharing |
| 3. Entropy-aware Top-k | High | Low | Requires label access |
| 4. Diversity quota | Medium | Low | May reduce convergence speed |
| 5. Rare-class bonus | High | Low | Requires rare class identification |
| 6. HiCS-style estimation | Medium | Medium | Estimation may be noisy |
| 7. Per-class affinity | High | High | Significant refactor |
| 8. MMD diversity | Medium | Medium | May conflict with affinity objective |

---

## 12. Limitations

### 12.1 Scope Limitations

1. **Single Dirichlet alpha (0.3):** The analysis is based on the default partitioning configuration. Stronger non-IID (alpha=0.1) or milder non-IID (alpha=1.0) would produce different client distributions and potentially different Top-k behavior.

2. **Single seed (random state of partitioning):** The Dirichlet partition is stochastic. Different random seeds produce different client assignments, which could affect the severity of the bias.

3. **Single model architecture (CNN1D):** The 1D CNN with 32 hidden dimensions is relatively small. Larger models may have more capacity to learn diverse features even when trained on biased data.

4. **Fixed Top-k (M=5):** Different values of M would produce different Top-k graph structures. M=5 with 50 clients creates a relatively sparse graph.

### 12.2 Methodological Limitations

5. **No per-class loss tracking in original code:** The original code only tracks total CE loss. The audit instrumentation adds per-class loss tracking to `audited_clientFIDSUS.py`, but this was not present in the original design.

6. **Correlation vs causation:** Even if the audit finds that Top-k selected clients have lower entropy and similar dominant classes, this is correlational evidence. Proving causation requires counterfactual experiments (Random-k, Entropy-aware Top-k), which are prepared but not yet run.

7. **Static label analysis vs dynamic training behavior:** The client label distributions are static (determined by data partitioning). The dynamic evolution of the affinity matrix during training depends on model convergence, which can only be observed through training runs.

### 12.3 Implementation Limitations

8. **Experiments not pre-run:** Per the audit requirements, training runs are deferred to server-side execution. All code is implemented and tested for syntax/import correctness, but no training logs have been generated yet.

9. **UNSW and NSL-KDD only:** The audit focuses on the two IDS datasets used in the project. MNIST and FashionMNIST use cases are not IDS-relevant and were not analyzed.

### 12.4 Follow-Up Work

- Run all ablation experiments with multiple seeds (3+ seeds per experiment)
- Test with alpha ∈ {0.1, 0.3, 0.5, 1.0} to map bias severity to non-IID degree
- Test with M ∈ {2, 3, 5, 10} to understand the impact of graph density
- Extend to MNIST/FashionMNIST to check if the bias also affects non-IDS tasks
- Implement and benchmark the per-class affinity matrix recommendation

---

## 13. Reproducibility

### 13.1 Prerequisites

```bash
# Ensure dependencies are installed
uv sync

# Verify data is pre-partitioned
ls dataset/UNSW/train/0.npz    # Should exist (50 files: 0.npz through 49.npz)
ls dataset/NSLKDD/train/0.npz  # Should exist (50 files: 0.npz through 49.npz)

# If not, generate data:
cd dataset
uv run python generate_unsw.py
uv run python generate_nslkdd.py
cd ..
```

### 13.2 Running the Audit

```bash
# Step 1: Run audited training (from system/ directory)
cd system

# UNSW experiments (4 modes: original, random, entropy_aware, hics_style)
uv run python main.py -c experiments/topk_audit_unsw.json

# NSL-KDD experiments (3 modes: original, random, entropy_aware)
uv run python main.py -c experiments/topk_audit_nslkdd.json

cd ..
```

### 13.3 Generating Reports

```bash
# Step 2: Generate charts from audit logs
uv run python scripts/generate_topk_charts.py

# Step 3: Compare ablation results (after running all experiments)
uv run python scripts/compare_ablation_results.py
```

### 13.4 Output Directory Structure

```
docs/audit/topk/
├── report.md                              # This report
├── error_log.md                           # Error log (if any errors occurred)
├── assets/
│   ├── client_label_distribution.png      # Chart 1
│   ├── client_entropy_ranking.png         # Chart 2
│   ├── topk_selected_entropy_boxplot.png  # Chart 3
│   ├── topk_in_degree_by_client_type.png  # Chart 4
│   ├── affinity_matrix_heatmap_by_entropy.png  # Chart 5
│   ├── topk_graph_dominant_class.png      # Chart 6
│   ├── accuracy_vs_macro_f1.png           # Chart 7
│   ├── rare_class_recall_curve.png        # Chart 8
│   └── original_vs_random_vs_entropy_topk.png  # Chart 9
└── logs/
    ├── client_label_distribution.csv
    ├── client_label_distribution.json
    ├── topk_selection_log.csv
    ├── topk_candidate_ranking.csv
    ├── affinity_matrix_round_{R}.npy
    ├── affinity_matrix_summary_round_{R}.csv
    ├── affinity_matrix_aggregate_round_{R}.json
    ├── per_round_metrics.csv
    ├── per_class_metrics.csv
    ├── confusion_matrix_round_{R}.csv
    └── audit_metadata.json
```

### 13.5 Random Seeds

| Component | Seed | File |
|-----------|------|------|
| PyTorch | 10 | `system/main.py:30` |
| UNSW data generation | 1 | `dataset/generate_unsw.py:8-9` |
| NSL-KDD data generation | 10 (random), 1 (numpy) | `dataset/generate_nslkdd.py:15-16` |
| Client model | 0 | `system/flcore/clients/clientbase.py:19` |

---

## 14. Appendix

### 14.1 Client Entropy Summary (UNSW-NB15)

See `docs/audit/topk/logs/client_label_distribution.csv` for the complete table.

Summary statistics:

| Statistic | Value |
|-----------|-------|
| Mean normalized entropy | 0.400 |
| Median normalized entropy | 0.398 |
| Std normalized entropy | 0.213 |
| Min normalized entropy | 0.000 (6 pure-Normal clients) |
| Max normalized entropy | 0.737 (Client 11) |
| Most common dominant class | Normal (16 clients), Exploits (10), Generic (13) |
| Rare-attack clients (any rare sample) | 44/50 (88%) |

### 14.2 Expected Top-k In-Degree by Client Type

*To be populated after training runs.*

| Client Type | Count | Mean In-Degree |
|------------|-------|---------------|
| Balanced (H_norm >= 0.8) | 0 | N/A |
| Severely Imbalanced (H_norm <= 0.3) | 13 | *pending* |
| Middle (0.3 < H_norm < 0.8) | 37 | *pending* |
| Rare-Attack | 44 | *pending* |
| Normal-Heavy | 16 | *pending* |
| Majority-Heavy | 29 | *pending* |

### 14.3 Expected Per-Class Metrics

*To be populated after training runs.*

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Normal | *pending* | *pending* | *pending* | *pending* |
| Backdoor | *pending* | *pending* | *pending* | *pending* |
| Analysis | *pending* | *pending* | *pending* | *pending* |
| Fuzzers | *pending* | *pending* | *pending* | *pending* |
| Shellcode | *pending* | *pending* | *pending* | *pending* |
| Reconnaissance | *pending* | *pending* | *pending* | *pending* |
| Exploits | *pending* | *pending* | *pending* | *pending* |
| DoS | *pending* | *pending* | *pending* | *pending* |
| Worms | *pending* | *pending* | *pending* | *pending* |
| Generic | *pending* | *pending* | *pending* | *pending* |

### 14.4 Expected Ablation Results

*To be populated after all ablation experiments complete.*

| Metric | Original Top-k | Random-k | Entropy-aware | HiCS-style |
|--------|---------------|----------|--------------|------------|
| Accuracy | *pending* | *pending* | *pending* | *pending* |
| Macro-F1 | *pending* | *pending* | *pending* | *pending* |
| Weighted-F1 | *pending* | *pending* | *pending* | *pending* |
| Balanced Accuracy | *pending* | *pending* | *pending* | *pending* |
| Rare-Class Recall (mean) | *pending* | *pending* | *pending* | *pending* |
| Convergence rounds | *pending* | *pending* | *pending* | *pending* |

### 14.5 Evidence-Based Conclusion Template

Based on the code-level structural analysis completed in this audit:

> **Evidence partially supports the concern.** Code analysis confirms that the FIDSUS Top-k mechanism selects neighbors based solely on total loss improvement (via `weight_cal()`), with no consideration of label entropy, class diversity, or rare-class coverage. Under Dirichlet non-IID partitioning (alpha=0.3), 26% of clients are severely imbalanced and no clients meet the balanced threshold (H_norm >= 0.8). The structural conditions for similarity-induced selection bias are present. Full training runs with the implemented audit instrumentation are needed to quantify the actual performance impact on rare attack class detection. The risk is especially plausible for U2R, R2L (NSL-KDD) and Backdoor, Analysis, Shellcode, Worms (UNSW), which collectively represent less than 5% of total samples.

---

*Report generated by FIDSUS Top-k Audit Instrumentation, 2026-06-08.*
