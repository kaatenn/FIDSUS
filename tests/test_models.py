"""models.py 单元测试：验证 CNN1D 与 BaseHeadSplit 前向传播。"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "system"))

from flcore.trainmodel.models import CNN1D, BaseHeadSplit


def test_cnn1d_output_shape():
    model = CNN1D(hidden_dim=32, num_classes=10)
    # 模拟 UNSW 42 维特征
    x = torch.randn(8, 42)
    out = model(x)
    assert out.shape == (8, 10)


def test_cnn1d_nslkdd_dim():
    model = CNN1D(hidden_dim=32, num_classes=5)
    # NSL-KDD 122 维
    x = torch.randn(4, 122)
    out = model(x)
    assert out.shape == (4, 5)


def test_base_head_split():
    model = CNN1D(hidden_dim=32, num_classes=5)
    head = model.fc
    model.fc = nn.Identity()
    bhs = BaseHeadSplit(model, head)
    x = torch.randn(4, 122)
    out = bhs(x)
    assert out.shape == (4, 5)
    # base 输出维度 = hidden_dim * 2 = 64
    rep = bhs.base(x)
    assert rep.shape == (4, 64)


def test_base_head_split_grad():
    """验证 base 与 head 可分离更新（FIDSUS 依赖此设计）。"""
    model = CNN1D(hidden_dim=32, num_classes=5)
    head = model.fc
    model.fc = nn.Identity()
    bhs = BaseHeadSplit(model, head)
    x = torch.randn(4, 122)
    y = torch.randint(0, 5, (4,))
    out = bhs(x)
    loss = nn.functional.nll_loss(out, y)
    loss.backward()
    # head 参数应有梯度
    assert bhs.head.weight.grad is not None
