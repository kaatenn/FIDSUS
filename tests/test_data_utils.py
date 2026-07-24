"""data_utils 单元测试：在合成 MINI 数据集上验证读取。"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "system"))

from utils.data_utils import read_client_data_un


def test_read_mini_dataset(mini_dataset, chdir_system):
    name, feature_dim, num_classes = mini_dataset
    data = read_client_data_un(name, 0, is_train=True)
    assert len(data) > 0
    x, y = data[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
    assert x.shape[0] == feature_dim


def test_read_test_split(mini_dataset, chdir_system):
    name, feature_dim, num_classes = mini_dataset
    data = read_client_data_un(name, 0, is_train=False)
    assert len(data) > 0
    for x, y in data:
        assert 0 <= int(y.item()) < num_classes
