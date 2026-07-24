"""pytest 公共 fixtures。

生成一个合成的 MINI 数据集（与真实 npz 格式一致），供端到端测试使用，
避免依赖大数据集、避免长时间训练。同时将 system/ 加入 sys.path。
"""

import os
import sys
import numpy as np
import pytest

# 将 system 目录加入 sys.path，使测试能 import flcore / utils
SYSTEM_DIR = os.path.join(os.path.dirname(__file__), "..", "system")
SYSTEM_DIR = os.path.abspath(SYSTEM_DIR)
if SYSTEM_DIR not in sys.path:
    sys.path.insert(0, SYSTEM_DIR)

# 测试用合成数据集根目录：放在 tests/_tmpdata 下（被 .gitignore 忽略）
TMP_DATA_ROOT = os.path.join(os.path.dirname(__file__), "_tmpdata")


def _make_mini_dataset(name, num_clients=3, num_classes=5, feature_dim=42,
                       samples_per_client=40, seed=0):
    """生成合成 npz 数据分片，模拟真实数据集目录结构。

    目录: tests/_tmpdata/dataset/<name>/{train,test}/<i>.npz
    每个 npz: {'data': {'x': ndarray(N, feature_dim), 'y': ndarray(N,)}}
    """
    rng = np.random.RandomState(seed)
    base = os.path.join(TMP_DATA_ROOT, "dataset", name)
    train_dir = os.path.join(base, "train")
    test_dir = os.path.join(base, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for i in range(num_clients):
        # 训练集：保证每个标签都有一些样本（避免单一类别导致 AUC 异常）
        n = samples_per_client
        x = rng.randn(n, feature_dim).astype(np.float32)
        # 让前 num_classes*4 个样本均匀覆盖各类，其余随机
        y = rng.randint(0, num_classes, size=n).astype(np.int64)
        head = np.tile(np.arange(num_classes), 4)
        y[:len(head)] = head
        np.savez(os.path.join(train_dir, "{}.npz".format(i)),
                 data={'x': x, 'y': y})
        # 测试集
        nt = max(20, samples_per_client // 2)
        xt = rng.randn(nt, feature_dim).astype(np.float32)
        yt = rng.randint(0, num_classes, size=nt).astype(np.int64)
        np.savez(os.path.join(test_dir, "{}.npz".format(i)),
                 data={'x': xt, 'y': yt})
    return base


@pytest.fixture(scope="session")
def mini_dataset():
    """返回 (dataset_name, feature_dim, num_classes)，供多个测试复用。"""
    name = "MINI"
    feature_dim = 42
    num_classes = 5
    _make_mini_dataset(name, num_clients=3, num_classes=num_classes,
                       feature_dim=feature_dim, samples_per_client=40, seed=42)
    return name, feature_dim, num_classes


@pytest.fixture
def chdir_system(monkeypatch):
    """切换工作目录到 system/，使相对路径 ../dataset / ../results 正常工作。

    同时把 dataset 根目录重定向到 tests/_tmpdata 下的合成数据。
    """
    # 将合成数据集目录链接/拷贝到 system 视角下的 ../dataset
    # system/main.py 使用 ../dataset/<name>，因此 system 工作目录 + ../dataset
    # 我们直接让测试在 system 目录运行，并在该处创建 ../dataset/<name> 软链。
    monkeypatch.chdir(SYSTEM_DIR)
    # 确保合成数据集对 system 可见
    src = os.path.join(TMP_DATA_ROOT, "dataset")
    dst = os.path.abspath(os.path.join(SYSTEM_DIR, "..", "dataset"))
    # dataset 目录可能已存在（真实数据集），我们只确保 MINI 子目录存在于其中
    mini_src = os.path.join(src, "MINI")
    mini_dst = os.path.join(dst, "MINI")
    if os.path.isdir(mini_src) and not os.path.isdir(mini_dst):
        try:
            os.symlink(mini_src, mini_dst)
        except (OSError, NotImplementedError):
            # Windows 无权限创建符号链接时回退为拷贝
            import shutil
            shutil.copytree(mini_src, mini_dst)
    yield SYSTEM_DIR
