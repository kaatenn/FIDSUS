# FIDSUS 项目使用指南

> 基于论文 *FIDSUS: Federated Intrusion Detection for Securing UAV Swarms in Smart Aerial Computing* 的代码实现。

## 1. 环境搭建

### 1.1 前置条件

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) 包管理器（已替代 Conda）
- CUDA 11.8 兼容的 GPU（可选，CPU 也可运行）

### 1.2 安装依赖

```bash
# 在项目根目录执行
uv sync
```

这会自动：
- 读取 `pyproject.toml` 中的依赖声明
- 从 PyPI + PyTorch CUDA 索引解析并安装所有包
- 创建 `.venv` 虚拟环境

### 1.3 激活环境

```bash
# Windows (PowerShell)
.venv\Scripts\activate

# Windows (CMD)
.venv\Scripts\activate.bat

# Linux / macOS
source .venv/bin/activate
```

## 2. 数据集准备

项目已内置 4 个预分区数据集（每个 50 个客户端）。如果你需要重新生成或自定义分区参数，按以下步骤操作：

### 2.1 UNSW-NB15（10 类入侵检测）

```bash
cd dataset
python generate_unsw.py
```

- 原始数据: `dataset/UNSW/rawdata/datasets/unsw_train.csv`, `unsw_test.csv`
- 预处理脚本: `dataset/UNSW/rawdata/datasets/preprocess.py`（将攻击类别映射为 1-10 标签）
- 输出: `dataset/UNSW/train/*.npz` + `dataset/UNSW/test/*.npz` (各 50 个)

### 2.2 NSL-KDD（5 类入侵检测）

```bash
cd dataset
python generate_nslkdd.py
```

- 原始数据: `dataset/NSLKDD/rawdata/datasets/KDDTrain.csv`, `KDDTest.csv`
- 预处理脚本: `dataset/NSLKDD/rawdata/datasets/preprocess.py`（将 41 种攻击映射为 5 个类别）
- 输出: `dataset/NSLKDD/train/*.npz` + `dataset/NSLKDD/test/*.npz` (各 50 个)

### 2.3 自定义分区参数

编辑 `dataset/utils/dataset_utils.py` 中的变量：

```python
train_ratio = 0.8   # 训练/测试切分比例
alpha = 0.3          # Dirichlet 分布浓度参数（越小越 Non-IID）
batch_size = 10      # 最小每客户端样本数约束
```

编辑 `generate_unsw.py` / `generate_nslkdd.py` 底部的调用参数：

```python
generate_unsw(dir_path, num_clients=50, niid=True, balance=False, partition="dir")
#                                   ↑客户端数   ↑非独立同分布  ↑不均衡    ↑Dirichlet方式
```

## 3. 运行实验

### 3.1 基本用法

所有命令在 `system/` 目录下执行：

```bash
cd system
```

#### 运行默认实验（FIDSUS on UNSW）

```bash
python main.py
```

这会自动读取 `experiments/default.json` 并运行其中定义的所有实验。

#### 运行指定的配置文件

```bash
python main.py --config experiments/algorithm_comparison.json
```

简写形式：

```bash
python main.py -c experiments/full_sweep.json
```

### 3.2 JSON 配置文件格式

配置文件由两部分组成：`defaults`（共享默认值）和 `experiments`（实验列表）。

```json
{
  "defaults": {
    "ignore": false,
    "device": "cuda",
    "device_id": "0",
    "dataset": "UNSW",
    "num_clients": 50,
    "join_ratio": 1.0,
    "global_rounds": 100,
    "local_epochs": 1,
    "batch_size": 64,
    "local_learning_rate": 0.01,
    ...
  },
  "experiments": [
    {
      "name": "FIDSUS on UNSW",
      "algorithm": "FIDSUS",
      "dataset": "UNSW",
      "times": 3
    },
    {
      "name": "FedAvg baseline",
      "algorithm": "FedAvg",
      "dataset": "NSLKDD",
      "num_classes": 5,
      "ignore": true
    }
  ]
}
```

**规则：**
- `defaults` 中的值应用于所有实验，单个实验可通过覆盖字段来定制
- `experiments` 数组中的每个对象代表一次实验
- `name` 字段可选，省略时自动生成 `{algorithm}_{dataset}_{goal}`
- `ignore` 设为 `true` 可临时跳过某个实验（无需删除配置）
- 合并方式为浅合并：`{...defaults, ...experiment}`

### 3.3 完整参数列表

| JSON 键 | 类型 | 默认值 | 说明 |
|---------|------|--------|------|
| `ignore` | bool | `false` | 是否跳过此实验 |
| `algorithm` | str | `"FIDSUS"` | 算法选择（见 §3.4） |
| `dataset` | str | `"UNSW"` | 数据集：`UNSW`, `NSLKDD`, `mnist`, `FashionMNIST` |
| `goal` | str | `"test"` | 实验标识（用于结果文件命名） |
| `device` | str | `"cuda"` | 设备：`cuda` / `cpu` |
| `device_id` | str | `"0"` | CUDA 设备 ID |
| `num_clients` | int | `50` | 客户端总数 |
| `join_ratio` | float | `1.0` | 每轮参与客户端比例 |
| `global_rounds` | int | `100` | 全局通信轮次 |
| `local_epochs` | int | `1` | 本地训练轮次（epoch） |
| `batch_size` | int | `64` | 本地 batch size |
| `local_learning_rate` | float | `0.01` | 本地学习率 |
| `server_learning_rate` | float | `0.01` | Server 端学习率（FedGH / FIDSUS 用） |
| `num_classes` | int | `10` | 类别数 |
| `model` | str | `"1dcnn"` | 模型架构 |
| `times` | int | `1` | 重复运行次数 |
| `eval_gap` | int | `1` | 评估间隔（每 N 轮评估一次） |
| `M` | int | `5` | FIDSUS：每个客户端收到的相似模型数 |
| `mu` | float | `0.01` | FedProx / FIDSUS 近端项系数 |
| `lamda` | float | `0.99` | FedProto / GPFL 正则化系数 |
| `tau` | float | `1.0` | MOON 温度系数 |
| `kl_weight` | float | `0.0` | FedAvgDBE KL 权重 |
| `client_activity_rate` | float | `1.0` | 客户端活跃率（模拟掉线） |
| `learning_rate_decay` | bool | `false` | 是否启用学习率衰减 |
| `learning_rate_decay_gamma` | float | `0.1` | 学习率衰减 gamma |
| `time_threthold` | float | `10000` | 慢客户端超时阈值（秒） |
| `batch_num_per_client` | int | `2` | 每客户端批次数 |
| `random_join_ratio` | bool | `false` | 随机参与比例 |
| `prev` | int | `0` | 续跑起始索引 |
| `save_folder_name` | str | `"items"` | 模型保存目录 |
| `beta` | float | `0.0` | 额外参数 |
| `p_learning_rate` | float | `0.01` | 个性化学习率 |
| `momentum` | float | `0.1` | 动量系数 |

### 3.4 可选算法

| 算法参数值 | 论文中角色 |
|-----------|-----------|
| `FIDSUS` | **本文提出方法** |
| `FedAvg` | 基线：加权联邦平均 |
| `FedProx` | 对比：近端正则化 |
| `FedProto` | 对比：原型聚合 |
| `MOON` | 对比：对比学习 |
| `FedGH` | 对比/消融：全局分类头（FIDSUS 无亲和矩阵版） |
| `GPFL` | 对比：通用/个性化条件变换 |
| `FedAvgDBE` | 对比：分布偏差消除 |

### 3.5 批量实验

在 JSON 配置文件的 `experiments` 数组中定义多个实验即可，一次运行顺序执行：

```json
{
  "experiments": [
    {"algorithm": "FedAvg",  "dataset": "UNSW",   "times": 3},
    {"algorithm": "FedProx", "dataset": "UNSW",   "times": 3},
    {"algorithm": "FIDSUS",  "dataset": "UNSW",   "times": 3},
    {"algorithm": "FedAvg",  "dataset": "NSLKDD", "num_classes": 5, "times": 3},
    {"algorithm": "FIDSUS",  "dataset": "NSLKDD", "num_classes": 5, "times": 3}
  ]
}
```

**预置配置文件：**

| 文件 | 说明 |
|------|------|
| `experiments/default.json` | 单次默认实验（FIDSUS on UNSW） |
| `experiments/algorithm_comparison.json` | 8 算法 × 2 数据集（UNSW + NSLKDD），各 3 次 |
| `experiments/full_sweep.json` | 全覆盖：4 数据集 × 8 算法 + mu/M/jr 参数扫描，MNIST/FashionMNIST 默认跳过 |

### 3.6 跳过实验

将实验中 `ignore` 设为 `true` 即可临时禁用，无需删除配置行：

```json
{
  "experiments": [
    {"name": "keep this",    "algorithm": "FedAvg", "dataset": "UNSW"},
    {"name": "skip this",    "algorithm": "FedAvg", "dataset": "NSLKDD", "ignore": true}
  ]
}
```

运行时输出 `[SKIPPED]` 标记。

### 3.7 复现论文主要实验

```bash
cd system

# FIDSUS 在 UNSW-NB15 上（50 客户端，100 轮，3 次重复）
python main.py -c experiments/default.json

# 全算法对比（UNSW + NSLKDD）
python main.py -c experiments/algorithm_comparison.json

# 全覆盖实验（含参数扫描，图像数据集默认跳过）
python main.py -c experiments/full_sweep.json
```

如需只运行特定数据集，编辑配置文件将其他实验的 `ignore` 设为 `true`，或创建自定义配置文件。

## 4. 结果解读

### 4.1 训练过程输出

```
============= Running time: 0th =============
Creating server and clients ...
Join ratio / total clients: 1.0 / 50
Finished creating server and clients.

-------------Round number: 0-------------
Evaluate personalized models
Averaged Train Loss: 2.1234
Averaged Test Accurancy: 0.4521
Averaged Test AUC: 0.5234
Std Test Accurancy: 0.1201
Std Test AUC: 0.0987
------------------------- time cost ------------------------- 12.34
...
Best accuracy.
0.8523
Average time cost per round.
10.56
```

关键指标：
- **Averaged Test Accuracy**: 所有客户端测试准确率的加权平均
- **Averaged Test AUC**: 微平均 ROC-AUC
- **Std Test Accuracy**: 客户端间准确率标准差（越低越稳定）
- **Best accuracy**: 全程最佳准确率
- **Average time cost per round**: 平均每轮耗时

### 4.2 结果文件位置

| 文件 | 路径 | 内容 |
|------|------|------|
| 训练曲线 | `results/{dataset}_{algo}_{goal}_{time}.h5` | `rs_test_acc`, `rs_test_auc`, `rs_train_loss` 数组 |
| 时间统计 | `system/timecost/time_cost_{algo}_{nc}_{dataset}_{goal}.txt` | 运行总时间 |
| 汇总统计 | 控制台输出（由 `result_utils.average_data` 打印） | Best Accuracy 的均值 ± 标准差 |

### 4.3 读取结果文件

```python
import h5py
import numpy as np

with h5py.File("results/UNSW_FIDSUS_test_0.h5", "r") as f:
    test_acc = np.array(f["rs_test_acc"])
    test_auc = np.array(f["rs_test_auc"])
    train_loss = np.array(f["rs_train_loss"])

print(f"Final accuracy: {test_acc[-1]:.4f}")
print(f"Best accuracy: {test_acc.max():.4f} at round {test_acc.argmax()}")
```

## 5. 项目扩展

### 5.1 添加新算法

1. **创建 Client**：在 `system/flcore/clients/` 下新建 `clientXXX.py`，继承 `Client`
2. **创建 Server**：在 `system/flcore/servers/` 下新建 `serverXXX.py`，继承 `Server`
3. **注册算法**：在 `system/main.py` 的算法选择分支中添加 `elif args.algorithm == "XXX":` 分支
4. **更新配置**：在 JSON 配置文件的实验中添加新算法名即可使用

### 5.2 添加新模型

在 `system/flcore/trainmodel/models.py` 中添加新模型类，然后在 `system/main.py` 的 `if model_str == "xxx":` 分支中注册。

### 5.3 添加新数据集

1. 创建 `dataset/NewData/` 目录结构
2. 编写 `dataset/generate_newdata.py`（参考 `generate_unsw.py`）
3. 在 `system/main.py` 中确保模型输入维度与数据集匹配

## 6. 常见问题

### Q: `cuda is not available` 怎么办？
A: 程序会自动回退到 CPU。也可以在 JSON 配置中显式指定：`"device": "cpu"`

### Q: 数据集已存在，如何强制重新生成？
A: 删除对应数据集的 `config.json` 后再运行 generate 脚本，或删除整个 `train/` 和 `test/` 目录。

### Q: 如何修改客户端数量？
A: 修改 generate 脚本中的 `num_clients` 并重新生成数据，在 JSON 配置中用 `"num_clients"` 指定相同数量。

### Q: torch 版本必须用 2.0.1 吗？
A: 这是论文原始环境版本。如需升级，编辑 `pyproject.toml` 中的 torch 相关版本号，注意同步更新 `torchaudio`/`torchvision`/`torchtext` 的兼容版本，然后执行 `uv lock --upgrade-package torch`。

### Q: 如何跳过某个实验而不删除配置？
A: 将该实验的 `"ignore"` 设为 `true`，运行时自动跳过并显示 `[SKIPPED]` 标记。

### Q: 如何创建新的实验配置？
A: 复制 `experiments/` 目录下的任一 `.json` 文件作为模板，修改 `experiments` 数组即可。然后用 `python main.py -c experiments/your_config.json` 运行。

### Q: 多实验配置中某个实验失败了怎么办？
A: 程序不会因单个实验失败而中断——后续实验会继续运行。检查失败实验的报错信息，修正配置后可将 `"prev"` 设为已完成次数来续跑。

### Q: 配置文件路径相对于哪里？
A: 相对于 `system/` 目录（即 `main.py` 所在目录）。建议将配置文件放在 `system/experiments/` 下。
