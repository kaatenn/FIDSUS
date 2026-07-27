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

#### 运行 FIDSUS（默认算法）在 UNSW 数据集上

```bash
python main.py -data UNSW -algo FIDSUS
```

#### 运行 FedAvg 基线在 NSL-KDD 上

```bash
python main.py -data NSLKDD -algo FedAvg
```

#### 指定客户端数和全局轮次

```bash
python main.py -data UNSW -algo FIDSUS -nc 50 -gr 100 -lbs 64
```

### 3.2 完整命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-data` | str | `UNSW` | 数据集选择: `UNSW`, `NSLKDD`, `mnist`, `FashionMNIST` |
| `-algo` | str | `FIDSUS` | 算法选择（见 §3.3） |
| `-go` | str | `test` | 实验标识（用于结果文件命名） |
| `-dev` | str | `cuda` | 设备: `cuda` / `cpu` |
| `-did` | str | `0` | CUDA 设备 ID |
| `-nc` | int | `50` | 客户端总数 |
| `-jr` | float | `1.0` | 每轮参与客户端比例 |
| `-gr` | int | `100` | 全局通信轮次 |
| `-ls` | int | `1` | 本地训练轮次（epoch） |
| `-lbs` | int | `64` | 本地 batch size |
| `-lr` | float | `0.01` | 本地学习率 |
| `-slr` | float | `0.01` | Server 端学习率（FedGH / FIDSUS 用） |
| `-nb` | int | `10` | 类别数 |
| `-m` | str | `1dcnn` | 模型架构 |
| `-t` | int | `1` | 重复运行次数 |
| `-eg` | int | `1` | 评估间隔（每 N 轮评估一次） |
| `-M` | int | `5` | FIDSUS: 每个客户端收到的相似模型数 |
| `-mu` | float | `0.01` | FedProx / FIDSUS 近端项系数 |
| `-lam` | float | `0.99` | FedProto / GPFL 正则化系数 |
| `-tau` | float | `1.0` | MOON 温度系数 |
| `-klw` | float | `0.0` | FedAvgDBE KL 权重 |
| `-car` | float | `1.0` | 客户端活跃率（模拟掉线） |
| `-ld` | bool | `False` | 是否启用学习率衰减 |
| `-ldg` | float | `0.1` | 学习率衰减 gamma |
| `-tth` | float | `10000` | 慢客户端超时阈值（秒） |
| `-uf` | bool | `True` | FIDSUS: 是否启用跨轮特征融合。`False` 即消融变体 `FIDSUS_no_fusion` |
| `-rtl` | str | `""` | 逗号分隔的少数标签列表，如 `2` 或 `8,4,1,2`；为空按数据集自动选取 |

### 3.3 可选算法

| 算法参数值 | 论文中角色 |
|-----------|-----------|
| `FIDSUS` | **本文提出方法** |
| `FIDSUS_no_fusion` | **消融**：去掉 FIDSUS 的跨轮特征融合（MMD）部分 |
| `FedAvg` | 基线：加权联邦平均 |
| `FedProx` | 对比：近端正则化 |
| `FedProto` | 对比：原型聚合 |
| `MOON` | 对比：对比学习 |
| `FedGH` | 对比/消融：全局分类头（FIDSUS 无亲和矩阵版） |
| `GPFL` | 对比：通用/个性化条件变换 |
| `FedAvgDBE` | 对比：分布偏差消除 |

### 3.4 批量实验示例

创建批量脚本 `run_experiments.sh`：

```bash
#!/bin/bash
cd system

algos=("FedAvg" "FedProx" "FedProto" "MOON" "FedGH" "GPFL" "FedAvgDBE" "FIDSUS")
datasets=("UNSW" "NSLKDD")

for algo in "${algos[@]}"; do
    for ds in "${datasets[@]}"; do
        echo "=== Running $algo on $ds ==="
        python main.py -algo "$algo" -data "$ds" -nc 50 -gr 100 -t 3 -go "full_comparison"
    done
done
```

### 3.5 复现论文主要实验

```bash
cd system

# FIDSUS 在 UNSW-NB15 上（50 客户端，100 轮）
python main.py -algo FIDSUS -data UNSW -nc 50 -gr 100 -t 3 -go "reproduce"

# FedAvg 基线对比
python main.py -algo FedAvg -data UNSW -nc 50 -gr 100 -t 3 -go "reproduce"

# FIDSUS 在 NSL-KDD 上
python main.py -algo FIDSUS -data NSLKDD -nc 50 -nb 5 -gr 100 -t 3 -go "reproduce"
```

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
3. **注册算法**：在 `system/main.py` 中添加 `elif args.algorithm == "XXX":` 分支

### 5.2 添加新模型

在 `system/flcore/trainmodel/models.py` 中添加新模型类，然后在 `system/main.py` 的 `if model_str == "xxx":` 分支中注册。

### 5.3 添加新数据集

1. 创建 `dataset/NewData/` 目录结构
2. 编写 `dataset/generate_newdata.py`（参考 `generate_unsw.py`）
3. 在 `system/main.py` 中确保模型输入维度与数据集匹配

## 6. 常见问题

### Q: `cuda is not available` 怎么办？
A: 程序会自动回退到 CPU。也可以显式指定: `-dev cpu`

### Q: 数据集已存在，如何强制重新生成？
A: 删除对应数据集的 `config.json` 后再运行 generate 脚本，或删除整个 `train/` 和 `test/` 目录。

### Q: 如何修改客户端数量？
A: 修改 generate 脚本中的 `num_clients` 并重新生成数据，运行实验时用 `-nc` 指定相同数量。

### Q: torch 版本必须用 2.0.1 吗？
A: 这是论文原始环境版本。如需升级，编辑 `pyproject.toml` 中的 torch 相关版本号，注意同步更新 `torchaudio`/`torchvision`/`torchtext` 的兼容版本，然后执行 `uv lock --upgrade-package torch`。

## 7. 少数标签对比实验（FIDSUS vs FedAvg vs FIDSUS_no_fusion）

本章节对应论文中针对"样本较少标签"的对比实验，覆盖四个目标：

| 编号 | 目标 | 对应方法 |
|------|------|---------|
| ① | 少数标签上的准确率/召回率 | **FIDSUS** |
| ② | 少数标签上的准确率/召回率 | **FedAvg** |
| ③ | 少数标签上的准确率/召回率 | **FIDSUS_no_fusion**（去掉跨轮特征融合） |
| ④ | 收敛速度对比 | **FIDSUS** vs **FIDSUS_no_fusion** |

### 7.1 少数标签的定义

少数标签按各数据集的**全局训练样本量**统计确定（见 `system/utils/metrics_utils.py` 的 `RARE_LABELS`）：

| 数据集 | 少数标签 | 样本量（参考） |
|--------|---------|---------------|
| NSL-KDD | `2` | 87（极少） |
| UNSW-NB15 | `8, 4, 1, 2` | 139 / 1210 / 1824 / 2154 |

可用命令行 `-rtl` 覆盖，例如 `-rtl 2,3`。

### 7.2 指标计算（三口径并列）

每一轮训练都会计算并记录**三个评估口径**，各自回答一个独立问题（不做 metric shopping，全部并列汇报）：

| 口径 | 含义 | 适用算法 | 回答的问题 |
|------|------|---------|-----------|
| **个性化** | `model_per` 本地个性化模型 | 全部 | 个性化适配效果 |
| **全局头** | `model_per.base` 特征 + 服务器全局 `head` 分类 | 仅 FIDSUS 系 | 全局知识共享是否帮到稀有类 |
| **Macro-F1** | 各类 F1 算术平均（稀有类等权重） | 全部 | 类别平衡表现（不被大类主导） |

> **关键设计**：全局头口径与个性化口径**共享同一个特征提取器**（`model_per.base`），仅换分类头（个性化 head vs 服务器全局 head）。这是一个干净的对照实验——只隔离"全局知识共享"这一个变量。
>
> **为什么 FedAvg 没有全局头口径**：FedAvg 没有独立训练的服务器分类头，所以该口径对它不适用，汇总表中标 N/A。

结果写入 h5 文件的字段：
- `rs_test_precision` / `rs_test_recall`: 个性化口径，形状 `(rounds, num_classes)`
- `rs_global_head_precision` / `rs_global_head_recall`: 全局头口径（仅 FIDSUS 系）
- `rs_macro_f1` / `rs_global_head_macro_f1`: 每轮 Macro-F1（个性化 / 全局头）
- `rare_labels`: 少数标签索引数组

> **稀有类指标统一用「末尾 10 轮均值 ± std」**汇报（见 `summarize_rare_labels` 的 `tail_*` 字段），过滤小样本噪声。历史最佳 `best_*` 保留作参考，但因稀有类测试样本极少（如 NSL-KDD label2 仅 32 个），易被噪声虚高，**不建议作为主结论**。


训练过程中每轮会打印形如：
```
Rare labels | label8: P=0.1200 R=0.3400  label4: P=0.5600 R=0.7100
```

### 7.3 FIDSUS_no_fusion 消融变体

通过参数 `-uf False`（或直接选算法 `FIDSUS_no_fusion`）禁用 FIDSUS 的**跨轮特征融合**（MMD）部分，其余保持一致：

```bash
# 方式一：直接用算法名（推荐，结果文件自动带 _no_fusion 后缀）
python main.py -algo FIDSUS_no_fusion -data UNSW -nc 50 -gr 100

# 方式二：用 -uf 参数
python main.py -algo FIDSUS -uf False -data UNSW -nc 50 -gr 100 -go nofusion
```

对应代码：`clientFIDSUS.py` 的 `train()` 中
```python
if self.use_fusion:
    self.protos = aggregation(self.protos_g, self.protos_per)  # MMD 融合
else:
    self.protos = self.protos_g                                # 仅全局原型
```

## 8. 收敛速度分析

收敛速度由 `system/utils/metrics_utils.py::compute_convergence_speed()` 计算，指标包括：

| 指标 | 含义 |
|------|------|
| `best_acc` | 全程最高准确率 |
| `best_round` | 达到最高准确率的轮次 |
| `converge_round` | 首次达到 `90% × best_acc` 的轮次（越小收敛越快） |
| `tail_std` | 末尾 10 轮准确率标准差（越小越稳定） |

对比 `FIDSUS` 与 `FIDSUS_no_fusion` 的 `converge_round` 即可判断跨轮特征融合对收敛速度的影响。

## 9. 一键运行实验

项目提供两种一键运行方式（功能等价），在 `system/` 目录运行：

### 9.1 Python 版（推荐，Windows 友好）

```bash
cd system

# 真实实验：NSLKDD + UNSW，三种算法，100 轮（需 GPU）
uv run python run_all.py

# 快速冒烟测试（CPU 也能跑，2 轮 / 3 客户端，用于验证可运行）
uv run python run_all.py --quick

# 仅指定数据集 / 算法
uv run python run_all.py --datasets UNSW
uv run python run_all.py --algos FIDSUS FIDSUS_no_fusion

# 跳过训练，仅汇总已有结果
uv run python run_all.py --summary-only --goal rare_label_exp
```

运行结束后：
- 每个组合的结果写入 `results/{dataset}_{algo}_{goal}_{run}.h5`
- 汇总表写入 `results/summary.csv`
- 终端打印少数标签 precision/recall 表 + 收敛速度对比表

`run_all.py` 关键参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--datasets` | `NSLKDD UNSW` | 参与的数据集 |
| `--algos` | `FedAvg FIDSUS FIDSUS_no_fusion` | 参与的算法 |
| `--rounds` | `100` | 全局轮次 |
| `--clients` | `50` | 客户端数 |
| `--times` | `1` | 每组合重复次数 |
| `--device` | `cuda` | 设备 |
| `--goal` | `rare_label_exp` | 实验标识 |
| `--quick` | off | 冒烟模式（2轮/3客户端/cpu） |

### 9.2 Bash 版（Linux / WSL）

```bash
# 真实实验
bash run_experiments.sh

# 快速冒烟
QUICK=1 bash run_experiments.sh

# 自定义轮次 / 设备
ROUNDS=200 DEVICE=cuda bash run_experiments.sh
```

汇总：`bash run_experiments.sh` 结束后运行 `uv run python run_all.py --summary-only`。

## 10. 测试

项目在 `tests/` 下提供测试，用 `uv`（或项目 `.venv`）运行：

```bash
# 安装测试依赖（首次）
uv sync --group dev

# 运行全部测试（CPU，约 10 秒）
uv run pytest tests/ -v
```

测试覆盖：
- `test_metrics.py`：逐标签 precision/recall、Macro-F1、收敛速度、末尾均值±std 计算逻辑
- `test_models.py`：CNN1D / BaseHeadSplit 前向 shape、梯度分离
- `test_data_utils.py`：合成数据集 npz 读取
- `test_client_server.py`：FedAvg / FIDSUS / FIDSUS_no_fusion 端到端跑通（合成小数据，CPU），含全局头口径与 Macro-F1 字段断言

> 测试使用 `tests/conftest.py` 自动生成的合成 MINI 数据集，**不依赖大数据集、不依赖 GPU**，可在任何机器秒级验证代码可运行。

## 11. 结果文件字段说明

每个 `results/{dataset}_{algo}_{goal}_{run}.h5` 包含：

| 字段 | 形状 | 适用 | 说明 |
|------|------|------|------|
| `rs_test_acc` | `(rounds,)` | 全部 | 每轮总体加权准确率 |
| `rs_test_auc` | `(rounds,)` | 全部 | 每轮总体 micro AUC |
| `rs_train_loss` | `(rounds,)` | 全部 | 每轮训练损失 |
| `rs_test_precision` | `(rounds, num_classes)` | 全部 | 个性化口径每轮每标签 precision |
| `rs_test_recall` | `(rounds, num_classes)` | 全部 | 个性化口径每轮每标签 recall |
| `rs_global_head_precision` | `(rounds, num_classes)` | FIDSUS 系 | 全局头口径每轮每标签 precision |
| `rs_global_head_recall` | `(rounds, num_classes)` | FIDSUS 系 | 全局头口径每轮每标签 recall |
| `rs_macro_f1` | `(rounds,)` | 全部 | 个性化口径每轮 Macro-F1 |
| `rs_global_head_macro_f1` | `(rounds,)` | FIDSUS 系 | 全局头口径每轮 Macro-F1 |
| `rare_labels` | `(k,)` | 全部 | 少数标签索引 |

> 向后兼容：改动前生成的旧 h5 文件不含新字段，`load_run_result` 会将这些字段读为 `None`，汇总时优雅降级（如 FedAvg 不显示全局头行）。

读取示例：

```python
import h5py, numpy as np
with h5py.File("results/UNSW_FIDSUS_rare_label_exp_0.h5", "r") as f:
    acc = np.array(f["rs_test_acc"])
    recall = np.array(f["rs_test_recall"])           # 个性化口径 (rounds, num_classes)
    gh_recall = np.array(f["rs_global_head_recall"]) # 全局头口径
    rare = np.array(f["rare_labels"])
# 少数标签 label8 的召回率曲线（两个口径并列）
print("个性化:", recall[:, 8])
print("全局头:", gh_recall[:, 8])
```

### 11.5 如何解读三口径结果

三口径并列的核心价值是**隔离变量、诚实汇报**。解读原则：

1. **看个性化口径 vs 全局头口径的稀有类 recall 差异**（仅 FIDSUS 系）：
   - 全局头 recall **明显高于**个性化 → FIDSUS 的全局知识共享确实帮到了稀有类（说法 B 在"全局模型"层面成立）
   - 两者**接近** → 全局头没带来稀有类增益，个性化路径屏蔽了优势（支持说法 A）
   - 全局头**更低** → 全局头反而被大类主导（对稀有类不利）

2. **看 Macro-F1 而非总体准确率**判断类别平衡：总体准确率被大类（如 NSL-KDD 的 label0/4 占 88%）主导，Macro-F1 给稀有类等权重，是更诚实的平衡指标。

3. **看末尾均值±std 而非单轮值或历史最佳**：稀有类测试样本极少（NSL-KDD label2 仅 32 个），单轮 P/R 每变 1 个样本就跳 0.03，末尾 10 轮均值才能反映真实稳态。

