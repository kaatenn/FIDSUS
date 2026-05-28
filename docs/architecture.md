# FIDSUS 代码架构梳理

> 论文: *FIDSUS: Federated Intrusion Detection for Securing UAV Swarms in Smart Aerial Computing* (IEEE IoT Journal, 2025)

## 1. 项目概览

FIDSUS 是一个联邦学习（FL）入侵检测系统，针对 UAV 集群场景中的数据异构、通信不稳定、拓扑频繁变化等问题而设计。项目实现了 **8 种 FL 算法**，在 **NSL-KDD** 和 **UNSW-NB15** 数据集上进行入侵检测实验。

## 2. 目录结构

```
FIDSUS/
├── system/                          # 核心联邦学习系统
│   ├── main.py                      # ★ 入口：解析参数、分发算法、运行实验
│   ├── run.sh                       # 示例启动脚本
│   ├── flcore/
│   │   ├── servers/
│   │   │   ├── serverbase.py        # ★ Server 基类
│   │   │   ├── FIDSUS.py            # ★ 本文算法
│   │   │   ├── serveravg.py         # FedAvg
│   │   │   ├── serverprox.py        # FedProx
│   │   │   ├── serverproto.py       # FedProto
│   │   │   ├── servermoon.py        # MOON
│   │   │   ├── servergh.py          # FedGH
│   │   │   ├── servergpfl.py        # GPFL
│   │   │   └── serveravgDBE.py      # FedAvgDBE
│   │   ├── clients/
│   │   │   ├── clientbase.py        # ★ Client 基类
│   │   │   ├── clientFIDSUS.py      # ★ 本文算法
│   │   │   ├── clientavg.py
│   │   │   ├── clientprox.py
│   │   │   ├── clientproto.py
│   │   │   ├── clientmoon.py
│   │   │   ├── clientgh.py
│   │   │   ├── clientgpfl.py
│   │   │   └── clientavgDBE.py
│   │   ├── trainmodel/
│   │   │   └── models.py            # ★ CNN1D + BaseHeadSplit
│   │   └── optimizers/
│   │       └── fedoptimizer.py      # ★ PerturbedGradientDescent
│   └── utils/
│       ├── data_utils.py            # 数据加载（读取 .npz 分片）
│       └── result_utils.py          # 结果汇总（h5 文件统计）
└── dataset/
    ├── utils/dataset_utils.py       # ★ 数据分区（Dirichlet 分配）
    ├── generate_unsw.py             # UNSW-NB15 预处理 + 分区
    ├── generate_nslkdd.py           # NSL-KDD 预处理 + 分区
    ├── UNSW/                        # UNSW-NB15（预分区 50 个 .npz）
    ├── NSLKDD/                      # NSL-KDD（预分区 50 个 .npz）
    ├── mnist/                       # MNIST（预分区）
    └── FashionMNIST/                # FashionMNIST（预分区）
```

## 3. 模块职责详解

### 3.1 `system/main.py` — 实验入口

- 解析全部命令行参数（算法、数据集、客户端数、学习率、轮次等）
- 根据 `--algorithm` 参数实例化对应的 Server（工厂模式）
- 调用 `server.train()` 执行 FL 训练循环
- 通过 `average_data()` 汇总多次运行的均值和标准差

关键参数一览：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--algorithm` | FL 算法选择 | FIDSUS |
| `--dataset` | 数据集 | UNSW |
| `--num_clients` | 客户端总数 | 50 |
| `--join_ratio` | 每轮参与客户端比例 | 1.0 |
| `--global_rounds` | 全局通信轮次 | 100 |
| `--local_epochs` | 本地训练轮次 | 1 |
| `--batch_size` | 批次大小 | 64 |
| `--local_learning_rate` | 本地学习率 | 0.01 |
| `--model` | 模型架构 | 1dcnn |
| `--M` | 每个客户端收到的模型数（FIDSUS） | 5 |
| `--mu` | 近端项系数 | 0.01 |
| `--lamda` | 原型正则系数 | 0.99 |
| `--tau` | MOON 温度系数 | 1.0 |

### 3.2 `flcore/servers/serverbase.py` — Server 基类

核心职责：
- `set_clients()`: 为每个数据分片初始化一个 Client 对象
- `select_clients()`: 随机选择参与本轮的客户端子集
- `send_models()`: 将全局模型下发到所有客户端
- `receive_models()`: 收集选中客户端的模型和权重
- `aggregate_parameters()`: 加权联邦平均（FedAvg 标准聚合）
- `evaluate()`: 汇总所有客户端测试指标（Accuracy、AUC、Loss）
- `save_results()`: 将结果写入 `.h5` 文件

### 3.3 `flcore/clients/clientbase.py` — Client 基类

核心职责：
- `load_train_data()` / `load_test_data()`: 从 `.npz` 文件加载该客户端的数据分片
- `set_parameters()`: 接收服务器下发的模型参数
- `train()`: 本地 SGD 训练（子类覆写实现差异化逻辑）
- `test_metrics()`: 计算 Accuracy 和 AUC
- `train_metrics()`: 计算训练 Loss

### 3.4 `flcore/trainmodel/models.py` — 模型架构

**CNN1D**: 所有算法统一使用的模型
```
Conv1d(1→32, kernel=3) → ReLU → MaxPool1d(2)
→ Conv1d(32→64, kernel=3) → ReLU → MaxPool1d(2)
→ AdaptiveMaxPool1d(1) → Flatten → Dropout(0.2)
→ Linear(64→num_classes) → LogSoftmax
```

**BaseHeadSplit**: 将模型拆分为 base（特征提取器）和 head（分类器），这是 FIDSUS 等算法的关键设计：
- `base`: Conv + Pool 层（提取特征表示）
- `head`: 最后的 Linear 层（分类决策）
- 分离后可以独立更新 head（如 FIDSUS 的全局分类器训练）

### 3.5 `flcore/optimizers/fedoptimizer.py` — 自定义优化器

**PerturbedGradientDescent**: 带近端项的 SGD 变体
```
更新公式: p = p - lr * (grad + μ * (p - p_global))
```
- 被 FedProx 和 FIDSUS 的个性化模型使用
- `μ` 控制本地模型向全局模型靠拢的强度

### 3.6 `dataset/utils/dataset_utils.py` — 数据分区工具

- `separate_data()`: 核心分区函数，支持两种模式：
  - `pat`: 按类别均匀/非均匀分配
  - `dir`: Dirichlet 分布（α=0.3）模拟 Non-IID 数据分布
- `split_data()`: 将每个客户端数据按 8:2 切分为训练集和测试集
- `save_file()`: 将分区结果保存为 `.npz` + `config.json`

### 3.7 `system/utils/data_utils.py` — 数据加载

- `read_client_data_un()`: 从 `.npz` 读取单个客户端的数据，返回 `[(x_tensor, y_tensor), ...]` 列表

### 3.8 `system/utils/result_utils.py` — 结果统计

- `average_data()`: 汇总多次实验结果，输出 Best Accuracy 的均值和标准差

## 4. 类继承关系

```
Server (serverbase.py)
├── FedAvg      ← 标准加权联邦平均，基线方法
├── FedProx     ← 近端正则化 μ||w-w_g||²
├── FedProto    ← 仅交换类别原型，无全局模型
├── MOON        ← 对比学习：rep 靠近全局、远离旧局部
├── FedGH       ← 全局分类头 + 原型传输（FIDSUS 简化版）
├── GPFL        ← GCE + CoV 通用/个性化条件特征变换
├── FedAvgDBE   ← 分布偏差消除（running mean 对齐）
└── FIDSUS      ← ★ 本文方法

Client (clientbase.py)
├── clientAVG      ← 标准 SGD 本地训练
├── clientProx     ← 带近端项的 PerturbedGradientDescent
├── clientProto    ← 原型匹配 MSE 正则
├── clientMOON     ← 对比损失（cosine similarity）
├── clientGH       ← 原型收集
├── clientGPFL     ← GCE + CoV 双变换
├── clientAvgDBE   ← running mean + client_mean 偏差纠正
└── clientFIDSUS   ← ★ 亲和权重 + MMD 融合 + 双模型
```

## 5. 数据流

```
                    ┌──────────────────────────────┐
                    │       Server (FIDSUS)         │
                    │                               │
                    │  P[i][j]: 亲和矩阵             │
                    │  client_models[]: 所有客户端模型 │
                    │  head: 全局分类器（可训练）      │
                    └──────────┬─────────┬──────────┘
                  send_models()│         │receive_models()
                               │         │
            ┌──────────────────┘         └──────────────────┐
            ▼                                               ▼
┌─────────────────────┐                         ┌─────────────────────┐
│     Client i        │  ◄— top-M similar —►    │   Other Clients     │
│                     │                         │                     │
│ 1. weight_cal()     │  计算验证集loss+梯度      │                     │
│ 2. aggregate()      │  加权聚合M个模型          │                     │
│ 3. train()          │  model(g) + model_per(p)│                     │
│ 4. MMD fusion       │  protos←agg(g, per)     │                     │
│ 5. upload           │  model + protos + w_vec │                     │
└─────────────────────┘                         └─────────────────────┘
                               │
                               ▼
                  ┌──────────────────────────────┐
                  │   Server: train_head()        │
                  │   用上传的 protos 训练全局 head │
                  │   更新 P[i] += weight_vector   │
                  └──────────────────────────────┘
```

## 6. 数据集

| 数据集 | 特征维度 | 类别数 | 客户端数 | 分区方式 | 用途 |
|--------|---------|--------|---------|---------|------|
| NSL-KDD | 122 | 5 | 50 | Dirichlet α=0.3 | 网络入侵检测（论文核心数据集） |
| UNSW-NB15 | 42 | 10 | 50 | Dirichlet α=0.3 | 网络入侵检测（论文核心数据集） |
| MNIST | 784 | 10 | 50 | Dirichlet α=0.3 | 图像分类验证 |
| FashionMNIST | 784 | 10 | 50 | Dirichlet α=0.3 | 图像分类验证 |

每个客户端的数据存储为独立 `.npz` 文件：`(train|test)/(0..49).npz`，包含 `{'x': array, 'y': array}`。
