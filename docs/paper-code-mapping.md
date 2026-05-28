# FIDSUS 论文研究内容与代码对应关系

> 论文: *FIDSUS: Federated Intrusion Detection for Securing UAV Swarms in Smart Aerial Computing* (IEEE IoT Journal, 2025)
>
> 代码: `system/flcore/servers/FIDSUS.py` + `system/flcore/clients/clientFIDSUS.py`

## 1. FIDSUS 四大创新点 → 代码映射

| 论文概念 | 代码位置 | 实现细节 |
|----------|----------|----------|
| **亲和矩阵 P** | `FIDSUS.py:17` — `self.P = torch.diag(torch.ones(num_clients))` | 初始化为单位对角阵，每轮用客户端的 `weight_vector` 增量更新 |
| **Top-M 模型选择** | `FIDSUS.py:58-67` — `send_models()` | `torch.topk(self.P[client.id], M_)` 选出 M 个最相似客户端的模型下发 |
| **亲和权重计算** | `clientFIDSUS.py:99-111` — `weight_cal()` | `(L_old - L_received) / \|\|param_diff\|\|` — 损失下降越多=越相似 |
| **双模型架构** | `clientFIDSUS.py:30-39` | `self.model`（全局） + `self.model_per`（个性化），后者用 `PerturbedGradientDescent` |
| **MMD 跨轮特征融合** | `clientFIDSUS.py:210-261` — `MMD()` + `aggregation()` | 计算全局原型与个性化原型间的 MMD 距离 → 按距离加权融合 |
| **全局分类器训练** | `FIDSUS.py:102-109` — `train_head()` | 收集所有客户端的类别原型 `protos`，集中训练 Server 端的 `head` |
| **个性化评估** | `FIDSUS.py:131-151` — `evaluate_personalized()` | 使用 `model_per`（个性化模型）而非全局模型进行测试评估 |

## 2. 核心算法流程（对应论文 Algorithm 1）

```
每轮 t:
┌─ Server ────────────────────────────────────────────┐
│ 1. selected_clients = select_clients()               │
│ 2. send_models():                                    │
│    对每个客户端 i:                                    │
│      indices = topk(P[i], M)  # 选M个最相似的         │
│      发送 {id_j: model_j for j in indices}           │
│                                                      │
│ (每 eval_gap 轮评估一次)                              │
│                                                      │
│ 3. receive_models():                                 │
│    收集 client.model, client.protos, client.weight_vector │
│    更新 client_models[id] = model                     │
│    更新 P[id] += weight_vector                        │
│                                                      │
│ 4. aggregate_parameters():                           │
│    FedAvg 加权聚合                                   │
│                                                      │
│ 5. train_head():                                     │
│    用上传的 protos 作为 DataLoader 训练全局 head       │
└──────────────────────────────────────────────────────┘

┌─ Client i ──────────────────────────────────────────┐
│ 1. receive_models(ids, models):                      │
│    存储收到的 M 个相似客户端模型                       │
│                                                      │
│ 2. aggregate_parameters(val_loader):                 │
│    weights = weight_scale(weight_cal(val_loader))    │
│      weight_cal:                                     │
│        L_old = loss(old_model, val)                  │
│        对每个收到模型 j:                               │
│          L_j = loss(model_j, val)                    │
│          w_j = (L_old - L_j) / ||param_j - param_old|| │
│      weight_scale:                                   │
│        weights = max(w, 0) / sum(w)                  │
│    model = Σ w_j * model_j                           │
│                                                      │
│ 3. train():                                          │
│    对每个 epoch:                                      │
│      # 全局模型训练                                   │
│      reg = model.base(x)                             │
│      loss = CELoss(model.head(reg), y)               │
│      optimizer.step()                                │
│                                                      │
│      # 个性化模型训练                                  │
│      reg_per = model_per.base(x)                     │
│      loss = CELoss(model_per.head(reg_per), y)       │
│      optimizer_per.step(model_per.params, device)    │
│                                                      │
│      # 收集原型                                       │
│      protos[y_c].append(reg[i])                      │
│      protos_per[y_c].append(reg_per[i])              │
│                                                      │
│    protos_g = agg_func(protos)  # 按类别平均          │
│    protos_per = agg_func(protos_per)                 │
│    protos = aggregation(protos_g, protos_per)        │
│                                                      │
│ 4. upload: model, protos, weight_vector              │
└──────────────────────────────────────────────────────┘
```

## 3. 创新点深度剖析

### 3.1 亲和矩阵 (Affinity Matrix) — `FIDSUS.py` + `clientFIDSUS.py`

**论文意图**: 量化 UAV 间局部特征提取器的相似性，让相似 UAV 互相分享知识，不相似的不分享，避免负迁移。

**代码实现**:
- **存储**: `self.P` 是 `num_clients × num_clients` 的对角矩阵，初始化为单位阵
- **更新**: 每轮训练后，`self.P[client.id] += client.weight_vector`（`FIDSUS.py:97`）
- **选择**: `torch.topk(self.P[client.id], M_)` 取 P 矩阵第 i 行中最大的 M 个值对应的客户端索引（`FIDSUS.py:59`）

**亲和权重计算** (`clientFIDSUS.py:99-111`):
```python
def weight_cal(self, val_loader):
    L = self.recalculate_loss(self.old_model, val_loader)    # 上一轮模型在验证集上的 loss
    for received_model in self.received_models:
        params_dif = ||received_model - old_model||           # 参数变化量
        weight = (L - loss(received_model)) / (params_dif + ε)  # 损失下降 / 参数差异
    # 损失下降越多 → 该模型对本客户端越有帮助 → 权重越大
```

### 3.2 跨轮特征融合 (Cross-Round Feature Fusion) — `clientFIDSUS.py:210-261`

**论文意图**: 用 MMD 衡量当前轮全局原型与个性化原型间的分布差异，按差异大小加权融合，缓解灾难性遗忘。

**MMD 计算** (`clientFIDSUS.py:210-234`):
```python
def MMD(x, y, kernel, device):
    # 使用 RBF 多带宽核: bandwidths = [10, 15, 20, 50]
    # MMD² = E[k(x,x)] + E[k(y,y)] - 2E[k(x,y)]
    return mean(XX + YY - 2*XY)
```

**融合策略** (`clientFIDSUS.py:248-261`):
```python
def aggregation(protos, protos_per):
    for label in protos:
        mmd_value = MMD(protos[label], protos_per[label])
        normalized_mmd = (mmd - min) / (max - min)   # 归一化到 [0,1]
        weight = 1 - normalized_mmd                   # MMD 越小 → 越相似 → 权重越大
        aggregated[label] = weight * protos[label] + (1-weight) * protos_per[label]
```

### 3.3 全局分类器 (Global Head Training) — `FIDSUS.py:102-109`

**论文意图**: 在 Server 端用所有客户端的类别原型训练一个共享分类器头，提升全局泛化能力。

**代码实现**:
```python
def train_head(self):
    proto_loader = DataLoader(self.uploaded_protos, batch_size, shuffle=True)
    for p, y in proto_loader:
        out = self.head(p)          # 全局 head 对原型进行分类
        loss = CEloss(out, y)
        self.opt_h.zero_grad()
        loss.backward()
        self.opt_h.step()
```

### 3.4 双模型与个性化评估 — `clientFIDSUS.py`

**论文意图**: 每个客户端维护全局模型和个性化模型两个副本，分别用于知识共享和本地适配。

**代码实现**:
- `self.model`（全局模型）: 标准 SGD → 上传到 Server 参与聚合
- `self.model_per`（个性化模型）: `PerturbedGradientDescent`（带近端项）→ 仅本地使用
- 评估时使用 `model_per` (`evaluate_personalized`)，反映个性化适配效果

## 4. 对比算法一览

| 算法 | 核心思想 | 论文中作用 | Server | Client |
|------|----------|-----------|--------|--------|
| **FedAvg** | 加权联邦平均 | 基线方法 | `serveravg.py` | `clientavg.py` |
| **FedProx** | 近端项 `μ\|\|w-w_g\|\|²` | 处理系统异构 | `serverprox.py` | `clientprox.py` |
| **FedProto** | 仅交换类别原型，无全局模型 | 通信高效对比 | `serverproto.py` | `clientproto.py` |
| **MOON** | 对比学习（rep→全局，rep↛旧局部） | 数据异构对比 | `servermoon.py` | `clientmoon.py` |
| **FedGH** | 全局分类头 + 原型传输 | FIDSUS 消融对比（无亲和矩阵） | `servergh.py` | `clientgh.py` |
| **GPFL** | GCE + CoV 通用/个性化条件变换 | 个性化 FL 对比 | `servergpfl.py` | `clientgpfl.py` |
| **FedAvgDBE** | 分布偏差消除（running mean） | Non-IID 处理对比 | `serveravgDBE.py` | `clientavgDBE.py` |

### 4.1 FIDSUS vs FedGH（最关键的消融对比）

FedGH 可视作 FIDSUS 的"简化版"——它也有全局 head 训练 + 原型传输，但缺少 FIDSUS 的三个核心创新：

| 特性 | FedGH | FIDSUS |
|------|-------|--------|
| 原型传输 | ✓ | ✓ |
| 全局 head 训练 | ✓ | ✓ |
| 亲和矩阵 P | ✗ | ✓ |
| Top-M 选择性模型下发 | ✗ | ✓ |
| MMD 跨轮特征融合 | ✗ | ✓ |
| 双模型（全局+个性化） | ✗ | ✓ |
| 个性化评估 | ✗ | ✓ |

## 5. 模型架构细节

所有算法共用 `CNN1D` + `BaseHeadSplit`（`models.py`）：

```python
BaseHeadSplit:
    base: Conv1d(1→32) → Conv1d(32→64) → MaxPool → AdaptiveMaxPool
    head: Linear(64 → num_classes)
    
    forward(x) → head(base(x))
```

- `hidden_dim = 32`（全局固定，见 `main.py:28`）
- 对于 NSL-KDD: `num_classes=5`, UNSW: `num_classes=10`
- 分类器 `head` 与特征提取器 `base` 的分离，是实现 FIDSUS 原型提取、全局 head 训练、MMD 融合的关键架构设计

## 6. 数据划分

`dataset/utils/dataset_utils.py` 使用 **Dirichlet 分布**（`α=0.3`）模拟 Non-IID：

```python
proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
```

- `α=0.3` 产生高度倾斜的类别分布（某些客户端只有 1-2 个类别）
- 模拟 UAV 集群中不同节点观测到的攻击类型不一致的真实场景
- 每个客户端数据保存为独立 `.npz` 文件，格式: `{'x': ndarray, 'y': ndarray}`
