# UAV-NIDD 数据集审计方法论

## 1. 数据集概述

UAV-NIDD 数据集包含两个视角的网络流量数据：

| 视角 | 来源 | 粒度 | 特征类型 |
|------|------|------|----------|
| **GCS** | 地面控制站 (Ground Control Station) 捕获 | 流级别 (flow-level) | Zeek/flowmeter 导出的 80 个流统计特征 |
| **UAV** | 无人机端 (Unmanned Aerial Vehicle) 捕获 | 包级别 (packet-level) | tshark 导出的 86~116 个无线/网络包特征 |

两个视角的特征空间是**完全不相交**的（disjoint feature spaces），因此无法直接合并，需要在审计中分别处理。

## 2. 审计目标

本审计围绕四个核心问题展开：

| 问题 | 章节 | 内容 |
|------|------|------|
| Q1 | Section 1 | 每种攻击在 GCS/UAV 各有多少样本？ |
| Q2 | Section 2 | 每种攻击有多少可用的数值特征？ |
| Q3 | Section 3 | 同一攻击，GCS vs UAV 哪个受冲击更大？（跨视角对比） |
| Q4 | Section 4 | 同一平台，不同攻击之间如何关联？（视角内分析） |

---

## 3. 数据加载与预处理

### 3.1 文件发现

脚本遍历 `dataset/UAV-NIDD/` 下的 `GCS Case3/` 和 `UAV-Case 1/` 两个目录树，按攻击类型组织 CSV 文件。处理了以下边界情况：

- UAV `De-Authentication/` 目录中的 `FakeLanding.csv` 与 `Fake-Landing Packets/` 中的文件完全相同，仅计数一次
- GCS `Scanning/` 目录中的 `conn.csv`（Zeek conn log，不同 schema）被排除，仅使用 `output1.csv`（flowmeter）
- GCS `MITM/` 目录中的 `Evil Twin.csv` 作为子类型单独处理
- UAV `Replay Attack-UAV/` 仅有 `.pcap` 文件，无 CSV，从 UAV 分析中排除

### 3.2 CSV 格式自动检测

支持三种 CSV 格式，通过读取首行自动识别：

| 格式 | 识别特征 | 分隔符 | 来源 |
|------|----------|--------|------|
| Zeek TSV | 首行含 `#fields` | 制表符 `\t` | GCS flowmeter |
| Plain flowmeter | 首行以 `"uid"` 或 `"ts"` 开头 | 逗号 `,` | GCS flowmeter |
| Tshark export | 首行含 `frame.` | 逗号或制表符 | GCS Replay / UAV 全部 |

### 3.3 数值特征筛选

对于每个攻击和 Normal 基线的 DataFrame，剔除以下非数值列：

- **GCS 非特征列**：标识列（uid, ts, id.orig_h, id.resp_h, ...）、协议元数据（proto, service, conn_state, history）、地址列（ip.src, ip.dst）等
- **UAV 非特征列**：帧元数据（frame.number, frame.time_epoch, ...）、地址列（ip.src, ip.dst, wlan.sa, wlan.da, ...）、枚举/分类列（wlan.fc.type, tcp.flags.syn, ...）

仅保留数值型列（int/float dtype），得到 $F_k$ 个数值特征。

### 3.4 特征统计量计算

对于每种攻击 $k$ 和 Normal 基线 $N$，对每个数值特征 $f_j$ 计算以下统计量：

**样本量**：$n_k$（攻击 $k$ 的数据行数）

**均值**：

$$\mu_{k,j} = \frac{1}{n_k} \sum_{i=1}^{n_k} x_{i,j}$$

**标准差**（总体）：

$$\sigma_{k,j} = \sqrt{\frac{1}{n_k} \sum_{i=1}^{n_k} (x_{i,j} - \mu_{k,j})^2}$$

**最小值**：

$$\min_{k,j} = \min_{i=1}^{n_k} x_{i,j}$$

**最大值**：

$$\max_{k,j} = \max_{i=1}^{n_k} x_{i,j}$$

所有计算使用 float32 精度以控制内存占用。

---

## 4. 偏离度计算（核心方法）

### 4.1 问题与动机

原方法直接使用 Cohen's d 计算原始特征上的归一化偏离度：

$$d_j^{\text{raw}} = \frac{|\mu_{k,j} - \mu_{N,j}|}{\sqrt{(\sigma_{k,j}^2 + \sigma_{N,j}^2)/2}}$$

然而，由于不同特征的量纲差异巨大（例如 `flow_SYN_flag_count` 的取值范围可达 $10^6$，而 `active.min` 仅有 $10^0 \sim 10^2$），原始 Cohen's d 使得大尺度特征天然主导了偏离度排名，不同特征之间不可比。

### 4.2 解决方案：Min-Max 归一化 + Cohen's d

两步处理消除量纲影响：

**Step 1 — Min-Max 归一化到 [0, 1]**

以 Normal 基线的观测范围为基准，将每个特征线性映射到 [0, 1] 区间：

$$\tilde{\mu}_{k,j} = \frac{\mu_{k,j} - \min_{N,j}}{\max_{N,j} - \min_{N,j}}$$

$$\tilde{\sigma}_{k,j} = \frac{\sigma_{k,j}}{\max_{N,j} - \min_{N,j}}$$

对于在 Normal 基线上范围为 0 的常数特征（$\max_{N,j} - \min_{N,j} \leq 10^{-12}$），设置 $\tilde{\mu}_{k,j} = \tilde{\sigma}_{k,j} = 0$（无判别力，$d = 0$）。

**Step 2 — 在归一化值上计算 Cohen's d**

$$d_{k,j} = \frac{|\tilde{\mu}_{k,j} - \tilde{\mu}_{N,j}|}{\sqrt{(\tilde{\sigma}_{k,j}^2 + \tilde{\sigma}_{N,j}^2)/2}}$$

归一化后，所有特征无量纲，$d_{k,j}$ 的值可直接跨特征比较。

### 4.3 偏离度矩阵

对于平台 $P \in \{\text{GCS}, \text{UAV}\}$，所有攻击的偏离度构成矩阵 $\mathbf{D} \in \mathbb{R}^{A \times F}$：

$$\mathbf{D} = \begin{bmatrix} d_{1,1} & d_{1,2} & \cdots & d_{1,F} \\ d_{2,1} & d_{2,2} & \cdots & d_{2,F} \\ \vdots & \vdots & \ddots & \vdots \\ d_{A,1} & d_{A,2} & \cdots & d_{A,F} \end{bmatrix}$$

其中 $A$ 为攻击数，$F$ 为共同特征数（仅保留所有攻击 + Normal 共有的数值特征列交集）。

---

## 5. 审计分析维度

### 5.1 样本量统计（Section 1）

统计每种攻击在 GCS/UAV 上的数据行数，并汇总总行数。标识仅在单侧存在的攻击。

### 5.2 特征数量统计（Section 2）

统计每种攻击的数值特征数量，以及 GCS/UAV 各自使用的 CSV 格式。

### 5.3 跨视角对比（Section 3）

对于同时存在于 GCS 和 UAV 的攻击，分别计算各侧 Top-10 最大偏离特征，并输出两侧的平均偏离度：

$$\overline{d}^{\text{GCS}}_k = \frac{1}{F} \sum_{j=1}^{F} d_{k,j}^{\text{GCS}}, \quad \overline{d}^{\text{UAV}}_k = \frac{1}{F} \sum_{j=1}^{F} d_{k,j}^{\text{UAV}}$$

**解读规则**：若 $\overline{d}^{\text{GCS}}_k > \overline{d}^{\text{UAV}}_k$，则该攻击对 GCS（流层面）的冲击大于对 UAV（包层面）的冲击；反之亦然。

对于仅在单侧存在的攻击，仅计算该侧的偏离度（Section 3b）。

### 5.4 视角内分析（Section 4）

在**同一平台**内分析攻击间的关系：

#### 4a. 攻击冲击力排名

按 $\overline{d}_k$ 降序排列：

$$\text{Rank}_k = \frac{1}{F} \sum_{j=1}^{F} d_{k,j}$$

值越大，该攻击对该平台的总体冲击越大。

#### 4b. 最具攻击区分度的特征

计算每个特征偏离度在攻击间的标准差（高方差 = 攻击类型敏感）：

$$\sigma_j^{(d)} = \sqrt{\frac{1}{A} \sum_{k=1}^{A} \left( d_{k,j} - \frac{1}{A}\sum_{m=1}^{A} d_{m,j} \right)^2}$$

$\sigma_j^{(d)}$ 大的特征说明其偏离程度依赖攻击类型，是攻击识别的关键标志。

#### 4c. 最能捕捉攻击共性的特征

计算每个特征偏离度在攻击间的均值（高均值 = 普适性冲击）：

$$\overline{d}_j = \frac{1}{A} \sum_{k=1}^{A} d_{k,j}$$

#### 4d. 攻击相似性矩阵

对每对攻击 $(k, l)$，计算偏离度向量的 Pearson 相关系数：

$$r_{k,l} = \frac{\sum_{j=1}^{F} (d_{k,j} - \overline{d}_k)(d_{l,j} - \overline{d}_l)}{\sqrt{\sum_{j=1}^{F} (d_{k,j} - \overline{d}_k)^2} \cdot \sqrt{\sum_{j=1}^{F} (d_{l,j} - \overline{d}_l)^2}}$$

$r_{k,l} \to 1$ 表示两种攻击的偏离模式相似（对相同特征产生类似冲击），$\to 0$ 或负值表示模式正交。

#### 4e. 单特征最大偏离速查

$$\max_j d_{k,j}$$

---

## 6. 报告输出

审计报告输出到 `docs/audit/uav_nidd_audit_report.txt`，包含：

| 章节 | 内容 |
|------|------|
| Data Sources | 数据源路径、文件清单、CSV 格式说明 |
| Section 1 | 样本量对比表 |
| Section 2 | 特征数量与格式表 |
| Section 3 | 跨视角偏离度对比（含 Top-10 特征和解读） |
| Section 3b | 单侧攻击偏离度 |
| Section 4a-e | 视角内分析（冲击排名、特征敏感性、相似性矩阵、速查表） |
| Section 5 | 解读指南 |

---

## 7. 运行方式

```bash
uv run python scripts/audit_uav_nidd_deep.py
```

前置依赖：`pandas`, `numpy`（由 `uv` 管理）。

## 8. 注意事项

1. **特征空间不交**：GCS（流特征）和 UAV（包特征）使用完全不同的特征集，偏离度值 $d$ 仅在**同平台内**严格可比。跨平台的 $\overline{d}$ 对比反映的是各自平台的相对冲击程度，而非绝对量级。
2. **Normal 基线样本量**：GCS Normal 仅 109 行（10 个子文件夹各约 10 行），UAV Normal 约 11,733 行。基线样本量差异可能影响统计稳定性和归一化范围。
3. **常数特征**：Normal 基线中范围为 0 的特征（常数）被置为 $d = 0$。这些特征可能在攻击中仍有变化值，但由于基线无变化，无法归一化。
4. **Replay 攻击**：GCS Replay 数据使用 tshark export 格式（10 个特征），与主流 flowmeter 格式不同（80 个特征），其偏离度仅基于 10 个共同特征计算。
