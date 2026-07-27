# FIDSUS

The source code is for the paper: FIDSUS: Federated Intrusion Detection for Securing UAV Swarms

## Overview

The dynamic nature of UAV swarms, characterized by communication instability, heterogeneous nodes, and frequent topology changes, makes them vulnerable to network attacks. To address these challenges, we propose FIDSUS, a federated learning-based intrusion detection system. FIDSUS quantifies the similarity between UAVs' local feature extractors using an affinity matrix, facilitating knowledge sharing and enhancing the robustness of local models. By employing cross-round feature fusion, FIDSUS mitigates the forgetting problem caused by data heterogeneity, improving detection accuracy in complex scenarios. The global classifier, trained on fused feature representations, further boosts generalization performance. Experimental results demonstrate FIDSUS's superior performance compared to existing FL approaches, achieving an average accuracy improvement of 4% to 34% in large-scale scenarios. FIDSUS exhibits exceptional robustness and accuracy in dynamic client environments while maintaining competitive training efficiency.


## Dependencies

This project requires the following dependencies to be installed:

### Packages (runtime)

Only what `system/` actually imports at runtime:

- `torch` (CUDA build for GPU; see GPU notes below)
- `numpy`
- `h5py`
- `scikit-learn`

Test deps (`uv sync --group dev`): `pytest`.
Dataset (re)generation (`uv sync --group dataset`): `ujson`.

> The legacy `env.yaml` listed many unused packages (pandas, torchaudio, torchtext,
> torchvision, opacus, cvxpy, ...). They are **not** imported by the codebase and
> have been removed from `pyproject.toml`.

### Installation (uv, recommended)

This project is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync            # 安装运行依赖
uv sync --group dev  # 额外安装测试依赖 (pytest)
```

### GPU notes (Pascal / sm_61 GPUs like GT 1030)

`pyproject.toml` locks `torch>=2.6,<2.7` from the **cu118** index because
PyTorch 2.7+ dropped support for Pascal (sm_61) GPUs. The cu118 wheels of 2.6.x
are the last builds that still include sm_61 binaries. A one-command Windows
script is provided:

```bat
run_gpu.bat            REM 完整实验 (100 轮)
run_gpu.bat --quick    REM 快速验证 GPU 是否可用
```

If you upgrade to a newer GPU (sm_75+), relax the constraint to `torch>=2.6`
and switch the index to cu126/cu128 in `pyproject.toml`.

### Installation (Conda, legacy)

You can also create a conda environment using the provided `env.yaml` file:

```bash
conda env create -f env.yaml
```

## Dataset

The dataset used for experimentation is the NSL-KDD and UNSW-NB15 dataset. We have pre-partitioned the USNW-NB15 dataset into 50 subsets according to a Dirichlet distribution to facilitate training. You can modify the partitioning method by adjusting the parameters in the `generate_unsw.py` file.

## Core Components

The core components of this project, which are central to the algorithms discussed in our paper, are implemented in the following files:


- **Client Implementation**: `system/flcore/clients/clientFIDSUS.py`
- **Server Implementation**: `system/flcore/servers/serverFIDSUS.py`

## Running the Project

### One-command experiments (recommended)

Run the full rare-label comparison (FIDSUS vs FedAvg vs FIDSUS_no_fusion on NSL-KDD + UNSW-NB15, 100 rounds) with a single command:

```bash
cd system
uv run python run_all.py              # full experiment (needs GPU)
uv run python run_all.py --quick      # quick smoke test (CPU, ~1 min)
```

This produces per-algorithm `.h5` result files plus a `results/summary.csv` with rare-label precision/recall and convergence-speed comparison. See [`docs/usage-guide.md`](docs/usage-guide.md) §7–§11 for details.

A bash equivalent is also provided at the repo root:

```bash
bash run_experiments.sh            # full
QUICK=1 bash run_experiments.sh    # smoke
```

### Single run

To run the project, navigate to the `system` directory and execute the following command:

```bash
cd system
uv run python main.py -algo FIDSUS -data UNSW -nc 50 -gr 100
```

### Tests

```bash
uv run pytest tests/ -v
```


