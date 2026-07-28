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
uv sync --group dev   # 安装运行依赖 + 测试依赖
```

### GPU support (auto-detected)

`pyproject.toml` declares **three mutually-exclusive torch variants** and the
run scripts auto-pick one based on your GPU's compute capability:

| GPU compute capability | Variant | Example hardware | torch build |
|------------------------|---------|------------------|-------------|
| `< 7.5` (Pascal/Maxwell) | `cu118` | GT 1030, GTX 1080 | torch 2.6.* + cu118 (last sm_61 build) |
| `>= 7.5` (Turing and newer) | `cu126` | RTX 20/30/40/50 series | torch (latest) + cu126 |
| no GPU | `cpu` | — | torch (latest) CPU |

The scripts (`run_experiments.sh` on Linux/WSL, `run_gpu.bat` on Windows) will:
1. query `nvidia-smi --query-gpu=compute_cap` to detect your GPU,
2. install `uv` if missing,
3. run `uv sync --extra <variant> --group dev` to build the right environment,
4. warn loudly and fall back to CPU if torch still has no CUDA.

```bash
# Linux / WSL (auto-detects GPU, e.g. RTX 4090 -> cu126)
bash run_experiments.sh
# Windows (auto-detects GPU, e.g. GT 1030 -> cu118)
run_gpu.bat
```

Manual override (skip auto-detection):

```bash
TORCH_VARIANT=cu126 bash run_experiments.sh     # force a specific variant
set TORCH_VARIANT=cu118 && run_gpu.bat          # Windows
```

You can also sync the environment directly without the scripts:

```bash
uv sync --extra cu126 --group dev   # pick cu118 | cu126 | cpu
```

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


