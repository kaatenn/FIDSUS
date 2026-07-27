@echo off
REM ==========================================================================
REM  FIDSUS local GPU training script (Windows .bat)
REM  --------------------------------------------------------------
REM  Runs the rare-label comparison and produces a 3-caliber summary:
REM    1. Personalized   (model_per, local)
REM    2. Global head    (model_per.base + server global head)  [FIDSUS only]
REM    3. Macro-F1       (class-balanced)
REM  Rare-class metrics are reported as tail-10-round mean +/- std.
REM
REM  Target GPU: NVIDIA Pascal (e.g. GT 1030 / GTX 1060, sm_61)
REM              torch is pinned to 2.6.* + cu118 (last build with sm_61)
REM
REM  Usage:
REM     run_gpu.bat                 Full experiment (100 rounds, NSLKDD + UNSW, 3 algos)
REM     run_gpu.bat --quick         Quick smoke (2 rounds / 3 clients, verify GPU + code)
REM     run_gpu.bat --summary-only  Only summarize existing results (no training)
REM     run_gpu.bat --datasets UNSW --algos FIDSUS FIDSUS_no_fusion
REM
REM  Note:
REM     - Prefers uv (uv sync); falls back to .venv pip if uv is not in PATH.
REM     - GT 1030 has only 4GB VRAM and is weak; it may be SLOWER than CPU.
REM     - This file is intentionally ASCII-only to avoid cmd codepage issues.
REM ==========================================================================
setlocal EnableDelayedExpansion

cd /d "%~dp0"
set "ROOT=%CD%"
set "VENV_PY=%ROOT%\.venv\Scripts\python.exe"

echo ============================================================
echo  FIDSUS local GPU training (3-caliber evaluation)
echo  Working dir: %ROOT%
echo ============================================================

REM ---- 1. Detect GPU ----
echo.
echo [1/4] Detecting NVIDIA GPU ...
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    if exist "C:\Windows\System32\nvidia-smi.exe" (
        "C:\Windows\System32\nvidia-smi.exe" --query-gpu=name,driver_version,memory.total --format=csv,noheader
    ) else (
        echo [WARN] nvidia-smi not found. Cannot confirm GPU. Will fall back to CPU if CUDA is unavailable.
    )
) else (
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
)

REM ---- 2. Install / sync dependencies ----
echo.
echo [2/4] Syncing dependencies (torch 2.6.* + cu118) ...
where uv >nul 2>&1
if errorlevel 1 (
    echo uv is not in PATH, trying existing .venv.
    if not exist "%VENV_PY%" (
        echo [ERROR] No uv found and no .venv. Install uv first: https://docs.astral.sh/uv/
        exit /b 1
    )
    echo Installing cu118 torch via pip with %VENV_PY% ...
    "%VENV_PY%" -m pip install "torch>=2.6,<2.7" --index-url https://download.pytorch.org/whl/cu118
    if errorlevel 1 (
        echo [ERROR] torch cu118 install failed.
        exit /b 1
    )
    set "RUN=%VENV_PY%"
) else (
    echo Syncing environment with uv sync ...
    uv sync
    if errorlevel 1 (
        echo [ERROR] uv sync failed.
        exit /b 1
    )
    set "RUN=uv run python"
)

REM ---- 3. Verify CUDA availability ----
echo.
echo [3/4] Verifying CUDA availability ...
%RUN% -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (CUDA unavailable)')"
if errorlevel 1 (
    echo [ERROR] torch import failed.
    exit /b 1
)

echo.
echo If cuda_available is False above, training will fall back to CPU.
echo Press Ctrl+C to cancel, or any key to continue ...
pause >nul

REM ---- 4. Run experiment ----
echo.
echo [4/4] Starting experiment ...
cd /d "%ROOT%\system"

REM Skip the pause/confirm for summary-only mode (no training needed)
echo "%~1" | findstr /C:"--summary-only" >nul 2>&1
if not errorlevel 1 goto :run

:run
if "%~1"=="" (
    echo Mode: full experiment (100 rounds, NSLKDD + UNSW, 3 algos)
    %RUN% run_all.py --device cuda %*
) else (
    echo Mode: args %*
    %RUN% run_all.py --device cuda %*
)
set "RC=%ERRORLEVEL%"

echo.
echo ============================================================
echo  Done. Results in results\ , summary in results\summary.csv
echo  Three calibers: personalized / global-head / Macro-F1
echo ============================================================
exit /b %RC%
