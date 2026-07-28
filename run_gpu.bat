@echo off
REM ==========================================================================
REM  FIDSUS local GPU training script (Windows .bat)
REM  --------------------------------------------------------------
REM  Produces a 3-caliber summary:
REM    1. Personalized   (model_per, local)
REM    2. Global head    (model_per.base + server global head)  [FIDSUS only]
REM    3. Macro-F1       (class-balanced)
REM  Rare-class metrics are reported as tail-10-round mean +/- std.
REM
REM  torch CUDA variant auto-detection (via nvidia-smi compute_cap):
REM    compute_cap <  7.5  -> cu118  (Pascal/Maxwell, e.g. GT 1030 sm_61)
REM    compute_cap >= 7.5  -> cu126  (Turing and newer, e.g. RTX 4090 sm_89)
REM    no GPU              -> cpu
REM  Override with: set TORCH_VARIANT=cu118   (or cu126 / cpu)
REM
REM  Usage:
REM     run_gpu.bat                 Full experiment (100 rounds)
REM     run_gpu.bat --quick         Quick smoke (2 rounds / 3 clients)
REM     run_gpu.bat --summary-only  Only summarize existing results
REM
REM  This file is intentionally ASCII-only to avoid cmd codepage issues.
REM ==========================================================================
setlocal EnableDelayedExpansion

cd /d "%~dp0"
set "ROOT=%CD%"

set "QUICK=0"
set "SUMMARY_ONLY=0"
if "%~1"=="--quick" set "QUICK=1"
if "%~1"=="--summary-only" set "SUMMARY_ONLY=1"

echo ============================================================
echo  FIDSUS GPU training (3-caliber evaluation)
echo  Working dir: %ROOT%
echo ============================================================

REM ---- 1. Detect GPU compute capability -> choose torch variant ----
echo.
echo [1/5] Detecting GPU ...
set "VARIANT=%TORCH_VARIANT%"
if "%VARIANT%"=="" (
    set "NSMI="
    where nvidia-smi >nul 2>&1 && set "NSMI=nvidia-smi"
    if not exist "NSMI" if exist "C:\Windows\System32\nvidia-smi.exe" set "NSMI=C:\Windows\System32\nvidia-smi.exe"

    if "!NSMI!"=="" (
        echo   nvidia-smi not found, will use CPU torch.
        set "VARIANT=cpu"
    ) else (
        "!NSMI!" --query-gpu=name,compute_cap --format=csv,noheader 2>nul | findstr /R "." >nul
        if errorlevel 1 (
            echo   nvidia-smi query failed, will use CPU torch.
            set "VARIANT=cpu"
        ) else (
            for /f "tokens=1,2 delims=," %%a in ('"!NSMI!" --query-gpu=name,compute_cap --format=csv,noheader 2^>nul') do (
                set "GPU_NAME=%%a"
                set "CC=%%b"
                goto :gotcap
            )
            :gotcap
            echo   GPU: !GPU_NAME!  compute_cap=!CC!
            REM Parse major.minor -> compare. Strip spaces.
            set "CC=!CC: =!"
            for /f "tokens=1,2 delims=." %%x in ("!CC!") do (
                set "CMAJ=%%x"
                set "CMIN=%%y"
            )
            if "!CMIN!"=="" set "CMIN=0"
            set /a "CCINT=CMAJ*10+CMIN" 2>nul
            if !CCINT! LSS 75 (
                set "VARIANT=cu118"
            ) else (
                set "VARIANT=cu126"
            )
        )
    )
)
echo   torch variant: %VARIANT%

REM ---- 2. Install uv if missing ----
echo.
echo [2/5] Ensuring uv ...
where uv >nul 2>&1
if errorlevel 1 (
    echo   uv not found, installing via powershell ...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" >nul 2>&1
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv not available. Install from https://docs.astral.sh/uv/
    exit /b 1
)

REM ---- 3. Sync dependencies with detected CUDA variant ----
echo.
echo [3/5] uv sync --extra %VARIANT% --group dev ...
uv sync --extra "%VARIANT%" --group dev
if errorlevel 1 (
    echo [ERROR] uv sync failed for variant=%VARIANT%
    exit /b 1
)
set "RUN=uv run python"

REM ---- 4. Verify CUDA availability (warn if CPU-only torch) ----
echo.
echo [4/5] Verifying torch / CUDA ...
for /f "delims=" %%i in ('%RUN% -c "import torch; print(torch.__version__); print('1' if torch.cuda.is_available() else '0'); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"') do (
    echo   %%i
)
REM Re-query cuda availability to decide fallback
%RUN% -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
    echo   ----------------------------------------------------------------
    echo   [WARN] torch is CPU-only but training requested CUDA.
    echo   This usually means nvidia driver missing, OR torch came from a
    echo   CPU index. Falling back to CPU training (will be slow).
    echo   To force a CUDA rebuild: set TORCH_VARIANT=cu126 ^& run again
    echo   ----------------------------------------------------------------
    set "DEVICE=cpu"
) else (
    set "DEVICE=cuda"
)

cd /d "%ROOT%\system"

if "%QUICK%"=="1" (
    echo.
    echo [quick mode] rounds=2 clients=3 device=cpu
    %RUN% run_all.py --quick
    goto :done
)
if "%SUMMARY_ONLY%"=="1" (
    echo.
    echo [summary-only] aggregating existing results ...
    %RUN% run_all.py --summary-only
    goto :done
)

REM ---- 5. Run experiment ----
echo.
echo [5/5] Starting experiment (device=%DEVICE%) ...
%RUN% run_all.py --device %DEVICE% %*

:done
echo.
echo ============================================================
echo  Done. Results in results\ , summary in results\summary.csv
echo  Three calibers: personalized / global-head / Macro-F1
echo ============================================================
endlocal
