# UAV-FGS Anonymous Review Code

This package contains the anonymous review code for the paper's RGB-T UAV 3D Gaussian Splatting pipeline. It is a minimal, self-contained adaptation of the original 3D Gaussian Splatting codebase, with the additional scripts required for:

- CFR-based raw RGB-T standardization from cross-FoV image pairs
- RGB-stage 3DGS reconstruction
- thermal-stage 3DGS transfer/training
- controllable RGB-T fusion and sweep evaluation
- GT-view and auxiliary evaluation used by the paper

This review package intentionally omits non-essential viewer/build artifacts, internal campaign scripts, summary generators, and local experiment records.

## Quick Start (Reviewer, Copy-Paste)

This repository is derived from the original 3D Gaussian Splatting codebase, and the review-time setup steps and checks needed for reproduction are listed here.

### 1) System prerequisites

The CUDA extensions in `submodules/` require both:

- a system CUDA toolkit with `nvcc`
- a system C/C++ compiler

The Conda environment below installs the PyTorch CUDA runtime, but it does not replace the system compiler toolchain needed to build the extensions.

Windows prerequisites:

- NVIDIA GPU and driver
- CUDA toolkit with `nvcc` available on `PATH` (CUDA 11.8 is the intended match here)
- Visual Studio 2019 or 2022 Build Tools with the MSVC x64 C/C++ toolchain
- a shell where `cl` is available
- COLMAP on `PATH` (or pass `--colmap <path>`)
- ExifTool on `PATH` (or pass `--exiftool <path>`)

Recommended Windows shell:

- Start from `x64 Native Tools Command Prompt for VS 2019/2022`, or any shell where `cl` is already available before installing the CUDA extensions.

Windows preflight checks:

```powershell
where.exe nvcc
where.exe cl
where.exe colmap
where.exe exiftool
```

Linux prerequisites:

- NVIDIA GPU and driver
- CUDA toolkit with `nvcc` available on `PATH`
- GCC/G++ toolchain
- COLMAP on `PATH`
- ExifTool on `PATH`

Linux example (Ubuntu):

```bash
sudo apt-get update
sudo apt-get install -y build-essential gcc g++ cmake ninja-build colmap libimage-exiftool-perl
```

Linux preflight checks:

```bash
which nvcc
which g++
which colmap
which exiftool
```

### 2) Create environment

Recommended explicit setup:

```bash
conda create -n uav-fgs python=3.10.18 pip=25.2 numpy=1.26.4 -y
conda activate uav-fgs
conda install -y pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia -c defaults
python -m pip install -r requirements.txt
```

Equivalent one-shot alternative:

```bash
conda env create -f environment.yml
conda activate uav-fgs
```

### 3) Install CUDA extensions (required)

Run these commands from the repository root, in the same shell that already sees `nvcc` and the system compiler:

```bash
python -m pip install --no-build-isolation submodules/simple-knn
python -m pip install --no-build-isolation submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation submodules/fused-ssim
```

Standard non-editable installs are intentional here. They are sufficient for review reproduction and are more robust than editable installs across different `pip` and `setuptools` versions.

### 4) Sanity checks

Do not rely only on the `conda` or `pip` exit text. Always run the checks below after installation, especially if package manager output included warnings such as cache, clobber, or safety messages.

Toolchain checks:

Windows:

```powershell
where.exe nvcc
where.exe cl
where.exe colmap
where.exe exiftool
```

Linux:

```bash
which nvcc
which g++
which colmap
which exiftool
```

Python/runtime checks:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import cv2, diff_gaussian_rasterization, simple_knn, fused_ssim; print('extensions OK')"
python train.py -h
python render.py -h
python metrics.py -h
python cfr.py -h
python run_uavfgs_pipeline.py -h
```

### 5) Smoke tests

Dataset layout:

```text
<DATA_ROOT>/
  rgb/
  thermal/
```

If `--rgb_dir` is omitted, the pipeline first looks for `<DATA_ROOT>/RGB` and then falls back to `<DATA_ROOT>/rgb`.

Lightweight CFR smoke test on a real scene:

```bash
python cfr.py --rgb_dir <DATA_ROOT>/rgb --th_dir <DATA_ROOT>/thermal --out_dir <TMP_CFR_OUT> --samples 3 --align fit --stage both --comparison
```

Lightweight pipeline smoke test:

1. Prepare a tiny paired subset at:

```text
<TINY_ROOT>/
  rgb/
  thermal/
```

2. Copy a few paired RGB/T images from one scene into the two folders above.

3. Run the first two pipeline steps:

```bash
python run_uavfgs_pipeline.py --data_root <TINY_ROOT> --out_root <TINY_OUT> --rgb_dir <TINY_ROOT>/rgb --to_step 2
```

### 6) Run full pipeline

Full dataset layout:

```text
<DATA_ROOT>/
  rgb/
  thermal/
```

Main command:

```bash
python run_uavfgs_pipeline.py --data_root <DATA_ROOT> --out_root <OUT_ROOT>
```

Equivalent explicit lowercase-RGB form:

```bash
python run_uavfgs_pipeline.py --data_root <DATA_ROOT> --out_root <OUT_ROOT> --rgb_dir <DATA_ROOT>/rgb
```

The pipeline is resumable by default. To rerun from scratch, use cleaning flags such as:

`--clean_fit --clean_input --clean_thermal_ud --clean_blend_out --force`

## Package Layout

The package keeps the same root-level structure expected by the original 3DGS codebase:

- `train.py`, `render.py`, `metrics.py`: inherited 3DGS training/rendering/metric entry points
- `run_uavfgs_pipeline.py`: one-command end-to-end pipeline used in this work
- `cfr.py`: crop/FoV/resolution standardization for raw RGB-T pairs
- `convert_uavfgs.py`: COLMAP + input conversion helper
- `eval_crop_metrics.py`: crop/alignment evaluation helper
- `metrics_plus.py`: extended GT-view metrics and auxiliary diagnostics
- `novel_view_metrics.py`: optional no-reference novel-view evaluation
- `blend_model_strict_endpoints.py`: RGB-T Gaussian fusion/export
- `eval_blend_sweep.py`: fusion-sweep evaluation
- `arguments/`, `gaussian_renderer/`, `scene/`, `utils/`, `lpipsPyTorch/`: required runtime code
- `submodules/`: vendored CUDA extensions required by the optimizer/renderer

## Environment

This repository is tested with:

- Python `3.10`
- PyTorch `2.0.1` + TorchVision `0.15.2`
- CUDA runtime `11.8` via `pytorch-cuda=11.8`
- Conda-based setup

The environment spec is:

- `environment.yml` for Conda + PyTorch/CUDA
- `requirements.txt` for the remaining Python packages

`requirements.txt` intentionally relies on the default PyPI index so that link-scrubbing on anonymous review platforms does not affect installation.

The package uses `opencv-python-headless` because the review code does not require OpenCV GUI windows. `opencv-python` is also acceptable if preferred locally.

Important note for extension builds:

- PyTorch CUDA runtime from Conda is not enough by itself
- the extension build additionally requires system `nvcc`
- the extension build additionally requires a visible system compiler (`cl` on Windows, `g++` on Linux)

## External Tools

The full raw-pair pipeline expects:

- COLMAP on `PATH`, or pass `--colmap <path>`
- ExifTool on `PATH`, or pass `--exiftool <path>`

`cfr.py` has fallback behavior when ExifTool is unavailable, but the full review pipeline should still be configured with COLMAP and ExifTool available.

## Data Assumptions

The end-to-end pipeline expects a dataset root of the form:

```text
<DATA_ROOT>/
  rgb/
  thermal/
```

The full benchmark and anonymous review access information should be provided separately in the review materials.

## Main Entry Point

The primary entry point used in this project is:

```bash
python run_uavfgs_pipeline.py --data_root <DATA_ROOT> --out_root <OUT_ROOT>
```

This orchestrates the review package pipeline in order:

1. CFR standardization from raw RGB-T pairs
2. crop/alignment evaluation
3. COLMAP conversion and reconstruction
4. stage-1 RGB 3DGS training/rendering/metrics
5. thermal undistortion and layout normalization
6. stage-2 thermal 3DGS training/rendering/metrics
7. RGB-T Gaussian blending
8. fusion-sweep evaluation

The script is resumable by default and writes per-step state files under:

```text
<DATA_ROOT>/_pipeline_state/
```

## Important Defaults

The review package follows the current code defaults, not older README snapshots.

- `run_metrics_plus=true`
- `run_novel_view_metrics=false`
- fusion `dc_y_from=lerp`
- blend `endpoint_mode=blend`

Please treat `run_uavfgs_pipeline.py -h` as the authoritative source of current CLI defaults.

## Minimal Direct Usage

If you want to run steps separately, the main scripts are:

```bash
python cfr.py -h
python convert_uavfgs.py -h
python train.py -h
python render.py -h
python metrics.py -h
python metrics_plus.py -h
python novel_view_metrics.py -h
python blend_model_strict_endpoints.py -h
python eval_blend_sweep.py -h
```

## Troubleshooting

- If `cl` is missing on Windows, reopen the session from a Visual Studio developer prompt and reinstall the CUDA extensions from that shell.
- If `nvcc` is missing, install a system CUDA toolkit and reopen the shell before building the extensions.
- If `conda env create` or `conda install` prints cache, safety, or clobber warnings, complete the installation, then run the sanity checks above. If the sanity checks fail, clean the Conda cache and recreate the environment.
- If extension installation fails, first confirm that `python -c "import torch"` works in the active environment, then confirm that `nvcc` and the system compiler are visible in the same shell.

## Notes on Scope

- This package is prepared for anonymous academic review.
- It intentionally excludes non-core viewers, build caches, internal analysis scripts, and local result tables.
- It preserves third-party license headers and upstream attribution where required by inherited components.
