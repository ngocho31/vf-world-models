# pytorch-liteflownet3

> Modified PyTorch reimplementation of [LiteFlowNet3 (ECCV 2020)](https://arxiv.org/abs/2007.09319), adapted for the **CaMo-JEPA** pipeline.
>
> Original repo: [sniklaus/pytorch-liteflownet3](https://github.com/sniklaus/pytorch-liteflownet3)

## Overview

This module runs **offline** to extract dense Optical Flow from sequential image frames.  
Output `.npy` files (shape `[2, H, W]`) are consumed by the CaMo-JEPA Dataloader via `PrecomputedFlowReader`.

```
Input images (t, t+1)  ──→  LiteFlowNet3  ──→  flow_<name>.npy [2, H, W]
                                                    ↓
                                          CaMo-JEPA Dataloader
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.x |
| CUDA Toolkit | 12.x+ (with `nvcc`) |
| GPU | NVIDIA with compute capability ≥ 8.9 (e.g. L4, RTX 4090) |

Additional Python packages:
```
numpy, Pillow, tqdm, opencv-python
```

---

## Setup

### 1. Create & activate environment

```bash
conda create -n camo_env python=3.10 -y
conda activate camo_env
pip install torch torchvision numpy Pillow tqdm opencv-python
```

### 2. Fix `libstdc++` (required on some Linux servers)

Add this to your `~/.bashrc` so it persists across sessions:
```bash
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3. Compile the Correlation CUDA kernel (one-time)

```bash
cd correlation_package
python setup.py install
cd ..
```

> **Troubleshooting:** If compilation fails, verify `nvcc --version` matches your PyTorch CUDA version.
> The `setup.py` is configured for `compute_89` (Ada Lovelace / L4). If using a different GPU architecture,
> edit the `nvcc_args` in `correlation_package/setup.py` accordingly.

### 4. Download pretrained weights

```bash
pip install gdown
gdown 1vUSEIxXGZa9d2PQ82SG_gbbIUWLNfH50
# Downloads: network-sintel.pytorch (~30MB) into current directory
```

---

## Usage

### Step 1: Extract Optical Flow

```bash
python extract_optical_flow.py \
  --img_dir <path/to/sequential/images> \
  --out_dir <path/to/output/flow>
```

- Reads all `.png` / `.jpg` / `.jpeg` images from `--img_dir`, sorted by filename.
- For **N** images, produces **N−1** flow files: `flow_<image_name>.npy`
- Each `.npy` has shape `[2, H, W]` — channel 0 is horizontal displacement (u), channel 1 is vertical displacement (v), both in pixels.

**Example with sample images:**
```bash
python extract_optical_flow.py \
  --img_dir ./images \
  --out_dir ./output/flow
```

### Step 2: Visualize Flow (Qualitative — Middlebury Color)

```bash
python visualize_flow.py \
  --input_dir <path/to/flow/npy/files> \
  --output_dir <path/to/output/heatmaps>
```

Uses the standard **Middlebury Color Wheel** encoding (same as academic papers):
- **Color** = direction of movement
- **Saturation** = speed (white = stationary, vivid = fast)

### Step 3: Evaluate Flow Quality (Quantitative — Photometric Warping Error)

```bash
python evaluate_warping.py \
  --img_dir <path/to/sequential/images> \
  --flow_dir <path/to/flow/npy/files>
```

Computes **L1 Photometric Warping Error** — warps Frame 2 backward using the predicted flow and compares it against Frame 1. This metric does **not** require ground truth, making it suitable for real-world data.

Optional: save warped images for visual verification:
```bash
python evaluate_warping.py \
  --img_dir <path/to/images> \
  --flow_dir <path/to/flow> \
  --save_warp_dir <path/to/output/warped>
```

---

## Input Requirements

| Requirement | Detail |
|---|---|
| Image format | `.png`, `.jpg`, `.jpeg` (RGB, 3 channels) |
| Resolution | **Must be multiples of 32** (e.g. 256×512, 384×640) |
| Naming | Files must be named so that `sorted()` gives temporal order |
| Folder structure | Each video/recording in its own directory |

---

## File Structure

```
pytorch-liteflownet3/
├── run.py                     # LiteFlowNet3 model definition (Network class)
├── extract_optical_flow.py    # Main tool: image pairs → .npy flow files
├── visualize_flow.py          # Middlebury Color heatmap visualization
├── evaluate_warping.py        # Photometric Warping Error (no GT needed)
├── network-sintel.pytorch     # Pretrained weights (download separately)
├── correlation_package/       # Custom CUDA correlation kernel
│   ├── setup.py               # Build script (configured for compute_89)
│   ├── correlation.py         # Python wrapper
│   ├── correlation_cuda.cc    # C++ binding
│   └── correlation_cuda_kernel.cu  # CUDA kernel
└── images/                    # Sample test images
    ├── first.png
    └── second.png
```

---

## Modifications from Original Repo

This fork was modified to run on **PyTorch 2.x + CUDA 12/13 + Ada Lovelace GPUs**:

### `correlation_package/setup.py`
- Changed C++ standard from `c++11` → `c++17` (required by PyTorch 2.x)
- Removed deprecated GPU architectures (`compute_50/52/60/61`)
- Added `compute_89` for NVIDIA L4 / RTX 4090

### `correlation_package/correlation_cuda_kernel.cu`
- Replaced deprecated `.type()` API calls with `.scalar_type()` (PyTorch 2.x compatibility)

### New scripts added
- `extract_optical_flow.py` — batch flow extraction with argparse CLI
- `visualize_flow.py` — Middlebury Color Wheel visualization
- `evaluate_warping.py` — Photometric Warping Error evaluation

---

## Integration with CaMo-JEPA

The `.npy` files produced by `extract_optical_flow.py` are consumed by [`src/camo_jepa/motion/flow.py`](../../motion/flow.py) → `PrecomputedFlowReader`, which passes them into the CaMo-JEPA training pipeline.

```
extract_optical_flow.py → .npy files → Dataloader → PrecomputedFlowReader → FlowTokenEncoder → CaMo-JEPA
```

---

## Citation

```bibtex
@inproceedings{hui2020liteflownet3,
    author    = {Tak-Wai Hui and Chen Change Loy},
    title     = {{LiteFlowNet3: Resolving Correspondence Ambiguity for More Accurate Optical Flow Estimation}},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year      = {2020}
}
```
