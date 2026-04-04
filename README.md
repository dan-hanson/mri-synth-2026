# MRI Missing Modality Synthesis (Diffusion-Based)

## Overview

This project implements a **3D diffusion-based model** for **missing MRI modality synthesis** using BraTS-style datasets.
Given 3 MRI modalities (e.g., T1, T2, FLAIR), the model learns to generate the missing fourth modality.

The system is designed to be:

* modular (multiple backbones supported)
* reproducible (config-driven)
* extensible (ready for experimentation with architectures and diffusion strategies)

---

## What This Repo Currently Does

### Core Pipeline

* ✅ Loads BraTS-format 3D MRI volumes (`.nii.gz`)
* ✅ Normalizes data (z-score)
* ✅ Randomly drops one modality during training
* ✅ Uses a **single model for all missing-modality combinations**
* ✅ Trains a **diffusion model (noise prediction)**
* ✅ Supports **full-volume inference via sliding window**
* ✅ Saves:

  * NIfTI outputs
  * PNG slice panels
  * JSON summaries

---

### Model & Training Features

* ✅ Config-driven architecture selection
* ✅ Modular model registry (UNet, ConvNeXt-ready, etc.)
* ✅ Diffusion timestep conditioning
* ✅ Mixed precision training (AMP)
* ✅ Learning rate scheduling
* ✅ Composite loss:

  * Noise MSE (core diffusion loss)
  * MAE (reconstruction)
  * SSIM (structure)
* ✅ EMA (Exponential Moving Average) for stabilization
* ✅ Validation:

  * light (loss only)
  * heavy (MAE, PSNR, SSIM)

---

### Inference Features

* ✅ Full 3D reconstruction using **sliding-window inference**
* ✅ Configurable reverse diffusion steps
* ✅ Reproducible case selection:

  * fixed list OR
  * seeded random subset
* ✅ Outputs:

  * predicted volume (.nii.gz)
  * visualization panel (.png)
  * metrics summary (.json)

---

### Conditioning Strategy

* 4 modality slots (T1, T1ce, T2, FLAIR)
* Missing modality handled via:

  * configurable fill value
  * optional presence mask (recommended)

---

## Project Structure (Simplified)

```
src/mri_missing/
│
├── config.py
├── data/
│   ├── dataset.py
│   └── test_loader.py
│
├── engine/
│   ├── train.py
│   ├── infer.py
│   └── train_minimal.py
│
├── models/
│   ├── registry.py
│   ├── unet3d.py
│   └── convnext3d.py (planned/partial)
│
├── diffusion/
│   └── scheduler.py
│
├── losses/
│   └── image_losses.py
│
├── metrics/
│   └── image_metrics.py
│
├── utils/
│   ├── ema.py
│   ├── io.py
│   ├── cases.py
│   └── vis.py
│
configs/
└── base.yaml
```

---

## Setup Instructions

### 1. Clone repo

```bash
git clone <repo-url>
cd mri-synth-2026
```

### 2. Create environment

Using micromamba (recommended):

```bash
micromamba create -f environment.yml
micromamba activate mri_v1
```

Alternative:

```bash
pip install -r requirements.txt
```

> ⚠️ You may want to install PyTorch manually with CUDA support depending on your system.

---

### 3. Download dataset

Use BraTS-style dataset structure:

```
data/
└── GLI/
    ├── TrainingData/
    │   ├── BraTS-GLI-xxxxx/
    │   │   ├── *-t1n.nii.gz
    │   │   ├── *-t1c.nii.gz
    │   │   ├── *-t2w.nii.gz
    │   │   ├── *-t2f.nii.gz
    │   │   └── *-seg.nii.gz
```

Update paths in:

```yaml
configs/base.yaml
```

---

### 4. Run training

```bash
python src/mri_missing/engine/train.py
```

---

### 5. Run inference

```bash
python src/mri_missing/engine/infer.py
```

Outputs:

```
outputs/
└── inference/
    ├── *.nii.gz
    ├── *.png
    └── summary.json
```

---

## Key Config Parameters

### Training

```yaml
train:
  max_steps: ...
  batch_size: ...
  use_amp: true
```

### Diffusion

```yaml
diffusion:
  timesteps: 1000
```

### Inference

```yaml
inference:
  reverse_steps: 50   # important for speed vs quality
```

### Missing Modality Strategy

```yaml
missing_policy:
  fill_value: 1.0
  use_presence_mask: true
```

---

## Current Limitations

* Small baseline model (~350k params)
* No pretrained backbone yet
* Diffusion variant is simplified (noise prediction only)
* Sliding window inference is correct but still slow at high steps
* No segmentation-based evaluation yet
* No distributed or multi-GPU training

---

## What We Still Need To Do

### High Priority

* [ ] Add **pretrained backbone support**

  * Swin (medical pretrained)
  * ConvNeXt-3D
* [ ] Implement **learned sigma / predict-x0 diffusion variants**
* [ ] Add **segmentation-based evaluation**
* [ ] Improve **inference speed**

  * reduced step schedules
  * DDIM / fast sampling

---

### Medium Priority

* [ ] Add **better normalization strategy**

  * brain masking
  * modality-specific scaling
* [ ] Add **dataset caching / prefetching**
* [ ] Add **training visualizations over time**
* [ ] Improve **logging (TensorBoard or similar)**

---

### Architecture / Research Extensions

* [ ] Multi-head modality prediction (all outputs simultaneously)
* [ ] Hybrid GAN + diffusion losses
* [ ] Temporal consistency (for sequences)
* [ ] Cross-attention between modalities

---

### Engineering / Infrastructure

* [ ] Docker setup
* [ ] CLI interface (argparse)
* [ ] experiment tracking (wandb or similar)
* [ ] better checkpoint/version management

---

## Notes for Contributors

* Everything is intended to be **config-driven**
* Avoid hardcoding paths or hyperparameters
* Keep modules independent and swappable
* Prefer clarity over premature optimization

---

## Current Status

The pipeline is **fully functional end-to-end**:

* training ✅
* validation ✅
* inference ✅
* visualization ✅
* reproducibility (seeded selection) ✅

Next phase is:

> **scaling model quality and efficiency**

---

## Acknowledgment

Inspired by BraTS 2025 challenge approaches and diffusion-based medical imaging literature.

---

## Contact / Collaboration

If you're working on this project:

* start with `configs/base.yaml`
* verify dataset paths
* run a short training test (50–100 steps)
* then move to full experiments

---
