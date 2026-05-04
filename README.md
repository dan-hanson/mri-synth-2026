# MRI Missing-Modality Synthesis

A diffusion-based pipeline for synthesizing missing MRI modalities (T1, T1ce, T2, FLAIR) from the modalities that are present, trained on BraTS-GLI 2023.

The model conditions on the available modalities plus a presence mask, and a denoising network learns to reconstruct whichever modality is held out. Conditioning is built from the four image channels and four binary mask channels (8 conditioning channels total), with a noisy version of the target stacked on top to give a 9-channel input to the denoiser.

---

## Requirements

- **Python 3.11** (required — newer or older versions are not currently supported)
- A CUDA-capable GPU is strongly recommended for training. Inference can run on CPU but will be slow.
- The BraTS-GLI 2023 dataset (training and validation splits)

The repo includes a `.python-version` file pinning the interpreter to 3.11. If you use `pyenv` or `uv`, the right Python will be selected automatically when you `cd` into the project.

---

## One-time setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd mri-synth-2026

# 2. Create a Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Make the BraTS data available under data/GLI/
#    Either copy the data into data/GLI/, or symlink it:
ln -s /path/to/your/brats/data data/GLI
```

After step 4, `data/GLI/` should contain these two folders:

```
data/GLI/
├── ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
└── ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/
```

You do not need to set `PYTHONPATH` or install the package. Each entry-point script imports `_path_bootstrap` first, which finds the project root and adds `src/` to `sys.path` automatically.

---

## Running the pipeline

All paths and hyperparameters live in `configs/base.yaml`. Paths in the YAML are interpreted relative to the project root, so the same config works on any machine without edits.

### Sanity check (optional)

Verify a case loads correctly and produces the expected tensor shapes:

```bash
python dataset_test.py
```

### Step 1 — Preprocess NIfTI data into `.pt` cache files

```bash
python scripts/preprocess_to_pt.py
```

Reads `configs/base.yaml` for the data and cache locations and writes one `.pt` file per case into `cache/train/` and `cache/val/`. Already-cached cases are skipped, so re-running is safe.

### Step 2 — Verify the cache (optional)

```bash
python scripts/inspect_data.py

# or pick a different case:
CASE_ID=BraTS-GLI-00001-000 python scripts/inspect_data.py
```

Compares one raw case to its cached version and prints per-modality diffs. Useful before kicking off a long training run.

### Step 3 — Train

```bash
python src/mri_missing/engine/train.py
```

Outputs land in `outputs/<run_name>/` with `checkpoints/`, `logs/`, an augmentation preview PNG, and a copy of the config that was used.

To stop a running job cleanly (saves a checkpoint first), drop a `STOP` file in the run directory:

```bash
touch outputs/COMBO_01/STOP
```

To resume from a checkpoint, set `train.resume` in `base.yaml` to point at the checkpoint path before relaunching.

### Step 4 — Inference

```bash
python src/mri_missing/engine/infer.py
```

Uses the `inference:` block in `base.yaml` — which run, which checkpoint, which cases, which modality to drop, etc. Outputs go to `outputs/<run_name>/inference/`.

### Step 5 — Evaluation across the validation set

```bash
python src/mri_missing/engine/eval.py
```

Runs metrics over the validation cache, controlled by `inference.max_eval_cases` in the YAML (`0` means run the full set). Outputs land in `outputs/evaluations/<run_name>_<timestamp>/`.

---

## Quick reference

| Stage | Command |
|---|---|
| Sanity check | `python dataset_test.py` |
| Preprocess | `python scripts/preprocess_to_pt.py` |
| Verify cache | `python scripts/inspect_data.py` |
| Train | `python src/mri_missing/engine/train.py` |
| Infer | `python src/mri_missing/engine/infer.py` |
| Eval | `python src/mri_missing/engine/eval.py` |
| Stop training | `touch outputs/<run_name>/STOP` |

---

## Repo layout

```
mri-synth-2026/
├── _path_bootstrap.py           # adds src/ to sys.path for repo-root scripts
├── dataset_test.py              # smoke test
├── README.md
├── requirements.txt
├── .python-version              # pins Python to 3.11
│
├── configs/
│   └── base.yaml                # all paths and hyperparameters
│
├── data/                        # BraTS data lives here (gitignored)
│   └── GLI/
│
├── cache/                       # populated by preprocess_to_pt.py (gitignored)
│   ├── train/
│   └── val/
│
├── outputs/                     # run artifacts (gitignored)
│
├── scripts/
│   ├── _path_bootstrap.py       # local copy for scripts in this folder
│   ├── inspect_data.py
│   └── preprocess_to_pt.py
│
└── src/
    └── mri_missing/             # the package
        ├── config.py            # YAML loader + project-root resolution
        ├── seed.py
        ├── data/dataset.py
        ├── diffusion/scheduler.py
        ├── engine/
        │   ├── _path_bootstrap.py
        │   ├── train.py
        │   ├── train_minimal.py
        │   ├── infer.py
        │   ├── infer_v8_latest.py
        │   └── eval.py
        ├── losses/image_losses.py
        ├── metrics/image_metrics.py
        ├── models/              # UNet, ConvNeXt3D, SwinUNETR, MONAI diffusion
        └── utils/               # cases, ema, io, nifti, stats, visualization
```

The three copies of `_path_bootstrap.py` are intentional. When you run a script with `python <path>`, Python only adds the script's own directory to `sys.path`, so the bootstrap has to live next to anything that imports it. The contents of all three copies are identical.

---

## Configuration

Almost everything is controlled by `configs/base.yaml`. The most common knobs:

- `project.run_name` — name for this run's output folder. Set to `null` for an auto-timestamped name.
- `data.backend` — `pt_cache` (fast, normal) or `nifti` (slow, raw `.nii.gz` on the fly).
- `train.max_steps`, `train.resume`, `optim.lr` — duration, checkpoint resume path, learning rate.
- `inference.run_dir`, `inference.checkpoint_name`, `inference.case_ids`, `inference.missing_keys` — what to run inference on.
- `inference.cfg_guidance_scale` — per-modality classifier-free guidance scale at inference time.
- `missing_policy.sampling_probs` — probability of dropping each modality during training.

All path values in the YAML are auto-resolved against the project root by `load_config()`, so they work on any machine without edits. Absolute paths are also accepted if you keep data on a separate disk.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'mri_missing'`**

The bootstrap didn't fire. Confirm `_path_bootstrap.py` exists in the same directory as the script you're running:

- Repo-root scripts (`dataset_test.py`) → root copy
- `scripts/*.py` → `scripts/_path_bootstrap.py`
- `src/mri_missing/engine/*.py` → `src/mri_missing/engine/_path_bootstrap.py`

**`FileNotFoundError` on a data path**

Either `data/GLI/` isn't pointing at the right folder, or the BraTS subfolder names don't match what's in `base.yaml`. Run `python scripts/inspect_data.py` to confirm one case loads correctly.

**CUDA out of memory during training**

Drop `loader.batch_size` to 1, reduce `patch.size`, enable `train.use_amp: true`, or enable sliding-window inference under `inference.sliding_window` if the OOM is in the inference path.

**`ImportError: cannot import name 'X' from 'mri_missing.utils.Y'`**

A function being imported wasn't found in the module. Usually means a refactor left a stale import behind — check the actual definitions in the target file with `grep "^def " src/mri_missing/utils/<file>.py`.

---

## Why Python 3.11 specifically?

The codebase uses syntax and typing features that are 3.10+ (PEP 604 union types, structural pattern matching in some places), and several of the model dependencies (a particular pin of `diffusers`, `monai`, and `torch`) were tested against 3.11 specifically. 3.12 changes the import system in ways that may break the bootstrap pattern, and 3.10 lacks some typing features used by the model registry. If you need to run on a different Python version, expect to do some debugging.