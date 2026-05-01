import os
import numpy as np
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from mri_missing.utils.nifti import load_case, normalize_strict_bound

CASE_ID = "BraTS-GLI-00723-000"
RAW_ROOT = PROJECT_ROOT / "data" / "GLI" / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
CACHE_ROOT = PROJECT_ROOT / "cache" / "train"

raw_case_dir = RAW_ROOT / CASE_ID
cache_path = CACHE_ROOT / f"{CASE_ID}.pt"

raw_mods, _, _ = load_case(str(raw_case_dir))
raw_norm = {k: normalize_strict_bound(v) for k, v in raw_mods.items()}

blob = torch.load(str(cache_path), map_location="cpu")
cache_mods = {
    "t1": blob["t1"].numpy(),
    "t1ce": blob["t1ce"].numpy(),
    "t2": blob["t2"].numpy(),
    "flair": blob["flair"].numpy(),
}

for k in ["t1", "t1ce", "t2", "flair"]:
    a = raw_norm[k]
    b = cache_mods[k]

    print(f"\n[{k}]")
    print("raw   shape/min/max/mean/std:", a.shape, a.min(), a.max(), a.mean(), a.std())
    print("cache shape/min/max/mean/std:", b.shape, b.min(), b.max(), b.mean(), b.std())

    if a.shape == b.shape:
        diff = np.abs(a - b)
        print("mean_abs_diff:", diff.mean())
        print("max_abs_diff :", diff.max())
    else:
        print("SHAPE MISMATCH")