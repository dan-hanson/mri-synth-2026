import numpy as np
import torch

import _path_bootstrap  # noqa: F401  (must come before mri_missing.*)

from pathlib import Path

from mri_missing.config import load_config
from mri_missing.utils.nifti import load_case, normalize_strict_bound


# Pick a case to compare. Override by setting CASE_ID env var if you want to
# point at a different one without editing the file.
import os
CASE_ID = os.environ.get("CASE_ID", "BraTS-GLI-00723-000")


def main():
    cfg = load_config("configs/base.yaml")

    # load_config returns absolute paths (resolved against the project root),
    # so these work no matter where the script is invoked from.
    raw_root = Path(cfg["data"]["train_root"])
    cache_root = Path(cfg["data"]["train_cache_root"])

    raw_case_dir = raw_root / CASE_ID
    cache_path = cache_root / f"{CASE_ID}.pt"

    print(f"Comparing case: {CASE_ID}")
    print(f"  raw  : {raw_case_dir}")
    print(f"  cache: {cache_path}")

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


if __name__ == "__main__":
    main()