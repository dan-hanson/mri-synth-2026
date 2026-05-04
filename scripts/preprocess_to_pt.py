import os
import sys
from pathlib import Path

import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Bootstrap: figure out the project root and add src/ to sys.path so we can
# import from mri_missing without any hardcoded paths.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()


def _find_project_root(start: Path) -> Path:
    """Walk upward looking for a marker that identifies the repo root."""
    markers = (".git", "pyproject.toml", "setup.py", "configs")
    for candidate in (start, *start.parents):
        for marker in markers:
            if (candidate / marker).exists():
                return candidate
    return Path.cwd()


PROJECT_ROOT = _find_project_root(_HERE)
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Now safe to import project modules
from mri_missing.utils.nifti import normalize_zscore, normalize_strict_bound  # noqa: E402
from mri_missing.config import load_config  # noqa: E402  (uses the same resolver)


FILE_MAP = {
    "t1": "t1n",
    "t1ce": "t1c",
    "t2": "t2w",
    "flair": "t2f",
}


def load_case(case_dir: str):
    mods = {}
    for mod_key, file_tag in FILE_MAP.items():
        found = None
        for f in os.listdir(case_dir):
            fl = f.lower()
            if fl.endswith(".nii.gz") and file_tag in fl and "seg" not in fl:
                found = os.path.join(case_dir, f)
                break
        if found is None:
            raise FileNotFoundError(f"Missing {mod_key} in {case_dir}")
        arr = nib.load(found).get_fdata().astype(np.float32)
        mods[mod_key] = normalize_strict_bound(arr)
    return mods


def preprocess_split(input_root: str, output_root: str):
    os.makedirs(output_root, exist_ok=True)
    case_names = sorted(
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    )

    for case_name in tqdm(case_names, desc=f"Processing {os.path.basename(output_root)}"):
        case_dir = os.path.join(input_root, case_name)
        out_path = os.path.join(output_root, f"{case_name}.pt")

        if os.path.exists(out_path):
            continue

        mods = load_case(case_dir)
        torch.save(
            {
                "case_id": case_name,
                "t1": torch.from_numpy(mods["t1"]),
                "t1ce": torch.from_numpy(mods["t1ce"]),
                "t2": torch.from_numpy(mods["t2"]),
                "flair": torch.from_numpy(mods["flair"]),
            },
            out_path,
        )


def main():
    # Pull the same paths the rest of the pipeline uses. load_config()
    # auto-resolves any relative paths in base.yaml against the project root,
    # so this stays in sync no matter who's running it.
    cfg = load_config("configs/base.yaml")

    train_root = cfg["data"]["train_root"]
    val_root = cfg["data"]["val_root"]
    train_cache = cfg["data"]["train_cache_root"]
    val_cache = cfg["data"]["val_cache_root"]

    print(f"[preprocess] project root: {PROJECT_ROOT}")
    print(f"[preprocess] train_root  : {train_root}")
    print(f"[preprocess] val_root    : {val_root}")
    print(f"[preprocess] train_cache : {train_cache}")
    print(f"[preprocess] val_cache   : {val_cache}")

    preprocess_split(train_root, train_cache)
    preprocess_split(val_root, val_cache)


if __name__ == "__main__":
    main()