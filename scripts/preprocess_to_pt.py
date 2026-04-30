import os
import sys
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from mri_missing.utils.nifti import normalize_zscore

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
        mods[mod_key] = normalize_zscore(arr)

    # Look for the seg file. BraTS naming is "{case}-seg.nii.gz".
    # Validation cases may not have seg — return None in that case.
    seg = None
    for f in os.listdir(case_dir):
        fl = f.lower()
        if fl.endswith(".nii.gz") and "seg" in fl:
            seg_path = os.path.join(case_dir, f)
            seg = nib.load(seg_path).get_fdata().astype(np.uint8)
            break

    return mods, seg


def preprocess_split(input_root: str, output_root: str):
    os.makedirs(output_root, exist_ok=True)
    case_names = sorted(
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    )

    cases_with_seg = 0
    cases_without_seg = 0

    for case_name in tqdm(case_names, desc=f"Processing {os.path.basename(output_root)}"):
        case_dir = os.path.join(input_root, case_name)
        out_path = os.path.join(output_root, f"{case_name}.pt")

        if os.path.exists(out_path):
            # Already cached. Skip — but track whether existing cache has seg.
            blob = torch.load(out_path, map_location="cpu", weights_only=False)
            if "seg" in blob and blob["seg"] is not None:
                cases_with_seg += 1
            else:
                cases_without_seg += 1
            continue

        mods, seg = load_case(case_dir)

        blob = {
            "case_id": case_name,
            "t1": torch.from_numpy(mods["t1"]),
            "t1ce": torch.from_numpy(mods["t1ce"]),
            "t2": torch.from_numpy(mods["t2"]),
            "flair": torch.from_numpy(mods["flair"]),
        }

        if seg is not None:
            blob["seg"] = torch.from_numpy(seg)
            cases_with_seg += 1
        else:
            blob["seg"] = None
            cases_without_seg += 1

        torch.save(blob, out_path)

    print(f"  cases with seg: {cases_with_seg}")
    print(f"  cases without seg: {cases_without_seg}")


def main():
    train_root = os.path.join(
        PROJECT_ROOT, "data", "GLI",
        "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    )
    val_root = os.path.join(
        PROJECT_ROOT, "data", "GLI",
        "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
    )

    train_cache = os.path.join(PROJECT_ROOT, "cache", "train")
    val_cache = os.path.join(PROJECT_ROOT, "cache", "val")

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("TRAIN ROOT exists:", os.path.exists(train_root))
    print("VAL ROOT exists:", os.path.exists(val_root))

    preprocess_split(train_root, train_cache)
    preprocess_split(val_root, val_cache)


if __name__ == "__main__":
    main()