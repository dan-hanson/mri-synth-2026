import os
import sys
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm

sys.path.append(r"C:\mri_synth_2026\src")
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
    train_root = r"C:\mri_synth_2026\data\GLI\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    val_root = r"C:\mri_synth_2026\data\GLI\ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"

    train_cache = r"C:\mri_synth_2026\cache\train"
    val_cache = r"C:\mri_synth_2026\cache\val"

    preprocess_split(train_root, train_cache)
    preprocess_split(val_root, val_cache)

if __name__ == "__main__":
    main()