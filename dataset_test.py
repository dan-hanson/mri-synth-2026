import os
import nibabel as nib
import numpy as np
import torch

# ---- CHANGE THIS ----
DATA_ROOT = r"C:\mri_synth_2026\data\GLI\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

def load_case(case_path):
    modalities = {
        "t1": None,
        "t1ce": None,
        "t2": None,
        "flair": None
    }

    for file in os.listdir(case_path):
        if not file.endswith(".nii.gz"):
            continue

        f = file.lower()

        if "t1n" in f:
            modalities["t1"] = nib.load(os.path.join(case_path, file)).get_fdata()

        elif "t1c" in f:
            modalities["t1ce"] = nib.load(os.path.join(case_path, file)).get_fdata()

        elif "t2w" in f:
            modalities["t2"] = nib.load(os.path.join(case_path, file)).get_fdata()

        elif "t2f" in f:
            modalities["flair"] = nib.load(os.path.join(case_path, file)).get_fdata()

    # 🔥 CRITICAL DEBUG CHECK
    for k, v in modalities.items():
        if v is None:
            raise ValueError(f"Missing modality: {k} in {case_path}")

    return modalities

def normalize(x):
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    return x


def random_missing(mods):
    keys = ["t1", "t1ce", "t2", "flair"]
    target_key = np.random.choice(keys)

    target = mods[target_key].copy()

    cond = []
    for k in keys:
        if k == target_key:
            cond.append(np.zeros_like(mods[k]))
        else:
            cond.append(mods[k])

    cond = np.stack(cond, axis=0)

    return cond, target, target_key


def main():
    cases = os.listdir(DATA_ROOT)
    case_path = os.path.join(DATA_ROOT, cases[0])

    mods = load_case(case_path)

    # Normalize
    for k in mods:
        mods[k] = normalize(mods[k])

    cond, target, target_key = random_missing(mods)

    print("Condition shape:", cond.shape)
    print("Target shape:", target.shape)
    print("Missing modality:", target_key)

    # Convert to torch
    cond = torch.tensor(cond, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)

    print("Tensor shape:", cond.shape)


if __name__ == "__main__":
    main()