import os
import numpy as np
import nibabel as nib

from mri_missing.utils.cases import FILE_MAP


def normalize_zscore(x):
    # 1. Create a boolean mask of the actual brain tissue (ignoring empty air)
    mask = x > x.min()
    
    # 2. Safety check: If the volume is completely empty, return it as-is
    if not mask.any():
        return x
        
    # 3. Calculate stats ONLY on the brain tissue
    mean_val = x[mask].mean()
    std_val = x[mask].std()
    
    # 4. Normalize the volume using the true tissue stats
    return (x - mean_val) / (std_val + 1e-8)


def normalize_per_case_01_np(x):
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + 1e-8)


def load_case(case_dir):
    data = {}
    affine = None
    header = None

    for mod_key, file_tag in FILE_MAP.items():
        found = None
        for f in os.listdir(case_dir):
            fl = f.lower()
            if fl.endswith(".nii.gz") and file_tag in fl and "seg" not in fl:
                found = os.path.join(case_dir, f)
                break

        if found is None:
            raise FileNotFoundError(f"Could not find {mod_key} in {case_dir}")

        img = nib.load(found)
        arr = img.get_fdata().astype(np.float32)

        data[mod_key] = arr
        if affine is None:
            affine = img.affine
            header = img.header

    return data, affine, header


def save_prediction_nifti(pred, affine, header, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img = nib.Nifti1Image(pred.astype(np.float32), affine, header)
    nib.save(img, out_path)