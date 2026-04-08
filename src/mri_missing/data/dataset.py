import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from mri_missing.utils.nifti import normalize_zscore

class BraTSDataset(Dataset):
    def __init__(self, root_dir, patch_size=(96, 96, 64), fill_value=1.0, use_presence_mask=True, sampling_probs=None, backend="nifti", augmentation=None):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.fill_value = fill_value
        self.use_presence_mask = use_presence_mask
        self.backend = backend
        self.sampling_probs = sampling_probs
        self.augmentation = augmentation or {}

        print("ROOT DIR:", root_dir)
        print("Exists?", os.path.exists(root_dir))
        print("Entries:", os.listdir(root_dir)[:5] if os.path.exists(root_dir) else "INVALID")

        if self.backend == "pt_cache":
            self.cases = [
                os.path.join(root_dir, f)
                for f in os.listdir(root_dir)
                if f.endswith(".pt")
            ]
        else:
            self.cases = [
                os.path.join(root_dir, d)
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]

        print(f"Loaded {len(self.cases)} cases")
    
    def crop_to_nonzero(self, mods):
        # Create a boolean mask of where any modality has tissue
        mask = (mods["t1"] != 0) | (mods["t1ce"] != 0) | (mods["t2"] != 0) | (mods["flair"] != 0)
        
        # If the case is completely empty (shouldn't happen, but safe)
        if not mask.any():
            return mods
            
        indices = np.nonzero(mask)
        
        # Find the tight bounding box
        d_min, d_max = indices[0].min(), indices[0].max() + 1
        h_min, h_max = indices[1].min(), indices[1].max() + 1
        w_min, w_max = indices[2].min(), indices[2].max() + 1
        
        # Crop all modalities
        for k in mods:
            mods[k] = mods[k][d_min:d_max, h_min:h_max, w_min:w_max]
            
        return mods
    
    def random_crop(self, cond, target, size=(96, 96, 64)):
        _, D, H, W = cond.shape
        d, h, w = size

        d0 = np.random.randint(0, D - d + 1)
        h0 = np.random.randint(0, H - h + 1)
        w0 = np.random.randint(0, W - w + 1)

        cond = cond[:, d0:d0+d, h0:h0+h, w0:w0+w]
        target = target[d0:d0+d, h0:h0+h, w0:w0+w]

        return cond, target
    
    def apply_augmentations(self, cond, target):
        if not self.augmentation.get("enabled", False):
            return cond, target

        # Safe flips only: sagittal / coronal style axes within patch space
        flip_prob = self.augmentation.get("flip_prob", 0.0)
        if np.random.rand() < flip_prob:
            # flip depth axis
            cond = np.flip(cond, axis=1).copy()
            target = np.flip(target, axis=0).copy()

        if np.random.rand() < flip_prob:
            # flip height axis
            cond = np.flip(cond, axis=2).copy()
            target = np.flip(target, axis=1).copy()

        # Mild intensity shift
        if np.random.rand() < self.augmentation.get("intensity_shift_prob", 0.0):
            shift = np.random.uniform(
                -self.augmentation.get("intensity_shift_max", 0.0),
                self.augmentation.get("intensity_shift_max", 0.0),
            )
            # only shift the first 4 channels (modalities), not the presence mask
            num_image_ch = 4
            cond[:num_image_ch] = cond[:num_image_ch] + shift
            target = target + shift

        return cond, target
    
    def load_case_pt(self, case_path):
        blob = torch.load(case_path, map_location="cpu")
        return {
            "t1": blob["t1"].numpy(),
            "t1ce": blob["t1ce"].numpy(),
            "t2": blob["t2"].numpy(),
            "flair": blob["flair"].numpy(),
        }

    def load_case(self, case_path):
        modalities = {
            "t1": None,
            "t1ce": None,
            "t2": None,
            "flair": None
        }

        for file in os.listdir(case_path):
            if "seg" in file.lower():
                continue

            if not file.endswith(".nii.gz"):
                continue

            f = file.lower()
            full_path = os.path.join(case_path, file)

            if "t1n" in f:
                modalities["t1"] = nib.load(full_path).get_fdata()
            elif "t1c" in f:
                modalities["t1ce"] = nib.load(full_path).get_fdata()
            elif "t2w" in f:
                modalities["t2"] = nib.load(full_path).get_fdata()
            elif "t2f" in f:
                modalities["flair"] = nib.load(full_path).get_fdata()

        # print("Loading case:", case_path)
        # print("Files:", os.listdir(case_path))
        for k, v in modalities.items():
            if v is None:
                raise ValueError(f"Missing {k} in {case_path}")

        return modalities

    def random_missing(self, mods):
        keys = ["t1", "t1ce", "t2", "flair"]

        probs_cfg = getattr(self, "sampling_probs", None)
        if probs_cfg is None:
            probs = None
        else:
            probs = np.array([probs_cfg[k] for k in keys], dtype=np.float32)
            probs = probs / probs.sum()

        target_key = str(np.random.choice(keys, p=probs))

        target = mods[target_key]

        cond = []
        mask = []

        fill_value = getattr(self, "fill_value", 1.0)
        use_presence_mask = getattr(self, "use_presence_mask", True)

        for k in keys:
            if k == target_key:
                cond.append(np.full_like(mods[k], fill_value, dtype=np.float32))
                mask.append(np.zeros_like(mods[k], dtype=np.float32))
            else:
                cond.append(mods[k].astype(np.float32))
                mask.append(np.ones_like(mods[k], dtype=np.float32))

        cond = np.stack(cond, axis=0) # [4, D, H, W]
        mask = np.stack(mask, axis=0) # [4, D, H, W]

        if use_presence_mask:
            cond = np.concatenate([cond, mask], axis=0) # [8, D, H, W]

        return cond, target, target_key

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_path = self.cases[idx]

        if self.backend == "pt_cache":
            mods = self.load_case_pt(case_path)
        else:
            mods = self.load_case(case_path)
            
            # 1. Strip the wasted background air first
            mods = self.crop_to_nonzero(mods)

        if self.backend != "pt_cache":
            for k in mods:
                mods[k] = normalize_zscore(mods[k])

        cond, target, target_key = self.random_missing(mods)
        cond, target = self.random_crop(cond, target, size=self.patch_size)
        cond, target = self.apply_augmentations(cond, target)

        cond = torch.tensor(cond, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

        return cond, target, target_key