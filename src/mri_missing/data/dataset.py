import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

class BraTSDataset(Dataset):
    def __init__(self, root_dir, patch_size=(96, 96, 64), fill_value=1.0, use_presence_mask=True):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.fill_value = fill_value
        self.use_presence_mask = use_presence_mask

        print("ROOT DIR:", root_dir)
        print("Exists?", os.path.exists(root_dir))
        print("Entries:", os.listdir(root_dir)[:5] if os.path.exists(root_dir) else "INVALID")

        self.cases = [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]

        print(f"Loaded {len(self.cases)} cases")

    def normalize(self, x):
        return (x - np.mean(x)) / (np.std(x) + 1e-8)
    
    def random_crop(self, cond, target, size=(96, 96, 64)):
        _, D, H, W = cond.shape
        d, h, w = size

        d0 = np.random.randint(0, D - d + 1)
        h0 = np.random.randint(0, H - h + 1)
        w0 = np.random.randint(0, W - w + 1)

        cond = cond[:, d0:d0+d, h0:h0+h, w0:w0+w]
        target = target[d0:d0+d, h0:h0+h, w0:w0+w]

        return cond, target

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
        target_key = str(np.random.choice(keys))

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

        cond = np.stack(cond, axis=0)   # [4, D, H, W]
        mask = np.stack(mask, axis=0)   # [4, D, H, W]

        if use_presence_mask:
            cond = np.concatenate([cond, mask], axis=0)  # [8, D, H, W]

        return cond, target, target_key

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_path = self.cases[idx]

        mods = self.load_case(case_path)

        for k in mods:
            mods[k] = self.normalize(mods[k])

        cond, target, target_key = self.random_missing(mods)
        cond, target = self.random_crop(cond, target, size=self.patch_size)

        cond = torch.tensor(cond, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

        return cond, target, target_key