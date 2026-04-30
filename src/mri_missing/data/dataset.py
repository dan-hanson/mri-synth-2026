import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from mri_missing.utils.nifti import normalize_zscore
from mri_missing.utils.cases import MOD_KEYS, MOD_TO_IDX


class BraTSDataset(Dataset):
    def __init__(
        self,
        root_dir,
        patch_size=(96, 96, 64),
        fill_value=1.0,
        use_presence_mask=True,
        sampling_probs=None,
        backend="nifti",
        augmentation=None,
        tumor_crop_prob=0.0,
    ):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.fill_value = fill_value
        self.use_presence_mask = use_presence_mask
        self.backend = backend
        self.sampling_probs = sampling_probs
        self.augmentation = augmentation or {}
        self.tumor_crop_prob = tumor_crop_prob

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

    def crop_to_nonzero(self, mods, seg=None):
        mask = (mods["t1"] != 0) | (mods["t1ce"] != 0) | (mods["t2"] != 0) | (mods["flair"] != 0)

        if not mask.any():
            return mods, seg

        indices = np.nonzero(mask)
        d_min, d_max = indices[0].min(), indices[0].max() + 1
        h_min, h_max = indices[1].min(), indices[1].max() + 1
        w_min, w_max = indices[2].min(), indices[2].max() + 1

        for k in mods:
            mods[k] = mods[k][d_min:d_max, h_min:h_max, w_min:w_max]

        if seg is not None:
            seg = seg[d_min:d_max, h_min:h_max, w_min:w_max]

        return mods, seg

    def random_crop(self, cond, target, seg, size=(96, 96, 64)):
        """
        Random spatial crop. With probability `self.tumor_crop_prob`, force the
        crop window to contain at least one tumor voxel (when the case has any).
        Otherwise, sample the crop origin uniformly.
        """
        _, D, H, W = cond.shape
        d, h, w = size

        # Sanity guard: every crop axis must be <= the volume axis.
        if D < d or H < h or W < w:
            raise ValueError(
                f"Volume too small for patch: vol={(D, H, W)}, patch={(d, h, w)}"
            )

        biased = (
            self.tumor_crop_prob > 0.0
            and np.random.rand() < self.tumor_crop_prob
            and seg is not None
            and (seg > 0).any()
        )

        if biased:
            tumor_coords = np.argwhere(seg > 0)
            # Pick a random tumor voxel to be the anchor
            td, th, tw = tumor_coords[np.random.randint(len(tumor_coords))]

            # For the patch [d0, d0+d) to contain td: d0 in [td - d + 1, td]
            # Combined with the volume bounds [0, D - d]: take the intersection.
            d0_lo = max(0, td - d + 1)
            d0_hi = min(D - d, td)
            h0_lo = max(0, th - h + 1)
            h0_hi = min(H - h, th)
            w0_lo = max(0, tw - w + 1)
            w0_hi = min(W - w, tw)

            # randint's high is exclusive; +1 so the inclusive range works
            d0 = np.random.randint(d0_lo, d0_hi + 1)
            h0 = np.random.randint(h0_lo, h0_hi + 1)
            w0 = np.random.randint(w0_lo, w0_hi + 1)
        else:
            d0 = np.random.randint(0, D - d + 1)
            h0 = np.random.randint(0, H - h + 1)
            w0 = np.random.randint(0, W - w + 1)

        cond = cond[:, d0:d0 + d, h0:h0 + h, w0:w0 + w]
        target = target[d0:d0 + d, h0:h0 + h, w0:w0 + w]
        seg = seg[d0:d0 + d, h0:h0 + h, w0:w0 + w]

        return cond, target, seg

    def apply_augmentations(self, cond, target, seg):
        if not self.augmentation.get("enabled", False):
            return cond, target, seg

        flip_prob = self.augmentation.get("flip_prob", 0.0)
        if np.random.rand() < flip_prob:
            cond = np.flip(cond, axis=1).copy()
            target = np.flip(target, axis=0).copy()
            seg = np.flip(seg, axis=0).copy()

        if np.random.rand() < flip_prob:
            cond = np.flip(cond, axis=2).copy()
            target = np.flip(target, axis=1).copy()
            seg = np.flip(seg, axis=1).copy()

        if np.random.rand() < self.augmentation.get("intensity_shift_prob", 0.0):
            shift = np.random.uniform(
                -self.augmentation.get("intensity_shift_max", 0.0),
                self.augmentation.get("intensity_shift_max", 0.0),
            )
            num_image_ch = 4
            cond[:num_image_ch] = cond[:num_image_ch] + shift
            target = target + shift
            # seg untouched — discrete labels

        return cond, target, seg

    def load_case_pt(self, case_path):
        blob = torch.load(case_path, map_location="cpu", weights_only=False)
        mods = {
            "t1": blob["t1"].numpy(),
            "t1ce": blob["t1ce"].numpy(),
            "t2": blob["t2"].numpy(),
            "flair": blob["flair"].numpy(),
        }
        # Seg may be absent or None for some validation cases
        seg_tensor = blob.get("seg", None)
        if seg_tensor is None:
            seg = np.zeros_like(mods["t1"], dtype=np.uint8)
        else:
            seg = seg_tensor.numpy().astype(np.uint8)
        return mods, seg

    def load_case(self, case_path):
        modalities = {"t1": None, "t1ce": None, "t2": None, "flair": None}
        seg = None

        for file in os.listdir(case_path):
            f = file.lower()
            if not f.endswith(".nii.gz"):
                continue
            full_path = os.path.join(case_path, file)

            if "seg" in f:
                seg = nib.load(full_path).get_fdata().astype(np.uint8)
                continue

            if "t1n" in f:
                modalities["t1"] = nib.load(full_path).get_fdata()
            elif "t1c" in f:
                modalities["t1ce"] = nib.load(full_path).get_fdata()
            elif "t2w" in f:
                modalities["t2"] = nib.load(full_path).get_fdata()
            elif "t2f" in f:
                modalities["flair"] = nib.load(full_path).get_fdata()

        for k, v in modalities.items():
            if v is None:
                raise ValueError(f"Missing {k} in {case_path}")

        if seg is None:
            seg = np.zeros_like(modalities["t1"], dtype=np.uint8)

        return modalities, seg

    def random_missing(self, mods):
        keys = MOD_KEYS

        probs_cfg = getattr(self, "sampling_probs", None)
        if probs_cfg is None:
            probs = None
        else:
            probs = np.array([probs_cfg[k] for k in keys], dtype=np.float32)
            probs = probs / probs.sum()

        target_key = str(np.random.choice(keys, p=probs))
        target_idx = MOD_TO_IDX[target_key]

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

        cond = np.stack(cond, axis=0)
        mask = np.stack(mask, axis=0)

        if use_presence_mask:
            cond = np.concatenate([cond, mask], axis=0)

        return cond, target, target_key, target_idx

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_path = self.cases[idx]

        if self.backend == "pt_cache":
            mods, seg = self.load_case_pt(case_path)
        else:
            mods, seg = self.load_case(case_path)
            mods, seg = self.crop_to_nonzero(mods, seg)

        if self.backend != "pt_cache":
            for k in mods:
                mods[k] = normalize_zscore(mods[k])

        cond, target, target_key, target_idx = self.random_missing(mods)
        cond, target, seg = self.random_crop(cond, target, seg, size=self.patch_size)
        cond, target, seg = self.apply_augmentations(cond, target, seg)

        cond = torch.tensor(cond, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        # tumor mask: 1.0 inside tumor (any non-zero seg label), 0.0 elsewhere
        tumor_mask = torch.tensor((seg > 0).astype(np.float32), dtype=torch.float32).unsqueeze(0)
        target_idx = torch.tensor(target_idx, dtype=torch.long)
        case_id = os.path.basename(case_path).replace(".pt", "")

        # 6-tuple return: cond, target, tumor_mask, target_key, target_idx, case_id
        return cond, target, tumor_mask, target_key, target_idx, case_id