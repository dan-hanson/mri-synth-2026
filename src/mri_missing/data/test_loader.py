from torch.utils.data import DataLoader
import sys

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from mri_missing.data.dataset import BraTSDataset

DATA_ROOT = "/home/heron/Desktop/PROJECTS/SP-Group/mri-synth-2026/data/GLI/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

dataset = BraTSDataset(DATA_ROOT)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

for cond, target, tumor_mask, key, target_idx, case_id in loader:
    print("Condition:  ", cond.shape)
    print("Target:     ", target.shape)
    print("Tumor mask: ", tumor_mask.shape, "fraction:", tumor_mask.float().mean().item())
    print("Missing key:", key)
    print("Target idx: ", target_idx)
    print("Case id:    ", case_id)
    break