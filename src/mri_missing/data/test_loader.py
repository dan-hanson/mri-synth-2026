from torch.utils.data import DataLoader
import sys

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from mri_missing.data.dataset import BraTSDataset

DATA_ROOT = ".../data/GLI/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

dataset = BraTSDataset(DATA_ROOT)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

for cond, target, key in loader:
    print("Condition:", cond.shape)
    print("Target:", target.shape)
    print("Missing:", key)
    break