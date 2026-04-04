from torch.utils.data import DataLoader
import sys
sys.path.append(r"C:\mri_synth_2026\src")
from mri_missing.data.dataset import BraTSDataset

DATA_ROOT = r"C:\mri_synth_2026\data\GLI\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

dataset = BraTSDataset(DATA_ROOT)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

for cond, target, key in loader:
    print("Condition:", cond.shape)
    print("Target:", target.shape)
    print("Missing:", key)
    break