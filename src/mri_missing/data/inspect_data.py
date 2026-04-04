import os

ROOT = r"C:\mri_synth_2026\data\GLI\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

print("Level 1:", os.listdir(ROOT)[:3])

first = os.listdir(ROOT)[0]
print("Level 2:", os.listdir(os.path.join(ROOT, first))[:5])