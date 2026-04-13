import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

ROOT = "/home/heron/Desktop/PROJECTS/SP-Group/mri-synth-2026-main/mri-synth-2026/data/GLI/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

print("Level 1:", os.listdir(ROOT)[:3])

first = os.listdir(ROOT)[0]
print("Level 2:", os.listdir(os.path.join(ROOT, first))[:5])