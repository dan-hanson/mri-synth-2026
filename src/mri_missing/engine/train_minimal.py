import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from mri_missing.diffusion.scheduler import DiffusionScheduler
from mri_missing.models.time_embedding import SinusoidalTimeEmbedding
from mri_missing.data.dataset import BraTSDataset
from mri_missing.models.unet_denoiser import TinyUNet3D
from mri_missing.models.registry import build_model


# Edit this to point at your training data directory if it lives outside the project
DATA_ROOT = os.path.join(
    PROJECT_ROOT, "data", "GLI",
    "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

scheduler = DiffusionScheduler().to(DEVICE)
time_embed = SinusoidalTimeEmbedding(128).to(DEVICE)


def add_noise(x):
    noise = torch.randn_like(x)
    alpha = torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device) * 0.5 + 0.25
    noisy = alpha * x + (1 - alpha) * noise
    return noisy, noise


def main():
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_ROOT:    {DATA_ROOT}")
    print(f"Exists?       {os.path.exists(DATA_ROOT)}")

    dataset = BraTSDataset(DATA_ROOT)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    model = build_model(
        name="convnext3d",
        in_channels=9,  # was 5
        out_channels=1
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = nn.MSELoss()

    model.train()

    # 5-tuple unpack: (cond, target, missing_key, target_idx, case_id)
    for step, (cond, target, key, target_idx, case_id) in enumerate(loader):
        cond = cond.to(DEVICE)
        target = target.to(DEVICE)
        target_idx = target_idx.to(DEVICE)

        t = scheduler.sample_timesteps(cond.shape[0], DEVICE)
        x_t, noise = scheduler.q_sample(target, t)
        t_emb = time_embed(t.float())

        x = torch.cat([cond, x_t], dim=1)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            # convnext3d takes precomputed t_emb and accepts class_labels (currently ignored by this model)
            pred_noise = model(x, t_emb, class_labels=target_idx)
            loss = loss_fn(pred_noise, noise)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 10 == 0:
            print(f"step={step} loss={loss.item():.6f} missing={key} idx={target_idx.tolist()}")

        if step == 50:
            break


if __name__ == "__main__":
    main()