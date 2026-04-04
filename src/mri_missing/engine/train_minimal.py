import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
torch.backends.cudnn.benchmark = True

sys.path.append(r"C:\mri_synth_2026\src")

from mri_missing.diffusion.scheduler import DiffusionScheduler
from mri_missing.models.time_embedding import SinusoidalTimeEmbedding
from mri_missing.data.dataset import BraTSDataset
from mri_missing.models.unet_denoiser import TinyUNet3D
from mri_missing.models.registry import build_model


DATA_ROOT = r"C:\mri_synth_2026\data\GLI\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

scheduler = DiffusionScheduler().to(DEVICE)
time_embed = SinusoidalTimeEmbedding(128).to(DEVICE)


def add_noise(x):
    noise = torch.randn_like(x)
    alpha = torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device) * 0.5 + 0.25
    noisy = alpha * x + (1 - alpha) * noise
    return noisy, noise


def main():
    dataset = BraTSDataset(DATA_ROOT)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    model = build_model(
        name="convnext3d", # Set model here (e.g. "unet", "convnext3d", "swin")
        in_channels=5,
        out_channels=1
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = nn.MSELoss()

    model.train()

    for step, (cond, target, key) in enumerate(loader):
        cond = cond.to(DEVICE)      # [B, 4, D, H, W]
        target = target.to(DEVICE)  # [B, 1, D, H, W]

        t = scheduler.sample_timesteps(cond.shape[0], DEVICE)
        x_t, noise = scheduler.q_sample(target, t)
        t_emb = time_embed(t.float())

        x = torch.cat([cond, x_t], dim=1)  # [B, 5, D, H, W]
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            pred_noise = model(x, t_emb)
            loss = loss_fn(pred_noise, noise)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 10 == 0:
            print(f"step={step} loss={loss.item():.6f} missing={key}")

        if step == 50:
            break


if __name__ == "__main__":
    main()