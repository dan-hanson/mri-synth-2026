import math
import torch


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-8, 0.999).float()


class DiffusionScheduler:
    def __init__(
        self,
        timesteps=1000,
        schedule="cosine",
        beta_start=1e-4,
        beta_end=0.02,
        cosine_s=0.008,
    ):
        self.timesteps = timesteps
        self.schedule = schedule

        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule == "cosine":
            self.betas = cosine_beta_schedule(timesteps, s=cosine_s)
        elif schedule == "scaled_linear_beta":
            # The High-Res DDPM Standard
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
        else:
            raise ValueError(f"Unknown diffusion schedule: {schedule}")

        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        return self

    def sample_timesteps(self, batch_size, device):
        return torch.randint(0, self.timesteps, (batch_size,), device=device)

    def q_sample(self, x0, t):
        noise = torch.randn_like(x0)
        alpha_bar = self.alpha_cumprod[t].view(-1, 1, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise