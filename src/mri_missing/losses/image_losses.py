import torch
import torch.nn.functional as F

try:
    from monai.losses import SSIMLoss
except ImportError:
    SSIMLoss = None


class CompositeSynthesisLoss:
    def __init__(
            self,
            use_noise_mse=True,
            noise_mse_weight=1.0,
            use_mae=True,
            mae_weight=0.25,
            use_ssim=True,
            ssim_weight=0.25,
            **kwargs # Catch the old clamp args from config
        ):
            self.use_noise_mse = use_noise_mse
            self.noise_mse_weight = noise_mse_weight
            self.use_mae = use_mae
            self.mae_weight = mae_weight
            self.use_ssim = use_ssim and SSIMLoss is not None
            self.ssim_weight = ssim_weight

            # CRITICAL: reduction="none" allows batch-wise masking later
            self.ssim_loss = SSIMLoss(spatial_dims=3, data_range=1.0, reduction="none") if self.use_ssim else None

    def _clamp_aux(self, x):
        return torch.clamp(x, min=self.aux_clamp_min, max=self.aux_clamp_max)

    def _to_01(self, x):
        x = self._clamp_aux(x)
        return (x - self.aux_clamp_min) / (self.aux_clamp_max - self.aux_clamp_min + 1e-8)

    def _aux_scale(self, t, timesteps):
        if self.aux_weight_mode == "constant":
            return torch.ones_like(t, dtype=torch.float32)

        t = t.float()
        T = float(timesteps - 1)

        if self.aux_weight_mode == "linear_decay":
            # low t => high weight, high t => low weight
            return 1.0 - (t / max(T, 1.0))

        if self.aux_weight_mode == "snr_like":
            # stronger at low-noise steps, softer at high-noise steps
            frac = t / max(T, 1.0)
            return (1.0 - frac) ** 2

        raise ValueError(f"Unknown aux_weight_mode: {self.aux_weight_mode}")


    def __call__(self, pred_noise, true_noise, pred_x0, target_x0, snr=None, **kwargs):
            total = 0.0
            parts = {}

            # 1. Noise MSE is ALWAYS applied at full weight
            if self.use_noise_mse:
                mse = F.mse_loss(pred_noise, true_noise)
                total = total + self.noise_mse_weight * mse
                parts["noise_mse"] = mse.item()

            # 2. Map Z-scored data [-3, 3] to [0, 1] safely
            pred_eval = torch.clamp((pred_x0 + 3.0) / 6.0, min=0.0, max=1.0)
            target_eval = torch.clamp((target_x0 + 3.0) / 6.0, min=0.0, max=1.0)

            # 3. Create the Min-SNR Weighting
            if snr is not None:
                # Clamp SNR to 5.0 to prevent gradient explosions at t=0
                aux_weight = torch.clamp(snr, max=5.0).view(-1)
            else:
                aux_weight = torch.ones(pred_x0.shape[0], device=pred_x0.device)

            # 4. Apply Min-SNR Weighted MAE
            if self.use_mae:
                mae_raw = F.l1_loss(pred_eval, target_eval, reduction="none")
                mae_per_batch = mae_raw.mean(dim=(1, 2, 3, 4))
                # Multiply by continuous SNR weight instead of binary mask
                mae_weighted = (mae_per_batch * aux_weight).mean()
                
                total = total + self.mae_weight * mae_weighted
                parts["mae_loss"] = mae_weighted.item()

            # 5. Apply Min-SNR Weighted SSIM
            if self.use_ssim:
                ssim_raw = self.ssim_loss(pred_eval, target_eval).view(-1)
                ssim_weighted = (ssim_raw * aux_weight).mean()
                
                total = total + self.ssim_weight * ssim_weighted
                parts["ssim_loss"] = ssim_weighted.item()

            return total, parts


    # def __call__(self, pred_noise, true_noise, pred_x0, target_x0, t, timesteps):
    #     total = 0.0
    #     parts = {}

    #     if self.use_noise_mse:
    #         mse = F.mse_loss(pred_noise, true_noise)
    #         total = total + self.noise_mse_weight * mse
    #         parts["noise_mse"] = mse.item()

    #     pred_x0_clamped = self._clamp_aux(pred_x0)
    #     target_x0_clamped = self._clamp_aux(target_x0)

    #     aux_scale = self._aux_scale(t, timesteps).view(-1, 1, 1, 1, 1)

    #     if self.use_mae:
    #         mae_map = torch.abs(pred_x0_clamped - target_x0_clamped)
    #         mae = (mae_map * aux_scale).mean()
    #         total = total + self.mae_weight * mae
    #         parts["mae_loss"] = mae.item()

    #     if self.use_ssim:
    #         pred_eval = self._to_01(pred_x0)
    #         target_eval = self._to_01(target_x0)
    #         ssim = self.ssim_loss(pred_eval, target_eval)
    #         aux_scalar = aux_scale.mean()
    #         ssim = ssim * aux_scalar
    #         total = total + self.ssim_weight * ssim
    #         parts["ssim_loss"] = ssim.item()

    #     parts["aux_scale"] = aux_scale.mean().item()
    #     return total, parts