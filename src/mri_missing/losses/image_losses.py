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
        use_texture_loss=True,
        texture_weight_max=0.05,
        texture_fade_start=0.4,
        texture_fade_end=0.6,
        snr_cap=5.0,
        snr_weight_mode="gate",   # "gate" or "amplify"
        **kwargs,
    ):
        self.use_noise_mse = use_noise_mse
        self.noise_mse_weight = noise_mse_weight

        self.use_mae = use_mae
        self.mae_weight = mae_weight

        self.use_ssim = use_ssim and SSIMLoss is not None
        self.ssim_weight = ssim_weight

        self.use_texture_loss = use_texture_loss
        self.texture_weight_max = texture_weight_max
        self.texture_fade_start = texture_fade_start
        self.texture_fade_end = texture_fade_end

        self.snr_cap = float(snr_cap)
        self.snr_weight_mode = str(snr_weight_mode).lower()
        if self.snr_weight_mode not in {"gate", "amplify"}:
            raise ValueError("snr_weight_mode must be 'gate' or 'amplify'.")

        self.ssim_loss = (
            SSIMLoss(spatial_dims=3, data_range=1.0, reduction="none")
            if self.use_ssim else None
        )

    def _to_eval_range(self, x):
        # strict-bound training manifold [-1, 1] -> [0, 1]
        return torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)

    def _make_aux_weight(self, snr, batch_size, device):
        if snr is None:
            return torch.ones(batch_size, device=device)

        aux_weight = torch.clamp(snr, max=self.snr_cap).view(-1)

        # "gate" = only emphasize low-noise steps without blowing up scale
        if self.snr_weight_mode == "gate":
            aux_weight = aux_weight / max(self.snr_cap, 1e-8)

        # "amplify" preserves the older behavior, just capped
        return aux_weight

    def _high_pass_texture_loss(self, pred, target):
        pred_d = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        true_d = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]

        pred_h = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        true_h = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]

        pred_w = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        true_w = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]

        loss_d = F.l1_loss(pred_d, true_d, reduction="none").mean(dim=(1, 2, 3, 4))
        loss_h = F.l1_loss(pred_h, true_h, reduction="none").mean(dim=(1, 2, 3, 4))
        loss_w = F.l1_loss(pred_w, true_w, reduction="none").mean(dim=(1, 2, 3, 4))

        return (loss_d + loss_h + loss_w) / 3.0

    def __call__(
        self,
        pred_noise,
        true_noise,
        pred_x0,
        target_x0,
        snr=None,
        current_step=0,
        max_steps=1,
        **kwargs,
    ):
        total = 0.0
        parts = {}

        if self.use_noise_mse:
            mse = F.mse_loss(pred_noise, true_noise)
            total = total + self.noise_mse_weight * mse
            parts["noise_mse"] = mse.item()

        pred_eval = self._to_eval_range(pred_x0)
        target_eval = self._to_eval_range(target_x0)

        aux_weight = self._make_aux_weight(
            snr=snr,
            batch_size=pred_x0.shape[0],
            device=pred_x0.device,
        )

        if self.use_mae:
            mae_raw = F.l1_loss(pred_eval, target_eval, reduction="none")
            mae_per_batch = mae_raw.mean(dim=(1, 2, 3, 4))
            mae_weighted = (mae_per_batch * aux_weight).mean()

            total = total + self.mae_weight * mae_weighted
            parts["mae_loss"] = mae_weighted.item()

        if self.use_ssim:
            ssim_raw = self.ssim_loss(pred_eval, target_eval)
            ssim_per_batch = ssim_raw.reshape(pred_eval.shape[0], -1).mean(dim=1)
            ssim_weighted = (ssim_per_batch * aux_weight).mean()

            total = total + self.ssim_weight * ssim_weighted
            parts["ssim_loss"] = ssim_weighted.item()

        if self.use_texture_loss:
            progress = current_step / max(1, max_steps)
            multiplier = (progress - self.texture_fade_start) / (
                self.texture_fade_end - self.texture_fade_start + 1e-8
            )
            multiplier = max(0.0, min(1.0, multiplier))

            if multiplier > 0:
                texture_raw = self._high_pass_texture_loss(pred_eval, target_eval)
                current_weight = self.texture_weight_max * multiplier
                texture_weighted = (texture_raw * aux_weight).mean() * current_weight

                total = total + texture_weighted
                parts["texture_loss"] = texture_weighted.item()
                parts["texture_weight"] = current_weight

        return total, parts