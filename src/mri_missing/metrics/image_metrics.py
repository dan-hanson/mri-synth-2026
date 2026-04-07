import torch
import torch.nn.functional as F

try:
    from monai.metrics import PSNRMetric, SSIMMetric
except ImportError:
    PSNRMetric = None
    SSIMMetric = None

def normalize_per_case_01(x):
    # x: [B, C, D, H, W]
    x_min = x.amin(dim=(2, 3, 4), keepdim=True)
    x_max = x.amax(dim=(2, 3, 4), keepdim=True)
    return (x - x_min) / (x_max - x_min + 1e-8)

class ImageMetricBundle:
    def __init__(self, compute_mae=True, compute_psnr=True, compute_ssim=True, eval_norm="per_case_minmax"):
        self.compute_mae = compute_mae
        self.compute_psnr = compute_psnr
        self.compute_ssim = compute_ssim and SSIMMetric is not None
        self.eval_norm = eval_norm

        self.ssim = SSIMMetric(spatial_dims=3, data_range=1.0) if self.compute_ssim else None

    def _prep(self, pred, target):
        if self.eval_norm == "per_case_minmax":
            pred = normalize_per_case_01(pred)
            target = normalize_per_case_01(target)
        elif self.eval_norm == "none":
            pass
        else:
            raise ValueError(f"Unknown eval_norm: {self.eval_norm}")
        return pred, target

    def __call__(self, pred, target, mask=None):
        pred, target = self._prep(pred, target)
        out = {}

        if mask is not None:
            # Ensure mask is boolean
            bool_mask = mask > 0

            # 1. Masked MAE (Only calculate where mask is True)
            if self.compute_mae:
                out["mae"] = F.l1_loss(pred[bool_mask], target[bool_mask], reduction="mean").item()

            # 2. Masked PSNR
            if self.compute_psnr:
                mse = F.mse_loss(pred[bool_mask], target[bool_mask], reduction="mean")
                if mse == 0:
                    out["psnr"] = 100.0
                else:
                    out["psnr"] = 10 * torch.log10(1.0 / mse).item()

            # 3. Tight-Cropped SSIM
            if self.compute_ssim:
                # Find the tightest bounding box to eliminate empty background from the SSIM window
                coords = torch.where(bool_mask)
                if len(coords[0]) > 0:
                    d0, d1 = coords[2].min(), coords[2].max() + 1
                    h0, h1 = coords[3].min(), coords[3].max() + 1
                    w0, w1 = coords[4].min(), coords[4].max() + 1
                    
                    pred_crop = pred[:, :, d0:d1, h0:h1, w0:w1]
                    targ_crop = target[:, :, d0:d1, h0:h1, w0:w1]
                    
                    ssim_val = self.ssim(pred_crop, targ_crop)
                    out["ssim"] = ssim_val.mean().item() if torch.is_tensor(ssim_val) else float(ssim_val)
                else:
                    out["ssim"] = 0.0

        else:
            # Fallback to standard full-volume metrics if no mask is provided
            if self.compute_mae:
                out["mae"] = F.l1_loss(pred, target, reduction="mean").item()
            if self.compute_psnr:
                mse = F.mse_loss(pred, target, reduction="mean")
                out["psnr"] = 10 * torch.log10(1.0 / mse).item() if mse > 0 else 100.0
            if self.compute_ssim:
                ssim_val = self.ssim(pred, target)
                out["ssim"] = ssim_val.mean().item() if torch.is_tensor(ssim_val) else float(ssim_val)

        return out