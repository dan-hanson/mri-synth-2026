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
        self.compute_psnr = compute_psnr and PSNRMetric is not None
        self.compute_ssim = compute_ssim and SSIMMetric is not None
        self.eval_norm = eval_norm

        self.psnr = PSNRMetric(max_val=1.0) if self.compute_psnr else None
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

    def __call__(self, pred, target):
        pred, target = self._prep(pred, target)

        out = {}

        if self.compute_mae:
            out["mae"] = F.l1_loss(pred, target, reduction="mean").item()

        if self.compute_psnr:
            psnr_val = self.psnr(pred, target)
            out["psnr"] = psnr_val.mean().item() if torch.is_tensor(psnr_val) else float(psnr_val)

        if self.compute_ssim:
            ssim_val = self.ssim(pred, target)
            out["ssim"] = ssim_val.mean().item() if torch.is_tensor(ssim_val) else float(ssim_val)

        return out