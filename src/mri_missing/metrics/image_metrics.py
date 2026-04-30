import torch
import torch.nn.functional as F

try:
    from monai.metrics import PSNRMetric, SSIMMetric
except ImportError:
    PSNRMetric = None
    SSIMMetric = None


# Minimum dimension required for MONAI's default 11x11x11 SSIM window
SSIM_MIN_DIM = 11


def normalize_per_case_01(x):
    # x: [B, C, D, H, W]
    x_min = x.amin(dim=(2, 3, 4), keepdim=True)
    x_max = x.amax(dim=(2, 3, 4), keepdim=True)
    return (x - x_min) / (x_max - x_min + 1e-8)


class ImageMetricBundle:
    """
    Computes MAE, PSNR, and SSIM with optional region-aware breakouts.

    Calling __call__ with `tumor_mask` returns:
      - {mae, psnr, ssim}                                  (global, brain-masked)
      - {tumor_mae, tumor_psnr, tumor_ssim}                (tumor region only)
      - {healthy_mae, healthy_psnr, healthy_ssim}          (brain minus tumor)

    SSIM is computed on a tight bounding-box crop for each region. If the
    region is smaller than the SSIM window in any axis, SSIM is omitted
    (the key won't appear in the output dict).
    """

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

    def _mae(self, pred, target, bool_mask):
        return F.l1_loss(pred[bool_mask], target[bool_mask], reduction="mean").item()

    def _psnr(self, pred, target, bool_mask):
        mse = F.mse_loss(pred[bool_mask], target[bool_mask], reduction="mean")
        if mse == 0:
            return 100.0
        return (10 * torch.log10(1.0 / mse)).item()

    def _ssim_cropped(self, pred, target, bool_mask):
        """Tight-crop SSIM. Returns None if the region is too small for the SSIM window."""
        if not bool_mask.any():
            return None

        coords = torch.where(bool_mask)
        d0, d1 = coords[2].min().item(), coords[2].max().item() + 1
        h0, h1 = coords[3].min().item(), coords[3].max().item() + 1
        w0, w1 = coords[4].min().item(), coords[4].max().item() + 1

        if (d1 - d0) < SSIM_MIN_DIM or (h1 - h0) < SSIM_MIN_DIM or (w1 - w0) < SSIM_MIN_DIM:
            return None

        pred_crop = pred[:, :, d0:d1, h0:h1, w0:w1]
        targ_crop = target[:, :, d0:d1, h0:h1, w0:w1]

        ssim_val = self.ssim(pred_crop, targ_crop)
        return ssim_val.mean().item() if torch.is_tensor(ssim_val) else float(ssim_val)

    def _region_metrics(self, pred, target, bool_mask, prefix=""):
        """Compute MAE/PSNR/SSIM over the boolean mask, with optional key prefix."""
        out = {}
        if not bool_mask.any():
            return out

        if self.compute_mae:
            out[f"{prefix}mae"] = self._mae(pred, target, bool_mask)
        if self.compute_psnr:
            out[f"{prefix}psnr"] = self._psnr(pred, target, bool_mask)
        if self.compute_ssim:
            ssim_val = self._ssim_cropped(pred, target, bool_mask)
            if ssim_val is not None:
                out[f"{prefix}ssim"] = ssim_val
        return out

    def __call__(self, pred, target, mask=None, tumor_mask=None):
        """
        pred, target: [B, C, D, H, W]
        mask:        [B, C, D, H, W] - brain mask (where to compute global metrics)
        tumor_mask:  [B, C, D, H, W] - tumor mask (subset of mask, for region split)

        Returns a dict with global metrics and, when tumor_mask is supplied,
        tumor_*/healthy_* breakouts.
        """
        pred, target = self._prep(pred, target)

        if mask is None:
            # No brain mask: compute everything on the full volume
            full_mask = torch.ones_like(target, dtype=torch.bool)
        else:
            full_mask = mask > 0

        out = self._region_metrics(pred, target, full_mask, prefix="")

        if tumor_mask is not None:
            tumor_bool = tumor_mask > 0
            healthy_bool = full_mask & (~tumor_bool)

            out.update(self._region_metrics(pred, target, tumor_bool, prefix="tumor_"))
            out.update(self._region_metrics(pred, target, healthy_bool, prefix="healthy_"))

        return out