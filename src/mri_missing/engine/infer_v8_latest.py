import os
import sys
import time
import json
import torch
import numpy as np
import torch.nn.functional as F

import _path_bootstrap  # noqa: F401  (must come before mri_missing.*)

from mri_missing.config import load_config, ensure_dir
from mri_missing.models.registry import build_model
from mri_missing.diffusion.scheduler import DiffusionScheduler
from mri_missing.models.time_embedding import SinusoidalTimeEmbedding
from mri_missing.utils.io import load_checkpoint, save_json
from mri_missing.utils.cases import MOD_KEYS, resolve_case_dirs
from mri_missing.utils.nifti import load_case, save_prediction_nifti, normalize_zscore, normalize_per_case_01_np, \
    normalize_strict_bound
from mri_missing.utils.visualization import save_slice_panel
from monai.inferers import sliding_window_inference
from mri_missing.utils.ema import EMA
from mri_missing.metrics.image_metrics import ImageMetricBundle
from diffusers import DDIMScheduler

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def pad_to_multiple_3d(x, multiple=32, value=-1.0, mode="constant"):
    """
    x: [B, C, D, H, W]
    Pads only on the max side of each spatial axis.
    Returns padded tensor and pad info tuple.
    """
    _, _, d, h, w = x.shape

    d_pad = (multiple - (d % multiple)) % multiple
    h_pad = (multiple - (h % multiple)) % multiple
    w_pad = (multiple - (w % multiple)) % multiple

    # F.pad order for 5D is (W_left, W_right, H_left, H_right, D_left, D_right)
    pad = (0, w_pad, 0, h_pad, 0, d_pad)
    x_pad = F.pad(x, pad, mode=mode, value=value)
    return x_pad, pad


def unpad_3d(x, pad):
    """
    x: [B, C, D, H, W]
    pad: (0, w_pad, 0, h_pad, 0, d_pad)
    """
    _, w_pad, _, h_pad, _, d_pad = pad

    if d_pad > 0:
        x = x[:, :, :-d_pad, :, :]
    if h_pad > 0:
        x = x[:, :, :, :-h_pad, :]
    if w_pad > 0:
        x = x[:, :, :, :, :-w_pad]

    return x


def prepare_condition(cfg, data_dict, missing_key):
    '''Prepares the conditioning tensor and target for inference.
    Handles modality masking and presence masks based on config.'''
    cond = []
    mask = []
    target = data_dict[missing_key].copy()

    fill_value = cfg["missing_policy"]["fill_value"]
    use_presence_mask = cfg["missing_policy"]["use_presence_mask"]

    for k in MOD_KEYS:
        if k == missing_key:
            cond.append(np.full_like(data_dict[k], fill_value, dtype=np.float32))
            mask.append(np.zeros_like(data_dict[k], dtype=np.float32))
        else:
            cond.append(data_dict[k].astype(np.float32))
            mask.append(np.ones_like(data_dict[k], dtype=np.float32))

    cond = np.stack(cond, axis=0)  # [4, D, H, W]

    if use_presence_mask:
        mask = np.stack(mask, axis=0)  # [4, D, H, W]
        cond = np.concatenate([cond, mask], axis=0)  # [8, D, H, W]

    return cond, target


def pad_to_multiple_3d(x, multiple=32, value=-1.0, mode="constant"):
    _, _, d, h, w = x.shape
    d_pad = (multiple - (d % multiple)) % multiple
    h_pad = (multiple - (h % multiple)) % multiple
    w_pad = (multiple - (w % multiple)) % multiple
    pad = (0, w_pad, 0, h_pad, 0, d_pad)
    x_pad = F.pad(x, pad, mode=mode, value=value)
    return x_pad, pad


def unpad_3d(x, pad):
    _, w_pad, _, h_pad, _, d_pad = pad
    if d_pad > 0: x = x[:, :, :-d_pad, :, :]
    if h_pad > 0: x = x[:, :, :, :-h_pad, :]
    if w_pad > 0: x = x[:, :, :, :, :-w_pad]
    return x


def resolve_checkpoint_path(cfg):
    '''Determines the checkpoint path to load for inference based on config settings.'''
    ckpt_path = cfg["inference"].get("checkpoint_path")
    run_dir = cfg["inference"].get("run_dir")
    ckpt_name = cfg["inference"].get("checkpoint_name", "best.pt")

    if ckpt_path:
        return ckpt_path

    if run_dir:
        return os.path.join(run_dir, "checkpoints", ckpt_name)

    raise ValueError("Set either inference.checkpoint_path or inference.run_dir in config.")


def build_infer_model(cfg, device):
    '''Builds the model for inference based on the config. Supports multiple architectures.'''
    model_name = cfg["model"]["name"].lower()
    model_kwargs = {
        "in_channels": cfg["model"]["in_channels"],
        "out_channels": cfg["model"]["out_channels"],
    }

    if model_name == "unet":
        model_kwargs["base_ch"] = cfg["model"]["base_ch"]

    elif model_name == "convnext3d":
        model_kwargs["base_dim"] = cfg["model"]["base_dim"]

    elif model_name == "monai_diffusion":
        model_kwargs["channels"] = tuple(cfg["model"]["channels"])
        model_kwargs["attention_levels"] = tuple(cfg["model"]["attention_levels"])
        model_kwargs["num_res_blocks"] = cfg["model"]["num_res_blocks"]
        model_kwargs["num_head_channels"] = cfg["model"]["num_head_channels"]
        model_kwargs["norm_num_groups"] = cfg["model"].get("norm_num_groups", 32)
        model_kwargs["norm_eps"] = cfg["model"].get("norm_eps", 1e-6)
        model_kwargs["resblock_updown"] = cfg["model"].get("resblock_updown", False)
        model_kwargs["transformer_num_layers"] = cfg["model"].get("transformer_num_layers", 1)
        model_kwargs["dropout_cattn"] = cfg["model"].get("dropout_cattn", 0.0)
        # --- NEW: Inject the class embedder size (Update to 5!) ---
        if cfg["model"].get("use_target_class_embed", True):
            model_kwargs["num_class_embeds"] = 5  # 0=T1, 1=T1ce, 2=T2, 3=FLAIR, 4=NULL

    elif model_name == "swin_ddpm":
        model_kwargs["img_size"] = tuple(cfg["patch"]["size"])
        model_kwargs["feature_size"] = cfg["model"].get("feature_size", 48)
        model_kwargs["depths"] = tuple(cfg["model"].get("depths", [2, 2, 2, 2]))
        model_kwargs["num_heads"] = tuple(cfg["model"].get("num_heads", [4, 4, 8, 16]))
        model_kwargs["window_size"] = tuple(cfg["model"].get("window_size", [4, 4, 4]))
        model_kwargs["drop_rate"] = cfg["model"].get("drop_rate", 0.0)
        model_kwargs["attn_drop_rate"] = cfg["model"].get("attn_drop_rate", 0.0)
        model_kwargs["use_checkpoint"] = cfg["model"].get("use_checkpoint", True)
        model_kwargs["spatial_dims"] = cfg["model"].get("spatial_dims", 3)
    # ----------------------

    model = build_model(model_name, **model_kwargs).to(device)
    return model


def build_available_support_mask(cond_np, missing_key):
    """
    cond_np: [4 or 8, D, H, W]
    Uses presence masks only to identify WHICH modality slots are available,
    then uses the image channels themselves to find tissue support.
    """
    if cond_np.shape[0] >= 8:
        # channels 4:8 are presence masks; each is all-ones or all-zeros
        available_idx = [i for i in range(4) if cond_np[4 + i].max() > 0.5]
    else:
        available_idx = [i for i, k in enumerate(MOD_KEYS) if k != missing_key]

    if len(available_idx) == 0:
        raise ValueError("No available input modalities found to build support mask.")

    support_masks = [(cond_np[i] > -0.999) for i in available_idx]
    return np.any(np.stack(support_masks, axis=0), axis=0)


@torch.inference_mode()
def infer_single_case(cfg, model, diffusion, case_dir, missing_key, device):
    '''Runs inference on a single case using a Gaussian sliding window to protect VRAM and eliminate seams.'''
    data_dict, affine, header = load_case(case_dir)

    raw_data_dict = {k: v.copy() for k, v in data_dict.items()}

    norm_data_dict = {}
    for k, vol in data_dict.items():
        norm_data_dict[k] = normalize_strict_bound(vol)

    cond_np, target_np = prepare_condition(cfg, norm_data_dict, missing_key)

    available_mask = build_available_support_mask(cond_np, missing_key)

    coords = np.argwhere(available_mask)
    if len(coords) > 0:
        d0, h0, w0 = coords.min(axis=0)
        d1, h1, w1 = coords.max(axis=0) + 1
    else:
        d0, h0, w0 = 0, 0, 0
        d1, h1, w1 = target_np.shape

    cond_crop = cond_np[:, d0:d1, h0:h1, w0:w1]
    target_crop = target_np[d0:d1, h0:h1, w0:w1]

    cond = torch.tensor(cond_crop[None], dtype=torch.float32, device=device)

    # No manual padding needed! Sliding window handles dimensions perfectly.
    x = torch.randn((1, 1, *cond.shape[2:]), device=device)

    trained_betas_np = diffusion.betas.detach().cpu().numpy()

    scheduler = DDIMScheduler(
        trained_betas=trained_betas_np,
        beta_schedule="scaled_linear",
        clip_sample=True,  # Anchors the healthy tissue
        clip_sample_range=1.5,  # The pressure valve for the tumor
        thresholding=False,  # Prevents global gray mush
    )
    scheduler.set_timesteps(cfg["inference"]["reverse_steps"])

    use_sw = cfg["inference"]["sliding_window"]["enabled"]
    roi_size = tuple(int(v) for v in cfg["inference"]["sliding_window"]["roi_size"])
    sw_batch_size = cfg["inference"]["sliding_window"]["sw_batch_size"]
    overlap = cfg["inference"]["sliding_window"]["overlap"]

    start = time.time()
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    use_class_embed = cfg["model"].get("use_target_class_embed", True)
    guidance_scale = cfg["inference"].get("cfg_guidance_scale", 1.0)

    mod_to_idx = {"t1": 0, "t1ce": 1, "t2": 2, "flair": 3}
    target_idx = torch.tensor([mod_to_idx[missing_key]], device=device, dtype=torch.long)
    null_idx = torch.tensor([5], device=device, dtype=torch.long)

    for t_idx in scheduler.timesteps:
        t_scalar = int(t_idx.item()) if torch.is_tensor(t_idx) else int(t_idx)
        t = torch.tensor([t_scalar], device=device, dtype=torch.long)

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            if guidance_scale > 1.0:
                cond_zero = torch.zeros_like(cond)

                # Wrap predictors to safely expand batch sizes for the sliding window
                def pred_cond_fn(patch_in):
                    bs = patch_in.shape[0]
                    return model(patch_in, t.expand(bs),
                                 class_labels=target_idx.expand(bs) if use_class_embed else None)

                def pred_uncond_fn(patch_in):
                    bs = patch_in.shape[0]
                    return model(patch_in, t.expand(bs), class_labels=null_idx.expand(bs) if use_class_embed else None)

                # --- CRITICAL FIX: mode="gaussian" ---
                noise_cond = sliding_window_inference(
                    inputs=torch.cat([cond, x], dim=1),
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=pred_cond_fn,
                    overlap=overlap,
                    mode="gaussian"
                )
                noise_uncond = sliding_window_inference(
                    inputs=torch.cat([cond_zero, x], dim=1),
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=pred_uncond_fn,
                    overlap=overlap,
                    mode="gaussian"
                )

                pred_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                def predictor(patch_in):
                    bs = patch_in.shape[0]
                    return model(patch_in, t.expand(bs),
                                 class_labels=target_idx.expand(bs) if use_class_embed else None)

                pred_noise = sliding_window_inference(
                    inputs=torch.cat([cond, x], dim=1),
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=predictor,
                    overlap=overlap,
                    mode="gaussian"
                )

        x = scheduler.step(pred_noise.float(), t_idx, x.float()).prev_sample

    elapsed = time.time() - start

    pred_crop = x[0, 0].detach().float().cpu().numpy()
    pred_crop = np.clip(pred_crop, -1.0, 1.0)

    pred_full = np.full_like(target_np, -1.0, dtype=np.float32)
    pred_full[d0:d1, h0:h1, w0:w1] = pred_crop

    pred_full = np.where(available_mask, pred_full, -1.0)
    eval_mask_full = available_mask.astype(np.float32)

    return (
        pred_full.astype(np.float32),
        target_np.astype(np.float32),
        cond_np.astype(np.float32),
        affine,
        header,
        elapsed,
        norm_data_dict,
        raw_data_dict,
        eval_mask_full,
    )


def main():
    cfg = load_config("configs/base.yaml")

    device = cfg["runtime"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    ckpt_path = resolve_checkpoint_path(cfg)

    model = build_infer_model(cfg, device)
    ema = EMA(model, decay=cfg["ema"]["decay"]) if cfg["ema"]["enabled"] else None
    load_checkpoint(
        ckpt_path,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        ema=ema,
        map_location=device,
    )
    infer_model = ema.shadow if (ema is not None and cfg["ema"]["infer_with_ema"]) else model
    infer_model.eval()

    # --- COMPILER TRICK FOR BLACKWELL ---
    if cfg["inference"].get("use_compile", False):
        if os.name == "nt":
            print("Warning: torch.compile is not fully supported on Windows. Skipping compile.")
        else:
            print("Compiling model for Linux (max-autotune). The first volume will take a few extra minutes...")
            # max-autotune optimizes the graph specifically for inference speed
            infer_model = torch.compile(infer_model, mode="max-autotune")
    # ------------------------------------

    diffusion = DiffusionScheduler(
        timesteps=cfg["diffusion"]["timesteps"],
        schedule=cfg["diffusion"].get("schedule", "linear"),
        cosine_s=cfg["diffusion"].get("cosine_s", 0.008),
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
    ).to(device)

    metric_bundle = ImageMetricBundle(
        compute_mae=cfg["metrics"]["compute_mae"],
        compute_psnr=cfg["metrics"]["compute_psnr"],
        compute_ssim=cfg["metrics"]["compute_ssim"],
        eval_norm=cfg.get("metrics", {}).get("eval_norm", "per_case_minmax"),
    )

    data_root = cfg["inference"]["data_root"]
    case_ids = cfg["inference"].get("case_ids", [])
    missing_keys = cfg["inference"].get("missing_keys", ["t1ce"])

    case_dirs = resolve_case_dirs(
        data_root,
        case_ids=case_ids,
        strict=False,
        random_case_count=cfg["inference"].get("random_case_count", 0),
        random_case_seed=cfg["inference"].get("random_case_seed", 42),
    )

    # --- NEW: Generate a Unique Run ID ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    custom_run_name = cfg["inference"].get("run_name", "run")
    unique_run_id = f"{custom_run_name}_{timestamp}"

    base_out = os.path.join(
        cfg["project"]["output_root"],
        cfg["inference"]["output_subdir"],
        cfg["model"]["name"],
        unique_run_id  # <--- Nests everything safely inside a unique folder!
    )
    ensure_dir(base_out)

    # --- FIX: SAFE JSON LOADING ---
    summary_path = os.path.join(base_out, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
    else:
        summary = []
    # ------------------------------
    for case_dir in case_dirs:
        case_id = os.path.basename(case_dir)

        for missing_key in missing_keys:
            pred, target, cond, affine, header, elapsed, norm_data_dict, raw_data_dict, eval_mask = infer_single_case(
                cfg, infer_model, diffusion, case_dir, missing_key, device
            )

            pred_t = torch.tensor(pred[None, None], dtype=torch.float32)
            target_t = torch.tensor(target[None, None], dtype=torch.float32)
            mask_t = torch.tensor(eval_mask[None, None], dtype=torch.float32)

            infer_metrics = metric_bundle(pred_t, target_t, mask=mask_t)
            # --- NEW: Parallel Full-Volume Metrics ---
            # Compute strict clinical metrics in [-1.0, 1.0] space
            infer_metrics = metric_bundle(pred_t, target_t, mask=mask_t)
            if cfg["metrics"].get("compute_full_volume", False):
                full_metrics = metric_bundle(pred_t, target_t, mask=None)
                for k, v in full_metrics.items():
                    infer_metrics[f"full_{k}"] = v

            item_dir = os.path.join(base_out, f"{case_id}_{missing_key}")
            ensure_dir(item_dir)

            # ---------------------------------------------------------
            # --- NEW: UN-NORMALIZE FOR CLINICAL EXPORT & VISUALS ---
            # ---------------------------------------------------------
            raw_target = raw_data_dict[missing_key]
            raw_min, raw_max = raw_target.min(), raw_target.max()

            # Map prediction from [-1, 1] to [0, 1]
            pred_01 = (pred + 1.0) / 2.0

            # Project [0, 1] to the patient's native physical intensity
            pred_native = (pred_01 * (raw_max - raw_min)) + raw_min

            # Mask out the background to perfectly match the raw file
            pred_native = np.where(eval_mask, pred_native, raw_target.min())

            if cfg["inference"]["save_nifti"]:
                nifti_path = os.path.join(item_dir, f"{case_id}_pred_{missing_key}.nii.gz")
                save_prediction_nifti(pred_native, affine, header, nifti_path)

                if cfg["inference"].get("save_composite_nifti", True):
                    composite = []
                    for k in MOD_KEYS:
                        if k == missing_key:
                            composite.append(pred_native)
                        else:
                            composite.append(raw_data_dict[k])  # Use RAW data

                    composite = np.stack(composite, axis=0).astype(np.float32)
                    comp_path = os.path.join(item_dir, f"{case_id}_composite_{missing_key}.nii.gz")
                    save_prediction_nifti(composite, affine, header, comp_path)

            if cfg["inference"]["save_png"]:
                png_path = os.path.join(item_dir, f"{case_id}_panel_{missing_key}.png")
                # Pass the native target and native prediction so the Abs Diff math is accurate
                save_slice_panel(cond, raw_target, pred_native, png_path, missing_key=missing_key, title=f"{case_id}")

            summary.append({
                "case_id": case_id,
                "missing_key": missing_key,
                "elapsed_sec": elapsed,
                "checkpoint": ckpt_path,
                "reverse_steps": cfg["inference"]["reverse_steps"],
                "overlap": cfg["inference"]["sliding_window"]["overlap"],
                "roi_size": cfg["inference"]["sliding_window"]["roi_size"],
                **infer_metrics,
            })

            print(f"[infer] case={case_id} missing={missing_key} time={elapsed:.2f}s")
            save_json(summary_path, summary)

    print(f"Saved inference outputs to: {base_out}")


if __name__ == "__main__":
    main()