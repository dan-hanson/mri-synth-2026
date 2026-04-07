import os
import sys
import time
import torch
import numpy as np

sys.path.append(r"C:\mri_synth_2026\src")

from mri_missing.config import load_config, ensure_dir
from mri_missing.models.registry import build_model
from mri_missing.diffusion.scheduler import DiffusionScheduler
from mri_missing.models.time_embedding import SinusoidalTimeEmbedding
from mri_missing.utils.io import load_checkpoint, save_json
from mri_missing.utils.cases import MOD_KEYS, resolve_case_dirs
from mri_missing.utils.nifti import load_case, save_prediction_nifti, normalize_zscore, normalize_per_case_01_np
from mri_missing.utils.visualization import save_slice_panel
from monai.inferers import sliding_window_inference
from mri_missing.utils.ema import EMA
from mri_missing.metrics.image_metrics import ImageMetricBundle

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def prepare_condition(cfg, data_dict, missing_key):
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

    cond = np.stack(cond, axis=0)   # [4, D, H, W]

    if use_presence_mask:
        mask = np.stack(mask, axis=0)   # [4, D, H, W]
        cond = np.concatenate([cond, mask], axis=0)   # [8, D, H, W]

    return cond, target


def resolve_checkpoint_path(cfg):
    ckpt_path = cfg["inference"].get("checkpoint_path")
    run_dir = cfg["inference"].get("run_dir")
    ckpt_name = cfg["inference"].get("checkpoint_name", "best.pt")

    if ckpt_path:
        return ckpt_path

    if run_dir:
        return os.path.join(run_dir, "checkpoints", ckpt_name)

    raise ValueError("Set either inference.checkpoint_path or inference.run_dir in config.")


def build_infer_model(cfg, device):
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

    model = build_model(model_name, **model_kwargs).to(device)
    return model


@torch.inference_mode()
def infer_single_case(cfg, model, diffusion, case_dir, missing_key, device):
    data_dict, affine, header = load_case(case_dir)

    # --- ADD THIS: Save a raw copy for the composite output ---
    raw_data_dict = {k: v.copy() for k, v in data_dict.items()}

    # training-style normalization
    for k in data_dict:
        data_dict[k] = normalize_zscore(data_dict[k])

    cond_np, target_np = prepare_condition(cfg, data_dict, missing_key)

    # --- THE BOUNDING BOX CROP ---
    # Find the bounds of the actual brain (ignore the background)
    mask = target_np > target_np.min()
    coords = np.argwhere(mask)
    if len(coords) > 0:
        d0, h0, w0 = coords.min(axis=0)
        d1, h1, w1 = coords.max(axis=0) + 1
    else:
        d0, h0, w0 = 0, 0, 0
        d1, h1, w1 = target_np.shape

    # Crop the inputs to just the brain
    cond_crop = cond_np[:, d0:d1, h0:h1, w0:w1]
    target_crop = target_np[d0:d1, h0:h1, w0:w1]

    # Initialize tensors based on the CROPPED shape
    cond = torch.tensor(cond_crop[None], dtype=torch.float32, device=device)
    x = torch.randn((1, 1, *target_crop.shape), device=device)

    reverse_steps = cfg["inference"]["reverse_steps"]
    total_train_steps = cfg["diffusion"]["timesteps"]

    if reverse_steps >= total_train_steps:
        t_schedule = list(reversed(range(total_train_steps)))
    else:
        idxs = np.linspace(0, total_train_steps - 1, reverse_steps, dtype=int)
        t_schedule = list(reversed(idxs.tolist()))

    use_sw = cfg["inference"]["sliding_window"]["enabled"]
    roi_size = tuple(int(x) for x in cfg["inference"]["sliding_window"]["roi_size"])
    sw_batch_size = cfg["inference"]["sliding_window"]["sw_batch_size"]
    overlap = cfg["inference"]["sliding_window"]["overlap"]

    start = time.time()

    # 1. FIX: Added enumerate() so 'i' exists
    for i, t_idx in enumerate(t_schedule):
        t = torch.tensor([t_idx], device=device, dtype=torch.long)
        model_in = torch.cat([cond, x], dim=1)  

        if use_sw:
            def predictor(patch_x):
                return model(patch_x, t)

            pred_noise = sliding_window_inference(
                inputs=model_in,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=predictor,
                overlap=overlap,
                mode="gaussian"   # <--- THE MAGIC BULLET
            )

            pred_noise = sliding_window_inference(
                inputs=model_in,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=predictor,
                overlap=overlap,
            )
        else:
            pred_noise = model(model_in, t)

        # --- THE DDIM MATH ---
        alpha_bar_t = diffusion.alpha_cumprod[t_idx].view(1, 1, 1, 1, 1)

        # Look ahead to the next timestep in our shortened schedule
        if i < len(t_schedule) - 1:
            t_prev = t_schedule[i+1]
            alpha_bar_prev = diffusion.alpha_cumprod[t_prev].view(1, 1, 1, 1, 1)
        else:
            alpha_bar_prev = torch.ones_like(alpha_bar_t)

        # Predict the fully clean image (x0)
        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        pred_x0 = torch.clamp(pred_x0, min=-3.0, max=3.0) # Stability clamp
        
        # Add back the exact amount of noise needed for the NEXT leap
        x = torch.sqrt(alpha_bar_prev) * pred_x0 + torch.sqrt(1 - alpha_bar_prev) * pred_noise

        # ---> OLD DDPM MATH HAS BEEN COMPLETELY DELETED FROM HERE <---

    elapsed = time.time() - start

    # Grab the cropped prediction
    pred_cropped = x[0, 0].detach().cpu().numpy()

    # --- NORMALIZATION FIX ---
    # Normalize ONLY the brain crop to 0-1 BEFORE pasting it
    pred_cropped = normalize_per_case_01_np(pred_cropped)

    # Create an empty black volume (0s)
    pred_full = np.zeros_like(target_np)
    
    # Paste the normalized brain back into the exact original coordinates
    pred_full[d0:d1, h0:h1, w0:w1] = pred_cropped
    
    # --- BACKGROUND MASKING ---
    # Find exactly where the real brain is
    brain_mask = target_np > target_np.min()
    # Multiply the prediction by the mask (Brain * 1, Background * 0)
    pred_full = pred_full * brain_mask
    # ----------------------------------------

    return pred_full, target_np, cond_np, affine, header, elapsed, raw_data_dict


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

    # time_embed = SinusoidalTimeEmbedding(128).to(device)

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

    base_out = os.path.join(
        cfg["project"]["output_root"],
        cfg["inference"]["output_subdir"],
        cfg["model"]["name"],
    )
    ensure_dir(base_out)

    summary = []

    for case_dir in case_dirs:
        case_id = os.path.basename(case_dir)

        for missing_key in missing_keys:
            pred, target, cond, affine, header, elapsed, raw_data_dict = infer_single_case(
                cfg, infer_model, diffusion, case_dir, missing_key, device
            )

            pred_t = torch.tensor(pred[None, None], dtype=torch.float32)
            target_t = torch.tensor(target[None, None], dtype=torch.float32)
            # --- Generate the binary mask tensor ---
            mask_t = torch.tensor((target > target.min())[None, None], dtype=torch.float32)
            
            # Pass the mask into the bundle
            infer_metrics = metric_bundle(pred_t, target_t, mask=mask_t)

            item_dir = os.path.join(base_out, f"{case_id}_{missing_key}")
            ensure_dir(item_dir)

            if cfg["inference"]["save_nifti"]:
                # Save just the prediction
                nifti_path = os.path.join(item_dir, f"{case_id}_pred_{missing_key}.nii.gz")
                save_prediction_nifti(pred, affine, header, nifti_path)

                # --- ADD THIS: Save the Composite 4-Modality NIfTI ---
                if cfg["inference"].get("save_composite_nifti", True):
                    composite = []
                    for k in MOD_KEYS:
                        if k == missing_key:
                            composite.append(pred) # Plug in the synthetic modality
                        else:
                            composite.append(raw_data_dict[k]) # Keep the real modalities unaltered
                    
                    composite = np.stack(composite, axis=0)
                    comp_path = os.path.join(item_dir, f"{case_id}_composite_{missing_key}.nii.gz")
                    save_prediction_nifti(composite, affine, header, comp_path)

            item_dir = os.path.join(base_out, f"{case_id}_{missing_key}")
            ensure_dir(item_dir)

            if cfg["inference"]["save_nifti"]:
                nifti_path = os.path.join(item_dir, f"{case_id}_pred_{missing_key}.nii.gz")
                save_prediction_nifti(pred, affine, header, nifti_path)

            if cfg["inference"]["save_png"]:
                png_path = os.path.join(item_dir, f"{case_id}_panel_{missing_key}.png")
                save_slice_panel(cond, target, pred, png_path, missing_key=missing_key, title=f"{case_id}")

            summary.append({
                "case_id": case_id,
                "missing_key": missing_key,
                "elapsed_sec": elapsed,
                "checkpoint": ckpt_path,
                **infer_metrics,
            })

            print(f"[infer] case={case_id} missing={missing_key} time={elapsed:.2f}s")
            

    save_json(os.path.join(base_out, "summary.json"), summary)
    print(f"Saved inference outputs to: {base_out}")


if __name__ == "__main__":
    main()