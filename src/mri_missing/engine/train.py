import os
import sys
import time
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import concurrent.futures

sys.path.append(r"C:\mri_synth_2026\src")

from mri_missing.config import load_config, ensure_dir, make_run_dir, save_config_copy
from mri_missing.seed import seed_everything
from mri_missing.data.dataset import BraTSDataset
from mri_missing.models.registry import build_model
from mri_missing.diffusion.scheduler import DiffusionScheduler
from mri_missing.utils.io import save_checkpoint, load_checkpoint, save_json
from mri_missing.utils.stats import StepTimer, RunTimer, get_vram_mb, reset_vram_stats, count_params
from mri_missing.metrics.image_metrics import ImageMetricBundle
from mri_missing.utils.visualization import save_history_plots, save_augmentation_panel
from mri_missing.losses.image_losses import CompositeSynthesisLoss
from mri_missing.utils.ema import EMA
from diffusers import DDIMScheduler
from contextlib import nullcontext

# Enable TF32 on compatible NVIDIA GPUs for faster training (with a potential minor impact on precision)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def heavy_rank_key(val_stats): # *** probably could be improved with a more complex weighted formula ***
    '''Ranking key for heavy validation checkpoints. Prioritizes SSIM, then MAE, then noise MSE.'''
    return (
        float(val_stats.get("val_raw_ssim", val_stats.get("val_ssim", -1e9))),
        -float(val_stats.get("val_raw_mae", val_stats.get("val_mae", 1e9))),
        -float(val_stats.get("val_noise_mse", 1e9)),
    )

def _norm_high(x, lo, hi):
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (float(x) - lo) / (hi - lo)))


def _norm_low(x, good, bad):
    if bad <= good:
        return 0.0
    return max(0.0, min(1.0, 1.0 - ((float(x) - good) / (bad - good))))


def heavy_should_consider(val_stats, step, max_steps, cfg):
    ckpt_cfg = cfg.get("checkpoint", {})
    min_progress = float(ckpt_cfg.get("heavy_min_progress", 0.20))

    if step < int(min_progress * max_steps):
        return False

    raw_ssim = float(val_stats.get("val_raw_ssim", val_stats.get("val_ssim", -1e9)))
    raw_psnr = float(val_stats.get("val_raw_psnr", val_stats.get("val_psnr", -1e9)))
    raw_mae = float(val_stats.get("val_raw_mae", val_stats.get("val_mae", 1e9)))
    noise = float(val_stats.get("val_noise_mse", 1e9))

    if raw_ssim < float(ckpt_cfg.get("heavy_min_raw_ssim", 0.08)):
        return False
    if raw_psnr < float(ckpt_cfg.get("heavy_min_raw_psnr", 10.0)):
        return False
    if raw_mae > float(ckpt_cfg.get("heavy_max_raw_mae", 0.35)):
        return False
    if noise > float(ckpt_cfg.get("heavy_max_noise_mse", 0.35)):
        return False

    return True


def heavy_rank_score(val_stats, cfg):
    ckpt_cfg = cfg.get("checkpoint", {})

    raw_ssim = float(val_stats.get("val_raw_ssim", val_stats.get("val_ssim", 0.0)))
    raw_psnr = float(val_stats.get("val_raw_psnr", val_stats.get("val_psnr", 0.0)))
    raw_mae = float(val_stats.get("val_raw_mae", val_stats.get("val_mae", 1.0)))
    noise = float(val_stats.get("val_noise_mse", 1.0))

    # Default normalization ranges: tune later if needed
    ssim_term = _norm_high(raw_ssim,
                           ckpt_cfg.get("rank_ssim_lo", 0.05),
                           ckpt_cfg.get("rank_ssim_hi", 0.50))

    psnr_term = _norm_high(raw_psnr,
                           ckpt_cfg.get("rank_psnr_lo", 10.0),
                           ckpt_cfg.get("rank_psnr_hi", 24.0))

    mae_term = _norm_low(raw_mae,
                         ckpt_cfg.get("rank_mae_good", 0.08),
                         ckpt_cfg.get("rank_mae_bad", 0.30))

    noise_term = _norm_low(noise,
                           ckpt_cfg.get("rank_noise_good", 0.03),
                           ckpt_cfg.get("rank_noise_bad", 0.25))

    w_ssim = float(ckpt_cfg.get("w_ssim", 0.45))
    w_psnr = float(ckpt_cfg.get("w_psnr", 0.20))
    w_mae = float(ckpt_cfg.get("w_mae", 0.20))
    w_noise = float(ckpt_cfg.get("w_noise", 0.15))

    score = (
        w_ssim * ssim_term +
        w_psnr * psnr_term +
        w_mae * mae_term +
        w_noise * noise_term
    )

    # Keep a tuple so ties are still resolved sensibly
    return (
        float(score),
        float(raw_ssim),
        -float(raw_mae),
        float(raw_psnr),
        -float(noise),
    )

def make_light_val_batch(val_loader, device=None):
    '''Extract a single batch from the validation loader for quick, frequent validation during training.'''
    batch = next(iter(val_loader))
    cond, target, keys, target_idx, case_ids = batch # Unpack 5 vars
    return [(cond.clone(), target.clone(), keys, target_idx.clone(), case_ids)]


def build_optimizer(cfg, model):
    '''Build optimizer based on configuration. Supports Adam and AdamW.'''
    name = cfg["optim"]["name"].lower()
    lr = cfg["optim"]["lr"]
    wd = cfg["optim"]["weight_decay"]

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(cfg, optimizer):
    '''Build learning rate scheduler based on config. Supports cosine_annealing and constant_with_warmup.'''
    name = cfg["scheduler"]["name"].lower()

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["scheduler"]["t_max"],
            eta_min=cfg["scheduler"]["eta_min"],
        )
    elif name == "constant_with_warmup":
        warmup_steps = cfg["scheduler"].get("warmup_steps", 2500)
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0  # Stay flat after warmup
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {name}")

def should_stop(run_dir, cfg):
    '''Check if a stop file exists in the run directory, indicating that training should be halted.'''
    stop_file = cfg.get("control", {}).get("stop_file", "STOP")
    stop_path = os.path.join(run_dir, stop_file)
    return os.path.exists(stop_path)

@torch.inference_mode()
def validate(
    model,
    batches,
    scheduler,
    loss_fn,
    metric_bundle,
    device,
    use_amp,
    timesteps,
    compute_heavy=False,
    current_step=0,
    max_steps=1,
    reverse_steps=15,
    raw_metric_bundle=None,
):
    was_training = model.training
    model.eval()

    loss_accum = {}
    metric_accum = {}
    modality_metrics = {}

    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)
        if device == "cuda"
        else nullcontext()
    )

    ddim_scheduler = None
    if compute_heavy:
        trained_betas_np = scheduler.betas.detach().cpu().numpy()
        ddim_scheduler = DDIMScheduler(
            trained_betas=trained_betas_np,
            beta_schedule="scaled_linear", # <-- CRITICAL ADDITION
            clip_sample=True,
            clip_sample_range=1.0,
        )

    for batch in batches:
        # Unpack the new structures
        if len(batch) == 6:
            cond, target, keys, target_idx, case_ids, fixed_latent = batch
        else:
            cond, target, keys, target_idx, case_ids = batch
            fixed_latent = None

        cond = cond.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target_idx = target_idx.to(device, non_blocking=True) # Send to GPU

        # -----------------------------
        # light validation forward pass
        # -----------------------------
        t = scheduler.sample_timesteps(cond.shape[0], device)
        x_t, noise = scheduler.q_sample(target, t)
        x_in = torch.cat([cond, x_t], dim=1)

        with amp_ctx:
            # Pass class_labels!
            pred_noise = model(x_in, t, class_labels=target_idx)

        alpha_bar = scheduler.alpha_cumprod[t].view(-1, 1, 1, 1, 1).float()
        snr = alpha_bar / (1.0 - alpha_bar + 1e-8)

        pred_x0_1step = (
            x_t.float() - torch.sqrt(1.0 - alpha_bar) * pred_noise.float()
        ) / (torch.sqrt(alpha_bar) + 1e-5)
        pred_x0_1step = torch.clamp(pred_x0_1step, min=-1.0, max=1.0)

        total_loss, loss_parts = loss_fn(
            pred_noise.float(),
            noise.float(),
            pred_x0_1step.float(),
            target.float(),
            snr=snr,
            current_step=current_step,
            max_steps=max_steps,
        )

        loss_accum.setdefault("val_total_loss", []).append(float(total_loss.item()))
        for k, v in loss_parts.items():
            loss_accum.setdefault(f"val_{k}", []).append(float(v))

        # -----------------------------
        # heavy validation forward pass
        # -----------------------------
        if compute_heavy:
            # RESTORED: Generate or load the starting noise
            if fixed_latent is not None:
                x_gen = fixed_latent.to(device, non_blocking=True).clone()
            else:
                x_gen = torch.randn_like(target)

            # RESTORED: Set the step jumps for the scheduler
            ddim_scheduler.set_timesteps(reverse_steps)

            with amp_ctx:
                for t_idx in ddim_scheduler.timesteps:
                    t_scalar = int(t_idx.item()) if torch.is_tensor(t_idx) else int(t_idx)
                    t_tensor = torch.full(
                        (cond.shape[0],), t_scalar, device=device, dtype=torch.long
                    )

                    model_in = torch.cat([cond, x_gen], dim=1)
                    
                    # Pass class_labels!
                    p_noise = model(model_in, t_tensor, class_labels=target_idx) 

                    x_gen = ddim_scheduler.step(p_noise, t_idx, x_gen).prev_sample

            for b in range(target.shape[0]):
                mod = keys[b] if isinstance(keys, (list, tuple)) else keys
                p_b = x_gen[b:b+1]
                t_b = target[b:b+1]

                if cond.shape[1] >= 8:
                    mask_b = (cond[b:b+1, 4:8] > 0.5).any(dim=1, keepdim=True).float()
                else:
                    mask_b = (cond[b:b+1, :4] > -0.999).any(dim=1, keepdim=True).float()

                metrics = metric_bundle(p_b, t_b, mask=mask_b)
                for k, v in metrics.items():
                    metric_accum.setdefault(k, []).append(float(v))
                    modality_metrics.setdefault(mod, {}).setdefault(k, []).append(float(v))

                if raw_metric_bundle is not None:
                    raw_metrics = raw_metric_bundle(p_b, t_b, mask=mask_b)
                    for k, v in raw_metrics.items():
                        rk = f"raw_{k}"
                        metric_accum.setdefault(rk, []).append(float(v))
                        modality_metrics.setdefault(mod, {}).setdefault(rk, []).append(float(v))

    out = {k: sum(vals) / max(len(vals), 1) for k, vals in loss_accum.items()}

    if compute_heavy:
        for k, vals in metric_accum.items():
            out[f"val_{k}"] = sum(vals) / max(len(vals), 1)
        for mod, m_dict in modality_metrics.items():
            for k, vals in m_dict.items():
                out[f"val_{mod}_{k}"] = sum(vals) / max(len(vals), 1)

    if was_training:
        model.train()

    return out


def main():
    cfg = load_config("configs/base.yaml")

    seed_everything(cfg["seed"]["value"], cfg["seed"]["deterministic"])

    device = cfg["runtime"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if cfg["runtime"]["cudnn_benchmark"] and device == "cuda" and not cfg["seed"]["deterministic"]:
        torch.backends.cudnn.benchmark = True

    run_dir, run_name = make_run_dir(cfg)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)
    save_config_copy(cfg, os.path.join(run_dir, "config.yaml"))

    train_ds = BraTSDataset(
        cfg["data"]["train_cache_root"] if cfg["data"]["backend"] == "pt_cache" else cfg["data"]["train_root"],
        patch_size=tuple(cfg["patch"]["size"]),
        fill_value=cfg["missing_policy"]["fill_value"],
        use_presence_mask=cfg["missing_policy"]["use_presence_mask"],
        sampling_probs=cfg["missing_policy"].get("sampling_probs", None),
        backend=cfg["data"]["backend"],
        augmentation=cfg.get("augmentation", {}),
    )

    val_aug_cfg = dict(cfg.get("augmentation", {}))
    val_aug_cfg["enabled"] = False

    val_ds = BraTSDataset(
            cfg["data"]["val_cache_root"] if cfg["data"]["backend"] == "pt_cache" else cfg["data"]["val_root"],
            patch_size=tuple(cfg["patch"]["size"]),
            fill_value=cfg["missing_policy"]["fill_value"],
            use_presence_mask=cfg["missing_policy"]["use_presence_mask"],
            # Force Validation to ONLY test the hard modalities
            sampling_probs={"t1": 0.15, "t1ce": 0.30, "t2": 0.25, "flair": 0.30},
            backend=cfg["data"]["backend"],
            augmentation=val_aug_cfg,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["loader"]["batch_size"],
        shuffle=cfg["loader"]["shuffle"],
        num_workers=cfg["loader"]["num_workers"],
        pin_memory=cfg["loader"]["pin_memory"],
        persistent_workers=cfg["loader"].get("persistent_workers", False),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["loader"]["batch_size"],
        shuffle=False,
        num_workers=cfg["loader"]["num_workers"],
        pin_memory=cfg["loader"]["pin_memory"],
        persistent_workers=cfg["loader"].get("persistent_workers", False),
    )

    # --- VAL MICRO-BATCH ---
    print("\n" + "=" * 60)
    print("Extracting Validation Micro-Batch...")

    golden_batches = []
    golden_gen = torch.Generator(device="cpu")
    golden_gen.manual_seed(cfg["seed"]["value"] + 123456)

    for cond, target, keys, target_idx, case_ids in val_loader: # Unpack 5 vars
        fixed_latent = torch.randn(
            target.shape,
            generator=golden_gen,
            dtype=target.dtype,
        )

        golden_batches.append((
            cond.clone(),
            target.clone(),
            keys,
            target_idx.clone(), # Append target_idx
            case_ids,
            fixed_latent.clone(),
        ))

        if len(golden_batches) >= 8:
            break

    print(f"Locked {len(golden_batches)} batches. Tracking the following volumes:")
    for _, _, keys, _, case_ids, _ in golden_batches: 
        for i in range(len(keys)):
            mod = keys[i] if isinstance(keys, (list, tuple)) else keys
            c_id = case_ids[i] if isinstance(case_ids, (list, tuple)) else case_ids
            print(f"  -> {c_id} [Missing: {mod.upper()}]")
    print("=" * 60 + "\n")
    # ------------------------------

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
        model_kwargs["use_flash_attention"] = cfg["model"].get("use_flash_attention", False)
        if cfg["model"].get("use_target_class_embed", False):
            model_kwargs["num_class_embeds"] = 4

    elif model_name == "swin":
        # Force img_size to perfectly match the patch size to avoid window mismatches
        model_kwargs["img_size"] = tuple(cfg["patch"]["size"])
        model_kwargs["feature_size"] = cfg["model"].get("feature_size", 48)
        model_kwargs["time_embed_dim"] = cfg["model"].get("time_embed_dim", 16)
        model_kwargs["depths"] = tuple(cfg["model"].get("depths", [2, 2, 2, 2]))
        model_kwargs["num_heads"] = tuple(cfg["model"].get("num_heads", [4, 4, 8, 16]))
        model_kwargs["window_size"] = tuple(cfg["model"].get("window_size", [4, 4, 4]))
        model_kwargs["drop_rate"] = cfg["model"].get("drop_rate", 0.0)
        model_kwargs["attn_drop_rate"] = cfg["model"].get("attn_drop_rate", 0.0)

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

    model = build_model(model_name, **model_kwargs).to(device)

    # --- OS-Safe Compiler Switch ---
    if cfg["train"].get("use_compile", False):
        if os.name == "nt":  # 'nt' means Windows
            print("Warning: torch.compile is not fully supported on Windows. Skipping compilation.")
        else:
            print("Compiling model for Linux (this will take a few minutes)...")
            model = torch.compile(model)

    ema = EMA(model, decay=cfg["ema"]["decay"]) if cfg["ema"]["enabled"] else None

    optimizer = build_optimizer(cfg, model)
    scheduler_lr = build_scheduler(cfg, optimizer)
    loss_fn = CompositeSynthesisLoss(
        use_noise_mse=cfg["loss"]["use_noise_mse"],
        noise_mse_weight=cfg["loss"]["noise_mse_weight"],
        use_mae=cfg["loss"]["use_mae"],
        mae_weight=cfg["loss"]["mae_weight"],
        use_ssim=cfg["loss"]["use_ssim"],
        ssim_weight=cfg["loss"]["ssim_weight"],
        # --- NEW: High-Pass Texture Loss ---
        use_texture_loss=cfg["loss"].get("use_texture_loss", False),
        texture_weight_max=cfg["loss"].get("texture_weight_max", 0.05),
        texture_fade_start=cfg["loss"].get("texture_fade_start", 0.4),
        texture_fade_end=cfg["loss"].get("texture_fade_end", 0.6),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["use_amp"] and device == "cuda")

    diffusion = DiffusionScheduler(
        timesteps=cfg["diffusion"]["timesteps"],
        schedule=cfg["diffusion"].get("schedule", "linear"),
        cosine_s=cfg["diffusion"].get("cosine_s", 0.008),
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
    ).to(device)

    # time_embed = SinusoidalTimeEmbedding(128).to(device) -> not needed for monai_diffusion

    metric_bundle = ImageMetricBundle(
        compute_mae=cfg["metrics"]["compute_mae"],
        compute_psnr=cfg["metrics"]["compute_psnr"],
        compute_ssim=cfg["metrics"]["compute_ssim"],
        eval_norm=cfg.get("metrics", {}).get("eval_norm", "per_case_minmax"),
    )

    raw_metric_bundle = ImageMetricBundle(
        compute_mae=cfg["metrics"]["compute_mae"],
        compute_psnr=cfg["metrics"]["compute_psnr"],
        compute_ssim=cfg["metrics"]["compute_ssim"],
        eval_norm=cfg.get("metrics", {}).get("raw_eval_norm", "none"),
    )

    start_step = 0
    best_val = float("inf")
    best_ssim = -float("inf") # track best SSIM separately for clinical checkpoint
    top_heavy = []
    top_k_heavy = int(cfg.get("checkpoint", {}).get("top_k_heavy", 3))

    if cfg["train"]["resume"]:
        ckpt = load_checkpoint(
            cfg["train"]["resume"],
            model,
            optimizer=optimizer,
            scheduler=scheduler_lr,
            scaler=scaler,
            ema=ema,
            map_location=device,
        )
        start_step = ckpt.get("step", 0) + 1
        best_val = ckpt.get("best_val", float("inf"))
        best_ssim = ckpt.get("best_ssim", -float("inf"))

    print(f"Device: {device}")
    print(f"Run: {run_name}")
    print(f"Model: {model_name}")
    print(f"Trainable params: {count_params(model):,}")

    step_timer = StepTimer()
    run_timer = RunTimer()
    history = []

    model.train()
    step = start_step

    # Fetch config value (default to 1 if missing)
    accum_steps = cfg["train"].get("accumulate_grad_batches", 1)
    
    # Initialize zero gradients before the loop starts
    optimizer.zero_grad(set_to_none=True)

    # --- Background IO pool for non-blocking saves ---
    io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    # --- AUGMENTATION PREVIEW ---
    print("\nGenerating Multi-Augmentation Preview Panel...")
    # Pass the raw train_ds (not the loader) so we can sample it repeatedly
    save_augmentation_panel(
        train_ds, 
        os.path.join(run_dir, "00_augmentation_preview.png"), 
        num_samples=5
    )
    print("Preview saved to run directory.\n")
    # ----------------------------

    # ------------------
    # Main Training Loop
    #------------------
    try:
        cond_dropout_prob = cfg["loss"].get("cond_dropout_prob", 0.0)
        use_class_embed = cfg["model"].get("use_target_class_embed", False)
        # 4 is the out-of-bounds index used as the "null" token for CFG
        null_label_index = 4 if use_class_embed else None 

        while step < cfg["train"]["max_steps"]:
            for cond, target, key, target_idx, current_case_id in train_loader:
                
                # RESTORED: Stop file logic
                if should_stop(run_dir, cfg):
                    print("\nStop file detected. Saving checkpoint...")
                    save_checkpoint(
                        os.path.join(ckpt_dir, "stop.pt"),
                        model, optimizer, scheduler_lr, scaler,
                        step, best_val, cfg, ema=ema
                    )
                    print("Saved stop checkpoint.")
                    return

                # RESTORED: VRAM tracking and Timer start
                reset_vram_stats()
                step_timer.tic()

                cond = cond.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                target_idx = target_idx.to(device, non_blocking=True)

                t = diffusion.sample_timesteps(cond.shape[0], device)
                x_t, noise = diffusion.q_sample(target, t)

                # Classifier-Free Guidance (CFG) Dropout Logic
                if cond_dropout_prob > 0.0:
                    drop_mask = torch.rand(cond.shape[0], device=device) < cond_dropout_prob
                    if drop_mask.any():
                        cond_view = cond.clone()
                        cond_view[drop_mask] = 0.0 # Zero out the context modalities
                        cond = cond_view

                        if null_label_index is not None:
                            target_idx_view = target_idx.clone()
                            target_idx_view[drop_mask] = null_label_index # Inject null token
                            target_idx = target_idx_view

                x = torch.cat([cond, x_t], dim=1)

                with torch.amp.autocast("cuda", enabled=cfg["train"]["use_amp"] and device == "cuda"):
                    # Pass the target class to the MONAI backbone
                    pred_noise = model(x, t, class_labels=target_idx if use_class_embed else None)

                alpha_bar_f32 = diffusion.alpha_cumprod[t].view(-1, 1, 1, 1, 1).float()
                snr = alpha_bar_f32 / (1.0 - alpha_bar_f32 + 1e-8)

                pred_x0 = (
                    x_t.float() - torch.sqrt(1.0 - alpha_bar_f32) * pred_noise.float()
                ) / (torch.sqrt(alpha_bar_f32) + 1e-5)

                pred_x0 = torch.clamp(pred_x0, min=-1.0, max=1.0)

                loss, loss_parts = loss_fn(
                    pred_noise.float(),
                    noise.float(),
                    pred_x0,
                    target.float(),
                    snr=snr,
                    t=t,
                    timesteps=cfg["diffusion"]["timesteps"],
                    # --- NEW: Curriculum tracking ---
                    current_step=step,
                    max_steps=cfg["train"]["max_steps"]
                )

                # --- SAFETY CHECKS BEFORE BACKWARD ---
                if not torch.isfinite(loss) or not torch.isfinite(pred_noise).all() or not torch.isfinite(pred_x0).all():
                    print(f"NaN detected at step={step}. Skipping batch and zeroing gradients...")
                    # Dump the poisoned gradients
                    optimizer.zero_grad(set_to_none=True)
                    # Move immediately to the next dataloader batch
                    continue

                # --- BACKWARD PASS (Scale loss for accumulation) ---
                loss_scaled = loss / accum_steps
                scaler.scale(loss_scaled).backward()

                # --- OPTIMIZER STEP (Only every N steps) ---
                if (step + 1) % accum_steps == 0 or (step + 1) == cfg["train"]["max_steps"]:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Zero gradients AFTER stepping
                    optimizer.zero_grad(set_to_none=True)
                    
                    if ema is not None:
                        ema.update(model)
                    
                    if scheduler_lr is not None:
                        scheduler_lr.step()

                if device == "cuda":
                    torch.cuda.synchronize()
                
                step_time = step_timer.toc()
                vram_mb = get_vram_mb() if device == "cuda" else 0.0
                lr = optimizer.param_groups[0]["lr"]
                missing_str = key[0] if isinstance(key, (list, tuple)) else key

                should_log = (step % cfg["train"]["log_every"] == 0)
                do_light_val = (step % cfg["validation"]["light_every"] == 0 and step > 0)
                do_heavy_val = (step % cfg["validation"]["heavy_every"] == 0 and step > 0)
                should_save = (step % cfg["train"]["save_every"] == 0 and step > 0)

                record = None
                if should_log or do_light_val or do_heavy_val or should_save:
                    record = {
                        "step": step,
                        "train_loss": float(loss.item()),
                        "missing": str(missing_str),
                        "lr": float(lr),
                        "step_time_sec": float(step_time),
                        "vram_mb": float(vram_mb),
                        "run_time_sec": float(run_timer.elapsed()),
                    }
                    record.update(loss_parts)

                if should_log:
                    print(
                        f"step={step} "
                        f"loss={loss.item():.6f} "
                        f"noise_mse={loss_parts.get('noise_mse', 0):.6f} "
                        f"mae_loss={loss_parts.get('mae_loss', 0):.6f} "
                        f"ssim_loss={loss_parts.get('ssim_loss', 0):.6f} "
                        f"missing={missing_str} "
                        f"lr={lr:.6e} "
                        f"time={step_time:.3f}s "
                        f"vram={vram_mb:.1f}MB"
                    )

                if do_light_val or do_heavy_val:
                    val_model = ema.shadow if (ema is not None and cfg["ema"]["validate_with_ema"]) else model

                    val_batches = golden_batches if do_heavy_val else make_light_val_batch(val_loader)

                    val_stats = validate(
                        val_model,
                        val_batches,
                        diffusion,
                        loss_fn,
                        metric_bundle,
                        device,
                        cfg["train"]["use_amp"],
                        timesteps=cfg["diffusion"]["timesteps"],
                        compute_heavy=do_heavy_val,
                        current_step=step,
                        max_steps=cfg["train"]["max_steps"],
                        reverse_steps=cfg["inference"]["reverse_steps"],
                        raw_metric_bundle=raw_metric_bundle,
                    )
                    if record is None:
                        record = {
                            "step": step,
                            "train_loss": float(loss.item()),
                            "missing": str(missing_str),
                            "lr": float(lr),
                            "step_time_sec": float(step_time),
                            "vram_mb": float(vram_mb),
                            "run_time_sec": float(run_timer.elapsed()),
                        }
                        record.update(loss_parts)

                    record.update(val_stats)

                    # --- CONSOLE UI ---
                    prefix = "Val Heavy" if do_heavy_val else "Val Light"
                    print(f"\n{'='*12} [ {prefix} ] {'='*12}")
                    
                    t_loss = val_stats.get('val_total_loss', 0)
                    n_mse = val_stats.get('val_noise_mse', 0)
                    m_loss = val_stats.get('val_mae_loss', 0)
                    s_loss = val_stats.get('val_ssim_loss', 0)
                    
                    if do_heavy_val:
                        print(f"Total Loss: {t_loss:>8.6f}  |  Noise MSE: {n_mse:>8.6f} ||  Global PSNR: {val_stats.get('val_psnr', 0):>8.4f}")
                        print(f"MAE Loss:   {m_loss:>8.6f}  |  SSIM Loss: {s_loss:>8.6f} ||  Global SSIM: {val_stats.get('val_ssim', 0):>8.4f}")
                        if "val_raw_ssim" in val_stats:
                            print(f"Raw PSNR:    {val_stats.get('val_raw_psnr', 0):>8.4f}  |  Raw SSIM:   {val_stats.get('val_raw_ssim', 0):>8.4f}  |  Raw MAE: {val_stats.get('val_raw_mae', 0):>8.4f}")
                        print("-" * 65)
                        
                        for mod in ["t1ce", "flair"]:
                            if f"val_{mod}_ssim" in val_stats:
                                print(f"[{mod.upper():>5}] PSNR: {val_stats[f'val_{mod}_psnr']:>7.4f}  |  SSIM: {val_stats[f'val_{mod}_ssim']:>7.4f}  |  MAE: {val_stats[f'val_{mod}_mae']:>7.4f}")
                        print(f"{'='*50}\n")
                    else:
                        print(f"Total Loss: {t_loss:>8.6f}  |  Noise MSE: {n_mse:>8.6f}")
                        print(f"MAE Loss:   {m_loss:>8.6f}  |  SSIM Loss: {s_loss:>8.6f}")
                        print(f"{'='*46}\n")

                    # --- DUAL CHECKPOINTING ---
                    # Save the Math Checkpoint (Noise) - Updates on Light AND Heavy
                    if val_stats["val_noise_mse"] < best_val:
                        best_val = val_stats["val_noise_mse"]
                        save_checkpoint(
                            os.path.join(ckpt_dir, "best_noise.pt"),
                            model, optimizer, scheduler_lr, scaler,
                            step, best_val, cfg, ema=ema
                        )
                    
                    if do_heavy_val:
                        if "val_ssim" in val_stats and val_stats["val_ssim"] > best_ssim:
                            best_ssim = float(val_stats["val_ssim"])

                        if heavy_should_consider(val_stats, step, cfg["train"]["max_steps"], cfg):
                            key = heavy_rank_score(val_stats, cfg)
                            candidate_primary = key[0]

                            margin = float(cfg.get("checkpoint", {}).get("heavy_score_margin", 0.005))
                            should_add = False

                            if len(top_heavy) < top_k_heavy:
                                should_add = True
                            else:
                                current_floor = float(top_heavy[-1]["key"][0])
                                should_add = (float(key[0]) > current_floor + margin)

                            if should_add:
                                ckpt_path = os.path.join(ckpt_dir, f"top_heavy_step_{step:07d}.pt")
                                save_checkpoint(
                                    ckpt_path,
                                    model, optimizer, scheduler_lr, scaler,
                                    step, candidate_primary, cfg, ema=ema
                                )

                                top_heavy.append({
                                    "key": key,
                                    "score": float(key[0]),
                                    "step": int(step),
                                    "path": ckpt_path,
                                    "val_ssim": float(val_stats.get("val_ssim", 0.0)),
                                    "val_mae": float(val_stats.get("val_mae", 0.0)),
                                    "val_psnr": float(val_stats.get("val_psnr", 0.0)),
                                    "val_raw_ssim": float(val_stats.get("val_raw_ssim", val_stats.get("val_ssim", 0.0))),
                                    "val_raw_mae": float(val_stats.get("val_raw_mae", val_stats.get("val_mae", 0.0))),
                                    "val_raw_psnr": float(val_stats.get("val_raw_psnr", val_stats.get("val_psnr", 0.0))),
                                    "val_noise_mse": float(val_stats.get("val_noise_mse", 0.0)),
                                })

                                top_heavy.sort(key=lambda x: x["key"], reverse=True)

                                while len(top_heavy) > top_k_heavy:
                                    doomed = top_heavy.pop(-1)
                                    if os.path.exists(doomed["path"]):
                                        os.remove(doomed["path"])

                                save_json(os.path.join(ckpt_dir, "top_heavy.json"), top_heavy)

                                print("*** Updated top-k heavy sentinel checkpoints ***")
                                for rank, item in enumerate(top_heavy, start=1):
                                    print(
                                        f"  #{rank} step={item['step']} "
                                        f"score={item['score']:.4f} "
                                        f"raw_ssim={item['val_raw_ssim']:.4f} "
                                        f"raw_psnr={item['val_raw_psnr']:.4f} "
                                        f"raw_mae={item['val_raw_mae']:.4f} "
                                        f"noise={item['val_noise_mse']:.6f}"
                                    )
                                print()

                    model.train()

                    # VRAM Flush -> require on windows to prevent memory fragmentation from causing OOMs during validation
                    # should not be needed on Linux, and may hurt performance
                    # ------------------------------#
                    # if torch.cuda.is_available():
                    #     torch.cuda.empty_cache()
                    # ------------------------------#

                # Persist history only when something meaningful happened
                if record is not None:
                    history.append(record)

                # JSON writes now follow log cadence and important events
                if should_log or do_light_val or do_heavy_val or should_save:
                    io_executor.submit(save_json, os.path.join(log_dir, "history.json"), list(history))

                # Plot updates only on validation/save cadence
                if do_light_val or do_heavy_val or should_save:
                    io_executor.submit(save_history_plots, list(history), os.path.join(run_dir, "plots"))

                if should_save:
                    save_checkpoint(
                        os.path.join(ckpt_dir, "latest.pt"),
                        model, optimizer, scheduler_lr, scaler,
                        step, best_val, cfg, ema=ema
                    )

                step += 1
                if step >= cfg["train"]["max_steps"]:
                    break

    except KeyboardInterrupt:
        print("\nInterrupted. Saving checkpoint...")
        save_checkpoint(
            os.path.join(ckpt_dir, "interrupt.pt"),
            model, optimizer, scheduler_lr, scaler,
            step, best_val, cfg, ema=ema
        )
        print("Saved interrupt checkpoint.")

if __name__ == "__main__":
    main()