import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def _norm_for_plot(vol):
    """Safely squeezes any array into exactly [0, 1] exclusively for matplotlib."""
    v_min, v_max = vol.min(), vol.max()
    if v_max - v_min < 1e-6:
        return np.zeros_like(vol)
    return (vol - v_min) / (v_max - v_min)

def save_history_plots(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # Training loss
    # -------------------------
    steps = [h["step"] for h in history if "train_loss" in h]
    train_loss = [h["train_loss"] for h in history if "train_loss" in h]

    if steps:
        plt.figure()
        plt.plot(steps, train_loss, label="train_loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "train_loss.png"))
        plt.close()

    # -------------------------
    # Validation Loss Components (Log Scale)
    # -------------------------
    val_steps = [h["step"] for h in history if "val_total_loss" in h]
    if val_steps:
        plt.figure(figsize=(10, 6))
        
        # Define all the possible loss parts we want to track
        loss_keys = ["val_total_loss", "val_noise_mse", "val_mae_loss", "val_ssim_loss", "val_texture_loss"]
        
        for key in loss_keys:
            # Check if this specific loss exists in the history at all
            if any(key in h for h in history):
                vals = [h.get(key, None) for h in history if "val_total_loss" in h]
                
                # Filter out None gaps if a metric was added mid-run
                valid_steps = [s for s, v in zip(val_steps, vals) if v is not None]
                valid_vals = [v for v in vals if v is not None]
                
                if valid_vals:
                    plt.plot(valid_steps, valid_vals, label=key)

        plt.xlabel("Step")
        plt.ylabel("Loss (Log Scale)")
        plt.yscale("log") # Prevents small losses from being squashed by total_loss
        plt.title("Validation Loss Components")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "val_loss_components.png"))
        plt.close()

    # -------------------------
    # Validation Structural Metrics
    # -------------------------
    heavy_steps = [h["step"] for h in history if "val_mae" in h]
    mae_vals = [h["val_mae"] for h in history if "val_mae" in h]
    psnr_vals = [h["val_psnr"] for h in history if "val_psnr" in h]
    ssim_vals = [h["val_ssim"] for h in history if "val_ssim" in h]

    if heavy_steps:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        
        axs[0].plot(heavy_steps, mae_vals, color="red")
        axs[0].set_title("Validation MAE (Lower is better)")
        axs[0].grid(True, alpha=0.3)
        
        axs[1].plot(heavy_steps, psnr_vals, color="green")
        axs[1].set_title("Validation PSNR (Higher is better)")
        axs[1].grid(True, alpha=0.3)
        
        axs[2].plot(heavy_steps, ssim_vals, color="blue")
        axs[2].set_title("Validation SSIM (Higher is better)")
        axs[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "val_metrics.png"))
        plt.close()

MOD_LABELS = ["T1", "T1ce", "T2", "FLAIR"]


def _to_display_01(x):
    # For strict-bound normalized MRI data in [-1, 1]
    x = (x + 1.0) / 2.0
    return np.clip(x, 0.0, 1.0)


def save_slice_panel(cond, target, pred, out_path, missing_key=None, title="sample"):
    """
    cond: [4 or 8, D, H, W] in normalized space
    target: [D, H, W] in normalized space
    pred: [D, H, W] in normalized space
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Use the last spatial axis as slice index, matching existing convention
    z = target.shape[2] // 2

    cond_vis = cond[:4].copy()
    target_vis = target.copy()
    pred_vis = pred.copy()

    # ---Dynamically scale all arrays for safe rendering ---
    cond_vis = _norm_for_plot(cond)
    target_vis = _norm_for_plot(target)
    pred_vis = _norm_for_plot(pred)

    diff = np.abs(target_vis[:, :, z] - pred_vis[:, :, z])
    diff_vmax = max(np.percentile(diff, 99), 1e-6)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor="white")

    for i in range(4):
        axes[0, i].imshow(
            cond_vis[i, :, :, z],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="none",
        )
        label = MOD_LABELS[i]
        if missing_key is not None:
            expected_missing_idx = {
                "t1": 0,
                "t1ce": 1,
                "t2": 2,
                "flair": 3,
            }[missing_key]
            if i == expected_missing_idx:
                label += " (missing slot)"
        axes[0, i].set_title(label)
        axes[0, i].axis("off")

    axes[1, 0].imshow(target_vis[:, :, z], cmap="gray", vmin=0.0, vmax=1.0, interpolation="none")
    axes[1, 0].set_title("Target")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pred_vis[:, :, z], cmap="gray", vmin=0.0, vmax=1.0, interpolation="none")
    axes[1, 1].set_title("Prediction")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(diff, cmap="hot", vmin=0.0, vmax=diff_vmax, interpolation="none")
    axes[1, 2].set_title("Abs Diff")
    axes[1, 2].axis("off")

    axes[1, 3].axis("off")
    meta = f"Missing: {missing_key}" if missing_key is not None else ""
    axes[1, 3].text(0.05, 0.8, meta, fontsize=12)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()

def save_augmentation_panel(dataset, out_path, num_samples=5):
    """Pulls the exact same case multiple times to visualize random augmentation variance."""
    fig, axes = plt.subplots(2, num_samples, figsize=(3 * num_samples, 6), facecolor="white")
    
    case_id = None
    for i in range(num_samples):
        # By calling dataset[0] repeatedly, we force the random_crop and augmentations to roll again
        cond, target, target_key, c_id = dataset[0] 
        if case_id is None:
            case_id = c_id
        
        c_np = cond.numpy()
        t_np = target.squeeze().numpy() # [X, Y, Z]
        
        z_mid = t_np.shape[-1] // 2
        
        # Find the first available conditional image channel to display
        cond_vis = np.zeros_like(t_np)
        for c_idx in range(4):
            if c_np[c_idx].max() > -0.99:
                cond_vis = c_np[c_idx]
                break

        c_plot = _norm_for_plot(cond_vis[:, :, z_mid])
        t_plot = _norm_for_plot(t_np[:, :, z_mid])
        
        # origin="lower" prevents the NIfTI arrays from rendering upside down on the plot
        axes[0, i].imshow(c_plot, cmap="gray", vmin=0.0, vmax=1.0, origin="lower")
        axes[0, i].set_title(f"Aug {i+1}: Cond")
        axes[0, i].axis("off")
        
        axes[1, i].imshow(t_plot, cmap="gray", vmin=0.0, vmax=1.0, origin="lower")
        axes[1, i].set_title(f"Target ({target_key})")
        axes[1, i].axis("off")
        
    plt.suptitle(f"Augmentation Variance: {case_id}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()