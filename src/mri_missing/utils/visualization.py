import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
    # Validation metrics
    # -------------------------
    val_steps = [h["step"] for h in history if "val_noise_mse" in h]
    val_noise = [h["val_noise_mse"] for h in history if "val_noise_mse" in h]

    heavy_steps = [h["step"] for h in history if "mae" in h]
    mae_vals = [h["mae"] for h in history if "mae" in h]
    psnr_vals = [h["psnr"] for h in history if "psnr" in h]
    ssim_vals = [h["ssim"] for h in history if "ssim" in h]

    if val_steps:
        plt.figure()
        plt.plot(val_steps, val_noise, label="val_noise_mse")

        if heavy_steps:
            plt.plot(heavy_steps, mae_vals, label="mae")
            plt.plot(heavy_steps, psnr_vals, label="psnr")
            plt.plot(heavy_steps, ssim_vals, label="ssim")

        plt.xlabel("step")
        plt.title("Validation Metrics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "val_metrics.png"))
        plt.close()

MOD_LABELS = ["T1", "T1ce", "T2", "FLAIR"]

def save_slice_panel(cond, target, pred, out_path, missing_key=None, title="sample"):
    """
    cond: [4, D, H, W]
    target: [D, H, W]
    pred: [D, H, W]
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    z = target.shape[2] // 2

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # top row: all 4 ordered condition slots
    for i in range(4):
        axes[0, i].imshow(cond[i, :, :, z], cmap="gray")
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

    # bottom row
    axes[1, 0].imshow(target[:, :, z], cmap="gray")
    axes[1, 0].set_title("Target")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pred[:, :, z], cmap="gray")
    axes[1, 1].set_title("Prediction")
    axes[1, 1].axis("off")

    diff = np.abs(target[:, :, z] - pred[:, :, z])
    axes[1, 2].imshow(diff, cmap="hot")
    axes[1, 2].set_title("Abs Diff")
    axes[1, 2].axis("off")

    # leave last panel for text / metadata
    axes[1, 3].axis("off")
    meta = f"Missing: {missing_key}" if missing_key is not None else ""
    axes[1, 3].text(0.05, 0.8, meta, fontsize=12)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()