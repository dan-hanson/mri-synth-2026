import os
import json
import torch


def save_checkpoint(path, model, optimizer, scheduler, scaler, step, best_val=None, cfg=None, ema=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "ema_state": ema.state_dict() if ema is not None else None,
            "step": step,
            "best_val": best_val,
            "config": cfg,
        },
        path,
    )


def _adapt_state_dict(saved_state, target_state):
    """
    Reconcile a saved state_dict with the target model's expected key prefixes.

    torch.compile() wraps the model and prefixes every parameter key with
    '_orig_mod.'. A checkpoint saved from a compiled model has the prefix;
    one saved from an uncompiled model does not. If the current and saved
    runs disagree on whether the model is compiled, we need to add or strip
    the prefix so load_state_dict succeeds.
    """
    saved_keys = list(saved_state.keys())
    target_keys = list(target_state.keys())

    saved_has = any(k.startswith("_orig_mod.") for k in saved_keys)
    target_has = any(k.startswith("_orig_mod.") for k in target_keys)

    if saved_has == target_has:
        # Both compiled or both not — leave keys alone and let load_state_dict
        # surface any genuine architectural mismatch.
        return saved_state

    if saved_has and not target_has:
        # Strip the prefix
        return {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in saved_state.items()
        }

    # target_has and not saved_has — add the prefix
    return {f"_orig_mod.{k}": v for k, v in saved_state.items()}


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, ema=None, map_location="cpu"):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found or not a file: {path}")

    ckpt = torch.load(path, map_location=map_location)

    # Adapt model state for compile prefix mismatch
    model_state = _adapt_state_dict(ckpt["model_state"], model.state_dict())
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        print(f"[load_checkpoint] model: {len(missing)} missing, {len(unexpected)} unexpected keys")
        if missing:
            print(f"  first missing: {missing[:3]}")
        if unexpected:
            print(f"  first unexpected: {unexpected[:3]}")

    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])

    if ema is not None and ckpt.get("ema_state") is not None:
        # EMA shadow may also need prefix adaptation
        ema_state = _adapt_state_dict(ckpt["ema_state"], ema.shadow.state_dict())
        ema.load_state_dict(ema_state)

    return ckpt


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)