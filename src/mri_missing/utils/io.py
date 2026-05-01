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

    Handles three cases:
      1. Saved keys match target keys exactly — return saved unchanged.
      2. Saved has '_orig_mod.' prefix but target does not — strip it.
      3. Target has '_orig_mod.' prefix but saved does not — add it.

    Case 3 is what happens when a checkpoint was saved from an uncompiled
    or transparently-unwrapped model and we're now loading into a
    torch.compile()-wrapped model.
    """
    saved_keys = set(saved_state.keys())
    target_keys = set(target_state.keys())

    # Already matches — nothing to do
    if saved_keys == target_keys or len(saved_keys & target_keys) > 0:
        # If at least some keys overlap, no rewriting needed; PyTorch will
        # surface any other mismatches itself.
        if saved_keys.issubset(target_keys) or target_keys.issubset(saved_keys):
            return saved_state

    saved_has_prefix = any(k.startswith("_orig_mod.") for k in saved_keys)
    target_has_prefix = any(k.startswith("_orig_mod.") for k in target_keys)

    # Case 2: saved has prefix, target does not — strip
    if saved_has_prefix and not target_has_prefix:
        return {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v
                for k, v in saved_state.items()}

    # Case 3: target has prefix, saved does not — add
    if target_has_prefix and not saved_has_prefix:
        return {f"_orig_mod.{k}": v for k, v in saved_state.items()}

    # Both have prefix or neither — leave as-is and let PyTorch report mismatches
    return saved_state


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, ema=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)

    # Adapt model state for the target model's prefix convention
    model_state = _adapt_state_dict(ckpt["model_state"], model.state_dict())
    model.load_state_dict(model_state)

    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])

    if ema is not None and ckpt.get("ema_state") is not None:
        # Adapt EMA state for the EMA model's prefix convention
        ema_state = _adapt_state_dict(ckpt["ema_state"], ema.shadow.state_dict())
        ema.load_state_dict(ema_state)

    return ckpt


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)