import os
import random
import hashlib

MOD_KEYS = ["t1", "t1ce", "t2", "flair"]
MOD_TO_IDX = {k: i for i, k in enumerate(MOD_KEYS)}  # {"t1": 0, "t1ce": 1, "t2": 2, "flair": 3}
NUM_MODALITIES = len(MOD_KEYS)

FILE_MAP = {
    "t1": "t1n",
    "t1ce": "t1c",
    "t2": "t2w",
    "flair": "t2f",
}


def case_split_label(case_id: str, seed: int = 42, val_fraction: float = 0.12):
    """
    Deterministic split: returns "internal_val" or "internal_train" for a case ID.

    Uses a stable hash so the same case_id always lands in the same split,
    regardless of file order, machine, or run. Hash is salted with `seed` so
    you can change splits later by changing the seed.

    With ~1251 cases and val_fraction=0.12 you get about 150 internal-val cases.
    """
    h = hashlib.md5(f"{seed}:{case_id}".encode()).hexdigest()
    # Map first 8 hex chars to a [0, 1) float
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    return "internal_val" if bucket < val_fraction else "internal_train"


def split_train_cases(case_ids, seed: int = 42, val_fraction: float = 0.12):
    """
    Given a list of training case IDs, return (train_ids, val_ids) sets.
    Deterministic for a given seed.
    """
    train_ids = set()
    val_ids = set()
    for cid in case_ids:
        if case_split_label(cid, seed=seed, val_fraction=val_fraction) == "internal_val":
            val_ids.add(cid)
        else:
            train_ids.add(cid)
    return train_ids, val_ids


def list_case_dirs(root_dir):
    return [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]


def resolve_case_dirs(root_dir, case_ids=None, strict=True, random_case_count=0, random_case_seed=42):
    all_cases = {os.path.basename(p): p for p in list_case_dirs(root_dir)}

    if case_ids:
        resolved = []
        missing = []

        for cid in case_ids:
            if cid not in all_cases:
                missing.append(cid)
                continue
            resolved.append(all_cases[cid])

        if missing:
            available = list(all_cases.keys())[:20]
            msg = (
                f"Missing case ids: {missing}\n"
                f"First available case ids: {available}"
            )
            if strict:
                raise FileNotFoundError(msg)
            else:
                print("[warn]", msg)

        return resolved

    all_case_paths = sorted(all_cases.values())

    if random_case_count and random_case_count > 0:
        rng = random.Random(random_case_seed)
        if random_case_count >= len(all_case_paths):
            return all_case_paths
        return rng.sample(all_case_paths, random_case_count)

    return all_case_paths