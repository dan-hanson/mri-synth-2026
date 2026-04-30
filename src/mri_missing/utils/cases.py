import os
import random

MOD_KEYS = ["t1", "t1ce", "t2", "flair"]
MOD_TO_IDX = {k: i for i, k in enumerate(MOD_KEYS)}  # {"t1": 0, "t1ce": 1, "t2": 2, "flair": 3}
NUM_MODALITIES = len(MOD_KEYS)

FILE_MAP = {
    "t1": "t1n",
    "t1ce": "t1c",
    "t2": "t2w",
    "flair": "t2f",
}


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