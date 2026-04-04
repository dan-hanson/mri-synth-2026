import time
import torch


class StepTimer:
    def __init__(self):
        self.start = None

    def tic(self):
        self.start = time.time()

    def toc(self):
        return time.time() - self.start if self.start is not None else 0.0


class RunTimer:
    def __init__(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start


def get_vram_mb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def reset_vram_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)