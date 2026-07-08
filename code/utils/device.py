"""Torch device helper: CUDA when available, else CPU."""
from __future__ import annotations

import torch


def get_torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_label() -> str:
    dev = get_torch_device()
    if dev.type == "cuda":
        name = torch.cuda.get_device_name(dev)
        mem_gb = torch.cuda.get_device_properties(dev).total_memory / (1024**3)
        return f"cuda ({name}, {mem_gb:.0f}GB)"
    return "cpu"


def dataloader_kwargs(batch_size: int = 32) -> dict:
    """DataLoader options: pin_memory on CUDA for faster host→device transfer."""
    use_cuda = torch.cuda.is_available()
    return {
        "batch_size": batch_size * (2 if use_cuda else 1),
        "shuffle": True,
        "pin_memory": use_cuda,
    }
