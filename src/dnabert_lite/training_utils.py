"""Small training helpers shared by pretraining and fine-tuning."""

from __future__ import annotations

import sys
import time

import torch
from torch import nn


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module when DataParallel is used."""

    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def parallelize_model(
    model: nn.Module,
    device: torch.device,
    use_multi_gpu: bool = True,
) -> tuple[nn.Module, int]:
    """Move a model to device and wrap it with DataParallel when available."""

    model = model.to(device)
    if device.type == "cuda" and use_multi_gpu and torch.cuda.device_count() > 1:
        gpu_count = torch.cuda.device_count()
        print(f"Using DataParallel on {gpu_count} GPUs.", file=sys.stderr)
        return nn.DataParallel(model), gpu_count
    return model, 1


def scalar_loss(loss: torch.Tensor) -> torch.Tensor:
    """Collapse DataParallel per-device losses into one scalar loss."""

    return loss.mean()


class ProgressBar:
    """A lightweight stderr progress bar with no extra dependency."""

    def __init__(self, total: int, desc: str, enabled: bool = True) -> None:
        self.total = max(1, total)
        self.desc = desc
        self.enabled = enabled
        self.current = 0
        self.start_time = time.time()

    def update(self, step: int = 1, **metrics: float) -> None:
        if not self.enabled:
            return

        self.current = min(self.total, self.current + step)
        fraction = self.current / self.total
        width = 28
        filled = int(width * fraction)
        bar = "=" * filled + "." * (width - filled)
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0.0
        metric_text = ""
        if metrics:
            metric_text = " - " + " ".join(
                f"{name}={value:.4f}" for name, value in metrics.items()
            )
        line = (
            f"\r{self.desc} [{bar}] {self.current}/{self.total} "
            f"{fraction * 100:5.1f}% {rate:5.2f} step/s{metric_text}"
        )
        sys.stderr.write(line)
        sys.stderr.flush()

    def close(self) -> None:
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()
