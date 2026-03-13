"""Device helpers for CPU, CUDA, and optional TPU/XLA execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.parallel_loader import MpDeviceLoader
except ImportError:
    xm = None
    MpDeviceLoader = None


def xla_available() -> bool:
    """Return True when torch-xla is installed in the current environment."""
    return xm is not None


def _normalize_device(device: str | torch.device) -> str | torch.device:
    """Normalize TPU aliases to the torch XLA device name."""
    if isinstance(device, str) and device.lower() == "tpu":
        return "xla"
    return device


def is_xla_device(device: str | torch.device | None) -> bool:
    """Return True for TPU/XLA devices."""
    if device is None:
        return False
    return torch.device(_normalize_device(device)).type == "xla"


def _infer_model_device(model: nn.Module | None) -> torch.device | None:
    """Infer the first concrete device used by a model."""
    if model is None:
        return None

    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.device.type != "meta":
            return tensor.device

    return None


def resolve_device(
    device: str | torch.device | None = None,
    model: nn.Module | None = None,
) -> torch.device:
    """
    Resolve a runtime device.

    `auto` preserves an already-placed accelerator model, otherwise prefers
    CUDA, then XLA, then CPU.
    """
    if device not in (None, "auto"):
        requested = torch.device(_normalize_device(device))
        if requested.type == "xla":
            if not xla_available():
                raise RuntimeError(
                    "Requested an XLA device but torch-xla is not installed."
                )
            return xm.xla_device()
        return requested

    model_device = _infer_model_device(model)
    if model_device is not None and model_device.type in {"cuda", "xla"}:
        return model_device

    if torch.cuda.is_available():
        return torch.device("cuda")

    if xla_available():
        return xm.xla_device()

    return model_device or torch.device("cpu")


def move_model_to_device(model: nn.Module, device: str | torch.device) -> nn.Module:
    """Move a model unless it is already placed or quantization forbids `.to()`."""
    target_device = torch.device(_normalize_device(device))
    current_device = _infer_model_device(model)

    if current_device == target_device:
        return model

    try:
        return model.to(target_device)
    except (RuntimeError, ValueError) as exc:
        message = str(exc).lower()
        if "4-bit" in message or "8-bit" in message or "bitsandbytes" in message:
            return model
        raise


def move_batch_to_device(batch: Any, device: str | torch.device) -> Any:
    """Recursively move tensors in a batch to the target device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(_normalize_device(device))
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    return batch


def wrap_dataloader(dataloader: Any, device: str | torch.device) -> Any:
    """Wrap dataloaders with MpDeviceLoader when running on XLA."""
    if dataloader is None or not is_xla_device(device) or MpDeviceLoader is None:
        return dataloader
    if dataloader.__class__.__name__ == "MpDeviceLoader":
        return dataloader
    return MpDeviceLoader(dataloader, _normalize_device(device))


def optimizer_step(optimizer: torch.optim.Optimizer, device: str | torch.device) -> None:
    """Execute an optimizer step across CPU/CUDA/XLA runtimes."""
    if is_xla_device(device) and xm is not None:
        xm.optimizer_step(optimizer, barrier=False)
        return
    optimizer.step()


def mark_step(device: str | torch.device) -> None:
    """Flush queued XLA ops when needed."""
    if is_xla_device(device) and xm is not None:
        xm.mark_step()


def is_primary_process(device: str | torch.device) -> bool:
    """Return True for the process that should write logs and checkpoints."""
    if is_xla_device(device) and xm is not None:
        return xm.is_master_ordinal()
    return True


def save_checkpoint(
    checkpoint: dict[str, Any],
    path: str | Path,
    device: str | torch.device,
) -> None:
    """Save checkpoints in an XLA-safe way."""
    if is_xla_device(device) and xm is not None:
        xm.save(checkpoint, str(path))
        return
    torch.save(checkpoint, path)
