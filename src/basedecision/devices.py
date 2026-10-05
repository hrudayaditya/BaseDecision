"""Choose a device and precision that work on the machine at hand.

``BaseDecision.from_pretrained`` deliberately defaults to the configuration the model was
measured on (CUDA, BF16) and fails loudly elsewhere. That is the right default for an explicit,
low-level constructor and the wrong one for a first run on a laptop. :func:`resolve_device`
supplies the portable choice and is used by :func:`basedecision.load` and
:func:`basedecision.load_from_hub`:

=================  ===========================================================
CUDA GPU present   ``("cuda", "bf16")`` if the GPU has native BF16, else ``("cuda", "fp32")``
no CUDA GPU        ``("cpu", "fp32")`` (also Apple-silicon Macs: Metal/MPS is not supported)
=================  ===========================================================

Explicit arguments always win and are returned unchanged, so an invalid combination such as
``("cpu", "bf16")`` is still rejected, with the model constructor's own clear message.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

__all__ = ["resolve_device"]


def _device_kind(device: object) -> str:
    """Return the device type ("cuda", "cpu", ...) of a string or ``torch.device``-like."""
    if isinstance(device, str):
        return device.split(":", 1)[0]
    return str(getattr(device, "type", ""))


def _import_torch() -> ModuleType:
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "Install basedecision[runtime] (Torch and Transformers) to run a local model"
        ) from exc
    return torch


def _native_bf16(torch: ModuleType) -> bool:
    """Return whether the current CUDA device computes BF16 natively (not by emulation)."""
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:  # older Torch without the keyword
        return bool(torch.cuda.is_bf16_supported())
    except Exception:  # an unusual CUDA setup: be conservative
        return False


def resolve_device(device: object = None, precision: str | None = None) -> tuple[Any, str]:
    """Return a ``(device, precision)`` pair for local inference.

    Args:
        device: ``None`` or ``"auto"`` selects CUDA when a GPU is available and CPU otherwise.
            Anything else (``"cpu"``, ``"cuda"``, ``"cuda:1"``, a ``torch.device``) is kept.
        precision: ``None`` selects ``"bf16"`` on a CUDA device with native BF16 support and
            ``"fp32"`` everywhere else (the only option on CPU). ``"bf16"``/``"fp32"`` are kept.

    Returns:
        The device (as given, or ``"cuda"``/``"cpu"``) and the precision.

    Raises:
        ImportError: Torch is needed to look for a GPU but is not installed.
    """
    torch: ModuleType | None = None
    if device is None or device == "auto":
        torch = _import_torch()
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if precision is None:
        if _device_kind(device) == "cuda":
            torch = torch or _import_torch()
            precision = "bf16" if _native_bf16(torch) else "fp32"
        else:
            precision = "fp32"
    return device, precision
