"""Explicit experimental CPU BF16 inference; no process-wide tuning."""
from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar
import warnings
from ._cpu_selftest import CPUFastUnavailable,verify,version_problem
from .client import BaseDecision
from .packing import Packed
from .types import Request,Result,InputError

if TYPE_CHECKING:
    from torch import Tensor

_T = TypeVar('_T', bound='CPUFastDecision')

@dataclass(frozen=True)
class CPUResult(Result):
    """A :class:`~basedecision.Result` produced by the experimental ``cpu_fast`` backend.

    Attributes:
        backend: Always ``'cpu_fast_experimental'``.
        inference_precision: How the forward pass ran: BF16 autocast over FP32 weights.
    """

    backend: str = 'cpu_fast_experimental'
    inference_precision: str = 'bf16_autocast_fp32_weights'

class CPUFastDecision(BaseDecision):
    """Experimental CPU backend: BF16 autocast with a tiled attention path for long inputs.

    Opt in with ``load(path, backend='cpu_fast')``. At construction it checks the installed Torch
    and Transformers versions and runs a short known-answer self-test, raising
    :class:`~basedecision.CPUFastUnavailable` (nothing is changed) if the fast path cannot be
    verified; use the normal CPU path then. Probabilities are uncalibrated, and it was measured
    slower than the default on Apple-silicon laptops; see ``docs/CPU.md``.
    """

    @classmethod
    def from_pretrained(cls: type[_T],path: str | Path,*,device: str='cpu',precision: str='bf16',max_batch_size: int=1,max_batch_tokens: int=8192) -> _T:
        """Load a checkpoint for the fast CPU path; ``device`` must be ``'cpu'`` and ``precision`` ``'bf16'``."""
        return cls(path,device=device,precision=precision,max_batch_size=max_batch_size,max_batch_tokens=max_batch_tokens)

    def __init__(self,path: str | Path,*,device: str='cpu',precision: str='bf16',max_batch_size: int=1,max_batch_tokens: int=8192) -> None:
        """Load the checkpoint, verify the fast path on this installation and install it.

        Raises:
            InputError: ``device``/``precision`` are not ``'cpu'``/``'bf16'``.
            CPUFastUnavailable: The Torch/Transformers versions are unsupported or the self-test
                failed (an ``InputError``; the message says to use ``load(path)`` instead).
        """
        if device!='cpu' or precision!='bf16':raise InputError('cpu_fast requires device="cpu", precision="bf16"')
        import torch,transformers
        problem=version_problem(torch.__version__,transformers.__version__)
        if problem:raise CPUFastUnavailable(problem)  # instantly, before any weights are loaded
        super().__init__(path,device='cpu',precision='fp32',max_batch_size=max_batch_size,max_batch_tokens=max_batch_tokens)
        # Prove the fast path works on THIS installation (known-answer test on a tiny model with this
        # checkpoint's attention geometry) before it is used. Raises CPUFastUnavailable otherwise.
        self._verification=verify(self._model.encoder.config)
        from ._cpu_attention import install,install_head
        self._cpu_counters=install(self._model);install_head(self._model)
        self.inference_precision='bf16_autocast_fp32_weights'
        warnings.warn('Experimental CPU fast mode: probabilities are uncalibrated; speed depends on hardware BF16 support. No thread or affinity settings were changed.',RuntimeWarning,stacklevel=2)

    def _forward(self,items: Sequence[Packed]) -> list[Tensor]:
        with self._torch.autocast('cpu',dtype=self._torch.bfloat16):
            return super()._forward(items)

    def _result(self,r: Request,p: Packed,z: Tensor) -> CPUResult:
        return CPUResult(**asdict(super()._result(r,p,z)))

    def backend_info(self) -> dict[str, Any]:
        """Describe the backend: precision, thread count, attention-path counters and the self-test."""
        with self._lock:
            return dict(backend='cpu_fast_experimental',device='cpu',weight_precision=self.precision,
                inference_precision=self.inference_precision,calibration_supported=False,
                attention_calls=dict(self._cpu_counters),threads=self._torch.get_num_threads(),
                self_test=dict(torch=self._verification.torch,transformers=self._verification.transformers,
                    seconds=round(self._verification.seconds,3),relative_difference=float(f'{self._verification.relative_difference:.3g}')),
                fallback='Original backbone attention for padded/batched/global attention; CPU BF16 head retained')
