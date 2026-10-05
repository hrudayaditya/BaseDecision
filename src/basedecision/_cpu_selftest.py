"""Version range and known-answer self-test for the experimental ``cpu_fast`` backend.

The fast path replaces some of ModernBERT's attention code with a tiled version derived from
Transformers 4.57.6. Instead of demanding that exact pair of versions, which would force people to
install particular torch/transformers releases and risk conflicts with everything else in their
environment, ``cpu_fast`` accepts the range the code is known to work in and then **proves it works
on the installed versions** before it is used:

1. A version range (torch >= 2.6, transformers >= 4.48 and < 5) rules out installs that are known
   not to work, instantly and before any weights are loaded. Transformers 5 removed the internals
   the adapter patches.
2. A short self-test builds a tiny random ModernBERT with the *real* model's attention geometry
   (window size, global-layer pattern, head size) and runs it with the original attention and with
   the fast path installed, on inputs that use the tiled path and on a padded batch that must use
   the reference. In FP32 a correct adapter agrees with the reference to about 1e-7, so the strict
   check (:data:`FP32_TOLERANCE`) detects even a 1% error; a looser BF16 pass confirms that the
   dtype the backend actually runs in works on this install. It takes about 0.3 s once the process
   is warm, and the result is cached.

If either step fails, ``cpu_fast`` refuses with :class:`CPUFastUnavailable` and a message saying
what to do; it never falls back silently and never returns results it has not verified. The
self-test changes no global state (no registry, no ``torch.backends`` flag, no seed outside a
restored RNG scope), and the patching it tests is local to one model instance.
"""

from __future__ import annotations

import importlib
import re
import threading
import time
import warnings
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Final, Protocol

from .types import InputError

__all__ = [
    "CPUFastUnavailable",
    "Verification",
    "verify",
    "version_problem",
]

TORCH_MINIMUM: Final = (2, 6)
TRANSFORMERS_MINIMUM: Final = (4, 48)
TRANSFORMERS_BELOW: Final = (5, 0)

FP32_TOLERANCE: Final = 1e-5
"""Largest allowed *relative* difference (to the largest reference magnitude) between the original
and the fast attention in FP32. A correct adapter measures ~1e-7 on torch 2.6 to 2.14; a 1% error in
one weight matrix measures ~1e-4."""
BF16_HIDDEN_TOLERANCE: Final = 5e-3
BF16_LOGIT_TOLERANCE: Final = 0.1
"""Looser limits for the BF16 pass, which only has to show that the dtype the backend runs in works
(BF16 rounding noise dominates the final logits of a tiny random model)."""

_LAYERS: Final = 6  # with the default pattern: global and local layers both present
_SELF_TEST_LENGTHS: Final = (300, 520)  # single inputs spanning 2 and 3 chunks of 256: tiled path
_PADDED_BATCH: Final = (2, 300)  # padding: must use the reference path
_FALLBACK_HINT: Final = (
    "The default CPU backend works with your installation: use load(path) "
    "(or load(path, device='cpu', precision='fp32'))."
)


class _HasHiddenState(Protocol):
    last_hidden_state: Any


class _EncoderConfig(Protocol):
    """The attention geometry of a ``ModernBertConfig`` that the self-test reproduces."""

    hidden_size: int
    num_attention_heads: int
    local_attention: int
    global_attn_every_n_layers: int


class CPUFastUnavailable(InputError):  # noqa: N818 - reads better than "...Error" next to InputError
    """``cpu_fast`` cannot be used with the installed torch/transformers."""


@dataclass(frozen=True)
class Verification:
    """Outcome of a successful self-test."""

    torch: str
    transformers: str
    seconds: float
    relative_difference: float
    """Largest FP32 difference to the reference, relative to the largest reference magnitude."""


def _major_minor(version: str) -> tuple[int, int] | None:
    match = re.match(r"\s*(\d+)\.(\d+)", version)
    return (int(match.group(1)), int(match.group(2))) if match else None


def version_problem(torch_version: str, transformers_version: str) -> str | None:
    """Return why these versions are unsupported, or ``None`` if they are inside the range."""
    torch_parsed = _major_minor(torch_version)
    transformers_parsed = _major_minor(transformers_version)
    if torch_parsed is None or torch_parsed < TORCH_MINIMUM or torch_parsed[0] >= 3:
        return f"cpu_fast needs torch 2.6 or newer (found {torch_version!r}). {_FALLBACK_HINT}"
    if (
        transformers_parsed is None
        or transformers_parsed < TRANSFORMERS_MINIMUM
        or transformers_parsed >= TRANSFORMERS_BELOW
    ):
        return (
            "cpu_fast supports transformers 4.48 to 4.57 "
            f"(found {transformers_version!r}; transformers 5 removed the internals it patches). "
            f"{_FALLBACK_HINT}"
        )
    return None


_LOCK = threading.Lock()
_CACHE: dict[tuple[Any, ...], Verification] = {}


def verify(config: _EncoderConfig) -> Verification:
    """Check that ``cpu_fast`` works on this installation for a model with this encoder config.

    Args:
        config: The loaded model's ``ModernBertConfig`` (its attention geometry is reproduced
            in a tiny model).

    Returns:
        A :class:`Verification` (cached per process for the same versions and geometry).

    Raises:
        CPUFastUnavailable: The versions are outside the supported range, the adapter fails on
            this installation, or its outputs differ from the reference.
    """
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    problem = version_problem(torch.__version__, transformers.__version__)
    if problem is not None:
        raise CPUFastUnavailable(problem)
    key = (
        torch.__version__,
        transformers.__version__,
        int(config.hidden_size) // int(config.num_attention_heads),
        int(config.local_attention),
        int(config.global_attn_every_n_layers),
    )
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is None:
            cached = _CACHE[key] = _run_self_test(config, torch, transformers)
        return cached


def _run_self_test(
    config: _EncoderConfig, torch: ModuleType, transformers: ModuleType
) -> Verification:
    started = time.perf_counter()
    reason: str | None = None
    difference = 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # e.g. "CPU does not support bf16" on old hardware
            difference = _compare(config, torch)
    except CPUFastUnavailable:
        raise
    except Exception as error:  # any failure means this installation is not supported
        reason = f"{type(error).__name__}: {str(error)[:120]}"
    if reason is not None:
        raise CPUFastUnavailable(
            f"cpu_fast failed its self-test on torch {torch.__version__} / transformers "
            f"{transformers.__version__} ({reason}). {_FALLBACK_HINT}"
        )
    return Verification(
        torch=torch.__version__,
        transformers=transformers.__version__,
        seconds=time.perf_counter() - started,
        relative_difference=difference,
    )


def _relative_difference(reference: list[Any], other: list[Any]) -> float:
    scale = max(1e-6, *(float(x.abs().max()) for x in reference))
    return max(float((a - b).abs().max()) for a, b in zip(reference, other, strict=True)) / scale


def _compare(config: _EncoderConfig, torch: ModuleType) -> float:
    """Run a tiny model with the original and the fast attention; return the FP32 difference."""
    transformers = importlib.import_module("transformers")
    modern_bert_config, modern_bert_model = (
        transformers.ModernBertConfig,
        transformers.ModernBertModel,
    )

    from ._cpu_attention import install, install_head
    from ._model import DecisionModel

    head_size = int(config.hidden_size) // int(config.num_attention_heads)
    heads = 2
    options: dict[str, Any] = {}
    for name in ("global_rope_theta", "local_rope_theta"):
        if hasattr(config, name):
            options[name] = getattr(config, name)
    small = modern_bert_config(
        vocab_size=100,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=head_size * heads,
        intermediate_size=head_size * heads * 2,
        num_hidden_layers=_LAYERS,
        num_attention_heads=heads,
        max_position_embeddings=2048,
        local_attention=int(config.local_attention),
        global_attn_every_n_layers=int(config.global_attn_every_n_layers),
        reference_compile=False,
        **options,
    )
    small._attn_implementation = "sdpa"
    with torch.random.fork_rng(devices=[]):  # restores the caller's RNG state afterwards
        torch.manual_seed(92)
        model = DecisionModel(modern_bert_model(small), head_layers=2).eval()
    model.requires_grad_(False)

    generator = torch.Generator().manual_seed(5)
    batches = []
    for size, length in [(1, _SELF_TEST_LENGTHS[0]), (1, _SELF_TEST_LENGTHS[1]), _PADDED_BATCH]:
        mask = torch.ones(size, length, dtype=torch.long)
        if size == 2:
            mask[1, -50:] = 0
        batches.append(
            {
                "input_ids": torch.randint(0, 100, (size, length), generator=generator),
                "attention_mask": mask,
                "marker_pos": torch.tensor([[1, 3]] * size),
                "marker_mask": torch.ones(size, 2, dtype=torch.bool),
                "qtype": torch.zeros(size, dtype=torch.long),
            }
        )

    def run(bf16: bool) -> tuple[list[Any], list[Any]]:
        """Run every batch; return the encoder hidden states and the final logits."""
        captured: list[Any] = []

        def capture(_module: object, _args: object, output: _HasHiddenState) -> None:
            captured.append(output.last_hidden_state.float().clone())

        hook = model.encoder.register_forward_hook(capture)
        try:
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                logits = [model(**batch)[0].float().clone() for batch in batches]
        finally:
            hook.remove()
        return captured, logits

    reference = {mode: run(mode) for mode in (False, True)}
    counters = install(model)  # type: ignore[no-untyped-call]  # instance-local: this model only
    install_head(model)  # type: ignore[no-untyped-call]
    fast = {mode: run(mode) for mode in (False, True)}

    for mode in (False, True):
        if not all(bool(torch.isfinite(x).all()) for part in fast[mode] for x in part):
            raise CPUFastUnavailable(f"non-finite outputs. {_FALLBACK_HINT}")
    if counters["tiled_calls"] == 0 or counters["reference_calls"] == 0:
        raise CPUFastUnavailable(
            f"the fast attention path was not exercised as expected {dict(counters)}. "
            f"{_FALLBACK_HINT}"
        )
    strict = max(_relative_difference(reference[False][part], fast[False][part]) for part in (0, 1))
    if strict > FP32_TOLERANCE:
        raise CPUFastUnavailable(
            f"its outputs differ from the reference by {strict:.1e} relative in FP32 "
            f"(limit {FP32_TOLERANCE:.0e}). {_FALLBACK_HINT}"
        )
    hidden = _relative_difference(reference[True][0], fast[True][0])
    logits = _relative_difference(reference[True][1], fast[True][1])
    if hidden > BF16_HIDDEN_TOLERANCE or logits > BF16_LOGIT_TOLERANCE:
        raise CPUFastUnavailable(
            f"in BF16 its outputs differ from the reference by {hidden:.1e} (hidden) / "
            f"{logits:.1e} (logits). {_FALLBACK_HINT}"
        )
    return strict
