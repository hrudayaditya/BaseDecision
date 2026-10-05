"""Small Python convenience API layered over the existing inference contract."""
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, overload
from .calibration import CalibratedDecision, CalibrationError, calibration_profiles
from .client import BaseDecision
from .cpu import CPUFastDecision
from .devices import resolve_device
from .types import Request, Option, InputError, options_from

_R_co = TypeVar('_R_co', covariant=True)
_M = TypeVar('_M')

class _BatchBackend(Protocol[_R_co]):
    """What :func:`decide` needs from a backend: ordered batch prediction."""

    def predict_batch(self, requests: list[Request]) -> Sequence[_R_co]:
        """Answer the requests in input order."""

def _hardware(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Fill in device/precision from resolve_device() unless the caller gave them explicitly."""
    device,precision=resolve_device(kwargs.pop('device',None),kwargs.pop('precision',None))
    return dict(kwargs,device=device,precision=precision)

def _check_options(backend: str | None,calibration_profile: str | None) -> None:
    """Validate the backend and calibration choice before any model is loaded or downloaded."""
    if backend not in (None,'default','cpu_fast'):raise InputError('backend must be default or cpu_fast')
    if backend=='cpu_fast' and calibration_profile is not None:
        raise CalibrationError('CPU calibration transfer has not been validated; use raw output')
    if calibration_profile is not None:
        info=calibration_profiles().get(calibration_profile)
        if info is None or not info['available']:raise CalibrationError('Unknown or unavailable calibration profile')

_LENGTH_NOTICE='sequence-length-is-longer-than-the-specified-maximum'

def _quiet(model: _M) -> _M:
    """Mark Transformers' "longer than the model maximum" notice as already shown.

    This package packs the text itself and refuses an over-long request with ``ContextLengthError``,
    so that notice ("... will result in indexing errors") is wrong as well as alarming. Setting the
    flag is Transformers' own warn-once mechanism; a model without a tokenizer is left alone.
    """
    notices=getattr(getattr(model,'_tokenizer',None),'deprecation_warnings',None)
    if isinstance(notices,dict):notices[_LENGTH_NOTICE]=True
    return model

def _calibrated(model: BaseDecision,calibration_profile: str | None) -> BaseDecision | CalibratedDecision:
    if calibration_profile is None:return model
    return CalibratedDecision(model,profile=calibration_profile)

@overload
def load(path: str | Path, *, calibration_profile: None = None, backend: Literal['default'] | None = None, **kwargs: Any) -> BaseDecision: ...
@overload
def load(path: str | Path, *, calibration_profile: None = None, backend: Literal['cpu_fast'], **kwargs: Any) -> CPUFastDecision: ...
@overload
def load(path: str | Path, *, calibration_profile: str, backend: Literal['default'] | None = None, **kwargs: Any) -> CalibratedDecision: ...
def load(path: str | Path, *, calibration_profile: str | None = None, backend: str | None = None, **kwargs: Any) -> BaseDecision | CalibratedDecision:
    """Load an explicitly named local checkpoint. Calibration is opt-in.

    Unless given, ``device`` and ``precision`` are chosen for this machine (see
    :func:`resolve_device`): CUDA/BF16 when a GPU with native BF16 is present, otherwise
    CPU/FP32. Pass them explicitly to override, e.g. ``load(path, device='cpu', precision='fp32')``.

    Args:
        path: Directory of the exported checkpoint.
        calibration_profile: Return a :class:`~basedecision.CalibratedDecision` for this reviewed
            profile instead of the raw model (CUDA/BF16 only; see ``docs/CALIBRATION.md``).
        backend: ``None``/``'default'`` for the standard backend, or ``'cpu_fast'`` for the
            experimental CPU backend (see ``docs/CPU.md``).
        **kwargs: ``device``, ``precision``, ``max_batch_size``, ``max_batch_tokens``.

    Returns:
        The model: a :class:`~basedecision.BaseDecision`, a ``CPUFastDecision`` for
        ``backend='cpu_fast'``, or a ``CalibratedDecision`` when a profile is given.

    Raises:
        ImportError: Torch/Transformers are not installed (``basedecision[runtime]``).
        FileNotFoundError: The checkpoint is incomplete.
        InputError: An unsupported backend, device, precision or batch setting (an explicit
            choice is never silently changed). ``CPUFastUnavailable`` and ``CalibrationError`` are
            subclasses.
    """
    _check_options(backend,calibration_profile)
    if backend=='cpu_fast':
        kwargs.setdefault('device','cpu');kwargs.setdefault('precision','bf16')
        return _quiet(CPUFastDecision.from_pretrained(path,**kwargs))
    return _calibrated(_quiet(BaseDecision.from_pretrained(path,**_hardware(kwargs))),calibration_profile)

@overload
def load_from_hub(repo_id: str, *, revision: str | None = None, calibration_profile: None = None, backend: Literal['default'] | None = None, **kwargs: Any) -> BaseDecision: ...
@overload
def load_from_hub(repo_id: str, *, revision: str | None = None, calibration_profile: None = None, backend: Literal['cpu_fast'], **kwargs: Any) -> CPUFastDecision: ...
@overload
def load_from_hub(repo_id: str, *, revision: str | None = None, calibration_profile: str, backend: Literal['default'] | None = None, **kwargs: Any) -> CalibratedDecision: ...
def load_from_hub(repo_id: str, *, revision: str | None = None, calibration_profile: str | None = None, backend: str | None = None, **kwargs: Any) -> BaseDecision | CalibratedDecision:
    """Download a checkpoint from the Hugging Face Hub, then load it like :func:`load`.

    The download is explicit (this call) and never executes remote code. ``revision`` pins a
    branch, tag or commit; use a commit hash for reproducible results. ``cache_dir`` and
    ``local_files_only`` are passed to the download; every other option (``device``,
    ``precision``, ``max_batch_size``, ...) behaves as in :func:`load`, including the
    machine-appropriate default for ``device`` and ``precision``.

    Raises:
        ImportError: ``huggingface_hub`` or Torch/Transformers are not installed
            (``basedecision[hub]``, ``basedecision[runtime]``).
        InputError: As for :func:`load`, or ``repo_id`` is empty.
    """
    _check_options(backend,calibration_profile)
    if backend=='cpu_fast':
        kwargs.setdefault('device','cpu');kwargs.setdefault('precision','bf16')
        return _quiet(CPUFastDecision.from_hub(repo_id,revision=revision,**kwargs))
    return _calibrated(_quiet(BaseDecision.from_hub(repo_id,revision=revision,**_hardware(kwargs))),calibration_profile)

def decide(model: _BatchBackend[_R_co], *, context: str, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, _R_co]:
    """Answer an ordered mapping of named question schemas over one context.

    Each question is packed and evaluated independently. Context tokenization may
    be reused by local batching; no shared transformer-state computation is claimed.

    Args:
        model: A local model from :func:`load`, or a cloud backend from
            ``BaseDecision.from_provider``.
        context: The text every question is about.
        questions: ``{name: schema}``. A schema has ``kind`` (``'choice'`` by default, ``'noul'``
            or ``'score'``), ``question``, ``options`` (not for ``noul``, which has fixed
            false/true options) and, for ``score``, optional ``values``. Unknown fields are rejected.

    Returns:
        ``{name: result}`` in question order.

    Raises:
        InputError: ``questions`` is empty or malformed, or a question is invalid. Nothing runs
            until every schema has been validated.
    """
    if not isinstance(questions,Mapping) or not questions:raise InputError('questions must be a nonempty mapping')
    requests=[];names=[]
    for name,q in questions.items():
        if not isinstance(name,str) or not name.strip() or not isinstance(q,Mapping):raise InputError('Each named question must have a schema mapping')
        if set(q)-{'kind','question','options','values'}:raise InputError(f'Unknown fields in question {name}')
        kind=q.get('kind','choice');options: tuple[Option, ...]
        if kind=='noul':
            if 'options' in q or 'values' in q:raise InputError('noul has fixed false/true options; omit options and values')
            options=(Option('false','false'),Option('true','true'))
        else:options=options_from(q.get('options',[]))
        values=q.get('values')
        requests.append(Request(context,q.get('question',''),options,kind,values));names.append(name)
    results=model.predict_batch(requests)
    if len(results)!=len(requests):raise RuntimeError('Backend returned an incorrect number of results')
    return dict(zip(names,results))
