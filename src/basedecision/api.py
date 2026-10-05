"""Small Python convenience API layered over the existing inference contract."""
from collections.abc import Mapping
from .client import BaseDecision
from .devices import resolve_device
from .types import Request, Option, InputError, options_from

def _hardware(kwargs):
    """Fill in device/precision from resolve_device() unless the caller gave them explicitly."""
    device,precision=resolve_device(kwargs.pop('device',None),kwargs.pop('precision',None))
    return dict(kwargs,device=device,precision=precision)

def _check_options(backend,calibration_profile):
    """Validate the backend and calibration choice before any model is loaded or downloaded."""
    if backend not in (None,'default','cpu_fast'):raise InputError('backend must be default or cpu_fast')
    if backend=='cpu_fast' and calibration_profile is not None:
        from .calibration import CalibrationError
        raise CalibrationError('CPU calibration transfer has not been validated; use raw output')
    if calibration_profile is not None:
        from .calibration import calibration_profiles, CalibrationError
        info=calibration_profiles().get(calibration_profile)
        if info is None or not info['available']:raise CalibrationError('Unknown or unavailable calibration profile')

def _calibrated(model,calibration_profile):
    if calibration_profile is None:return model
    from .calibration import CalibratedDecision
    return CalibratedDecision(model,profile=calibration_profile)

def load(path, *, calibration_profile=None, backend=None, **kwargs):
    """Load an explicitly named local checkpoint. Calibration is opt-in.

    Unless given, ``device`` and ``precision`` are chosen for this machine (see
    :func:`resolve_device`): CUDA/BF16 when a GPU with native BF16 is present, otherwise
    CPU/FP32. Pass them explicitly to override, e.g. ``load(path, device='cpu', precision='fp32')``.
    """
    _check_options(backend,calibration_profile)
    if backend=='cpu_fast':
        from .cpu import CPUFastDecision
        kwargs.setdefault('device','cpu');kwargs.setdefault('precision','bf16')
        return CPUFastDecision.from_pretrained(path,**kwargs)
    return _calibrated(BaseDecision.from_pretrained(path,**_hardware(kwargs)),calibration_profile)

def load_from_hub(repo_id, *, revision=None, calibration_profile=None, backend=None, **kwargs):
    """Download a checkpoint from the Hugging Face Hub, then load it like :func:`load`.

    The download is explicit (this call) and never executes remote code. ``revision`` pins a
    branch, tag or commit; use a commit hash for reproducible results. ``cache_dir`` and
    ``local_files_only`` are passed to the download; every other option (``device``,
    ``precision``, ``max_batch_size``, ...) behaves as in :func:`load`, including the
    machine-appropriate default for ``device`` and ``precision``.
    """
    _check_options(backend,calibration_profile)
    if backend=='cpu_fast':
        from .cpu import CPUFastDecision
        kwargs.setdefault('device','cpu');kwargs.setdefault('precision','bf16')
        return CPUFastDecision.from_hub(repo_id,revision=revision,**kwargs)
    return _calibrated(BaseDecision.from_hub(repo_id,revision=revision,**_hardware(kwargs)),calibration_profile)

def decide(model, *, context, questions):
    """Answer an ordered mapping of named question schemas over one context.

    Each question is packed and evaluated independently. Context tokenization may
    be reused by local batching; no shared transformer-state computation is claimed.
    """
    if not isinstance(questions,Mapping) or not questions:raise InputError('questions must be a nonempty mapping')
    requests=[];names=[]
    for name,q in questions.items():
        if not isinstance(name,str) or not name.strip() or not isinstance(q,Mapping):raise InputError('Each named question must have a schema mapping')
        if set(q)-{'kind','question','options','values'}:raise InputError(f'Unknown fields in question {name}')
        kind=q.get('kind','choice')
        if kind=='noul':
            if 'options' in q or 'values' in q:raise InputError('noul has fixed false/true options; omit options and values')
            options=(Option('false','false'),Option('true','true'))
        else:options=options_from(q.get('options',[]))
        values=q.get('values')
        requests.append(Request(context,q.get('question'),options,kind,values));names.append(name)
    results=model.predict_batch(requests)
    if len(results)!=len(requests):raise RuntimeError('Backend returned an incorrect number of results')
    return dict(zip(names,results))
