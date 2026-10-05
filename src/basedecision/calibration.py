"""Reviewed local calibration profiles; explicit opt-in, with no refitting at inference."""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
from .types import InputError, Request, Result, options_from

class CalibrationError(InputError):
    """Unsupported scope, incompatible inference contract, or invalid calibration input."""

@dataclass(frozen=True)
class CalibratedResult(Result):
    raw_probabilities: dict | None = None
    calibration: dict | None = None

_DATA = Path(__file__).parent / 'data/calibration_v1.json'
_WARNINGS = {
    'sgd_identifier': 'Opt-in only. Short and two-option slices regressed; aggregate improvement is not universal.',
    'sgd_schema': 'Opt-in only. Short and two-option slices regressed; aggregate improvement is not universal.',
    'clinc': 'Experimental, unavailable for application: assessment uncertainty gate missed.',
    'vast': 'Near-identity temperature; raw output remains the recommended default.',
}

def _read(path):
    try:
        data=json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise CalibrationError('Cannot read calibration artifact') from exc
    if not isinstance(data,dict) or data.get('version')!='1' or not isinstance(data.get('scopes'),dict):
        raise CalibrationError('Unsupported calibration artifact schema')
    return data

def calibration_profiles():
    """Return packaged profile coverage and release limitations. No model load required."""
    data=_read(_DATA)
    return {name: dict(available=spec['assessment_acceptable'],method=spec['method'],
        temperature=math.exp(spec['theta'][0]),kind='choice',
        option_range=[spec['min_options'],spec['max_options']],
        packed_token_range=[spec['min_tokens'],spec['max_tokens']],
        note=_WARNINGS[name]) for name,spec in data['scopes'].items()}

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def _verify(model,data):
    # RC4 intentionally preserves the four measured RC3 inference modules byte-for-byte.
    # Their hashes are the compatibility contract, rather than a cosmetic version string.
    root=Path(__file__).parent
    required={'client.py','packing.py','_model.py','types.py'}
    contract=data.get('sdk_contract',{})
    if not isinstance(contract,dict) or set(contract)!=required or any(_sha(root/name)!=contract[name] for name in required):
        raise CalibrationError('SDK packing/inference contract differs from calibration')
    if data.get('precision')!=getattr(model,'precision',None) or data.get('device_type')!=getattr(getattr(model,'device',None),'type',None):
        raise CalibrationError('Calibration requires its assessed inference precision and device type')
    path=Path(model.model_id).resolve(); expected=data.get('checkpoint',{})
    files=['model.safetensors','rl_agent_config.json','encoder/config.json']
    if not (path/'tokenizer/tokenizer.json').is_file(): raise CalibrationError('Missing local tokenizer')
    files+=sorted(str(p.relative_to(path)) for p in (path/'tokenizer').rglob('*') if p.is_file())
    if not isinstance(expected,dict) or set(files)!=set(expected):raise CalibrationError('Checkpoint/tokenizer file contract differs')
    try:
        if any(_sha(path/name)!=expected[name] for name in files):raise CalibrationError('Checkpoint/tokenizer differs from calibration')
    except OSError as exc:raise CalibrationError('Cannot verify calibration checkpoint') from exc

class CalibratedDecision:
    """Wrap a local BaseDecision instance for one caller-declared, assessed choice workload.

    No automatic workload detection. Raw model stays available to the caller.
    The wrapper does not own or close the underlying model.
    """
    def __init__(self, model, *, profile, artifact=None):
        data=_read(_DATA if artifact is None else artifact)
        spec=data['scopes'].get(profile)
        if spec is None:raise CalibrationError('Unknown calibration profile; inspect calibration_profiles()')
        if not isinstance(spec,dict):raise CalibrationError('Invalid profile schema')
        if spec.get('assessment_acceptable') is not True:raise CalibrationError('Profile did not pass assessment; retain raw probabilities')
        if spec.get('method')!='scalar':raise CalibrationError('This release integrates reviewed scalar profiles only')
        theta=spec.get('theta')
        if not isinstance(theta,list) or len(theta)!=1 or isinstance(theta[0],bool) or not isinstance(theta[0],(int,float)) or not math.isfinite(theta[0]) or not math.log(.05)<=theta[0]<=math.log(100):
            raise CalibrationError('Invalid scalar temperature')
        for lo,hi in [('min_options','max_options'),('min_tokens','max_tokens')]:
            if any(type(spec.get(k)) is not int for k in (lo,hi)) or not 1<=spec[lo]<=spec[hi]:raise CalibrationError('Invalid calibration coverage')
        if not hasattr(model,'model_id') or not hasattr(model,'count_tokens'):raise CalibrationError('Calibration is supported for local logits only')
        _verify(model,data)
        self._model=model;self.model_id=model.model_id;self.profile=profile;self._spec=dict(spec)
        self.temperature=math.exp(theta[0]);self._artifact_sha=_sha(_DATA if artifact is None else artifact)

    def _coverage(self,kind,options,tokens):
        if kind!='choice':raise CalibrationError('Profile covers choice only; use the raw model for noul or score')
        s=self._spec
        if not s['min_options']<=options<=s['max_options'] or not s['min_tokens']<=tokens<=s['max_tokens']:
            raise CalibrationError('Request outside assessed option/token range; use raw output or a representative calibration set')

    def apply(self,result):
        """Apply once to a raw result from the bound model. Preserve the selected answer."""
        if not isinstance(result,Result) or result.model_id!=self.model_id:raise CalibrationError('Result belongs to another model/backend')
        if result.calibration_status!='raw_softmax_uncalibrated':raise CalibrationError('Expected raw output; cannot calibrate twice')
        ids=list(result.raw_logits);z=list(result.raw_logits.values())
        self._coverage(result.kind,len(ids),result.packed_tokens)
        if set(ids)!=set(result.probabilities) or result.option_id not in ids or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in z):
            raise CalibrationError('Invalid result logits/options')
        m=max(z);e=[math.exp((v-m)/self.temperature) for v in z];total=math.fsum(e)
        p={k:v/total for k,v in zip(ids,e)}
        fields=asdict(result)
        fields.update(probabilities=p,calibration_status=f'posthoc_scalar:scope={self.profile}:v1')
        return CalibratedResult(**fields,raw_probabilities=dict(result.probabilities),calibration=dict(
            profile=self.profile,temperature=self.temperature,artifact_sha256=self._artifact_sha,
            note=_WARNINGS.get(self.profile,'Caller-supplied calibration artifact; validate workload coverage.')))

    def count_tokens(self,request):return self._model.count_tokens(request)
    def predict(self,request):return self.predict_batch([request])[0]
    def choose(self,*,context,question,options):return self.predict(Request(context,question,options_from(options)))
    def check(self,**kwargs):raise CalibrationError('No reviewed noul profile; call the raw model.check()')
    def score(self,**kwargs):raise CalibrationError('No reviewed score profile; call the raw model.score()')
    def predict_batch(self,requests):
        if isinstance(requests,(str,bytes)):raise InputError('requests must contain Request objects')
        requests=list(requests)
        # Validate every item's scope and full packing before invoking inference.
        for r in requests:
            if not isinstance(r,Request):raise InputError('requests must contain Request objects')
            self._coverage(r.kind,len(r.options),self._model.count_tokens(r))
        return [self.apply(r) for r in self._model.predict_batch(requests)]
    def predict_iter(self,requests,*,buffer_size=64):
        if type(buffer_size) is not int or buffer_size<1:raise InputError('buffer_size must be a positive integer')
        buffer=[]
        for r in requests:
            buffer.append(r)
            if len(buffer)==buffer_size:yield from self.predict_batch(buffer);buffer=[]
        if buffer:yield from self.predict_batch(buffer)
