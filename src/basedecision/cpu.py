"""Explicit experimental CPU BF16 inference; no process-wide tuning."""
from dataclasses import dataclass, asdict
import warnings
from .client import BaseDecision
from .types import Result,InputError

@dataclass(frozen=True)
class CPUResult(Result):
    backend: str = 'cpu_fast_experimental'
    inference_precision: str = 'bf16_autocast_fp32_weights'

class CPUFastDecision(BaseDecision):
    @classmethod
    def from_pretrained(cls,path,*,device='cpu',precision='bf16',max_batch_size=1,max_batch_tokens=8192):
        return cls(path,device=device,precision=precision,max_batch_size=max_batch_size,max_batch_tokens=max_batch_tokens)

    def __init__(self,path,*,device='cpu',precision='bf16',max_batch_size=1,max_batch_tokens=8192):
        if device!='cpu' or precision!='bf16':raise InputError('cpu_fast requires device="cpu", precision="bf16"')
        import torch,transformers
        if transformers.__version__!='4.57.6' or torch.__version__.split('+')[0]!='2.9.1':
            raise InputError('Experimental cpu_fast requires torch 2.9.1 and transformers 4.57.6; use the default CPU FP32 backend otherwise')
        super().__init__(path,device='cpu',precision='fp32',max_batch_size=max_batch_size,max_batch_tokens=max_batch_tokens)
        from ._cpu_attention import install,install_head
        self._cpu_counters=install(self._model);install_head(self._model)
        self.inference_precision='bf16_autocast_fp32_weights'
        warnings.warn('Experimental CPU fast mode: probabilities are uncalibrated; speed depends on hardware BF16 support. No thread or affinity settings were changed.',RuntimeWarning,stacklevel=2)

    def _forward(self,items):
        with self._torch.autocast('cpu',dtype=self._torch.bfloat16):
            return super()._forward(items)

    def _result(self,r,p,z):
        return CPUResult(**asdict(super()._result(r,p,z)))

    def backend_info(self):
        with self._lock:
            return dict(backend='cpu_fast_experimental',device='cpu',weight_precision=self.precision,
                inference_precision=self.inference_precision,calibration_supported=False,
                attention_calls=dict(self._cpu_counters),threads=self._torch.get_num_threads(),
                fallback='Original backbone attention for padded/batched/global attention; CPU BF16 head retained')
