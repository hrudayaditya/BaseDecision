"""Local frozen inference. Token-budgeted batching; no global backend settings changed."""
import contextlib,json,threading
from pathlib import Path
from .types import Request,Result,Option,InputError,options_from
from .packing import pack,encode,batch_plan

class BaseDecision:
    @staticmethod
    def from_provider(provider, model, **kwargs):
        """Explicitly opt in to cloud processing via OpenAI or Anthropic."""
        from .providers import ProviderDecision
        return ProviderDecision(provider, model, **kwargs)

    @classmethod
    def from_hub(cls, repo_id, *, revision=None, cache_dir=None, local_files_only=False, **kwargs):
        """Download a model snapshot explicitly; never load remote Python code."""
        if not isinstance(repo_id, str) or not repo_id.strip():
            raise InputError('repo_id must be a nonempty Hub repository ID')
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError('Install basedecision[hub]') from None
        path = snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir,
            local_files_only=local_files_only,
            allow_patterns=['model.safetensors','rl_agent_config.json','encoder/config.json','tokenizer/*'],
            ignore_patterns=['*.py','*.pkl','*.pt','*.bin'])
        return cls.from_pretrained(path, **kwargs)

    @classmethod
    def from_pretrained(cls,path,*,device='cuda',precision='bf16',max_batch_size=1,max_batch_tokens=8192):
        """Load a local exported checkpoint. Remote downloads are not implicit."""
        return cls(path,device=device,precision=precision,max_batch_size=max_batch_size,max_batch_tokens=max_batch_tokens)

    def __init__(self,path,*,device='cuda',precision='bf16',max_batch_size=1,max_batch_tokens=8192):
        list(batch_plan([],max_batch_size,max_batch_tokens))
        if precision not in ('bf16','fp32'):raise InputError('precision must be bf16 or fp32')
        try:
            import torch
            from transformers import AutoConfig,AutoModel,AutoTokenizer
            from safetensors.torch import load_file
        except ImportError as e:raise ImportError('Install basedecision[runtime], or use the existing basedecision ML environment.') from e
        from ._model import DecisionModel
        path=Path(path).expanduser().resolve()
        for f in ('model.safetensors','rl_agent_config.json','encoder/config.json','tokenizer/tokenizer.json'):
            if not (path/f).is_file():raise FileNotFoundError(path/f)
        self._torch=torch;self.device=torch.device(device);self.precision=precision
        if self.device.type not in ('cpu','cuda'):raise InputError('This release candidate supports CPU and CUDA')
        if self.device.type=='cuda':
            if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
            with torch.cuda.device(self.device):
                if precision=='bf16' and not torch.cuda.is_bf16_supported():raise InputError('BF16 is unsupported on this device')
        elif precision!='fp32':raise InputError('CPU inference requires precision="fp32"')
        # B615 reviewed: these two loads use local directories with downloads disabled.
        cfg=AutoConfig.from_pretrained(str(path/'encoder'),local_files_only=True,trust_remote_code=False)  # nosec B615
        if cfg.model_type!='modernbert':raise InputError('Only the evaluated ModernBERT architecture is supported')
        cfg.reference_compile=False
        self.maximum_tokens=min(8192,int(cfg.max_position_embeddings))
        self._tokenizer=AutoTokenizer.from_pretrained(str(path/'tokenizer'),local_files_only=True,trust_remote_code=False)  # nosec B615
        settings=json.loads((path/'rl_agent_config.json').read_text())
        # Construction initializes CPU tensors before strict load. Preserve caller RNG state.
        with torch.random.fork_rng(devices=[]):
            encoder=AutoModel.from_config(cfg,attn_implementation='sdpa',trust_remote_code=False)
            self._model=DecisionModel(encoder,settings.get('head_layers',2),len(settings.get('act_costs',{}))+1)
        state=load_file(str(path/'model.safetensors'),device='cpu')
        self._model.load_state_dict(state,strict=True);del state
        self._model.float().to(self.device).eval().requires_grad_(False)
        self.model_id=str(path);self.max_batch_size=max_batch_size;self.max_batch_tokens=max_batch_tokens
        self._lock=threading.RLock()

    def count_tokens(self,request):
        """Exact complete-input token count; raises if over the supported limit."""
        with self._lock:return pack(self._tokenizer,request,self.maximum_tokens).tokens

    def choose(self,*,context,question,options):
        return self.predict(Request(context,question,options_from(options)))

    def check(self,*,context,question):
        return self.predict(Request(context,question,(Option('false','false'),Option('true','true')),'noul'))

    def score(self,*,context,question,levels,values=None):
        return self.predict(Request(context,question,options_from(levels),'score',None if values is None else tuple(values)))

    def predict(self,request):return self.predict_batch([request])[0]

    def predict_batch(self,requests):
        """Return results in input order. Validate/pack all requests before any forward pass."""
        if isinstance(requests,(str,bytes)):raise InputError('requests must contain Request objects')
        requests=list(requests)
        if not all(isinstance(r,Request) for r in requests):raise InputError('requests must contain Request objects')
        with self._lock:
            # Batch-local context reuse; no retained cross-request cache of user text.
            contexts={};packed=[]
            for r in requests:
                if r.context not in contexts:contexts[r.context]=encode(self._tokenizer,r.context)
                packed.append(pack(self._tokenizer,r,self.maximum_tokens,contexts[r.context]))
            plan=list(batch_plan([p.tokens for p in packed],self.max_batch_size,self.max_batch_tokens))
            results=[None]*len(requests)
            for indices in plan:
                logits=self._forward([packed[i] for i in indices])
                for i,z in zip(indices,logits):results[i]=self._result(requests[i],packed[i],z)
            return results

    def predict_iter(self,requests,*,buffer_size=64):
        """Bound host-memory usage. Earlier buffers may complete before a later invalid request."""
        if isinstance(buffer_size,bool) or not isinstance(buffer_size,int) or buffer_size<1:raise InputError('buffer_size must be a positive integer')
        buffer=[]
        for r in requests:
            buffer.append(r)
            if len(buffer)==buffer_size:yield from self.predict_batch(buffer);buffer=[]
        if buffer:yield from self.predict_batch(buffer)

    def _forward(self,items):
        t=self._torch;n=len(items);length=max(p.tokens for p in items);k=max(len(p.markers) for p in items)
        ids=t.full((n,length),self._tokenizer.pad_token_id,dtype=t.long)
        attention=t.zeros((n,length),dtype=t.long);markers=t.zeros((n,k),dtype=t.long);mask=t.zeros((n,k),dtype=t.bool)
        for i,p in enumerate(items):
            ids[i,:p.tokens]=t.tensor(p.ids);attention[i,:p.tokens]=1
            markers[i,:len(p.markers)]=t.tensor(p.markers);mask[i,:len(p.markers)]=True
        batch=dict(input_ids=ids,attention_mask=attention,marker_pos=markers,marker_mask=mask,qtype=t.tensor([p.qtype for p in items]))
        amp=t.autocast('cuda',dtype=t.bfloat16) if self.precision=='bf16' else contextlib.nullcontext()
        with t.inference_mode(),amp:
            z,action=self._model(**{k:v.to(self.device) for k,v in batch.items()})
        if not bool(t.isfinite(z).all() and t.isfinite(action).all()):raise FloatingPointError('Nonfinite model outputs')
        # One transfer per batch; action head is deliberately not interpreted as abstention.
        return [v[:len(p.markers)] for v,p in zip(z.float().cpu(),items)]

    def _result(self,r,p,z):
        probs=z.softmax(-1).tolist();values=z.tolist();j=int(z.argmax());o=r.options[j]
        return Result(answer=(o.id=='true') if r.kind=='noul' else o.id,option_id=o.id,label=o.text,
            probabilities={o.id:v for o,v in zip(r.options,probs)},raw_logits={o.id:v for o,v in zip(r.options,values)},
            packed_tokens=p.tokens,model_id=self.model_id,kind=r.kind,
            selected_value=r.values[j] if r.values is not None else None,
            expected_value=sum(v*q for v,q in zip(r.values,probs)) if r.values is not None else None)
