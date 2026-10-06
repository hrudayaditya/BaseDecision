"""Local frozen inference. Token-budgeted batching; no global backend settings changed."""
from __future__ import annotations
import contextlib,json,threading
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar
from .types import Request,Result,Option,InputError,options_from
from .packing import pack,encode,batch_plan,Packed

if TYPE_CHECKING:
    import torch
    from torch import Tensor
    from .providers import ProviderDecision

_T = TypeVar('_T', bound='BaseDecision')
_R = TypeVar('_R')
_R_co = TypeVar('_R_co', covariant=True)

class _Predicts(Protocol[_R_co]):
    """Anything that answers one request: the local model and the cloud backends."""

    def predict(self, request: Request) -> _R_co:
        """Answer one request."""

class _PredictsBatch(Protocol[_R_co]):
    """Anything that answers a list of requests in order."""

    def predict_batch(self, requests: list[Request]) -> Sequence[_R_co]:
        """Answer the requests in input order."""

class BaseDecision:
    """A local BaseDecision checkpoint, loaded onto a CPU or CUDA device for inference.

    Answers typed questions about a text: :meth:`choose` (pick one option), :meth:`check`
    (yes/no) and :meth:`score` (an ordered level), or any :class:`~basedecision.Request` through
    :meth:`predict`, :meth:`predict_batch` and :meth:`predict_iter`. Results carry the raw softmax
    probabilities. The same convenience methods exist on the cloud backend returned by
    :meth:`from_provider`.

    Most callers should use :func:`basedecision.load`, which picks the device and precision that
    work on this machine. Constructing this class directly is strict: it defaults to CUDA/BF16
    and fails on a machine without a suitable GPU.

    Attributes:
        device: The ``torch.device`` the weights are on.
        precision: ``'bf16'`` (CUDA only) or ``'fp32'``.
        maximum_tokens: Largest packed request in tokens (at most 8,192).
        model_id: Path of the checkpoint.
        max_batch_size: Most requests per forward pass (default 1).
        max_batch_tokens: Padded-token budget per forward pass.

    Calls are serialized by an internal lock, so one instance is safe to share between threads.

    Call :meth:`close` (or use the model as a context manager) to release the weights, and the GPU
    memory they hold, when you are done; the model is otherwise freed when it is garbage collected.
    """

    _closed: bool = False  # a class default, so instances built without __init__ stay usable


    @staticmethod
    def from_provider(provider: str, model: str, **kwargs: Any) -> ProviderDecision:
        """Explicitly opt in to cloud processing via OpenAI or Anthropic.

        Args:
            provider: ``'openai'`` or ``'anthropic'``.
            model: The provider's model id.
            **kwargs: Passed to :class:`~basedecision.providers.ProviderDecision` (``api_key``,
                ``timeout``, ``max_retries``, ``max_output_tokens``, ``max_input_bytes``,
                ``reasoning_effort``).

        Returns:
            A backend with the same ``choose``/``check``/``score``/``predict*`` methods. Use it as
            a context manager so its HTTP client is closed. Results have no probabilities.
        """
        from .providers import ProviderDecision
        return ProviderDecision(provider, model, **kwargs)

    @classmethod
    def from_hub(cls: type[_T], repo_id: str, *, revision: str | None=None, cache_dir: str | Path | None=None, local_files_only: bool=False, **kwargs: Any) -> _T:
        """Download a model snapshot explicitly; never load remote Python code.

        Args:
            repo_id: Hugging Face repository id, ``'owner/name'``.
            revision: Branch, tag or commit hash; a commit hash pins the exact weights.
            cache_dir: Download cache location (the Hugging Face default if ``None``).
            local_files_only: Use only what is already cached; never touch the network.
            **kwargs: Passed to :meth:`from_pretrained` (``device``, ``precision``, ...).

        Raises:
            InputError: ``repo_id`` is empty.
            ImportError: ``huggingface_hub`` is not installed (``pip install "basedecision[hub]"``).
        """
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
    def from_pretrained(cls: type[_T],path: str | Path,*,device: str='cuda',precision: str='bf16',max_batch_size: int=1,max_batch_tokens: int=8192) -> _T:
        """Load a local exported checkpoint. Remote downloads are not implicit.

        The strict constructor: ``device`` and ``precision`` default to CUDA/BF16 and are never
        adjusted. :func:`basedecision.load` is the friendlier entry point.
        """
        return cls(path,device=device,precision=precision,max_batch_size=max_batch_size,max_batch_tokens=max_batch_tokens)

    def __init__(self,path: str | Path,*,device: str='cuda',precision: str='bf16',max_batch_size: int=1,max_batch_tokens: int=8192) -> None:
        """Load the checkpoint in ``path`` onto ``device``; see :meth:`from_pretrained`.

        Args:
            path: Directory with ``model.safetensors``, ``rl_agent_config.json``, ``encoder/`` and
                ``tokenizer/``.
            device: ``'cpu'`` or ``'cuda'`` (or ``'cuda:N'``).
            precision: ``'bf16'`` (CUDA only) or ``'fp32'`` (required on the CPU).
            max_batch_size: Most requests per forward pass.
            max_batch_tokens: Padded-token budget per forward pass.

        Raises:
            ImportError: Torch/Transformers are not installed (``basedecision[runtime]``).
            FileNotFoundError: A required checkpoint file is missing.
            InputError: Unsupported device, precision, architecture or batch limits.
            RuntimeError: CUDA was requested but is unavailable.
        """
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
        self._torch: Any=torch;self.device: torch.device=torch.device(device);self.precision=precision
        if self.device.type not in ('cpu','cuda'):raise InputError('Only CPU and CUDA devices are supported')
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
        self._tokenizer=AutoTokenizer.from_pretrained(str(path/'tokenizer'),local_files_only=True,trust_remote_code=False)  # type: ignore[no-untyped-call,unused-ignore]  # nosec B615
        settings=json.loads((path/'rl_agent_config.json').read_text())
        # Construction initializes CPU tensors before strict load. Preserve caller RNG state.
        with torch.random.fork_rng(devices=[]):
            encoder=AutoModel.from_config(cfg,attn_implementation='sdpa',trust_remote_code=False)  # type: ignore[no-untyped-call,unused-ignore]
            self._model=DecisionModel(encoder,settings.get('head_layers',2),len(settings.get('act_costs',{}))+1)
        state=load_file(str(path/'model.safetensors'),device='cpu')
        self._model.load_state_dict(state,strict=True);del state
        self._model.float().to(self.device).eval().requires_grad_(False)
        self.model_id=str(path);self.max_batch_size=max_batch_size;self.max_batch_tokens=max_batch_tokens
        self._lock=threading.RLock()

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has been called; a closed model cannot answer requests."""
        return self._closed

    def close(self) -> None:
        """Release the weights and the tokenizer, and the GPU memory they hold.

        Waits for a running request to finish first. Safe to call more than once. Afterwards
        ``choose``, ``predict`` and the other request methods raise
        :class:`~basedecision.InputError`; ``model_id``, ``device`` and ``precision`` stay readable.
        """
        with self._lock:
            if self._closed:return
            self._closed=True
            del self._model,self._tokenizer
        import gc
        gc.collect()  # the model and its hooks form reference cycles
        if self.device.type=='cuda':self._torch.cuda.empty_cache()

    def __enter__(self: _T) -> _T:
        """Return the model itself for use in a ``with`` statement."""
        return self

    def __exit__(self,*args: object) -> None:
        """Close the model when the ``with`` block ends."""
        self.close()

    def _check_open(self) -> None:
        if self._closed:raise InputError('The model is closed; load it again to run more requests')

    def count_tokens(self,request: Request) -> int:
        """Exact complete-input token count; raises if over the supported limit.

        Raises:
            InputError: ``request`` is invalid, or the model is closed.
            ContextLengthError: The packed request needs more than ``maximum_tokens``.
        """
        with self._lock:self._check_open();return pack(self._tokenizer,request,self.maximum_tokens).tokens

    def choose(self: _Predicts[_R],*,context: str,question: str,options: Sequence[str | Option]) -> _R:
        """Pick one of ``options`` as the answer to ``question`` about ``context``.

        Args:
            context: The text the question is about.
            question: The question.
            options: 2 to 255 distinct strings and/or :class:`~basedecision.Option` objects.

        Returns:
            The result; ``answer`` is the chosen option's id (the string itself for plain strings).
        """
        return self.predict(Request(context,question,options_from(options)))

    def check(self: _Predicts[_R],*,context: str,question: str) -> _R:
        """Answer a yes/no ``question`` about ``context``; ``answer`` is a Python ``bool``."""
        return self.predict(Request(context,question,(Option('false','false'),Option('true','true')),'noul'))

    def score(self: _Predicts[_R],*,context: str,question: str,levels: Sequence[str | Option],values: Sequence[float] | None=None) -> _R:
        """Pick an ordered level (``levels``, lowest first) for ``question`` about ``context``.

        Args:
            context: The text the question is about.
            question: The question.
            levels: 2 to 255 distinct strings and/or :class:`~basedecision.Option` objects.
            values: Optional strictly increasing number per level. The result then carries
                ``selected_value`` and, locally, the probability-weighted ``expected_value``.
        """
        return self.predict(Request(context,question,options_from(levels),'score',None if values is None else tuple(values)))

    def predict(self,request: Request) -> Result:
        """Answer one :class:`~basedecision.Request`."""
        return self.predict_batch([request])[0]

    def predict_batch(self,requests: Iterable[Request]) -> list[Result]:
        """Return results in input order. Validate/pack all requests before any forward pass.

        Requests are grouped into forward passes of at most ``max_batch_size`` requests and
        ``max_batch_tokens`` padded tokens. A request that cannot be packed raises before any
        request is run.

        Raises:
            InputError: ``requests`` holds something other than ``Request`` objects, a request is
                invalid, or the model is closed.
            ContextLengthError: A request needs more than ``maximum_tokens``.
        """
        if isinstance(requests,(str,bytes)):raise InputError('requests must contain Request objects')
        requests=list(requests)
        if not all(isinstance(r,Request) for r in requests):raise InputError('requests must contain Request objects')
        with self._lock:
            self._check_open()
            # Batch-local context reuse; no retained cross-request cache of user text.
            contexts={};packed=[]
            for r in requests:
                if r.context not in contexts:contexts[r.context]=encode(self._tokenizer,r.context,self.maximum_tokens,'context')
                packed.append(pack(self._tokenizer,r,self.maximum_tokens,contexts[r.context]))
            plan=list(batch_plan([p.tokens for p in packed],self.max_batch_size,self.max_batch_tokens))
            results: list[Any]=[None]*len(requests)  # every index is filled by the plan below
            for indices in plan:
                logits=self._forward([packed[i] for i in indices])
                for i,z in zip(indices,logits):results[i]=self._result(requests[i],packed[i],z)
            return results

    def predict_iter(self: _PredictsBatch[_R],requests: Iterable[Request],*,buffer_size: int=64) -> Iterator[_R]:
        """Bound host-memory usage. Earlier buffers may complete before a later invalid request.

        Lazily yields results in input order, running ``buffer_size`` requests at a time, so a
        long stream never has to be held in memory.

        Raises:
            InputError: ``buffer_size`` is not a positive integer, or a request is invalid.
        """
        if isinstance(buffer_size,bool) or not isinstance(buffer_size,int) or buffer_size<1:raise InputError('buffer_size must be a positive integer')
        buffer: list[Request]=[]
        for r in requests:
            buffer.append(r)
            if len(buffer)==buffer_size:yield from self.predict_batch(buffer);buffer=[]
        if buffer:yield from self.predict_batch(buffer)

    def _forward(self,items: Sequence[Packed]) -> list[Tensor]:
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

    def _result(self,r: Request,p: Packed,z: Tensor) -> Result:
        probs=z.softmax(-1).tolist();values=z.tolist();j=int(z.argmax());o=r.options[j]
        return Result(answer=(o.id=='true') if r.kind=='noul' else o.id,option_id=o.id,label=o.text,
            probabilities={o.id:v for o,v in zip(r.options,probs)},raw_logits={o.id:v for o,v in zip(r.options,values)},
            packed_tokens=p.tokens,model_id=self.model_id,kind=r.kind,
            selected_value=r.values[j] if r.values is not None else None,
            expected_value=sum(v*q for v,q in zip(r.values,probs)) if r.values is not None else None)
