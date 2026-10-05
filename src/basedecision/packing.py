"""Complete-description-v1 packing: the exact token sequence the model reads.

Three safety properties hold for every input; for ordinary text the packed tokens are identical
to earlier releases (``tests/test_input_hardening.py`` replays a golden corpus to prove it):

* **User text is text.** Special-token strings such as ``[SEP]`` or ``[MASK]`` typed into a
  context, question or option are tokenized as ordinary characters, so untrusted text can never
  forge the structural tokens that delimit the question, the options and the context.
* **Invalid Unicode is rejected clearly.** A lone surrogate is an ``InputError`` naming the part
  that contains it, never a raw ``TypeError`` from the tokenizer.
* **Too long means "too long", quickly.** The window is fixed (8,192 tokens) and nothing is ever
  truncated, so an input that cannot fit is rejected after a bounded amount of work instead of
  tokenizing all of it first; see :data:`EXACT_CHARS`.
"""
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from .types import InputError,ContextLengthError,Request
QTYPES={'choice':0,'score':1,'noul':2}
"""Question kind -> the type embedding the model was trained with."""

class Tokenizer(Protocol):
    """The part of a Hugging Face tokenizer that packing uses.

    Calling it returns a mapping with ``input_ids``. ``cls_token_id``, ``sep_token_id``,
    ``mask_token_id`` and ``pad_token_id`` must be set; :func:`pack` refuses a tokenizer without them.
    """

    def __call__(self, text: str, *, add_special_tokens: bool, split_special_tokens: bool) -> Mapping[str, Any]:
        """Tokenize ``text`` without adding special tokens, treating special-token strings as text."""

EXACT_CHARS=262_144
"""Up to this many characters of input are always tokenized exactly, so that an input that is
only slightly too long is reported with its exact token count. Beyond it, tokenization stops as
soon as the window is clearly exceeded and the error says "at least N tokens". The cost of
rejecting an input is therefore bounded by a few times this constant, however large the input."""
_PROBE_MARGIN=4
"""A prefix cut in the middle of a word can tokenize into a few more tokens than the same span of
the whole text. Early rejection counts only tokens beyond this margin, so it can never reject an
input that would have fit."""

@dataclass(frozen=True)
class Packed:
    """One request packed into the exact token sequence the model reads.

    Attributes:
        ids: All token ids, including the structural ``[CLS]``/``[SEP]``/``[MASK]`` tokens.
        markers: Position of each option's marker token; the model scores the options there.
        qtype: Type-embedding index of the question kind (see :data:`QTYPES`).
    """

    ids: tuple[int,...]
    markers: tuple[int,...]
    qtype: int
    @property
    def tokens(self) -> int:
        """Total number of tokens."""
        return len(self.ids)

def _is_valid_text(text: str) -> bool:
    try:text.encode('utf-8')
    except UnicodeEncodeError:return False
    return True

def _tokenize(tok: Tokenizer,text: str,what: str) -> list[int]:
    """Token ids of ``text`` with special-token strings treated as ordinary text."""
    try:
        return list(tok(text,add_special_tokens=False,split_special_tokens=True)['input_ids'])
    except TypeError as error:
        problem=error
    # Decided outside the handler so that no exception holding the text is attached to ours.
    if not _is_valid_text(text):
        raise InputError(f'The {what} contains a character that is not valid Unicode text (a lone surrogate); remove or replace it') from None
    raise problem

def encode(tok: Tokenizer,text: str,budget: int | None=None,what: str='input') -> list[int]:
    """Token ids of ``text``. With a ``budget`` (the tokens this text may still use), an input far
    beyond it is rejected without tokenizing all of it.

    Raises:
        InputError: ``text`` is not valid Unicode.
        ContextLengthError: more than ``budget`` tokens are clearly needed and ``text`` is longer
            than :data:`EXACT_CHARS`; ``required_tokens`` is then a lower bound (``at_least``).
    """
    if budget is None or len(text)<=EXACT_CHARS:return _tokenize(tok,text,what)
    budget=max(budget,0);size=max(4*budget,1024)
    while size<len(text):
        counted=len(_tokenize(tok,text[:size],what))-_PROBE_MARGIN
        if counted>budget:raise ContextLengthError(max(counted,budget+1),budget,at_least=True)
        size*=2
    return _tokenize(tok,text,what)

class _Window:
    """Running token total of one packed request, stopping early once it is clearly too long."""

    def __init__(self,tok: Tokenizer,maximum: int) -> None:self.tok=tok;self.maximum=maximum;self.used=4;self.chars=0  # [CLS] and three [SEP]
    def take(self,text: str,what: str,overhead: int=0,last: bool=False) -> list[int]:
        """Tokenize one part of the request and add it to the running total."""
        try:ids=encode(self.tok,text,self.maximum-self.used-overhead,what)
        except ContextLengthError as early:
            raise ContextLengthError(self.used+overhead+early.required_tokens,self.maximum,at_least=True) from None
        self.used+=overhead+len(ids);self.chars+=len(text)
        if self.used>self.maximum and self.chars>EXACT_CHARS and not last:
            raise ContextLengthError(self.used,self.maximum,at_least=True)
        return ids

def pack(tok: Tokenizer,request: Request,maximum: int=8192,state_ids: Sequence[int] | None=None) -> Packed:
    """Pack ``request`` into the exact token sequence the model reads ("complete-description-v1").

    Args:
        tok: The checkpoint's tokenizer.
        request: The request to pack. Nothing is truncated or dropped.
        maximum: Window size in tokens (8,192 unless the checkpoint is smaller).
        state_ids: The context's token ids, if already computed (batches share them).

    Returns:
        The packed ids, the option marker positions and the question-type index.

    Raises:
        InputError: Not a ``Request``, the tokenizer lacks special tokens, the text is not valid
            Unicode, or two options tokenize identically (or to nothing).
        ContextLengthError: The complete input needs more than ``maximum`` tokens.
    """
    if not isinstance(request,Request):raise InputError('Expected a Request')
    special: list[Any]=[getattr(tok,n,None) for n in ('cls_token_id','sep_token_id','mask_token_id','pad_token_id')]
    if any(v is None for v in special):raise InputError('Tokenizer is missing required special tokens')
    texts=[o.text if request.kind=='choice' else f'level {i}: {o.text}' if request.kind=='score' else f'{o.id}: {o.text}' for i,o in enumerate(request.options)]
    window=_Window(tok,maximum)
    parts=[window.take(' '+s,'option text',overhead=1) for s in texts]
    if any(not p for p in parts) or len({tuple(p) for p in parts})!=len(parts):raise InputError('Options tokenize identically or to empty text; supply distinct descriptions')
    instruction=window.take(f'{request.kind} question: {request.question}','question')
    state: Sequence[int]
    if state_ids is None:state=window.take(request.context,'context',last=True)
    else:state=state_ids;window.used+=len(state)
    required=window.used
    if required>maximum:raise ContextLengthError(required,maximum)
    ids=[special[0]]+instruction+[special[1]];markers: list[int]=[]
    for part in parts:markers.append(len(ids));ids.extend([special[2]]+part)
    ids.extend([special[1]]+list(state)+[special[1]])
    if len(ids)!=required:raise RuntimeError('Internal packing token-count mismatch')
    return Packed(tuple(ids),tuple(markers),QTYPES[request.kind])

def batch_plan(lengths: Sequence[int],max_batch_size: int,max_batch_tokens: int) -> Iterator[list[int]]:
    """Group request indices into batches, shortest first, within both budgets.

    Yields lists of indices into ``lengths``. A batch holds at most ``max_batch_size`` requests and
    at most ``max_batch_tokens`` padded tokens (longest request x batch size).

    Raises:
        InputError: A budget is not a positive integer, or one request alone exceeds
            ``max_batch_tokens``.
    """
    if isinstance(max_batch_size,bool) or not isinstance(max_batch_size,int) or max_batch_size<1:raise InputError('max_batch_size must be a positive integer')
    if isinstance(max_batch_tokens,bool) or not isinstance(max_batch_tokens,int) or max_batch_tokens<1:raise InputError('max_batch_tokens must be a positive integer')
    batch: list[int]=[]
    for i in sorted(range(len(lengths)),key=lambda j:lengths[j]):
        if lengths[i]>max_batch_tokens:raise InputError(f'Request {i} needs {lengths[i]} tokens, exceeding batch token budget {max_batch_tokens}')
        if batch and (len(batch)==max_batch_size or lengths[i]*(len(batch)+1)>max_batch_tokens):yield batch;batch=[]
        batch.append(i)
    if batch:yield batch
