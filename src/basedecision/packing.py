"""Complete-description-v1 packing, compatible with the repaired evaluator."""
from dataclasses import dataclass
from .types import InputError,ContextLengthError,Request
QTYPES={'choice':0,'score':1,'noul':2}
@dataclass(frozen=True)
class Packed:
    ids: tuple[int,...]
    markers: tuple[int,...]
    qtype: int
    @property
    def tokens(self):return len(self.ids)

def encode(tok,text):return list(tok(text,add_special_tokens=False)['input_ids'])

def pack(tok,request,maximum=8192,state_ids=None):
    if not isinstance(request,Request):raise InputError('Expected a Request')
    special=[getattr(tok,n,None) for n in ('cls_token_id','sep_token_id','mask_token_id','pad_token_id')]
    if any(v is None for v in special):raise InputError('Tokenizer is missing required special tokens')
    texts=[o.text if request.kind=='choice' else f'level {i}: {o.text}' if request.kind=='score' else f'{o.id}: {o.text}' for i,o in enumerate(request.options)]
    parts=[encode(tok,' '+s) for s in texts]
    if any(not p for p in parts) or len({tuple(p) for p in parts})!=len(parts):raise InputError('Options tokenize identically or to empty text; supply distinct descriptions')
    instruction=encode(tok,f'{request.kind} question: {request.question}')
    state=encode(tok,request.context) if state_ids is None else state_ids
    required=4+len(instruction)+sum(1+len(p) for p in parts)+len(state)
    if required>maximum:raise ContextLengthError(required,maximum)
    ids=[special[0]]+instruction+[special[1]];markers=[]
    for part in parts:markers.append(len(ids));ids.extend([special[2]]+part)
    ids.extend([special[1]]+list(state)+[special[1]])
    if len(ids)!=required:raise RuntimeError('Internal packing token-count mismatch')
    return Packed(tuple(ids),tuple(markers),QTYPES[request.kind])

def batch_plan(lengths,max_batch_size,max_batch_tokens):
    if isinstance(max_batch_size,bool) or not isinstance(max_batch_size,int) or max_batch_size<1:raise InputError('max_batch_size must be a positive integer')
    if isinstance(max_batch_tokens,bool) or not isinstance(max_batch_tokens,int) or max_batch_tokens<1:raise InputError('max_batch_tokens must be a positive integer')
    batch=[]
    for i in sorted(range(len(lengths)),key=lambda j:lengths[j]):
        if lengths[i]>max_batch_tokens:raise InputError(f'Request {i} needs {lengths[i]} tokens, exceeding batch token budget {max_batch_tokens}')
        if batch and (len(batch)==max_batch_size or lengths[i]*(len(batch)+1)>max_batch_tokens):yield batch;batch=[]
        batch.append(i)
    if batch:yield batch
