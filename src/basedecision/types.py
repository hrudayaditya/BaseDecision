"""Public value types. No ML runtime import needed for validation."""
from dataclasses import dataclass, asdict
import math

class InputError(ValueError):
    """Invalid request schema or ambiguous option rendering."""

class ContextLengthError(InputError):
    def __init__(self, required, maximum):
        self.required_tokens, self.maximum_tokens = required, maximum
        super().__init__(f'Complete input requires {required} tokens; maximum is {maximum}. No text was truncated.')

@dataclass(frozen=True)
class Option:
    id: str
    text: str
    def __post_init__(self):
        if not isinstance(self.id,str) or not self.id.strip():raise InputError('Option ID must be a nonempty string')
        if not isinstance(self.text,str) or not self.text.strip():raise InputError('Option text must be a nonempty string')

@dataclass(frozen=True)
class Request:
    context: str
    question: str
    options: tuple[Option,...]
    kind: str = 'choice'
    values: tuple[float,...] | None = None
    def __post_init__(self):
        if not isinstance(self.context,str):raise InputError('context must be a string; serialize structured input explicitly')
        if not isinstance(self.question,str) or not self.question.strip():raise InputError('question must be a nonempty string')
        if self.kind not in ('choice','noul','score'):raise InputError('kind must be choice, noul or score')
        if not isinstance(self.options,(tuple,list)):raise InputError('options must be a sequence of Option objects')
        object.__setattr__(self,'options',tuple(self.options))
        if not 2<=len(self.options)<=255 or not all(isinstance(o,Option) for o in self.options):raise InputError('Provide 2..255 Option objects')
        if len({o.id for o in self.options})!=len(self.options):raise InputError('Option IDs must be unique')
        if len({o.text for o in self.options})!=len(self.options):raise InputError('Option descriptions must be distinct')
        if self.kind=='noul' and [o.id for o in self.options]!=['false','true']:raise InputError('noul options must be ordered false, true')
        if self.values is not None:
            if self.kind!='score' or not isinstance(self.values,(tuple,list)) or len(self.values)!=len(self.options):raise InputError('values must match score levels')
            if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in self.values):raise InputError('Score values must be finite numbers')
            if any(a>=b for a,b in zip(self.values,self.values[1:])):raise InputError('Score values must be strictly increasing')
            object.__setattr__(self,'values',tuple(float(v) for v in self.values))

@dataclass(frozen=True)
class Result:
    answer: str | bool
    option_id: str
    label: str
    probabilities: dict[str,float]
    raw_logits: dict[str,float]
    packed_tokens: int
    model_id: str
    kind: str
    calibration_status: str = 'raw_softmax_uncalibrated'
    selected_value: float | None = None
    expected_value: float | None = None
    def to_dict(self):return asdict(self)

def options_from(labels):
    if isinstance(labels,(str,bytes)):raise InputError('options must be a sequence, not a string')
    try:values=list(labels)
    except TypeError as e:raise InputError('options must be iterable') from e
    if not all(isinstance(x,(str,Option)) for x in values):raise InputError('Options must be strings or Option objects')
    return tuple(Option(x,x) if isinstance(x,str) else x for x in values)
