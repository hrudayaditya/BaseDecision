"""Public value types. No ML runtime import needed for validation."""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, asdict
from typing import Any, Literal
import math

Kind = Literal['choice','noul','score']
"""The three question types: pick one option, answer yes/no (``noul``), or pick an ordered level."""

class InputError(ValueError):
    """Invalid request schema or ambiguous option rendering."""

class ContextLengthError(InputError):
    """The complete input does not fit the model's window. Nothing is ever truncated.

    ``required_tokens`` is exact unless ``at_least`` is true: an input far beyond the window is
    rejected before all of it has been tokenized, and the count is then a lower bound.
    """

    def __init__(self, required: int, maximum: int, at_least: bool = False) -> None:
        """Record the token counts and build the message (``at_least`` marks a lower bound)."""
        self.required_tokens, self.maximum_tokens, self.at_least = required, maximum, bool(at_least)
        qualifier = 'at least ' if at_least else ''
        super().__init__(f'Complete input requires {qualifier}{required} tokens; maximum is {maximum}. No text was truncated.')

@dataclass(frozen=True)
class Option:
    """One answer the model may choose: a stable ``id`` and the ``text`` it reads.

    Plain strings passed as options become ``Option(text, text)``. Give an explicit description
    (``Option('refund', 'Customer wants money back')``) when a label alone is ambiguous.

    Raises:
        InputError: ``id`` or ``text`` is not a nonempty string.
    """

    id: str
    text: str
    def __post_init__(self) -> None:
        """Validate that both fields are nonempty strings."""
        if not isinstance(self.id,str) or not self.id.strip():raise InputError('Option ID must be a nonempty string')
        if not isinstance(self.text,str) or not self.text.strip():raise InputError('Option text must be a nonempty string')

@dataclass(frozen=True)
class Request:
    """One question about one text: the unit every backend answers.

    Attributes:
        context: The text the question is about (a string; serialize structured data yourself).
        question: The question to answer.
        options: 2 to 255 distinct :class:`Option` objects. For ``noul`` they must be exactly
            ``false`` then ``true``; for ``score`` they are the ordered levels.
        kind: ``'choice'``, ``'noul'`` (yes/no) or ``'score'``.
        values: Optional strictly increasing numbers, one per ``score`` level.

    Raises:
        InputError: Any field is invalid; the message names the problem.
    """

    context: str
    question: str
    options: tuple[Option,...]
    kind: Kind = 'choice'
    values: tuple[float,...] | None = None
    def __post_init__(self) -> None:
        """Validate every field and normalize ``options`` and ``values`` to tuples."""
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
    """A local model's answer, with the full probability distribution behind it.

    Attributes:
        answer: The chosen option's id, or a ``bool`` for ``noul`` questions.
        option_id: Id of the chosen option (``'true'``/``'false'`` for ``noul``).
        label: Text of the chosen option.
        probabilities: Raw softmax over the options, keyed by option id (uncalibrated).
        raw_logits: The logits behind ``probabilities``, keyed by option id.
        packed_tokens: Tokens the packed input used (at most the 8,192-token window).
        model_id: Path of the checkpoint that produced the result.
        kind: The question type that was answered.
        calibration_status: ``'raw_softmax_uncalibrated'`` unless a calibration profile was applied.
        selected_value: For ``score`` questions with ``values``: the chosen level's value.
        expected_value: For ``score`` questions with ``values``: the probability-weighted value.
    """

    answer: str | bool
    option_id: str
    label: str
    probabilities: dict[str,float]
    raw_logits: dict[str,float]
    packed_tokens: int
    model_id: str
    kind: Kind
    calibration_status: str = 'raw_softmax_uncalibrated'
    selected_value: float | None = None
    expected_value: float | None = None
    def to_dict(self) -> dict[str, Any]:
        """Return the result as a plain, JSON-serializable dictionary."""
        return asdict(self)

def options_from(labels: Sequence[str | Option]) -> tuple[Option, ...]:
    """Normalize a list or tuple of strings and/or :class:`Option` objects to ``Option`` tuples.

    A string ``s`` becomes ``Option(s, s)``.

    Raises:
        InputError: ``labels`` is a string, bytes, mapping or set (the order would be lost or the
            meaning guessed), is not iterable, or holds anything but strings and ``Option``s.
    """
    if isinstance(labels,(str,bytes)):raise InputError('options must be a sequence, not a string')
    if isinstance(labels,(Mapping,set,frozenset)):
        raise InputError('options must be an ordered sequence (a list or tuple) of strings or Option objects, not a mapping or set; to give an option a description use Option(id, text)')
    try:values=list(labels)
    except TypeError as e:raise InputError('options must be iterable') from e
    if not all(isinstance(x,(str,Option)) for x in values):raise InputError('Options must be strings or Option objects')
    return tuple(Option(x,x) if isinstance(x,str) else x for x in values)
