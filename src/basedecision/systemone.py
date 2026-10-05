"""Jev/SystemOne API adapter for local BaseDecision models.

The Jev/SystemOne API (``POST /v1/systemone``) takes a *state* and a schema of typed
*questions* and returns, for every question, a probability for each allowed option::

    request  = {"model": "...", "state": <str | JSON>,
                "questions": {<id>: {"type": "choice" | "score" | "noul",
                                     "instructions": <optional>, "criteria": ...}}}
    response = {"model": "...", "answers": {<id>: {...}}, "usage": {...}}

:class:`SystemOne` (or the one-shot :func:`systemone`) maps such a request body onto
BaseDecision requests, runs them on a local model, and returns the response body in the
documented wire format. It is a pure ``dict -> dict`` function: no web framework, no
network access, and no import of Torch. See ``docs/SYSTEMONE.md`` for the full mapping
and for the differences from the reference implementation.

Design rules, in line with the rest of the package:

* Nothing is silently dropped, truncated or guessed. A request that BaseDecision cannot
  honour (images, an over-long input, ambiguous options) is rejected with a
  :class:`SystemOneError` that carries a stable ``code`` and the offending ``field``.
* Every request is validated *before* any inference runs.
* Probabilities are the model's raw softmax. Cloud providers return no probabilities, so
  they cannot serve this API and are rejected when the service is created.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, NoReturn, Protocol, TypedDict

from .providers import ProviderDecision
from .types import ContextLengthError, InputError, Option, Request, Result

__all__ = [
    "DEFAULT_MAX_QUESTIONS",
    "DEFAULT_MAX_STATE_CHARS",
    "ChoiceAnswer",
    "DecisionBackend",
    "ErrorCode",
    "NoulAnswer",
    "ScoreAnswer",
    "SystemOne",
    "SystemOneError",
    "SystemOneResponse",
    "Usage",
    "parse_request_body",
    "systemone",
]

DEFAULT_MAX_QUESTIONS: Final = 64
"""Default cap on questions per request. Each question is one forward pass."""

DEFAULT_MAX_STATE_CHARS: Final = 1_000_000
"""Default cap on the rendered state, in characters.

The tokenizer's longest token is 512 characters, so no text above ``8192 * 512`` characters
can fit the model's window; the default is far below that bound yet far above any realistic
state, and keeps the cost of rejecting an oversized input small.
"""

_MAX_OPTIONS: Final = 255
_ROUND_DIGITS: Final = 4
_QUESTION_TYPES: Final = ("choice", "score", "noul")
_REQUEST_FIELDS: Final = frozenset(
    {"model", "state", "questions", "images", "videos", "media_kwargs", "id"}
)
_QUESTION_FIELDS: Final = frozenset({"type", "instructions", "criteria"})

QuestionType = Literal["choice", "score", "noul"]
ErrorCode = Literal[
    "invalid_request",
    "unknown_field",
    "unsupported_modality",
    "invalid_question",
    "invalid_criteria",
    "limit_exceeded",
    "model_not_found",
    "context_length_exceeded",
    "unsupported_backend",
]


class NoulAnswer(TypedDict):
    """Answer to a ``noul`` (true/false) question: the probability that it is true."""

    type: Literal["noul"]
    noul: float


class ChoiceAnswer(TypedDict):
    """Answer to a ``choice`` question."""

    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(TypedDict):
    """Answer to a ``score`` question; ``score`` is the expected zero-based level index."""

    type: Literal["score"]
    score: float
    confidence: float
    legend: dict[str, Any]
    probabilities: dict[str, float]


class Usage(TypedDict):
    """Token accounting. ``output_tokens`` is always 0: the models never generate text."""

    input_tokens: int
    output_tokens: int


class SystemOneResponse(TypedDict):
    """The Jev/SystemOne response body."""

    model: str
    answers: dict[str, ChoiceAnswer | ScoreAnswer | NoulAnswer]
    usage: Usage


class DecisionBackend(Protocol):
    """What :class:`SystemOne` needs from a model: ordered batch prediction.

    Satisfied by :class:`basedecision.BaseDecision` and ``CPUFastDecision``, and by
    ``CalibratedDecision`` for ``choice`` questions. The results must carry
    per-option ``probabilities``.
    """

    def predict_batch(self, requests: Sequence[Request]) -> Sequence[Result]:
        """Return one result per request, in input order."""


class SystemOneError(InputError):
    """A request that cannot be answered.

    Subclasses :class:`~basedecision.InputError`, so existing ``except InputError``
    handlers also catch it. The ``code`` is stable and meant for programmatic handling;
    the message is meant for humans. Messages never contain the request ``state``.

    Attributes:
        code: Machine-readable error code (see :data:`ErrorCode`).
        field: Dotted path of the offending request field, e.g. ``questions.urgency``,
            or ``None`` when the error is not tied to a field.
        details: Extra integers for some codes; ``context_length_exceeded`` carries
            ``required_tokens`` and ``maximum_tokens``.
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        field: str | None = None,
        **details: int,
    ) -> None:
        """Create an error; ``details`` become the ``details`` attribute."""
        super().__init__(message)
        self.code: ErrorCode = code
        self.field = field
        self.details: dict[str, int] = dict(details)

    def __reduce__(self) -> tuple[Any, ...]:
        """Support pickling, which keyword-only arguments would otherwise break."""
        # The default Exception pickling would call __init__ with the message only.
        return (_rebuild_error, (self.code, str(self), self.field, self.details))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable ``{"error": {...}}`` body (a BaseDecision convention)."""
        error: dict[str, Any] = {"code": self.code, "message": str(self)}
        if self.field is not None:
            error["field"] = self.field
        error.update(self.details)
        return {"error": error}


def _rebuild_error(
    code: ErrorCode, message: str, field: str | None, details: dict[str, int]
) -> SystemOneError:
    return SystemOneError(code, message, field=field, **details)


def parse_request_body(data: bytes | str) -> dict[str, Any]:
    """Parse the raw JSON text of a request body, strictly.

    Rejects what a standard JSON parser would silently accept: ``NaN``/``Infinity``,
    duplicate object keys (a repeated question ID would otherwise drop a question), and
    invalid UTF-8. The top-level value must be an object.

    Raises:
        SystemOneError: ``invalid_request`` if the body is not a valid JSON object.
    """

    def reject_constant(name: str) -> NoReturn:
        raise ValueError(f"non-standard JSON constant {name}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate object key")  # never echo the key itself
            result[key] = value
        return result

    value: Any = None
    reason: str | None = None
    try:
        text = data.decode("utf-8") if isinstance(data, bytes) else data
        value = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_pairs)
    except (ValueError, RecursionError) as exc:
        # json.JSONDecodeError and UnicodeDecodeError are ValueError subclasses and hold the
        # whole body (.doc / .object). Keep a short description only and raise below, outside
        # the handler, so the exception that holds the request text is not attached.
        reason = _brief(exc)
    if reason is not None:
        raise SystemOneError("invalid_request", f"request body is not valid JSON: {reason}")
    if not isinstance(value, dict):
        raise SystemOneError("invalid_request", "request body must be a JSON object")
    return value


def _brief(exc: BaseException) -> str:
    """Describe a parser error briefly, without echoing any input text."""
    if isinstance(exc, json.JSONDecodeError):
        return f"{exc.msg} at line {exc.lineno} column {exc.colno}"
    if isinstance(exc, RecursionError):
        return "nesting too deep"
    if isinstance(exc, UnicodeDecodeError):
        return "invalid UTF-8"
    return str(exc)


def _render(value: object, field: str) -> str:
    """Render a state/instruction value as text: strings as-is, other JSON compactly."""
    if isinstance(value, str):
        return value
    with contextlib.suppress(TypeError, ValueError, RecursionError):
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    # Raised outside the suppressed handler, so no exception about the value is attached.
    raise SystemOneError(
        "invalid_request",
        f"{field} must be a string or a finite, JSON-serializable value",
        field=field,
    )


def _option_text(value: object, fallback: str, field: str) -> str:
    """Option text from a criteria description; blank or missing falls back to the id."""
    if value is None:
        return fallback
    text = _render(value, field)
    return text if text.strip() else fallback


@dataclass(frozen=True)
class _Question:
    qid: str
    kind: QuestionType
    request: Request
    order: tuple[str, ...]  # option ids in the order the response must list them
    legend: tuple[Any, ...] = ()  # raw level descriptions, for score questions


@dataclass(frozen=True)
class _Parsed:
    model: str
    questions: tuple[_Question, ...]


class SystemOne:
    """Answers Jev/SystemOne request bodies using a local BaseDecision model.

    Args:
        backend: A local model (``load(...)``/``BaseDecision``/``CPUFastDecision``, or a
            ``CalibratedDecision`` for choice-only schemas). Cloud providers are rejected:
            they return no probabilities and BaseDecision does not invent them.
        accepted_models: If given, only these values of the request's ``model`` field are
            served; anything else raises ``model_not_found``. By default any value is
            accepted and echoed back.
        max_questions: Maximum questions per request (each costs one forward pass).
            ``None`` disables the limit.
        max_state_chars: Maximum length of the rendered state in characters, checked
            before tokenization. ``None`` disables the limit.
        strict: Reject unknown request/question fields (default). ``False`` ignores them,
            for clients that attach extra metadata.

    The instance holds no per-request state and is safe to share between threads (the
    backend serializes inference).

    Raises:
        TypeError: ``backend`` has no ``predict_batch``.
        SystemOneError: ``unsupported_backend`` for a cloud provider backend.
        ValueError: A limit is not a positive integer.
    """

    def __init__(
        self,
        backend: DecisionBackend,
        *,
        accepted_models: Collection[str] | None = None,
        max_questions: int | None = DEFAULT_MAX_QUESTIONS,
        max_state_chars: int | None = DEFAULT_MAX_STATE_CHARS,
        strict: bool = True,
    ) -> None:
        """Create the service; see the class docstring for the arguments."""
        if not callable(getattr(backend, "predict_batch", None)):
            raise TypeError("backend must provide predict_batch(requests)")
        if isinstance(backend, ProviderDecision):
            raise SystemOneError(
                "unsupported_backend",
                "cloud providers return no probabilities, which SystemOne answers require; "
                "use a local model",
            )
        for name, limit in (("max_questions", max_questions), ("max_state_chars", max_state_chars)):
            if limit is not None and (
                isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        if isinstance(accepted_models, str):
            raise TypeError("accepted_models must be a collection of names, not a string")
        self._backend = backend
        self._accepted = None if accepted_models is None else frozenset(accepted_models)
        self._max_questions = max_questions
        self._max_state_chars = max_state_chars
        self._strict = strict

    def __call__(self, request: Mapping[str, Any]) -> SystemOneResponse:
        """Answer one request body (already parsed JSON) and return the response body.

        Raises:
            SystemOneError: The request is invalid or cannot be answered.
        """
        parsed = self._parse(request)
        results = self._predict(parsed)
        answers: dict[str, ChoiceAnswer | ScoreAnswer | NoulAnswer] = {}
        input_tokens = 0
        for question, result in zip(parsed.questions, results, strict=True):
            answers[question.qid] = _answer(question, result)
            packed = getattr(result, "packed_tokens", None)
            input_tokens += packed if isinstance(packed, int) else 0
        return {
            "model": parsed.model,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }

    # ------------------------------------------------------------------ parsing

    def _parse(self, body: Mapping[str, Any]) -> _Parsed:
        if not isinstance(body, Mapping):
            raise SystemOneError("invalid_request", "request must be a JSON object")
        if self._strict:
            unknown = sorted(str(key) for key in body if key not in _REQUEST_FIELDS)
            if unknown:
                raise SystemOneError(
                    "unknown_field",
                    f"unknown request field(s): {', '.join(unknown)}",
                    field=unknown[0],
                )
        model = body.get("model")
        if not isinstance(model, str) or not model.strip():
            raise SystemOneError(
                "invalid_request",
                "'model' is required and must be a non-empty string",
                field="model",
            )
        if self._accepted is not None and model not in self._accepted:
            raise SystemOneError(
                "model_not_found", f"model {model!r} is not served here", field="model"
            )
        self._reject_media(body)
        if "state" not in body:
            raise SystemOneError("invalid_request", "'state' is required", field="state")
        context = _render(body["state"], "state")
        if self._max_state_chars is not None and len(context) > self._max_state_chars:
            raise SystemOneError(
                "limit_exceeded",
                f"state is {len(context)} characters; the limit is {self._max_state_chars}",
                field="state",
            )
        raw_questions = body.get("questions")
        if not isinstance(raw_questions, Mapping) or not raw_questions:
            raise SystemOneError(
                "invalid_request",
                "'questions' is required and must be a non-empty object",
                field="questions",
            )
        if self._max_questions is not None and len(raw_questions) > self._max_questions:
            raise SystemOneError(
                "limit_exceeded",
                f"{len(raw_questions)} questions; the limit is {self._max_questions}",
                field="questions",
            )
        questions = tuple(
            self._parse_question(qid, raw, context) for qid, raw in raw_questions.items()
        )
        return _Parsed(model=model, questions=questions)

    @staticmethod
    def _reject_media(body: Mapping[str, Any]) -> None:
        for key in ("images", "videos"):
            value = body.get(key)
            if value is None:
                continue
            if not isinstance(value, (list, tuple)):
                raise SystemOneError("invalid_request", f"'{key}' must be a list", field=key)
            if value:
                raise SystemOneError(
                    "unsupported_modality",
                    f"'{key}' are not supported: BaseDecision models read text only",
                    field=key,
                )
        if body.get("media_kwargs"):
            raise SystemOneError(
                "unsupported_modality",
                "'media_kwargs' are not supported: BaseDecision models read text only",
                field="media_kwargs",
            )

    def _parse_question(self, qid: object, raw: object, context: str) -> _Question:
        if not isinstance(qid, str) or not qid.strip():
            raise SystemOneError(
                "invalid_question", "question IDs must be non-blank strings", field="questions"
            )
        field = f"questions.{qid}"
        if not isinstance(raw, Mapping):
            raise SystemOneError("invalid_question", f"{field} must be an object", field=field)
        if self._strict:
            unknown = sorted(str(key) for key in raw if key not in _QUESTION_FIELDS)
            if unknown:
                raise SystemOneError(
                    "unknown_field",
                    f"{field}: unknown field(s): {', '.join(unknown)}",
                    field=f"{field}.{unknown[0]}",
                )
        kind = raw.get("type")
        if kind not in _QUESTION_TYPES:
            raise SystemOneError(
                "invalid_question",
                f"{field}.type must be one of: {', '.join(_QUESTION_TYPES)}",
                field=f"{field}.type",
            )
        instructions = raw.get("instructions")
        text = qid if instructions is None else _render(instructions, f"{field}.instructions")
        if not text.strip():
            text = qid  # an empty instruction falls back to the question ID
        criteria = raw.get("criteria")
        if kind == "choice":
            return self._choice(qid, field, text, criteria, context)
        if kind == "score":
            return self._score(qid, field, text, criteria, context)
        return self._noul(qid, field, text, criteria, context)

    @staticmethod
    def _build(
        field: str, context: str, text: str, options: tuple[Option, ...], kind: QuestionType
    ) -> Request:
        try:
            return Request(context, text, options, kind)
        except InputError as exc:  # e.g. identical descriptions
            raise SystemOneError(
                "invalid_criteria", f"{field}: {exc}", field=f"{field}.criteria"
            ) from exc

    def _choice(self, qid: str, field: str, text: str, criteria: object, context: str) -> _Question:
        where = f"{field}.criteria"
        if not isinstance(criteria, Mapping) or not criteria:
            raise SystemOneError(
                "invalid_criteria",
                f"{where} must be a non-empty object mapping option ID to description",
                field=where,
            )
        texts: dict[str, str] = {}
        for option_id, description in criteria.items():
            if not isinstance(option_id, str) or not option_id.strip():
                raise SystemOneError(
                    "invalid_criteria",
                    f"{where}: option IDs must be non-blank strings",
                    field=where,
                )
            texts[option_id] = _option_text(description, option_id, f"{where}.{option_id}")
        _check_option_count(len(texts), where)
        _check_distinct(texts.values(), where)
        # Options are packed in sorted ID order so that the JSON key order sent by a
        # client can never change the decision; the response keeps the request order.
        options = tuple(Option(option_id, texts[option_id]) for option_id in sorted(texts))
        return _Question(
            qid, "choice", self._build(field, context, text, options, "choice"), tuple(texts)
        )

    def _score(self, qid: str, field: str, text: str, criteria: object, context: str) -> _Question:
        where = f"{field}.criteria"
        if not isinstance(criteria, (list, tuple)) or not criteria:
            raise SystemOneError(
                "invalid_criteria",
                f"{where} must be a non-empty list of level descriptions",
                field=where,
            )
        levels = [str(index) for index in range(len(criteria))]
        texts = [
            _option_text(description, level, f"{where}[{level}]")
            for level, description in zip(levels, criteria, strict=True)
        ]
        _check_option_count(len(texts), where)
        _check_distinct(texts, where)
        options = tuple(Option(level, label) for level, label in zip(levels, texts, strict=True))
        return _Question(
            qid,
            "score",
            self._build(field, context, text, options, "score"),
            tuple(levels),
            tuple(criteria),
        )

    def _noul(self, qid: str, field: str, text: str, criteria: object, context: str) -> _Question:
        where = f"{field}.criteria"
        labels = {"false": "false", "true": "true"}
        if criteria not in (None, {}):
            if not isinstance(criteria, Mapping) or set(criteria) - {"true", "false"}:
                raise SystemOneError(
                    "invalid_criteria",
                    f"{where} for a noul question may only describe 'true' and 'false'",
                    field=where,
                )
            for key in labels:
                if key in criteria:
                    labels[key] = _option_text(criteria[key], key, f"{where}.{key}")
        options = (Option("false", labels["false"]), Option("true", labels["true"]))
        return _Question(
            qid, "noul", self._build(field, context, text, options, "noul"), ("false", "true")
        )

    # ---------------------------------------------------------------- inference

    def _predict(self, parsed: _Parsed) -> Sequence[Result]:
        requests = [question.request for question in parsed.questions]
        try:
            results = self._backend.predict_batch(requests)
        except InputError as exc:  # includes ContextLengthError and CalibrationError
            raise self._explain(parsed, exc) from exc
        if len(results) != len(requests):
            raise RuntimeError("backend returned an incorrect number of results")
        for question, result in zip(parsed.questions, results, strict=True):
            probabilities = getattr(result, "probabilities", None)
            if not isinstance(probabilities, Mapping) or set(probabilities) != set(question.order):
                raise SystemOneError(
                    "unsupported_backend",
                    "the backend did not return per-option probabilities",
                    field=f"questions.{question.qid}",
                )
        return results

    def _explain(self, parsed: _Parsed, exc: InputError) -> SystemOneError:
        """Turn a backend input error into a SystemOneError attributed to a question."""
        count_tokens = getattr(self._backend, "count_tokens", None)
        if callable(count_tokens):
            for question in parsed.questions:
                try:
                    count_tokens(question.request)
                except ContextLengthError as inner:
                    return _length_error(question.qid, inner)
                except InputError as inner:
                    where = f"questions.{question.qid}"
                    return SystemOneError("invalid_criteria", f"{where}: {inner}", field=where)
        if isinstance(exc, ContextLengthError):
            return SystemOneError(
                "context_length_exceeded",
                f"input requires {exc.required_tokens} tokens; maximum is {exc.maximum_tokens}",
                required_tokens=exc.required_tokens,
                maximum_tokens=exc.maximum_tokens,
            )
        return SystemOneError("invalid_request", str(exc))


def _length_error(qid: str, exc: ContextLengthError) -> SystemOneError:
    where = f"questions.{qid}"
    return SystemOneError(
        "context_length_exceeded",
        f"{where}: complete input requires {exc.required_tokens} tokens; maximum is "
        f"{exc.maximum_tokens}. Nothing was truncated; shorten the state or the criteria.",
        field=where,
        required_tokens=exc.required_tokens,
        maximum_tokens=exc.maximum_tokens,
    )


def _check_option_count(count: int, where: str) -> None:
    if count < 2:
        raise SystemOneError("invalid_criteria", f"{where} needs at least two options", field=where)
    if count > _MAX_OPTIONS:
        raise SystemOneError(
            "limit_exceeded",
            f"{where} has {count} options; the limit is {_MAX_OPTIONS}",
            field=where,
        )


def _check_distinct(texts: Collection[str], where: str) -> None:
    if len(set(texts)) != len(texts):
        raise SystemOneError(
            "invalid_criteria",
            f"{where}: options must have distinct descriptions (the model reads the "
            "description; the ID is only used to report the answer)",
            field=where,
        )


def _round(value: float) -> float:
    return round(float(value), _ROUND_DIGITS)


def _answer(question: _Question, result: Result) -> ChoiceAnswer | ScoreAnswer | NoulAnswer:
    """Build one wire-format answer from a model result (probabilities already checked)."""
    probabilities = result.probabilities
    if question.kind == "noul":
        return {"type": "noul", "noul": _round(probabilities["true"])}
    rounded = {option_id: _round(probabilities[option_id]) for option_id in question.order}
    if question.kind == "choice":
        return {
            "type": "choice",
            "choice": result.option_id,
            "confidence": _round(probabilities[result.option_id]),
            "probabilities": rounded,
        }
    expected = sum(index * probabilities[level] for index, level in enumerate(question.order))
    return {
        "type": "score",
        "score": _round(expected),
        "confidence": _round(max(probabilities[level] for level in question.order)),
        "legend": dict(zip(question.order, question.legend, strict=True)),
        "probabilities": rounded,
    }


def systemone(
    backend: DecisionBackend,
    request: Mapping[str, Any],
    *,
    accepted_models: Collection[str] | None = None,
    max_questions: int | None = DEFAULT_MAX_QUESTIONS,
    max_state_chars: int | None = DEFAULT_MAX_STATE_CHARS,
    strict: bool = True,
) -> SystemOneResponse:
    """Answer one Jev/SystemOne request body; see :class:`SystemOne` for the arguments.

    Equivalent to ``SystemOne(backend, ...)(request)``. When serving many requests, create
    one :class:`SystemOne` and reuse it.
    """
    return SystemOne(
        backend,
        accepted_models=accepted_models,
        max_questions=max_questions,
        max_state_chars=max_state_chars,
        strict=strict,
    )(request)
