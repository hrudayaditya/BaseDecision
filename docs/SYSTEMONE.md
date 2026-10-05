# Jev / SystemOne API

BaseDecision can answer requests in the **Jev/SystemOne** wire format: a `state` plus a schema of
typed `questions` in, a probability for every allowed option out. The same request body that a
Jev/SystemOne service accepts (`POST /v1/systemone`) works with a local BaseDecision model:

```python
from basedecision import SystemOne, load

service = SystemOne(load("/path/to/model"))        # or load(..., device="cpu", precision="fp32")
response = service({
    "model": "basedecision",
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle the message?",
            "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
        },
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "outage": {"type": "noul", "instructions": "Is a service down?"},
    },
})
```

```json
{
  "model": "basedecision",
  "answers": {
    "department": {"type": "choice", "choice": "technical", "confidence": 0.9612,
                   "probabilities": {"billing": 0.0388, "technical": 0.9612}},
    "urgency": {"type": "score", "score": 2.0, "confidence": 1.0,
                "legend": {"0": "Can wait", "1": "This week", "2": "Today"},
                "probabilities": {"0": 0.0, "1": 0.0, "2": 1.0}},
    "outage": {"type": "noul", "noul": 0.9874}
  },
  "usage": {"input_tokens": 102, "output_tokens": 0}
}
```

`SystemOne` is a pure `dict -> dict` function: no web framework, no network access, and it does not
import Torch. `systemone(backend, request, ...)` is the one-shot form. To serve it over HTTP see
[Serving it](#serving-it).

**Compatibility basis.** The format follows the request/response documented in the
[Cloudflare/clef](https://huggingface.co/Cloudflare/clef) model card and implemented by `systemone()`
in its `joint_schema_model.py` (Hub revision `2f3de3dd`, which states "the Clef API is fully
compatible with Jev and SystemOne"). It has **not** been tested against a hosted Jev service. Where
BaseDecision cannot behave identically it refuses or documents the difference; see
[Differences](#differences-from-the-reference-implementation).

## Request

| Field | Type | Notes |
|---|---|---|
| `model` | string, required | Echoed in the response. Not used for routing; use `accepted_models` to restrict it. |
| `state` | string or any JSON, required | A string is used verbatim; anything else is rendered as compact JSON with sorted keys, so key order never matters. |
| `questions` | object, required, non-empty | Question ID → question. Up to 64 by default. |
| `id` | any | Accepted and ignored. |
| `images`, `videos`, `media_kwargs` | — | Accepted only when empty/absent. Anything else is rejected (`unsupported_modality`); BaseDecision models read text only. |

A question has `type` (`"choice"`, `"score"` or `"noul"`), an optional `instructions`, and `criteria`:

| `type` | `criteria` | Answer |
|---|---|---|
| `choice` | object mapping option ID → description (2–255 options) | the most likely option and the probability of each |
| `score` | array of level descriptions, indexed from 0 (2–255 levels) | the expected level index, plus per-level probabilities |
| `noul` | optional object with `"true"` and/or `"false"` descriptions | the probability that the statement is true |

`instructions` defaults to the question ID when missing, `null`, or blank. Unknown fields are rejected
by default (`strict=True`) so that a typo such as `instruction` cannot silently change a question.

## Response

```
{ "model": str,
  "answers": { <question id>: <answer>, ... },     # same order as the request
  "usage":   { "input_tokens": int, "output_tokens": 0 } }
```

| Answer | Fields |
|---|---|
| choice | `type`, `choice` (option ID), `confidence` (its probability), `probabilities` (option ID → p, in the order the request listed the options) |
| score | `type`, `score` (Σ index·p), `confidence` (largest p), `legend` (index → your description, as given), `probabilities` (index → p) |
| noul | `type`, `noul` (p of true) |

All numbers are rounded to 4 decimal places, as in the reference. Probabilities are the model's raw
softmax: they are **not calibrated** and tend to be over-confident (see `MODEL_CARD.md`). Wrapping a
CUDA/BF16 model in a reviewed `CalibratedDecision` profile works for `choice` questions only; `noul`
and `score` questions are then refused (`invalid_request`).

`usage.input_tokens` is the sum of the packed tokens of every question's forward pass.
`output_tokens` is always 0.

## How a request is mapped

Each question becomes one native BaseDecision `Request`, all answered in a single `predict_batch`
call, so the state is tokenized once and everything is validated before any inference runs.

* **Choice.** `criteria` becomes `Option(id, description)`. The model reads the *description* only;
  the ID is used to report the answer. A blank, missing or non-string description falls back to the
  ID or its compact JSON text. Descriptions must be distinct. Options are packed in sorted-ID order,
  so the JSON key order sent by a client can never change a decision; the response lists
  `probabilities` in the request's order.
* **Score.** Level *i* becomes `Option(str(i), description)`; `score` is the expected index.
* **Noul.** The options are `false` and `true`. Without `criteria` they use the same wording as the
  native `check()` API; a custom description is passed through. The model is sensitive to option
  wording, so prefer the default unless you have measured otherwise.

## Differences from the reference implementation

| Topic | Reference (Clef) | BaseDecision |
|---|---|---|
| Questions | decided jointly in one forward pass | each question is an independent pass (cost grows with the number of questions) |
| Modalities | text, images, video | text only; images/videos are **rejected, not ignored** |
| Too-long input | state silently truncated to `max_length` | **never truncated**: `context_length_exceeded` with `required_tokens`/`maximum_tokens` (window: 8,192) |
| What the model reads per option | `{"option_id", "description"}` JSON | the description only |
| `noul` default wording | "The proposition is true or the answer is yes." | `true` / `false` (the validated `check()` wording) |
| Unknown `noul` criteria keys / unknown fields | ignored | rejected (`strict=False` ignores unknown fields) |
| `usage.input_tokens` | tokens of the joint sequence | sum over all questions |
| Failure type | `ValueError` | `SystemOneError` with a stable `code` and `field` |

Not supported: cloud providers as a backend (`BaseDecision.from_provider`). They return no
probabilities, which this API requires, and BaseDecision does not invent them; constructing
`SystemOne` with one raises `unsupported_backend` before any request is sent.

## Errors

Every rejected request raises `SystemOneError`, a subclass of `InputError` (and `ValueError`), with:
`code` (stable), `field` (dotted path such as `questions.urgency.criteria`), `details`
(`required_tokens`/`maximum_tokens` for length errors) and `to_dict()` →
`{"error": {"code", "message", "field"?, ...}}`. Messages never contain the request `state`. The error
body shape is a BaseDecision convention, not part of the Jev/SystemOne specification.

| `code` | Meaning | Suggested HTTP status |
|---|---|---|
| `invalid_request` | not valid JSON/object; missing or malformed `model`, `state`, `questions`; unserializable state; backend scope error | 400 |
| `unknown_field` | a field the API does not define (strict mode) | 400 |
| `model_not_found` | `model` is not in `accepted_models` | 404 |
| `limit_exceeded` | too many questions/options, or `state` over `max_state_chars` | 413 |
| `context_length_exceeded` | the complete input does not fit the model window | 413 |
| `unsupported_modality` | images, videos or `media_kwargs` supplied | 422 |
| `invalid_question` | bad question ID, shape or `type` | 422 |
| `invalid_criteria` | bad, missing or duplicate-description options | 422 |
| `unsupported_backend` | the backend cannot return probabilities | 501 |

Unexpected backend failures (for example `FloatingPointError` on non-finite outputs) propagate
unchanged; they are bugs or environment problems, not bad requests.

## Limits and options

`SystemOne(backend, *, accepted_models=None, max_questions=64, max_state_chars=1_000_000, strict=True)`

* `max_questions` bounds the work per request: every question is a forward pass.
* `max_state_chars` rejects an oversized state *before* tokenization, so the cost of refusing it is
  small. The tokenizer's longest token is 512 characters, so nothing above 8192 × 512 characters
  could ever fit; the default is far below that and far above any realistic state.
* `parse_request_body(bytes_or_str)` parses raw JSON strictly: no `NaN`/`Infinity`, no duplicate keys
  (a repeated question ID would otherwise silently drop a question), valid UTF-8, an object at the top.

**Cost.** A three-question request over ~100 tokens took about 0.2 s end to end on a 10-core Apple CPU
(FP32, measured with the reference server below). A single question over an 8,191-token state took
about 31 s on the same CPU. GPU numbers were not measured for this feature. Inference is serialized by
the model's lock, so concurrent requests queue.

## Serving it

`examples/systemone_server.py` is a dependency-free reference server for `POST /v1/systemone`:

```bash
export SYSTEMONE_API_KEY=...        # optional on loopback, required elsewhere
python examples/systemone_server.py --model /path/to/model --api-key-env SYSTEMONE_API_KEY
curl -s localhost:8080/v1/systemone -H 'Content-Type: application/json' \
     -H "Authorization: Bearer $SYSTEMONE_API_KEY" -d @request.json
```

It binds to `127.0.0.1` and refuses a non-loopback address without an API key; caps the request body
(1 MiB); requires `Content-Type: application/json` and `Content-Length`; applies a socket timeout;
checks the bearer token (constant time) before reading the body; never logs bodies, answers or
exception messages; and returns JSON errors only. It is a **reference, not a hardened production
server** (see `SECURITY.md`): put it behind a gateway that provides TLS, rate limits and quotas. The
device is chosen automatically (CUDA/BF16, else CPU/FP32).

To embed the adapter in an existing framework, parse with `parse_request_body`, call the service, and
map `SystemOneError.code` to a status (table above). FastAPI example (inference blocks, so it is kept
off the event loop):

```python
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from basedecision import SystemOne, SystemOneError, load
from basedecision.systemone import parse_request_body

STATUS = {"model_not_found": 404, "limit_exceeded": 413, "context_length_exceeded": 413,
          "unsupported_modality": 422, "invalid_question": 422, "invalid_criteria": 422,
          "unsupported_backend": 501}

app = FastAPI()
service = SystemOne(load("/path/to/model"))


@app.post("/v1/systemone")
async def systemone_endpoint(request: Request):
    try:
        body = parse_request_body(await request.body())
        return await run_in_threadpool(service, body)
    except SystemOneError as error:
        return JSONResponse(error.to_dict(), status_code=STATUS.get(error.code, 400))
```

## Testing

`tests/test_systemone.py` (71 scripted tests; 100% line and branch coverage of
`basedecision/systemone.py`) covers the wire format against hand-computed probabilities, every validation path, key-order
invariance, backend-failure mapping, thread safety, and a seeded fuzz test that corrupts valid
requests 4,000 times and requires that only a valid response or a `SystemOneError` ever results.
`tests/test_systemone_server.py` runs the reference server on a loopback port. Four further tests
check the adapter against the native `predict`/`check`/`score` API on a real checkpoint:

```bash
BASEDECISION_TEST_MODEL=/path/to/model python -m unittest tests.test_systemone -v
```
