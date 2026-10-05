## RC5 experimental CPU fast mode

```python
from basedecision import load
model = load('/path/to/model', backend='cpu_fast')
result = model.choose(context='Please refund my purchase.',
                      question='What does the customer request?',
                      options=['Refund', 'Delivery status', 'Change address'])
print(result.to_dict())
print(model.backend_info())
```

Requires Torch 2.9.1 and Transformers 4.57.6. This is explicit opt-in; the default loader and GPU calibration behavior are preserved. For the portable reference CPU path use `load(path, device='cpu', precision='fp32')`. GPU calibration profiles remain workload-specific. CPU calibration is not enabled.

See [CPU support and limitations](docs/CPU.md). RC5 needs the short hardware verification below before deployment; no model inference was available in the package build environment.

# BaseDecision

Typed decisions from a local Laya/ModernBERT checkpoint, OpenAI, or Anthropic.
Choose a label, answer a boolean question, or select an ordered score level through
one Python interface. No label definitions are required; optional descriptions
can be supplied using `Option(id, text)`.

**Release candidate 0.1.0rc4.** Local inference is grounded in RC2's 8,751-decision
regression with identical logits and labels. Cloud backends passed official-SDK HTTP transport tests (OpenAI 2.54.0,
Anthropic 0.125.0); live provider acceptance is still pending. This is a decision SDK, not a
causal language model or a general chat/agent server.

## Installation

Install the included wheel without changing your working ML dependencies:

```bash
python -m pip install --no-deps dist/basedecision-0.1.0rc4-py3-none-any.whl
```

From a source checkout, choose the dependencies you need:

```bash
python -m pip install '.[runtime,hub]'   # local model and explicit Hub download
python -m pip install '.[providers]'    # OpenAI and Anthropic only; no Torch needed
```

These commands do not imply a published PyPI package or model repository exists.
Dependency ranges are compatibility targets. The tested local environment was
Python 3.12, Torch 2.9.1+cu126, Transformers 4.57.6, A100-SXM4-80GB, BF16 autocast.

## Quick Python integration

```python
from basedecision import load, decide

model = load('/path/to/exported/model')
answers = decide(model, context='Please refund the duplicate payment.', questions={
    'intent': {'kind': 'choice', 'question': 'What does the customer request?',
               'options': ['Refund request', 'Delivery status', 'Other request']},
    'refund_requested': {'kind': 'noul', 'question': 'Is a refund explicitly requested?'},
})
print(answers['intent'].answer)
```

Each question is evaluated independently. No definitions are required for plain labels.
The existing `choose`, `check`, `score` and batch interfaces remain available.
See [Python API](docs/API.md), [calibration coverage](docs/CALIBRATION.md), and the
runnable `examples/python_quickstart.py`.

## Reviewed calibration

Calibration is opt-in, with raw probabilities as the default. `CalibratedDecision`
wraps the same loaded model for an explicit assessed workload. It returns both raw
and calibrated probabilities while preserving the selected answer. The bundled
SGD profiles have aggregate improvements and documented short/two-option regressions.
VAST remains raw by default; CLINC is blocked by its uncertainty assessment.
These profiles do not apply to arbitrary refund questions or all 8K requests.

```python
from basedecision import CalibratedDecision, calibration_profiles
print(calibration_profiles())
# Only for correctly formatted SGD service-intent requests:
sgd = CalibratedDecision(model, profile='sgd_schema')
# result = sgd.predict(sgd_request)
```

RC4 preserves the measured RC3 inference modules byte-for-byte. Its additions are
Python convenience functions and a post-processing wrapper. No additional GPU
quality evaluation is claimed for this packaging change.

## Local inference

```python
from basedecision import BaseDecision

model = BaseDecision.from_pretrained('/path/to/exported/model')
result = model.choose(
    context='Please refund this purchase.',
    question='What does the customer request?',
    options=['Refund request', 'Delivery status', 'Other request'],
)
print(result.answer)
print(result.probabilities)  # raw, uncalibrated softmax
```

For a published checkpoint, download explicitly:

```python
model = BaseDecision.from_hub('YOUR_NAMESPACE/YOUR_MODEL', revision='YOUR_REVISION')
```

Replace the placeholders with an actual repository and revision. Authentication
uses standard Hugging Face configuration; no token is embedded in code. Use
`local_files_only=True` for cached-only loading. No remote Python is executed.

## OpenAI and Anthropic

Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` using your shell's secure secret setup.
Do not put credentials in source files or chat. Choose a model that supports the
provider's structured JSON output API.

```python
from basedecision import BaseDecision

with BaseDecision.from_provider('openai', 'YOUR_OPENAI_MODEL_ID') as model:
    result = model.choose(context='Please refund this purchase.',
                          question='What does the customer request?',
                          options=['Refund request', 'Delivery status', 'Other request'])

with BaseDecision.from_provider('anthropic', 'YOUR_ANTHROPIC_MODEL_ID') as model:
    result = model.check(context='The account is active.', question='Is the account active?')
```

Cloud calls send the entire supplied context/question/options to that provider.
They never occur as an automatic fallback from local inference. Official clients
are reused across requests and closed by the context manager. OpenAI uses
Responses structured output; Anthropic uses Messages `output_config.format`.
Provider model availability and schema support must be verified for your account.

Cloud results contain `probabilities=None`, `raw_logits=None`, and
`calibration_status='not_available_from_provider'`. We do not turn a selected
label or generated confidence into a probability distribution. `usage` reports
provider input/output token counts when available. Provider limits differ from
the local model; the byte-size guard does not claim an exact provider token count.

## Boolean, scores, and batching

```python
from basedecision import Request, Option

decision = model.check(context='The account is active.', question='Is the account active?')
# decision.answer is a Python bool

rating = model.score(context='I am satisfied.', question='Rate satisfaction.',
                     levels=['Dissatisfied', 'Neutral', 'Satisfied'], values=[1, 2, 3])
# selected_value is the chosen level; expected_value is available only locally.

requests = [Request('Please refund this purchase.', 'What is requested?',
                    (Option('refund', 'Refund request'), Option('status', 'Delivery status')))]
results = model.predict_batch(requests)
```

Use an open client for these calls. Local batch size defaults to 1. For offline
throughput on similar-length inputs, explicitly set `max_batch_size=8` and an
appropriate `max_batch_tokens` when loading the model. Near-8K batch=4 was the
largest measured configuration in the supplied 32768-token benchmark. No universal
speedup or GPU memory guarantee is claimed. `predict_iter` bounds the buffered
request count; `predict_batch` materializes the submitted iterable.

Cloud `predict_batch` is sequential, not the providers' asynchronous Batch API.
All request schemas are checked before its first API call. Earlier successful
calls may have been billed if a later call fails. There are no implicit fallback
calls or repeated sampling until a preferred answer appears.

## Jev / SystemOne API

Requests in the Jev/SystemOne wire format (`POST /v1/systemone`: a `state` and typed `questions`,
answered with per-option probabilities) can be answered by a local model:

```python
from basedecision import SystemOne, load

service = SystemOne(load('/path/to/exported/model'))  # no GPU: load(path, device='cpu', precision='fp32')
response = service({'model': 'basedecision',
                    'state': 'Our checkout started returning errors and orders are blocked.',
                    'questions': {
                        'department': {'type': 'choice', 'instructions': 'Which team should handle it?',
                                       'criteria': {'billing': 'Payments or invoices',
                                                    'technical': 'Bugs or outages'}},
                        'outage': {'type': 'noul', 'instructions': 'Is a service down?'}}})
print(response['answers'])
```

Text only, one forward pass per question, no silent truncation, cloud providers not supported (they
return no probabilities). `examples/systemone_server.py` serves the endpoint over HTTP (a reference,
not a hardened server). See [Jev / SystemOne API](docs/SYSTEMONE.md) for the exact mapping, the
differences from the reference implementation, and the error codes.

## Errors and limits

- Local inputs must fit 8192 total packed tokens (or a smaller checkpoint limit),
  including the question, all 2–255 options, delimiters and source text.
- No input or option is silently truncated. `ContextLengthError` includes
  `required_tokens` and `maximum_tokens`; `count_tokens(request)` performs exact
  local packing and rejects oversized requests.
- `InputError` means a schema/configuration problem. Context is a string; serialize
  structured inputs explicitly. Option IDs and descriptions must be distinct.
- Provider errors expose `code`, `retryable`, and `status_code` without returning
  upstream bodies. Codes include authentication/permission, rate limit, timeout,
  request rejection, refusal, incomplete output and invalid selection.
- The official SDK owns retries (default 2, configurable 0–5). There is no second
  retry loop in BaseDecision. Timeout is a request timeout, not a total batch or
  retry wall-clock deadline. Retries can increase elapsed time and request costs.
- Invalid/refused responses are errors, not a guessed label, NONE, or false.
- Local action-head output is not interpreted as abstention or authorization.

```python
from basedecision import ContextLengthError
from basedecision.errors import ProviderError

try:
    result = model.predict(requests[0])
except ContextLengthError as exc:
    print(exc.required_tokens, exc.maximum_tokens)
except ProviderError as exc:
    print(exc.code, exc.retryable)  # sanitized; avoid printing provider internals
```

## Verification and release

```bash
python -m unittest discover -s tests -v
```

Read `MODEL_CARD.md` for measured quality/limitations, `BENCHMARKS.md` for batching,
and `SECURITY.md` for deployment boundaries and repository settings. `tools/`
contains the existing HPC regression/benchmark runners; they require the research
checkout and datasets, while the installed SDK does not.

Before public publication: run the supplied CI audits, perform one live smoke
call per provider with your own keys, verify upstream weight/data licensing,
choose repository/package namespaces, and review the model card. Do not claim
that this RC supports `AutoModelForCausalLM`, `.generate()`, chat completion
endpoints, arbitrary Hugging Face architectures, or reasoning equivalent to
Qwen/DeepSeek/Llama. It serves this decision architecture and optional cloud APIs.
