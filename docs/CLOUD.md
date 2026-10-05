# OpenAI and Anthropic backends

```python
from basedecision import BaseDecision

with BaseDecision.from_provider('openai', 'YOUR_OPENAI_MODEL_ID') as client:
    result = client.choose(context='Please refund this purchase.',
                           question='What does the customer request?',
                           options=['Refund request', 'Delivery status', 'Other request'])

with BaseDecision.from_provider('anthropic', 'YOUR_ANTHROPIC_MODEL_ID') as client:
    result = client.check(context='The account is active.', question='Is the account active?')
```

Install with `pip install ".[providers]"` (or `.[openai]` / `.[anthropic]`); no Torch is needed. Set
`OPENAI_API_KEY` or `ANTHROPIC_API_KEY` using your shell's secure secret setup. Do not put
credentials in source files or chat. Choose a model that supports the provider's structured JSON
output API.

Cloud calls send the entire supplied context/question/options to that provider. They never occur as
an automatic fallback from local inference. Official clients are reused across requests and closed by
the context manager. OpenAI uses Responses structured output; Anthropic uses Messages
`output_config.format`. Provider model availability and schema support must be verified for your
account.

Cloud results contain `probabilities=None`, `raw_logits=None`, and
`calibration_status='not_available_from_provider'`. We do not turn a selected label or generated
confidence into a probability distribution. `usage` reports provider input/output token counts when
available. Provider limits differ from the local model; the byte-size guard does not claim an exact
provider token count.

**Reasoning models.** Responses that contain reasoning items (OpenAI) or thinking blocks (Anthropic)
are supported: that deliberation is ignored, only the structured decision is used, and OpenAI's
hidden reasoning tokens are reported as `usage['reasoning_tokens']` (they are billed and count
against `max_output_tokens`). If a model spends its whole output budget before answering you get the
error code `output_budget_exhausted`: raise `max_output_tokens` or, for OpenAI reasoning models,
lower the effort with `from_provider('openai', model, reasoning_effort='low')` (sent as
`reasoning.effort`; the option is only valid for reasoning models). Tool calls and unknown item types
are rejected (`unexpected_output_block`), never guessed at.

**Batches.** Cloud `predict_batch` is sequential, not the providers' asynchronous Batch API. All
request schemas are checked before its first API call. Earlier successful calls may have been billed
if a later call fails. There are no implicit fallback calls or repeated sampling until a preferred
answer appears.

**Errors.** Provider errors expose `code`, `retryable`, and `status_code` without returning upstream
bodies. Codes include authentication/permission, rate limit, timeout, request rejection, refusal,
incomplete output, `output_budget_exhausted`, `content_filtered`, `unexpected_output_block` and
invalid selection. An optional `detail` names the offending item type (an API identifier, never
response text). Invalid or refused responses are errors, not a guessed label, NONE, or false.

```python
from basedecision import ProviderError

try:
    result = client.predict(request)
except ProviderError as error:
    print(error.code, error.retryable)  # sanitized; avoid printing provider internals
```

The official SDK owns retries (default 2, configurable 0-5). There is no second retry loop in
BaseDecision. Timeout is a request timeout, not a total batch or retry wall-clock deadline. Retries
can increase elapsed time and request costs.

**Status.** Both backends passed official-SDK HTTP transport tests (OpenAI 2.54.0, Anthropic
0.125.0) with mocked servers. They have not been run against the live services: do one real call per
provider with your own key before relying on them.
