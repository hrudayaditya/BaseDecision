# Unreleased

- Jev/SystemOne API support: `SystemOne` / `basedecision.systemone.systemone` answer `POST /v1/systemone`
  request bodies (choice, score and noul questions) with local models; strict validation, stable error
  codes (`SystemOneError`), no silent truncation, images/videos rejected, cloud providers refused.
- `examples/systemone_quickstart.py` and `examples/systemone_server.py` (reference HTTP server; auto device
  selection), `docs/SYSTEMONE.md`.
- No change to the hash-pinned inference modules (`client.py`, `packing.py`, `_model.py`, `types.py`).

## Fixed

- OpenAI backend raised a raw `TypeError` on any response containing a reasoning item (every
  reasoning model returns one). Provider responses are now normalized to plain data and handled by one
  closed, total policy (`basedecision/_wire.py`): reasoning items and Anthropic thinking blocks are
  ignored, tool calls and unknown types are rejected with `unexpected_output_block`, and malformed
  responses can only ever produce a `ProviderResponseError`.
- Unpickling a `ProviderError` doubled its message prefix.

## Added

- Error codes `output_budget_exhausted` (reasoning consumed `max_output_tokens`) and `content_filtered`;
  `ProviderError.detail` (safe API identifier); `usage['reasoning_tokens']`;
  `from_provider('openai', ..., reasoning_effort=...)`.
- Tests that enforce the invariant "any JSON in, a decision or a `ProviderResponseError` out", and that
  enumerate the installed SDKs' own output-item unions so SDK upgrades are covered automatically.

# 0.1.0rc5

- Explicit experimental CPU fast backend with instance-local backbone and head attention.
- Raw output backend/precision metadata, strict runtime-version checks, unchanged GPU calibration contract.
- Bounded hardware verification command; CPU calibration deferred.

# 0.1.0rc4

- Python `load` and named-question `decide` helpers.
- Reviewed, explicit scalar calibration profiles with raw probability retention.
- Checkpoint/inference compatibility checks, scope errors, profile inspection.
- Python examples and API/calibration documentation.
- Original inference modules unchanged; CLINC remains blocked.

# Changes

## 0.1.0rc3
- Optional OpenAI Responses and Anthropic Messages decision backends.
- Explicit Hub snapshot loading; no remote Python execution.
- Sanitized provider errors, strict output validation, bounded SDK retries.
- Documentation, example, CI checks and repository security guidance.
- Local model forward, packing and batch planner retained from RC2.

## 0.1.0rc2
- Single-request default; resumable full-suite SDK parity runner.
- 8751 decisions passed with exact logits and labels on the supplied GPU report.
