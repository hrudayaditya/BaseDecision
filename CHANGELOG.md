# Unreleased

- Jev/SystemOne API support: `SystemOne` / `basedecision.systemone.systemone` answer `POST /v1/systemone`
  request bodies (choice, score and noul questions) with local models; strict validation, stable error
  codes (`SystemOneError`), no silent truncation, images/videos rejected, cloud providers refused.
- `examples/systemone_quickstart.py` and `examples/systemone_server.py` (reference HTTP server; auto device
  selection), `docs/SYSTEMONE.md`.
- Input hardening (changes `packing.py`, `types.py` and one line of `client.py`; `_model.py`, the network,
  is untouched): see Fixed. For ordinary inputs the packed tokens are identical to the earlier release,
  verified by a 450-case golden corpus (`tests/data/packing_golden.json`, including the exact over-limit
  error counts) and by 120 real-model predictions with bit-identical logits. The `sdk_contract` hashes
  in `calibration_v1.json` were refreshed accordingly; the GPU quality assessment behind the calibration
  profiles was not repeated for this change.

## Fixed

- Oversized input no longer costs time and memory proportional to its size. The window is fixed at 8,192
  tokens and nothing is truncated, so an input that cannot fit is now rejected after a bounded amount of
  work (a 100 MB text: 46 s and 6.7 GiB before, 0.02 s now, and the model lock is no longer held while
  tokenizing it). Inputs up to 262,144 characters still report their exact token count; larger ones report
  "at least N tokens" (`ContextLengthError.at_least`; `SystemOneError.details['at_least']`).
- Special-token strings in user text (`[SEP]`, `[MASK]`, `[CLS]`, `[PAD]`, `<|endoftext|>`, ...) were
  tokenized into the real structural tokens, so untrusted text could forge the packed format. They are now
  ordinary text in the context, the question and every option.
- A lone surrogate in a context, question or option raised a raw `TypeError` from the tokenizer; it is now
  an `InputError` naming the part that contains it.
- `choose(options={...})` with a dict (or a set) silently used the keys as labels and dropped the
  descriptions (or ordered the options randomly). Mappings and sets are now rejected with a message that
  points to `Option(id, text)`.

- First run without an NVIDIA GPU: `load(path)`, the README quickstart and the examples crashed with
  `CUDA is unavailable`. `load()` now chooses CUDA/BF16 when a GPU with native BF16 is present and CPU/FP32
  otherwise (`resolve_device()`); explicit `device`/`precision` are never overridden. The inference modules
  keep their CUDA/BF16 defaults, so `BaseDecision.from_pretrained` stays strict.

- OpenAI backend raised a raw `TypeError` on any response containing a reasoning item (every
  reasoning model returns one). Provider responses are now normalized to plain data and handled by one
  closed, total policy (`basedecision/_wire.py`): reasoning items and Anthropic thinking blocks are
  ignored, tool calls and unknown types are rejected with `unexpected_output_block`, and malformed
  responses can only ever produce a `ProviderResponseError`.
- Unpickling a `ProviderError` doubled its message prefix.
- Provider errors no longer keep the original SDK exception attached as `__context__` (it holds the HTTP
  request including the API key header and the request text); the same for `parse_selection` and the
  SystemOne `parse_request_body`/state rendering, whose JSON errors hold the whole document. Sanitized
  errors are now raised outside the `except` block. The duplicate-key message no longer echoes the key.
- `ContextLengthError` could not be unpickled, so it surfaced as `BrokenProcessPool` (and killed the pool)
  when raised in a `ProcessPoolExecutor`/`multiprocessing` worker. Fixed from `errors.py` via `copyreg`,
  leaving the byte-pinned `types.py` untouched.

## Added

- `load_from_hub(repo_id, revision=...)`: explicit Hub download with the same machine-appropriate defaults;
  `resolve_device()`; `--device`/`--precision` flags in all example scripts; a Hardware section in the README
  with measured CPU timings and memory.

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
