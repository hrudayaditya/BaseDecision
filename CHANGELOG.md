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

## Changed

- The README now installs from PyPI (`pip install "basedecision[runtime]"`, `[runtime,hub]`, `[providers]`;
  `pip install --no-deps basedecision` to leave an existing Torch alone) and says "from a clone" where a step
  needs the repository (`examples/`, the tests, `pip install ".[runtime]"`). The link to `MODEL_CARD.md` is
  gone because that file is no longer in the repository. Package author is now "Hrudayaditya Jallu", and the
  README has a License section.
- The README gains a "How it performs" section: the benchmark image (`docs/assets/benchmarks.png`, with a
  descriptive alt text) and a short write-up (highest average, ahead on five of eight benchmarks, behind
  on two, one tie). The two baseline names that were cut off in the source figure are still placeholders
  (`GLiNER2.5-Decid…`, `Decision 1.0 Kai…`) and must be filled in before the README is published.
- Typing and documentation. The whole package is now typed: `mypy --strict` passes (configuration in
  `pyproject.toml`, run `mypy`), including on machines where Torch, Transformers and the provider SDKs are
  not installed, and every public class, function and method has a docstring. `tests/test_api_quality.py`
  enforces both. For users of the typed API: `load()` / `load_from_hub()` return the precise type
  (`BaseDecision`, `CPUFastDecision` for `backend='cpu_fast'`, `CalibratedDecision` when a profile is
  given), `Request.kind` / `Result.kind` are the literal `choice | noul | score`, `calibration_profiles()`
  returns `ProfileInfo` dictionaries, and the convenience methods return the right result type on the
  local model and on the cloud backend. Runtime behaviour is unchanged. The four pinned modules differ from
  the previous commit only in annotations, docstrings and typing imports (checked by comparing their syntax
  trees with those removed), 120 real-model predictions remain bit-identical to the original baseline, and
  the `sdk_contract` hashes were refreshed.
- `cpu_fast` no longer requires exactly torch 2.9.1 and transformers 4.57.6 (which forced a specific, and
  security-advisory-laden, install). It accepts torch >= 2.6 with transformers 4.48-4.57 and verifies itself
  on the installed versions at load time with a short known-answer self-test (strict in FP32: a correct
  adapter agrees to ~1e-7 and a 1% error is caught; plus a BF16 pass), changing no global state. Outside the
  range or on a failed self-test it raises `CPUFastUnavailable` (an `InputError`) with advice, never
  silently falling back. `backend_info()` gains `self_test`. Verified with the real checkpoint on torch
  2.6.0/transformers 4.48.0, 2.9.1/4.57.6 and 2.14.1/4.57.6; refused cleanly on transformers 5.18.
- The README now starts with a three-line install and a six-line first run, followed by one short,
  runnable example per supported feature. The tests execute those examples in order against a real
  checkpoint and assert the outputs the README prints; the Hugging Face, OpenAI/Anthropic and
  GPU-calibration examples are only compiled and import-checked. Provider details moved to the new
  `docs/CLOUD.md`, the batching guidance to `docs/API.md`; the stale rc4 wheel command is gone.

## Removed

- `tools/` (the HPC regression, GPU benchmark, calibration-smoke and `cpu_fast` verification runners, which
  needed the research checkout and datasets), `audits/`, `LOCAL_VERIFICATION.json` and
  `cpu_fast_verification.json` (internal release evidence), and the research-adapter packing parity test.
  The hardware-independent part of the `cpu_fast` verification now lives in
  `tests/test_cpu_fast_kernels.py`. Fine-tuning, RL and evaluation code is not part of this repository.

## Fixed

- Passing an over-long text to a model loaded with `load()` / `load_from_hub()` made Transformers print
  "Token indices sequence length is longer than the specified maximum ... will result in indexing
  errors" next to the clean `ContextLengthError`. The notice was wrong (nothing is sent to the model), so
  the loaders now mark it as already shown. `BaseDecision.from_pretrained` / `from_hub` are unchanged.
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

- `close()`, `closed` and context-manager support on local models (`BaseDecision`, `CPUFastDecision`,
  `CalibratedDecision`); the cloud backend gains `closed`. `close()` waits for a running request, releases
  the network and tokenizer (and empties the CUDA cache), and is idempotent; afterwards request methods
  raise `InputError`. `with load(path) as model:` is the idiomatic form. `load(..., calibration_profile=...)`
  closes the model it loaded if calibration cannot be applied, instead of leaving it in memory. Changes
  `client.py` (the `sdk_contract` hash was refreshed); no inference code changed.
- `load_from_hub(repo_id, revision=...)`: explicit Hub download with the same machine-appropriate defaults;
  `resolve_device()`; `--device`/`--precision` flags in all example scripts; CPU timings and memory in the
  README's "Force the CPU or the GPU" example.

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
