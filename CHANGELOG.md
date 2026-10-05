# Unreleased

- Jev/SystemOne API support: `SystemOne` / `basedecision.systemone.systemone` answer `POST /v1/systemone`
  request bodies (choice, score and noul questions) with local models; strict validation, stable error
  codes (`SystemOneError`), no silent truncation, images/videos rejected, cloud providers refused.
- `examples/systemone_quickstart.py` and `examples/systemone_server.py` (reference HTTP server; auto device
  selection), `docs/SYSTEMONE.md`.
- No change to the hash-pinned inference modules (`client.py`, `packing.py`, `_model.py`, `types.py`).

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
