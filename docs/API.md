# Python API

`load(local_path, device='cuda', precision='bf16', ...)` loads a local exported checkpoint. Equivalent to `BaseDecision.from_pretrained`. Pass `calibration_profile='sgd_schema'` only when all requests belong to that assessed scope. For mixed workloads, keep the raw client and wrap it separately.

`BaseDecision.from_hub(repo_id, revision=..., ...)` explicitly downloads a supported checkpoint. No remote Python code is executed. Actual published checkpoint namespace is still to be assigned.

`BaseDecision.from_provider('openai'|'anthropic', model_id, ...)` explicitly opts into cloud inference. Context managers close clients. No automatic cloud fallback. Provider results do not include fabricated probabilities.

- `choose(context=str, question=str, options=[str|Option,...])`: choice result.
- `check(context=str, question=str)`: boolean answer.
- `score(context=str, question=str, levels=[str|Option,...], values=[...])`: ordered score level.
- `predict(Request(...))`: one typed request.
- `predict_batch(requests)`: ordered results; local budgeted batching, cloud sequential calls.
- `predict_iter(requests, buffer_size=64)`: bounded request buffering.
- `count_tokens(request)`: local complete packed length; rejects over-budget inputs.
- `result.to_dict()`: serializable output.
- `SystemOne(model)(request_body)`: answers a Jev/SystemOne request body; see [SYSTEMONE.md](SYSTEMONE.md).

Named questions use `decide(model, context=..., questions={name: schema})`. Schemas contain `kind`, `question`, optional `options`, and score-only `values`. The kind defaults to choice; noul has fixed false/true options and does not accept an options field. Unknown fields and invalid schemas are rejected before inference. Returns a dict of typed results in question order. Each question is an independent model input; this is not shared-state attention or a single-pass multi-question head.

`Option('stable_id', 'Readable description')` is optional. Plain strings work as both ID and label. Users do not need to supply custom definitions. Descriptions can help ambiguous labels, but no model can infer an undocumented meaning for an arbitrary opaque identifier.

Local input limit: 8192 **total packed tokens**, including question and all intact options; the checkpoint may impose a smaller limit. Overflow and token-identical choices raise explicit errors. The package does not summarize or truncate inputs automatically.

Importing the public package does not import Torch. Optional runtime/provider dependencies are imported when used. Local batch size remains 1 by default. Supported local devices: CPU/FP32 and CUDA/BF16 or FP32; packaged calibration covers CUDA/BF16 only.

This is a native decision-model package. It does not implement `.generate()`, AutoModelForCausalLM, TypeScript, or an autonomous agent, and the library itself is not an HTTP service (`examples/systemone_server.py` is a small reference server for the Jev/SystemOne API). Its model quality must be assessed for each application.
