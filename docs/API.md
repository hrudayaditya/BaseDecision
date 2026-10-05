# Python API

`load(local_path, device=None, precision=None, ...)` loads a local exported checkpoint. `device` and `precision` default to what works on the machine: CUDA/BF16 when a GPU with native BF16 is present, otherwise CPU/FP32 (`resolve_device()` returns that choice). Explicit values are used as given. `BaseDecision.from_pretrained` is the strict low-level constructor and defaults to CUDA/BF16. Pass `calibration_profile='sgd_schema'` only when all requests belong to that assessed scope. For mixed workloads, keep the raw client and wrap it separately.

`load_from_hub(repo_id, revision=..., ...)` explicitly downloads a supported checkpoint and loads it with the same machine-appropriate defaults (`BaseDecision.from_hub` is the strict variant). No remote Python code is executed. Actual published checkpoint namespace is still to be assigned.

`BaseDecision.from_provider('openai'|'anthropic', model_id, ...)` explicitly opts into cloud inference. Context managers close clients. No automatic cloud fallback. Provider results do not include fabricated probabilities.

- `choose(context=str, question=str, options=[str|Option,...])`: choice result.
- `check(context=str, question=str)`: boolean answer.
- `score(context=str, question=str, levels=[str|Option,...], values=[...])`: ordered score level.
- `predict(Request(...))`: one typed request.
- `predict_batch(requests)`: ordered results; local budgeted batching, cloud sequential calls.
- `predict_iter(requests, buffer_size=64)`: bounded request buffering.
- `count_tokens(request)`: local complete packed length; rejects over-budget inputs.
- `result.to_dict()`: serializable output.
- `close()` / `closed` / `with load(path) as model:`: release the local weights (and the GPU memory they held). It waits for a running request, is safe to repeat, and afterwards the request methods raise `InputError`. `model_id`, `device` and `precision` stay readable. The cloud backend has the same three; `load(path, calibration_profile=...)` returns a wrapper that owns its model, so closing it closes the model too, while a `CalibratedDecision(model, ...)` you build yourself never closes your model.
- `SystemOne(model)(request_body)`: answers a Jev/SystemOne request body; see [SYSTEMONE.md](SYSTEMONE.md).

Named questions use `decide(model, context=..., questions={name: schema})`. Schemas contain `kind`, `question`, optional `options`, and score-only `values`. The kind defaults to choice; noul has fixed false/true options and does not accept an options field. Unknown fields and invalid schemas are rejected before inference. Returns a dict of typed results in question order. Each question is an independent model input; this is not shared-state attention or a single-pass multi-question head.

`Option('stable_id', 'Readable description')` is optional. Plain strings work as both ID and label. Users do not need to supply custom definitions. Descriptions can help ambiguous labels, but no model can infer an undocumented meaning for an arbitrary opaque identifier.

**Batching.** Local batch size defaults to 1. For offline throughput on similar-length inputs, explicitly set `max_batch_size=8` and an appropriate `max_batch_tokens` when loading the model. Near-8K batch=4 was the largest measured configuration in the supplied 32768-token benchmark (see [BENCHMARKS.md](../BENCHMARKS.md)). No universal speedup or GPU memory guarantee is claimed; on a CPU, batching mixed-length requests is slower than one at a time. `predict_iter` bounds the buffered request count; `predict_batch` materializes the submitted iterable.

Cloud backends, their errors and their limits: [CLOUD.md](CLOUD.md).

Local input limit: 8192 **total packed tokens**, including question and all intact options; the checkpoint may impose a smaller limit. Overflow and token-identical choices raise explicit errors. The package does not summarize or truncate inputs automatically.

Importing the public package does not import Torch. Optional runtime/provider dependencies are imported when used. Local batch size remains 1 by default. Supported local devices: CPU/FP32 and CUDA/BF16 or FP32; packaged calibration covers CUDA/BF16 only.

This is a native decision-model package. It does not implement `.generate()`, AutoModelForCausalLM, TypeScript, or an autonomous agent, and the library itself is not an HTTP service (`examples/systemone_server.py` is a small reference server for the Jev/SystemOne API). Its model quality must be assessed for each application.
