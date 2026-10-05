# Calibration profiles

Raw output is the default. Use `calibration_profiles()` without loading weights to inspect coverage. Only explicitly selected profiles are applied:

```python
from basedecision import load, CalibratedDecision, calibration_profiles

print(calibration_profiles())
raw_model = load('/path/to/model')
sgd_model = CalibratedDecision(raw_model, profile='sgd_schema')
result = sgd_model.predict(sgd_request)
print(result.probabilities)
print(result.raw_probabilities)
print(result.calibration)
```

`sgd_request` must use the assessed SGD current-service intent task, service schema, history, choices and rendering. A generic refund example is not SGD merely because both involve intent recognition. The caller declares the workload; we cannot infer membership reliably from text. Option/token checks are additional bounds, not proof of semantic coverage.

| Profile | Release behavior | Temperature | Assessment NLL raw → scaled | Equal-width ECE raw → scaled |
|---|---|---:|---:|---:|
| sgd_identifier | Explicit opt-in | 1.4646 | .6116 → .5830 | 8.05% → 2.97% |
| sgd_schema | Explicit opt-in | 1.4793 | .6034 → .5606 | 7.88% → 2.96% |
| vast | Available explicitly; raw recommended | 1.0071 | .6246 → .6242 | 7.68% → 7.43% |
| clinc | Unavailable; experimental | 1.5831 | .4408 → .3955 | 5.43% → 2.68% |

SGD assessment: 1,270 questions / 125 dialogue groups per format. These formats share groups. VAST: 118 questions / 32 groups. CLINC: 156 questions / groups. Labels and accuracy were unchanged.

CLINC's NLL-difference interval [-.1257,+.0220] exceeded the +.02 upper-bound tolerance; no silent override is provided. VAST's change is negligible. SGD's short and two-option slices regressed: identifier two-option NLL .2516 → .2827, schema two-option .1929 → .2087. No post-hoc two-option routing rule has been added. The opt-in profile retains the assessed aggregate tradeoff.

The metadata are attached to results and accessible before inference. The wrapped client accepts choice requests only. For boolean and score tasks, use the raw model explicitly; those types have no reviewed profiles. Batch scope/packing is validated before inference. The wrapper preserves labels, raw logits, and raw probabilities, and does not introduce abstention thresholds. CalibrationError is an InputError subtype.

Model/encoder/tokenizer hashes and the four local inference module hashes are verified once per wrapper creation. Create the wrapper once and reuse it. RC4 keeps those RC3 files byte-identical, so the original artifact remains valid without refitting or changing its scientific provenance. The checkpoint must match the assessed CUDA/BF16 configuration. The metadata's RC3 version records the original fit; compatibility is checked by the unchanged inference implementation, not by relabeling that artifact.

These results do not establish calibration for arbitrary schemas, FinEntity, long-context reasoning, unknown distributions, or cloud providers. Existing reserved data had historical project development use. This was a group-disjoint fit/selection/assessment, not a fresh external certification.

`artifact='/local/calibration.json'` is supported for compatible scalar artifacts. This is a caller-supplied assessment claim, not an official endorsement; editing an artifact's acceptance field is not validation. Untrusted JSON is never executed. No fitting dependency, SciPy, model training, or network request is required at inference.

Run the bounded integration smoke on your GPU host:

```bash
python tools/calibration_smoke.py \
  --model "$BASEDECISION_DATA/runs/model" \
  --records "$BASEDECISION_DATA/data/records.jsonl"
```
