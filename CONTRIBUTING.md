# Contributing

Use Python 3.10+ and a virtual environment. Install editable source with the optional dependencies your change requires. Run `python -m unittest discover -s tests -v` and build the wheel before submitting changes. Provider transport tests use mocks and require provider extras, not API keys. Runtime/GPU checks are separate.

Keep model loading explicit, preserve full input/option packing, avoid global backend-setting changes, and never add implicit cloud calls. Inference-module edits invalidate the packaged calibration contract and require fresh equivalence and compatibility review. Preserve upstream license notices. Do not commit weights, datasets, credentials, API responses containing user text, or local cache directories.

Include the behavior change, verification performed, and limitations in the change description. Report security issues privately through the repository owner's designated channel once the public repository is configured; do not publish secrets in issues.
