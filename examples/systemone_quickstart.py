"""Answer a Jev/SystemOne request with a local BaseDecision model.

    python examples/systemone_quickstart.py --model /path/to/model

The device is chosen automatically (CUDA/BF16 if available, otherwise CPU/FP32); pass
--device/--precision to override. No network access is used.
"""

import argparse
import json

from basedecision import SystemOne, SystemOneError, load

REQUEST = {
    "model": "basedecision",
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle the message?",
            "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
        },
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "outage": {"type": "noul", "instructions": "Is a service down?"},
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="path to the exported checkpoint")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--precision", choices=["fp32", "bf16"])
    args = parser.parse_args()
    model = load(args.model, device=args.device or "auto", precision=args.precision)
    service = SystemOne(model)
    try:
        response = service(REQUEST)
    except SystemOneError as error:
        raise SystemExit(json.dumps(error.to_dict())) from error
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
