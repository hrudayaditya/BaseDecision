"""Ask several named questions about one text with a local model.

    python examples/python_quickstart.py --model /path/to/model
    python examples/python_quickstart.py --model /path/to/model --device cpu

The device is chosen automatically: CUDA/BF16 when a suitable GPU is present, otherwise CPU/FP32.
"""

import argparse
import json

from basedecision import decide, load


def main() -> None:
    """Parse the arguments, run the example and print the result."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="path to the exported checkpoint")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--precision", choices=["fp32", "bf16"], help="default: bf16 on CUDA, fp32 on CPU"
    )
    args = parser.parse_args()
    model = load(args.model, device=args.device, precision=args.precision)
    answers = decide(
        model,
        context="I was charged twice for this purchase. Please refund the duplicate payment.",
        questions={
            "request": {
                "kind": "choice",
                "question": "What does the customer request?",
                "options": ["Refund request", "Delivery status", "Change shipping address"],
            },
            "duplicate_charge": {
                "kind": "noul",
                "question": "Does the customer report being charged twice?",
            },
        },
    )
    print(json.dumps({name: result.to_dict() for name, result in answers.items()}, indent=2))


if __name__ == "__main__":
    main()
