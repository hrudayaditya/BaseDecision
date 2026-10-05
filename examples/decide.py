"""Answer one choice question with a local model or a cloud provider.

    python examples/decide.py --model /path/to/model                  # local: GPU if any, else CPU
    python examples/decide.py --model /path/to/model --device cpu     # force CPU
    python examples/decide.py --backend openai --model YOUR_OPENAI_MODEL_ID

The local backend chooses CUDA/BF16 when a suitable GPU is present and CPU/FP32 otherwise.
Cloud backends send the text to the provider and need OPENAI_API_KEY / ANTHROPIC_API_KEY.
"""

import argparse
import json

from basedecision import BaseDecision, load
from basedecision.errors import ProviderError


def main() -> None:
    """Parse the arguments, run the example and print the result."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=["local", "openai", "anthropic"], default="local")
    parser.add_argument("--model", required=True, help="Local export path or provider model ID")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="local backend only (default: auto)",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16"],
        help="local backend only (default: bf16 on CUDA, fp32 on CPU)",
    )
    args = parser.parse_args()
    try:
        client = (
            load(args.model, device=args.device, precision=args.precision)
            if args.backend == "local"
            else BaseDecision.from_provider(args.backend, args.model)
        )
        try:
            result = client.choose(
                context="Please refund this purchase.",
                question="What is requested?",
                options=["Refund request", "Delivery status", "Other request"],
            )
            print(json.dumps(result.to_dict(), indent=2))
        finally:
            if args.backend != "local":
                client.close()
    except ProviderError as error:
        print(
            json.dumps(
                {
                    "error": error.code,
                    "retryable": error.retryable,
                    "status_code": error.status_code,
                }
            )
        )
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
