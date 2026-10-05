"""A deterministic corpus of *ordinary* requests, used to pin the packing behaviour.

The corpus deliberately contains no special-token strings, no lone surrogates, no mapping options and
no input larger than the exact-tokenization threshold: exactly the inputs whose packed tokens must
never change. ``tests/data/packing_golden.json`` holds the digest of what the reference
implementation produced for each case; ``test_input_hardening`` replays the corpus and compares.

To regenerate the golden file after a *deliberate* change to the token format, run
``python tests/packing_corpus.py MODEL_DIR tests/data/packing_golden.json`` and review the diff.
"""

# ruff: noqa: E501, RUF001  (the corpus deliberately contains long literals and CJK punctuation)
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

from basedecision import Option, Request
from basedecision.packing import pack
from basedecision.types import InputError

WORDS = ["account", "refund", "order", "payment", "invoice", "delivery", "customer", "shipping", "address", "package", "status", "contract", "clause", "party", "breach", "notice", "termination", "renewal", "liability", "warranty", "damages", "patient", "dose", "symptom", "diagnosis", "treatment", "allergy", "appointment", "schedule", "urgent", "normal", "the", "a", "of", "to", "and", "in", "is", "it", "for", "on", "with", "as", "was", "be", "by", "at", "this", "that", "from", "or", "an", "are", "not", "have", "price", "total", "tax", "discount", "subscription", "cancel", "upgrade", "downgrade", "login", "password", "error", "server", "timeout", "database", "request", "response", "latency", "deploy", "release", "version", "update", "install"]
EXTRAS = [
    "été café naïve über",
    "你好，世界。这是一个测试。",
    "مرحبا بالعالم",
    "שלום עולם",
    "🙂 💸 📦 ❓ 👍🏽",
    "Ωmega µ± ≤ ≥ ∞",
    "á é ö",
    "한국어 텍스트입니다",
    "tab\tseparated\tvalues",
    "line1\nline2\r\nline3",
    "multiple    spaces     here",
    "https://example.com/path?x=1&y=2",
    "user@example.com +1 (555) 010-9999",
    "def f(x):\n    return x ** 2  # square",
    '{"a": 1, "b": [true, null, 2.5]}',
    "2026-10-05T12:30:00Z 1,234.56 $99.99 50% #42",
]


def sentence(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n)).capitalize() + "."


def context(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.03:
        return ""
    if kind < 0.40:
        return " ".join(sentence(rng, rng.randint(4, 14)) for _ in range(rng.randint(1, 6)))
    if kind < 0.55:
        data = {
            rng.choice(WORDS): {
                rng.choice(WORDS): rng.randint(0, 999),
                "tags": [rng.choice(WORDS) for _ in range(3)],
            }
            for _ in range(rng.randint(1, 8))
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":"))
    if kind < 0.70:
        return " ".join(rng.choice(EXTRAS) for _ in range(rng.randint(1, 5)))
    if kind < 0.80:
        return " ".join(
            sentence(rng, 10) + " " + rng.choice(EXTRAS) for _ in range(rng.randint(20, 120))
        )
    if kind < 0.90:
        return rng.choice([" ", "  ", "\n", "\n\n  \t "]) * rng.randint(1, 60) + sentence(rng, 5)
    if kind < 0.97:
        return " ".join(
            sentence(rng, 12) for _ in range(rng.randint(150, 520))
        )  # long, up to ~8K tokens
    return " ".join(sentence(rng, 14) for _ in range(rng.randint(700, 1100)))  # over the 8192 limit


def build_corpus(count: int = 450, seed: int = 20261005) -> list[Request]:
    rng = random.Random(seed)
    requests: list[Request] = []
    while len(requests) < count:
        text, question = context(rng), sentence(rng, rng.randint(3, 9)).rstrip(".") + "?"
        roll = rng.random()
        try:
            if roll < 0.2:
                options = (Option("false", "false"), Option("true", "true"))
                requests.append(Request(text, question, options, "noul"))
            elif roll < 0.4:
                n = rng.randint(2, 6)
                levels = tuple(
                    Option(str(i), sentence(rng, rng.randint(1, 4)).rstrip(".")) for i in range(n)
                )
                requests.append(Request(text, question, levels, "score", tuple(range(n))))
            else:
                n = rng.choice([2, 2, 3, 3, 4, 6, 12, 40])
                opts = tuple(
                    Option(
                        f"id{i}",
                        rng.choice(EXTRAS)
                        if rng.random() < 0.15
                        else sentence(rng, rng.randint(1, 8)).rstrip("."),
                    )
                    for i in range(n)
                )
                requests.append(Request(text, question, opts))
        except InputError:  # duplicate random option text: draw again
            continue
    return requests


def digest(tokenizer: object, request: Request) -> str:
    """Digest of the packed tokens (or of the exact error) for one request."""
    try:
        packed = pack(tokenizer, request, 8192)
    except Exception as error:
        required = getattr(error, "required_tokens", "")
        return f"E:{type(error).__name__}:{required}"
    payload = json.dumps([list(packed.ids), list(packed.markers), packed.qtype])
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


if __name__ == "__main__":
    from transformers import AutoTokenizer

    model_dir, target = Path(sys.argv[1]), Path(sys.argv[2])
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"), local_files_only=True)
    corpus = build_corpus()
    digests = [digest(tokenizer, request) for request in corpus]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"seed": 20261005, "cases": len(digests), "digests": digests}, indent=0) + "\n"
    )
    errors = sum(d.startswith("E:") for d in digests)
    print(f"wrote {len(digests)} digests ({errors} are exact over-limit errors) to {target}")
