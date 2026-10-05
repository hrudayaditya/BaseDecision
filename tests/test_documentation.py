"""Executes the documentation and the example scripts against a real checkpoint.

Set ``BASEDECISION_TEST_MODEL`` to a checkpoint directory. Nothing is configured for the
hardware: the snippets run exactly as a new user would run them (only the placeholder path is
replaced), so on a machine without a GPU this proves the first run works on a CPU.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

import basedecision

ROOT = Path(__file__).resolve().parents[1]
MODEL = os.environ.get("BASEDECISION_TEST_MODEL", "")
PACKAGE_PARENT = str(Path(basedecision.__file__).resolve().parents[1])

# (file, heading whose first ```python block is executed, name of the variable it must define)
SNIPPETS = [
    ("README.md", "## Quick Python integration", "answers"),
    ("README.md", "## Local inference", "result"),
    ("README.md", "## Jev / SystemOne API", "response"),
    ("docs/SYSTEMONE.md", None, "response"),
]


def first_python_block(text: str, heading: str | None) -> str:
    if heading is not None:
        text = text.split(heading, 1)[1].split("\n## ", 1)[0]
    match = re.search(r"```python\n(.*?)```", text, re.S)
    if match is None:
        raise AssertionError(f"no python block after {heading!r}")
    return match.group(1)


@unittest.skipUnless(MODEL, "set BASEDECISION_TEST_MODEL to a checkpoint directory")
class DocumentationTests(unittest.TestCase):
    def test_snippets_run_exactly_as_documented(self) -> None:
        for filename, heading, variable in SNIPPETS:
            code = first_python_block((ROOT / filename).read_text(), heading)
            code = re.sub(r"""(['"])/path/to/[^'"]*\1""", "MODEL", code)  # the only substitution
            with self.subTest(file=filename, section=heading):
                self.assertIn("load(MODEL", code, "snippet no longer loads a model")
                namespace: dict[str, Any] = {"MODEL": MODEL}
                exec(compile(code, filename, "exec"), namespace)
                self.assertIn(variable, namespace)
                self.assertIsNotNone(namespace[variable])


@unittest.skipUnless(MODEL, "set BASEDECISION_TEST_MODEL to a checkpoint directory")
class ExampleScriptTests(unittest.TestCase):
    def run_example(self, script: str, *args: str) -> Any:
        env = {**os.environ, "PYTHONPATH": PACKAGE_PARENT, "PYTHONDONTWRITEBYTECODE": "1"}
        done = subprocess.run(
            [sys.executable, str(ROOT / "examples" / script), "--model", MODEL, *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
            check=False,
        )
        self.assertEqual(done.returncode, 0, done.stderr[-600:])
        return json.loads(done.stdout)

    def test_decide(self) -> None:
        for args in ((), ("--device", "auto"), ("--device", "cpu", "--precision", "fp32")):
            with self.subTest(args=args):
                result = self.run_example("decide.py", *args)
                self.assertEqual(result["answer"], "Refund request")
                self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)

    def test_python_quickstart(self) -> None:
        result = self.run_example("python_quickstart.py")
        self.assertEqual(result["request"]["answer"], "Refund request")
        self.assertIs(result["duplicate_charge"]["answer"], True)

    def test_systemone_quickstart(self) -> None:
        result = self.run_example("systemone_quickstart.py")
        self.assertEqual(set(result["answers"]), {"department", "urgency", "outage"})

    def test_bad_combinations_fail_clearly_instead_of_silently_changing(self) -> None:
        env = {**os.environ, "PYTHONPATH": PACKAGE_PARENT}
        done = subprocess.run(
            [
                sys.executable,
                str(ROOT / "examples" / "decide.py"),
                "--model",
                MODEL,
                "--device",
                "cpu",
                "--precision",
                "bf16",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
            check=False,
        )
        self.assertNotEqual(done.returncode, 0)
        self.assertIn('CPU inference requires precision="fp32"', done.stderr)


if __name__ == "__main__":
    unittest.main()
