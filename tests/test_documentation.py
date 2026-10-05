"""Executes the documentation and the example scripts against a real checkpoint.

Set ``BASEDECISION_TEST_MODEL`` to a checkpoint directory. Nothing is configured for the
hardware: the snippets run exactly as a new user would run them (only the placeholder path is
replaced), so on a machine without a GPU this proves the first run works on a CPU.

The README is also checked without a checkpoint: every install extra it names exists, every
relative link resolves, and every Python example compiles and imports only names that exist.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import unittest
import warnings
from pathlib import Path
from typing import Any

import basedecision

ROOT = Path(__file__).resolve().parents[1]
MODEL = os.environ.get("BASEDECISION_TEST_MODEL", "")
PACKAGE_PARENT = str(Path(basedecision.__file__).resolve().parents[1])

# (file, heading whose first ```python block is executed, name of the variable it must define)
SNIPPETS = [("docs/SYSTEMONE.md", None, "response")]

# README examples that cannot run here, by a marker in their code, and why. They are still compiled
# and their imports checked.
NOT_RUNNABLE = {
    "YOUR_": "placeholder repository or model id (Hugging Face download, cloud providers)",
    "CalibratedDecision(": "calibration needs a CUDA GPU with BF16",
}

# README example -> text its printed output must contain, so the "# ..." comments cannot drift.
# Keyed by a string that appears in exactly one example.
README_OUTPUT = {
    "options=['Refund', 'Delivery status', 'Change address']": [
        "Refund",
        "{'Refund': 0.99",
    ],
    "model.check(": ["True"],
    "model.score(": ["Satisfied 3.0"],
    "decide(model": ["Refund request True"],
    "Option('status', 'Customer asks": ["status - Customer asks where the parcel is"],
    "predict_batch": ["['refund', 'status']"],
    "'Refund', 'Other']).to_dict()": [
        "'answer': 'Refund'",
        "'probabilities'",
        "'raw_logits'",
    ],
    "device='cpu', precision='fp32'": ["cpu fp32"],
    "SystemOne(model)": ["technical"],
    "count_tokens": ["22\n", "20014 8192"],
    "scratch.check(": ["True\nTrue"],
}


def first_python_block(text: str, heading: str | None) -> str:
    if heading is not None:
        text = text.split(heading, 1)[1].split("\n## ", 1)[0]
    match = re.search(r"```python\n(.*?)```", text, re.S)
    if match is None:
        raise AssertionError(f"no python block after {heading!r}")
    return match.group(1)


def python_blocks(text: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", text, re.S)


def not_runnable_reason(code: str) -> str | None:
    return next((why for marker, why in NOT_RUNNABLE.items() if marker in code), None)


class ReadmeStructureTests(unittest.TestCase):
    """What can be checked without a model."""

    readme = (ROOT / "README.md").read_text()

    def test_install_commands_name_real_extras(self) -> None:
        section = (ROOT / "pyproject.toml").read_text().split("[project.optional-dependencies]")[1]
        extras = re.findall(r"^(\w+)\s*=", section.split("\n[", 1)[0], re.M)  # 3.10: no tomllib
        named = {
            extra.strip()
            for group in re.findall(r"pip install[^\n`]*?\.\[([^\]]+)\]", self.readme)
            for extra in group.split(",")
        }
        self.assertTrue(named, "the README no longer shows an install command")
        self.assertLessEqual(named, set(extras), "install extra that pyproject does not define")

    def test_relative_links_and_example_files_exist(self) -> None:
        links = re.findall(r"\]\((?!https?://|#)([^)#\s]+)", self.readme)
        scripts = re.findall(r"examples/[\w.]+\.py", self.readme)
        self.assertTrue(links and scripts)
        for target in {*links, *scripts}:
            with self.subTest(target=target):
                self.assertTrue((ROOT / target).exists(), f"README points to missing {target}")

    def test_every_example_compiles_and_imports_existing_names(self) -> None:
        for filename in [
            "README.md",
            *sorted(str(p.relative_to(ROOT)) for p in ROOT.glob("docs/*.md")),
        ]:
            for number, code in enumerate(python_blocks((ROOT / filename).read_text()), 1):
                with self.subTest(file=filename, example=number):
                    tree = ast.parse(code, filename)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom) and node.module == "basedecision":
                            for alias in node.names:
                                self.assertTrue(
                                    hasattr(basedecision, alias.name),
                                    f"{filename} imports missing name {alias.name}",
                                )

    def test_only_the_documented_examples_are_skipped(self) -> None:
        skipped = [c for c in python_blocks(self.readme) if not_runnable_reason(c)]
        # Hugging Face download, OpenAI/Anthropic, calibration: nothing else may hide here.
        self.assertEqual(len(skipped), 3)

    def test_first_example_is_the_short_first_run(self) -> None:
        first = python_blocks(self.readme)[0]
        self.assertIn("load('/path/to/model')", first)
        self.assertLessEqual(len(first.strip().splitlines()), 8)


@unittest.skipUnless(MODEL, "set BASEDECISION_TEST_MODEL to a checkpoint directory")
class DocumentationTests(unittest.TestCase):
    def test_readme_examples_run_in_order_and_print_what_they_claim(self) -> None:
        namespace: dict[str, Any] = {"MODEL": MODEL}
        claims = dict(README_OUTPUT)
        executed = 0
        for number, block in enumerate(python_blocks((ROOT / "README.md").read_text()), 1):
            if not_runnable_reason(block):
                continue
            code = re.sub(r"""(['"])/path/to/[^'"]*\1""", "MODEL", block)  # the only substitution
            printed = io.StringIO()
            with self.subTest(example=number, first_line=code.strip().splitlines()[0]):
                with warnings.catch_warnings(), contextlib.redirect_stdout(printed):
                    warnings.simplefilter("ignore", RuntimeWarning)  # cpu_fast's opt-in notice
                    exec(compile(code, f"README.md example {number}", "exec"), namespace)
                executed += 1
                for marker in [m for m in claims if m in code]:
                    for expected in claims.pop(marker):
                        self.assertIn(expected, printed.getvalue(), f"example {number} output")
        self.assertEqual(claims, {}, "an expected-output marker matched no README example")
        blocks = python_blocks((ROOT / "README.md").read_text())
        self.assertEqual(executed, len([c for c in blocks if not not_runnable_reason(c)]))

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
