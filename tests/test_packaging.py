"""The package metadata PyPI shows, checked without building anything.

``pyproject.toml`` is read as text so the checks also run on Python 3.10 (no ``tomllib``).
"""

from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path

import basedecision

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/hrudayaditya/BaseDecision"


def section(text: str, name: str) -> str:
    body = text.split(f"[{name}]", 1)[1]
    return re.split(r"^\[", body, maxsplit=1, flags=re.MULTILINE)[0]


class PackageMetadataTests(unittest.TestCase):
    pyproject = (ROOT / "pyproject.toml").read_text()
    project = section(pyproject, "project")

    def test_version_is_the_same_everywhere(self) -> None:
        declared = re.search(r'^version = "([^"]+)"', self.project, re.MULTILINE)
        self.assertIsNotNone(declared)
        assert declared is not None
        self.assertEqual(declared.group(1), basedecision.__version__)

    def test_license_is_an_spdx_expression_not_the_license_text(self) -> None:
        self.assertRegex(self.project, r'(?m)^license = "Apache-2\.0"$')
        self.assertNotIn("license = {", self.project)  # the deprecated table dumps the whole text
        self.assertNotIn("License ::", self.project)  # superseded by the expression
        self.assertRegex(self.pyproject, r'requires = \["setuptools>=(7[7-9]|[89]\d)"')

    def test_project_urls_point_at_the_repository(self) -> None:
        urls = dict(
            re.findall(
                r'^(\w+) = "(https://[^"]+)"$', section(self.pyproject, "project.urls"), re.M
            )
        )
        for name in ("Homepage", "Source", "Issues", "Changelog", "Documentation"):
            with self.subTest(url=name):
                self.assertTrue(
                    urls.get(name, "").startswith(REPO_URL), f"{name} must point at {REPO_URL}"
                )

    def test_classifiers_describe_what_is_tested(self) -> None:
        for expected in (
            "Typing :: Typed",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.14",
        ):
            self.assertIn(f'"{expected}"', self.project)
        self.assertNotIn('Aditya"', self.project)  # the old short author name

    def test_a_plain_install_brings_everything_the_readme_uses(self) -> None:
        listed = re.search(r"(?ms)^dependencies = \[(.*?)^\]", self.project)
        self.assertIsNotNone(listed)
        assert listed is not None
        declared = set(re.findall(r'"([A-Za-z0-9_.-]+)', listed.group(1)))
        for name in (
            "torch",
            "transformers",
            "safetensors",
            "tokenizers",
            "huggingface-hub",
            "openai",
            "anthropic",
        ):
            self.assertIn(name, declared)

    def test_old_install_commands_still_work_through_alias_extras(self) -> None:
        extras = section(self.pyproject, "project.optional-dependencies")
        for name in (
            "runtime",
            "hub",
            "providers",
            "openai",
            "anthropic",
        ):  # printed by 0.1.0 and by error messages
            self.assertRegex(extras, rf"(?m)^{name} = \[\]$")

    def test_every_third_party_import_is_a_declared_dependency(self) -> None:
        listed = re.search(r"(?ms)^dependencies = \[(.*?)^\]", self.project)
        assert listed is not None
        declared = {
            n.lower().replace("-", "_") for n in re.findall(r'"([A-Za-z0-9_.-]+)', listed.group(1))
        }
        imported: set[str] = set()
        for path in (ROOT / "src" / "basedecision").glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    imported.add(node.module.split(".")[0])
        third_party = {
            m for m in imported if m not in sys.stdlib_module_names and m != "basedecision"
        }
        self.assertEqual(
            sorted(third_party - declared), [], "imported but not declared in dependencies"
        )

    def test_authors_are_not_the_old_placeholder(self) -> None:
        self.assertIn('authors = [{name = "Hrudayaditya Jallu"}]', self.project)


if __name__ == "__main__":
    unittest.main()
