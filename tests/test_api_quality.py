"""Keeps the public API documented and typed.

The package ships ``py.typed``, so its annotations are part of the interface. These checks need
nothing installed except, for the last one, ``mypy``.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "basedecision"
DOCUMENTED_DUNDERS = {
    "__init__"
}  # the rest (__enter__, __exit__, ...) have fixed, standard meanings


def modules() -> list[Path]:
    return sorted(PACKAGE.glob("*.py"))


def is_overload(node: ast.FunctionDef) -> bool:
    return any(isinstance(d, ast.Name) and d.id == "overload" for d in node.decorator_list)


def functions(tree: ast.Module) -> list[tuple[str, ast.FunctionDef, bool]]:
    """Every function with its dotted name and whether it is public (reachable by users)."""
    found: list[tuple[str, ast.FunctionDef, bool]] = []

    def visit(node: ast.AST, prefix: str, public: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.", public and not child.name.startswith("_"))
            elif isinstance(child, ast.FunctionDef):
                name = child.name
                shown = public and (not name.startswith("_") or name in DOCUMENTED_DUNDERS)
                found.append((f"{prefix}{name}", child, shown))
                # nested functions are implementation details: typed (mypy), not documented
                for inner in ast.walk(child):
                    if isinstance(inner, ast.FunctionDef) and inner is not child:
                        found.append((f"{prefix}{name}.<{inner.name}>", inner, False))

    visit(tree, "", True)
    return found


class DocumentationAndTypingTests(unittest.TestCase):
    def test_every_public_class_and_function_has_a_docstring(self) -> None:
        missing: list[str] = []
        for path in modules():
            tree = ast.parse(path.read_text())
            if path.name != "__init__.py" and ast.get_docstring(tree) is None:
                missing.append(f"{path.name} (module)")
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ClassDef)
                    and not node.name.startswith("_")
                    and ast.get_docstring(node) is None
                ):
                    missing.append(f"{path.name}:{node.name}")
            for name, function, public in functions(tree):
                if public and not is_overload(function) and ast.get_docstring(function) is None:
                    missing.append(f"{path.name}:{name}")
        self.assertEqual(missing, [], "public items without a docstring")

    def test_every_function_is_fully_annotated(self) -> None:
        missing: list[str] = []
        for path in modules():
            for name, function, _public in functions(ast.parse(path.read_text())):
                args = function.args
                every = [*args.posonlyargs, *args.args, *args.kwonlyargs]
                if args.vararg:
                    every.append(args.vararg)
                if args.kwarg:
                    every.append(args.kwarg)
                unannotated = [
                    a.arg for a in every if a.annotation is None and a.arg not in {"self", "cls"}
                ]
                if unannotated or function.returns is None:
                    missing.append(f"{path.name}:{name} {unannotated or ''}")
        self.assertEqual(missing, [], "functions without complete annotations")

    def test_the_package_declares_its_types(self) -> None:
        self.assertTrue((PACKAGE / "py.typed").is_file())
        manifest = (ROOT / "pyproject.toml").read_text()
        self.assertIn('"py.typed"', manifest, "py.typed must be listed as package data")

    @unittest.skipUnless(importlib.util.find_spec("mypy"), "mypy is not installed")
    def test_mypy_strict_passes(self) -> None:
        from mypy import api

        stdout, stderr, status = api.run(
            [
                "--config-file",
                str(ROOT / "pyproject.toml"),
                "--cache-dir",
                os.devnull,
                str(PACKAGE),
            ]
        )
        self.assertEqual(status, 0, stdout + stderr)


if __name__ == "__main__":
    unittest.main()
