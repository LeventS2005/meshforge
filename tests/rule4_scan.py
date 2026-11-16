"""AST-based rule-4 scan (V43, D65): the UI layers hold no meshing logic.

Replaces the old substring scan, which failed on a forbidden token anywhere
in the file — including docstrings and comments. This walks the AST instead,
so only *real code references* count:

- ``import``/``from ... import`` of a forbidden module (absolute or
  relative — ``from ..meshing import x`` inside ``meshforge.web`` resolves
  to ``meshforge.meshing``),
- dynamic imports with a literal module string
  (``importlib.import_module("meshforge.meshing")`` / ``__import__``),
- any ``Name``/``Attribute`` reference to a forbidden entry point.

Shared by the web-layer and desktop-shell scans. The static *JS* scans stay
substring checks — the frontend is not Python.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Modules that hold meshing/geometry/solve logic — the GUI-agnostic layers
# (web) may only talk to MeshingSession (rule 4).
FORBIDDEN_MODULES = (
    "meshforge.geometry",
    "meshforge.meshing",
    "meshforge.synth",
    "meshforge.solve",
)
# Entry points of the meshing/export internals; a reference from a UI layer
# means logic leaked past the service boundary.
FORBIDDEN_NAMES = (
    "mesh_wall",
    "get_backend",
    "export_surface",
    "export_abaqus",
    "write_polymesh",
)
# Tails of the forbidden modules, to resolve relative imports from inside
# the meshforge package ("..geometry" -> meshforge.geometry).
_FORBIDDEN_TAILS = tuple(mod.split(".", 1)[1] for mod in FORBIDDEN_MODULES)


def _is_forbidden_module(name: str) -> bool:
    return any(
        name == mod or name.startswith(mod + ".") for mod in FORBIDDEN_MODULES
    )


def _is_forbidden_relative(module: str) -> bool:
    return any(
        module == tail or module.startswith(tail + ".")
        for tail in _FORBIDDEN_TAILS
    )


def meshing_logic_references(path: Path) -> list[str]:
    """Every real code reference to meshing internals in *path* (a .py file)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    hits: list[str] = []

    def hit(node: ast.AST, what: str) -> None:
        hits.append(f"{path.name}:{node.lineno}: {what}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden_module(alias.name):
                    hit(node, f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0:
                forbidden_base = _is_forbidden_module(module)
            else:  # relative import inside the meshforge package
                forbidden_base = _is_forbidden_relative(module)
            if forbidden_base:
                hit(node, f"from {'.' * node.level}{module} import ...")
                continue
            for alias in node.names:
                dotted = f"{module}.{alias.name}" if module else alias.name
                if node.level == 0 and _is_forbidden_module(dotted):
                    hit(node, f"from {module} import {alias.name}")
                elif alias.name in FORBIDDEN_NAMES:
                    hit(
                        node,
                        f"from {'.' * node.level}{module} import {alias.name}",
                    )
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                hit(node, f"reference to {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_NAMES:
                hit(node, f"reference to .{node.attr}")
        elif isinstance(node, ast.Call):
            func = node.func
            dynamic = (
                isinstance(func, ast.Name) and func.id == "__import__"
            ) or (
                isinstance(func, ast.Attribute) and func.attr == "import_module"
            )
            if dynamic and node.args:
                arg = node.args[0]
                if (
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and _is_forbidden_module(arg.value)
                ):
                    hit(node, f"dynamic import of {arg.value}")
    return hits


def assert_holds_no_meshing_logic(*paths: Path) -> None:
    problems = [h for p in paths for h in meshing_logic_references(p)]
    assert not problems, "rule-4 violation(s):\n" + "\n".join(problems)
