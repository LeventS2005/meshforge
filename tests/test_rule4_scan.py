"""The rule-4 AST scanner itself (V43, D65).

The scanner guards the rule-4 boundary for the web + shell layers; a bug
that never flags anything would silently void that guard (rule 6). Each
violation shape is proven to be caught, and the docstring/comment false
positive the old substring scan suffered from is proven gone.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rule4_scan import assert_holds_no_meshing_logic, meshing_logic_references


def _scan(tmp_path: Path, source: str) -> list[str]:
    path = tmp_path / "candidate.py"
    path.write_text(source)
    return meshing_logic_references(path)


def test_clean_module_passes(tmp_path: Path) -> None:
    assert _scan(tmp_path, "from meshforge.pipeline import MeshingSession\n") == []


def test_docstring_and_comment_mentions_are_not_violations(tmp_path: Path) -> None:
    # The exact false positive the old substring scan had.
    source = (
        '"""Docs may say meshforge.meshing or mesh_wall freely."""\n'
        "# comment: meshforge.solve, export_abaqus\n"
        "x = 1\n"
    )
    assert _scan(tmp_path, source) == []


@pytest.mark.parametrize(
    "line",
    [
        "import meshforge.meshing\n",
        "import meshforge.meshing.wall as w\n",
        "from meshforge.geometry import frames\n",
        "from meshforge import synth\n",
        "from ..meshing import wall\n",  # relative, inside the package
        "from meshforge.meshing.swept import mesh_wall\n",
        "from somewhere import mesh_wall\n",
        "import importlib\nm = importlib.import_module('meshforge.solve')\n",
        "m = __import__('meshforge.meshing')\n",
        "y = mesh_wall(x)\n",
        "y = pkg.export_abaqus(x)\n",
    ],
)
def test_each_violation_shape_is_caught(tmp_path: Path, line: str) -> None:
    hits = _scan(tmp_path, line)
    assert hits, f"scanner missed a rule-4 violation: {line!r}"
    path = tmp_path / "candidate.py"
    path.write_text(line)
    with pytest.raises(AssertionError, match="rule-4 violation"):
        assert_holds_no_meshing_logic(path)


def test_unrelated_solve_names_do_not_trip(tmp_path: Path) -> None:
    # "solve" itself is not forbidden — only the meshforge.solve module and
    # the listed entry points are.
    source = "def solve(x):\n    return x\n\nresult = solve(2)\n"
    assert _scan(tmp_path, source) == []
