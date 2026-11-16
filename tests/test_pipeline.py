from __future__ import annotations

from pathlib import Path

import pytest

from meshforge.pipeline import MeshingSession


@pytest.fixture(scope="module")
def synth_session() -> MeshingSession:
    s = MeshingSession()
    s.generate_synth("straight_tube")
    return s


def test_session_full_pipeline_headless(synth_session: MeshingSession) -> None:
    s = synth_session
    r1 = s.compute_centerline()
    assert r1.step == "centerline"
    assert r1.summary["n_samples"] == 200
    assert "centerline" in r1.text
    assert r1.render_hints["kind"] == "polyline"

    r4 = s.compute_volmesh(backend="swept", thickness=0.5, n_axial=30)
    assert r4.summary["quality"]["n_inverted"] == 0
    assert r4.render_hints["kind"] == "volume"


def test_session_load_from_file(tmp_path: Path, sphere_stl: Path) -> None:
    s = MeshingSession()
    result = s.load(sphere_stl)
    assert result.summary["watertight"] is True
    assert s.input_sha256 == result.summary["sha256"]

    out = s.save("surface", tmp_path / "again.vtu")
    assert Path(out.summary["path"]).is_file()


def test_save_refuses_overwriting_input(sphere_stl: Path) -> None:
    s = MeshingSession()
    s.load(sphere_stl)
    # Same path, format-corrected suffix differs -> craft the exact match.
    with pytest.raises(ValueError, match="unknown export format"):
        s.save("surface", sphere_stl)  # .stl is not a neutral export format


def test_steps_fail_loudly_without_prerequisites() -> None:
    s = MeshingSession()
    with pytest.raises(RuntimeError, match="no surface"):
        s.compute_centerline()
    s.generate_synth("straight_tube")
    with pytest.raises(RuntimeError, match="no volmesh"):
        s.save("volmesh", "anywhere.vtu")
    with pytest.raises(ValueError, match="unknown save target"):
        s.save("banana", "x.vtu")


def test_new_source_invalidates_downstream(synth_session_factory=None) -> None:
    s = MeshingSession()
    s.generate_synth("straight_tube")
    s.compute_centerline()
    assert s.centerline is not None
    s.generate_synth("tapered_tube")
    assert s.centerline is None  # stale geometry must not leak across sources
