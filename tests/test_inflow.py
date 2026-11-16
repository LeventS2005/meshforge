"""D80/D82: scalar-speed inlets, inflow report, flow-ends picker, cardiac
presets, source length units, and the centreline flow section.

Ungated tests stub the OpenFOAM run (the case files are still written and
inspected); the OpenFOAM-gated test proves the scalar and vector forms of the
same inlet solve identically (trap #16: skip cleanly without OpenFOAM).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from meshforge.solve.foam_runner import CfdResult, find_foam
from meshforge.solve.inflow import (
    LAMINAR_RE_LIMIT,
    cardiac_presets,
    flow_ends,
    inflow_notes,
    inflow_report,
    patch_geometry,
    resolve_inlet_speed,
)

# ---------------------------------------------------------------------------
# fixtures


@pytest.fixture(scope="module")
def tube_session():
    """Straight synthetic tube (r 3, along z) -> centreline -> O-grid lumen."""
    from meshforge.pipeline import MeshingSession

    s = MeshingSession()
    s.generate_synth("straight_tube")
    s.compute_centerline()
    s.compute_lumen(n_axial=12, n_theta=16)
    return s


def _fresh_tube_session():
    from meshforge.pipeline import MeshingSession

    s = MeshingSession()
    s.generate_synth("straight_tube")
    s.compute_centerline()
    s.compute_lumen(n_axial=12, n_theta=16)
    return s


def _stub_solve(monkeypatch, session):
    """Replace the OpenFOAM run: the case is still written; the 'result' is
    the session mesh with a uniform field (point data like the reader's)."""
    import meshforge.solve as solve_pkg

    calls = {}

    def fake(case_dir, *, solver=None, timeout=600.0, foam=None, series=False):
        calls["case"] = Path(case_dir)
        grid = session.volmesh.grid.copy()
        n = grid.n_points
        grid.point_data["U"] = np.tile([0.0, 0.0, 1.0], (n, 1))
        grid.point_data["p"] = np.linspace(0.0, 1.0, n)
        grid.point_data["U_mag"] = np.ones(n)
        return CfdResult(
            case=Path(case_dir), solver=solver or "simpleFoam", elapsed=0.0,
            latest_time=1.0, grid=grid, n_cells=grid.n_cells, u_max=1.0,
            p_min=0.0, p_max=1.0, notes=[],
        )

    monkeypatch.setattr(solve_pkg, "solve_case", fake)
    return calls


def _u_entry(case: Path, patch: str) -> str:
    text = (case / "0" / "U").read_text()
    m = re.search(rf"^\s*{patch} (\{{.*\}})$", text, re.M)
    assert m, f"no {patch} entry in 0/U:\n{text}"
    return m.group(1)


# ---------------------------------------------------------------------------
# patch geometry (D80)


def test_inward_normals_follow_the_centreline(tube_session):
    cl, vol = tube_session.centerline, tube_session.volmesh
    inlet = patch_geometry(vol, "inlet")
    outlet = patch_geometry(vol, "outlet")
    wall = patch_geometry(vol, "wall")
    # station 0 is the first centreline point: flow enters along +tangent[0]
    assert np.dot(inlet.normal_in, cl.tangents[0]) > 0.9999
    assert np.dot(outlet.normal_in, -cl.tangents[-1]) > 0.9999
    # centred on the TRUE axis (x = y = 0); the voxel centreline itself sits
    # ~0.06 off it
    assert math.hypot(inlet.centroid[0], inlet.centroid[1]) < 0.02
    # ... and ON the flat end cap nearest station 0 (D81 anchors the end
    # stations on the caps), not at the centreline's first point, which the
    # voxel skeleton leaves ~0.04 short of the cap
    zmin, zmax = tube_session.surface.bounds[4:6]
    cap_z = min((zmin, zmax), key=lambda z: abs(z - cl.points[0][2]))
    assert inlet.centroid[2] == pytest.approx(cap_z, abs=1e-6)
    assert inlet.is_cap and outlet.is_cap and not wall.is_cap
    assert inlet.planarity == pytest.approx(1.0, abs=1e-9)
    assert wall.planarity < 0.05  # a closed tube wall has no mean direction
    # the area IS the patch faces' area (independent VTK cell-size sum) ...
    faces = vol.boundary["inlet"].to_polydata(vol.grid.points)
    assert inlet.area == pytest.approx(faces.area, rel=1e-9)
    # ... and a coarse 16-ray O-grid disk in the r = 3 tube is ~ pi r^2
    assert inlet.area == pytest.approx(math.pi * 9.0, rel=0.05)
    assert inlet.d_eq == pytest.approx(2.0 * math.sqrt(inlet.area / math.pi))


def test_patch_geometry_ignores_face_winding():
    from meshforge.meshing.base import BoundaryGroup

    s = _fresh_tube_session()
    vol = s.volmesh
    before = patch_geometry(vol, "inlet")
    g = vol.boundary["inlet"]
    vol.boundary["inlet"] = BoundaryGroup(
        "inlet", g.cell_ids, np.ascontiguousarray(g.face_nodes[:, ::-1])
    )
    after = patch_geometry(vol, "inlet")
    np.testing.assert_allclose(after.normal_in, before.normal_in, atol=1e-12)
    assert after.area == pytest.approx(before.area)


def test_unknown_patch_lists_the_real_ones(tube_session):
    with pytest.raises(ValueError, match=r"no boundary patch 'front'.*inlet"):
        patch_geometry(tube_session.volmesh, "front")


# ---------------------------------------------------------------------------
# scalar speed -> vector (D80)


def test_scalar_speed_resolves_along_the_inward_normal(tube_session):
    vol = tube_session.volmesh
    vector, geo = resolve_inlet_speed(vol, "inlet", 2.5)
    np.testing.assert_allclose(vector, 2.5 * np.asarray(geo.normal_in), atol=1e-12)
    # an axis-aligned normal is written with exact zeros, not 1e-17 noise
    assert sorted(abs(c) for c in vector)[:2] == [0.0, 0.0]
    assert math.hypot(*vector) == pytest.approx(2.5)


@pytest.mark.parametrize("speed", [0.0, -1.0, float("nan"), float("inf")])
def test_non_positive_or_non_finite_speed_is_refused(tube_session, speed):
    with pytest.raises(ValueError, match="positive finite"):
        resolve_inlet_speed(tube_session.volmesh, "inlet", speed)


def test_scalar_speed_on_a_curved_patch_is_refused(tube_session):
    with pytest.raises(ValueError, match="not a flat end cap"):
        resolve_inlet_speed(tube_session.volmesh, "wall", 1.0)


def test_parse_inlet_knob_forms():
    from meshforge.pipeline import _parse_inlet_knob

    assert _parse_inlet_knob("inlet=150") == ("inlet", 150.0)
    assert _parse_inlet_knob("front=0,0,1") == ("front", (0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="NAME=vx,vy,vz"):
        _parse_inlet_knob("front")  # the pre-D80 message still names the form
    with pytest.raises(ValueError, match="NAME=SPEED"):
        _parse_inlet_knob("front=1,2")
    with pytest.raises(ValueError, match="not a number"):
        _parse_inlet_knob("front=fast")
    with pytest.raises(ValueError, match="positive finite"):
        _parse_inlet_knob("front=-3")
    with pytest.raises(ValueError, match="invalid boundary-group name"):
        _parse_inlet_knob("bad name=3")


# ---------------------------------------------------------------------------
# inflow report (Q, Re, warnings)


def test_inflow_report_flow_rate_reynolds_and_litres(tube_session):
    geo = patch_geometry(tube_session.volmesh, "inlet")
    v = 150.0 * np.asarray(geo.normal_in)
    rep = inflow_report(geo, v, 3.3, "mm")
    assert rep["normal_speed"] == pytest.approx(150.0)
    assert rep["flow_rate"] == pytest.approx(150.0 * geo.area)
    assert rep["reynolds"] == pytest.approx(150.0 * geo.d_eq / 3.3)
    # mm^3/s -> L/min: x 1e-9 m^3/mm^3 x 1e3 L/m^3 x 60 s/min
    assert rep["flow_rate_l_per_min"] == pytest.approx(150.0 * geo.area * 6e-5)
    assert "flow_rate_l_per_min" not in inflow_report(geo, v, 3.3, None)
    note = inflow_notes(rep)[0]
    assert "L/min" in note and "Re " in note and "-0.0000" not in note


def test_inflow_notes_warn_on_outward_vector_and_high_re(tube_session):
    geo = patch_geometry(tube_session.volmesh, "inlet")
    outward = -1.0 * np.asarray(geo.normal_in)
    notes = inflow_notes(inflow_report(geo, outward, 1.0))
    assert any("does not point into the domain" in n for n in notes)
    fast = 1e4 * np.asarray(geo.normal_in)
    rep = inflow_report(geo, fast, 1.0)
    assert rep["reynolds"] > LAMINAR_RE_LIMIT
    assert any("laminar solution may not exist" in n for n in inflow_notes(rep))
    assert not any(
        "laminar solution" in n for n in inflow_notes(rep, laminar=False)
    )


# ---------------------------------------------------------------------------
# flow-ends picker


def test_flow_ends_picks_the_nearest_cap(tube_session):
    cl, vol = tube_session.centerline, tube_session.volmesh
    near_start = flow_ends(vol, cl.points[0] + 0.5 * cl.tangents[0])
    assert (near_start["inflow"]["name"], near_start["outflow"]["name"]) == ("inlet", "outlet")
    near_end = flow_ends(vol, cl.points[-1])
    assert (near_end["inflow"]["name"], near_end["outflow"]["name"]) == ("outlet", "inlet")
    assert set(near_end["inflow"]) >= {"area", "d_eq", "normal_in", "centroid"}


def test_flow_ends_refuses_bad_points_and_ambiguous_meshes():
    from meshforge.meshing.base import split_boundary_group

    s = _fresh_tube_session()
    with pytest.raises(ValueError, match="three finite numbers"):
        flow_ends(s.volmesh, [0.0, float("nan"), 1.0])
    with pytest.raises(ValueError, match="three finite numbers"):
        flow_ends(s.volmesh, [0.0, 1.0])
    # half of the outlet becomes its own flat patch: three caps, no guess
    split_boundary_group(s.volmesh, "outlet", axis="x", threshold=0.0, name="outlet_b")
    with pytest.raises(ValueError, match="exactly two flat end caps; this mesh has 3"):
        flow_ends(s.volmesh, s.centerline.points[0])


# ---------------------------------------------------------------------------
# presets + source length units


def test_cardiac_presets_follow_the_length_unit():
    mm = cardiac_presets("mm")
    m = cardiac_presets("m")
    assert {p["id"]: p["value"] for p in mm["speed"]} == {"aorta_mean": 150.0, "aorta_peak": 1000.0}
    assert {p["id"]: p["value"] for p in m["speed"]} == {"aorta_mean": 0.15, "aorta_peak": 1.0}
    assert mm["nu"][0]["value"] == 3.3 and mm["nu"][0]["unit"] == "mm^2/s"
    assert m["nu"][0]["value"] == pytest.approx(3.3e-6)
    assert mm["time_unit"] == "s"
    with pytest.raises(ValueError, match="length unit"):
        cardiac_presets(None)
    with pytest.raises(ValueError, match="no cardiac presets for length unit 'furlong'"):
        cardiac_presets("furlong")


def test_length_unit_follows_the_source(tmp_path):
    from meshforge.pipeline import MeshingSession

    s = MeshingSession()
    assert s.length_unit is None
    s.generate_synth("straight_tube")
    assert s.length_unit is None
    with pytest.raises(ValueError, match="length unit"):
        s.cfd_presets()
    s.length_unit = "mm"  # a millimetre source (segment/sac set this)
    assert s.cfd_presets()["speed"][0]["unit"] == "mm/s"
    stl = tmp_path / "tube.stl"
    s.surface.save(stl)
    s.load(stl)
    assert s.length_unit is None  # an STL declares no unit


# ---------------------------------------------------------------------------
# the cfd step (OpenFOAM stubbed): scalar form end to end


def test_cfd_scalar_inlet_writes_the_resolved_vector(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = _fresh_tube_session()
    calls = _stub_solve(monkeypatch, s)
    geo = patch_geometry(s.volmesh, "inlet")
    result = s.compute_cfd(nu=0.5, inlet="inlet=2", outlet="outlet=0")

    entry = _u_entry(calls["case"], "inlet")
    vec = [float(v) for v in re.search(r"uniform \(([^)]*)\)", entry).group(1).split()]
    np.testing.assert_allclose(vec, 2.0 * np.asarray(geo.normal_in), atol=1e-9)
    assert "zeroGradient" in _u_entry(calls["case"], "outlet")
    # recorded as typed: a replay re-resolves against the same mesh
    assert s.journal[-1]["params"]["inlet"] == "inlet=2"
    inflow = result.summary["inflow"]
    assert inflow["patch"] == "inlet"
    assert inflow["flow_rate"] == pytest.approx(2.0 * geo.area)
    assert inflow["reynolds"] == pytest.approx(2.0 * geo.d_eq / 0.5)
    assert "inflow 'inlet'" in result.text
    # D82: a centreline exists, so the result carries the flow section
    assert result.render_hints["section"].n_cells > 0
    assert len(result.render_hints["view_dir"]) == 3


def test_cfd_vector_inlet_still_works_and_warns_when_outward(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = _fresh_tube_session()
    calls = _stub_solve(monkeypatch, s)
    n_in = np.asarray(patch_geometry(s.volmesh, "inlet").normal_in)
    out = -n_in  # points OUT of the domain: allowed, but loudly flagged
    result = s.compute_cfd(
        nu=0.5, inlet=f"inlet={out[0]:.6g},{out[1]:.6g},{out[2]:.6g}", outlet="outlet=0"
    )
    assert "fixedValue" in _u_entry(calls["case"], "inlet")
    assert "does not point into the domain" in result.text


def test_cfd_scalar_inlet_errors_before_or_without_a_run(tmp_path, monkeypatch):
    from meshforge.pipeline import MeshingSession

    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="volmesh"):
        MeshingSession().compute_cfd(inlet="inlet=1", outlet="outlet=0")
    s = _fresh_tube_session()
    calls = _stub_solve(monkeypatch, s)
    with pytest.raises(ValueError, match="not a flat end cap"):
        s.compute_cfd(inlet="wall=1", outlet="outlet=0")
    with pytest.raises(ValueError, match="no boundary patch 'front'"):
        s.compute_cfd(inlet="front=1", outlet="outlet=0")
    assert "case" not in calls  # nothing ran


def test_cfd_without_a_centreline_says_why_there_is_no_section(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = _fresh_tube_session()
    s.centerline = None
    _stub_solve(monkeypatch, s)
    result = s.compute_cfd(nu=0.5, inlet="inlet=1", outlet="outlet=0")
    assert "section" not in result.render_hints
    assert "no session centreline" in result.text


def test_cli_cfd_refuses_the_scalar_form_with_a_reason(tmp_path, capsys):
    from meshforge.cli import main

    rc = main(["cfd", str(tmp_path), "--inlet", "front=1", "--outlet", "back=0"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "needs the mesh in memory" in err
    # the remedy must exist: recipes have no cfd step, replay does
    assert "meshforge replay" in err and "recipe" not in err


# ---------------------------------------------------------------------------
# centreline flow section (D82)


def test_section_follows_a_curved_tube_and_stays_inside():
    import pyvista as pv

    from meshforge.geometry.ribbon import centerline_ribbon
    from meshforge.pipeline import MeshingSession

    s = MeshingSession()
    s.generate_synth("curved_tube")
    s.compute_centerline()
    s.compute_lumen(n_axial=16, n_theta=16)
    cl, grid = s.centerline, s.volmesh.grid
    sec = centerline_ribbon(cl, grid)
    assert sec.surface.n_cells > 0 and sec.n_rows == len(cl.points)
    # the bend lies in a plane: the view direction is that plane's normal
    assert np.abs(cl.tangents @ np.asarray(sec.view_dir)).max() < 0.05
    # every section point is inside the volume (the display probe is total)
    mask = pv.PolyData(np.asarray(sec.surface.points)).sample(grid)["vtkValidPointMask"]
    assert np.all(np.asarray(mask) == 1)
    # and the section runs through the centreline
    tree_d = np.min(
        np.linalg.norm(cl.points[:, None, :] - np.asarray(sec.surface.points)[None], axis=2),
        axis=1,
    )
    assert tree_d.max() < 1e-6


def test_section_rows_stop_at_the_first_wall_they_leave_through():
    """A row wide enough to reach a SECOND body (the descending aorta behind
    the arch) must stop where it leaves the first."""
    import pyvista as pv

    from meshforge.geometry.centerline import Centerline
    from meshforge.geometry.ribbon import centerline_ribbon

    def box(x0, x1):
        img = pv.ImageData(dimensions=(9, 9, 21), spacing=((x1 - x0) / 8, 0.25, 0.5),
                          origin=(x0, -1.0, 0.0))
        return img.cast_to_unstructured_grid()

    volume = box(-1.0, 1.0).merge(box(3.0, 5.0))  # body A, gap, body B
    z = np.linspace(0.5, 9.5, 19)
    pts = np.column_stack([0.05 * np.sin(z / 3.0), np.zeros_like(z), z])  # bends in x-z
    tan = np.gradient(pts, axis=0)
    tan /= np.linalg.norm(tan, axis=1)[:, None]
    cl = Centerline(
        points=pts, tangents=tan, arc_lengths=z - z[0], radii=np.full(len(z), 1.0),
        length=float(z[-1] - z[0]), closed=False, voxel_size=0.1,
    )
    sec = centerline_ribbon(cl, volume, margin=6.0)  # rows reach x = +-6
    xs = np.asarray(sec.surface.points)[:, 0]
    assert xs.max() <= 1.0 + 1e-9  # never jumps the gap into body B
    assert xs.min() >= -1.0 - 1e-9
    assert xs.max() > 0.8  # but does reach A's wall


def test_section_refuses_a_centreline_outside_the_volume():
    import pyvista as pv

    from meshforge.geometry.centerline import Centerline
    from meshforge.geometry.ribbon import centerline_ribbon

    volume = pv.ImageData(dimensions=(5, 5, 5)).cast_to_unstructured_grid()
    z = np.linspace(10.0, 12.0, 5)
    pts = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    pts[:, 0] += 0.01 * z**2
    tan = np.gradient(pts, axis=0)
    tan /= np.linalg.norm(tan, axis=1)[:, None]
    cl = Centerline(points=pts, tangents=tan, arc_lengths=z - z[0], radii=np.ones(5),
                    length=2.0, closed=False, voxel_size=0.1)
    with pytest.raises(ValueError, match="does not run inside the volume"):
        centerline_ribbon(cl, volume)


# ---------------------------------------------------------------------------
# OpenFOAM-gated: the scalar and vector forms of one inlet solve identically

needs_foam = pytest.mark.skipif(find_foam() is None, reason="OpenFOAM not available")


@needs_foam
def test_scalar_and_vector_inlets_solve_identically(tmp_path, monkeypatch):
    from test_cfd import _disk_wall

    from meshforge.pipeline import MeshingSession

    monkeypatch.chdir(tmp_path)
    results = {}
    for form in ("front=1", "front=0,0,1"):  # the pipe runs +z from z = 0
        s = MeshingSession()
        s.volmesh = _disk_wall(0.35, 4.0, 8)
        r = s.compute_cfd(nu=0.1, inlet=form, outlet="back=0", end_time=1500.0)
        assert any("converged" in n for n in r.summary["notes"])
        results[form] = (s.cfd.u_max, s.cfd.p_min, s.cfd.p_max, r.summary["inflow"])
    a, b = results["front=1"], results["front=0,0,1"]
    assert a[:3] == pytest.approx(b[:3], rel=1e-9)
    assert a[3]["velocity"] == pytest.approx([0.0, 0.0, 1.0])
    assert a[3]["flow_rate"] == pytest.approx(b[3]["flow_rate"])
