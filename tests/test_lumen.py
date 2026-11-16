"""V47: structured lumen (blood-volume) fill — swept O-grid + guarded
smoothing, tet fallback (V47, decision D75); end stations anchored on flat
end caps (D81)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from meshforge.geometry.centerline import Centerline
from meshforge.meshing.lumen import fill_lumen
from meshforge.pipeline import MeshingSession, load_workfile, replay
from meshforge.synth import generate


def _tube_radius_length(surf, truth):
    b = surf.bounds
    return (b[1] - b[0]) / 2.0, truth.length


def _patch_area(wm, name):
    """Area of a boundary patch from the mesh itself (quads or triangles)."""
    p = np.asarray(wm.grid.points)
    f = wm.boundary[name].face_nodes
    if f.shape[1] == 4:  # planar quad: half the diagonals' cross product
        d = np.cross(p[f[:, 2]] - p[f[:, 0]], p[f[:, 3]] - p[f[:, 1]])
    else:
        d = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    return 0.5 * float(np.linalg.norm(d, axis=1).sum())


def _patch_nodes(wm, name):
    return np.asarray(wm.grid.points)[np.unique(wm.boundary[name].face_nodes)]


# ---------------------------------------------------------------------------
# structured O-grid path
# ---------------------------------------------------------------------------
def test_straight_tube_ogrid_quality_groups_volume():
    surf, truth = generate("straight_tube")
    wm = fill_lumen(surf, method="ogrid", n_theta=16, n_axial=10, smooth=0)
    assert wm.backend == "lumen-ogrid"
    q = wm.quality
    assert q.n_inverted == 0
    # the V26 O-grid layout floor (~0.7071) survives the straight sweep
    assert q.sj_min > 0.65
    assert set(wm.boundary) == {"wall", "inlet", "outlet"}
    n_disk_quads = wm.boundary["inlet"].n_faces
    assert wm.boundary["outlet"].n_faces == n_disk_quads
    # lateral faces: one per rim edge per station interval; the O-grid rim
    # carries 4*n_t edges and n_t tracks n_theta/4 -> n_theta edges.
    assert wm.boundary["wall"].n_faces == 9 * 16
    r, length = _tube_radius_length(surf, truth)
    exact = math.pi * r * r * length
    # a 16-gon ring inscribes the circle: area factor sin(2pi/16)/(2pi/16)
    assert abs(q.total_volume - exact) / exact < 0.06
    assert any(n.startswith("fold number") for n in wm.notes)


def test_curved_tube_guarded_smoothing_never_regresses():
    surf, _ = generate("curved_tube")
    raw = fill_lumen(surf, method="ogrid", n_theta=16, n_axial=12, smooth=0)
    sm = fill_lumen(surf, method="ogrid", n_theta=16, n_axial=12, smooth=4)
    assert raw.quality.n_inverted == 0
    assert sm.quality.n_inverted == 0
    # the do-no-harm guard: global sj_min can never regress
    assert sm.quality.sj_min >= raw.quality.sj_min - 1e-9
    assert any("guarded smoothing" in n for n in sm.notes)
    assert not any("guarded smoothing" in n for n in raw.notes)


def test_torus_closed_centerline_refused():
    surf, _ = generate("torus")
    with pytest.raises(ValueError, match="open centrelines only"):
        fill_lumen(surf, n_theta=16, n_axial=10, smooth=0)


def test_knob_refusals_are_loud():
    surf, _ = generate("straight_tube")
    with pytest.raises(ValueError, match="auto \\| ogrid \\| tet"):
        fill_lumen(surf, method="hex")
    with pytest.raises(ValueError, match="too coarse"):
        fill_lumen(surf, n_theta=7)
    with pytest.raises(ValueError, match="too coarse"):
        fill_lumen(surf, n_axial=1)
    with pytest.raises(ValueError, match="smooth iterations"):
        fill_lumen(surf, smooth=-1)
    with pytest.raises(ValueError, match="mesh_size must be positive"):
        fill_lumen(surf, mesh_size=0.0)


def test_ogrid_count_overrides_validate():
    from meshforge.meshing.ogrid import ogrid_disk

    theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    loop = np.column_stack([np.cos(theta), np.sin(theta)])
    with pytest.raises(ValueError, match="n_t override"):
        ogrid_disk([loop], target_size=0.2, n_t=1)
    with pytest.raises(ValueError, match="n_r override"):
        ogrid_disk([loop], target_size=0.2, n_r=0)
    bm = ogrid_disk([loop], target_size=0.2, n_t=5, n_r=3)
    # pinned counts land in the layout note verbatim
    assert "n_t 5" in bm.notes[0] and "n_r 3" in bm.notes[0]


# ---------------------------------------------------------------------------
# tet fallback path (gmsh-gated)
# ---------------------------------------------------------------------------
def test_tet_method_direct():
    pytest.importorskip("gmsh")
    surf, truth = generate("straight_tube")
    wm = fill_lumen(surf, method="tet", n_theta=16, n_axial=10)
    assert wm.backend == "lumen-tet"
    assert wm.quality.n_inverted == 0
    assert set(wm.boundary) == {"wall", "inlet", "outlet"}
    r, length = _tube_radius_length(surf, truth)
    exact = math.pi * r * r * length
    assert abs(wm.quality.total_volume - exact) / exact < 0.06


def test_auto_falls_back_to_tet_with_loud_note(monkeypatch):
    pytest.importorskip("gmsh")
    import meshforge.meshing.lumen as lumen_mod

    def boom(*a, **k):
        raise ValueError("synthetic structured failure")

    monkeypatch.setattr(lumen_mod, "_station_disks", boom)
    surf, _ = generate("straight_tube")
    wm = fill_lumen(surf, method="auto", n_theta=16, n_axial=10, smooth=0)
    assert wm.backend == "lumen-tet"
    fallback = [n for n in wm.notes if "structured O-grid fill failed" in n]
    assert fallback and "fold number" in fallback[0]
    # a forced ogrid keeps the failure LOUD instead of falling back
    with pytest.raises(ValueError, match="synthetic structured failure"):
        fill_lumen(surf, method="ogrid", n_theta=16, n_axial=10, smooth=0)


# ---------------------------------------------------------------------------
# end caps (D81): the inlet/outlet patch IS the surface's flat end cap
# ---------------------------------------------------------------------------
_R, _L, _TILT_IN, _TILT_OUT = 3.0, 30.0, 25.0, 35.0


def _tilted_cap_tube():
    """A straight tube along +z whose two flat caps are tilted 25° and 35°.

    z is sheared near each end — z' = z + y·tan(25°)·w_in(z) and
    z' = z + x·tan(35°)·w_out(z), w ramping 1 -> 0 over 6 from the cap — so
    x, y (and the wall's cylinder) are untouched, every ring stays planar and
    each cap stays a flat 64-gon on its tilted plane. Returns (surface,
    centreline, caps): the centreline is the tube axis, ending 0.05 inside each
    cap centre — a centreline-normal end station there sends half its rays out
    through the cap within ~0.1 of the axis (the defect). caps[end] = (exact
    cap area, outward unit normal, cap centre)."""
    surf, _ = generate("straight_tube", radius=_R, length=_L, n_theta=64, n_axial=41)
    p = np.asarray(surf.points, dtype=float).copy()
    blend = 6.0
    w_in = np.clip(1.0 - p[:, 2] / blend, 0.0, 1.0)
    w_out = np.clip(1.0 - (_L - p[:, 2]) / blend, 0.0, 1.0)
    t_in, t_out = math.tan(math.radians(_TILT_IN)), math.tan(math.radians(_TILT_OUT))
    p[:, 2] += p[:, 1] * t_in * w_in + p[:, 0] * t_out * w_out
    surf.points = p
    z = np.linspace(0.05, _L - 0.05, 201)
    cl = Centerline(
        points=np.column_stack([np.zeros_like(z), np.zeros_like(z), z]),
        tangents=np.tile([0.0, 0.0, 1.0], (len(z), 1)),
        arc_lengths=z - z[0],
        radii=np.full(len(z), _R),
        length=float(z[-1] - z[0]),
        closed=False,
        voxel_size=0.0,
    )
    a_xy = 0.5 * 64 * _R**2 * math.sin(2.0 * math.pi / 64)  # the 64-gon in plan
    n_in = np.array([0.0, t_in, -1.0]) / math.hypot(t_in, 1.0)
    n_out = np.array([-t_out, 0.0, 1.0]) / math.hypot(t_out, 1.0)
    caps = {
        "inlet": (a_xy / math.cos(math.radians(_TILT_IN)), n_in, np.zeros(3)),
        "outlet": (
            a_xy / math.cos(math.radians(_TILT_OUT)), n_out, np.array([0.0, 0.0, _L])
        ),
    }
    return surf, cl, caps


def _assert_patch_is_cap(wm, end, caps):
    """The patch lies ON its cap's plane, reaches the cap's rim, and matches
    the cap's area: 32 equal-angle spokes inscribed in the 64-gon rim (which
    the tilt stretches by 1/cos) fall up to ~1.2% short of it."""
    nodes = _patch_nodes(wm, end)
    area, normal, centre = min(
        caps.values(), key=lambda c: np.linalg.norm(nodes.mean(axis=0) - c[2])
    )
    assert np.abs((nodes - centre) @ normal).max() < 1e-9 * _L, end
    rho = np.hypot(nodes[:, 0], nodes[:, 1])  # distance from the tube axis
    assert rho.max() <= _R * (1 + 1e-9) and rho.max() >= _R * math.cos(math.pi / 64)
    assert _patch_area(wm, end) == pytest.approx(area, rel=0.02), end


@pytest.mark.parametrize("method", ["ogrid", "tet"])
def test_tilted_end_caps_patches_are_the_caps(method):
    if method == "tet":
        pytest.importorskip("gmsh")
    surf, cl, caps = _tilted_cap_tube()
    wm = fill_lumen(surf, method=method, centerline=cl, n_theta=32, n_axial=20)
    assert wm.backend == f"lumen-{method}"
    assert wm.quality.n_inverted == 0
    for end in ("inlet", "outlet"):
        _assert_patch_is_cap(wm, end, caps)
        anchored = f"{end}: anchored on its flat end cap"
        assert any(n.startswith(anchored) for n in wm.notes)
    # the whole tube is filled (the shear preserves volume): 32-gon x length
    exact = 0.5 * 32 * _R**2 * math.sin(2.0 * math.pi / 32) * _L
    assert wm.quality.total_volume == pytest.approx(exact, rel=0.01)


def test_tilted_end_caps_reproduce_the_clipped_patch_without_anchoring(monkeypatch):
    """The fixture is a faithful reproduction: with cap detection off the old
    centreline-normal end sections come back clipped to well under the cap."""
    import meshforge.meshing.lumen as lumen_mod

    monkeypatch.setattr(lumen_mod, "_end_caps", lambda *a, **k: {})
    surf, cl, caps = _tilted_cap_tube()
    wm = fill_lumen(surf, method="ogrid", centerline=cl, n_theta=32, n_axial=20)
    for end, (area, _, _) in caps.items():
        assert _patch_area(wm, end) < 0.8 * area, end
        assert any(n.startswith(f"{end}: no flat end cap found") for n in wm.notes)


def test_tilted_end_caps_with_the_default_centreline():
    """Same tube through the default path (skeleton centreline, no knob)."""
    surf, _, caps = _tilted_cap_tube()
    wm = fill_lumen(surf, method="ogrid", n_theta=32, n_axial=20)
    assert wm.quality.n_inverted == 0
    for end in ("inlet", "outlet"):  # (the skeleton may run either way)
        _assert_patch_is_cap(wm, end, caps)


def test_end_without_a_flat_cap_keeps_its_section_loudly():
    """Cone-shaped ends (each fan apex pushed out along the axis) have no flat
    cap: the old centreline-normal end section is kept and the notes say so."""
    surf, _ = generate("straight_tube", radius=_R, length=20.0, n_theta=64, n_axial=41)
    p = np.asarray(surf.points, dtype=float).copy()
    apex = np.nonzero(np.hypot(p[:, 0], p[:, 1]) < 1e-9)[0]
    assert len(apex) == 2
    p[apex, 2] += np.where(p[apex, 2] > 10.0, 1.0, -1.0)
    surf.points = p
    wm = fill_lumen(surf, method="ogrid", n_theta=16, n_axial=10, smooth=0)
    assert wm.quality.n_inverted == 0
    for end in ("inlet", "outlet"):
        assert any(n.startswith(f"{end}: no flat end cap found") for n in wm.notes)


def test_cap_filter_rejects_coplanar_wall_facets():
    """A flat region is only a cap if it closes a whole ring (>= 12 rim
    vertices) and creases against the wall (>= 10°). A straight tube's flat
    wall strips (64 of them, 5.6° creases) are coplanar disks but not caps;
    a 4-triangle planar fan is too small a ring to be one."""
    from meshforge.meshing import lumen as lumen_mod

    surf, _ = generate("straight_tube", radius=_R, length=20.0, n_theta=64, n_axial=41)
    tri, pts, regions = lumen_mod._coplanar_regions(surf)
    caps = [c for c in (lumen_mod._as_cap(tri, pts, *r) for r in regions) if c]
    assert len(regions) == 66 and len(caps) == 2
    assert sorted(len(c.loop) for c in caps) == [64, 64]

    def fan(n):
        th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        fpts = np.vstack([np.column_stack([np.cos(th), np.sin(th), 0 * th]), [0, 0, 0]])
        ftri = np.array([[n, j, (j + 1) % n] for j in range(n)])
        area = 0.5 * n * math.sin(2.0 * math.pi / n)
        return ftri, fpts, np.arange(n), area, np.zeros(3), np.array([0.0, 0.0, 1.0])

    assert lumen_mod._as_cap(*fan(4), 90.0) is None  # rim of 4 < 12
    assert lumen_mod._as_cap(*fan(16), 90.0) is not None
    assert lumen_mod._as_cap(*fan(16), 5.0) is None  # no crease: a wall facet


# ---------------------------------------------------------------------------
# session step / workfile / CLI
# ---------------------------------------------------------------------------
def test_session_step_records_and_replays(tmp_path):
    s = MeshingSession()
    s.generate_synth("straight_tube")
    r = s.compute_lumen(method="ogrid", n_theta=16, n_axial=10, smooth=0)
    assert r.step == "lumen"
    assert s.volmesh is not None and s.volmesh.backend == "lumen-ogrid"
    assert s.journal[-1]["step"] == "lumen"
    assert s.journal[-1]["params"]["method"] == "ogrid"
    wf = s.save_workfile(tmp_path / "lumen.mfwork")
    s2 = load_workfile(wf)
    replay(s2)
    assert s2.volmesh.backend == "lumen-ogrid"
    assert s2.volmesh.summary()["quality"]["n_inverted"] == 0


def test_cli_lumen_end_to_end(tmp_path, capsys):
    from meshforge import cli

    surf, _ = generate("straight_tube")
    src = tmp_path / "tube.vtp"
    surf.save(src)
    rc = cli.main(
        [
            "lumen", str(src),
            "--n-theta", "16", "--n-axial", "10", "--smooth", "0",
            "--outputs", str(tmp_path),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "lumen-ogrid" in out and "inverted=0" in out
    assert "fold number" in out
    assert (tmp_path / "tube_lumen.vtu").is_file()
    assert (tmp_path / "tube_lumen.summary.json").is_file()


def test_cli_lumen_bad_method_exits_nonzero(tmp_path, capsys):
    from meshforge import cli

    surf, _ = generate("straight_tube")
    src = tmp_path / "tube.vtp"
    surf.save(src)
    rc = cli.main(["lumen", str(src), "--method", "hex", "--outputs", str(tmp_path)])
    assert rc == 1
    assert "auto | ogrid | tet" in capsys.readouterr().err


# --- surface-tet fallback (a centreline that bends tighter than the vessel) --


def test_surface_tet_fills_the_surface_itself_and_tags_the_caps():
    surf, _ = generate("straight_tube")
    wm = fill_lumen(surf, method="surface-tet")
    assert wm.backend == "lumen-surface-tet"
    assert wm.quality.n_inverted == 0
    assert set(wm.boundary) == {"wall", "inlet", "outlet"}
    assert abs(wm.grid.volume) == pytest.approx(abs(surf.volume), rel=0.01)
    # each patch is the WHOLE cap after the gmsh remesh, not a rim-less core
    from meshforge.meshing.lumen import flat_cap_cells
    from meshforge.solve.inflow import patch_geometry

    tri = surf.triangulate()
    cap_areas = sorted(float(tri.extract_cells(c).area) for c in flat_cap_cells(tri))
    patch_areas = sorted(patch_geometry(wm, n).area for n in ("inlet", "outlet"))
    assert patch_areas == pytest.approx(cap_areas, rel=0.02)


def test_surface_tet_refuses_a_surface_without_two_flat_caps(monkeypatch):
    # inlet/outlet come from the two flat caps; one (or three) is ambiguous
    import meshforge.meshing.lumen as lumen

    real = lumen.flat_cap_cells
    monkeypatch.setattr(lumen, "flat_cap_cells", lambda s: real(s)[:1])
    surf, _ = generate("straight_tube")
    with pytest.raises(ValueError, match="exactly two flat end caps"):
        fill_lumen(surf, method="surface-tet")


def test_auto_falls_back_to_surface_tet_when_the_ring_shell_is_invalid(monkeypatch):
    # A centreline bending tighter than the vessel makes the station rings
    # overlap: the O-grid fails AND the ring shell self-intersects. Simulate
    # both and check auto still delivers a mesh, loudly.
    import meshforge.meshing.lumen as lumen

    def bad_sweep(*a, **k):
        raise ValueError("simulated O-grid failure")

    real_tet_fill = lumen._tet_fill
    calls = {"n": 0}

    def tet_fill(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:  # the station-ring shell
            raise RuntimeError("lumen tet fill failed — simulated self-intersecting ring shell")
        return real_tet_fill(*a, **k)

    monkeypatch.setattr(lumen, "_sweep", bad_sweep)
    monkeypatch.setattr(lumen, "_tet_fill", tet_fill)
    surf, _ = generate("straight_tube")
    wm = fill_lumen(surf, method="auto")
    assert wm.backend == "lumen-surface-tet"
    assert wm.quality.n_inverted == 0
    assert any("surface-tet fallback" in n for n in wm.notes)
