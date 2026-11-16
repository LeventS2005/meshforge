"""Parallel-plane swept wall (D87): stations are sections of planes parallel
to two flat parallel end caps.

Ground truth is the synthetic ``eccentric_sac``: a 60 mm segment between the
caps z = 0 and z = 60 whose every z-section is an exact circle — a 30 mm sac
whose centre sits 8 mm off the neck axis, a shoulder into a 12 mm neck and a
15 mm sideways kink into it. The centreline-ray wall folds on it (the same
family as the real Anev_02 sac and Zenodo P05/P16); the plane sweep must be
clean, keep the thickness along the surface normal, keep the caps planar
and match the analytic shell volume.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest
import pyvista as pv

from meshforge.meshing import get_backend
from meshforge.meshing.planes import build_plane_lattice, parallel_caps
from meshforge.synth import curved_tube, extrude_loops, straight_tube, tapered_tube
from meshforge.synth.generators import _lattice_tube, eccentric_sac, eccentric_sac_profile

T = 1.5  # wall thickness, mm
S, N, L = 60, 32, 4  # default stations, samples, layers


@pytest.fixture(scope="module")
def sac():
    return eccentric_sac()


@pytest.fixture(scope="module")
def reference():
    """The same profile, finer and run 20 mm past both caps: the exact
    surface the wall's thickness is measured against."""
    ref, _ = eccentric_sac(z0=-20.0, z1=80.0, n_theta=360, n_axial=1001)
    return ref


def _centre_path(truth):
    from meshforge.geometry.crop import _as_centerline

    idx = np.linspace(0, len(truth.points) - 1, 241).astype(int)
    return _as_centerline(truth.points[idx], truth.radii[idx], "eccentric_sac centre path")


def _wall(surface, **kw):
    kw.setdefault("curvature_cap", 0.9)
    return get_backend("swept").mesh_wall(surface, T, **kw)


def _layers(wall, stations=S, samples=N):
    return np.asarray(wall.grid.points).reshape(stations, samples, L + 1, 3)


def _analytic_shell_volume(side: str) -> float:
    """∫dz ∮ the annulus between each circle and its in-plane offset by
    t·sqrt(1 + (dρ/dz)²) — a wall of normal thickness t cut by the planes."""
    z = np.linspace(0.0, 60.0, 6001)
    th = np.linspace(0.0, 2.0 * math.pi, 721)[:-1]
    cx, cy, r = eccentric_sac_profile(z)
    slope = (np.gradient(cx, z)[:, None] * np.cos(th) + np.gradient(cy, z)[:, None] * np.sin(th)
             + np.gradient(r, z)[:, None])
    d = T * np.sqrt(1.0 + slope**2)
    R = r[:, None]
    ring = 0.5 * ((R + d) ** 2 - R**2) if side == "outward" else 0.5 * (R**2 - (R - d) ** 2)
    return float(np.trapezoid(ring.mean(axis=1) * 2.0 * math.pi, z))


# ---------------------------------------------------------------------------
# the failure and the fix
# ---------------------------------------------------------------------------
def test_eccentric_sac_is_a_segment_between_parallel_caps(sac):
    surface, truth = sac
    assert surface.n_open_edges == 0 and surface.is_manifold
    assert surface.volume == pytest.approx(truth.volume, rel=0.005)
    caps = parallel_caps(surface)
    assert caps.angle_deg < 1e-9 and caps.separation == pytest.approx(60.0, abs=1e-9)
    assert caps.first.centre[2] == pytest.approx(0.0, abs=1e-12)  # caudal cap first


def test_centreline_sweep_folds_on_the_eccentric_sac(sac):
    surface, truth = sac
    wall = _wall(surface, centerline=_centre_path(truth))
    lat = wall.lattice
    assert lat["sweep"] == "centreline" and lat["fold_number"] > 1.0
    assert wall.quality.n_inverted > 100
    assert lat["n_capped"] > 0 and lat["n_floored"] > 0


def test_plane_sweep_passes_the_wall_gate(sac):
    surface, _ = sac
    wall = _wall(surface, sweep="planes")
    lat = wall.lattice
    assert wall.quality.n_inverted == 0 and wall.quality.sj_min > 0.3
    assert lat["n_capped"] == 0 and lat["n_floored"] == 0
    assert lat["sweep"] == "planes" and lat["side"] == "outward"
    assert lat["n_rays"] == S * N and lat["cap_angle_deg"] == 0.0
    assert lat["dropped_loops"] == 0 and lat["surface_offset_max"] < 0.02
    assert 50.0 < lat["tilt_max_deg"] < 70.0  # the kink side of the neck
    assert wall.grid.n_cells == (S - 1) * N * L
    faces = {k: g.n_faces for k, g in wall.boundary.items()}
    assert faces == {"inner": (S - 1) * N, "outer": (S - 1) * N, "inlet": N * L, "outlet": N * L}


@pytest.mark.parametrize("side", ["outward", "inward"])
def test_plane_sweep_keeps_the_thickness_along_the_normal(sac, reference, side):
    surface, _ = sac
    wall = _wall(surface, sweep="planes", side=side)
    X = _layers(wall)
    far = X[:, :, L if side == "outward" else 0].reshape(-1, 3)
    near = X[:, :, 0 if side == "outward" else L].reshape(-1, 3)

    def dist(p):
        return np.abs(pv.PolyData(p).compute_implicit_distance(reference)["implicit_distance"])

    d = dist(far)
    # measured: outward 1.404..1.608, inward 1.395..1.582 (tilt up to 60 deg)
    assert d.min() > 0.9 * T and d.max() < 1.1 * T
    assert np.median(d) == pytest.approx(T, rel=0.01)
    assert dist(near).max() < 0.02  # the input's own faceting (96 vs 360 per ring)
    # "inner" is the lumen side either way: the input surface when the wall
    # grows outward, the offset when it grows inward
    pts = np.asarray(wall.grid.points)
    inner_d = dist(pts[np.unique(wall.boundary["inner"].face_nodes)])
    if side == "outward":
        assert inner_d.max() < 0.02
    else:
        assert inner_d.min() > 0.9 * T


def test_plane_sweep_keeps_the_caps_planar(sac):
    surface, _ = sac
    wall = _wall(surface, sweep="planes")
    pts = np.asarray(wall.grid.points)
    inlet = np.unique(wall.boundary["inlet"].face_nodes)
    outlet = np.unique(wall.boundary["outlet"].face_nodes)
    assert np.abs(pts[inlet, 2]).max() < 1e-6
    assert np.abs(pts[outlet, 2] - 60.0).max() < 1e-6
    # every node of a station stays in its plane, all layers
    z = _layers(wall)[..., 2]
    assert np.ptp(z.reshape(S, -1), axis=1).max() < 1e-9


@pytest.mark.parametrize("side", ["outward", "inward"])
def test_plane_sweep_shell_volume_matches_the_analytic_shell(sac, side):
    surface, _ = sac
    wall = _wall(surface, sweep="planes", side=side)
    # measured: -0.75% outward, -0.73% inward (32-gon sections)
    assert wall.quality.total_volume == pytest.approx(_analytic_shell_volume(side), rel=0.015)


# ---------------------------------------------------------------------------
# bad sections: counted or refused, never silent
# ---------------------------------------------------------------------------
def _notch(n=400, radius=10.0, depth=4.0, width=0.12):
    th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    a = np.angle(np.exp(1j * th))
    r = radius - depth * np.exp(-((a / width) ** 2))
    return np.column_stack([r * np.cos(th), r * np.sin(th)])


def _banana(n=120):
    """An annular sector (radii 6..10, ±120 deg) with round ends: its
    centroid (r ~ 2.7) lies outside it — not star-shaped about it."""
    a0, a1 = -2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0

    def end(a, betas):  # semicircle of radius 2 joining the arcs at angle a
        v = np.array([math.cos(a), math.sin(a)])
        u = np.array([-math.sin(a), math.cos(a)])
        return [tuple(8.0 * v + 2.0 * (math.cos(b) * v + math.sin(b) * u)) for b in betas]

    out = [(10.0 * math.cos(a), 10.0 * math.sin(a)) for a in np.linspace(a0, a1, n)]
    out += end(a1, np.linspace(0.0, math.pi, 24)[1:-1])
    out += [(6.0 * math.cos(a), 6.0 * math.sin(a)) for a in np.linspace(a1, a0, n)]
    out += end(a0, np.linspace(math.pi, 2.0 * math.pi, 24)[1:-1])
    return np.array(out)


def _prism(poly, height=20.0):
    surface, _ = extrude_loops([poly], distance=height)
    return surface.triangulate()


def test_sections_need_not_be_star_shaped():
    poly = _banana()
    x, y = poly[:, 0], poly[:, 1]
    cr = x * np.roll(y, -1) - np.roll(x, -1) * y
    centroid = np.array([((x + np.roll(x, -1)) * cr).sum(), ((y + np.roll(y, -1)) * cr).sum()])
    centroid /= 3.0 * cr.sum()
    assert np.linalg.norm(centroid) < 6.0  # outside the section: a ray fan would fail
    wall = get_backend("swept").mesh_wall(_prism(poly), 0.5, sweep="planes", n_theta=64,
                                          n_axial=10, curvature_cap=0.9)
    assert wall.quality.n_inverted == 0 and wall.quality.sj_min > 0.5
    assert wall.lattice["n_capped"] == 0 and wall.lattice["n_floored"] == 0


def test_a_concavity_tighter_than_the_offset_is_counted():
    notch = _prism(_notch())
    tight = get_backend("swept").mesh_wall(notch, T, sweep="planes", n_theta=96, n_axial=10,
                                           curvature_cap=0.9)
    lat = tight.lattice
    assert lat["n_capped_concave"] > 0 and lat["n_capped"] == lat["n_capped_concave"]
    assert tight.quality.n_inverted == 0  # capped, not folded
    assert any("concavity cap" in n for n in tight.notes)
    assert lat["thickness_min"] < T
    thin = get_backend("swept").mesh_wall(notch, 0.05, sweep="planes", n_theta=96, n_axial=10,
                                          curvature_cap=0.9)
    assert thin.lattice["n_capped"] == 0 and thin.quality.n_inverted == 0


def test_an_inward_wall_thicker_than_the_section_is_counted():
    tube, _ = straight_tube(radius=1.0, length=10.0)
    wall = get_backend("swept").mesh_wall(tube, T, sweep="planes", side="inward",
                                          curvature_cap=0.9)
    assert wall.lattice["n_capped_concave"] == wall.lattice["n_rays"]
    assert wall.lattice["thickness_max"] < 1.0


def test_the_tilt_bound_is_counted():
    cone, _ = tapered_tube()
    steep = get_backend("swept").mesh_wall(cone, 0.5, sweep="planes", max_tilt=1.0)
    assert steep.lattice["n_capped_tilt"] > 0
    assert steep.lattice["n_capped"] >= steep.lattice["n_capped_tilt"]
    assert any("tilt cap" in n for n in steep.notes)
    ok = get_backend("swept").mesh_wall(cone, 0.5, sweep="planes")
    assert ok.lattice["n_capped"] == 0 and ok.lattice["max_tilt_deg"] == 80.0


def test_refuses_caps_that_are_not_parallel():
    bent, _ = curved_tube()
    with pytest.raises(ValueError, match="deg from parallel"):
        get_backend("swept").mesh_wall(bent, 0.5, sweep="planes")


def _zigzag_tube(radius=3.0, bend=8.0, n_theta=48):
    """Up 0->40, over a semicircle, down to z=20, under one, up to z=60:
    parallel caps at z=0 and z=60, but a plane between them meets 2-3 legs."""
    def line(p, q, n):
        t = np.linspace(0.0, 1.0, n)[:, None]
        return p + t * (q - p), np.tile((q - p) / np.linalg.norm(q - p), (n, 1))

    def arc(cx, cz, a0, a1, n, sign):
        a = np.linspace(a0, a1, n)
        pts = np.column_stack([cx + bend * np.cos(a), np.zeros(n), cz + bend * np.sin(a)])
        return pts, sign * np.column_stack([-np.sin(a), np.zeros(n), np.cos(a)])

    parts = [line(np.array([0.0, 0, 0]), np.array([0.0, 0, 40]), 41),
             arc(bend, 40.0, math.pi, 0.0, 41, -1.0),
             line(np.array([2 * bend, 0, 40.0]), np.array([2 * bend, 0, 20.0]), 21),
             arc(3 * bend, 20.0, math.pi, 2 * math.pi, 41, 1.0),
             line(np.array([4 * bend, 0, 20.0]), np.array([4 * bend, 0, 60.0]), 41)]
    pts = np.vstack([p[0][1 if k else 0:] for k, p in enumerate(parts)])
    tan = np.vstack([p[1][1 if k else 0:] for k, p in enumerate(parts)])
    b = np.tile([0.0, 1.0, 0.0], (len(pts), 1))
    return _lattice_tube(pts, np.cross(b, tan), b, np.full(len(pts), radius), n_theta)


def test_refuses_a_plane_that_meets_the_segment_twice():
    tube = _zigzag_tube()
    assert tube.n_open_edges == 0 and parallel_caps(tube).angle_deg < 1e-9
    with pytest.raises(ValueError, match="meets the surface in 2 loops"):
        get_backend("swept").mesh_wall(tube, 0.5, sweep="planes")


def test_knob_validation():
    tube, _ = straight_tube()
    backend = get_backend("swept")
    with pytest.raises(ValueError, match="sweep must be one of"):
        backend.mesh_wall(tube, 0.5, sweep="rays")
    with pytest.raises(ValueError, match="inward wall needs the plane sweep"):
        backend.mesh_wall(tube, 0.5, side="inward")
    with pytest.raises(ValueError, match="plane sweep only"):
        backend.mesh_wall(tube, 0.5, max_tilt=60.0)
    with pytest.raises(ValueError, match="plane sweep only"):
        backend.mesh_wall(tube, 0.5, section_smoothing=1.0)
    with pytest.raises(ValueError, match="curvature_cap < 1"):
        backend.mesh_wall(tube, 0.5, sweep="planes", curvature_cap=1.0)
    from meshforge.geometry.thickness import ThicknessSpec

    spec = ThicknessSpec.from_outer_surface(straight_tube(radius=1.5)[0])
    with pytest.raises(ValueError, match="scalar thickness"):
        build_plane_lattice(tube, spec)


def test_section_smoothing_moves_nodes_and_says_how_far():
    notch = _prism(_notch())
    exact = get_backend("swept").mesh_wall(notch, T, sweep="planes", n_theta=96, n_axial=10,
                                           curvature_cap=0.9)
    smooth = get_backend("swept").mesh_wall(notch, T, sweep="planes", n_theta=96, n_axial=10,
                                            curvature_cap=0.9, section_smoothing=2.0)
    assert exact.lattice["surface_offset_max"] < 1e-6
    lat = smooth.lattice
    assert lat["section_smoothing"] == 2.0 and lat["surface_offset_max"] > 0.5
    assert lat["n_capped"] < exact.lattice["n_capped"]
    assert any("sections smoothed" in n and "from the input surface" in n for n in smooth.notes)
    with pytest.raises(ValueError, match="section_smoothing"):
        build_plane_lattice(notch, T, section_smoothing=-1.0)


# ---------------------------------------------------------------------------
# sweep="auto" and the unchanged default
# ---------------------------------------------------------------------------
def test_default_sweep_is_the_centreline_sweep_unchanged():
    tube, _ = straight_tube()
    default = get_backend("swept").mesh_wall(tube, 0.5)
    explicit = get_backend("swept").mesh_wall(tube, 0.5, sweep="centreline")
    assert np.array_equal(default.grid.points, explicit.grid.points)
    assert default.lattice["sweep"] == "centreline"
    assert default.notes[0].startswith("lattice 60 stations")


def test_auto_takes_the_planes_only_when_the_centreline_wall_is_not_clean(sac):
    surface, truth = sac
    auto = _wall(surface, centerline=_centre_path(truth), sweep="auto")
    assert auto.lattice["sweep"] == "planes" and auto.quality.n_inverted == 0
    assert auto.notes[0].startswith("auto: plane sweep")
    assert auto.lattice["auto_centreline"]["inverted"] > 100

    tube, _ = straight_tube()  # parallel caps, clean centreline wall
    kept = get_backend("swept").mesh_wall(tube, 0.5, sweep="auto")
    plain = get_backend("swept").mesh_wall(tube, 0.5)
    assert kept.lattice["sweep"] == "centreline"
    assert np.array_equal(kept.grid.points, plain.grid.points)
    assert kept.notes[0].startswith("auto: centreline sweep (clean wall")

    bent, _ = curved_tube(radius=2.0, bend_radius=4.0)  # folds, caps not parallel
    held = get_backend("swept").mesh_wall(bent, 1.5, sweep="auto", n_theta=16, n_axial=30)
    assert held.lattice["sweep"] == "centreline" and held.lattice["n_capped"] > 0
    assert "the plane sweep does not apply" in held.notes[0]


# ---------------------------------------------------------------------------
# service layer + CLI
# ---------------------------------------------------------------------------
def test_session_volmesh_plane_sweep_journal_and_replay(tmp_path):
    from meshforge.pipeline import STEP_KNOBS, MeshingSession, load_workfile, replay

    assert STEP_KNOBS["volmesh"]["sweep"][1] == "centreline"
    assert STEP_KNOBS["volmesh"]["side"][1] == "outward"
    s = MeshingSession()
    s.generate_synth("aneurysm_tube")
    s.compute_crop()
    r = s.compute_volmesh(thickness=1.5, curvature_cap=0.9, sweep="planes")
    assert r.summary["lattice"]["sweep"] == "planes" and "plane sweep (outward)" in r.text
    assert r.summary["quality"]["n_inverted"] == 0
    assert s.journal[-1]["params"]["sweep"] == "planes"
    cells = s.volmesh.grid.n_cells
    wf = s.save_workfile(tmp_path / "planes.mfwork")
    fresh = load_workfile(wf)
    assert [x.step for x in replay(fresh)] == ["generate_synth", "crop", "volmesh"]
    assert fresh.volmesh.lattice["sweep"] == "planes" and fresh.volmesh.grid.n_cells == cells


def test_cli_volmesh_plane_sweep(tmp_path, capsys, sac):
    from meshforge.cli import main

    surface, _ = sac
    src = tmp_path / "eccentric sac.vtp"
    surface.save(src)
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    out = tmp_path / "wall.vtu"
    rc = main(["volmesh", str(src), "--thickness", "1.5", "--curvature-cap", "0.9",
               "--sweep", "planes", "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert "inverted=0" in printed and "wall rays   : capped=0  floored=0" in printed
    assert "plane sweep : side=outward" in printed
    summary = json.loads(out.with_suffix(".summary.json").read_text())
    assert summary["lattice"]["sweep"] == "planes"
    assert hashlib.sha256(src.read_bytes()).hexdigest() == before
