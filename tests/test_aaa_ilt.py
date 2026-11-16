"""D91: intraluminal thrombus (ILT) as a tied solid in the AAA wall-stress model.

Mesh checks run everywhere. The two CalculiX benchmarks (two-layer Lamé, and
a soft thrombus passing the pressure through like a fluid) skip without ccx.
"""
from __future__ import annotations

import functools
import math

import numpy as np
import pytest

import meshforge.solve.aaa_metrics as am
from meshforge.geometry.crop import planes_from_report
from meshforge.pipeline import MeshingSession
from meshforge.solve import find_ccx
from meshforge.solve.ilt import build_ilt_mesh
from meshforge.synth import generate

P140 = 140 * 1.33322e-4  # MPa
requires_ccx = pytest.mark.skipif(find_ccx() is None, reason="CalculiX (ccx) not available")
L = 120.0


def _tube(radius: float):
    surf, _ = generate("straight_tube", radius=radius, length=L, n_theta=96, n_axial=121)
    return surf


def _report(surface) -> dict:
    """Crop-report planes on a straight tube's two flat end caps (z extremes)."""
    z0, z1 = surface.bounds[4], surface.bounds[5]
    return {"planes": {
        "distal": {"origin": [0.0, 0.0, z0], "normal_out": [0.0, 0.0, -1.0], "height_mm": 0.0},
        "proximal": {"origin": [0.0, 0.0, z1], "normal_out": [0.0, 0.0, 1.0], "height_mm": z1 - z0},
    }, "landmarks": None}


@functools.lru_cache(maxsize=None)
def _wall(radius: float):
    s = MeshingSession()
    s.generate_synth("straight_tube", radius=radius, length=L, n_theta=96, n_axial=121)
    s.compute_centerline(voxel_size=1.0)
    # the plane sweep runs cap to cap (end stations = the cap rims), like the demo walls
    s.compute_volmesh(thickness=1.5, n_theta=32, n_axial=40, n_layers=2, curvature_cap=0.9,
                      sweep="planes")
    assert s.volmesh.quality.n_inverted == 0
    return s.volmesh


def test_thrombus_mesh_is_the_shell_between_lumen_and_sac():
    sac, lumen = _tube(10.0), _tube(6.0)
    m = build_ilt_mesh(lumen, sac, planes_from_report(_report(sac)), size=3.0, inset=0.5)
    expected = (abs(sac.volume) - math.pi * 36.0 * (L - 1.0)) / 1000.0
    assert m.volume_ml == pytest.approx(expected, rel=0.02)
    assert m.tets.shape[1] == 10 and m.sj_min > 0.02
    assert m.inner_faces and m.outer_faces
    # straight mid-edge nodes, CalculiX C3D10 order (5 = 1-2, 8 = 1-4, 10 = 3-4)
    t = m.tets[0]
    p = m.points
    assert np.allclose(p[t[4]], 0.5 * (p[t[0]] + p[t[1]]))
    assert np.allclose(p[t[7]], 0.5 * (p[t[0]] + p[t[3]]))
    assert np.allclose(p[t[9]], 0.5 * (p[t[2]] + p[t[3]]))
    # every corner tet positively oriented (CalculiX refuses a negative Jacobian)
    a, b, c, d = (p[m.tets[:, i]] for i in range(4))
    assert (np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) > 0).all()


def test_thrombus_mesh_refuses_a_lumen_outside_the_sac():
    sac, lumen = _tube(10.0), _tube(12.0)
    with pytest.raises(ValueError, match="outside the sac"):
        build_ilt_mesh(lumen, sac, planes_from_report(_report(sac)))


THIN_SHIFT = 3.2  # lumen r 6 off-centre in a sac r 10: an 0.8 mm thin side


@functools.lru_cache(maxsize=None)
def _thin_case():
    """The thrombus between an eccentric lumen and its sac (gap 0.8 mm on +x,
    7.2 mm on -x), plus the arguments ``_boundary_faces`` was called with."""
    import meshforge.solve.ilt as ilt

    sac, lumen = _tube(10.0), _tube(6.0).translate((THIN_SHIFT, 0.0, 0.0), inplace=False)
    seen, real = {}, ilt._boundary_faces

    def spy(*args):
        seen["args"] = args
        return real(*args)

    ilt._boundary_faces = spy
    try:
        m = build_ilt_mesh(lumen, sac, planes_from_report(_report(sac)), size=3.0, inset=0.5)
    finally:
        ilt._boundary_faces = real
    return m, seen["args"], sac.bounds[4], sac.bounds[5]


def _face_corners(m, faces):
    from meshforge.solve.ilt import _TET_FACES

    corners = [m.points[m.tets[t, list(_TET_FACES[label])]] for t, label in faces]
    return np.array(corners, dtype=float).reshape(-1, 3, 3)


def test_thin_thrombus_faces_are_labelled_by_the_surface_they_lie_on():
    """Where the thrombus is thinner than a face's centroid-to-vertex reach
    (~1.7 mm at 3 mm), the old nearest-VERTEX test swapped lumen and sac
    labels and unloaded the wall (P07, P12). Labels here come from the
    analytic geometry: every lumen face carries the pressure, every sac face
    is tied, and both kinds of end face are free."""
    m, _, z0, z1 = _thin_case()
    groups = {"pressure": m.inner_faces, "tied": m.outer_faces, "free": m.free_faces}
    where = {}
    for name, faces in groups.items():
        c = _face_corners(m, faces)
        x, y, z = c[..., 0], c[..., 1], c[..., 2]
        r_sac, r_lumen = np.hypot(x, y), np.hypot(x - THIN_SHIFT, y)
        cap_plane = (np.abs(z - z0) < 1e-6).all(1) | (np.abs(z - z1) < 1e-6).all(1)
        recut = (np.abs(z - z0 - 0.5) < 1e-3).all(1) | (np.abs(z - z1 + 0.5) < 1e-3).all(1)
        on_sac = (np.abs(r_sac - 10.0) < 0.05).all(1) & ~cap_plane
        on_lumen = (np.abs(r_lumen - 6.0) < 0.05).all(1) & ~recut
        where[name] = {"sac": on_sac, "lumen": on_lumen, "cap_plane": cap_plane,
                       "recut": recut, "thin": (x > 8.5).all(1)}
    # every lumen face is pressure, every sac face is tied, the ends are free
    assert where["pressure"]["lumen"].all(), "a pressure face is not on the lumen"
    assert where["tied"]["sac"].all(), "a tied face is not on the sac"
    assert (where["free"]["cap_plane"] | where["free"]["recut"]).all()
    for name, w in where.items():
        if name != "pressure":
            assert not w["lumen"].any(), f"a lumen face is {name}"
        if name != "tied":
            assert not w["sac"].any(), f"a sac face is {name}"
        if name != "free":
            assert not (w["cap_plane"] | w["recut"]).any(), f"an end face is {name}"
    # the thin side is really there and really meshed (x > 8.5: gap <= ~1 mm)
    assert where["tied"]["thin"].sum() >= 20 and where["pressure"]["thin"].sum() >= 20
    assert where["free"]["cap_plane"].sum() >= 20 and where["free"]["recut"].sum() >= 20
    s = m.summary()
    assert s["n_free_faces"] == len(m.free_faces) > 0
    note = next(n for n in m.notes if n.startswith("thrombus boundary faces"))
    assert f"{len(m.inner_faces)} pressure (lumen), {len(m.outer_faces)} tied (sac), " \
           f"{len(m.free_faces)} free" in note


def test_the_face_classifier_refuses_rather_than_guesses():
    import meshforge.solve.ilt as ilt

    _, (points, tets, lumen_r, sac_r, caps, recut, tol), _, _ = _thin_case()
    # the sac moved 0.2 mm along the tube: its cap faces lie on neither surface
    moved = sac_r.translate((0.0, 0.0, 0.2), inplace=False)
    with pytest.raises(ValueError, match="neither the lumen nor the sac"):
        ilt._boundary_faces(points, tets, lumen_r, moved, caps, recut, tol)
    # a "sac" that also holds the lumen: the lumen faces lie on both
    with pytest.raises(ValueError, match="within .* of both the lumen and the sac"):
        ilt._boundary_faces(points, tets, lumen_r, sac_r.append_polydata(lumen_r), caps,
                            recut, tol)
    # cap planes the sac does not end on: its ends cannot be left free
    off = [(name, origin + 5.0 * normal, normal) for name, origin, normal in caps]
    with pytest.raises(ValueError, match="no thrombus face lies on the .* cap plane"):
        ilt._boundary_faces(points, tets, lumen_r, sac_r, off, recut, tol)


def _with_near_zero_edges(surface, n: int, gap: float = 4e-4):
    """Split ``n`` interior edges at ``gap`` mm from one end: each split
    leaves two needles carrying an edge of ``gap`` mm, like the cropped
    s1364 sac that hung gmsh (D93)."""
    import pyvista as pv

    tri = surface.triangulate().clean()
    pts, faces = tri.points.copy(), tri.faces.reshape(-1, 4)[:, 1:].copy()
    touched, new_faces, done = set(), [], 0
    for t in range(0, len(faces), max(1, len(faces) // (4 * n))):
        a, b, c = (int(x) for x in faces[t])
        (u,) = [k for k in range(len(faces)) if k != t and a in faces[k] and b in faces[k]]
        d = next(int(x) for x in faces[u] if x not in (a, b))
        if {a, b, c, d} & touched:
            continue
        m = len(pts)
        pts = np.vstack([pts, pts[a] + gap * (pts[b] - pts[a]) / np.linalg.norm(pts[b] - pts[a])])
        faces[t] = (a, m, c)  # a -> b split at m: (a, m, c) + (m, b, c)
        new_faces.append((m, b, c))
        faces[u] = (b, m, d)  # the neighbour runs b -> a: (b, m, d) + (m, a, d)
        new_faces.append((m, a, d))
        touched |= {a, b, c, d}
        done += 1
        if done == n:
            break
    faces = np.vstack([faces, new_faces])
    return pv.PolyData(pts, np.column_stack([np.full(len(faces), 3), faces]).ravel())


def test_short_edge_collapse_removes_near_zero_edges_and_keeps_the_shape():
    from meshforge.solve.ilt import collapse_short_edges

    sac = _tube(10.0)
    bad = _with_near_zero_edges(sac, 12)
    assert bad.n_open_edges == 0 and bad.n_points == sac.triangulate().clean().n_points + 12
    out, n = collapse_short_edges(bad, 0.009)
    assert n == 12
    assert out.n_points == bad.n_points - 12 and out.n_open_edges == 0 and out.is_manifold
    f = out.faces.reshape(-1, 4)[:, 1:]
    p = out.points[f]
    shortest = min(np.linalg.norm(p[:, (k + 1) % 3] - p[:, k], axis=1).min() for k in range(3))
    assert shortest >= 0.009
    assert abs(out.volume) == pytest.approx(abs(sac.volume), rel=1e-6)
    # nothing to do on a clean surface
    assert collapse_short_edges(sac, 0.009)[1] == 0


def test_the_gmsh_guard_stops_a_call_that_overruns():
    import time

    from meshforge.solve.ilt import _run_gmsh, _tri_arrays

    points, tris = _tri_arrays(_tube(6.0))
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match=r"did not finish within 0\.05 s and was stopped"):
        _run_gmsh("remesh", "remeshing the lumen", size=3.0, timeout=0.05, points=points,
                  tris=tris)
    assert time.monotonic() - t0 < 10.0


def test_the_gmsh_guard_passes_the_child_error_through():
    from meshforge.solve.ilt import _run_gmsh, _tri_arrays

    points, tris = _tri_arrays(_tube(6.0))
    with pytest.raises(RuntimeError, match="unexpected keyword argument 'stray'"):
        _run_gmsh("remesh", "x", size=3.0, points=points, tris=tris, stray=np.zeros(1))


def test_the_thrombus_ladder_survives_a_hung_attempt(monkeypatch):
    """The first rung times out (as gmsh did on s1364): the next one meshes,
    and the notes say what happened. Every rung failing is a loud error."""
    import meshforge.solve.ilt as ilt

    real = ilt._remesh

    def hangs_when_collapsing(surface, size, what="surface", tol=0.0):
        if tol > 0:
            raise RuntimeError(f"thrombus mesh: remeshing the {what}: gmsh did not finish")
        return real(surface, size, what, tol)

    sac, lumen = _tube(10.0), _tube(6.0)
    monkeypatch.setattr(ilt, "_remesh", hangs_when_collapsing)
    m = build_ilt_mesh(lumen, sac, planes_from_report(_report(sac)))
    assert m.sj_min > ilt.SJ_MIN_ACCEPT
    tried = next(n for n in m.notes if n.startswith("thrombus mesh attempts"))
    assert "did not finish" in tried and "surfaces as given: sj_min" in tried

    def always_hangs(surface, size, what="surface", tol=0.0):
        raise RuntimeError("gmsh did not finish")

    monkeypatch.setattr(ilt, "_remesh", always_hangs)
    with pytest.raises(RuntimeError, match="failed on every attempt") as err:
        build_ilt_mesh(lumen, sac, planes_from_report(_report(sac)))
    assert str(err.value).count("did not finish") == len(ilt.COLLAPSE_LADDER)


def test_sac_from_wall_closes_the_inner_surface():
    wall = _wall(14.0)
    sac = am.sac_from_wall(wall)
    assert sac.n_open_edges == 0
    assert abs(sac.volume) == pytest.approx(math.pi * 14.0 ** 2 * L, rel=0.02)


def test_thrombus_needs_the_crop_planes(tmp_path):
    wall = _wall(14.0)
    with pytest.raises(ValueError, match="cap planes"):
        am.run_aaa_stress(wall, json_summary(wall), out_dir=tmp_path, lumen=_tube(8.0),
                          sac=_tube(14.0), crop_report=None, end_zone="30")


def test_thrombus_needs_the_sac_surface(tmp_path):
    wall, sac = _wall(14.0), _tube(14.0)
    with pytest.raises(ValueError, match="sac surface"):
        am.run_aaa_stress(wall, json_summary(wall), out_dir=tmp_path, lumen=_tube(8.0),
                          crop_report=_report(sac), end_zone="30")


def json_summary(wall) -> dict:
    import json

    return json.loads(json.dumps(wall.summary()))


def _two_layer_hoop(a, b, c, E1, n1, E2, n2, p, r):
    """Plane-strain Lamé, inner layer a..b (E1, n1) bonded to b..c (E2, n2),
    pressure p at a, free at c: the hoop stress at radius r in the outer layer."""
    def u(E, n, r):  # radial displacement coefficients of (A, B)
        return (1 + n) / E * (1 - 2 * n) * r, (1 + n) / E / r
    M = np.array([
        [1, -1 / a**2, 0, 0],
        [0, 0, 1, -1 / c**2],
        [1, -1 / b**2, -1, 1 / b**2],
        [*u(E1, n1, b), *(-np.array(u(E2, n2, b)))],
    ])
    A1, B1, A2, B2 = np.linalg.solve(M, [-p, 0, 0, 0])
    return A2 + B2 / r**2


@requires_ccx
def test_two_layer_lame_through_the_tied_thrombus(tmp_path):
    a, b, t = 8.0, 14.0, 1.5
    wall, lumen, sac = _wall(b), _tube(a), _tube(b)
    res = am.run_aaa_stress(wall, json_summary(wall), out_dir=tmp_path, crop_report=_report(sac),
                            lumen=lumen, sac=sac, E_ilt=5.0, end_zone="30", systolic_mmhg=140.0,
                            whatif_mmhg=120.0)
    assert not res.refused and res.summary["thrombus"]["modelled"]
    g = res.grid
    zc = g.cell_centers().points[:, 2]
    mid = np.abs(zc - 0.5 * (zc.min() + zc.max())) < L / 6
    vol = g.compute_cell_sizes()["Volume"]
    fe = np.average(g.cell_data["max_principal"][mid], weights=vol[mid])
    exact = _two_layer_hoop(a, b, b + t, 5.0, 0.49, 100.0, 0.49, P140, b + t / 2)
    print(f"two-layer Lame: FE {fe:.5f} MPa vs exact {exact:.5f} MPa")
    # 2%: the FE lands within 0.6% here. Pressure wrongly left on the wall's
    # inner face (the interface) would read +4% (0.11325 vs 0.10885 MPa
    # analytically): a tied, nearly incompressible thrombus shields the wall
    # almost as much either way, so the tolerance must stay tight
    assert fe == pytest.approx(exact, rel=0.02)
    # and the thrombus really shields the wall (here by ~38%): Lamé for the
    # wall alone, loaded at its own inner radius, is far higher
    wall_alone = _two_layer_hoop(b - 1e-9, b, b + t, 1.0, 0.49, 100.0, 0.49, P140, b + t / 2)
    assert fe < 0.75 * wall_alone
    assert "intraluminal thrombus modelled" in res.to_text()


@requires_ccx
def test_a_soft_thrombus_passes_the_pressure_through(tmp_path):
    wall, lumen, sac = _wall(14.0), _tube(8.0), _tube(14.0)
    rep = _report(sac)
    soft = am.run_aaa_stress(wall, json_summary(wall), out_dir=tmp_path / "soft", crop_report=rep,
                             lumen=lumen, sac=sac, E_ilt=0.01, end_zone="30")
    bare = am.run_aaa_stress(wall, json_summary(wall), out_dir=tmp_path / "bare", crop_report=rep,
                             end_zone="30")
    assert soft.pws_mpa == pytest.approx(bare.pws_mpa, rel=0.05)
    assert bare.summary["thrombus"] == {"modelled": False}
