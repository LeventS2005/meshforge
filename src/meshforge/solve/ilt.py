"""Intraluminal thrombus (ILT) as a solid in the AAA wall-stress model (D91).

The wall-only model loads the wall's inner surface with the full blood
pressure. In an aneurysm the pressure acts on the LUMEN, and the thrombus
between lumen and wall carries part of it. Leaving it out overstates wall
stress: on a native AAA, 674 kPa without it vs 287 kPa with it. Against
the Zenodo AAA dataset's published FE result, which includes thrombus, the
wall-only tension read 1.42x high, and the gap tracked thrombus thickness;
with it the median p99 ratio is 1.14 (12 cases).

The thrombus is the region between the blood-lumen surface and the sac
surface, the wall's inner surface. Both are cropped with the same two cap
planes (D85).
1. The lumen is re-cut ``inset`` mm inside each cap plane, so the hole
   never touches the outer boundary: no coplanar facets for gmsh.
2. Both surfaces are remeshed by gmsh at ``size``. Marching-cubes needles
   would otherwise become sliver tets (D90).
3. The region is filled with linear tets, oriented positive, then made
   C3D10 with STRAIGHT mid-edge nodes. That matches the wall's
   straight-edged C3D20R; node order is 1-2, 2-3, 3-1, 1-4, 2-4, 3-4.

Each boundary face of the thrombus is labelled by the surface it lies ON
(the point-to-surface distance of its corners and centroid; the fill keeps
the remeshed triangles as its boundary). Lumen faces are the pressure
surface; sac faces are tied to the wall (``*TIE ... ADJUST=NO``: moving
slave nodes onto the wall inverted thin tets). The thrombus ends on the cap
planes stay free; the wall carries the clamp. The lumen's re-cut caps are
free too: they sit ``inset`` mm inside the planes only for gmsh, and
pressure there would push that thin lid outward with an axial piston load
of P x lumen area. Neither the wall-only model nor the blood, which
continues through the cut, has that load. A face on neither surface, or on
both, is refused. The first version measured the face centroid to the
nearest surface VERTEX instead. Where the thrombus was thinner than about
1.7 mm (at a 3 mm mesh), that swapped the lumen and sac labels and left the
wall there unloaded (D91 addendum, 19 Oct).

gmsh can loop forever, with no error, on a surface carrying near-zero
edges: two cropped s1364 sacs (edges of 0.0004 mm, one of exactly zero)
never returned from the remesh (D93). So every gmsh call runs in a child
process killed after ``GMSH_TIMEOUT_S``, and the mesh is tried down
``COLLAPSE_LADDER``: short edges collapsed first at 0.009 mm, then the
surfaces as given, then two more tolerances. gmsh's response to these tiny
changes is erratic (one tolerance hangs, fails or leaves slivers on one
sac and is clean on the next), so each attempt is checked, not trusted:
it must mesh, match the volume and reach ``SJ_MIN_ACCEPT``.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

DEFAULT_E_ILT_MPA = 5.0  # the Zenodo AAA dataset's FE deck (wall 100 MPa)
DEFAULT_NU_ILT = 0.49
DEFAULT_ILT_SIZE_MM = 3.0
DEFAULT_INSET_MM = 0.5
GMSH_TIMEOUT_S = 60.0  # one gmsh call; the s1364 remesh/fill each take ~1 s
#: short-edge collapse tolerances tried in order, as fractions of the mesh size
COLLAPSE_LADDER = (0.003, 0.0, 0.001, 0.01)
SJ_MIN_ACCEPT = 0.02  # an attempt below this is kept only if none does better
#: a boundary face lies ON a surface or cap plane when its corners and centroid
#: are all within this fraction of the mesh size of it. The fill keeps the
#: remeshed triangles exactly (measured <= 1.3e-13 mm on s1364 and three Zenodo
#: cases) and a short-edge collapse moves a node <= 0.005 x size, so this only
#: clears those; it stays far below the inset and any gap a tet spans
ON_SURFACE_FRACTION = 0.01
_TET10_EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
_TET_FACES = {"S1": (0, 1, 2), "S2": (0, 3, 1), "S3": (1, 3, 2), "S4": (2, 3, 0)}


@dataclass
class IltMesh:
    points: np.ndarray  # (n, 3): corner nodes, then straight mid-edge nodes
    tets: np.ndarray  # (m, 10) C3D10 connectivity, 0-based into ``points``
    inner_faces: list[tuple[int, str]]  # (tet index, S1..S4) on the lumen: pressure
    outer_faces: list[tuple[int, str]]  # on the sac: tied to the wall's inner faces
    volume_ml: float
    expected_volume_ml: float
    sj_min: float
    notes: list[str] = field(default_factory=list)
    # the ends on the cap planes and the lumen's re-cut caps: no load, no tie
    free_faces: list[tuple[int, str]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "n_tets": int(len(self.tets)),
            "n_nodes": int(len(self.points)),
            "volume_ml": round(self.volume_ml, 3),
            "expected_volume_ml": round(self.expected_volume_ml, 3),
            "sj_min": round(self.sj_min, 4),
            "n_pressure_faces": len(self.inner_faces),
            "n_tied_faces": len(self.outer_faces),
            "n_free_faces": len(self.free_faces),
            "notes": list(self.notes),
        }


def _gmsh():
    try:
        import gmsh
    except ImportError as exc:  # pragma: no cover - gmsh ships with the env
        raise RuntimeError(
            "the thrombus mesh needs gmsh — install with: "
            "pip install -e '.[meshing]' in the meshforge clone"
        ) from exc
    return gmsh


def _tri_arrays(surface: pv.PolyData) -> tuple[np.ndarray, np.ndarray]:
    tri = surface.triangulate().clean()
    return np.asarray(tri.points, dtype=float), tri.faces.reshape(-1, 4)[:, 1:].copy()


def _polydata(points: np.ndarray, tris: np.ndarray) -> pv.PolyData:
    return pv.PolyData(points, np.column_stack([np.full(len(tris), 3), tris]).ravel())


def collapse_short_edges(surface: pv.PolyData, tol: float) -> tuple[pv.PolyData, int]:
    """Collapse every edge shorter than ``tol`` to its midpoint (both
    triangles on it vanish; each node moves at most ``tol / 2``). Returns the
    surface and the number of edges collapsed. Refuses, loudly, if the
    result is not a closed manifold."""
    points, tris = _tri_arrays(surface)
    n_collapsed = 0
    for _ in range(50):
        p = points[tris]
        lengths = np.stack([np.linalg.norm(p[:, (k + 1) % 3] - p[:, k], axis=1) for k in range(3)], 1)
        short = np.argwhere(lengths < tol)
        if len(short) == 0:
            break
        target = np.arange(len(points))
        moved = np.zeros(len(points), dtype=bool)
        points = points.copy()
        for t, k in short[np.argsort(lengths[short[:, 0], short[:, 1]])]:
            a, b = int(tris[t, k]), int(tris[t, (k + 1) % 3])
            if moved[a] or moved[b]:
                continue  # one collapse per node per round
            points[a] = 0.5 * (points[a] + points[b])
            target[b] = a
            moved[a] = moved[b] = True
            n_collapsed += 1
        tris = target[tris]
        keep = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
        tris = tris[keep]
        used = np.unique(tris)
        renum = np.full(len(points), -1, dtype=np.int64)
        renum[used] = np.arange(len(used))
        points, tris = points[used], renum[tris]
    else:  # pragma: no cover - 50 rounds of halving never leave a short edge
        raise ValueError(f"short edges (< {tol:g} mm) remain after 50 collapse rounds")
    out = _polydata(points, tris)
    if n_collapsed and (out.n_open_edges or not out.is_manifold):
        raise ValueError(
            f"collapsing {n_collapsed} edges shorter than {tol:g} mm left the surface "
            f"{'open' if out.n_open_edges else 'non-manifold'}"
        )
    return out, n_collapsed


def _gmsh_remesh_job(points: np.ndarray, tris: np.ndarray, size: float):
    """gmsh reparametrize (40 deg creases) + remesh at ``size``. Runs in the
    guarded child process (``_run_gmsh``)."""
    gmsh = _gmsh()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("meshforge_ilt_remesh")
        gmsh.model.addDiscreteEntity(2, 1)
        gmsh.model.mesh.addNodes(2, 1, np.arange(1, len(points) + 1), points.ravel())
        gmsh.model.mesh.addElementsByType(1, 2, [], (tris + 1).ravel())
        gmsh.model.mesh.classifySurfaces(math.radians(40.0), True, True, math.pi)
        gmsh.model.mesh.createGeometry()
        for name, value in (("Mesh.MeshSizeMax", size), ("Mesh.MeshSizeMin", size),
                            ("Mesh.MeshSizeFromPoints", 0), ("Mesh.MeshSizeExtendFromBoundary", 0)):
            gmsh.option.setNumber(name, value)
        gmsh.model.mesh.generate(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        index = np.zeros(int(tags.max()) + 1, dtype=np.int64)
        index[np.asarray(tags, dtype=np.int64)] = np.arange(len(tags))
        out = []
        for dim, tag in gmsh.model.getEntities(2):
            types, _, nodes = gmsh.model.mesh.getElements(dim, tag)
            for etype, enodes in zip(types, nodes):
                if etype == 2:
                    out.append(index[np.asarray(enodes, dtype=np.int64)].reshape(-1, 3))
        return {"points": np.asarray(xyz, dtype=float).reshape(-1, 3), "tris": np.vstack(out)}
    finally:
        gmsh.finalize()


def _gmsh_fill_job(outer_points, outer_tris, inner_points, inner_tris, size: float):
    """Linear tets of inside(outer) minus inside(inner). Runs in the guarded
    child process (``_run_gmsh``)."""
    gmsh = _gmsh()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("meshforge_ilt")
        offset = 0
        for tag, pts, tris in ((1, outer_points, outer_tris), (2, inner_points, inner_tris)):
            gmsh.model.addDiscreteEntity(2, tag)
            gmsh.model.mesh.addNodes(2, tag, np.arange(offset + 1, offset + len(pts) + 1), pts.ravel())
            gmsh.model.mesh.addElementsByType(tag, 2, [], (tris + offset + 1).ravel())
            offset += len(pts)
        loops = [gmsh.model.geo.addSurfaceLoop([1]), gmsh.model.geo.addSurfaceLoop([2])]
        gmsh.model.geo.addVolume(loops)
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5 * size)
        try:
            gmsh.model.mesh.generate(3)
        except Exception as exc:  # gmsh raises a bare Exception
            raise RuntimeError(
                f"thrombus tet fill failed — gmsh: {exc}. The lumen probably touches or "
                f"crosses the sac surface"
            ) from None
        try:
            gmsh.model.mesh.optimize("Netgen")
        except Exception:  # pragma: no cover - optimization is best effort
            pass
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        index = np.zeros(int(tags.max()) + 1, dtype=np.int64)
        index[np.asarray(tags, dtype=np.int64)] = np.arange(len(tags))
        _, tet_nodes = gmsh.model.mesh.getElementsByType(4)
        return {"points": np.asarray(xyz, dtype=float).reshape(-1, 3),
                "tets": index[np.asarray(tet_nodes, dtype=np.int64)].reshape(-1, 4)}
    finally:
        gmsh.finalize()


_JOBS = {"remesh": _gmsh_remesh_job, "fill": _gmsh_fill_job}


def _child_main(job: str, in_path: str, out_path: str) -> None:
    """Entry point of the guarded gmsh child: arrays in, arrays (or the
    error) out, both as .npz."""
    with np.load(in_path) as data:
        args = {k: data[k] for k in data.files}
    size = float(args.pop("size"))
    try:
        result = _JOBS[job](**args, size=size)
    except Exception as exc:
        kind = "ValueError" if isinstance(exc, ValueError) else "RuntimeError"
        np.savez(out_path, error=np.array(str(exc)), kind=np.array(kind))
        return
    np.savez(out_path, **result)


def _run_gmsh(job: str, what: str, *, size: float, timeout: float = GMSH_TIMEOUT_S,
              **arrays: np.ndarray) -> dict[str, np.ndarray]:
    """Run one gmsh job in a child process killed after ``timeout`` s. gmsh
    cannot be interrupted in-process, and it can spin forever on a bad
    surface without raising (D93)."""
    from meshforge.io.segment import _kill

    src = str(Path(__file__).resolve().parents[2])  # the dir holding meshforge/
    code = ("import sys; sys.path.insert(0, sys.argv[4]); "
            "from meshforge.solve.ilt import _child_main; _child_main(*sys.argv[1:4])")
    with tempfile.TemporaryDirectory(prefix="meshforge_gmsh_") as tmp:
        in_path, out_path = os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npz")
        log_path = os.path.join(tmp, "child.log")
        np.savez(in_path, size=np.array(float(size)), **arrays)
        with open(log_path, "wb") as log:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-c", code, job, in_path, out_path, src],
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=tmp,
                start_new_session=(os.name == "posix"),
            )
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"{what}: gmsh did not finish within {timeout:g} s and was stopped. gmsh "
                    f"can spin forever on a degenerate surface; nothing was reported"
                ) from None
            finally:
                _kill(proc)  # no-op once it exited
        if not os.path.exists(out_path):
            tail = Path(log_path).read_text(errors="replace").strip().splitlines()[-5:]
            raise RuntimeError(f"{what}: the gmsh process exited with code {rc} and no "
                               f"result" + (": " + " | ".join(tail) if tail else ""))
        with np.load(out_path) as data:
            out = {k: data[k] for k in data.files}
    if "error" in out:
        exc = ValueError if str(out["kind"]) == "ValueError" else RuntimeError
        raise exc(str(out["error"]))
    return out


def _remesh(surface: pv.PolyData, size: float, what: str = "surface",
            tol: float = 0.0) -> tuple[pv.PolyData, int]:
    """Collapse edges shorter than ``tol`` (0 = none), then gmsh-remesh the
    closed surface at ``size``. Returns it and the number of collapsed
    edges."""
    n_collapsed = 0
    if tol > 0:
        surface, n_collapsed = collapse_short_edges(surface, tol)
    points, tris = _tri_arrays(surface)
    out = _run_gmsh("remesh", f"thrombus mesh: remeshing the {what}", size=size,
                    points=points, tris=tris)
    out = _polydata(out["points"], out["tris"]).clean()
    if out.n_open_edges:
        raise ValueError(f"the remeshed {what} is not closed ({out.n_open_edges} open edges)")
    return out, n_collapsed


def _fill_between(outer: pv.PolyData, inner: pv.PolyData, size: float):
    """Linear tets of inside(outer) minus inside(inner)."""
    (op, ot), (ip, it) = _tri_arrays(outer), _tri_arrays(inner)
    out = _run_gmsh("fill", "thrombus mesh: the tet fill", size=size, outer_points=op,
                    outer_tris=ot, inner_points=ip, inner_tris=it)
    points, tets = out["points"], np.asarray(out["tets"], dtype=np.int64)
    if len(tets) == 0:
        raise RuntimeError("gmsh produced no thrombus tetrahedra")
    return points, tets


def _orient(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    a, b, c, d = (points[tets[:, i]] for i in range(4))
    neg = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) < 0
    tets = tets.copy()
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    return tets


def _tet10(points: np.ndarray, tets: np.ndarray):
    ends = np.sort(tets[:, _TET10_EDGES], axis=2).reshape(-1, 2)
    uniq, inv = np.unique(ends, axis=0, return_inverse=True)
    mids = (len(points) + inv.ravel()).reshape(len(tets), 6)
    allpts = np.vstack([points, 0.5 * (points[uniq[:, 0]] + points[uniq[:, 1]])])
    return allpts, np.hstack([tets, mids])


def _off_surface(probes: np.ndarray, surface: pv.PolyData) -> np.ndarray:
    """Per face, the largest point-to-SURFACE distance of its probe points
    (``probes`` is (faces, k, 3)): 0 when the whole face lies on ``surface``."""
    q = pv.PolyData(probes.reshape(-1, 3)).compute_implicit_distance(surface)
    d = np.abs(np.asarray(q.point_data["implicit_distance"], dtype=float))
    return d.reshape(probes.shape[:2]).max(axis=1)


def _boundary_faces(points, tets, lumen: pv.PolyData, sac: pv.PolyData, cap_planes,
                    lumen_caps, tol: float):
    """Every boundary face of the thrombus, labelled by the surface it lies ON
    (its corners and centroid within ``tol`` of it):
    - lumen -> pressure;
    - sac -> tied to the wall;
    - sac faces on ``cap_planes`` (the thrombus ends) and lumen faces on
      ``lumen_caps`` (its re-cut caps) -> free.
    Planes are (name, origin, unit normal). Returns the (tet, S1..S4) lists of
    pressure, tied and free faces, and the free counts. Refuses a face on
    neither surface or on both, and a cap plane with no thrombus end on it."""
    grid = pv.UnstructuredGrid(
        np.column_stack([np.full(len(tets), 4), tets]).ravel(),
        np.full(len(tets), pv.CellType.TETRA, dtype=np.uint8), points,
    )
    surf = grid.extract_surface(algorithm="dataset_surface")
    faces = surf.faces.reshape(-1, 4)[:, 1:]
    cell = np.asarray(surf.cell_data["vtkOriginalCellIds"], dtype=np.int64)
    pid = np.asarray(surf.point_data["vtkOriginalPointIds"], dtype=np.int64)
    corners = np.asarray(surf.points, dtype=float)[faces]  # (faces, 3, 3)
    probes = np.concatenate([corners, corners.mean(axis=1, keepdims=True)], axis=1)
    d_lumen, d_sac = _off_surface(probes, lumen), _off_surface(probes, sac)
    on_lumen, on_sac = d_lumen <= tol, d_sac <= tol
    neither, both = ~(on_lumen | on_sac), on_lumen & on_sac
    if neither.any():
        raise ValueError(
            f"{int(neither.sum())} of {len(faces)} thrombus boundary faces lie on neither the "
            f"lumen nor the sac surface (up to {np.minimum(d_lumen, d_sac)[neither].max():.3g} mm "
            f"off, tolerance {tol:.2g} mm): the fill's boundary is not the two surfaces, so "
            f"which side carries the pressure is unknown — refusing to guess"
        )
    if both.any():
        raise ValueError(
            f"{int(both.sum())} thrombus boundary faces lie within {tol:.2g} mm of both the lumen "
            f"and the sac: the lumen touches the wall there, so the blood side cannot be told "
            f"from the wall side — refusing to guess"
        )

    def on_planes(planes, mask):
        hit, count = np.zeros(len(faces), dtype=bool), {}
        for name, origin, normal in planes:
            here = mask & (np.abs((corners - origin) @ normal).max(axis=1) <= tol)
            count[name] = int(here.sum())
            hit |= here
        return hit, count

    ends, per_plane = on_planes(cap_planes, on_sac)
    recut, _ = on_planes(lumen_caps, on_lumen)
    bare = [name for name, n in per_plane.items() if n == 0]
    if bare:
        raise ValueError(
            f"no thrombus face lies on the {' or '.join(bare)} cap plane: the sac does not end "
            f"on the crop's planes, so the thrombus ends cannot be left free as specified"
        )
    kind = np.where(ends | recut, 2, np.where(on_lumen, 0, 1))  # pressure, tied, free
    out: tuple[list, list, list] = ([], [], [])
    for k, f in enumerate(faces):
        nodes = set(pid[f].tolist())
        row = tets[cell[k]]
        label = next(lab for lab, loc in _TET_FACES.items() if {int(row[i]) for i in loc} == nodes)
        out[kind[k]].append((int(cell[k]), label))
    if not out[0] or not out[1]:
        raise ValueError(
            f"the thrombus has {len(out[0])} pressure and {len(out[1])} tied faces: "
            f"it needs both a lumen side and a sac side"
        )
    return (*out, {"ends": int(ends.sum()), "recut": int(recut.sum())})


def build_ilt_mesh(lumen: pv.PolyData, sac: pv.PolyData, planes, *,
                   size: float = DEFAULT_ILT_SIZE_MM,
                   inset: float = DEFAULT_INSET_MM,
                   remesh_sac: bool = True) -> IltMesh:
    """The thrombus solid between ``lumen`` and ``sac``, which share the two cut
    ``planes`` (``geometry.crop.planes_from_report``). Refuses a lumen that
    pokes through the sac, and a fill whose volume misses sac - lumen by
    more than 2%. ``remesh_sac=False`` keeps the sac triangulation as given:
    right for a sac rebuilt from a structured wall's inner faces (no needles,
    and the thrombus's outer nodes then sit exactly on the wall's)."""
    size, inset = float(size), float(inset)
    if not (np.isfinite(size) and size > 0 and np.isfinite(inset) and inset > 0):
        raise ValueError(f"thrombus mesh size and inset must be positive, got {size}, {inset}")
    # re-cut the lumen inset mm inside each cap plane with the crop's own
    # planar cut + snapped flat caps (no centreline needed). Not
    # clip_closed_surface: its caps carried needles of aspect ~4000 on a real
    # lumen, and gmsh's remesh then hung.
    from meshforge.geometry.crop import CutPlane, cut_and_cap

    inside = [
        CutPlane(name=pl.name, origin=np.asarray(pl.origin, dtype=float)
                 - inset * np.asarray(pl.normal, dtype=float),
                 normal=np.asarray(pl.normal, dtype=float), height=pl.height)
        for pl in planes
    ]
    keep = 0.5 * (inside[0].origin + inside[1].origin)
    lumen_cut, _ = cut_and_cap(lumen.triangulate(), inside, keep, what="lumen",
                               require_enclosure=False)
    probe = lumen_cut.points[:: max(1, lumen_cut.n_points // 2000)]
    enclosed = pv.PolyData(probe).select_enclosed_points(sac, check_surface=False)
    outside = 1.0 - float(np.mean(enclosed.point_data["SelectedPoints"]))
    if outside > 0.01:
        raise ValueError(
            f"{100 * outside:.1f}% of the lumen surface lies outside the sac surface — "
            f"the thrombus region is not a shell between them"
        )
    expected = (abs(float(sac.volume)) - abs(float(lumen_cut.volume))) / 1000.0
    if expected <= 0:
        raise ValueError(f"the sac encloses no more than the lumen (sac - lumen {expected:.2f} mL)")
    tried: list[str] = []
    best = None
    for fraction in COLLAPSE_LADDER:
        tol = fraction * size
        label = f"edges < {tol:.2g} mm collapsed" if tol > 0 else "surfaces as given"
        try:
            attempt = _attempt(lumen_cut, sac, size, tol, remesh_sac, expected)
        except (RuntimeError, ValueError) as exc:
            tried.append(f"{label}: {exc}")
            continue
        tried.append(f"{label}: sj_min {attempt['sj_min']:.3g}")
        if best is None or attempt["sj_min"] > best["sj_min"]:
            best = attempt
        if attempt["sj_min"] >= SJ_MIN_ACCEPT:
            break
    if best is None:
        raise RuntimeError("the thrombus mesh failed on every attempt — " + "; ".join(tried))
    points, tets, volume = best["points"], best["tets"], best["volume_ml"]

    def unit(pls):
        return [(pl.name, np.asarray(pl.origin, dtype=float),
                 np.asarray(pl.normal, dtype=float) / np.linalg.norm(pl.normal)) for pl in pls]

    tol = ON_SURFACE_FRACTION * size
    inner_faces, outer_faces, free_faces, n_free = _boundary_faces(
        points, tets, best["lumen_r"], best["sac_r"], unit(planes), unit(inside), tol)
    pts10, tets10 = _tet10(points, tets)
    notes = [
        f"thrombus = sac - lumen, the lumen re-cut {inset:g} mm inside both cap planes; "
        f"surfaces remeshed by gmsh at {size:g} mm; {len(tets)} C3D10 (straight mid-edge nodes)",
        f"volume {volume:.2f} mL (sac - lumen {expected:.2f} mL)",
        f"thrombus boundary faces: {len(inner_faces)} pressure (lumen), {len(outer_faces)} tied "
        f"(sac), {len(free_faces)} free ({n_free['ends']} ends on the cap planes + "
        f"{n_free['recut']} on the lumen's re-cut caps); each labelled by the surface it lies "
        f"on (within {tol:.2g} mm)",
    ]
    if best["n_collapsed"]:
        notes.append(
            f"{best['n_collapsed']} edges shorter than {best['tol']:.2g} mm collapsed before the "
            f"remesh (each node moved <= {best['tol'] / 2:.2g} mm): gmsh can hang on near-zero edges"
        )
    if len(tried) > 1:
        notes.append("thrombus mesh attempts (D93): " + "; ".join(tried))
    if best["sj_min"] < SJ_MIN_ACCEPT:
        notes.append(
            f"WARNING: the thrombus mesh's scaled-Jacobian minimum {best['sj_min']:.3g} is below "
            f"{SJ_MIN_ACCEPT:g} on every attempt; the best attempt is used"
        )
    return IltMesh(
        points=pts10, tets=tets10, inner_faces=inner_faces, outer_faces=outer_faces,
        volume_ml=volume, expected_volume_ml=expected, sj_min=best["sj_min"], notes=notes,
        free_faces=free_faces,
    )


def _attempt(lumen_cut, sac, size: float, tol: float, remesh_sac: bool, expected: float):
    """One remesh + fill with short edges below ``tol`` (0 = none) collapsed;
    raises if gmsh fails or hangs, or the volume misses sac - lumen by > 2%."""
    lumen_r, n_collapsed = _remesh(lumen_cut, size, "lumen", tol)
    if remesh_sac:
        sac_r, n_sac = _remesh(sac, size, "sac", tol)
        n_collapsed += n_sac
    else:
        sac_r = sac.triangulate().clean()
    points, tets = _fill_between(sac_r, lumen_r, size)
    tets = _orient(points, tets)
    a, b, c, d = (points[tets[:, i]] for i in range(4))
    volume = float((np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0).sum()) / 1000.0
    if abs(volume - expected) > 0.02 * expected:
        raise ValueError(
            f"thrombus fill volume {volume:.2f} mL vs sac - lumen {expected:.2f} mL (> 2% off)"
        )
    grid = pv.UnstructuredGrid(
        np.column_stack([np.full(len(tets), 4), tets]).ravel(),
        np.full(len(tets), pv.CellType.TETRA, dtype=np.uint8), points,
    )
    sj = grid.cell_quality("scaled_jacobian")["scaled_jacobian"]
    return {"points": points, "tets": tets, "volume_ml": volume, "sj_min": float(np.min(sj)),
            "lumen_r": lumen_r, "sac_r": sac_r, "n_collapsed": n_collapsed, "tol": tol}
