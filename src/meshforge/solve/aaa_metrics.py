"""AAA wall stress under blood pressure (M3, D88) — an INVESTIGATIONAL metric.

The method is the stiff-linear solve on the LOADED geometry (BioPARR,
Joldes & Miller): the CT shows the wall already pressurised, so a linear
elastic, small-strain (nlgeom off) solve on that geometry gives a stress that
is set by geometry and pressure, not by the modulus. Units are N, mm, MPa
(1 mmHg = 133.322 Pa = 1.33322e-4 MPa).

* Solve (CalculiX): uniform pressure on the wall's ``inner`` (lumen-side)
  group, both end caps (``inlet``/``outlet``) encastre, as in the Zenodo AAA
  deck. Wall E 100 MPa (their 1e5 kPa), nu 0.49. Elements: the swept hex
  wall solved as C3D20R (straight mid-edge nodes added, same geometry). ccx
  has no hybrid C3D8H; C3D8 locks at nu >= 0.45 and C3D8I, clean on regular
  meshes, reads 10-21 % low on the distorted cells of a real sac wall (D88).
  Stress is read per INTEGRATION POINT (2x2x2) from the .dat.
* PWS: the volume-weighted 99th percentile of the integration-point von Mises
  stress over the wall OUTSIDE the end zones (default: one neck diameter from
  each cap plane — the crop report's neck, else each cap's own equivalent
  diameter). Also the 99th percentile of the max principal stress, the max,
  and the peak location (height above the distal cap).
* Wall tension T (N/mm): the through-thickness resultant of the in-plane
  stress over each radial column of hexes (projected on the column's tangent
  plane) — the wall's uniform nominal thickness t x the layer-mean in-plane
  stress, or, when any column is not t thick along the surface normal (rays
  crossing a sac's shoulder at an angle), the stress integrated along each
  column's measured normal thickness — its larger principal value; mapped to
  the outer-surface nodes. On loaded geometry it is ~independent of the
  assumed thickness.

Refusals (fail loudly, rule 6): no metric is reported — the sentinel ``None``
plus a human-readable reason — when the mesh summary is missing, the wall has
any inverted cell, any capped or floored wall ray (both sweep kinds), or when
the end zones leave too little of the wall to evaluate.

Never a rupture probability or a time to rupture: the output says
"investigational" and states its assumptions.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: 1 mmHg = 133.322 Pa = 1.33322e-4 MPa (N/mm²).
MMHG_TO_MPA = 1.33322e-4
DEFAULT_E_MPA = 100.0  # the Zenodo AAA deck's wall: *Elastic 100000 (kPa)
DEFAULT_NU = 0.49  # near-incompressible tissue, as in the Zenodo deck
DEFAULT_SYSTOLIC_MMHG = 140.0
DEFAULT_WHATIF_MMHG = 120.0
PWS_PERCENTILE = 99.0
#: below this fraction of the wall volume left outside the end zones, the
#: "99th percentile" would be a near-maximum of a sliver — refused.
MIN_EVALUATED_FRACTION = 0.10
#: end zone "bending": this many shell bending lengths per cap.
BENDING_LENGTHS = 3.0
#: the tension is t_nominal x the layer-mean stress only while every radial
#: column is t_nominal thick along the inner surface's normal, within this
#: fraction. A swept wall's rays can cross a sac's shoulder at an angle: the
#: column is then thinner than its ray (Zenodo P17: 0.57 mm under 1.5 mm rays),
#: its stress ~t/h higher, and t x that stress over-reads the resultant by t/h
#: (P17's p99 read 1.91 x the reference, 0.95 integrated along the normal).
UNIFORM_NORMAL_TOL = 0.02
ELEMENT = "C3D20R"
METHOD = (
    "stiff-linear on the loaded geometry: linear elastic, small strain (nlgeom off), "
    "C3D20R hexahedra, CalculiX"
)

# Swept-wall hex convention (meshing/swept.py _hex_mesh): local nodes 0-3 on
# layer l, 4-7 on layer l+1; 0-1 / 3-2 run around the section, 0-3 along the
# stations. Boundary faces by local face:
_LOCAL_FACES: tuple[tuple[str, tuple[int, int, int, int]], ...] = (
    ("inner", (0, 1, 2, 3)),
    ("outer", (4, 5, 6, 7)),
    ("inlet", (0, 1, 5, 4)),
    ("outlet", (3, 2, 6, 7)),
    ("side", (1, 2, 6, 5)),
    ("side", (0, 3, 7, 4)),
)


def mmhg_to_mpa(mmhg: float) -> float:
    """Blood pressure in mmHg -> MPa (N/mm²), the solve's unit."""
    value = float(mmhg)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"pressure must be a positive number of mmHg, got {mmhg!r}")
    return value * MMHG_TO_MPA


# ---------------------------------------------------------------------------
# wall input


def wall_from_grid(grid, *, backend: str = "swept (from file)"):
    """A swept hex wall saved as a bare .vtu -> WallMesh with its boundary groups.

    The groups are recovered from the swept lattice's fixed hex convention
    (each boundary face is a known local face of its parent hex); a boundary
    face on a side face, or a non-hex cell, means the file is not a swept
    wall, and is refused.
    """
    import pyvista as pv

    from meshforge.meshing.base import BoundaryGroup, WallMesh
    from meshforge.quality.volume import volume_quality

    if not isinstance(grid, pv.UnstructuredGrid) or grid.n_cells == 0:
        raise ValueError("the wall file holds no volume cells")
    if set(np.unique(grid.celltypes).tolist()) != {int(pv.CellType.HEXAHEDRON)}:
        raise ValueError(
            "the wall must be all hexahedra (a swept wall from 'volmesh'); a tet or "
            "mixed mesh has no wall-ray counts to check"
        )
    conn = np.asarray(grid.cells).reshape(-1, 9)[:, 1:]
    n = len(conn)
    local = np.array([f for _, f in _LOCAL_FACES])  # (6, 4)
    faces = conn[:, local].reshape(-1, 4)  # cell-major: row = 6*cell + k
    keys = np.sort(faces, axis=1)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    boundary_rows = np.nonzero(counts[inv.ravel()] == 1)[0]
    k_local = boundary_rows % 6
    if np.any(k_local >= 4):
        raise ValueError(
            f"{int((k_local >= 4).sum())} boundary face(s) lie on the hexes' "
            f"circumferential sides — not a closed swept wall"
        )
    groups = {}
    for k, (name, _) in enumerate(_LOCAL_FACES[:4]):
        rows = boundary_rows[k_local == k]
        if rows.size == 0:
            raise ValueError(f"the wall has no {name!r} faces — not a swept tube wall")
        groups[name] = BoundaryGroup(
            name=name, cell_ids=(rows // 6).astype(np.int64), face_nodes=faces[rows]
        )
    if "region" not in grid.cell_data:
        grid.cell_data["region"] = np.ones(n, dtype=np.int32)
    return WallMesh(grid=grid, boundary=groups, backend=backend, quality=volume_quality(grid))


def read_wall(path: str | Path, summary_path: str | Path | None = None):
    """Read a swept wall .vtu and its volmesh summary JSON (default: the
    sibling ``<stem>.summary.json``). Returns (WallMesh, summary or None) —
    a missing summary is NOT an error here: :func:`wall_refusals` refuses it."""
    import pyvista as pv

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"wall mesh not found: {path}")
    wall = wall_from_grid(pv.read(str(path)))
    spath = Path(summary_path) if summary_path else path.with_suffix(".summary.json")
    summary = json.loads(spath.read_text()) if spath.is_file() else None
    return wall, summary


def wall_refusals(wall, summary: dict | None) -> list[str]:
    """Why the wall must not carry a stress result (empty = cleared).

    Checks the mesh summary (present, matches this mesh), the inverted count
    (summary AND recomputed), and the swept lattice's capped/floored ray
    counts — for both sweep kinds (planes: capped = tilt + concavity)."""
    reasons: list[str] = []
    n_inv = int(wall.quality.n_inverted)
    if n_inv:
        reasons.append(f"the wall has {n_inv} inverted cell(s)")
    if summary is None:
        reasons.append(
            "no mesh summary (the volmesh <stem>.summary.json): the wall's inverted / "
            "capped / floored counts cannot be checked"
        )
        return reasons
    if int(summary.get("n_cells", -1)) != int(wall.grid.n_cells) or int(
        summary.get("n_points", -1)
    ) != int(wall.grid.n_points):
        reasons.append(
            f"the summary describes {summary.get('n_cells')} cells / "
            f"{summary.get('n_points')} points, the mesh has {wall.grid.n_cells} / "
            f"{wall.grid.n_points} — it belongs to a different mesh"
        )
    faces = summary.get("boundary_faces") or {}
    for name in ("inner", "outer", "inlet", "outlet"):
        if name in wall.boundary and int(faces.get(name, -1)) != wall.boundary[name].n_faces:
            reasons.append(
                f"boundary group {name!r}: {wall.boundary[name].n_faces} faces in the mesh, "
                f"{faces.get(name)} in the summary"
            )
    q_inv = (summary.get("quality") or {}).get("n_inverted")
    if q_inv is None:
        reasons.append("the summary carries no inverted-cell count")
    elif int(q_inv) and not n_inv:
        reasons.append(f"the summary records {int(q_inv)} inverted cell(s)")
    lat = summary.get("lattice")
    if not lat or "n_capped" not in lat or "n_floored" not in lat:
        reasons.append(
            "the summary has no wall-ray counts (lattice n_capped / n_floored): only a "
            "swept hex wall's capped/floored check can clear a stress result"
        )
        return reasons
    if int(lat["n_capped"]):
        detail = ""
        if "n_capped_tilt" in lat:
            detail = (
                f" (tilt {lat['n_capped_tilt']}, concavity {lat['n_capped_concave']})"
            )
        reasons.append(
            f"{int(lat['n_capped'])} of {lat.get('n_rays')} wall rays capped{detail}: the "
            f"wall is thinned there and its stress would be a mesh artefact"
        )
    if int(lat["n_floored"]):
        reasons.append(
            f"{int(lat['n_floored'])} of {lat.get('n_rays')} wall rays floored: the wall "
            f"is thinner than its floor there"
        )
    return reasons


def crop_report_from_journal(journal: list[dict]) -> dict | None:
    """The newest crop report recorded AFTER the newest source step (load,
    synth, segment, sac) — i.e. the crop of the current surface."""
    sources = {"load", "generate_synth", "segment", "sac"}
    for rec in reversed(journal):
        if rec.get("step") in sources:
            return None
        if rec.get("step") == "crop":
            return (rec.get("summary") or {}).get("crop")
    return None


# ---------------------------------------------------------------------------
# geometry of the swept wall: radial columns, caps, end zones


@dataclass
class _Columns:
    cell_column: np.ndarray  # (n_cells,) column id
    n_columns: int
    node_root: np.ndarray  # (n_points,) layer-0 node of each node's radial line
    node_outer: np.ndarray  # (n_points,) outer-surface node of that radial line
    normal: np.ndarray  # (n_cells, 3) unit mid-surface normal, inner -> outer
    h_normal: np.ndarray  # (n_cells,) layer thickness along the inner vertex normals
    layer_frac: np.ndarray  # (n_cells,) the cell's share of its column (radial edges)
    volume: np.ndarray  # (n_cells,)
    outer_area: np.ndarray  # (n_columns,) area of the column's outer face
    outer_face: np.ndarray  # (n_columns, 4) outer-face node ids


def _median_column_thickness(cols: _Columns) -> float:
    return float(np.median(np.bincount(cols.cell_column, weights=cols.h_normal)))


def tension_thickness(cols: _Columns, t_nominal: float | None) -> float | None:
    """The thickness :func:`stress_fields` integrates the tension over:
    ``t_nominal`` when every radial column is that thick along the inner
    surface's vertex normals (within ``UNIFORM_NORMAL_TOL``), else None — each
    column's measured normal thickness. Equal ray lengths are not enough: a
    ray that leaves the surface at an angle builds a thinner column."""
    if t_nominal is None:
        return None
    h = np.bincount(cols.cell_column, weights=cols.h_normal)
    off = np.abs(h / float(t_nominal) - 1.0)
    return float(t_nominal) if bool(np.all(off <= UNIFORM_NORMAL_TOL)) else None


def annotate_wall_thickness(wall) -> float | None:
    """Measure a swept wall's thickness along the inner surface's normals
    (median over its radial columns, mm) and record it in
    ``wall.lattice["thickness_normal_median"]``: the thickness the stress
    actually sees. A centreline sweep's rays run square to the centreline,
    so on a sac's shoulders they leave the surface at an angle, and the
    wall comes out thinner than its ray length (s1364: 1.30 vs 1.5 mm, read
    as +37% peak stress; D93). Returns None, with a note, for a mesh without
    radial columns."""
    if getattr(wall, "lattice", None) is None:
        return None
    try:
        value = round(_median_column_thickness(_columns(wall)), 4)
    except (ValueError, KeyError, IndexError) as exc:
        wall.lattice["thickness_normal_median"] = None
        wall.lattice["thickness_normal_note"] = f"not measurable: {exc}"
        return None
    wall.lattice["thickness_normal_median"] = value
    return value


def _columns(wall) -> _Columns:
    grid = wall.grid
    conn = np.asarray(grid.cells).reshape(-1, 9)[:, 1:]
    pts = np.asarray(grid.points, dtype=float)
    n_pts = grid.n_points
    pred = np.full(n_pts, -1, dtype=np.int64)
    pred[conn[:, 4:8].ravel()] = conn[:, 0:4].ravel()
    succ = np.full(n_pts, -1, dtype=np.int64)
    succ[conn[:, 0:4].ravel()] = conn[:, 4:8].ravel()
    root = np.arange(n_pts)
    for _ in range(n_pts):
        nxt = np.where(pred[root] >= 0, pred[root], root)
        if np.array_equal(nxt, root):
            break
        root = nxt
    top = np.arange(n_pts)
    for _ in range(n_pts):
        nxt = np.where(succ[top] >= 0, succ[top], top)
        if np.array_equal(nxt, top):
            break
        top = nxt
    node_outer = np.empty(n_pts, dtype=np.int64)
    node_outer[root] = top  # every node shares its line's outer node
    node_outer = node_outer[root]

    col_key = np.sort(root[conn[:, 0:4]], axis=1)
    _, cell_column = np.unique(col_key, axis=0, return_inverse=True)
    cell_column = cell_column.ravel()
    n_col = int(cell_column.max()) + 1

    bot, topq = pts[conn[:, 0:4]], pts[conn[:, 4:8]]
    radial = (topq - bot).mean(axis=1)
    mid = 0.5 * (bot + topq)
    nrm = np.cross(mid[:, 2] - mid[:, 0], mid[:, 3] - mid[:, 1])
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    flip = np.einsum("ij,ij->i", nrm, radial) < 0
    nrm[flip] *= -1.0
    # Layer thickness along the inner surface's VERTEX normals (the wall was
    # offset node by node): facet-to-facet it would read t cos(pi/n) on an
    # n-sided ring, a faceting artefact (-0.5 % at 32 samples).
    inner = np.asarray(wall.boundary["inner"].face_nodes)
    q = pts[inner]
    fn = 0.5 * np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1])
    vn = np.zeros((n_pts, 3))
    np.add.at(vn, inner.ravel(), np.repeat(fn, 4, axis=0))
    edges = topq - bot  # (n, 4, 3) the radial edges
    vroot = vn[root[conn[:, 0:4]]]
    vroot /= np.maximum(np.linalg.norm(vroot, axis=2), 1e-300)[:, :, None]
    h_n = np.abs(np.einsum("nki,nki->nk", edges, vroot).mean(axis=1))
    edge_len = np.linalg.norm(edges, axis=2).mean(axis=1)

    vol = np.abs(np.asarray(grid.compute_cell_sizes(length=False, area=False)["Volume"]))
    outermost = np.nonzero(succ[conn[:, 4]] < 0)[0]  # its top nodes go no further
    if len(outermost) != n_col or len(np.unique(cell_column[outermost])) != n_col:
        raise ValueError("the wall's radial columns do not each end in one outer cell")
    outer_cells = np.empty(n_col, dtype=np.int64)
    outer_cells[cell_column[outermost]] = outermost
    oface = conn[outer_cells][:, 4:8]
    q = pts[oface]
    area = 0.5 * np.linalg.norm(np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1]), axis=1)
    frac = edge_len / np.bincount(cell_column, weights=edge_len)[cell_column]
    return _Columns(cell_column, n_col, root, node_outer, nrm, h_n, frac, vol, area, oface)


@dataclass
class _Cap:
    name: str  # wall group: inlet / outlet
    centroid: np.ndarray
    normal_in: np.ndarray  # unit, from the cap plane into the vessel
    d_equiv: float  # 2 sqrt(A/pi) of the area inside the inner rim
    distal: bool = False
    end_zone_mm: float = 0.0
    end_zone_rule: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.name,
            "centroid_mm": [round(float(v), 3) for v in self.centroid],
            "normal_in": [round(float(v), 6) for v in self.normal_in],
            "inner_d_equiv_mm": round(self.d_equiv, 3),
            "end_zone_mm": round(self.end_zone_mm, 3),
            "end_zone_rule": self.end_zone_rule,
        }


def _cap(wall, name: str) -> _Cap:
    """Cap plane from the inner rim of a cap group (Newell vector area)."""
    grid = wall.grid
    pts = np.asarray(grid.points, dtype=float)
    conn = np.asarray(grid.cells).reshape(-1, 9)[:, 1:]
    group = wall.boundary[name]
    inner_nodes = set(np.unique(wall.boundary["inner"].face_nodes).tolist())
    edges = []
    for row in group.face_nodes:
        on = [int(v) for v in row if int(v) in inner_nodes]
        if len(on) == 2:
            edges.append(tuple(on))
    if len(edges) < 3:
        raise ValueError(f"the {name!r} cap has no inner rim ({len(edges)} edges)")
    nbr: dict[int, list[int]] = {}
    for a, b in edges:
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)
    if any(len(v) != 2 for v in nbr.values()):
        raise ValueError(f"the {name!r} cap's inner rim is not one closed loop")
    loop = [edges[0][0]]
    prev, cur = None, edges[0][0]
    while True:
        a, b = nbr[cur]
        nxt = b if a == prev else a
        if nxt == loop[0]:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
        if len(loop) > len(nbr):
            raise ValueError(f"the {name!r} cap's inner rim does not close")
    if len(loop) != len(nbr):
        raise ValueError(f"the {name!r} cap's inner rim is more than one loop")
    ring = pts[loop]
    c0 = ring.mean(axis=0)
    tri = np.cross(ring - c0, np.roll(ring, -1, axis=0) - c0)  # 2x triangle areas
    vec = 0.5 * tri.sum(axis=0)
    area = float(np.linalg.norm(vec))
    normal = vec / area
    w = 0.5 * np.einsum("ij,j->i", tri, normal)
    centroid = ((ring + np.roll(ring, -1, axis=0) + c0) / 3.0 * w[:, None]).sum(axis=0) / w.sum()
    cell_c = pts[conn[np.unique(group.cell_ids)]].mean(axis=1).mean(axis=0)
    if np.dot(cell_c - centroid, normal) < 0:
        normal = -normal
    return _Cap(name, centroid, normal, 2.0 * float(np.sqrt(area / np.pi)))


def bending_length(radius: float, thickness: float, nu: float) -> float:
    """Shell bending length λ = sqrt(R t) / (3 (1 − ν²))^¼: a clamp's local
    bending disturbance decays as exp(−x/λ) along a cylinder."""
    return float(np.sqrt(radius * thickness) / (3.0 * (1.0 - nu**2)) ** 0.25)


def _end_zones(caps: list[_Cap], end_zone: str, crop_report: dict | None,
               thickness: float, nu: float) -> list[str]:
    notes: list[str] = []
    rule = str(end_zone).strip().lower() or "neck"
    neck = None
    if crop_report:
        neck = (crop_report.get("landmarks") or {}).get("neck_d_equiv_mm")
    if rule == "neck":
        for cap in caps:
            if neck is not None and float(neck) > 0:
                cap.end_zone_mm = float(neck)
                cap.end_zone_rule = "one neck diameter, from the crop report"
            else:
                cap.end_zone_mm = cap.d_equiv
                cap.end_zone_rule = "one diameter of this cap, no neck in a crop report"
    elif rule == "bending":
        for cap in caps:
            lam = bending_length(0.5 * cap.d_equiv + 0.5 * thickness, thickness, nu)
            cap.end_zone_mm = BENDING_LENGTHS * lam
            cap.end_zone_rule = (
                f"{BENDING_LENGTHS:g} shell bending lengths, sqrt(R t)/(3(1-nu^2))^0.25 "
                f"each, R {0.5 * cap.d_equiv + 0.5 * thickness:.1f} mm"
            )
    else:
        try:
            mm = float(rule)
        except ValueError:
            raise ValueError(
                f"end_zone must be 'neck', 'bending' or a length in mm, got {end_zone!r}"
            ) from None
        if not np.isfinite(mm) or mm < 0:
            raise ValueError(f"end_zone length must be >= 0 mm, got {end_zone!r}")
        for cap in caps:
            cap.end_zone_mm = mm
            cap.end_zone_rule = "a set length"
    if rule != "neck" and neck is not None:
        notes.append(
            f"end zones set by '{rule}' instead of the default one neck diameter "
            f"({float(neck):.1f} mm)"
        )
    return notes


def _identify_distal(caps: list[_Cap], crop_report: dict | None) -> str:
    plane = ((crop_report or {}).get("planes") or {}).get("distal")
    if plane and "origin" in plane and "normal_out" in plane:
        o, nrm = np.asarray(plane["origin"], float), np.asarray(plane["normal_out"], float)
        dist = [abs(float(np.dot(c.centroid - o, nrm))) for c in caps]
        distal = int(np.argmin(dist))
        how = "the cap on the crop report's distal plane"
    else:
        distal = int(np.argmin([c.centroid[2] for c in caps]))
        how = "the lower-z cap (no crop report; +z superior assumed)"
    for i, c in enumerate(caps):
        c.distal = i == distal
    return how


# ---------------------------------------------------------------------------
# solve + read-back


def _von_mises(s: np.ndarray) -> np.ndarray:
    """ccx order sxx, syy, szz, sxy, sxz, syz."""
    sxx, syy, szz, sxy, sxz, syz = (s[..., k] for k in range(6))
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy**2 + sxz**2 + syz**2)
    )


def _tensor(s: np.ndarray) -> np.ndarray:
    t = np.empty(s.shape[:-1] + (3, 3))
    t[..., 0, 0], t[..., 1, 1], t[..., 2, 2] = s[..., 0], s[..., 1], s[..., 2]
    t[..., 0, 1] = t[..., 1, 0] = s[..., 3]
    t[..., 0, 2] = t[..., 2, 0] = s[..., 4]
    t[..., 1, 2] = t[..., 2, 1] = s[..., 5]
    return t


# C3D20 node order (Abaqus/ccx): corners 1-8, then the mid-edge nodes of
# edges 1-2, 2-3, 3-4, 4-1, 5-6, 6-7, 7-8, 8-5, 1-5, 2-6, 3-7, 4-8.
_HEX20_EDGES = np.array(
    [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
     (0, 4), (1, 5), (2, 6), (3, 7)]
)


def _hex20(wall) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
    """The wall's hexes as 20-node quadratic hexes with STRAIGHT mid-edge
    nodes (identical geometry). Corner ids are kept (0..N-1); mid nodes
    follow. Returns (points, conn20 (n, 20), {sorted corner edge: mid id})."""
    pts = np.asarray(wall.grid.points, dtype=float)
    conn = np.asarray(wall.grid.cells).reshape(-1, 9)[:, 1:]
    ends = np.sort(conn[:, _HEX20_EDGES], axis=2).reshape(-1, 2)
    uniq, inv = np.unique(ends, axis=0, return_inverse=True)
    n0 = len(pts)
    mids = (n0 + inv.ravel()).reshape(len(conn), 12)
    allpts = np.vstack([pts, 0.5 * (pts[uniq[:, 0]] + pts[uniq[:, 1]])])
    edge_mid = {(int(a), int(b)): n0 + k for k, (a, b) in enumerate(uniq)}
    return allpts, np.hstack([conn, mids]), edge_mid


def write_wall_deck(wall, out_path: str | Path, *, pressure_mpa: float,
                    E: float = DEFAULT_E_MPA, nu: float = DEFAULT_NU,
                    input_path: str | Path | None = None, ilt=None,
                    E_ilt: float | None = None, nu_ilt: float | None = None) -> Path:
    """The AAA wall-stress CalculiX deck: the swept wall as C3D20R (straight
    mid-edge nodes added), linear elastic, one *STATIC step (nlgeom off),
    pressure on ``inner``, ``inlet`` and ``outlet`` encastre (every node of
    their faces, mid-edge nodes included); integration-point stresses to the
    .dat, nodal U to the .frd. Units: mm, N, MPa."""
    from meshforge.export.abaqus import _chunk_ids, _face_label
    from meshforge.export.deck import Material
    from meshforge.export.neutral import guard_output_path

    material = Material.elastic(E, nu, name="WALL_MAT")  # validates E, nu
    p = float(pressure_mpa)
    if not np.isfinite(p) or p <= 0:
        raise ValueError(f"pressure must be positive (MPa), got {pressure_mpa!r}")
    for name in ("inner", "inlet", "outlet"):
        if name not in wall.boundary:
            raise ValueError(f"the wall has no {name!r} boundary group")
    out = Path(out_path).with_suffix(".inp")
    guard_output_path(out, input_paths=[input_path])
    out.parent.mkdir(parents=True, exist_ok=True)
    pts, conn20, edge_mid = _hex20(wall)
    conn = conn20[:, :8]
    lines = [
        "*HEADING",
        f"meshforge AAA wall stress (investigational): p {p:.10g} MPa on INNER, INLET + "
        f"OUTLET encastre, E {E:g} MPa, nu {nu:g}, C3D20R",
        "** UNITS: length=mm, force=N -> stress = N/mm^2 = MPa",
        "*NODE",
    ]
    lines += [f"{i}, {x:.10g}, {y:.10g}, {z:.10g}" for i, (x, y, z) in enumerate(pts, start=1)]
    lines.append("*ELEMENT, TYPE=C3D20R, ELSET=WALL")
    for eid, row in enumerate(conn20 + 1, start=1):
        ids = [str(int(v)) for v in row]
        lines.append(f"{eid}, " + ", ".join(ids[:15]) + ",")
        lines.append(", ".join(ids[15:]))
    for name in ("inlet", "outlet"):
        group = wall.boundary[name]
        nodes = set(np.unique(group.face_nodes).tolist())
        for f in group.face_nodes:
            for a, b in zip(f, np.roll(f, -1)):
                nodes.add(edge_mid[(int(min(a, b)), int(max(a, b)))])
        lines.append(f"*NSET, NSET={name.upper()}")
        lines.extend(_chunk_ids(np.array(sorted(nodes)) + 1))
    lines.append("*SURFACE, NAME=INNER_SURF, TYPE=ELEMENT")
    inner = wall.boundary["inner"]
    for cid, fnodes in zip(inner.cell_ids, inner.face_nodes):
        lines.append(f"{int(cid) + 1}, {_face_label(conn[int(cid)], fnodes, 'hexahedron')}")
    lines.extend(material.lines())
    lines.append("*SOLID SECTION, ELSET=WALL, MATERIAL=WALL_MAT")
    load_surface = "INNER_SURF"
    if ilt is not None:
        lines.extend(_ilt_lines(ilt, n_nodes=len(pts), n_elements=len(conn20),
                                E=DEFAULT_E_ILT_MPA if E_ilt is None else E_ilt,
                                nu=DEFAULT_NU_ILT if nu_ilt is None else nu_ilt))
        load_surface = "ILT_IN"
    lines += [
        "*STEP",
        "*STATIC",
        "*BOUNDARY",
        "INLET, 1, 3",
        "OUTLET, 1, 3",
        "*DSLOAD",
        f"{load_surface}, P, {p:.10g}",
        "*NODE FILE",
        "U",
        "*EL PRINT, ELSET=WALL",
        "S",
        "*END STEP",
    ]
    out.write_text("\n".join(lines) + "\n")
    return out


def _ilt_lines(ilt, *, n_nodes: int, n_elements: int, E: float, nu: float) -> list[str]:
    """The thrombus block (D91): C3D10 after the wall's nodes and elements,
    ILT_IN (lumen faces, the pressure surface), ILT_OUT tied to the wall's
    INNER_SURF without moving nodes (ADJUST=NO: adjusting inverted thin tets)."""
    from meshforge.export.deck import Material

    mat = Material.elastic(E, nu, name="ILT_MAT")  # validates E, nu
    lines = ["*NODE"]
    lines += [f"{n_nodes + i}, {x:.10g}, {y:.10g}, {z:.10g}"
              for i, (x, y, z) in enumerate(ilt.points, start=1)]
    lines.append("*ELEMENT, TYPE=C3D10, ELSET=ILT")
    for k, row in enumerate(ilt.tets + n_nodes + 1, start=1):
        lines.append(f"{n_elements + k}, " + ", ".join(str(int(v)) for v in row))
    for name, faces in (("ILT_IN", ilt.inner_faces), ("ILT_OUT", ilt.outer_faces)):
        lines.append(f"*SURFACE, NAME={name}, TYPE=ELEMENT")
        lines += [f"{n_elements + int(t) + 1}, {label}" for t, label in faces]
    lines.extend(mat.lines())
    lines += [
        "*SOLID SECTION, ELSET=ILT, MATERIAL=ILT_MAT",
        f"*TIE, NAME=ILT_WALL, POSITION TOLERANCE={ILT_TIE_TOLERANCE_MM:g}, ADJUST=NO",
        "ILT_OUT, INNER_SURF",
    ]
    return lines


def sac_from_wall(wall):
    """The closed sac surface the thrombus is tied to: the wall's ``inner``
    faces (each quad split in two), with its two end openings closed by
    planar caps."""
    import pyvista as pv

    quads = np.asarray(wall.boundary["inner"].face_nodes, dtype=np.int64)
    tris = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    surf = pv.PolyData(np.asarray(wall.grid.points, dtype=float),
                       np.column_stack([np.full(len(tris), 3), tris]).ravel()).clean()
    closed = surf.fill_holes(1e9).triangulate().clean()
    if closed.n_open_edges:
        raise ValueError(
            f"the wall's inner surface does not close with two planar caps "
            f"({closed.n_open_edges} open edges)"
        )
    # one consistent outward winding: fill_holes caps need not match the
    # wall faces, and a mixed winding corrupts the enclosed volume
    return closed.compute_normals(cell_normals=True, point_normals=False,
                                  consistent_normals=True, auto_orient_normals=True)


def solve_pressure(wall, pressure_mpa: float, out_dir: Path, stem: str, *,
                   E: float = DEFAULT_E_MPA, nu: float = DEFAULT_NU,
                   ccx: str | None = None, timeout: float = 600.0,
                   input_path: str | Path | None = None, ilt=None,
                   E_ilt: float | None = None, nu_ilt: float | None = None):
    """One CalculiX solve of :func:`write_wall_deck`. Returns (ip_stress
    (n_cells, n_ip, 6) in ccx order sxx, syy, szz, sxy, sxz, syz; corner-node
    displacement (n_points, 3) or None; SolveResult)."""
    from meshforge.solve.dat import parse_stresses
    from meshforge.solve.runner import solve_deck

    deck = write_wall_deck(wall, Path(out_dir) / stem, pressure_mpa=pressure_mpa, E=E,
                           nu=nu, input_path=input_path, ilt=ilt, E_ilt=E_ilt, nu_ilt=nu_ilt)
    res = solve_deck(deck, out_dir, ccx=ccx, timeout=timeout, read_elements=False)
    if res.dat_path is None:
        raise RuntimeError(f"ccx wrote no .dat for {deck.name} — no integration-point stress")
    raw = parse_stresses(res.dat_path.read_text())
    n_cells = wall.grid.n_cells
    if not raw:
        raise RuntimeError(f"no stress block in {res.dat_path.name}")
    keys = np.array(list(raw.keys()), dtype=np.int64)
    n_ip = int(keys[:, 1].max())
    if len(keys) != n_cells * n_ip or keys[:, 0].min() != 1 or keys[:, 0].max() != n_cells:
        raise RuntimeError(
            f"{res.dat_path.name} holds {len(keys)} integration-point stresses, expected "
            f"{n_cells} elements x {n_ip}"
        )
    ip = np.empty((n_cells, n_ip, 6))
    ip[keys[:, 0] - 1, keys[:, 1] - 1] = np.array(list(raw.values()))
    u = None
    disp = res.frd.fields.get("DISP")
    n = wall.grid.n_points
    if disp is not None and np.array_equal(res.frd.node_ids[:n], np.arange(1, n + 1)):
        u = np.asarray(disp.values[:n, :3], dtype=float)
    return ip, u, res


def weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """The smallest value v with (weight of values <= v) >= q % of the total
    (inverted CDF). ``q`` in percent."""
    values, weights = np.asarray(values, float).ravel(), np.asarray(weights, float).ravel()
    if values.size == 0:
        raise ValueError("weighted_percentile of an empty set")
    order = np.argsort(values, kind="stable")
    cw = np.cumsum(weights[order])
    k = int(np.searchsorted(cw, q / 100.0 * cw[-1], side="left"))
    return float(values[order][min(k, values.size - 1)])


@dataclass
class StressFields:
    """Per-solve fields: integration-point von Mises / max principal (MPa)
    and, per radial column of hexes, the in-plane stress resultant (N/mm)."""

    vm_ip: np.ndarray  # (n_cells, n_ip)
    p1_ip: np.ndarray  # (n_cells, n_ip)
    resultant: np.ndarray  # (n_columns, 3, 3): sum over the column of h_n * P sigma P
    col_normal: np.ndarray  # (n_columns, 3) unit, inner -> outer
    tension_col: np.ndarray  # (n_columns,) larger in-plane principal of the resultant


def stress_fields(ip: np.ndarray, cols: _Columns, thickness: float | None = None
                  ) -> StressFields:
    """``ip`` (n_cells, n_ip, 6), ccx order. The tension of a column is the
    larger principal value of N = sum_l h_l P sigma_l P (sigma_l the element's
    integration-point mean, P = I - n n^T with n the column's normal): the
    through-thickness integral of the in-plane stress. With ``thickness`` (the
    wall's nominal, uniform t) h_l = t x the layer's share of the column, i.e.
    N = t x the layer-mean in-plane stress, free of any normal estimate;
    without it, h_l is measured along the inner surface's vertex normals.
    :func:`tension_thickness` says which applies to a wall."""
    vm = _von_mises(ip)
    p1 = np.linalg.eigvalsh(_tensor(ip))[..., -1]
    mean = _tensor(ip.mean(axis=1))
    h = cols.h_normal if thickness is None else float(thickness) * cols.layer_frac
    n_col = cols.n_columns
    nsum = np.zeros((n_col, 3))
    np.add.at(nsum, cols.cell_column, cols.normal * h[:, None])
    ncol = nsum / np.linalg.norm(nsum, axis=1)[:, None]
    full = np.zeros((n_col, 3, 3))
    np.add.at(full, cols.cell_column, mean * h[:, None, None])
    proj = np.eye(3)[None] - ncol[:, :, None] * ncol[:, None, :]
    res = proj @ full @ proj
    helper = np.where(np.abs(ncol[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    e1 = np.cross(ncol, helper)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(ncol, e1)
    a = np.einsum("ci,cij,cj->c", e1, res, e1)
    b = np.einsum("ci,cij,cj->c", e1, res, e2)
    d = np.einsum("ci,cij,cj->c", e2, res, e2)
    t1 = 0.5 * (a + d) + np.sqrt((0.5 * (a - d)) ** 2 + b**2)
    return StressFields(vm, p1, res, ncol, t1)


def _to_points(cell_values: np.ndarray, conn: np.ndarray, weights: np.ndarray,
               n_points: int) -> np.ndarray:
    acc = np.zeros(n_points)
    wsum = np.zeros(n_points)
    np.add.at(acc, conn.ravel(), np.repeat(cell_values * weights, conn.shape[1]))
    np.add.at(wsum, conn.ravel(), np.repeat(weights, conn.shape[1]))
    return acc / np.where(wsum > 0, wsum, 1.0)


def _tension_points(cols: _Columns, t_col: np.ndarray, n_points: int) -> np.ndarray:
    """Column tension -> outer-surface nodes (outer-face-area weighted), then
    along each radial line so every node of it carries its outer node's value."""
    acc = np.zeros(n_points)
    wsum = np.zeros(n_points)
    w = np.repeat(cols.outer_area, 4)
    np.add.at(acc, cols.outer_face.ravel(), np.repeat(t_col, 4) * w)
    np.add.at(wsum, cols.outer_face.ravel(), w)
    on_outer = wsum > 0
    outer_val = np.full(n_points, np.nan)
    outer_val[on_outer] = acc[on_outer] / wsum[on_outer]
    return outer_val[cols.node_outer]


@dataclass
class EndZones:
    excluded: np.ndarray  # (n_cells,) bool
    caps: list  # [_Cap] with end_zone_mm, distal flag
    evaluated_fraction: float
    notes: list[str]

    @property
    def distal(self):
        return next(c for c in self.caps if c.distal)

    def to_dict(self, rule: str) -> dict[str, Any]:
        return {
            "rule": rule,
            "caps": {("distal" if c.distal else "proximal"): c.to_dict() for c in self.caps},
            "evaluated_volume_fraction": round(self.evaluated_fraction, 4),
            "evaluated_cells": int((~self.excluded).sum()),
            "distance": "along each cap's normal from its plane (element centroids)",
        }


def end_zones(wall, *, end_zone: str = "neck", crop_report: dict | None = None,
              thickness: float, nu: float = DEFAULT_NU,
              volume: np.ndarray | None = None) -> EndZones:
    """The cells inside each cap's end zone: element centroids closer than
    the zone length to the cap plane, measured along the cap normal into the
    vessel. ``end_zone``: 'neck' (one neck diameter from the crop report, else
    each cap's own inner equivalent diameter), 'bending' (3 shell bending
    lengths per cap) or a length in mm."""
    caps = [_cap(wall, "inlet"), _cap(wall, "outlet")]
    notes = [f"distal cap: {_identify_distal(caps, crop_report)}"]
    notes += _end_zones(caps, end_zone, crop_report, thickness, nu)
    centres = np.asarray(wall.grid.cell_centers().points)
    excluded = np.zeros(wall.grid.n_cells, dtype=bool)
    for cap in caps:
        d = (centres - cap.centroid) @ cap.normal_in
        excluded |= (d < cap.end_zone_mm) & (d > -1e-6 * max(1.0, cap.end_zone_mm))
    if volume is None:
        volume = np.abs(np.asarray(
            wall.grid.compute_cell_sizes(length=False, area=False)["Volume"]))
    frac = float(volume[~excluded].sum() / volume.sum())
    return EndZones(excluded, caps, frac, notes)


def pws_region_thickness(cols: _Columns, hot: np.ndarray, ip_w: np.ndarray) -> dict[str, float]:
    """The wall under the PWS: the measured normal thickness (inner vertex
    normals, per radial column) of the integration points ``hot`` (n_cells,
    n_ip) at or above the 99th percentile; volume-weighted median and the
    minimum, mm. A thin patch carries a stress ~t/h too high and can set the
    PWS while the wall's median thickness passes (Zenodo P04: median -3.3 %,
    a 0.77 mm patch over 1.6 % of the area; D94)."""
    h = np.bincount(cols.cell_column, weights=cols.h_normal)[cols.cell_column]
    h_ip = np.broadcast_to(h[:, None], hot.shape)[hot]
    return {"median": round(weighted_percentile(h_ip, ip_w[hot], 50.0), 4),
            "min": round(float(h_ip.min()), 4)}


def stress_metrics(fields: StressFields, cols: _Columns, keep: np.ndarray,
                   centres: np.ndarray, distal) -> dict[str, Any]:
    """PWS and the other numbers of one solve, over the cells in ``keep``."""
    n_ip = fields.vm_ip.shape[1]
    ip_w = np.repeat((cols.volume / n_ip)[:, None], n_ip, axis=1)
    vm_k, w_k = fields.vm_ip[keep], ip_w[keep]
    pws = weighted_percentile(vm_k, w_k, PWS_PERCENTILE)
    p1p = weighted_percentile(fields.p1_ip[keep], w_k, PWS_PERCENTILE)
    flat = np.where(keep[:, None], fields.vm_ip, -np.inf)
    c_pk = int(np.unravel_index(int(np.argmax(flat)), flat.shape)[0])
    hot = keep[:, None] & (fields.vm_ip >= pws)
    hot_cells = np.nonzero(hot.any(axis=1))[0]
    hw = (hot * ip_w).sum(axis=1)[hot_cells]
    hot_c = (centres[hot_cells] * hw[:, None]).sum(axis=0) / hw.sum()
    col_keep = np.zeros(cols.n_columns, dtype=bool)
    col_keep[cols.cell_column[keep]] = True  # a column counts if any of its cells does
    tk, ta = fields.tension_col[col_keep], cols.outer_area[col_keep]

    def height(x):
        return round(float((x - distal.centroid) @ distal.normal_in), 3)

    return {
        "von_mises_p99_mpa": round(pws, 6),
        "pws_kpa": round(pws * 1000.0, 3),
        "max_principal_p99_mpa": round(p1p, 6),
        "von_mises_max_mpa": round(float(vm_k.max()), 6),
        "peak": {
            "location_mm": [round(float(v), 3) for v in centres[c_pk]],
            "height_above_distal_cap_mm": height(centres[c_pk]),
            "element": c_pk + 1,
            "top1pct_centroid_mm": [round(float(v), 3) for v in hot_c],
            "top1pct_height_above_distal_cap_mm": height(hot_c),
            "top1pct_cells": int(hot_cells.size),
            "top1pct_wall_thickness_mm": pws_region_thickness(cols, hot, ip_w),
        },
        "tension_n_per_mm": {
            "p50": round(weighted_percentile(tk, ta, 50.0), 6),
            "p99": round(weighted_percentile(tk, ta, PWS_PERCENTILE), 6),
            "max": round(float(tk.max()), 6),
        },
    }


# ---------------------------------------------------------------------------
# the whole metric


from meshforge.solve.ilt import (  # noqa: E402  (D91 defaults, re-exported)
    DEFAULT_E_ILT_MPA,
    DEFAULT_ILT_SIZE_MM,
    DEFAULT_NU_ILT,
)

#: slave (thrombus) nodes within this distance of the wall's inner faces are tied
ILT_TIE_TOLERANCE_MM = 1.5

ASSUMPTIONS = (
    "investigational: not a diagnosis, not a rupture prediction",
    "wall thickness assumed {t} mm, uniform (not measured from the scan)",
    "no intraluminal thrombus modelled: wall stress is overstated where thrombus is present",
    "stiff-linear on the loaded (imaged) geometry: linear elastic E {E:g} MPa, nu {nu:g}, "
    "small strain (nlgeom off) - stress follows geometry and pressure, not the modulus",
    "uniform static pressure on the lumen-side wall surface; both ends clamped (encastre), "
    "stress within the end zones excluded",
)


@dataclass
class AaaStressResult:
    summary: dict[str, Any]
    grid: Any = None  # pv.UnstructuredGrid with the fields; None when refused before solving
    refused: list[str] = field(default_factory=list)

    @property
    def pws_mpa(self) -> float | None:
        """The summary's single ``pws_mpa`` (systolic von Mises p99, MPa), or None."""
        return self.summary.get("pws_mpa")

    def to_text(self) -> str:
        return aaa_stress_text(self.summary)


def run_aaa_stress(
    wall,
    summary: dict | None,
    *,
    out_dir: str | Path,
    stem: str = "wall",
    crop_report: dict | None = None,
    systolic_mmhg: float = DEFAULT_SYSTOLIC_MMHG,
    whatif_mmhg: float = DEFAULT_WHATIF_MMHG,
    E: float = DEFAULT_E_MPA,
    nu: float = DEFAULT_NU,
    end_zone: str = "neck",
    ccx: str | None = None,
    timeout: float = 600.0,
    input_path: str | Path | None = None,
    lumen=None,
    sac=None,
    E_ilt: float = DEFAULT_E_ILT_MPA,
    nu_ilt: float = DEFAULT_NU_ILT,
    ilt_size: float = DEFAULT_ILT_SIZE_MM,
) -> AaaStressResult:
    """Wall stress at a systolic and a what-if pressure (two real solves),
    PWS / principal / tension outside the end zones, fields on a copy of the
    wall grid. A refusal returns the sentinel (metrics None) with the reasons
    in ``refused`` and ``summary['refused']``; a wall that fails the gate is
    not solved at all."""
    t_start = time.perf_counter()
    p_sys, p_wif = mmhg_to_mpa(systolic_mmhg), mmhg_to_mpa(whatif_mmhg)
    lat = (summary or {}).get("lattice") or {}
    t_nom = lat.get("thickness_max")
    out: dict[str, Any] = {
        "metric": "AAA wall stress (investigational)",
        "method": METHOD,
        "units": {"length": "mm", "force": "N", "stress": "MPa", "tension": "N/mm"},
        "material": {"E_mpa": float(E), "nu": float(nu), "element": ELEMENT},
        "boundary": "pressure on 'inner'; 'inlet' and 'outlet' encastre",
        "pressures_mmhg": {"systolic": float(systolic_mmhg), "whatif": float(whatif_mmhg)},
        "pressures_mpa": {"systolic": p_sys, "whatif": p_wif},
        "pws_definition": (
            f"volume-weighted {PWS_PERCENTILE:g}th percentile of the integration-point von "
            f"Mises stress outside the end zones (each of an element's 2x2x2 points weighs "
            f"1/8 of its volume)"
        ),
        "wall": {
            "n_cells": int(wall.grid.n_cells), "n_points": int(wall.grid.n_points),
            "thickness_mm": None if t_nom is None else round(float(t_nom), 4),
            "sweep": lat.get("sweep"),
            "inverted": int(wall.quality.n_inverted),
            "capped": lat.get("n_capped"), "floored": lat.get("n_floored"),
        },
        # THE peak wall stress, once in the file (a reader may search any depth):
        # the systolic Cauchy von Mises p99, MPa; None = refused (the sentinel).
        "pws_mpa": None,
        "pws_measure": (
            "Cauchy von Mises stress, volume-weighted 99th percentile over the "
            "integration points outside the end zones, at the systolic pressure; MPa"
        ),
        "pws_pressure_mmhg": float(systolic_mmhg),
        "thrombus": {"modelled": False},
        "systolic": None, "whatif": None, "whatif_ratio": None,
        "refused": [], "notes": [],
    }
    reasons = wall_refusals(wall, summary)
    if reasons:
        out["refused"] = reasons
        out["elapsed_s"] = round(time.perf_counter() - t_start, 2)
        return AaaStressResult(out, None, reasons)

    cols = _columns(wall)
    measured = _median_column_thickness(cols)
    out["wall"]["thickness_measured_mm"] = round(measured, 4)
    t_min = lat.get("thickness_min")
    uniform = t_nom is not None and t_min is not None and abs(t_nom - t_min) <= 1e-6 * t_nom
    thickness = float(t_nom) if uniform else measured
    out["wall"]["thickness_mm"] = round(thickness, 4)
    # equal rays (uniform) are not a uniform wall: a ray crossing the sac at an
    # angle builds a thinner column, so the nominal t holds only if every column
    # is t thick along the normal
    t_tension = tension_thickness(cols, thickness) if uniform else None
    out["tension_definition"] = (
        f"t x the layer-mean in-plane stress resultant (t = {thickness:g} mm, the wall's "
        f"uniform nominal thickness), larger principal value; per column of hexes, "
        f"mapped to the outer-surface nodes"
        if t_tension is not None else
        "the in-plane stress integrated along the measured column thickness (inner-surface "
        "vertex normals), larger principal value; mapped to the outer-surface nodes"
        + ("" if not uniform else
           f" (the rays are {thickness:g} mm, but not every column is that thick along the "
           f"surface normal, within {UNIFORM_NORMAL_TOL:.0%})")
    )
    ez = end_zones(wall, end_zone=end_zone, crop_report=crop_report, thickness=thickness,
                   nu=nu, volume=cols.volume)
    out["notes"] += ez.notes
    out["end_zones"] = ez.to_dict(str(end_zone))
    if ez.evaluated_fraction < MIN_EVALUATED_FRACTION:
        rules = "; ".join(
            f"{'distal' if c.distal else 'proximal'} {c.end_zone_mm:.1f} mm, {c.end_zone_rule}"
            for c in ez.caps
        )
        reasons.append(
            f"the end zones ({rules}) leave {ez.evaluated_fraction:.1%} of the wall volume "
            f"(< {MIN_EVALUATED_FRACTION:.0%}): this crop is too short for a clamped-end PWS "
            f"under that rule - re-crop with longer ends, or set the end zone explicitly "
            f"('bending' or a length in mm; it is reported)"
        )

    ilt = None
    if lumen is not None and not reasons:
        from meshforge.geometry.crop import planes_from_report
        from meshforge.solve.ilt import build_ilt_mesh

        if not crop_report or not crop_report.get("planes"):
            raise ValueError(
                "modelling the thrombus needs the crop report's two cap planes "
                "(--crop-report): the lumen and the wall must share them"
            )
        if sac is None:
            # gmsh's 3-D fill hung (no error, no timeout) against a sac rebuilt
            # from a plane-swept wall's steep shoulder: ask for the real one
            raise ValueError(
                "modelling the thrombus needs the sac surface the wall was built from "
                "(--sac): the thrombus fills the space between it and the lumen"
            )
        ilt = build_ilt_mesh(lumen, sac, planes_from_report(crop_report), size=ilt_size)
        out["thrombus"] = {"modelled": True, "E_mpa": float(E_ilt), "nu": float(nu_ilt),
                           "element": "C3D10", **ilt.summary()}
        out["boundary"] = ("pressure on the lumen (thrombus inner faces); thrombus tied to "
                           "the wall's inner faces; wall 'inlet' and 'outlet' encastre")
        out["notes"] += ilt.notes

    grid = wall.grid.copy()
    conn = np.asarray(grid.cells).reshape(-1, 9)[:, 1:]
    centres = np.asarray(grid.cell_centers().points)
    keep = ~ez.excluded
    for key, p, mmhg in (("systolic", p_sys, systolic_mmhg), ("whatif", p_wif, whatif_mmhg)):
        ip, u, res = solve_pressure(
            wall, p, Path(out_dir), f"{stem}_p{mmhg:g}mmHg", E=E, nu=nu, ccx=ccx,
            timeout=timeout, input_path=input_path, ilt=ilt, E_ilt=E_ilt, nu_ilt=nu_ilt,
        )
        f = stress_fields(ip, cols, t_tension)
        suffix = "" if key == "systolic" else "_whatif"
        vm_cell, p1_cell = f.vm_ip.mean(axis=1), f.p1_ip.mean(axis=1)
        grid.cell_data[f"von_mises{suffix}"] = vm_cell
        grid.cell_data[f"max_principal{suffix}"] = p1_cell
        grid.cell_data[f"tension{suffix}"] = f.tension_col[cols.cell_column]
        grid.point_data[f"von_mises{suffix}"] = _to_points(vm_cell, conn, cols.volume, grid.n_points)
        grid.point_data[f"max_principal{suffix}"] = _to_points(p1_cell, conn, cols.volume,
                                                               grid.n_points)
        grid.point_data[f"tension{suffix}"] = _tension_points(cols, f.tension_col, grid.n_points)
        if key == "systolic" and u is not None:
            grid.point_data["DISP"] = u
        block: dict[str, Any] = {
            "pressure_mmhg": float(mmhg),
            "solve_s": round(res.elapsed, 2),
            "deck": str(res.deck), "dat": str(res.dat_path),
            "u_max_mm": None if res.u_max is None else round(res.u_max, 6),
            "von_mises_p99_mpa": None, "pws_kpa": None, "max_principal_p99_mpa": None,
            "von_mises_max_mpa": None, "peak": None, "tension_n_per_mm": None,
        }
        if not reasons:
            block.update(stress_metrics(f, cols, keep, centres, ez.distal))
        out[key] = block
    grid.cell_data["excluded_end_zone"] = ez.excluded.astype(np.uint8)
    pt_ex = np.zeros(grid.n_points, dtype=np.uint8)
    pt_ex[np.unique(conn[ez.excluded])] = 1
    grid.point_data["excluded_end_zone"] = pt_ex
    grid.cell_data.active_scalars_name = "von_mises"

    s, w = out["systolic"], out["whatif"]
    out["whatif_ratio"] = {
        "expected": round(float(whatif_mmhg) / float(systolic_mmhg), 6),
        "pws": (None if s["von_mises_p99_mpa"] is None
                else round(w["von_mises_p99_mpa"] / s["von_mises_p99_mpa"], 6)),
        "note": "two independent solves; linear (nlgeom off), so stress scales with pressure",
    }
    out["pws_mpa"] = s["von_mises_p99_mpa"]
    out["refused"] = reasons
    out["elapsed_s"] = round(time.perf_counter() - t_start, 2)
    return AaaStressResult(out, grid, reasons)


def aaa_stress_text(summ: dict[str, Any]) -> str:
    """Human-readable result (step text and CLI share it); always ends with
    the assumptions."""
    lines = ["aaa-stress  : AAA wall stress under blood pressure - INVESTIGATIONAL"]
    w = summ["wall"]
    lines.append(
        f"wall        : {w['n_cells']} hex, inverted {w['inverted']}, capped "
        f"{w['capped']}, floored {w['floored']}, sweep {w['sweep']}, thickness "
        f"{w['thickness_mm']} mm"
    )
    lines.append(f"method      : {summ['method']}")
    m = summ["material"]
    lines.append(
        f"material    : E {m['E_mpa']:g} MPa, nu {m['nu']:g}; {summ['boundary']}"
    )
    if summ["refused"]:
        for r in summ["refused"]:
            lines.append(f"REFUSED     : {r}")
    ez = summ.get("end_zones")
    if ez:
        for side, c in ez["caps"].items():
            lines.append(
                f"end zone    : {side} cap ({c['group']}, inner d_equiv "
                f"{c['inner_d_equiv_mm']:.1f} mm): {c['end_zone_mm']:.1f} mm, "
                f"{c['end_zone_rule']}"
            )
        lines.append(
            f"evaluated   : {ez['evaluated_volume_fraction']:.1%} of the wall volume "
            f"({ez['evaluated_cells']} cells)"
        )
    for key in ("systolic", "whatif"):
        b = summ.get(key)
        if not b:
            continue
        label = f"{b['pressure_mmhg']:g} mmHg"
        if b.get("von_mises_p99_mpa") is None:
            lines.append(f"{label:<12}: PWS n/a (refused); solve {b['solve_s']:.1f} s")
            continue
        pk, tn = b["peak"], b["tension_n_per_mm"]
        lines.append(
            f"{label:<12}: PWS {b['von_mises_p99_mpa'] * 1000:.1f} kPa (p99 von Mises); max principal "
            f"p99 {b['max_principal_p99_mpa'] * 1000:.1f} kPa; von Mises max "
            f"{b['von_mises_max_mpa'] * 1000:.1f} kPa"
        )
        lines.append(
            f"{'':<12}  peak at ({', '.join(f'{v:.1f}' for v in pk['location_mm'])}) mm, "
            f"{pk['height_above_distal_cap_mm']:.1f} mm above the distal cap; top-1% region "
            f"centroid {pk['top1pct_height_above_distal_cap_mm']:.1f} mm above it "
            f"({pk['top1pct_cells']} cells)"
        )
        hw = pk.get("top1pct_wall_thickness_mm")
        if hw:
            lines.append(
                f"{'':<12}  wall under the top-1% region: median {hw['median']:.3f} mm, min "
                f"{hw['min']:.3f} mm along the surface normal"
            )
        lines.append(
            f"{'':<12}  wall tension p50 {tn['p50']:.3f}, p99 {tn['p99']:.3f}, max "
            f"{tn['max']:.3f} N/mm; solve {b['solve_s']:.1f} s"
        )
    r = summ.get("whatif_ratio")
    if r and r.get("pws") is not None:
        lines.append(
            f"what-if     : PWS {summ['pressures_mmhg']['whatif']:g}/"
            f"{summ['pressures_mmhg']['systolic']:g} mmHg = {r['pws']:.4f} (pressure ratio "
            f"{r['expected']:.4f})"
        )
    for note in summ.get("notes", []):
        lines.append(f"note        : {note}")
    lines.append(f"runtime     : {summ.get('elapsed_s', 0):.1f} s")
    t = summ["wall"].get("thickness_mm")
    lines.append("assumptions :")
    thr = summ.get("thrombus") or {}
    for a in ASSUMPTIONS:
        if a.startswith("no intraluminal thrombus") and thr.get("modelled"):
            a = (f"intraluminal thrombus modelled: a linear-elastic solid between lumen and "
                 f"wall (E {thr['E_mpa']:g} MPa, nu {thr['nu']:g}, C3D10), tied to the wall; "
                 f"the pressure acts on the lumen")
        lines.append(
            "  - " + a.format(t="?" if t is None else f"{float(t):g}", E=m["E_mpa"], nu=m["nu"])
        )
    return "\n".join(lines) + "\n"
