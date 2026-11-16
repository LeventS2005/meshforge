"""Structured lumen (blood-volume) fill (V47).

Fills the interior of a tubular surface with hexahedra: stations along the
validated centreline (embedded > skeleton, the same choice the wall lattice
makes), per-station lumen rings ray-cast from the centreline, each ring
O-grid-filled in its own frame plane (V26 ``ogrid_disk`` with a PINNED block
topology so every station meshes identically), consecutive disks swept into
hexes. Where the structured fill fails — a section too far from a disk, or a
bend so tight the swept cells fold (the fold number F = r*kappa, probed
session 34/35) — ``method="auto"`` falls back to a gmsh tetrahedral fill of
the SAME station-ring boundary shell, with a loud note naming the fold
number. Boundary groups are ``wall`` / ``inlet`` / ``outlet`` either way.

Probe evidence (D75): swept from the validated centreline, 19 of the 20 real
cardiac arches reach ``inverted == 0`` structured; smoothing is a
do-no-harm-guarded pass (V21 pattern in 3-D) because a plain Laplacian
BREAKS valid meshes and true Winslow cannot fix the fold obstruction.

End caps (D81): a centreline-normal section at the centreline end is NOT the
vessel's end section when the surface ends in a tilted flat cap — rays on one
side exit through the cap within a millimetre and the ``inlet``/``outlet``
patch comes out a sliver (−62% to −95% on the real arches). Each end whose
surface has a flat cap is therefore anchored ON that cap: the end station is
the cap itself (radii taken exactly from the cap's rim polygon), reached by a
short transition from the first interior station that clears the cap, so the
patch IS the cap. Interior stations are untouched.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyvista as pv

from meshforge.geometry import parallel_transport_frames
from meshforge.meshing.base import BoundaryGroup, WallMesh
from meshforge.quality.volume import volume_quality

_LABELS = {0: "wall", 2: "inlet", 3: "outlet"}

# Cap detection / anchoring constants (D81).
_CAP_ANGLE_TOL = 0.5  # deg: adjacent faces this close in normal are coplanar
_CAP_MIN_RIM = 12  # rim vertices: a cap closes a whole ring of the tube
_CAP_MIN_CREASE = 10.0  # deg: smallest cap-to-wall dihedral along the rim
_CAP_MAX_TILT = 60.0  # deg: a cap must face along the centreline end
_CAP_SEARCH = 3.0  # cap centroid within this many radii of the end station
_ANCHOR_CLEARANCE = 1.25  # anchor section clears the cap by this x its radius
_ANCHOR_MAX_SHEAR = 30.0  # deg: preferred max angle, transition path vs planes
_ANCHOR_MAX_LAYERS = 4  # a deeper, less sheared anchor is sought this far


def _cast_ring(surface, centre, u, v, cos_t, sin_t, ray_len):
    """Ray-cast one station ring in the plane spanned by (u, v).

    Returns (radii, cells): first-hit distance and hit cell id per ray, with
    NaN / -1 where the ray misses (the caller decides whether that is fatal).
    """
    radii = np.full(len(cos_t), np.nan)
    cells = np.full(len(cos_t), -1, dtype=np.int64)
    for j in range(len(cos_t)):
        d = cos_t[j] * u + sin_t[j] * v
        hits, ids = surface.ray_trace(centre, centre + d * ray_len, first_point=True)
        if len(hits) == 0:
            continue
        radii[j] = np.linalg.norm(hits - centre)
        cells[j] = int(ids[0])
    return radii, cells


def _ring_area(r: np.ndarray) -> float:
    """Area of the station polygon with equal-angle spokes ``r``."""
    return 0.5 * math.sin(2.0 * math.pi / len(r)) * float(np.sum(r * np.roll(r, -1)))


@dataclass
class _Cap:
    """A flat end cap: a disk-shaped set of coplanar surface triangles."""

    cells: np.ndarray  # cell ids of the cap triangles
    centre: np.ndarray  # area centroid (on the cap plane)
    normal: np.ndarray  # unit plane normal (sign arbitrary)
    area: float
    loop: np.ndarray  # (m, 3) ordered rim polygon


def _coplanar_regions(surface: pv.PolyData):
    """Edge-connected regions of >= 3 triangles whose normals agree within
    ``_CAP_ANGLE_TOL`` — the end-cap candidates.

    Returns (tri, pts, regions), regions a list of (cells, area, area
    centroid, mean unit normal, crease) where crease is the smallest dihedral
    (deg) between the region and a neighbouring face along its rim; None when
    the surface is not all triangles (its cell ids would then not be the
    ray-trace cell ids)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    faces = np.asarray(surface.faces)
    if surface.n_cells == 0 or len(faces) != 4 * surface.n_cells:
        return None
    tri = faces.reshape(-1, 4)
    if not (tri[:, 0] == 3).all():
        return None
    tri = tri[:, 1:].astype(np.int64)
    pts = np.asarray(surface.points, dtype=float)
    cross = np.cross(pts[tri[:, 1]] - pts[tri[:, 0]], pts[tri[:, 2]] - pts[tri[:, 0]])
    dbl = np.linalg.norm(cross, axis=1)
    nrm = np.divide(
        cross, dbl[:, None], out=np.zeros_like(cross), where=dbl[:, None] > 0
    )
    n_face = len(tri)
    edge = np.sort(
        np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1
    )
    owner = np.tile(np.arange(n_face), 3)
    key = edge[:, 0] * len(pts) + edge[:, 1]
    order = np.argsort(key, kind="stable")
    shared = key[order][1:] == key[order][:-1]
    f0, f1 = owner[order][:-1][shared], owner[order][1:][shared]
    flat = np.abs(np.einsum("ij,ij->i", nrm[f0], nrm[f1])) >= math.cos(
        math.radians(_CAP_ANGLE_TOL)
    )
    graph = coo_matrix(
        (np.ones(int(flat.sum())), (f0[flat], f1[flat])), shape=(n_face, n_face)
    )
    _, label = connected_components(graph, directed=False)
    sizes = np.bincount(label)
    rim = label[f0] != label[f1]
    dihedral = np.degrees(np.arccos(np.clip(
        np.abs(np.einsum("ij,ij->i", nrm[f0[rim]], nrm[f1[rim]])), 0.0, 1.0
    )))
    crease = np.full(len(sizes), 180.0)
    np.minimum.at(crease, label[f0[rim]], dihedral)
    np.minimum.at(crease, label[f1[rim]], dihedral)
    by_label = np.argsort(label, kind="stable")
    start = np.concatenate([[0], np.cumsum(sizes)])
    area_f = 0.5 * dbl
    lab_area = np.bincount(label, weights=area_f)
    cen = pts[tri].mean(axis=1)
    # sign-align every face normal with its region's first face, then average
    ref = nrm[by_label[start[:-1]]][label]
    sgn = np.sign(np.einsum("ij,ij->i", nrm, ref))
    regions = []
    for lab in np.nonzero((sizes >= 3) & (lab_area > 0))[0]:
        cells = by_label[start[lab]:start[lab + 1]]
        w = area_f[cells]
        normal = ((w * sgn[cells])[:, None] * nrm[cells]).sum(axis=0)
        normal /= np.linalg.norm(normal)
        centre = (w[:, None] * cen[cells]).sum(axis=0) / lab_area[lab]
        regions.append(
            (cells, float(lab_area[lab]), centre, normal, float(crease[lab]))
        )
    return tri, pts, regions


def _as_cap(tri, pts, cells, area, centre, normal, crease) -> _Cap | None:
    """The region as a flat end cap, or None.

    A cap is a planar disk with one simple rim that closes a whole ring of the
    tube (>= ``_CAP_MIN_RIM`` rim vertices) and meets the wall at a real crease
    (>= ``_CAP_MIN_CREASE`` everywhere along the rim). The two rim tests keep
    out the small coplanar facet groups a loft's lateral wall can contain
    (<= 9 rim vertices and creases <= 14.3 deg on the real arches, against
    48-vertex caps creased >= 17.6 deg) and a straight tube's flat wall strips
    (5.6 deg creases)."""
    if crease < _CAP_MIN_CREASE:
        return None
    ftri = tri[cells]
    verts = np.unique(ftri)
    if np.abs((pts[verts] - centre) @ normal).max() > 1e-3 * math.sqrt(area / math.pi):
        return None  # gently curved, not flat
    fedge = np.sort(
        np.concatenate([ftri[:, [0, 1]], ftri[:, [1, 2]], ftri[:, [2, 0]]]), axis=1
    )
    uniq, count = np.unique(fedge, axis=0, return_counts=True)
    if (count > 2).any() or len(verts) - len(uniq) + len(cells) != 1:
        return None  # non-manifold, or not a disk (e.g. an annulus)
    loop = _boundary_loop(uniq[count == 1])
    if loop is None or len(loop) < _CAP_MIN_RIM:
        return None
    return _Cap(cells=cells, centre=centre, normal=normal, area=area, loop=pts[loop])


def _boundary_loop(bedges: np.ndarray) -> np.ndarray | None:
    """Order boundary edges into ONE simple cycle of vertex ids, else None."""
    nbr: dict[int, list[int]] = {}
    for a, b in bedges:
        nbr.setdefault(int(a), []).append(int(b))
        nbr.setdefault(int(b), []).append(int(a))
    if len(nbr) < 3 or any(len(v) != 2 for v in nbr.values()):
        return None
    start = int(bedges[0, 0])
    loop, prev, cur = [start], -1, start
    while True:
        a, b = nbr[cur]
        prev, cur = cur, (a if a != prev else b)
        if cur == start:
            break
        loop.append(cur)
        if len(loop) > len(nbr):
            return None
    return np.asarray(loop) if len(loop) == len(nbr) else None


def flat_cap_cells(surface: pv.PolyData) -> list[np.ndarray]:
    """Cell ids of every flat end cap of ``surface`` — each a planar disk the
    D81 anchoring would accept (coplanar within ``_CAP_ANGLE_TOL``, one rim
    of >= ``_CAP_MIN_RIM`` vertices, creased >= ``_CAP_MIN_CREASE`` to the
    wall). Empty when the surface is not all triangles."""
    found = _coplanar_regions(surface)
    if found is None:
        return []
    tri, pts, regions = found
    return [r[0] for r in regions if _as_cap(tri, pts, *r) is not None]


def _end_caps(surface: pv.PolyData, ends: dict) -> dict[str, _Cap]:
    """The flat cap belonging to each centreline end.

    ``ends`` maps an end name to (end station, end tangent, search radius). A
    candidate region must face along the end tangent (within
    ``_CAP_MAX_TILT``) with its centroid within the search radius plus its own
    equivalent radius; the nearest one that is a planar disk wins. A cap
    claimed by both ends is ambiguous and anchors neither."""
    found = _coplanar_regions(surface)
    if found is None:
        return {}
    tri, pts, regions = found
    min_cos = math.cos(math.radians(_CAP_MAX_TILT))
    picked: dict[str, tuple[int, _Cap]] = {}
    for end, (p, t, reach) in ends.items():
        dist = [float(np.linalg.norm(r[2] - p)) for r in regions]
        cands = sorted(
            (dist[i], i)
            for i, (_, area, _, n, _) in enumerate(regions)
            if abs(float(n @ t)) >= min_cos
            and dist[i] <= reach + math.sqrt(area / math.pi)
        )
        for _, i in cands:
            cap = _as_cap(tri, pts, *regions[i])
            if cap is not None:
                picked[end] = (i, cap)
                break
    if len(picked) == 2 and picked["inlet"][0] == picked["outlet"][0]:
        return {}
    return {end: cap for end, (_, cap) in picked.items()}


def _cap_radii(cap: _Cap, centre, u, v, cos_t, sin_t) -> np.ndarray | None:
    """Exact spokes from ``centre`` to the cap rim, in the cap plane.

    A ray cast IN the cap plane is coplanar with the cap triangles (the VTK
    hit is then arbitrary), so the end station takes its radii from the rim
    polygon itself: the nearest crossing of each 2-D spoke with a rim edge.
    None when a spoke crosses no edge (the centre is not inside the rim)."""
    rel = cap.loop - centre
    a = np.column_stack([rel @ u, rel @ v])
    e = np.roll(a, -1, axis=0) - a
    d = np.column_stack([cos_t, sin_t])
    # spoke t*d meets edge a + w*e:  t = (a x e)/(d x e),  w = (a x d)/(d x e)
    den = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (a[None, :, 0] * e[None, :, 1] - a[None, :, 1] * e[None, :, 0]) / den
        w = (a[None, :, 0] * d[:, None, 1] - a[None, :, 1] * d[:, None, 0]) / den
    ok = (np.abs(den) > 1e-14) & (t > 0) & (w >= -1e-12) & (w <= 1.0 + 1e-12)
    t = np.where(ok, t, np.inf).min(axis=1)
    return t if np.isfinite(t).all() else None


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix about the unit ``axis``."""
    k = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


class _AnchorError(ValueError):
    """This end cannot be anchored on its cap; the caller keeps the old end."""


def _transition(surface, cap, frame, t_out, n_out, tau, spacing, ray_len,
                cos_t, sin_t, cap_cells):
    """Stations from an anchor (exclusive) to its cap (inclusive).

    The centre moves straight to the cap centroid while the anchor's frame
    rotates through ``tau`` about t_out x n_out — twist-free, and the last
    plane IS the cap plane, whose spokes come exactly from the rim. None when
    a ring misses, exits through a cap, or a centre leaves the surface."""
    p_a, t_a, u_a, v_a = frame
    axis = np.cross(t_out, n_out)
    sin_a = float(np.linalg.norm(axis))
    k = max(1, round(float(np.linalg.norm(cap.centre - p_a)) / spacing))
    out = []
    for j in range(1, k + 1):
        s = j / k
        rot = _rotation(axis / sin_a, s * tau) if sin_a > 1e-12 else np.eye(3)
        centre = p_a + s * (cap.centre - p_a)
        t_j, u_j, v_j = rot @ t_a, rot @ u_a, rot @ v_a
        if j == k:
            r_j = _cap_radii(cap, centre, u_j, v_j, cos_t, sin_t)
            if r_j is None:
                raise _AnchorError("the cap centroid is not inside its own rim")
        else:
            r_j, hit = _cast_ring(surface, centre, u_j, v_j, cos_t, sin_t, ray_len)
            if np.isnan(r_j).any() or np.isin(hit, cap_cells).any():
                return None
        out.append((centre, t_j, u_j, v_j, r_j, s))
    if k > 1:
        from meshforge.geometry.compat import interior_point_mask

        probe = pv.PolyData(np.array([o[0] for o in out[:-1]]))
        if not interior_point_mask(probe, surface, check_surface=False).all():
            return None
    return out


def _anchor_end(surface, cap, inward, sgn, geo, cap_cells, spacing, ray_len,
                cos_t, sin_t):
    """Anchor one end of the station sequence on its flat cap.

    ``inward`` lists station indices from this end inward; ``sgn`` is +1 at the
    outlet (outward = +tangent) and -1 at the inlet. A valid anchor sits at
    least half a spacing inside the cap, is not clipped by a cap, and has room
    to turn: moving the centre straight to the cap centroid while rotating the
    plane through the tilt tau keeps every transition plane's line of contact
    with its neighbour — and the last one's with the cap plane — at least
    min(h, ahead)/tau from its centre (h = depth below the cap plane, ahead =
    distance to the cap along the tangent), which must clear the section
    radius; its transition rings must stay inside the lumen.

    The outermost valid station wins unless the straight path to the cap
    centroid shears its hexes by more than ``_ANCHOR_MAX_SHEAR`` against the
    station planes (an eccentric cap); then the first valid station within
    ``_ANCHOR_MAX_LAYERS`` transition layers that is sheared less is taken.
    Stations between the anchor and the cap are dropped. Returns (anchor,
    transition, tilt in degrees, h); raises _AnchorError when no station
    qualifies."""
    P, T, U, V, radii, cells = geo
    n_out = cap.normal if float(cap.normal @ (sgn * T[inward[0]])) >= 0 else -cap.normal
    why: list[str] = []
    first = None
    for a in inward[: max(1, len(inward) // 2)]:
        h = float((cap.centre - P[a]) @ n_out)
        if h < 0.5 * spacing:
            continue  # at (or beyond) the cap plane: no room for a layer
        reach = float(np.linalg.norm(cap.centre - P[a]))
        if first is not None and reach > _ANCHOR_MAX_LAYERS * spacing:
            break
        if np.isnan(radii[a]).any() or np.isin(cells[a], cap_cells).any():
            why.append(f"station {a} is clipped")
            continue
        t_out = sgn * T[a]
        tau = math.acos(min(1.0, max(-1.0, float(t_out @ n_out))))
        ahead = float((cap.centre - P[a]) @ t_out)
        room = min(h, ahead) / max(tau, 1e-12)
        if room < _ANCHOR_CLEARANCE * float(radii[a].max()):
            why.append(
                f"station {a} has no room for its {math.degrees(tau):.0f}° tilt"
            )
            continue
        path = (cap.centre - P[a]) / reach
        shear = math.degrees(
            math.acos(min(1.0, float(path @ t_out), float(path @ n_out)))
        )
        if first is not None and shear > _ANCHOR_MAX_SHEAR:
            continue
        trans = _transition(surface, cap, (P[a], T[a], U[a], V[a]), t_out, n_out,
                            tau, spacing, ray_len, cos_t, sin_t, cap_cells)
        if trans is None:
            why.append(f"the transition from station {a} leaves the lumen")
            continue
        if room < max(float(o[4].max()) for o in trans):
            why.append(f"the transition from station {a} would fold")
            continue
        found = (a, trans, math.degrees(tau), h)
        if shear <= _ANCHOR_MAX_SHEAR:
            return found
        first = first or found
    if first is not None:
        return first
    raise _AnchorError(
        "no station near the end clears the cap"
        + (f": {'; '.join(why[:3])}" if why else "")
    )


def _anchor_ends(surface, geo, arc, end_radius, cos_t, sin_t, ray_len):
    """Anchor both ends on their flat caps (D81).

    Stations strictly between the two anchors are kept exactly as sampled — a
    ray miss at any of them is still fatal. Returns the final (stations,
    tangents, normals, binormals, radii, arc) and the notes."""
    P, T, U, V, radii, cells = geo
    S, n_theta = radii.shape
    notes: list[str] = []
    picked: dict[str, _Cap] = {}
    if S >= 3:
        picked = _end_caps(surface, {
            "inlet": (P[0], T[0], _CAP_SEARCH * end_radius[0]),
            "outlet": (P[-1], T[-1], _CAP_SEARCH * end_radius[1]),
        })
    cap_cells = (
        np.concatenate([c.cells for c in picked.values()]) if picked
        else np.empty(0, dtype=np.int64)
    )
    spacing = float(np.median(np.diff(arc))) if S > 1 else 0.0
    ends: dict[str, tuple] = {}
    for end, sgn in (("inlet", -1), ("outlet", 1)):
        keep = f"the {end} patch is the centreline-normal section at the centreline end"
        cap = picked.get(end)
        if cap is None:
            notes.append(
                f"{end}: no flat end cap found — {keep}" if S >= 3
                else f"{end}: too few stations to anchor an end cap — {keep}"
            )
            continue
        inward = list(range(S)) if sgn < 0 else list(range(S - 1, -1, -1))
        try:
            ends[end] = (cap, *_anchor_end(surface, cap, inward, sgn, geo, cap_cells,
                                           spacing, ray_len, cos_t, sin_t))
        except _AnchorError as exc:
            notes.append(f"{end}: not anchored on its flat end cap ({exc}) — {keep}")
    lo = ends["inlet"][1] if "inlet" in ends else 0
    hi = ends["outlet"][1] if "outlet" in ends else S - 1
    if lo > hi:
        notes.append(
            "end-cap anchoring dropped: the two end transitions overlap (too few "
            "stations) — both patches are centreline-normal sections"
        )
        ends, lo, hi = {}, 0, S - 1

    for i in range(lo, hi + 1):
        miss = np.nonzero(np.isnan(radii[i]))[0]
        if len(miss):
            raise ValueError(
                f"lumen ray missed the surface at station {i}, θ index "
                f"{int(miss[0])}; centerline may exit the surface or ray length "
                "is too short"
            )
    parts = [tuple(x[lo:hi + 1] for x in (P, T, U, V, radii, arc))]
    for end, sgn in (("inlet", -1), ("outlet", 1)):
        if end not in ends:
            i_end = lo if sgn < 0 else hi
            n_clip = int(np.isin(cells[i_end], cap_cells).sum())
            if n_clip:
                notes.append(
                    f"{end}: {n_clip} of {n_theta} rays exit through the end cap — "
                    f"the {end} patch (area {_ring_area(radii[i_end]):.4g}) is "
                    "clipped smaller than the vessel section"
                )
            continue
        cap, a, trans, tilt, depth = ends[end]
        length = float(np.linalg.norm(cap.centre - P[a]))
        block = (
            np.array([o[0] for o in trans]), np.array([o[1] for o in trans]),
            np.array([o[2] for o in trans]), np.array([o[3] for o in trans]),
            np.array([o[4] for o in trans]),
            arc[a] + sgn * length * np.array([o[5] for o in trans]),
        )
        if sgn < 0:
            parts.insert(0, tuple(x[::-1] for x in block))
        else:
            parts.append(block)
        notes.append(
            f"{end}: anchored on its flat end cap — patch area "
            f"{_ring_area(trans[-1][4]):.4g} (cap {cap.area:.4g}); the cap is "
            f"tilted {tilt:.1f}° to the centreline at station {a}, "
            f"{depth:.3g} inside it, reached in {len(trans)} transition layer(s)"
        )
        inner = _ring_area(radii[a])
        if cap.area < 0.5 * inner:
            notes.append(
                f"{end}: the surface itself narrows sharply at this end — the "
                f"cap (area {cap.area:.4g}) is {100.0 * cap.area / inner:.0f}% of "
                f"the section {depth:.3g} inside it (area {inner:.4g})"
            )
    out = tuple(np.concatenate([p[k] for p in parts]) for k in range(6))
    return out, notes


def _station_geometry(surface, centerline, *, n_theta: int, n_axial: int):
    """Stations, frames and ray-cast lumen radii (the wall-lattice sampling),
    each end anchored on the surface's flat end cap where it has one (D81).

    Returns (stations, tangents, normals, binormals, radii, arc, cos_t, sin_t,
    notes). Fails loud on a ray miss at a kept station — a centreline that
    exits the surface cannot anchor ANY lumen fill (structured or tet), so
    there is no fallback from here.
    """
    if centerline.closed:
        raise ValueError(
            "lumen fill supports open centrelines only (V47); a closed-loop "
            "lumen is a solid torus — a different block graph"
        )
    idx = np.unique(
        np.linspace(0, len(centerline.points) - 1, n_axial).round().astype(int)
    )
    stations = centerline.points[idx]
    tangents = centerline.tangents[idx]
    normals, binormals = parallel_transport_frames(tangents, closed=False)
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    ray_len = 5.0 * float(centerline.radii.max())
    s_count = len(stations)
    radii = np.empty((s_count, n_theta))
    cells = np.empty((s_count, n_theta), dtype=np.int64)
    for i in range(s_count):
        radii[i], cells[i] = _cast_ring(
            surface, stations[i], normals[i], binormals[i], cos_t, sin_t, ray_len
        )
    arc = centerline.arc_lengths[idx]
    end_radius = [
        float(np.nanmax(np.append(radii[k], centerline.radii[idx[k]]))) for k in (0, -1)
    ]
    (stations, tangents, normals, binormals, radii, arc), notes = _anchor_ends(
        surface, (stations, tangents, normals, binormals, radii, cells), arc,
        end_radius, cos_t, sin_t, ray_len,
    )
    return stations, tangents, normals, binormals, radii, arc, cos_t, sin_t, notes


def _fold_numbers(tangents, arc, radii) -> np.ndarray:
    """Per-station fold number F = r_max * kappa (folds where F >= 1)."""
    ds = np.gradient(arc)
    kvec = np.gradient(tangents, axis=0) / ds[:, None]
    kappa = np.linalg.norm(kvec, axis=1)
    return radii.max(axis=1) * kappa


def _fold_note(F: np.ndarray) -> str:
    return (
        f"fold number F=r·κ: max {float(F.max()):.2f}, "
        f"{int((F >= 1.0).sum())} station(s) with F >= 1"
    )


def _station_disks(radii, cos_t, sin_t, *, core_fraction: float):
    """O-grid every station ring with ONE pinned block topology.

    Counts are pinned from the median-perimeter station; a station whose
    section is too far from a disk raises ``ogrid_disk``'s loud ValueError
    (the auto cascade turns that into the tet fallback)."""
    from meshforge.meshing.ogrid import ogrid_disk

    S = radii.shape[0]
    rings2d = [
        np.column_stack([radii[i] * cos_t, radii[i] * sin_t]) for i in range(S)
    ]
    perims = np.array([
        float(np.linalg.norm(np.diff(np.vstack([r, r[:1]]), axis=0), axis=1).sum())
        for r in rings2d
    ])
    ref = int(np.argsort(perims)[S // 2])
    # target ~ ring segment length at the median station -> n_t ~ n_theta/4.
    # The per-quadrant count must not depend on rounding: for n_theta % 4 == 0
    # the ratio quarter/target is exactly n_theta/4, a knife edge for ceil, so
    # it is computed from the integer n_theta instead and handed to ogrid_disk.
    target = perims[ref] / len(cos_t)
    pin_t = max(2, -(-len(cos_t) // 4))
    bm_ref = ogrid_disk([rings2d[ref]], target_size=target,
                        core_fraction=core_fraction, n_t=pin_t)
    # Pin the reference radial count: parse it from the V26 note (the mesh's
    # own declared layout — kept in one place there).
    note = bm_ref.notes[0]
    pin_r = int(note.split("n_r ")[1].split(" ")[0])
    disks = []
    quads = None
    for i in range(S):
        bm = ogrid_disk([rings2d[i]], target_size=target,
                        core_fraction=core_fraction, n_t=pin_t, n_r=pin_r)
        if quads is None:
            quads, n_node, rim = bm.quads, len(bm.points), bm.edge_groups["outer_rim"]
        elif len(bm.points) != n_node or bm.quads.shape != quads.shape:
            raise ValueError(
                f"O-grid topology drift at station {i} despite pinned counts "
                f"({len(bm.points)} vs {n_node} nodes) — cannot sweep"
            )
        disks.append(bm.points)
    return disks, quads, rim, n_node


def _sweep(stations, normals, binormals, disks, quads, n_node):
    """Lift each 2-D disk into its station plane and sweep hexes."""
    S = len(stations)
    pts = np.vstack([
        stations[i]
        + disks[i][:, 0:1] * normals[i][None, :]
        + disks[i][:, 1:2] * binormals[i][None, :]
        for i in range(S)
    ])
    nq = len(quads)
    hexes = np.empty(((S - 1) * nq, 8), dtype=np.int64)
    for i in range(S - 1):
        b0, b1 = i * n_node, (i + 1) * n_node
        hexes[i * nq:(i + 1) * nq, 0:4] = quads + b0
        hexes[i * nq:(i + 1) * nq, 4:8] = quads + b1
    return pts, hexes


def _hex_grid(pts, hexes) -> pv.UnstructuredGrid:
    cells = np.column_stack(
        [np.full(len(hexes), 8, dtype=np.int64), hexes]
    ).ravel()
    ctypes = np.full(len(hexes), pv.CellType.HEXAHEDRON, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, ctypes, pts)
    grid.cell_data["region"] = np.ones(grid.n_cells, dtype=np.int32)
    return grid


def _ogrid_boundary(quads, rim, n_node, S) -> dict[str, BoundaryGroup]:
    """wall (swept rim quads) / inlet (station-0 disk) / outlet (last disk)."""
    nq = len(quads)
    edge_quad = {}
    for qi, q in enumerate(quads):
        for k in range(4):
            edge_quad[frozenset((int(q[k]), int(q[(k + 1) % 4])))] = qi
    wall_cells, wall_faces = [], []
    for i in range(S - 1):
        b0, b1 = i * n_node, (i + 1) * n_node
        for a, b in rim:
            qi = edge_quad[frozenset((int(a), int(b)))]
            wall_cells.append(i * nq + qi)
            wall_faces.append([b0 + a, b0 + b, b1 + b, b1 + a])
    b_last = (S - 1) * n_node
    inlet_faces = quads[:, ::-1].copy()  # outward = -tangent
    outlet_faces = quads + b_last
    return {
        "wall": BoundaryGroup(
            "wall",
            cell_ids=np.asarray(wall_cells, dtype=np.int64),
            face_nodes=np.asarray(wall_faces, dtype=np.int64),
        ),
        "inlet": BoundaryGroup(
            "inlet",
            cell_ids=np.arange(nq, dtype=np.int64),
            face_nodes=np.asarray(inlet_faces, dtype=np.int64),
        ),
        "outlet": BoundaryGroup(
            "outlet",
            cell_ids=np.arange((S - 2) * nq, (S - 1) * nq, dtype=np.int64),
            face_nodes=np.asarray(outlet_faces, dtype=np.int64),
        ),
    }


# --- guarded smoothing (V21 do-no-harm, in 3-D) -----------------------------


def _boundary_nodes(quads, rim, n_node, S) -> np.ndarray:
    """Node ids on the lumen boundary: the swept rim + both end disks."""
    mask = np.zeros(S * n_node, dtype=bool)
    rim_nodes = np.unique(rim)
    for i in range(S):
        mask[i * n_node + rim_nodes] = True
    mask[:n_node] = True
    mask[(S - 1) * n_node:] = True
    return mask


def guarded_smooth(
    pts: np.ndarray,
    hexes: np.ndarray,
    boundary: np.ndarray,
    *,
    iterations: int,
) -> tuple[np.ndarray, int]:
    """Do-no-harm smoothing: per interior node, a Laplacian (edge-neighbour
    mean) proposal is accepted — full step, else half, else quarter — only
    if the min VTK scaled Jacobian over the node's incident hexes does not
    drop. The guard uses THE reported metric (a cheaper corner-Jacobian
    proxy let a real 0.22 -> 0.19 VTK regression through on a clinical scan, D75).
    Nodes are greedily coloured so no two simultaneously-moved nodes share
    a hex — within a colour class every node's guard depends only on its
    own position, so the class moves as one vectorized batch and each
    accepted move can only RAISE its local min: global sj_min and
    inverted-0 can never regress (the V21 guarantee, probed on the real
    arches). Returns (new points, accepted move count)."""
    interior = np.nonzero(~boundary)[0]
    if len(interior) == 0 or iterations <= 0:
        return pts, 0
    n = len(pts)
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7))
    nbr: list[set[int]] = [set() for _ in range(n)]
    node_hex: list[list[int]] = [[] for _ in range(n)]
    for hi, h in enumerate(hexes):
        for a, b in edges:
            nbr[h[a]].add(int(h[b]))
            nbr[h[b]].add(int(h[a]))
        for v in h:
            node_hex[int(v)].append(hi)
    # Greedy colouring of the shares-a-hex conflict graph.
    hex_nodes = [set(map(int, h)) for h in hexes]
    color: dict[int, int] = {}
    for v in interior:
        used = set()
        for hi in node_hex[v]:
            for w in hex_nodes[hi]:
                if w != v and w in color:
                    used.add(color[w])
        c = 0
        while c in used:
            c += 1
        color[int(v)] = c
    classes: dict[int, list[int]] = {}
    for v, c in color.items():
        classes.setdefault(c, []).append(v)
    # Per-class CSR: neighbour lists (for the Laplacian proposal) and
    # incident-hex lists (for the guard). Within a class no hex repeats.
    class_data = []
    for cls in classes.values():
        nodes = np.asarray(sorted(cls), dtype=np.int64)
        nb_lists = [np.fromiter(sorted(nbr[v]), dtype=np.int64) for v in nodes]
        nptr = np.concatenate([[0], np.cumsum([len(x) for x in nb_lists])])
        hs_lists = [np.asarray(node_hex[v], dtype=np.int64) for v in nodes]
        hptr = np.concatenate([[0], np.cumsum([len(x) for x in hs_lists])])
        class_data.append(
            (nodes, np.concatenate(nb_lists), nptr,
             np.concatenate(hs_lists), hptr)
        )
    pts = np.array(pts, dtype=float)
    h_scale = float(
        np.mean(np.linalg.norm(pts[hexes[:, 1]] - pts[hexes[:, 0]], axis=1))
    )
    # Guard evaluations run on THE reported metric: one scratch grid whose
    # points are swapped per evaluation (VTK recomputes quality on call).
    scratch = _hex_grid(pts, hexes)

    def vtk_sj(p: np.ndarray) -> np.ndarray:
        scratch.points = p
        sj = np.asarray(
            scratch.cell_quality(["scaled_jacobian"])["scaled_jacobian"],
            dtype=float,
        )
        return np.where(np.isfinite(sj), sj, -np.inf)

    accepted = 0
    for _ in range(iterations):
        moved = False
        for nodes, nb, nptr, hs, hptr in class_data:
            counts = np.diff(nptr)[:, None]
            prop = np.add.reduceat(pts[nb], nptr[:-1], axis=0) / counts
            old = pts[nodes].copy()
            before = np.minimum.reduceat(vtk_sj(pts)[hs], hptr[:-1])
            pending = np.ones(len(nodes), dtype=bool)
            best = old.copy()
            for t in (1.0, 0.5, 0.25):
                idx = np.nonzero(pending)[0]
                if len(idx) == 0:
                    break
                pts[nodes[idx]] = old[idx] + t * (prop[idx] - old[idx])
                after = np.minimum.reduceat(vtk_sj(pts)[hs], hptr[:-1])
                ok = pending & (after >= before - 1e-12)
                best[ok] = pts[nodes[ok]]
                pending &= ~ok
            pts[nodes] = best  # rejected nodes revert to their old position
            disp = np.linalg.norm(best - old, axis=1)
            real = disp > 1e-9 * h_scale
            accepted += int(real.sum())
            if real.any():
                moved = True
        if not moved:
            break
    return pts, accepted


# --- tet fallback on the same station-ring shell -----------------------------


def _ring_shell(stations, normals, binormals, radii, cos_t, sin_t):
    """Watertight labelled boundary shell of the lumen straight from the
    ray-cast rings (INDEPENDENT of the O-grid — a station whose section
    defeats the O-grid still has a valid star-shaped ring): lateral ring
    quads split to triangles (label 0 wall, outward radial), end caps as
    centre fans (labels 2 inlet / 3 outlet, outward ∓ tangent)."""
    S, T = radii.shape
    ring_pts = np.concatenate([
        stations[i]
        + (radii[i] * cos_t)[:, None] * normals[i][None, :]
        + (radii[i] * sin_t)[:, None] * binormals[i][None, :]
        for i in range(S)
    ])
    pts = np.vstack([ring_pts, stations[0], stations[S - 1]])
    c0, c1 = S * T, S * T + 1
    tris: list[tuple[int, int, int]] = []
    labels: list[int] = []
    for i in range(S - 1):
        b0, b1 = i * T, (i + 1) * T
        for j in range(T):
            j1 = (j + 1) % T
            tris.append((b0 + j, b0 + j1, b1 + j1))
            tris.append((b0 + j, b1 + j1, b1 + j))
            labels.extend((0, 0))
    b_last = (S - 1) * T
    for j in range(T):
        j1 = (j + 1) % T
        tris.append((c0, j1, j))  # inlet fan, outward -tangent
        labels.append(2)
        tris.append((c1, b_last + j, b_last + j1))  # outlet fan, outward +t
        labels.append(3)
    return pts, np.asarray(tris, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def _tet_fill(shell_pts, shell_tris, patch_labels, *, mesh_size, optimize=True, remesh=False):
    """gmsh-fill the labelled shell; classify boundary faces by nearest shell
    patch (wall/inlet/outlet labels).

    ``remesh``: reparametrize the shell (patches split at 40 deg creases) and
    remesh it at ``mesh_size`` before the fill. A marching-cubes surface
    carries needle triangles at its cap rims; filled as-is they become sliver
    tets (scaled Jacobian ~1e-4) on which simpleFoam diverged."""
    try:
        import gmsh
    except ImportError as exc:
        raise RuntimeError(
            "the lumen tet fallback needs gmsh — install with: "
            "pip install -e '.[meshing]' in the meshforge clone"
        ) from exc
    from scipy.spatial import cKDTree

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("meshforge_lumen")
        node_tags = np.arange(1, len(shell_pts) + 1)
        gmsh.model.addDiscreteEntity(2, 1)
        gmsh.model.mesh.addNodes(2, 1, node_tags, shell_pts.ravel())
        gmsh.model.mesh.addElementsByType(1, 2, [], (shell_tris + 1).ravel())
        if remesh:
            gmsh.model.mesh.classifySurfaces(math.radians(40.0), True, True, math.pi)
            gmsh.model.mesh.createGeometry()
            surfaces = [tag for _, tag in gmsh.model.getEntities(2)]
            loop = gmsh.model.geo.addSurfaceLoop(surfaces)
            gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5 * mesh_size)
        else:
            loop = gmsh.model.geo.addSurfaceLoop([1])
        gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        try:
            gmsh.model.mesh.generate(3)
        except Exception as gmsh_exc:  # gmsh raises a bare Exception
            # D85: fail loudly with a readable cause instead of a raw gmsh
            # traceback (seen on a real eccentric sac: the station rings of a
            # folding sweep overlap, so the ring shell self-intersects)
            raise RuntimeError(
                f"lumen tet fill failed — gmsh: {gmsh_exc}. The station-ring shell "
                f"is invalid, typically because the rings overlap where the "
                f"centreline bends tighter than the section radius"
            ) from None
        opt_note = "optimization disabled (optimize=False)"
        if optimize:
            try:
                gmsh.model.mesh.optimize("Netgen")
                opt_note = "Netgen mesh optimization applied"
            except Exception as netgen_exc:
                try:
                    gmsh.model.mesh.optimize("")
                    opt_note = (
                        f"Netgen optimization failed ({netgen_exc}); "
                        "default gmsh optimizer applied"
                    )
                except Exception as default_exc:
                    opt_note = (
                        f"mesh optimization skipped (Netgen: {netgen_exc}; "
                        f"default: {default_exc})"
                    )
        tags, coords, _ = gmsh.model.mesh.getNodes()
        points = np.asarray(coords, dtype=float).reshape(-1, 3)
        tag_to_idx = np.zeros(int(tags.max()) + 1, dtype=np.int64)
        tag_to_idx[np.asarray(tags, dtype=np.int64)] = np.arange(len(tags))
        _, tet_nodes = gmsh.model.mesh.getElementsByType(4)
        tets = tag_to_idx[np.asarray(tet_nodes, dtype=np.int64)].reshape(-1, 4)
    finally:
        gmsh.finalize()
    if len(tets) == 0:
        raise ValueError("gmsh produced no tetrahedra for the lumen shell")

    def build(tets):
        cells = np.column_stack(
            [np.full(len(tets), 4, dtype=np.int64), tets]
        ).ravel()
        ctypes = np.full(len(tets), pv.CellType.TETRA, dtype=np.uint8)
        g = pv.UnstructuredGrid(cells, ctypes, points)
        g.cell_data["region"] = np.ones(g.n_cells, dtype=np.int32)
        return g

    grid = build(tets)
    quality = volume_quality(grid)
    if quality.n_inverted:
        q = grid.cell_quality(["volume"])
        flip = np.asarray(q["volume"], dtype=float) <= 0
        tets[flip] = tets[flip][:, [0, 1, 3, 2]]
        grid = build(tets)
        quality = volume_quality(grid)
        quality.notes.append(
            f"reoriented {int(flip.sum())} negatively-wound tets from gmsh"
        )

    surf = grid.extract_surface(algorithm="dataset_surface")
    faces = surf.faces.reshape(-1, 4)[:, 1:]
    if remesh:
        # A remeshed cap's faces sit far from the ORIGINAL cap triangles'
        # centroids (big fan triangles), so nearest-centroid tagging handed
        # ~30% of each cap rim to the wall. Caps are planar: a face is a cap
        # face when all three of its nodes lie on that cap's plane.
        face_labels = np.zeros(len(faces), dtype=np.int64)
        tol = 1e-3 * float(mesh_size)
        for label in (2, 3):
            cap_nodes = np.unique(shell_tris[patch_labels == label])
            if len(cap_nodes) < 3:
                continue
            p = shell_pts[cap_nodes]
            c = p.mean(axis=0)
            n = np.linalg.svd(p - c, full_matrices=False)[2][-1]
            off = np.abs((surf.points[faces] - c) @ n)
            face_labels[off.max(axis=1) < tol] = label
    else:
        tree = cKDTree(shell_pts[shell_tris].mean(axis=1))
        centroids = surf.points[faces].mean(axis=1)
        _, nearest = tree.query(centroids)
        face_labels = patch_labels[nearest]
    orig_cell = np.asarray(surf.cell_data["vtkOriginalCellIds"], dtype=np.int64)
    orig_point = np.asarray(surf.point_data["vtkOriginalPointIds"], dtype=np.int64)
    face_nodes = orig_point[faces]
    boundary: dict[str, BoundaryGroup] = {}
    for label, name in _LABELS.items():
        sel = face_labels == label
        if not sel.any():
            continue
        boundary[name] = BoundaryGroup(
            name=name, cell_ids=orig_cell[sel], face_nodes=face_nodes[sel]
        )
    return grid, boundary, quality, opt_note


# --- public entry -------------------------------------------------------------


def fill_lumen(
    surface: pv.PolyData,
    *,
    method: str = "auto",
    n_theta: int = 32,
    n_axial: int = 40,
    core_fraction: float = 0.5,
    smooth: int = 4,
    mesh_size: float | None = None,
    centerline=None,
) -> WallMesh:
    """Fill the blood volume of a tubular surface (V47).

    ``method``: ``ogrid`` = structured swept O-grid hexes (loud failure where
    the geometry obstructs), ``tet`` = gmsh tetra fill of the station-ring
    shell, ``surface-tet`` = gmsh tetra fill of the input surface itself (no
    centreline rings; needs exactly two flat end caps), ``auto`` (default) =
    ogrid, falling back to tet with a loud fold-number note when the
    structured fill fails or sweeps inverted, and to surface-tet when the
    ring shell itself is invalid (rings overlapping on a tight bend).

    Each end the surface closes with a flat cap is anchored on that cap
    (D81), so the ``inlet``/``outlet`` patch is the cap itself; the notes
    give each patch's area against its cap's, or say loudly why an end kept
    the centreline-normal section.
    """
    if method not in ("auto", "ogrid", "tet", "surface-tet"):
        raise ValueError(
            f"lumen method must be auto | ogrid | tet | surface-tet, got {method!r}"
        )
    if n_theta < 8 or n_axial < 2:
        raise ValueError(
            f"lumen lattice too coarse: n_theta={n_theta}, n_axial={n_axial}"
        )
    if smooth < 0:
        raise ValueError(f"smooth iterations must be >= 0, got {smooth}")
    if mesh_size is not None and not mesh_size > 0:
        raise ValueError(f"mesh_size must be positive, got {mesh_size}")
    if centerline is None:
        from meshforge.geometry import centerline_from_surface, compute_centerline

        centerline = centerline_from_surface(surface)
        if centerline is None:
            centerline = compute_centerline(surface)

    stations, tangents, normals, binormals, radii, arc, cos_t, sin_t, notes = (
        _station_geometry(surface, centerline, n_theta=n_theta, n_axial=n_axial)
    )
    F = _fold_numbers(tangents, arc, radii)
    fold = _fold_note(F)

    def structured() -> WallMesh:
        disks, quads, rim, n_node = _station_disks(
            radii, cos_t, sin_t, core_fraction=core_fraction
        )
        pts, hexes = _sweep(stations, normals, binormals, disks, quads, n_node)
        grid = _hex_grid(pts, hexes)
        quality = volume_quality(grid)
        if quality.n_inverted:
            raise ValueError(
                f"structured lumen sweep has {quality.n_inverted} inverted "
                f"cell(s) ({fold}) — the section crosses the fold radius"
            )
        smooth_note = None
        if smooth > 0:
            bmask = _boundary_nodes(quads, rim, n_node, len(stations))
            new_pts, accepted = guarded_smooth(
                pts, hexes, bmask, iterations=smooth
            )
            if accepted:
                sj_before = quality.sj_min
                grid = _hex_grid(new_pts, hexes)
                quality = volume_quality(grid)
                smooth_note = (
                    f"guarded smoothing: {accepted} accepted moves in <= "
                    f"{smooth} sweep(s), sj_min {sj_before:.4g} -> "
                    f"{quality.sj_min:.4g} (do-no-harm guard)"
                )
                if quality.n_inverted:
                    raise ValueError(
                        "guarded smoothing produced inverted cells — the "
                        "guard metric disagrees with VTK on this mesh; "
                        "re-run with smooth=0 and report"
                    )
        boundary = _ogrid_boundary(quads, rim, n_node, len(stations))
        mesh_notes = [
            f"swept O-grid lumen: {len(stations)} stations x {len(quads)} "
            f"disk quads (pinned topology), n_theta {n_theta}",
            fold,
        ]
        if smooth_note:
            mesh_notes.append(smooth_note)
        mesh_notes.extend(notes)
        mesh_notes.extend(quality.notes)
        return WallMesh(
            grid=grid,
            boundary=boundary,
            backend="lumen-ogrid",
            quality=quality,
            notes=mesh_notes,
        )

    def tet(reason: str | None) -> WallMesh:
        shell_pts, shell_tris, labels = _ring_shell(
            stations, normals, binormals, radii, cos_t, sin_t
        )
        size = mesh_size
        if size is None:
            # Median ring edge — the boundary discretization scale. (A
            # median over ALL shell triangle edges mixes in the long axial
            # edges and produced 4x-coarser slivers on the real arches.)
            rings2d = np.stack([radii * cos_t, radii * sin_t], axis=2)
            ring_edges = np.linalg.norm(
                np.roll(rings2d, -1, axis=1) - rings2d, axis=2
            )
            size = float(np.median(ring_edges)) * 1.5
        grid, boundary, quality, opt_note = _tet_fill(
            shell_pts, shell_tris, labels, mesh_size=size
        )
        mesh_notes = [
            f"tet lumen fill on the station-ring shell ({len(shell_pts)} "
            f"shell nodes, mesh_size={size:.4g})",
            opt_note,
            fold,
        ]
        if reason:
            mesh_notes.append(reason)
        mesh_notes.extend(notes)
        mesh_notes.extend(quality.notes)
        return WallMesh(
            grid=grid,
            boundary=boundary,
            backend="lumen-tet",
            quality=quality,
            notes=mesh_notes,
        )

    def surface_tet(reason: str | None) -> WallMesh:
        # The input surface IS a valid closed shell even where the centreline
        # bends tighter than the vessel (the station rings then overlap and
        # the ring shell self-intersects): fill it directly. Its two flat end
        # caps become inlet (the cap at the centreline's start) and outlet.
        tri = surface.triangulate()
        faces = tri.faces.reshape(-1, 4)[:, 1:]
        caps = flat_cap_cells(tri)
        if len(caps) != 2:
            raise ValueError(
                f"surface-tet lumen fill needs exactly two flat end caps, found "
                f"{len(caps)} — crop the vessel with two planar caps first"
            )
        start = np.asarray(stations[0], dtype=float)
        cents = [tri.points[faces[c]].mean(axis=(0, 1)) for c in caps]
        order = np.argsort([np.linalg.norm(c - start) for c in cents])
        labels = np.zeros(len(faces), dtype=np.int64)
        labels[caps[order[0]]] = 2  # inlet
        labels[caps[order[1]]] = 3  # outlet
        edges = np.linalg.norm(
            tri.points[faces] - tri.points[np.roll(faces, -1, axis=1)], axis=2
        )
        size = mesh_size if mesh_size is not None else float(np.median(edges)) * 1.5
        grid, boundary, quality, opt_note = _tet_fill(
            np.asarray(tri.points, dtype=float), faces, labels, mesh_size=size,
            remesh=True,
        )
        mesh_notes = [
            f"tet lumen fill of the input surface itself, remeshed by gmsh at "
            f"mesh_size={size:.4g} ({tri.n_points} input surface nodes); inlet = "
            f"the flat cap nearest the centreline start",
            opt_note,
            fold,
        ]
        if reason:
            mesh_notes.append(reason)
        mesh_notes.extend(quality.notes)
        return WallMesh(
            grid=grid,
            boundary=boundary,
            backend="lumen-surface-tet",
            quality=quality,
            notes=mesh_notes,
        )

    if method == "ogrid":
        return structured()
    if method == "tet":
        return tet(None)
    if method == "surface-tet":
        return surface_tet(None)
    try:
        return structured()
    except ValueError as exc:
        # The station geometry itself is fine (rays hit) — the STRUCTURED
        # fill is what failed; the tet fallback meshes the same shell.
        reason = f"structured O-grid fill failed ({exc}); tet fallback ({fold})"
        try:
            return tet(reason)
        except RuntimeError as ring_exc:
            # The ring shell itself self-intersects (tight bend): fill the
            # input surface directly instead.
            return surface_tet(
                f"{reason}; the station-ring shell is invalid ({ring_exc}); "
                f"surface-tet fallback"
            )
