"""Parallel-plane wall lattice (D87): stations are planes between two flat caps.

The centreline lattice (``lattice.py``) casts its rays square to a
centreline; on a short, wide, eccentric segment (an aneurysm sac cropped
between two caps) the centreline bends tighter than the section radius and
the ray fans of neighbouring stations cross (fold number r·κ > 1), so the
wall folds however the thickness is capped. Here the stations are instead
planes perpendicular to the common normal of the two flat end caps, evenly
spaced from cap to cap. Planes never cross, which makes hex validity a 2-D
question: every hex between stations i and i+1 has four edges running from
plane i to plane i+1 and its other edges in those planes, so each corner
Jacobian is h × (an in-plane corner cross product) — positive iff that
in-plane corner is, whatever the shear between the stations. (VTK's scaled
Jacobian also tests the hex centre, which mixes the two stations' quads:
positive while consecutive stations' quads are not rotated against each
other, which is why the node correspondence avoids sliding.)

Per station: the plane's section of the surface (the cap's own rim at the
two ends) — one loop; small extra loops (bumps or dimples where the wall
runs nearly parallel to the plane) are dropped and counted, a large one
refuses. Station 0 is sampled at ``n_theta`` points evenly in arc length;
every later station takes the previous nodes' nearest points on its section
(the surface's meridians: nodes do not slide), relaxed toward even spacing.
No star-shape is required (a ray fan from the section centroid crossed the
real Anev_02 sac sections 1-3 times on 7 of 1920 rays). The wall layers are
offset IN-PLANE along the section's 2-D outward normal by t/cosα, α the
angle between the surface normal and the plane (measured from the sections
±t away), so the thickness along the surface normal stays t while every
node stays in its plane — the inlet/outlet faces lie exactly on the caps.

Counts (the gate's "capped" / "floored"):

* tilt-capped: α > ``max_tilt`` — the offset is held at t/cos(max_tilt), so
  the wall is thinner than t along the normal there;
* concavity-capped: the section turns tighter than the offset (the offset
  curve would cross itself; for an inward wall, a convex turn): the offset
  edge must keep (1 − curvature_cap) of its inner length — the discrete
  d·κ ≤ cap — and both its corners convex;
* floored: an offset raised back to ``floor_frac`` of t.

``section_smoothing`` (off by default) Gaussian-smooths each section along
its length before sampling, for surfaces rougher than the wall thickness
(the recovered Anev_02 sac carries mm-scale axial terraces); how far the
nodes then sit from the input surface is reported, never hidden.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyvista as pv

from meshforge.meshing.lattice import WallLattice

PARALLEL_TOL_DEG = 2.0  # the two caps' normals agree within this
MAX_TILT_DEG = 80.0  # default bound on α (surface normal vs station plane)
LOOP_DROP_MAX_FRAC = 0.02  # an extra section loop above this x main area refuses
SPACING_RELAX = 0.2  # per station, nodes move this far toward even spacing
MIN_GAP = 0.25  # ... or further while a gap is under this x the even one
SIDES = ("outward", "inward")


@dataclass
class CapPair:
    """The two flat end caps of a segment, ordered first → last.

    ``normal0``/``normal1`` are unit normals oriented from the first cap
    toward the last (into the segment at the first cap)."""

    first: object  # meshing.lumen._Cap
    last: object
    normal0: np.ndarray
    normal1: np.ndarray
    angle_deg: float
    separation: float


def parallel_caps(
    surface: pv.PolyData, *, tol_deg: float = PARALLEL_TOL_DEG, centerline=None
) -> CapPair:
    """The surface's two flat end caps, required parallel within ``tol_deg``.

    Caps are the D81 flat-cap regions (``lumen.flat_cap_cells``). The first cap
    is the one nearer the start of ``centerline`` (or the centreline embedded in
    the surface), else the one with the lower z centroid. Raises ``ValueError``
    naming the reason when the plane sweep does not apply."""
    from meshforge.meshing.lumen import _as_cap, _coplanar_regions

    found = _coplanar_regions(surface)
    if found is None:
        raise ValueError("the plane sweep needs an all-triangle surface")
    tri, pts, regions = found
    caps = [c for c in (_as_cap(tri, pts, *r) for r in regions) if c is not None]
    if len(caps) != 2:
        raise ValueError(
            f"the plane sweep needs exactly two flat end caps; the surface has {len(caps)}"
        )
    a, b = caps
    if centerline is None:
        from meshforge.geometry import centerline_from_surface

        centerline = centerline_from_surface(surface)
    if centerline is not None and not centerline.closed:
        p0 = np.asarray(centerline.points[0], dtype=float)
        if np.linalg.norm(b.centre - p0) < np.linalg.norm(a.centre - p0):
            a, b = b, a
    elif (b.centre[2], b.centre[0], b.centre[1]) < (a.centre[2], a.centre[0], a.centre[1]):
        a, b = b, a
    axis = b.centre - a.centre
    n0 = a.normal if float(a.normal @ axis) > 0 else -a.normal
    n1 = b.normal if float(b.normal @ axis) > 0 else -b.normal
    angle = math.degrees(math.acos(min(1.0, float(n0 @ n1))))
    if angle > tol_deg:
        raise ValueError(
            f"the end caps are {angle:.2f} deg from parallel (the plane sweep needs "
            f"<= {tol_deg:g}); crop with parallel caps or use the centreline sweep"
        )
    sep = 0.5 * (float(axis @ n0) + float(axis @ n1))
    diag = float(np.linalg.norm(np.ptp(pts, axis=0)))
    eps = 1e-6 * diag
    if sep <= 1e-3 * diag:
        raise ValueError(
            f"the end caps are {sep:.4g} apart along their normal — they do not face "
            f"each other across the segment"
        )
    below = float(((pts - a.centre) @ n0).min())
    above = float(((pts - b.centre) @ n1).max())
    if below < -eps or above > eps:
        raise ValueError(
            f"the surface reaches {max(-below, above):.4g} beyond its cap planes — not a "
            f"segment between two end caps"
        )
    return CapPair(first=a, last=b, normal0=n0, normal1=n1, angle_deg=angle, separation=sep)


def _plane_basis(normal: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e1 = seed - float(seed @ normal) * normal
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(normal, e1)


def _plane_loops(tri: np.ndarray, pts: np.ndarray, origin, normal) -> list[np.ndarray]:
    """Closed section loops (3-D, ordered) of a closed triangle surface with a
    plane. Vertices ON the plane count as above it (a symbolic shift of the
    plane), so every crossing edge is well defined."""
    d = (pts - origin) @ normal
    pos = d >= 0.0
    npos = pos[tri].sum(axis=1)
    cross = tri[(npos == 1) | (npos == 2)]
    if not len(cross):
        return []
    ea, eb = cross, np.roll(cross, -1, axis=1)
    straddle = pos[ea] != pos[eb]
    n_pts = len(pts)
    key = (np.minimum(ea, eb) * n_pts + np.maximum(ea, eb))[straddle].reshape(-1, 2)
    uniq, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(-1, 2)
    if not (np.bincount(inv.ravel(), minlength=len(uniq)) == 2).all():
        raise ValueError("the surface is not a closed manifold where a station plane cuts it")
    # each node has exactly two segments: its two neighbours along the loop
    flat = inv.ravel()
    other = inv[:, ::-1].ravel()
    order = np.argsort(flat, kind="stable")
    nbr = other[order].reshape(-1, 2)
    ua, ub = uniq // n_pts, uniq % n_pts
    w = d[ua] / (d[ua] - d[ub])
    xyz = pts[ua] + w[:, None] * (pts[ub] - pts[ua])
    seen = np.zeros(len(uniq), dtype=bool)
    loops = []
    for start in range(len(uniq)):
        if seen[start]:
            continue
        loop, prev, cur = [start], -1, start
        seen[start] = True
        while True:
            a, b = nbr[cur]
            nxt = a if a != prev else b
            if nxt == start:
                break
            loop.append(nxt)
            seen[nxt] = True
            prev, cur = cur, nxt
        loops.append(xyz[loop])
    return loops


def _area(uv: np.ndarray) -> float:
    x, y = uv[:, 0], uv[:, 1]
    return 0.5 * float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum())


def _dedupe(uv: np.ndarray, tol: float) -> np.ndarray:
    keep = np.linalg.norm(uv - np.roll(uv, 1, axis=0), axis=1) > tol
    return uv[keep] if keep.sum() >= 3 else uv


def _smooth_loop(uv: np.ndarray, sigma: float) -> np.ndarray:
    """The closed polygon Gaussian-smoothed along its arc length (std
    ``sigma``), as ``max(1024, 2·len)`` points evenly spaced in arc length."""
    seg = np.linalg.norm(np.roll(uv, -1, axis=0) - uv, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(cum[-1])
    M = max(1024, 2 * len(uv))
    pts = _resample(uv, cum, np.arange(M) * (L / M))
    gain = np.exp(-2.0 * (np.pi * np.fft.rfftfreq(M, d=L / M) * sigma) ** 2)
    return np.stack([np.fft.irfft(np.fft.rfft(pts[:, k]) * gain, n=M) for k in range(2)],
                    axis=-1)


def _resample(uv: np.ndarray, cum: np.ndarray, s: np.ndarray) -> np.ndarray:
    closed = np.vstack([uv, uv[:1]])
    s = np.mod(s, cum[-1])
    return np.stack([np.interp(s, cum, closed[:, k]) for k in range(2)], axis=-1)


def _closest_param(uv: np.ndarray, cum: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Arc-length parameter of the point of the closed polygon ``uv`` nearest
    each of ``q``."""
    a = uv
    e = np.roll(uv, -1, axis=0) - a
    ee = np.maximum((e * e).sum(axis=1), 1e-300)
    rel = q[:, None, :] - a[None, :, :]
    t = np.clip((rel * e[None]).sum(axis=-1) / ee[None], 0.0, 1.0)
    dist = np.linalg.norm(rel - t[..., None] * e[None], axis=-1)
    k = np.argmin(dist, axis=1)
    rows = np.arange(len(q))
    return cum[k] + t[rows, k] * np.sqrt(ee[k])


def _increasing_subsequence(x: np.ndarray) -> np.ndarray:
    """Indices of a longest strictly increasing subsequence of ``x``."""
    import bisect

    tails: list[float] = []
    tail_idx: list[int] = []
    parent = np.full(len(x), -1)
    for k, v in enumerate(x):
        pos = bisect.bisect_left(tails, v)
        if pos == len(tails):
            tails.append(v)
            tail_idx.append(k)
        else:
            tails[pos] = v
            tail_idx[pos] = k
        parent[k] = tail_idx[pos - 1] if pos else -1
    out, k = [], tail_idx[-1]
    while k >= 0:
        out.append(k)
        k = parent[k]
    return np.asarray(out[::-1], dtype=int)


def _march(uv: np.ndarray, cum: np.ndarray, prev: np.ndarray, offs: np.ndarray):
    """Arc-length parameters of this station's nodes: each previous node's
    nearest point on this section — the surface's meridians, so the nodes do
    not slide — then ``SPACING_RELAX`` of the way toward even spacing (further,
    up to fully even, while any gap is under ``MIN_GAP`` of the even one).
    Nearest points out of cyclic order (a node pulled across a notch) are
    dropped and re-spaced between their in-order neighbours; None when fewer
    than half keep the order."""
    L = float(cum[-1])
    N = len(prev)
    sig = _closest_param(uv, cum, prev)
    best = None
    for r in range(N):
        rel = np.mod(np.roll(sig, -r) - sig[r], L)  # rel[0] = 0, the minimum
        keep = np.concatenate([[0], 1 + _increasing_subsequence(rel[1:])])
        keep = keep[np.concatenate([[True], np.diff(rel[keep]) > 0])]
        if best is None or len(keep) > len(best[1]):
            best = (r, keep, rel)
        if len(keep) == N:
            break
    r, keep, rel = best
    if len(keep) < max(3, N // 2):
        return None
    if len(keep) < N:
        rel = np.interp(np.arange(N), np.append(keep, N), np.append(rel[keep], L))
    even = L / N
    target = float(np.mean(rel - offs)) + offs
    for lam in (SPACING_RELAX, 0.5, 1.0):
        new = rel + lam * (target - rel)
        gaps = np.diff(np.append(new, new[0] + L))
        if gaps.min() >= MIN_GAP * even:
            break
    return sig[r] + np.roll(new, r)


def _best_shift(uv: np.ndarray, cum: np.ndarray, prev: np.ndarray, offs: np.ndarray):
    """Even arc-length samples at the cyclic shift that moves them least
    from ``prev`` (fallback when nearest points lose the order)."""
    L = float(cum[-1])
    N = len(prev)

    def cost(cands: np.ndarray) -> np.ndarray:
        smp = _resample(uv, cum, cands[:, None] + offs[None, :])
        return ((smp - prev[None]) ** 2).sum(axis=(1, 2))

    cands = np.arange(8 * N) * (L / (8 * N))
    best = float(cands[int(np.argmin(cost(cands)))])
    width = L / (8 * N)
    for _ in range(3):
        fine = best + np.linspace(-width, width, 17)
        best = float(fine[int(np.argmin(cost(fine)))])
        width /= 8.0
    return best + offs


def _signed_distance(q: np.ndarray, loop: np.ndarray) -> np.ndarray:
    """Distance from points ``q`` to a closed polygon, negative inside."""
    a = loop
    e = np.roll(loop, -1, axis=0) - a
    ee = np.maximum((e * e).sum(axis=1), 1e-300)
    rel = q[:, None, :] - a[None, :, :]
    t = np.clip((rel * e[None]).sum(axis=-1) / ee[None], 0.0, 1.0)
    dist = np.linalg.norm(rel - t[..., None] * e[None], axis=-1).min(axis=1)
    # even-odd crossing count along +x
    y0, y1 = a[None, :, 1], (a + e)[None, :, 1]
    straddle = (y0 > q[:, None, 1]) != (y1 > q[:, None, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = a[None, :, 0] + (q[:, None, 1] - y0) * e[None, :, 0] / (y1 - y0)
    inside = (straddle & (xc > q[:, None, 0])).sum(axis=1) % 2 == 1
    return np.where(inside, -dist, dist)


def _simple(poly: np.ndarray) -> bool:
    """True when no two non-adjacent edges of the closed polygon cross."""
    n = len(poly)
    a, b = poly, np.roll(poly, -1, axis=0)

    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (
            q[..., 1] - p[..., 1]
        ) * (r[..., 0] - p[..., 0])

    i, j = np.triu_indices(n, k=2)
    keep = ~((i == 0) & (j == n - 1))
    i, j = i[keep], j[keep]
    d1 = orient(a[i], b[i], a[j])
    d2 = orient(a[i], b[i], b[j])
    d3 = orient(a[j], b[j], a[i])
    d4 = orient(a[j], b[j], b[i])
    return not bool(((d1 * d2 < 0) & (d3 * d4 < 0)).any())


def _cross2(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def build_plane_lattice(
    surface: pv.PolyData,
    thickness,
    *,
    centerline=None,
    n_theta: int = 32,
    n_axial: int = 60,
    n_layers: int = 4,
    layer_ratio: float = 1.0,
    curvature_cap: float = 0.5,
    floor_frac: float = 0.2,
    side: str = "outward",
    max_tilt: float = MAX_TILT_DEG,
    section_smoothing: float = 0.0,
    caps: CapPair | None = None,
) -> WallLattice:
    """Wall lattice on stations = planes parallel to the two flat end caps.

    ``side`` "outward" grows the wall out of the surface (the surface is the
    lumen side, as the centreline sweep does); "inward" grows it into the
    surface (the surface is the OUTER wall — e.g. a recovered aneurysm sac,
    whose boundary is the fat plane). ``section_smoothing`` > 0: std (surface
    units) of a Gaussian smoothing of each section along its length before it
    is sampled. ``centerline`` only orders the caps (station 0 = the cap at
    its start)."""
    from meshforge.geometry.thickness import ThicknessSpec

    if isinstance(thickness, ThicknessSpec):
        raise ValueError(
            "the plane sweep takes a scalar thickness (a measured outer surface or a "
            "thickness field needs the centreline sweep)"
        )
    t_nom = float(thickness)
    if not t_nom > 0:
        raise ValueError(f"thickness must be positive, got {thickness}")
    if n_theta < 8 or n_axial < 2 or n_layers < 1:
        raise ValueError(
            f"lattice too coarse: n_theta={n_theta}, n_axial={n_axial}, n_layers={n_layers}"
        )
    if layer_ratio <= 0:
        raise ValueError(f"layer_ratio must be positive, got {layer_ratio}")
    if not 0.0 < floor_frac <= 1.0:
        raise ValueError(f"floor_frac must be in (0,1], got {floor_frac}")
    if not 0.0 < curvature_cap < 1.0:
        raise ValueError(
            f"the plane sweep needs 0 < curvature_cap < 1 (the offset edge keeps "
            f"1 - cap of its length), got {curvature_cap}"
        )
    if not 0.0 < max_tilt < 90.0:
        raise ValueError(f"max_tilt must be in (0, 90) degrees, got {max_tilt}")
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    if not (math.isfinite(section_smoothing) and section_smoothing >= 0.0):
        raise ValueError(f"section_smoothing must be >= 0, got {section_smoothing}")
    if caps is None:
        caps = parallel_caps(surface, centerline=centerline)

    tri = np.asarray(surface.faces).reshape(-1, 4)[:, 1:].astype(np.int64)
    pts = np.asarray(surface.points, dtype=float)
    S, N = int(n_axial), int(n_theta)
    u = np.linspace(0.0, 1.0, S)
    n0, n1 = caps.normal0, caps.normal1
    normals = (1.0 - u)[:, None] * n0[None] + u[:, None] * n1[None]
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    origins = (1.0 - u)[:, None] * caps.first.centre[None] + u[:, None] * caps.last.centre[None]
    seed = np.eye(3)[int(np.argmin(np.abs(n0)))]
    bases = [_plane_basis(normals[i], seed) for i in range(S)]
    spacing = caps.separation / (S - 1)
    tol = 1e-9 * max(caps.separation, 1.0)

    def to2d(i: int, xyz: np.ndarray) -> np.ndarray:
        e1, e2 = bases[i]
        rel = xyz - origins[i]
        return np.column_stack([rel @ e1, rel @ e2])

    def to3d(i: int, uv: np.ndarray) -> np.ndarray:
        e1, e2 = bases[i]
        return origins[i] + uv[..., :1] * e1 + uv[..., 1:2] * e2

    # ---- sections: one loop per station, CCW in (e1, e2) ----------------
    dense: list[np.ndarray] = []  # 3-D loop per station (smoothed if asked)
    dropped, drop_frac, area_change = 0, 0.0, 0.0
    for i in range(S):
        if i in (0, S - 1):
            loops3 = [(caps.first if i == 0 else caps.last).loop]
        else:
            loops3 = _plane_loops(tri, pts, origins[i], normals[i])
            if not loops3:
                raise ValueError(f"station {i}: its plane misses the surface")
        uvs = [to2d(i, lp) for lp in loops3]
        areas = np.abs([_area(x) for x in uvs])
        k = int(np.argmax(areas))
        if len(uvs) > 1:
            others = np.delete(areas, k) / areas[k]
            dropped += len(others)
            drop_frac = max(drop_frac, float(others.max()))
            if others.max() > LOOP_DROP_MAX_FRAC:
                raise ValueError(
                    f"station {i}: the plane meets the surface in {len(uvs)} loops (a second "
                    f"piece of {others.max():.1%} of the section) — the segment is not one "
                    f"tube along the cap normal; use the centreline sweep"
                )
        uv = _dedupe(uvs[k], tol)
        if _area(uv) < 0:
            uv = uv[::-1]
        if section_smoothing > 0:
            smooth = _smooth_loop(uv, section_smoothing)
            area_change = max(area_change, abs(_area(smooth) / _area(uv) - 1.0))
            uv = smooth
        dense.append(to3d(i, uv))

    # ---- samples: station 0 evenly in arc length; each later station's
    # nodes are the previous nodes' nearest points on its section (the
    # surface's meridians, no sliding), relaxed toward even spacing --------
    P2 = np.empty((S, N, 2))
    centroid = np.empty((S, 2))
    marched_fallback = 0
    for i in range(S):
        uv = to2d(i, dense[i])
        seg = np.linalg.norm(np.roll(uv, -1, axis=0) - uv, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        L = float(cum[-1])
        step = L / N
        offs = np.arange(N) * step
        if i == 0:
            sig = float(cum[int(np.argmax(uv[:, 0]))]) + offs
        else:
            prev = to2d(i, to3d(i - 1, P2[i - 1]))
            sig = _march(uv, cum, prev, offs)
            if sig is None:
                marched_fallback += 1
                sig = _best_shift(uv, cum, prev, offs)
        P2[i] = _resample(uv, cum, sig)
        x, y = uv[:, 0], uv[:, 1]
        cr = x * np.roll(y, -1) - np.roll(x, -1) * y
        a6 = 3.0 * cr.sum()
        centroid[i] = [((x + np.roll(x, -1)) * cr).sum() / a6, ((y + np.roll(y, -1)) * cr).sum() / a6]
        if _area(P2[i]) <= 0 or not _simple(P2[i]):
            raise ValueError(
                f"station {i}: the {N} samples of its section cross each other (a fold "
                f"or slot narrower than the sample spacing {step:.3g}); raise n_theta or "
                f"use the centreline sweep"
            )

    # ---- in-plane outward normals and the surface tilt α ----------------
    # vertex normal = bisector of the two adjacent unit edge normals: on the
    # outward side of BOTH edges whatever their lengths (a chord normal is
    # not, at a sharp sampled turn), so the inner ring itself is always valid
    edge = np.roll(P2, -1, axis=1) - P2
    nu = np.stack([edge[..., 1], -edge[..., 0]], axis=-1)
    nu /= np.linalg.norm(nu, axis=-1)[..., None]
    m = nu + np.roll(nu, 1, axis=1)
    m /= np.linalg.norm(m, axis=-1)[..., None]
    k_win = max(1, int(round(t_nom / spacing)))
    slope = np.empty((S, N))  # outward motion of the section per unit height
    for i in range(S):
        lo, hi = max(0, i - k_win), min(S - 1, i + k_win)
        q3 = to3d(i, P2[i])
        sd_lo = 0.0 if lo == i else _signed_distance(to2d(i, q3), to2d(i, dense[lo]))
        sd_hi = 0.0 if hi == i else _signed_distance(to2d(i, q3), to2d(i, dense[hi]))
        dh = float((origins[hi] - origins[lo]) @ normals[i])
        slope[i] = (sd_lo - sd_hi) / dh
    sec = np.sqrt(1.0 + slope**2)  # 1 / cos α
    tilt = np.degrees(np.arctan(np.abs(slope)))
    sec_max = 1.0 / math.cos(math.radians(max_tilt))
    tilt_capped = sec > sec_max
    d = t_nom * np.minimum(sec, sec_max)

    # ---- in-plane offset validity (concavity cap), then the floor -------
    sgn = 1.0 if side == "outward" else -1.0
    conc_capped = np.zeros((S, N), dtype=bool)
    unresolved = 0
    for i in range(S):
        P, mi, di = P2[i], m[i], d[i]
        e0 = np.roll(P, -1, axis=0) - P
        l0 = np.linalg.norm(e0, axis=1)
        e0h = e0 / l0[:, None]
        m1 = np.roll(mi, -1, axis=0)
        for _ in range(400):
            Q = P + sgn * di[:, None] * mi
            e1 = np.roll(Q, -1, axis=0) - Q
            # each layer quad keeps (1 - cap) of its inner edge, and both of its
            # corners on the offset edge stay convex (layers always step +m)
            bad = (
                ((e1 * e0h).sum(axis=1) < (1.0 - curvature_cap) * l0)
                | (_cross2(e1, mi) >= 0.0)
                | (_cross2(e1, m1) >= 0.0)
            )
            if not bad.any():
                break
            idx = np.nonzero(bad)[0]
            idx = np.unique(np.concatenate([idx, (idx + 1) % N]))
            di[idx] *= 0.9
            conc_capped[i, idx] = True
        else:
            unresolved += 1
        d[i] = di
    sec_eff = np.minimum(sec, sec_max)
    floor = floor_frac * t_nom * sec_eff
    floored = d < floor - 1e-12 * t_nom
    d = np.maximum(d, floor)

    n_tilt = int(tilt_capped.sum())
    n_conc = int((conc_capped & ~tilt_capped).sum())
    n_capped = int((tilt_capped | conc_capped).sum())
    n_floored = int(floored.sum())
    t_field = d / sec  # first-order thickness along the surface normal

    k = np.arange(n_layers + 1, dtype=float)
    if abs(layer_ratio - 1.0) < 1e-12:
        fracs = k / n_layers
    else:
        fracs = (layer_ratio**k - 1.0) / (layer_ratio**n_layers - 1.0)
    if side == "inward":
        fracs = fracs - 1.0  # layer n_layers is the surface itself
    nodes2 = P2[:, :, None, :] + (d[:, :, None, None] * fracs[None, None, :, None]) * m[:, :, None, :]
    nodes = np.empty((S, N, n_layers + 1, 3))
    for i in range(S):
        nodes[i] = to3d(i, nodes2[i])
    inner_radii = np.linalg.norm(P2 - centroid[:, None, :], axis=-1)
    # how far the wall's surface-side nodes sit from the input surface (0 up to
    # round-off unless the sections were smoothed)
    on_surface = nodes[:, :, 0 if side == "outward" else n_layers, :].reshape(-1, 3)
    offset = np.abs(np.asarray(
        pv.PolyData(on_surface).compute_implicit_distance(surface)["implicit_distance"]
    ))

    notes = [
        f"plane sweep: {S} stations square to the cap normal (caps "
        f"{caps.angle_deg:.3g} deg from parallel, {caps.separation:.4g} apart), "
        f"{N} arc-length samples per section, wall {side} of the surface",
        f"surface tilt to the station planes: max {float(tilt.max()):.1f} deg, "
        f"p95 {float(np.percentile(tilt, 95)):.1f} (in-plane offset t/cos(tilt), "
        f"bound {max_tilt:g} deg, normal from sections +-{k_win * spacing:.3g} apart)",
    ]
    if section_smoothing > 0:
        notes.append(
            f"sections smoothed (Gaussian std {section_smoothing:g} along their length): "
            f"the wall's surface-side nodes sit up to {float(offset.max()):.3g} (p95 "
            f"{float(np.percentile(offset, 95)):.3g}) from the input surface; section "
            f"areas changed by up to {area_change:.1%}"
        )
    if marched_fallback:
        notes.append(
            f"{marched_fallback} station(s) sampled evenly at the least-moving shift "
            f"(most nearest points to the previous nodes lost the cyclic order)"
        )
    if dropped:
        notes.append(
            f"{dropped} small extra section loop(s) dropped (largest {drop_frac:.2%} of "
            f"its station's section): bumps/dimples finer than the samples"
        )
    if n_tilt:
        notes.append(
            f"tilt cap: {n_tilt} node(s) steeper than {max_tilt:g} deg — the wall there "
            f"is thinner than {t_nom:g} along the normal"
        )
    if n_conc:
        notes.append(
            f"concavity cap: {n_conc} node(s) where the {side} offset would cross "
            f"itself (the section turns tighter than the offset; curvature_cap "
            f"{curvature_cap:g})"
        )
    if n_floored:
        notes.append(f"floor raised {n_floored} node(s) to {floor_frac:g} x the thickness")
    if unresolved:
        notes.append(f"{unresolved} station(s) kept an offset fold the cap could not remove")
    info = {
        "side": side,
        "n_capped_tilt": n_tilt,
        "n_capped_concave": n_conc,
        "tilt_max_deg": round(float(tilt.max()), 3),
        "tilt_p95_deg": round(float(np.percentile(tilt, 95)), 3),
        "max_tilt_deg": float(max_tilt),
        "cap_angle_deg": round(float(caps.angle_deg), 6),
        "dropped_loops": int(dropped),
        "dropped_area_frac_max": round(drop_frac, 6),
        "even_fallback_stations": int(marched_fallback),
        "section_smoothing": float(section_smoothing),
        "surface_offset_max": round(float(offset.max()), 4),
        "surface_offset_p95": round(float(np.percentile(offset, 95)), 4),
        "section_area_change_max": round(area_change, 5),
    }
    return WallLattice(
        nodes=nodes,
        inner_radii=inner_radii,
        thickness=t_nom,
        thickness_field=t_field,
        n_capped=n_capped,
        n_floored=n_floored,
        notes=notes,
        closed_s=False,
        sweep="planes",
        info=info,
    )
