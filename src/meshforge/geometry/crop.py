"""Crop an aorta surface to its infrarenal aneurysm segment (M2, D85).

Input: a closed aorta surface in a patient frame whose +z is superior (RAS
or LPS millimetres — what ``segment`` writes), e.g. TotalSegmentator's
``aorta`` label from the bifurcation up to the arch. Output: the segment
between two cut planes, each square to the local centreline and closed by an
exactly planar cap, plus a landmark report.

Landmarks come from the surface's own centreline (``aaa_centerline``: the
skeleton centreline smoothed to at least ``CL_SMOOTHING_MM`` and run out to
the surface at both ends) and the equivalent-diameter profile d(h) of its
centreline-normal sections (``compute_sections``), h = arc length above the
caudal end, median-smoothed over ``SMOOTH_WINDOW_MM``:

* **bifurcation** — the caudal end of the label (where the aorta label ends
  or splits): h = 0.
* **flare top** — walking up from h = 0, past the end zone's rise (a
  rounded label end, or a section grazing a flat end cap), the first local
  minimum of d(h): where the bifurcation flare stops narrowing. 0 when d
  rises straight into the sac (a sac that reaches the end of the label).
* **sac** — the maximum of d(h) between the flare top and
  ``SAC_SEARCH_MM`` above the bifurcation.
* **neck** — the first stretch above the sac where d(h) <= T for at least
  ``NECK_MIN_LENGTH_MM``, with T = max(``NECK_RATIO`` x d_ref, d_ref +
  ``NECK_FLOOR_MM``) and d_ref = the narrowest d(h) within
  ``NECK_SEARCH_MM`` above the sac (the proximal reference). It ends where
  d(h) first exceeds T again (or at the end of the profile: "open-ended").
  Renal arteries are not part of the aorta label, so the neck length is the
  non-dilated length above the sac, an upper bound of the infrarenal neck.

Refusals (``CropError``, the measured profile in the message): no
non-dilated stretch of ``NECK_MIN_LENGTH_MM`` above the sac (**no neck**),
or a widest section less than ``SAC_MIN_RATIO`` x the neck (**no sac**).
Sections that run through a flat end cap the input already has (a
pre-trimmed surface) are dropped from the profile, not read as a narrowing.

Cut planes: distal = flare top + ``distal_margin_mm``; proximal = neck start
+ ``neck_extension_mm`` (the flow/clamp extension; held inside the neck).
Each is square to the local vessel axis — the line through the section
centres within +-``PLANE_FIT_MM`` when they are straight, else (a dome, an
eccentric sac, a kink) the segment axis, the chord between the two cut
points; if two locally squared planes would cross inside the vessel, both
take the chord. A cut that opens more than one loop (a stub, strut or
branch), meets the wall under ``CAP_MIN_CREASE_DEG`` (the lumen's cap
anchoring, D81, needs >= 10 deg) or runs through the input's own end cap
moves toward the sac in ``PLANE_SHIFT_MM`` steps, at most
``PLANE_SHIFT_MAX`` times, then refuses. Each opening gets an exactly planar
cap (rim snapped onto the plane; a fan from the centroid when the rim is
star-shaped about it, else ear clipping into strictly positive triangles).

Every cropped surface carries a segment centreline (``embed_centerline``)
that ``volmesh`` and ``lumen`` pick up: both ends leave the cap's inscribed
centre along the cap normal, and between them the least-bent of a cubic
cap-to-cap family and the skeleton that stays inside (fold number max(r x
kappa) from ray-cast ring spokes) — the radial wall lattice folds, and its
curvature cap floors the wall, where a wandering skeleton bends through a
wide sac (D85). A second surface (the blood lumen, or later an outer sac)
is cut by the SAME planes (``lumen=``, or ``planes_from_report`` +
``crop_with_planes``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyvista as pv

# --- landmark rules (D85) -----------------------------------------------------
PROFILE_STEP_MM = 1.0  # spacing of the centreline-normal sections
SMOOTH_WINDOW_MM = 5.0  # running-median window of the diameter profile
FLARE_SEARCH_MM = 30.0  # a "flare" falling for longer than this is not a flare
SAC_SEARCH_MM = 200.0  # the sac lies within this height above the bifurcation
NECK_SEARCH_MM = 150.0  # proximal reference sought this far above the sac
NECK_RATIO = 1.2  # non-dilated: d <= 1.2 x the proximal reference ...
NECK_FLOOR_MM = 2.0  # ... and never a margin under 2 mm (segmentation noise)
NECK_MIN_LENGTH_MM = 10.0  # a neck is non-dilated for at least this long
SAC_MIN_RATIO = 1.5  # an aneurysm sac is at least 1.5 x its neck
# --- centreline / cut constants ---------------------------------------------
CL_SMOOTHING_MM = 2.0  # minimum centreline smoothing (RMS path deviation)
CL_SPACING_MM = 0.5  # target centreline sample spacing
PLANE_FIT_MM = 8.0  # cut plane squared to the section centres within +-8 mm ...
PLANE_FIT_RMS_MM = 0.5  # ... when they lie this close to a straight axis ...
PLANE_FIT_MAX_DEG = 30.0  # ... that leans at most this far off the skeleton
END_CLEARANCE_MM = 2.0  # a cut stays this far inside the profile's ends
PLANE_SHIFT_MM = 2.0  # a cut that opens >1 loop moves this far toward the sac ...
PLANE_SHIFT_MAX = 10  # ... at most this many times
CAP_MIN_CREASE_DEG = 10.5  # an auto cut must meet the wall at least this steeply (D81: 10)
CAP_STANDOFF = 0.02  # embedded centreline stops this x radius short of a cap
SEGMENT_FOLD_MAX = 0.9  # a cap-normal segment centreline must keep r x kappa below this
ON_PLANE_TOL_MM = 1e-6  # a clipped rim vertex lies on its plane within this
DEFAULT_NECK_EXTENSION_MM = 25.0
DEFAULT_DISTAL_MARGIN_MM = 10.0


class CropError(ValueError):
    """A loud refusal: no sac, no neck, or a cut that cannot be capped flat."""


class _PlanesCross(CropError):
    """The two cut planes intersect inside the kept vessel segment."""


class _MovePlane(CropError):
    """A cut the caller may retry with the named plane moved toward the sac:
    it opens more than one loop (a stub, strut or branch is cut too), or it
    meets the wall at so shallow an angle that the lumen's cap anchoring
    (D81, creases >= 10 deg) would not recognise the cap."""

    def __init__(self, message: str, plane: str):
        super().__init__(message)
        self.plane = plane


# ---------------------------------------------------------------------------
# centreline + profile
# ---------------------------------------------------------------------------
def aaa_centerline(surface: pv.PolyData, *, smoothing_mm: float = CL_SMOOTHING_MM):
    """The landmark centreline: caudal end first (lower z), smoothed to at
    least ``smoothing_mm`` RMS, sampled every ~``CL_SPACING_MM``, and run out
    along the end tangent to the surface at both ends (any face counts, so a
    label that ends in a rounded dome or a sac is measured to its tip)."""
    from meshforge.geometry.centerline import Centerline, compute_centerline

    b = surface.bounds
    diag = math.dist(b[::2], b[1::2])
    if not diag > 0:
        raise CropError("the surface is empty or degenerate (zero extent)")
    voxel = diag / 150.0
    smoothing = max(voxel, float(smoothing_mm))
    n = int(np.clip(math.ceil(1.5 * diag / CL_SPACING_MM) + 1, 200, 6000))
    cl = compute_centerline(
        surface, voxel_size=voxel, smoothing=smoothing, n_samples=n,
        cap_normal_tol=0.0,
    )
    if cl.closed:
        raise CropError("the surface's centreline is a closed loop — not an aorta segment")
    dz = float(cl.points[-1, 2] - cl.points[0, 2])
    if abs(dz) < 0.1 * cl.length:
        raise CropError(
            f"cannot tell the caudal end: the centreline's two ends differ by only "
            f"{abs(dz):.3g} mm in z over {cl.length:.4g} mm (expects a patient "
            f"frame with +z superior, RAS or LPS)"
        )
    notes = list(cl.notes) + [
        f"landmark centreline: voxel {voxel:.4g} mm, smoothing {smoothing:.4g} mm "
        f"RMS, {n} samples, both ends run out to the surface"
    ]
    if dz < 0:  # orient caudal -> cranial
        cl = Centerline(
            points=cl.points[::-1].copy(),
            tangents=-cl.tangents[::-1].copy(),
            arc_lengths=cl.length - cl.arc_lengths[::-1],
            radii=cl.radii[::-1].copy(),
            length=cl.length,
            closed=False,
            voxel_size=cl.voxel_size,
            notes=notes,
        )
    else:
        cl.notes = notes
    return cl


@dataclass
class DiameterProfile:
    """Equivalent-diameter profile along the landmark centreline."""

    h: np.ndarray  # mm above the caudal end
    d_equiv: np.ndarray  # 2 sqrt(A / pi) of the centreline-normal section
    d_max: np.ndarray  # largest opposite-spoke diameter of the section
    d_smooth: np.ndarray  # running median of d_equiv over SMOOTH_WINDOW_MM
    index: np.ndarray  # centreline sample of each station
    centers: np.ndarray  # (K, 3) centre of each section loop

    def text(self, every_mm: float = 10.0) -> str:
        """Compact 'h:d' listing of the smoothed profile (for messages)."""
        out, nxt = [], -1e9
        for h, d in zip(self.h, self.d_smooth):
            if h >= nxt:
                out.append(f"{h:.0f}:{d:.1f}")
                nxt = h + every_mm
        return "h mm:d mm " + " ".join(out)


def diameter_profile(surface: pv.PolyData, cl, *, h_max: float | None = None) -> DiameterProfile:
    """Sections every ~``PROFILE_STEP_MM`` up to ``h_max`` (default: all)."""
    from meshforge.geometry.sections import compute_sections

    from meshforge.meshing.lumen import flat_cap_cells

    ds = cl.length / max(len(cl.points) - 1, 1)
    step = max(1, int(round(PROFILE_STEP_MM / ds)))
    top = cl.length if h_max is None else min(cl.length, h_max)
    stations = [i for i in range(0, len(cl.points), step) if cl.arc_lengths[i] <= top]
    # a section that runs through a flat end cap the input already has (a
    # pre-trimmed surface with a tilted cap) is cut short by it: not a
    # vessel section — drop it rather than let it fake a narrow neck
    tri = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
    pts = np.asarray(surface.points)
    caps = []
    for cells in flat_cap_cells(surface):
        v = pts[np.unique(tri[cells])]
        c = v.mean(axis=0)
        caps.append((v, c, float(np.linalg.norm(v - c, axis=1).max())))

    def grazes(i: int) -> bool:
        o, t = cl.points[i], cl.tangents[i]
        for v, c, r in caps:
            if np.linalg.norm(c - o) <= 2.0 * r:
                d = (v - o) @ t
                if d.min() < 0.0 < d.max():
                    return True
        return False

    stations = [i for i in stations if not grazes(i)]
    if not stations:
        raise CropError("every centreline station's section runs through an end cap")
    secs = compute_sections(surface, cl, stations=stations)
    h = np.array([s.s for s in secs])
    d_eq = np.array([s.d_equiv for s in secs])
    d_mx = np.array([s.d_max for s in secs])
    half = 0.5 * SMOOTH_WINDOW_MM
    d_s = np.array([np.median(d_eq[np.abs(h - hi) <= half + 1e-9]) for hi in h])
    return DiameterProfile(
        h=h, d_equiv=d_eq, d_max=d_mx, d_smooth=d_s,
        index=np.array([s.index for s in secs]),
        centers=np.array([s.center for s in secs]),
    )


# ---------------------------------------------------------------------------
# landmarks
# ---------------------------------------------------------------------------
@dataclass
class Landmarks:
    h_flare_top: float
    h_sac: float
    d_sac: float  # smoothed equivalent diameter at the sac
    d_sac_max: float  # largest raw opposite-spoke diameter within the sac window
    h_neck_start: float
    h_neck_end: float
    neck_open_ended: bool  # the neck runs to the end of the measured profile
    d_neck: float  # median smoothed d over the first NECK_MIN_LENGTH_MM of the neck
    d_ref: float  # proximal reference: narrowest d within NECK_SEARCH_MM above the sac
    threshold: float  # T: non-dilated means d <= T
    notes: list[str] = field(default_factory=list)

    @property
    def neck_length(self) -> float:
        return self.h_neck_end - self.h_neck_start

    @property
    def sac_neck_ratio(self) -> float:
        return self.d_sac / self.d_neck


def find_landmarks(p: DiameterProfile) -> Landmarks:
    """Apply the D85 landmark rules to a profile; ``CropError`` when there is
    no neck or no sac (the message carries the measured profile)."""
    h, d = p.h, p.d_smooth
    n = len(h)
    if n < 5 or h[-1] - h[0] < 3.0 * NECK_MIN_LENGTH_MM:
        raise CropError(
            f"the diameter profile is too short to find landmarks ({n} sections over "
            f"{(h[-1] - h[0]) if n else 0:.3g} mm); {p.text()}"
        )
    notes: list[str] = []
    # the end zone first rises (a rounded label end, or a section grazing a
    # flat end cap); the flare then falls to its top, the first local minimum
    i_pk = 0
    while i_pk + 1 < n and d[i_pk + 1] > d[i_pk]:
        i_pk += 1
    i_f = i_pk
    while i_f + 1 < n and d[i_f + 1] <= d[i_f]:  # across median plateaus too
        i_f += 1
    if h[i_pk] - h[0] > FLARE_SEARCH_MM:
        i_f = 0  # the end rises straight into the sac: no flare
    elif h[i_f] - h[i_pk] > FLARE_SEARCH_MM:
        notes.append(
            f"the profile falls for {h[i_f] - h[i_pk]:.0f} mm above the caudal end — "
            f"not a bifurcation flare; the sac may reach the end of the label (flare "
            f"top = 0)"
        )
        i_f = 0
    win = np.flatnonzero((h >= h[i_f]) & (h <= h[0] + SAC_SEARCH_MM))
    i_s = int(win[np.argmax(d[win])])
    above = np.flatnonzero((h > h[i_s]) & (h <= h[i_s] + NECK_SEARCH_MM))
    if len(above) == 0 or h[above[-1]] - h[i_s] < NECK_MIN_LENGTH_MM:
        reach = (h[above[-1]] - h[i_s]) if len(above) else 0.0
        raise CropError(
            f"no neck: the widest section (d {d[i_s]:.1f} mm at h {h[i_s]:.0f} mm) is "
            f"only {reach:.1f} mm below the end of the surface — it ends inside the "
            f"aneurysm, with no non-dilated stretch of {NECK_MIN_LENGTH_MM:g} mm above "
            f"the sac; {p.text()}"
        )
    d_ref = float(d[above].min())
    thr = max(NECK_RATIO * d_ref, d_ref + NECK_FLOOR_MM)
    start = None
    for k in above:
        if d[k] > thr:
            continue
        stop = h[k] + NECK_MIN_LENGTH_MM
        if h[-1] < stop - 1e-9:
            break  # the profile ends before the stretch could be sustained
        run = (h >= h[k]) & (h <= stop + 1e-9)
        if (d[run] <= thr).all():
            start = int(k)
            break
    if start is None:
        raise CropError(
            f"no neck: above the widest section (d {d[i_s]:.1f} mm at h {h[i_s]:.0f} mm) "
            f"no stretch of {NECK_MIN_LENGTH_MM:g} mm stays non-dilated (d <= "
            f"{thr:.1f} mm = max({NECK_RATIO:g} x {d_ref:.1f}, {d_ref:.1f} + "
            f"{NECK_FLOOR_MM:g})); {p.text()}"
        )
    j = start
    while j + 1 < n and d[j + 1] <= thr:
        j += 1
    run = (h >= h[start]) & (h <= h[start] + NECK_MIN_LENGTH_MM + 1e-9)
    d_neck = float(np.median(d[run]))
    ratio = float(d[i_s]) / d_neck
    if ratio < SAC_MIN_RATIO:
        raise CropError(
            f"no sac: the widest section (d {d[i_s]:.1f} mm at h {h[i_s]:.0f} mm) is only "
            f"{ratio:.2f} x the non-dilated stretch above it (d {d_neck:.1f} mm from "
            f"h {h[start]:.0f} mm); an aneurysm sac is at least {SAC_MIN_RATIO:g} x its "
            f"neck; {p.text()}"
        )
    sac_zone = np.abs(h - h[i_s]) <= max(SMOOTH_WINDOW_MM, 0.5 * (h[start] - h[i_s]))
    return Landmarks(
        h_flare_top=float(h[i_f]),
        h_sac=float(h[i_s]),
        d_sac=float(d[i_s]),
        d_sac_max=float(np.nanmax(p.d_max[sac_zone])),
        h_neck_start=float(h[start]),
        h_neck_end=float(h[j]),
        neck_open_ended=j == n - 1,
        d_neck=d_neck,
        d_ref=d_ref,
        threshold=float(thr),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# cutting + capping
# ---------------------------------------------------------------------------
@dataclass
class CutPlane:
    name: str  # "distal" | "proximal"
    origin: np.ndarray  # on the centreline (the cut itself is nudged < 1e-3 mm off any vertex)
    normal: np.ndarray  # unit, pointing OUT of the kept segment
    height: float  # mm above the caudal end of the landmark centreline

    def to_dict(self) -> dict[str, Any]:
        return {
            "height_mm": round(self.height, 3),
            "origin": [round(float(v), 9) for v in self.origin],
            "normal_out": [round(float(v), 9) for v in self.normal],
        }


@dataclass
class CapInfo:
    plane: str
    area: float
    centroid: np.ndarray
    n_rim: int
    planarity: float  # max |distance| of a cap vertex from its plane
    triangulation: str  # "fan" | "polygon"
    inner: np.ndarray | None = None  # the cap point farthest from its rim
    inner_radius: float = 0.0  # its distance to the rim (inscribed radius)
    crease: float = 90.0  # smallest cap-to-wall angle along the rim, degrees

    @property
    def d_equiv(self) -> float:
        return 2.0 * math.sqrt(self.area / math.pi)

    def to_dict(self) -> dict[str, Any]:
        return {
            "area_mm2": round(self.area, 4),
            "d_equiv_mm": round(self.d_equiv, 3),
            "centroid": [round(float(v), 4) for v in self.centroid],
            "rim_vertices": self.n_rim,
            "planarity_mm": float(self.planarity),
            "triangulation": self.triangulation,
            "inscribed_radius_mm": round(float(self.inner_radius), 3),
            "rim_crease_min_deg": round(float(self.crease), 2),
        }


def _tri_array(surface: pv.PolyData) -> np.ndarray:
    faces = np.asarray(surface.faces)
    if surface.n_cells == 0 or len(faces) != 4 * surface.n_cells or not (
        faces.reshape(-1, 4)[:, 0] == 3
    ).all():
        raise CropError("the surface must be all triangles")
    return faces.reshape(-1, 4)[:, 1:].astype(np.int64)


def _signed_volume(pts: np.ndarray, tri: np.ndarray) -> float:
    return float(
        np.einsum("ij,ij->i", pts[tri[:, 0]], np.cross(pts[tri[:, 1]], pts[tri[:, 2]])).sum()
        / 6.0
    )


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seed = np.eye(3)[int(np.argmin(np.abs(normal)))]
    e1 = seed - float(seed @ normal) * normal
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(normal, e1)


def _nudge(points: np.ndarray, origin: np.ndarray, normal: np.ndarray, eps: float) -> np.ndarray:
    """Shift the plane along its normal by a few eps so that no vertex sits
    within eps of it (a vertex ON the plane makes zero-area clip slivers)."""
    dist = (points - origin) @ normal
    for k in range(64):
        shift = (k + 1) // 2 * (1 if k % 2 else -1) * 2.0 * eps
        if np.abs(dist - shift).min() >= eps:
            return origin + shift * normal
    raise CropError("could not place the cut plane clear of the surface vertices")


def _winding(loop2d: np.ndarray, q: np.ndarray) -> float:
    a = loop2d - q
    ang = np.arctan2(a[:, 1], a[:, 0])
    dang = np.diff(np.append(ang, ang[0]))
    dang = (dang + np.pi) % (2.0 * np.pi) - np.pi
    return float(dang.sum() / (2.0 * np.pi))


def _inscribed_centre(uv: np.ndarray, n_grid: int = 48) -> tuple[np.ndarray, float]:
    """The point of a simple 2-D polygon farthest from its rim (grid search,
    then a finer grid around the best cell) and that distance. A crescent
    lumen's centroid can lie outside it; this point never does."""
    a, b = uv, np.roll(uv, -1, axis=0)

    def inside(q: np.ndarray) -> np.ndarray:  # even-odd ray casting along +x
        cross = (a[None, :, 1] > q[:, None, 1]) != (b[None, :, 1] > q[:, None, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = a[None, :, 0] + (q[:, None, 1] - a[None, :, 1]) * (
                (b[None, :, 0] - a[None, :, 0]) / (b[None, :, 1] - a[None, :, 1])
            )
        return ((cross & (q[:, None, 0] < x_at)).sum(axis=1) % 2) == 1

    ab = b - a
    ab2 = np.maximum((ab**2).sum(axis=1), 1e-300)

    def rim_dist(q: np.ndarray) -> np.ndarray:
        t = np.clip(((q[:, None, :] - a[None]) * ab[None]).sum(axis=2) / ab2[None], 0.0, 1.0)
        near = a[None] + t[:, :, None] * ab[None]
        return np.sqrt(((q[:, None, :] - near) ** 2).sum(axis=2)).min(axis=1)

    lo, hi = uv.min(axis=0), uv.max(axis=0)
    best, best_d = uv.mean(axis=0), -1.0
    step = (hi - lo) / n_grid
    for _ in range(2):
        gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], n_grid + 1),
                             np.linspace(lo[1], hi[1], n_grid + 1))
        q = np.column_stack([gx.ravel(), gy.ravel()])
        q = q[np.concatenate([inside(q[k:k + 512]) for k in range(0, len(q), 512)])]
        if len(q):
            d = np.concatenate([rim_dist(q[k:k + 512]) for k in range(0, len(q), 512)])
            k = int(np.argmax(d))
            if d[k] > best_d:
                best, best_d = q[k], float(d[k])
        lo, hi = best - 2.0 * step, best + 2.0 * step
        step = (hi - lo) / n_grid
    return best, max(best_d, 0.0)


def _ear_clip(poly: np.ndarray) -> np.ndarray | None:
    """Triangulate a simple counter-clockwise 2-D polygon by ear clipping,
    every triangle strictly positive (a zero-area cap triangle would have no
    normal and split the flat-cap region the lumen anchoring looks for).

    Returns (n - 2, 3) local vertex indices, or None when no positive ear is
    left (a self-touching or numerically collinear rim)."""
    n = len(poly)
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum())
    if not area > 0:
        return None
    tiny = 1e-14 * area
    prv = np.roll(np.arange(n), 1)
    nxt = np.roll(np.arange(n), -1)
    alive = np.ones(n, dtype=bool)

    def twice(a: int, b: int, c: int) -> float:
        return float((poly[b, 0] - poly[a, 0]) * (poly[c, 1] - poly[a, 1])
                     - (poly[b, 1] - poly[a, 1]) * (poly[c, 0] - poly[a, 0]))

    def is_ear(i: int) -> bool:
        a, b, c = prv[i], i, nxt[i]
        if twice(a, b, c) <= 2.0 * tiny:
            return False
        others = alive.copy()
        others[[a, b, c]] = False
        q = poly[others]
        if len(q) == 0:
            return True
        pa, pb, pc = poly[a], poly[b], poly[c]

        def side(u, v):
            return (v[0] - u[0]) * (q[:, 1] - u[1]) - (v[1] - u[1]) * (q[:, 0] - u[0])

        inside = (side(pa, pb) >= -tiny) & (side(pb, pc) >= -tiny) & (side(pc, pa) >= -tiny)
        return not inside.any()

    tris: list[tuple[int, int, int]] = []
    left, i, stall = n, 0, 0
    while left > 3:
        if is_ear(i):
            a, c = prv[i], nxt[i]
            tris.append((a, i, c))
            nxt[a], prv[c] = c, a
            alive[i] = False
            left -= 1
            i, stall = a, 0
        else:
            i = nxt[i]
            stall += 1
            if stall > left:
                return None
    a = int(np.flatnonzero(alive)[0])
    b, c = nxt[a], nxt[nxt[a]]
    if twice(a, b, c) <= 2.0 * tiny:
        return None
    tris.append((a, b, c))
    return np.asarray(tris, dtype=np.int64)


def cut_and_cap(
    surface: pv.PolyData,
    planes: list[CutPlane],
    keep_point: np.ndarray,
    *,
    what: str = "surface",
    require_enclosure: bool = True,
    min_crease_deg: float | None = None,
) -> tuple[pv.PolyData, dict[str, CapInfo]]:
    """Keep the piece of a closed surface on the inner side of every plane
    that lies nearest ``keep_point``; close each plane's opening with a flat
    cap. Every structural surprise is a ``CropError`` naming ``what``.

    ``min_crease_deg``: refuse (retryable) a cap meeting the wall more
    shallowly than this anywhere along its rim.

    ``require_enclosure``: each opening must wind around its plane's origin
    (the centreline) — true for the outer surface, not for an eccentric
    blood lumen cut by the same planes.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from meshforge.meshing.lumen import flat_cap_cells

    if surface.n_open_edges:
        raise CropError(f"{what}: {surface.n_open_edges} open edge(s) — a crop needs a closed surface")
    tri0 = _tri_array(surface)
    pts0 = np.asarray(surface.points, dtype=float)
    edges = np.linalg.norm(pts0[tri0[:, 1]] - pts0[tri0[:, 0]], axis=1)
    eps = 1e-4 * float(np.median(edges[edges > 0])) if (edges > 0).any() else 1e-9
    work = pv.PolyData(pts0.copy(), np.asarray(surface.faces).copy())
    flag = np.zeros(surface.n_cells, dtype=np.int8)
    for cells in flat_cap_cells(surface):
        flag[cells] = 1
    work.cell_data["mf_flat_cap"] = flag
    used: list[CutPlane] = []
    for pl in planes:
        o = _nudge(np.asarray(work.points), pl.origin, pl.normal, eps)
        work = work.clip(normal=pl.normal, origin=o, invert=True)
        if not isinstance(work, pv.PolyData):
            work = work.extract_surface()
        if work.n_cells == 0:
            raise CropError(f"{what}: nothing is left inside the {pl.name} plane")
        used.append(CutPlane(pl.name, o, pl.normal, pl.height))
    work = work.triangulate()
    tri = _tri_array(work)
    pts = np.asarray(work.points, dtype=float)
    cflag = np.asarray(work.cell_data["mf_flat_cap"])

    # the connected piece nearest keep_point
    m = len(pts)
    rows = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]])
    cols = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0]])
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(m, m))
    _, lab = connected_components(graph, directed=False)
    used_v = np.unique(tri)
    near = used_v[np.argmin(np.linalg.norm(pts[used_v] - keep_point, axis=1))]
    keep = lab[tri[:, 0]] == lab[near]
    tri, cflag = tri[keep], cflag[keep]
    if cflag.any():
        bad = pts[np.unique(tri[cflag == 1])].mean(axis=0)
        which = min(used, key=lambda pl: abs(float((bad - pl.origin) @ pl.normal)))
        raise _MovePlane(
            f"{what}: the {which.name} plane runs through a flat end cap the input "
            f"already had — move it further inside (a square cut must clear a tilted "
            f"cap by about radius x tan(tilt))",
            which.name,
        )

    # boundary half-edges -> loops
    he = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    key = he[:, 0] * m + he[:, 1]
    if len(np.unique(key)) != len(key):
        raise CropError(f"{what}: inconsistently oriented or non-manifold triangles")
    rev = he[:, 1] * m + he[:, 0]
    on_rim = ~np.isin(rev, key)
    bnd = he[on_rim]
    rim_tri = dict(zip(bnd[:, 0].tolist(), np.tile(np.arange(len(tri)), 3)[on_rim].tolist()))
    nxt: dict[int, int] = {}
    for a, b in bnd:
        if int(a) in nxt:
            raise CropError(f"{what}: a pinched (non-manifold) opening at vertex {int(a)}")
        nxt[int(a)] = int(b)
    loops: list[list[int]] = []
    seen: set[int] = set()
    for start in nxt:
        if start in seen:
            continue
        loop, cur = [], start
        while cur not in seen:
            seen.add(cur)
            loop.append(cur)
            cur = nxt.get(cur, -1)
            if cur < 0:
                raise CropError(f"{what}: an opening whose rim does not close")
        if cur != start:
            raise CropError(f"{what}: an opening whose rim does not close")
        loops.append(loop)

    by_plane: dict[str, list[list[int]]] = {pl.name: [] for pl in used}
    for loop in loops:
        dist = {pl.name: float(np.abs((pts[loop] - pl.origin) @ pl.normal).max()) for pl in used}
        name = min(dist, key=dist.get)
        if dist[name] > ON_PLANE_TOL_MM:
            on_some = np.zeros(len(loop), dtype=bool)
            for pl in used:
                on_some |= np.abs((pts[loop] - pl.origin) @ pl.normal) <= ON_PLANE_TOL_MM
            if on_some.all() and len(used) == 2:
                raise _PlanesCross(
                    f"{what}: the two cut planes cross inside the vessel (one opening "
                    f"runs along both)"
                )
            raise CropError(
                f"{what}: an opening that no cut plane made ({len(loop)} rim vertices, "
                f"{dist[name]:.3g} mm off the {name} plane) — was the input closed?"
            )
        by_plane[name].append(loop)

    new_pts: list[np.ndarray] = []
    caps: list[np.ndarray] = []
    info: dict[str, CapInfo] = {}
    for pl in used:
        found = by_plane[pl.name]
        if len(found) != 1:
            raise _MovePlane(
                f"{what}: the {pl.name} plane opens the surface in {len(found)} loops "
                f"(expected exactly 1 — a branch, a second limb or a fold is cut too)",
                pl.name,
            )
        loop = np.asarray(found[0])
        pts[loop] -= np.outer((pts[loop] - pl.origin) @ pl.normal, pl.normal)  # snap
        wt = tri[[rim_tri[int(a)] for a in loop]]
        wn = np.cross(pts[wt[:, 1]] - pts[wt[:, 0]], pts[wt[:, 2]] - pts[wt[:, 0]])
        wl = np.linalg.norm(wn, axis=1)
        ok_n = wl > 0
        crease = float(np.degrees(np.arccos(np.clip(
            np.abs(wn[ok_n] @ pl.normal) / wl[ok_n], 0.0, 1.0))).min()) if ok_n.any() else 90.0
        if min_crease_deg is not None and crease < min_crease_deg:
            raise _MovePlane(
                f"{what}: the {pl.name} cap meets the wall at only {crease:.1f} deg "
                f"somewhere on its rim (< {min_crease_deg:g}: the lumen's flat-cap "
                f"anchoring would not recognise it)",
                pl.name,
            )
        e1, e2 = _plane_basis(pl.normal)
        rel = pts[loop] - pl.origin
        uv = np.column_stack([rel @ e1, rel @ e2])
        if require_enclosure and abs(_winding(uv, np.zeros(2))) < 0.5:
            raise CropError(
                f"{what}: the {pl.name} opening does not surround the centreline — the "
                f"cut caught the wrong piece of the surface"
            )
        x, y = uv[:, 0], uv[:, 1]
        cr = x * np.roll(y, -1) - np.roll(x, -1) * y
        area2 = float(cr.sum())  # twice the signed area, in the (e1, e2) frame
        if abs(area2) <= 0:
            raise CropError(f"{what}: the {pl.name} opening has zero area")
        c_uv = np.array([((x + np.roll(x, -1)) * cr).sum(), ((y + np.roll(y, -1)) * cr).sum()]) / (
            3.0 * area2
        )
        centre = pl.origin + c_uv[0] * e1 + c_uv[1] * e2
        a_uv, b_uv = uv - c_uv, np.roll(uv, -1, axis=0) - c_uv
        # cap triangle (c, b, a) for each boundary half-edge a -> b
        fan = b_uv[:, 0] * a_uv[:, 1] - b_uv[:, 1] * a_uv[:, 0]
        want = 1.0 if float(np.cross(e1, e2) @ pl.normal) > 0 else -1.0
        tiny = 1e-12 * abs(area2)
        if (want * fan > tiny).all():
            cid = m + len(new_pts)
            new_pts.append(centre)
            caps.append(np.column_stack([np.full(len(loop), cid), np.roll(loop, -1), loop]))
            kind = "fan"
        elif (want * fan < -tiny).all():
            raise CropError(
                f"{what}: the {pl.name} cap would face inward — the input surface is "
                f"not outward-oriented"
            )
        else:  # not star-shaped about its centroid: ear-clip the rim polygon
            rloop = loop[::-1]  # counter-clockwise seen from outside the cap
            loc = _ear_clip(uv[::-1])
            if loc is None:
                raise CropError(
                    f"{what}: the {pl.name} opening ({len(loop)} rim vertices) could not "
                    f"be triangulated into positive triangles"
                )
            caps.append(rloop[loc])
            kind = "polygon"
        cap_v = np.append(pts[loop], [centre] if kind == "fan" else np.empty((0, 3)), axis=0)
        in_uv, in_r = _inscribed_centre(uv)
        info[pl.name] = CapInfo(
            plane=pl.name,
            area=0.5 * abs(area2),
            centroid=centre,
            n_rim=len(loop),
            planarity=float(np.abs((cap_v - pl.origin) @ pl.normal).max()),
            triangulation=kind,
            inner=pl.origin + in_uv[0] * e1 + in_uv[1] * e2,
            inner_radius=in_r,
            crease=crease,
        )

    all_pts = np.vstack([pts] + ([np.asarray(new_pts)] if new_pts else []))
    all_tri = np.vstack([tri] + caps)
    used_ids = np.unique(all_tri)
    remap = np.full(len(all_pts), -1, dtype=np.int64)
    remap[used_ids] = np.arange(len(used_ids))
    out_tri = remap[all_tri]
    out_pts = all_pts[used_ids]
    faces = np.column_stack([np.full(len(out_tri), 3), out_tri]).ravel()
    out = pv.PolyData(out_pts, faces)
    nonmanifold = out.extract_feature_edges(
        boundary_edges=False, feature_edges=False, manifold_edges=False,
        non_manifold_edges=True,
    ).n_cells
    if out.n_open_edges or nonmanifold:
        raise CropError(
            f"{what}: the capped crop has {out.n_open_edges} open and {nonmanifold} "
            f"non-manifold edge(s)"
        )
    if _signed_volume(out_pts, out_tri) <= 0:
        raise CropError(f"{what}: the capped crop is inward-oriented (signed volume <= 0)")
    return out, info


# ---------------------------------------------------------------------------
# the whole crop
# ---------------------------------------------------------------------------
@dataclass
class CropResult:
    surface: pv.PolyData  # cropped, capped outer surface (centreline embedded)
    lumen: pv.PolyData | None  # the blood lumen cut by the same planes
    landmarks: Landmarks | None  # None only when manual heights skip the rules
    planes: list[CutPlane]
    caps: dict[str, CapInfo]
    lumen_caps: dict[str, CapInfo] | None
    centerline: Any  # landmark centreline (full length, caudal first)
    profile: DiameterProfile
    notes: list[str]

    def report(self) -> dict[str, Any]:
        lm = self.landmarks
        cl = self.centerline
        surf = self.surface
        tri = _tri_array(surf)
        rep: dict[str, Any] = {
            "frame": "input frame (RAS/LPS, +z superior), millimetres",
            "rules": {
                "profile": f"equivalent diameter 2*sqrt(A/pi) of centreline-normal "
                f"sections every {PROFILE_STEP_MM:g} mm, running median over "
                f"{SMOOTH_WINDOW_MM:g} mm; h = mm above the caudal end",
                "bifurcation": "caudal end of the label (h = 0)",
                "flare_top": "first local minimum of d(h) above the end zone's rise "
                "(0 when d rises straight into the sac)",
                "sac": f"maximum of d(h) from the flare top to h = {SAC_SEARCH_MM:g} mm",
                "neck": f"first stretch above the sac with d <= max({NECK_RATIO:g} x d_ref, "
                f"d_ref + {NECK_FLOOR_MM:g} mm) for >= {NECK_MIN_LENGTH_MM:g} mm; d_ref = "
                f"narrowest d within {NECK_SEARCH_MM:g} mm above the sac",
                "sac_min_ratio": SAC_MIN_RATIO,
            },
            "centerline": {
                "length_mm": round(float(cl.length), 3),
                "caudal_end": [round(float(v), 3) for v in cl.points[0]],
                "cranial_end": [round(float(v), 3) for v in cl.points[-1]],
            },
            "landmarks": None,
            "planes": {pl.name: pl.to_dict() for pl in self.planes},
            "caps": {k: v.to_dict() for k, v in self.caps.items()},
            "surface": {
                "n_points": int(surf.n_points),
                "n_triangles": int(surf.n_cells),
                "watertight": int(surf.n_open_edges) == 0,
                "volume_ml": round(_signed_volume(np.asarray(surf.points), tri) / 1000.0, 4),
            },
            "profile": {
                "h_mm": [round(float(v), 2) for v in self.profile.h],
                "d_equiv_mm": [round(float(v), 2) for v in self.profile.d_equiv],
                "d_smooth_mm": [round(float(v), 2) for v in self.profile.d_smooth],
            },
            "notes": list(self.notes),
        }
        if lm is not None:
            rep["landmarks"] = {
                "bifurcation_height_mm": 0.0,
                "flare_top_height_mm": round(lm.h_flare_top, 2),
                "sac_height_mm": round(lm.h_sac, 2),
                "sac_d_equiv_mm": round(lm.d_sac, 2),
                "sac_max_diameter_mm": round(lm.d_sac_max, 2),
                "neck_start_height_mm": round(lm.h_neck_start, 2),
                "neck_end_height_mm": round(lm.h_neck_end, 2),
                "neck_length_mm": round(lm.neck_length, 2),
                "neck_open_ended": bool(lm.neck_open_ended),
                "neck_d_equiv_mm": round(lm.d_neck, 2),
                "proximal_reference_d_mm": round(lm.d_ref, 2),
                "non_dilated_threshold_mm": round(lm.threshold, 2),
                "sac_neck_ratio": round(lm.sac_neck_ratio, 3),
            }
        for pl in self.planes:
            i = int(np.argmin(np.abs(cl.arc_lengths - pl.height)))
            cosang = abs(float(cl.tangents[i] @ pl.normal))
            rep["planes"][pl.name]["tilt_to_skeleton_deg"] = round(
                math.degrees(math.acos(min(1.0, cosang))), 4
            )
        if self.lumen is not None:
            lt = _tri_array(self.lumen)
            rep["lumen"] = {
                "n_points": int(self.lumen.n_points),
                "n_triangles": int(self.lumen.n_cells),
                "watertight": int(self.lumen.n_open_edges) == 0,
                "volume_ml": round(
                    _signed_volume(np.asarray(self.lumen.points), lt) / 1000.0, 4
                ),
                "caps": {k: v.to_dict() for k, v in (self.lumen_caps or {}).items()},
            }
        return rep


def _plane_at(cl, profile: DiameterProfile, height: float, name: str,
              sign: float) -> tuple[CutPlane, str, bool]:
    """The cut plane at ``height``: square to the local vessel axis.

    The skeleton centreline is a smoothing spline of a voxel path and can
    lean a few degrees off the true axis (on a short straight tube it is one
    cubic whose ends tilt ~3 degrees). A section cut slightly askew is still
    centred on the axis, so where the section centres within
    +-``PLANE_FIT_MM`` lie on a straight line (RMS <= ``PLANE_FIT_RMS_MM``,
    within ``PLANE_FIT_MAX_DEG`` of the skeleton) the plane is squared to
    that line (third return value True); elsewhere (a dome, an eccentric sac,
    a kink) the caller squares it to the segment axis instead."""
    i = int(np.argmin(np.abs(cl.arc_lengths - height)))
    o = np.asarray(cl.points[i], dtype=float)
    t = np.asarray(cl.tangents[i], dtype=float)
    t = t / np.linalg.norm(t)
    sel = np.abs(profile.h - cl.arc_lengths[i]) <= PLANE_FIT_MM
    if sel.sum() < 5:
        note = f"{name} plane: too few sections nearby for a local axis"
    else:
        c = profile.centers[sel]
        mean = c.mean(axis=0)
        axis = np.linalg.svd(c - mean)[2][0]
        axis = axis if float(axis @ t) >= 0 else -axis
        off = (c - mean) - np.outer((c - mean) @ axis, axis)
        rms = float(np.sqrt((off**2).sum(axis=1).mean()))
        lean = math.degrees(math.acos(min(1.0, abs(float(axis @ t)))))
        if rms <= PLANE_FIT_RMS_MM and lean <= PLANE_FIT_MAX_DEG:
            o = mean + float((o - mean) @ axis) * axis
            t = axis
            note = (
                f"{name} plane squared to the axis of {int(sel.sum())} section centres "
                f"(RMS {rms:.2f} mm; {lean:.2f} deg off the skeleton tangent)"
            )
        else:
            note = (
                f"{name} plane: the section centres within +-{PLANE_FIT_MM:g} mm are no "
                f"straight axis (RMS {rms:.2f} mm, {lean:.1f} deg off the skeleton)"
            )
    plane = CutPlane(name=name, origin=o, normal=sign * t, height=float(cl.arc_lengths[i]))
    return plane, note, note.startswith(f"{name} plane squared")


def _as_centerline(pts: np.ndarray, radii: np.ndarray, note: str):
    from meshforge.geometry.centerline import Centerline

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    deriv = np.gradient(pts, axis=0)
    tangents = deriv / np.linalg.norm(deriv, axis=1)[:, None]
    return Centerline(
        points=pts, tangents=tangents, arc_lengths=arc, radii=radii,
        length=float(arc[-1]), closed=False, voxel_size=0.0, notes=[note],
    )


def _fold_max(pts: np.ndarray, obb, n_probe: int = 40, n_rays: int = 16) -> float:
    """max over the curve of r x kappa — the radial lattices fold where this
    reaches 1 (the lumen O-grid) and the wall's curvature cap bites from
    ``curvature_cap``. r = the longest of ``n_rays`` ring spokes ray-cast in
    the plane square to the curve (the lattices' own sampling), at
    ``n_probe`` stations; a spoke that hits nothing makes the curve unusable
    (inf)."""
    # never ``import vtk``: the umbrella module loads the Viskores (vtkm)
    # accelerators, whose factory overrides then reroute every later slice and
    # contour — the SDF wall's contour segfaulted in a worker thread (D85)
    from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    if len(pts) <= 6:
        return float("inf")
    # curvature at the lattices' own station spacing (they sample 40-60
    # stations): a 0.5 mm finite difference would read sample noise as bends
    s = np.linspace(0.0, arc[-1], n_probe)
    pts = np.column_stack([np.interp(s, arc, pts[:, k]) for k in range(3)])
    t = np.gradient(pts, s, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    kappa = np.linalg.norm(np.gradient(t, s, axis=0), axis=1)
    reach = 4.0 * float(np.ptp(pts, axis=0).max() + 1.0)
    theta = np.linspace(0.0, 2.0 * math.pi, n_rays, endpoint=False)
    hit_pts, hit_ids = vtkPoints(), vtkIdList()
    worst = 0.0
    for i in range(1, n_probe - 1):
        e1, e2 = _plane_basis(t[i])
        r_max = 0.0
        for th in theta:
            d = math.cos(th) * e1 + math.sin(th) * e2
            obb.IntersectWithLine(pts[i], pts[i] + reach * d, hit_pts, hit_ids)
            if hit_pts.GetNumberOfPoints() == 0:
                return float("inf")
            r_max = max(r_max, float(np.linalg.norm(np.asarray(hit_pts.GetPoint(0)) - pts[i])))
        worst = max(worst, r_max * float(kappa[i]))
    return worst


def _segment_centerline(out: pv.PolyData, planes: list[CutPlane],
                        caps: dict[str, CapInfo], cl):
    """The centreline a cropped surface carries (``embed_centerline``), used
    by ``volmesh`` (wall lattice) and ``lumen`` (O-grid).

    Both ends start ``CAP_STANDOFF`` x the local radius inside their cap's
    inscribed centre (the cap point farthest from its rim — a crescent's
    centroid can lie outside it) and leave along the cap normal, so the end rings run parallel
    to the caps (a ring tilted more than asin(standoff) ~ 1.1 deg would hit
    its cap). Between them, the candidate with the smallest fold number
    max(r x kappa) whose every sample lies inside ``out``:

    * cubic Hermite curves cap to cap (tangent length 0.6-1.4 x the chord) —
      the least-bent curves with those ends. A skeleton wandering through a
      large sac folds the radial lattice wherever r x kappa nears 1, which is
      what capped and floored the wall there (D85);
    * the skeleton centreline ``cl`` between the caps, bent onto the cap
      normals over one radius — for a tortuous segment no cubic can follow.

    Returns (Centerline or None, note).""" 
    from scipy.spatial import cKDTree

    from meshforge.geometry.compat import interior_point_mask

    lo_pl, hi_pl = planes
    c0, c1 = caps[lo_pl.name].inner, caps[hi_pl.name].inner
    r0, r1 = caps[lo_pl.name].inner_radius, caps[hi_pl.name].inner_radius
    p0 = c0 - CAP_STANDOFF * r0 * lo_pl.normal
    p1 = c1 - CAP_STANDOFF * r1 * hi_pl.normal
    t0, t1 = -lo_pl.normal, hi_pl.normal

    tree = cKDTree(cl.points)
    i0, i1 = int(tree.query(p0)[1]), int(tree.query(p1)[1])
    span = abs(float(cl.arc_lengths[i1] - cl.arc_lengths[i0]))
    n = max(50, int(math.ceil(max(span, float(np.linalg.norm(p1 - p0))) / CL_SPACING_MM)) + 1)
    cands = []
    if span > 0:
        s = np.linspace(cl.arc_lengths[i0], cl.arc_lengths[i1], n)
        skel = np.column_stack([np.interp(s, cl.arc_lengths, cl.points[:, k]) for k in range(3)])
        u_end = np.abs(s - s[0]), np.abs(s[-1] - s)
        for start, direc, u, r in ((p0, t0, u_end[0], r0), (p1, -t1, u_end[1], r1)):
            blend = min(max(r, 5.0), 0.4 * span)
            line = start[None, :] + u[:, None] * direc[None, :]
            x = np.clip(u / blend, 0.0, 1.0)
            w = 1.0 - x * x * (3.0 - 2.0 * x)  # smoothstep: w(0)=1, w'(0)=0, w(blend)=0
            skel = skel + w[:, None] * (line - skel)
        cands.append(("skeleton centreline", skel))
    chord = float(np.linalg.norm(p1 - p0))
    u = np.linspace(0.0, 1.0, 20 * n)[:, None]
    h00, h10 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u
    h01, h11 = -2 * u**3 + 3 * u**2, u**3 - u**2
    for scale in (0.6, 0.8, 1.0, 1.2, 1.4):
        dense = h00 * p0 + h10 * scale * chord * t0 + h01 * p1 + h11 * scale * chord * t1
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
        tgt = np.linspace(0.0, cum[-1], n)
        pts = np.column_stack([np.interp(tgt, cum, dense[:, k]) for k in range(3)])
        cands.append((f"cap-to-cap cubic (tangent {scale:g} x chord)", pts))

    from vtkmodules.vtkFiltersGeneral import vtkOBBTree

    surf_pts = np.asarray(out.points, dtype=float)
    obb = vtkOBBTree()
    obb.SetDataSet(out)
    obb.BuildLocator()
    best, rejected = None, 0
    for label, pts in cands:
        if not interior_point_mask(pv.PolyData(pts), out, check_surface=False).all():
            rejected += 1
            continue
        fold = _fold_max(pts, obb)
        if not np.isfinite(fold):
            rejected += 1
            continue
        if best is None or fold < best[0]:
            best = (fold, label, pts)
    tilt_note = ""
    if best is None or best[0] >= SEGMENT_FOLD_MAX:
        # every cap-normal curve folds the rings (a short, wide, eccentric
        # sac): the straight line between the cap centres cannot fold. Its
        # end rings lean on the caps — the lumen O-grid anchors on its caps
        # (D81), a wall lattice would not, and says so through its counts.
        line = p0[None, :] + np.linspace(0.0, 1.0, n)[:, None] * (p1 - p0)[None, :]
        if interior_point_mask(pv.PolyData(line), out, check_surface=False).all():
            axis = (p1 - p0) / np.linalg.norm(p1 - p0)
            lean = [math.degrees(math.acos(min(1.0, abs(float(axis @ pl.normal)))))
                    for pl in planes]
            best = (0.0, "straight line between the cap centres", line)
            tilt_note = (
                f"; every cap-normal curve folds (max r x kappa >= {SEGMENT_FOLD_MAX:g}), "
                f"so its end rings lean {lean[0]:.0f} / {lean[1]:.0f} deg on the caps "
                f"(the lumen fill anchors on them; a wall lattice is not fit for this "
                f"segment)"
            )
        else:
            rejected += 1
    if best is None:
        return None, (
            "no segment centreline stays inside (the skeleton and every cap-to-cap "
            "curve leave the surface) — none embedded; volmesh/lumen compute their own"
        )
    fold, label, pts = best
    note = (
        f"segment centreline: {label}, fold number max(r x kappa) {fold:.3f}"
        + (f" ({rejected} candidate(s) left the surface)" if rejected else "")
        + tilt_note
    )
    from scipy.spatial import cKDTree

    radii = cKDTree(surf_pts).query(pts)[0]  # inscribed-ish: nearest wall distance
    return _as_centerline(pts, radii, f"crop step: {label}"), note


def _cut_with_centerline(surface: pv.PolyData, planes: list[CutPlane], keep: np.ndarray,
                         *, what: str, require_enclosure: bool, cl=None,
                         min_crease_deg: float | None = None):
    """cut_and_cap + embed a segment centreline (computed on the cropped
    surface itself when ``cl`` is not given). Returns (surface, caps, note)."""
    from meshforge.geometry.centerline import compute_centerline, embed_centerline

    out, caps = cut_and_cap(surface, planes, keep, what=what,
                            require_enclosure=require_enclosure,
                            min_crease_deg=min_crease_deg)
    if cl is None:
        b = out.bounds
        voxel = math.dist(b[::2], b[1::2]) / 150.0
        cl = compute_centerline(out, voxel_size=voxel, smoothing=max(voxel, CL_SMOOTHING_MM),
                                n_samples=400)
    seg_cl, note = _segment_centerline(out, planes, caps, cl)
    if seg_cl is not None:
        embed_centerline(out, seg_cl)
    shallow = [f"{k} {c.crease:.1f}" for k, c in caps.items() if c.crease < 10.0]
    if shallow:
        note += (
            f"; cap(s) meeting the wall under 10 deg ({', '.join(shallow)} deg): the "
            f"lumen fill will not anchor on them"
        )
    return out, caps, f"{what}: {note}"


def planes_from_report(report: dict[str, Any]) -> list[CutPlane]:
    """The two cut planes of an earlier crop (its report / summary JSON), so a
    surface made later (e.g. a recovered outer sac) is cut identically."""
    try:
        raw = report["planes"]
        planes = [
            CutPlane(
                name=name,
                origin=np.asarray(raw[name]["origin"], dtype=float),
                normal=np.asarray(raw[name]["normal_out"], dtype=float),
                height=float(raw[name]["height_mm"]),
            )
            for name in ("distal", "proximal")
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"not a crop report: needs planes.distal/proximal with origin, normal_out "
            f"and height_mm ({exc})"
        ) from None
    for pl in planes:
        if pl.origin.shape != (3,) or pl.normal.shape != (3,) or not (
            np.isfinite(pl.origin).all() and np.isfinite(pl.normal).all()
        ):
            raise ValueError(f"crop report: the {pl.name} plane is not a finite 3-vector pair")
        nn = float(np.linalg.norm(pl.normal))
        if not abs(nn - 1.0) < 1e-6:
            raise ValueError(f"crop report: the {pl.name} normal is not unit length ({nn:.6g})")
    if float(planes[0].normal @ planes[1].normal) > -0.0 and float(
        (planes[1].origin - planes[0].origin) @ planes[1].normal
    ) <= 0:
        raise ValueError("crop report: the proximal plane does not face away from the distal one")
    return planes


def crop_with_planes(surface: pv.PolyData, planes: list[CutPlane], *,
                     what: str = "surface") -> tuple[pv.PolyData, dict[str, CapInfo], str]:
    """Cut a closed surface with given planes (``planes_from_report``) and
    embed its segment centreline — the same cut as the crop that made the
    planes. The kept piece is the one nearest the midpoint of the plane
    origins; no landmark rules run."""
    if surface.n_open_edges:
        raise CropError(f"{what}: {surface.n_open_edges} open edges; it must be closed")
    keep = 0.5 * (planes[0].origin + planes[1].origin)
    return _cut_with_centerline(surface.triangulate(), planes, keep, what=what,
                                require_enclosure=False)


def crop_aaa(
    surface: pv.PolyData,
    *,
    neck_extension_mm: float = DEFAULT_NECK_EXTENSION_MM,
    distal_margin_mm: float = DEFAULT_DISTAL_MARGIN_MM,
    distal_height_mm: float | None = None,
    proximal_height_mm: float | None = None,
    lumen: pv.PolyData | None = None,
) -> CropResult:
    """Crop a closed aorta surface (mm, +z superior) to its infrarenal AAA
    segment; see the module docstring for the rules.

    ``distal_height_mm`` / ``proximal_height_mm`` (both or neither) place
    the planes by hand, in mm above the caudal end of the landmark
    centreline; the landmarks are then reported when found, never enforced.
    ``lumen`` — a second closed surface (the blood lumen inside an outer
    surface, or an outer sac around a lumen) — is cut by the SAME two
    planes, so both share their cap planes. Each cropped surface carries its
    own segment centreline.
    """
    for name, v in (("neck_extension_mm", neck_extension_mm), ("distal_margin_mm", distal_margin_mm)):
        if not (np.isfinite(v) and v >= 0):
            raise ValueError(f"{name} must be a finite number >= 0, got {v}")
    manual = (distal_height_mm is not None, proximal_height_mm is not None)
    if manual[0] != manual[1]:
        raise ValueError("give both distal_height_mm and proximal_height_mm, or neither")
    if surface.n_open_edges:
        raise CropError(
            f"surface has {surface.n_open_edges} open edges; the crop needs a "
            f"watertight surface (run 'meshforge check')"
        )
    if lumen is not None and lumen.n_open_edges:
        raise CropError(f"second surface has {lumen.n_open_edges} open edges; it must be closed")
    surface = surface.triangulate()
    cl = aaa_centerline(surface)
    notes = list(cl.notes)
    h_max = None if manual[0] else SAC_SEARCH_MM + NECK_SEARCH_MM
    profile = diameter_profile(surface, cl, h_max=h_max)

    landmarks: Landmarks | None
    lo_lim, hi_lim = END_CLEARANCE_MM, cl.length - END_CLEARANCE_MM
    if manual[0]:
        h_d, h_p = float(distal_height_mm), float(proximal_height_mm)
        if not (np.isfinite(h_d) and np.isfinite(h_p) and lo_lim <= h_d < h_p <= hi_lim):
            raise ValueError(
                f"manual heights must satisfy {lo_lim:g} <= distal < proximal <= "
                f"{hi_lim:.4g} mm (centreline length {cl.length:.4g} mm), got "
                f"{distal_height_mm} / {proximal_height_mm}"
            )
        try:
            landmarks = find_landmarks(profile)
        except CropError as exc:
            landmarks = None
            notes.append(f"manual heights; landmarks not found ({str(exc).split(';')[0]})")
        notes.append(f"manual heights: distal {h_d:.4g} mm, proximal {h_p:.4g} mm")
    else:
        landmarks = find_landmarks(profile)
        notes.extend(landmarks.notes)
        h_d = max(landmarks.h_flare_top + distal_margin_mm, lo_lim)
        if h_d >= landmarks.h_sac - PLANE_SHIFT_MM:
            room = landmarks.h_sac - max(landmarks.h_flare_top, lo_lim)
            if room < 2.0 * PLANE_SHIFT_MM:
                raise CropError(
                    f"no room below the sac: the flare top (h {landmarks.h_flare_top:.1f} "
                    f"mm) is within {room:.1f} mm of the sac (h {landmarks.h_sac:.1f} mm). "
                    f"{profile.text()}"
                )
            h_d = max(landmarks.h_flare_top, lo_lim) + 0.5 * room
            notes.append(
                f"distal plane held at h {h_d:.1f} mm, halfway between the flare top and "
                f"the sac: the margin ({distal_margin_mm:g} mm) would reach the sac"
            )
        h_p = landmarks.h_neck_start + neck_extension_mm
        cap_at = min(hi_lim, landmarks.h_neck_end) if not landmarks.neck_open_ended else hi_lim
        if h_p > cap_at:
            notes.append(
                f"proximal plane held at h {cap_at:.1f} mm: the requested neck extension "
                f"({neck_extension_mm:g} mm above the neck start at "
                f"{landmarks.h_neck_start:.1f} mm) runs past the "
                f"{'end of the surface' if cap_at == hi_lim else 'non-dilated neck'}"
            )
            h_p = cap_at
        if h_p <= landmarks.h_neck_start:
            raise CropError(
                f"no neck to extend into: the neck starts at h {landmarks.h_neck_start:.1f} "
                f"mm but the surface/neck ends at {cap_at:.1f} mm"
            )
    keep = cl.points[int(np.argmin(np.abs(cl.arc_lengths - 0.5 * (h_d + h_p))))]
    if landmarks is not None and h_d < landmarks.h_sac < h_p:
        keep = cl.points[int(np.argmin(np.abs(cl.arc_lengths - landmarks.h_sac)))]
    second = lumen.triangulate() if lumen is not None else None
    # a plane that opens more than one loop (a stub, a strut, a branch origin)
    # moves toward the sac in PLANE_SHIFT_MM steps, never past the sac or
    # below the neck start; both surfaces always share the final planes
    lo_stop = landmarks.h_sac - PLANE_SHIFT_MM if landmarks is not None else h_p
    hi_stop = landmarks.h_neck_start if landmarks is not None else h_d
    shifts = {"distal": 0, "proximal": 0}
    parallel = False
    while True:
        (pd, nd, okd), (pp, np_, okp) = (_plane_at(cl, profile, h_d, "distal", -1.0),
                                         _plane_at(cl, profile, h_p, "proximal", 1.0))
        # where no straight local axis exists (a dome, an eccentric sac, a
        # kink) the skeleton tangent can lean 50 deg off the vessel; square
        # that plane to the segment axis — the chord between the cut points
        chord = pp.origin - pd.origin
        chord = chord / np.linalg.norm(chord)
        if not okd or parallel:
            pd = CutPlane(pd.name, pd.origin, -chord, pd.height)
            nd += " — squared to the segment axis (chord between the cut points)"
        if not okp or parallel:
            pp = CutPlane(pp.name, pp.origin, chord, pp.height)
            np_ += " — squared to the segment axis (chord between the cut points)"
        planes = [pd, pp]
        try:
            crease = None if manual[0] else CAP_MIN_CREASE_DEG
            out, caps, seg_note = _cut_with_centerline(
                surface, planes, keep, what="outer surface", require_enclosure=True,
                cl=cl, min_crease_deg=crease,
            )
            lumen_out = lumen_caps = lumen_note = None
            if second is not None:
                lumen_out, lumen_caps, lumen_note = _cut_with_centerline(
                    second, planes, keep, what="second surface", require_enclosure=False,
                    min_crease_deg=crease,
                )
            break
        except _PlanesCross:
            if parallel or manual[0]:
                raise
            parallel = True  # square both to the chord: parallel planes never cross
            notes.append(
                "the two planes, each square to its own local axis, crossed inside the "
                "vessel — both squared to the segment axis instead"
            )
            continue
        except _MovePlane as exc:
            name = exc.plane
            nxt = h_d + PLANE_SHIFT_MM if name == "distal" else h_p - PLANE_SHIFT_MM
            ok = nxt < lo_stop if name == "distal" else nxt > hi_stop
            if manual[0] or shifts[name] >= PLANE_SHIFT_MAX or not ok:
                raise CropError(
                    f"{exc} — after moving the {name} plane {shifts[name]} x "
                    f"{PLANE_SHIFT_MM:g} mm toward the sac"
                ) from None
            shifts[name] += 1
            if name == "distal":
                h_d = nxt
            else:
                h_p = nxt
    notes.extend([nd, np_])
    for name, k in shifts.items():
        if k:
            notes.append(
                f"{name} plane moved {k * PLANE_SHIFT_MM:g} mm toward the sac: closer to "
                f"the end it opened more than one loop (a stub, strut or branch), met "
                f"the wall under {CAP_MIN_CREASE_DEG:g} deg, or ran through the input's "
                f"own end cap"
            )
    notes.append(seg_note)
    if lumen_note:
        notes.append(lumen_note)
    return CropResult(
        surface=out,
        lumen=lumen_out,
        landmarks=landmarks,
        planes=planes,
        caps=caps,
        lumen_caps=lumen_caps,
        centerline=cl,
        profile=profile,
        notes=notes,
    )
