"""Aneurysm sac recovery (D86): contrast CT + lumen mask -> outer sac.

On a native (non-stented) abdominal aortic aneurysm an AI ``aorta`` label
can trace only the contrast-filled lumen; wall-stress work needs the OUTER
sac: lumen + intraluminal thrombus (ILT) + wall. This module recovers it
from the CT intensities around the lumen, one axial slice at a time.

Method (per axial slice of the region of interest)
--------------------------------------------------
1. Origin = centroid of the slice's contrast lumen (lumen-mask voxels at or
   above ``lumen_hu``; the whole mask slice if none are). ``n_rays`` rays in
   equal world angles leave it; the lumen radius along a ray is its last
   lumen sample.
2. Walking outward from the lumen on a lightly smoothed slice (Gaussian
   0.5 mm), a ray stops at the FIRST of:

   * ``resolved`` — fat or air: HU < ``fat_hu`` sustained over
     ``FAT_RUN_MM``; the sac edge is the start of that run;
   * ``bright``   — a run of at least ``BRIGHT_RUN_MM`` at or above
     ``ENHANCED_HU`` (100 HU): another vessel, bone, or an enhancing organ
     such as the duodenum crossing in front of the sac (thrombus is
     ~20-60 HU). A shorter run is wall calcification and is walked through;
   * ``fov``      — the field-of-view edge;
   * ``no_fat``   — ``max_thickness_mm`` past the lumen with no fat:
     unenhanced soft tissue (IVC, psoas, bowel) abuts the sac.

   Only ``resolved`` rays measure the sac. Every ray also gives an UPPER
   bound for the sac edge (the sac holds no fat, bone or organ): the fat
   start, the bright start, the FOV edge, or lumen + ``max_thickness_mm``.
3. Errors are one-sided — a ray that crosses an abutting organ meets the fat
   BEHIND it — so outliers are rejected outward: (a) a bright/FOV stop bounds
   its neighbours too, relaxing by ``SLOPE_MM_PER_DEG`` around the circle,
   and a resolved ray more than 1 mm beyond that bound is demoted when it
   sits in a short run (<= ``SHORT_RUN`` rays) between stops — a long run of
   resolved rays is a real lobe of thrombus, not fat behind an organ; (b) a
   one-sided robust Fourier fit (order ``FOURIER_ORDER``) of thrombus+wall
   thickness vs angle demotes rays more than max(``OUTLIER_MM``, 2.5 robust
   sigma) outside it. Demoted rays are ``outlier``.
4. Unresolved rays are filled from their resolved neighbours by circular
   linear interpolation of the THICKNESS (so a gap follows the lumen's own
   shape); a slice with fewer than ``MIN_SLICE_RESOLVED`` of its rays
   resolved is filled from the nearest reliable slices along z. Bound, then
   Gaussian smoothing around the circumference (``SMOOTH_DEG``) and along z
   (``SMOOTH_Z_MM``), then the bounds again: lumen + ``MIN_WALL_MM`` <=
   radius <= the ray's (propagated) upper bound.
5. The slice mask is the star-shaped region under the radius curve, united
   with the lumen mask (so the sac always contains the lumen).

Uncertainty is reported, never hidden: the resolved fraction per slice and
over the sac, the count of every stop reason, and a per-vertex
``sac_resolved`` flag beside the ``ilt_thickness_mm`` field (lumen-to-sac
distance along the ray: thrombus PLUS wall). Too few resolved rays, no
lumen, a lumen touching the in-plane field-of-view edge, or a non-axial
grid are loud refusals (rule 6).

Axial slices, not centreline-normal planes: the sac MASK is the stack of
axial cross-sections of the 3-D sac, which is exact for an oblique vessel
too; only the per-ray thickness reads long by 1/cos(tilt) on a tilted
segment (reported as an axial in-plane distance).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: ray stop reasons (``SacResult.status``)
RESOLVED, NO_FAT, BRIGHT, FOV, OUTLIER = 0, 1, 2, 3, 4
STATUS_NAMES = ("resolved", "no_fat", "bright", "fov", "outlier")

# --- defaults of the knobs (pipeline step + CLI) ---------------------------
LUMEN_HU = 150.0  # contrast lumen, other vessels, bone
FAT_HU = -30.0  # fat / air below this
MAX_THICKNESS_MM = 40.0  # lumen-to-sac search limit along a ray
N_RAYS = 72  # 5 degrees apart
ROI_MM = 150.0  # height above the caudal lumen end (0 = whole lumen)
MIN_RESOLVED = 0.5  # refuse below this resolved fraction over the sac

# --- fixed method constants (documented, not knobs) ------------------------
FAT_RUN_MM = 1.5  # sustained fat needed to call the edge
MIN_WALL_MM = 1.5  # sac edge >= lumen + wall
BRIGHT_RUN_MM = 3.0  # a bright run at least this long = vessel / bone
SLICE_SIGMA_MM = 0.5  # in-plane Gaussian before sampling (noise)
OUTLIER_MM = 3.0  # floor of the circumferential outlier test
FOURIER_ORDER = 2  # robust thickness-vs-angle model per slice
MIN_SLICE_RESOLVED = 0.25  # below: the slice is filled along z
SMOOTH_DEG = 7.5  # circumferential Gaussian sigma
SMOOTH_Z_MM = 1.5  # along-z Gaussian sigma
MAX_TILT_DEG = 20.0  # slice normal vs world z
ENHANCED_HU = 100.0  # a sustained run at/above this: vessel, bone, enhancing organ
SLOPE_MM_PER_DEG = 0.5  # how fast a hard stop's bound relaxes around the circle
SHORT_RUN = 3  # resolved runs this short between hard stops can be demoted


@dataclass
class SacResult:
    """Everything one recovery produced, on the input grid."""

    mask: np.ndarray  # bool, the sac (contains the lumen), input-grid shape
    affine: np.ndarray  # voxel -> world (mm)
    slices: np.ndarray  # (S,) grid k index of each processed slice
    heights_mm: np.ndarray  # (S,) height above the caudal lumen slice
    origins: np.ndarray  # (S, 2) ray origin (i, j) per slice
    angles: np.ndarray  # (A,) ray angles, radians, in the (u1, u2) basis
    basis: np.ndarray  # (2, 3) in-plane world unit vectors u1, u2
    dirs: np.ndarray  # (A, 2) index-space step per mm along each ray
    lumen_radius: np.ndarray  # (S, A) mm
    measured: np.ndarray  # (S, A) mm, the resolved fat start (nan otherwise)
    upper: np.ndarray  # (S, A) mm, upper bound of the sac edge per ray
    radius: np.ndarray  # (S, A) mm, final sac radius
    status: np.ndarray  # (S, A) int8 stop reason (STATUS_NAMES)
    z_filled: np.ndarray  # (S,) bool, slice filled from its z neighbours
    params: dict[str, Any]
    summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def thickness(self) -> np.ndarray:
        """(S, A) lumen-to-sac distance along each ray (thrombus + wall), mm."""
        return self.radius - self.lumen_radius


def _check_params(lumen_hu, fat_hu, max_thickness_mm, n_rays, roi_mm, min_resolved):
    vals = dict(
        lumen_hu=float(lumen_hu),
        fat_hu=float(fat_hu),
        max_thickness_mm=float(max_thickness_mm),
        roi_mm=float(roi_mm),
        min_resolved=float(min_resolved),
    )
    for k, v in vals.items():
        if not np.isfinite(v):
            raise ValueError(f"sac: {k} must be finite, got {v}")
    if not vals["fat_hu"] < vals["lumen_hu"]:
        raise ValueError(
            f"sac: fat_hu ({fat_hu}) must be below lumen_hu ({lumen_hu})"
        )
    if vals["max_thickness_mm"] <= MIN_WALL_MM:
        raise ValueError(
            f"sac: max_thickness_mm must exceed the {MIN_WALL_MM} mm wall, got "
            f"{max_thickness_mm}"
        )
    if vals["roi_mm"] < 0:
        raise ValueError(f"sac: roi_mm must be >= 0 (0 = whole lumen), got {roi_mm}")
    if not 0.0 < vals["min_resolved"] <= 1.0:
        raise ValueError(f"sac: min_resolved must be in (0, 1], got {min_resolved}")
    n = int(n_rays)
    if n != n_rays or not 16 <= n <= 720:
        raise ValueError(f"sac: n_rays must be an integer in 16..720, got {n_rays}")
    vals["n_rays"] = n
    return vals


def _slice_frame(a: np.ndarray):
    """In-plane orthonormal world basis (u1, u2), the index-space step per mm
    along a world direction, and the slice tilt vs world z (degrees)."""
    e_i, e_j = a[:3, 0], a[:3, 1]
    normal = np.cross(e_i, e_j)
    normal /= np.linalg.norm(normal)
    tilt = float(np.degrees(np.arccos(min(1.0, abs(float(normal[2]))))))
    u1 = e_i / np.linalg.norm(e_i)
    u2 = e_j - (e_j @ u1) * u1
    u2 /= np.linalg.norm(u2)
    basis = np.stack([u1, u2])
    big = np.stack([e_i, e_j], axis=1)  # (3, 2): index (di, dj) -> world
    to_index = np.linalg.pinv(big)  # (2, 3): world in-plane vector -> (di, dj)
    return basis, to_index, tilt


def _run_starts(flag: np.ndarray, n: int) -> np.ndarray:
    """``out[..., t]`` = flag[..., t : t + n] all True (False past the end)."""
    n = max(int(n), 1)
    c = np.cumsum(np.concatenate([np.zeros(flag.shape[:-1] + (1,)), flag], axis=-1), axis=-1)
    m = flag.shape[-1]
    out = np.zeros(flag.shape, dtype=bool)
    if m >= n:
        out[..., : m - n + 1] = (c[..., n:] - c[..., : m - n + 1]) == n
    return out


def _robust_fourier_outliers(ang, r, order, floor_mm):
    """Mask of radii that disagree with a robust Fourier fit of r(angle).

    One-sided on purpose: a ray that crossed abutting soft tissue and met
    the fat behind it reads TOO FAR, never too near, so a point more than
    max(``floor_mm``, 2.5 robust sigma) OUTSIDE the fit is rejected and the
    fit is redone until stable (it walks inward off a wide bulge); inside
    the fit only a gross miss (2 x that) is rejected.
    """
    cols = [np.ones_like(ang)]
    for h in range(1, order + 1):
        cols += [np.cos(h * ang), np.sin(h * ang)]
    X = np.stack(cols, axis=1)
    keep = np.ones(len(r), dtype=bool)
    for _ in range(20):
        if keep.sum() < X.shape[1] + 3:
            break
        coef, *_ = np.linalg.lstsq(X[keep], r[keep], rcond=None)
        res = r - X @ coef
        sig = 1.4826 * float(np.median(np.abs(res[keep] - np.median(res[keep]))))
        hi = max(floor_mm, 2.5 * sig)
        new = (res <= hi) & (res >= -2.0 * hi)
        if np.array_equal(new, keep):
            break
        keep = new
    return ~keep


def _short_runs(flag: np.ndarray, max_len: int) -> np.ndarray:
    """Elements of circular runs of True no longer than ``max_len``."""
    n = flag.size
    out = np.zeros(n, dtype=bool)
    if flag.all() or not flag.any():
        return out
    start = int(np.flatnonzero(~flag)[0])  # rotate so a run never wraps
    f = np.roll(flag, -start)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], f.astype(int), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a <= max_len:
            out[np.arange(a, b)] = True
    return np.roll(out, start)


def _star_inside(points_ij, origin, basis_idx, angles, radius):
    """Mask of index points inside the star-shaped curve ``radius(angle)``
    around ``origin`` (radius linear in angle, periodic)."""
    v = (points_ij - origin) @ basis_idx.T  # (N, 2) world in-plane (u1, u2) mm
    theta = np.mod(np.arctan2(v[:, 1], v[:, 0]), 2 * np.pi)
    rho = np.hypot(v[:, 0], v[:, 1])
    r_at = np.interp(theta, angles, radius, period=2 * np.pi)
    return rho <= r_at


def recover_sac(
    ct: np.ndarray,
    lumen: np.ndarray,
    affine,
    *,
    lumen_hu: float = LUMEN_HU,
    fat_hu: float = FAT_HU,
    max_thickness_mm: float = MAX_THICKNESS_MM,
    n_rays: int = N_RAYS,
    roi_mm: float = ROI_MM,
    min_resolved: float = MIN_RESOLVED,
) -> SacResult:
    """Recover the outer sac around ``lumen`` from the CT ``ct`` (HU).

    ``ct`` and ``lumen`` share one grid; ``affine`` maps voxel index to world
    millimetres (any rotation/reflection, but the grid's slices must be
    axial within ``MAX_TILT_DEG``). Only the slices within ``roi_mm`` above
    the lumen's caudal end (lowest world z) are processed (0 = all), and
    only the largest 6-connected lumen component there. The array edges
    are the field of view. See the module docstring for the method.
    """
    from scipy import ndimage

    from meshforge.io.image import check_affine

    p = _check_params(lumen_hu, fat_hu, max_thickness_mm, n_rays, roi_mm, min_resolved)
    a = check_affine(affine)
    ct = np.asarray(ct)
    lumen = np.asarray(lumen).astype(bool, copy=False)
    if ct.ndim != 3 or lumen.shape != ct.shape:
        raise ValueError(
            f"sac: CT {ct.shape} and lumen mask {lumen.shape} must be the same 3-D grid"
        )
    if min(ct.shape) < 2:
        raise ValueError(f"sac: grid {ct.shape} is too small")
    basis, to_index, tilt = _slice_frame(a)
    if tilt > MAX_TILT_DEG:
        raise ValueError(
            f"sac: the grid's slices are {tilt:.1f} deg off axial (limit "
            f"{MAX_TILT_DEG:g}) — the method scans axial slices; resample the CT "
            f"to an axial grid first"
        )
    notes: list[str] = []
    if not lumen.any():
        raise ValueError("sac: the lumen mask is empty — there is no lumen to grow the sac from")

    # ---- region of interest: caudal roi_mm of the lumen, largest component
    nk = ct.shape[2]
    zc = a[2, 0] * (ct.shape[0] - 1) / 2 + a[2, 1] * (ct.shape[1] - 1) / 2 + a[2, 3]
    z_k = zc + a[2, 2] * np.arange(nk)
    has = lumen.any(axis=(0, 1))
    z_low = float(z_k[has].min())
    in_roi = has & (z_k <= z_low + p["roi_mm"] + 1e-6) if p["roi_mm"] > 0 else has
    ks = np.flatnonzero(in_roi)
    k0, k1 = int(ks.min()), int(ks.max())
    sub = lumen[:, :, k0 : k1 + 1] & in_roi[None, None, k0 : k1 + 1]
    lab, n_comp = ndimage.label(sub)
    if n_comp > 1:
        sizes = np.bincount(lab.ravel())[1:]
        keep = int(np.argmax(sizes)) + 1
        notes.append(
            f"kept the largest lumen component in the region of interest "
            f"({int(sizes[keep - 1])} voxels); dropped {n_comp - 1} other "
            f"component(s) ({int(sizes.sum() - sizes[keep - 1])} voxels)"
        )
        sub = lab == keep
    del lab
    if not sub.any():  # pragma: no cover - has[] guarantees a voxel
        raise ValueError("sac: no lumen voxels in the region of interest")
    kk = np.flatnonzero(sub.any(axis=(0, 1)))
    k0, k1 = k0 + int(kk.min()), k0 + int(kk.max())
    sub = sub[:, :, int(kk.min()) : int(kk.max()) + 1]
    edge = np.zeros(ct.shape[:2], dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    touching = np.flatnonzero((sub & edge[:, :, None]).any(axis=(0, 1)))
    if touching.size:
        raise ValueError(
            f"sac: the lumen touches the in-plane field-of-view edge on "
            f"{touching.size} slice(s) (first at grid k={k0 + int(touching[0])}) — "
            f"the sac around it lies partly outside the scan; refusing to "
            f"recover a sac the CT cannot show"
        )

    # ---- per-slice ray casting
    A = p["n_rays"]
    angles = 2 * np.pi * np.arange(A) / A
    world_dirs = np.cos(angles)[:, None] * basis[0] + np.sin(angles)[:, None] * basis[1]
    dirs = world_dirs @ to_index.T  # (A, 2) index step per mm
    basis_idx = (basis @ a[:3, :2])  # (2, 2): index offset (di, dj) -> (u1, u2) mm
    sp_in = np.linalg.norm(a[:3, :2], axis=0)
    dt = 0.5 * float(sp_in.min())
    n_fat = int(np.ceil(FAT_RUN_MM / dt))
    n_bright = int(np.ceil(BRIGHT_RUN_MM / dt))
    sigma_px = SLICE_SIGMA_MM / sp_in
    lh, fh, mt = p["lumen_hu"], p["fat_hu"], p["max_thickness_mm"]
    ni, nj = ct.shape[:2]

    S = k1 - k0 + 1
    origins = np.zeros((S, 2))
    r_lum = np.zeros((S, A))
    measured = np.full((S, A), np.nan)
    upper = np.full((S, A), np.inf)
    status = np.full((S, A), NO_FAT, dtype=np.int8)
    prop = np.full((S, A), np.inf)  # bounds propagated from hard stops
    ring = np.abs(np.arange(A)[:, None] - np.arange(A)[None, :])
    ring = np.minimum(ring, A - ring)  # circular ray distance
    slope = SLOPE_MM_PER_DEG * 360.0 / A  # mm per ray
    trimmed = 0
    for s in range(S):
        k = k0 + s
        L = sub[:, :, s]
        hu = np.asarray(ct[:, :, k], dtype=np.float32)
        Lb = L & (hu >= lh)
        if not Lb.any():
            Lb = L
        trimmed += int(np.count_nonzero(L & ~Lb))
        ci, cj = ndimage.center_of_mass(Lb)
        if not Lb[int(round(ci)), int(round(cj))]:
            edt = ndimage.distance_transform_edt(Lb, sampling=sp_in)
            ci, cj = np.unravel_index(int(np.argmax(edt)), edt.shape)
        o = np.array([float(ci), float(cj)])
        origins[s] = o
        li, lj = np.nonzero(Lb)
        r_far = float(np.max(np.hypot(*((np.stack([li, lj], 1) - o) @ basis_idx.T).T))) + 2 * dt
        t = np.arange(0.0, r_far + mt + FAT_RUN_MM + 2 * dt, dt)
        pts = o[None, None, :] + t[None, :, None] * dirs[:, None, :]  # (A, T, 2)
        inside = (
            (pts[..., 0] >= 0) & (pts[..., 0] <= ni - 1)
            & (pts[..., 1] >= 0) & (pts[..., 1] <= nj - 1)
        )
        sm = ndimage.gaussian_filter(hu, sigma=tuple(sigma_px), mode="nearest")
        prof = ndimage.map_coordinates(sm, [pts[..., 0].ravel(), pts[..., 1].ravel()],
                                       order=1, mode="nearest").reshape(pts.shape[:2])
        pi = np.clip(np.rint(pts[..., 0]).astype(int), 0, ni - 1)
        pj = np.clip(np.rint(pts[..., 1]).astype(int), 0, nj - 1)
        in_lumen = Lb[pi, pj] & inside & (t[None, :] <= r_far)
        fat_start = _run_starts((prof < fh) & inside, n_fat)
        bright_start = _run_starts((prof >= min(ENHANCED_HU, lh)) & inside, n_bright)
        for r in range(A):
            idx_l = np.flatnonzero(in_lumen[r])
            jl = int(idx_l[-1]) if idx_l.size else 0
            rl = float(t[jl])
            r_lum[s, r] = rl
            stop = int(np.searchsorted(t, rl + mt, side="right"))
            seg = slice(jl + 1, stop)
            ev_out = np.flatnonzero(~inside[r, seg])
            ev_fat = np.flatnonzero(fat_start[r, seg])
            ev_bright = np.flatnonzero(bright_start[r, seg])
            firsts = [
                (int(ev_fat[0]) if ev_fat.size else 10**9, RESOLVED),
                (int(ev_bright[0]) if ev_bright.size else 10**9, BRIGHT),
                (int(ev_out[0]) if ev_out.size else 10**9, FOV),
            ]
            j_ev, kind = min(firsts)
            if j_ev >= 10**9:
                status[s, r] = NO_FAT
                upper[s, r] = rl + mt
                continue
            te = float(t[jl + 1 + j_ev])
            status[s, r] = kind
            if kind == RESOLVED:
                measured[s, r] = te
                upper[s, r] = te + 0.5
            else:
                upper[s, r] = te
        # a hard stop (vessel / bone / enhancing organ, FOV edge) bounds its
        # neighbours too, relaxing by SLOPE_MM_PER_DEG: a fat hit far behind
        # an organ that blocks the next rays is fat BEHIND that organ
        hard = np.where((status[s] == BRIGHT) | (status[s] == FOV), upper[s], np.inf)
        if np.isfinite(hard).any():
            prop[s] = np.min(hard[None, :] + slope * ring, axis=1)
            res_s = status[s] == RESOLVED
            behind = res_s & (measured[s] > prop[s] + 1.0) & _short_runs(res_s, SHORT_RUN)
            status[s, behind] = OUTLIER
        # circumferential consistency of the resolved thickness
        res = status[s] == RESOLVED
        if res.sum() >= 2 * FOURIER_ORDER + 4:
            bad = _robust_fourier_outliers(
                angles[res], measured[s, res] - r_lum[s, res], FOURIER_ORDER, OUTLIER_MM
            )
            status[s, np.flatnonzero(res)[bad]] = OUTLIER
        # an outlier's "fat" is not trusted as a bound
        out = status[s] == OUTLIER
        upper[s, out] = r_lum[s, out] + mt

    # ---- fill, smooth, bound
    resolved = status == RESOLVED
    frac_slice = resolved.mean(axis=1)
    frac_all = float(resolved.mean())
    counts = {name: int(np.count_nonzero(status == i)) for i, name in enumerate(STATUS_NAMES)}
    if frac_all < p["min_resolved"]:
        raise ValueError(
            f"sac: only {frac_all:.1%} of the {status.size} rays found a fat plane "
            f"(need >= {p['min_resolved']:.0%}); stop reasons "
            + ", ".join(f"{k} {v}" for k, v in counts.items())
            + " — the sac edge is not visible on this CT (no fat around it, a "
            "tight field of view, or a wrong lumen); refusing to guess it"
        )
    # a slice is filled around its own circle when enough of it is seen (a
    # lowered min_resolved lowers this bar with it), else along z
    slice_bar = min(MIN_SLICE_RESOLVED, p["min_resolved"])
    reliable = (frac_slice >= slice_bar) & (resolved.sum(axis=1) >= 3)
    if not reliable.any():
        raise ValueError(
            f"sac: no slice has {slice_bar:.0%} of its rays (and at least 3) on a "
            f"fat plane — refusing to guess the sac"
        )
    # fill in THICKNESS (lumen-to-sac), so a gap follows the lumen's own shape
    R = np.zeros((S, A))
    for s in np.flatnonzero(reliable):
        m = resolved[s]
        R[s] = np.interp(angles, angles[m], measured[s, m] - r_lum[s, m], period=2 * np.pi)
    z_filled = ~reliable
    if z_filled.any():
        rs = np.flatnonzero(reliable)
        for r in range(A):
            R[z_filled, r] = np.interp(np.flatnonzero(z_filled), rs, R[rs, r])
        notes.append(
            f"{int(z_filled.sum())} of {S} slice(s) had under "
            f"{slice_bar:.0%} of their rays resolved; filled from the "
            f"nearest reliable slices along z"
        )
    R = R + r_lum
    # a resolved ray keeps its own (measured) bound; every other ray also
    # obeys the bounds propagated from the hard stops around it
    upper = np.where(resolved, upper, np.minimum(upper, prop))
    lower = r_lum + MIN_WALL_MM
    dz = abs(float(a[2, 2])) or 1.0
    R = np.minimum(R, upper)  # bound first, so smoothing does not spread a spike
    R = ndimage.gaussian_filter1d(R, SMOOTH_DEG / (360.0 / A), axis=1, mode="wrap")
    if S > 1:
        R = ndimage.gaussian_filter1d(R, SMOOTH_Z_MM / dz, axis=0, mode="nearest")
    R = np.maximum(np.minimum(R, upper), lower)

    # ---- rasterise: star-shaped region under R, united with the lumen
    mask = np.zeros(ct.shape, dtype=bool)
    buf = np.zeros(ct.shape[:2], dtype=bool)
    for s in range(S):
        k = k0 + s
        o = origins[s]
        rmax_idx = float(R[s].max()) * float(np.abs(dirs).max()) + 2
        i_lo, i_hi = max(int(o[0] - rmax_idx), 0), min(int(o[0] + rmax_idx) + 1, ni - 1)
        j_lo, j_hi = max(int(o[1] - rmax_idx), 0), min(int(o[1] + rmax_idx) + 1, nj - 1)
        gi, gj = np.mgrid[i_lo : i_hi + 1, j_lo : j_hi + 1]
        pts = np.stack([gi.ravel(), gj.ravel()], axis=1).astype(float)
        inside = _star_inside(pts, o, basis_idx, angles, R[s]).reshape(gi.shape)
        buf[:] = False
        buf[i_lo : i_hi + 1, j_lo : j_hi + 1] = inside
        mask[:, :, k] = buf | sub[:, :, s]

    heights = z_k[k0 : k1 + 1] - z_low
    result = SacResult(
        mask=mask,
        affine=a,
        slices=np.arange(k0, k1 + 1),
        heights_mm=heights,
        origins=origins,
        angles=angles,
        basis=basis,
        dirs=dirs,
        lumen_radius=r_lum,
        measured=measured,
        upper=upper,
        radius=R,
        status=status,
        z_filled=z_filled,
        params={
            **{k: p[k] for k in ("lumen_hu", "fat_hu", "max_thickness_mm", "n_rays", "roi_mm", "min_resolved")},
            "fat_run_mm": FAT_RUN_MM,
            "min_wall_mm": MIN_WALL_MM,
            "bright_run_mm": BRIGHT_RUN_MM,
            "slice_sigma_mm": SLICE_SIGMA_MM,
            "enhanced_hu": min(ENHANCED_HU, lh),
            "slope_mm_per_deg": SLOPE_MM_PER_DEG,
            "short_run_rays": SHORT_RUN,
            "outlier_mm": OUTLIER_MM,
            "fourier_order": FOURIER_ORDER,
            "min_slice_resolved": MIN_SLICE_RESOLVED,
            "smooth_deg": SMOOTH_DEG,
            "smooth_z_mm": SMOOTH_Z_MM,
            "ray_step_mm": dt,
        },
        notes=notes,
    )
    if trimmed:
        notes.append(
            f"{trimmed} lumen-mask voxel(s) in the region of interest are below "
            f"lumen_hu {lh:g} HU: not used as lumen for the rays (kept inside the sac)"
        )
    result.summary = _summarise(result, sub, frac_slice, frac_all, counts, tilt)
    return result


def _calipers(points: np.ndarray) -> tuple[float, float]:
    """(max, min) caliper width of a 2-D point set (mm)."""
    from scipy.spatial import ConvexHull

    hull = points[ConvexHull(points).vertices]
    d = hull[:, None, :] - hull[None, :, :]
    feret = float(np.sqrt((d**2).sum(-1)).max())
    th = np.deg2rad(np.arange(0.0, 180.0, 0.5))
    proj = hull @ np.stack([np.cos(th), np.sin(th)])
    return feret, float(np.ptp(proj, axis=0).min())


def _summarise(res: SacResult, lumen_sub, frac_slice, frac_all, counts, tilt) -> dict[str, Any]:
    a = res.affine
    S, A = res.radius.shape
    px_area = float(np.linalg.norm(np.cross(a[:3, 0], a[:3, 1])))
    voxel_ml = abs(float(np.linalg.det(a[:3, :3]))) / 1000.0
    th = res.thickness
    resolved = res.status == RESOLVED
    per_slice = []
    for s in range(S):
        k = int(res.slices[s])
        world = (
            res.radius[s][:, None] * np.stack([np.cos(res.angles), np.sin(res.angles)], 1)
        )
        feret, minw = _calipers(world)
        area = float(np.count_nonzero(res.mask[:, :, k])) * px_area
        lum_area = float(np.count_nonzero(lumen_sub[:, :, s])) * px_area
        per_slice.append({
            "k": k,
            "h_mm": round(float(res.heights_mm[s]), 2),
            "resolved_frac": round(float(frac_slice[s]), 3),
            "z_filled": bool(res.z_filled[s]),
            "sac_area_equiv_d_mm": round(float(2 * np.sqrt(area / np.pi)), 2),
            "sac_max_caliper_mm": round(feret, 2),
            "sac_min_caliper_mm": round(minw, 2),
            "lumen_area_equiv_d_mm": round(float(2 * np.sqrt(lum_area / np.pi)), 2),
            "thickness_median_mm": round(float(np.median(th[s])), 2),
            "thickness_max_mm": round(float(th[s].max()), 2),
        })

    def best(key):
        row = max(per_slice, key=lambda r: r[key])
        return {"value_mm": row[key], "h_mm": row["h_mm"], "k": row["k"]}

    sac_ml = float(np.count_nonzero(res.mask)) * voxel_ml
    lumen_ml = float(np.count_nonzero(lumen_sub)) * voxel_ml
    imax = np.unravel_index(int(np.argmax(th)), th.shape)
    return {
        "n_slices": int(S),
        "n_rays": int(A),
        "h_range_mm": [round(float(res.heights_mm.min()), 2), round(float(res.heights_mm.max()), 2)],
        "grid_k_range": [int(res.slices[0]), int(res.slices[-1])],
        "slice_tilt_deg": round(tilt, 2),
        "resolved_fraction": round(frac_all, 4),
        "resolved_fraction_per_slice": {
            "min": round(float(frac_slice.min()), 3),
            "p10": round(float(np.percentile(frac_slice, 10)), 3),
            "median": round(float(np.median(frac_slice)), 3),
        },
        "ray_status_counts": counts,
        "z_filled_slices": int(res.z_filled.sum()),
        "sac_volume_ml": round(sac_ml, 3),
        "lumen_volume_ml": round(lumen_ml, 3),
        "ilt_wall_volume_ml": round(sac_ml - lumen_ml, 3),
        "max_diameter": {
            "max_caliper": best("sac_max_caliper_mm"),
            "min_caliper": best("sac_min_caliper_mm"),
            "area_equivalent": best("sac_area_equiv_d_mm"),
        },
        "thickness_mm": {
            "definition": "lumen edge to outer sac edge along each axial ray "
                          "(intraluminal thrombus + wall)",
            "all_rays": {
                "median": round(float(np.median(th)), 2),
                "p90": round(float(np.percentile(th, 90)), 2),
                "max": round(float(th.max()), 2),
                "max_at_h_mm": round(float(res.heights_mm[imax[0]]), 2),
            },
            "resolved_rays": (
                {
                    "median": round(float(np.median(th[resolved])), 2),
                    "p90": round(float(np.percentile(th[resolved], 90)), 2),
                    "max": round(float(th[resolved].max()), 2),
                }
                if resolved.any() else None
            ),
        },
        "per_slice": per_slice,
    }


# ---------------------------------------------------------------------------
# surface + per-vertex fields
# ---------------------------------------------------------------------------
def ray_fields_at(res: SacResult, points: np.ndarray) -> dict[str, np.ndarray]:
    """Per-ray quantities interpolated at world ``points`` (N, 3):
    ``ilt_thickness_mm`` (linear in angle and between slices) and
    ``sac_resolved`` (1 where the nearest ray found its fat plane)."""
    a = res.affine
    inv = np.linalg.inv(a)
    q = np.asarray(points, dtype=float) @ inv[:3, :3].T + inv[:3, 3]
    S, A = res.radius.shape
    sf = np.clip(q[:, 2] - res.slices[0], 0.0, S - 1)
    s0 = np.floor(sf).astype(int)
    s1 = np.minimum(s0 + 1, S - 1)
    w = sf - s0
    basis_idx = res.basis @ a[:3, :2]
    th = res.thickness
    step = 2 * np.pi / A

    def at(s_idx):
        v = (q[:, :2] - res.origins[s_idx]) @ basis_idx.T
        theta = np.mod(np.arctan2(v[:, 1], v[:, 0]), 2 * np.pi)
        f = theta / step
        r0 = np.floor(f).astype(int) % A
        r1 = (r0 + 1) % A
        u = f - np.floor(f)
        val = (1 - u) * th[s_idx, r0] + u * th[s_idx, r1]
        near = np.where(u < 0.5, r0, r1)
        return val, res.status[s_idx, near] == RESOLVED

    v0, ok0 = at(s0)
    v1, ok1 = at(s1)
    thick = (1 - w) * v0 + w * v1
    resolved = np.where(w < 0.5, ok0, ok1)
    return {
        "ilt_thickness_mm": thick.astype(np.float32),
        "sac_resolved": resolved.astype(np.uint8),
    }


def sac_surface(res: SacResult, *, sigma: float = 1.0):
    """Closed outward sac surface (world mm) with the two vertex fields."""
    from meshforge.io.image import mask_to_surface

    surface, prov = mask_to_surface(res.mask, res.affine, sigma=sigma)
    for name, val in ray_fields_at(res, np.asarray(surface.points)).items():
        surface.point_data[name] = val
    return surface, prov


def write_sac_mask(res: SacResult, path: str | Path) -> Path:
    """Save the sac mask (uint8) as NIfTI on its own grid (the region-of-
    interest box of the CT; its affine places it in the CT's world)."""
    from meshforge.io.segment import _nib

    nib = _nib()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(res.mask.astype(np.uint8), res.affine)
    img.header.set_xyzt_units("mm", "sec")
    img.set_qform(res.affine, code=1)
    img.set_sform(res.affine, code=1)
    nib.save(img, str(path))
    return path


# ---------------------------------------------------------------------------
# files -> recovery (the pipeline step and the CLI)
# ---------------------------------------------------------------------------
@dataclass
class SacRun:
    result: SacResult
    surface: Any  # pv.PolyData, world (RAS) mm, with the vertex fields
    summary: dict[str, Any]
    grid_shape: tuple[int, int, int]  # the FULL CT grid
    grid_affine: np.ndarray
    input_sha256: dict[str, str]
    notes: list[str]


def _load_header(path: Path):
    """``(kind, shape, affine, notes)`` without reading voxels (NIfTI) or with
    the NRRD read (bounded) — returns the volume too for NRRD."""
    from meshforge.io.image import check_affine, nifti_affine_mm
    from meshforge.io.segment import MAX_CT_VOXELS, _nib

    name = path.name.lower()
    if name.endswith(".nrrd"):
        from meshforge.io.nrrd import read_nrrd

        vol = read_nrrd(path, max_voxels=MAX_CT_VOXELS)
        return vol.array, tuple(vol.array.shape), check_affine(vol.affine), list(vol.notes)
    if name.endswith((".nii", ".nii.gz")):
        img = _nib().load(str(path))
        shape = tuple(int(s) for s in img.header.get_data_shape())
        if len(shape) != 3:
            raise ValueError(f"{path.name}: a 3-D volume is required, got shape {shape}")
        if int(np.prod(shape)) > MAX_CT_VOXELS:
            raise ValueError(
                f"{path.name}: {shape} is over the {MAX_CT_VOXELS} voxel ceiling"
            )
        affine, notes = nifti_affine_mm(img)
        return img.dataobj, shape, affine, notes
    raise ValueError(
        f"unsupported volume {path.name!r}; supported: .nrrd, .nii, .nii.gz"
    )


def recover_sac_from_files(
    ct_path: str | Path,
    lumen_mask_path: str | Path,
    *,
    lumen_hu: float = LUMEN_HU,
    fat_hu: float = FAT_HU,
    max_thickness_mm: float = MAX_THICKNESS_MM,
    n_rays: int = N_RAYS,
    roi_mm: float = ROI_MM,
    min_resolved: float = MIN_RESOLVED,
    smoothing: float = 1.0,
) -> SacRun:
    """CT (.nrrd/.nii/.nii.gz, HU) + lumen mask on the same grid -> sac.

    Only the region of interest is read from a NIfTI CT: the caudal
    ``roi_mm`` of the lumen, boxed with an in-plane margin of
    ``max_thickness_mm`` + 10 mm (a ray can never reach a box edge that is
    not the scan's own edge). Both inputs are checksummed before and after
    (rule 1). ``smoothing`` = Gaussian sigma (voxels) of the sac mask before
    the surface.
    """
    from meshforge.io.load import file_sha256

    p = _check_params(lumen_hu, fat_hu, max_thickness_mm, n_rays, roi_mm, min_resolved)
    ct_path, lumen_mask_path = Path(ct_path), Path(lumen_mask_path)
    for label, path in (("CT", ct_path), ("lumen mask", lumen_mask_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    sha = {"ct": file_sha256(ct_path), "lumen_mask": file_sha256(lumen_mask_path)}

    ct_data, shape, a_ct, notes = _load_header(ct_path)
    m_data, m_shape, a_m, m_notes = _load_header(lumen_mask_path)
    if m_shape != shape or not np.allclose(a_m, a_ct, atol=1e-3):
        raise ValueError(
            f"the lumen mask grid {m_shape} does not match the CT grid {shape} "
            f"(or their affines differ) — resample the mask onto the CT first"
        )
    lumen = np.asarray(m_data) > 0.5
    if not lumen.any():
        raise ValueError(
            f"sac: the lumen mask {lumen_mask_path.name} is empty — there is no "
            f"lumen to grow the sac from"
        )
    # box: ROI slices of the lumen, in-plane bbox + margin
    zc = a_ct[2, 0] * (shape[0] - 1) / 2 + a_ct[2, 1] * (shape[1] - 1) / 2 + a_ct[2, 3]
    z_k = zc + a_ct[2, 2] * np.arange(shape[2])
    has = lumen.any(axis=(0, 1))
    z_low = float(z_k[has].min())
    roi = has & (z_k <= z_low + p["roi_mm"] + 1e-6) if p["roi_mm"] > 0 else has
    ks = np.flatnonzero(roi)
    sp = np.linalg.norm(a_ct[:3, :3], axis=0)
    margin = np.ceil((p["max_thickness_mm"] + 10.0) / sp[:2]).astype(int)
    ij = lumen[:, :, ks].any(axis=2)
    ii, jj = np.flatnonzero(ij.any(axis=1)), np.flatnonzero(ij.any(axis=0))
    lo = np.array([max(int(ii[0]) - margin[0], 0), max(int(jj[0]) - margin[1], 0), int(ks[0])])
    hi = np.array([
        min(int(ii[-1]) + margin[0], shape[0] - 1),
        min(int(jj[-1]) + margin[1], shape[1] - 1),
        int(ks[-1]),
    ])
    # one empty slice past each ROI end (when the scan has it): the sac's
    # flat ROI-end caps are then not mistaken for field-of-view caps
    lo[2], hi[2] = max(lo[2] - 1, 0), min(hi[2] + 1, shape[2] - 1)
    box = tuple(slice(int(l), int(h) + 1) for l, h in zip(lo, hi))
    ct_box = np.asarray(ct_data[box], dtype=np.float32)
    lumen_box = lumen[box] & roi[None, None, box[2]]
    del lumen, ct_data, m_data
    shift = np.eye(4)
    shift[:3, 3] = lo
    a_box = a_ct @ shift
    res = recover_sac(
        ct_box, lumen_box, a_box,
        lumen_hu=p["lumen_hu"], fat_hu=p["fat_hu"],
        max_thickness_mm=p["max_thickness_mm"], n_rays=p["n_rays"],
        roi_mm=0.0, min_resolved=p["min_resolved"],
    )
    surface, prov = sac_surface(res, sigma=smoothing)
    for label, path in (("CT", ct_path), ("lumen mask", lumen_mask_path)):
        key = "ct" if label == "CT" else "lumen_mask"
        if file_sha256(path) != sha[key]:
            raise RuntimeError(f"input changed on disk during sac recovery: {path}")
    all_notes = notes + m_notes + res.notes + list(prov["notes"])
    summary = {
        "inputs": {"ct": str(ct_path), "lumen_mask": str(lumen_mask_path)},
        "input_sha256": sha,
        "ct_grid": {
            "shape": [int(s) for s in shape],
            "spacing_mm": [round(float(s), 6) for s in sp],
            "roi_box_lo": [int(v) for v in lo],
            "roi_box_hi": [int(v) for v in hi],
        },
        "params": {**res.params, "roi_mm": p["roi_mm"], "smoothing": float(smoothing)},
        **res.summary,
        "surface": {
            "n_points": prov["n_points"],
            "n_triangles": prov["n_triangles"],
            "watertight": prov["watertight"],
            "enclosed_volume_ml": (
                None if prov["enclosed_volume"] is None
                else round(prov["enclosed_volume"] / 1000.0, 3)
            ),
            "point_data": ["ilt_thickness_mm", "sac_resolved"],
            "frame": "RAS world (NIfTI convention), millimetres",
        },
        "notes": all_notes,
    }
    return SacRun(
        result=res,
        surface=surface,
        summary=summary,
        grid_shape=tuple(int(s) for s in shape),
        grid_affine=a_ct,
        input_sha256=sha,
        notes=all_notes,
    )
