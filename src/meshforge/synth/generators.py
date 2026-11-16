from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pyvista as pv


@dataclass
class AnalyticTruth:
    """Exact ground truth paired with a generated surface.

    ``volume``/``surface_area`` are closed-form where one exists; the
    constricted tube's volume is a dense numeric quadrature of the exact
    radius profile (accurate far beyond mesh discretization error).
    """

    shape: str
    params: dict[str, Any]
    points: np.ndarray  # (N, 3) centerline samples, uniform in arc length
    tangents: np.ndarray  # (N, 3) unit tangents
    radii: np.ndarray  # (N,) tube radius at each sample
    arc_lengths: np.ndarray  # (N,) 0..length
    length: float
    closed: bool  # closed centerline loop (torus)
    surface_area: float | None = None
    volume: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "params": self.params,
            "length": self.length,
            "closed": self.closed,
            "surface_area": self.surface_area,
            "volume": self.volume,
            "notes": self.notes,
            "points": self.points.tolist(),
            "tangents": self.tangents.tolist(),
            "radii": self.radii.tolist(),
            "arc_lengths": self.arc_lengths.tolist(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _lattice_tube(
    centers: np.ndarray,  # (S, 3)
    normals: np.ndarray,  # (S, 3) in-plane frame vector 1
    binormals: np.ndarray,  # (S, 3) in-plane frame vector 2 (n x b = t)
    radii: np.ndarray,  # (S,)
    n_theta: int,
    *,
    periodic_s: bool = False,
) -> pv.PolyData:
    """Watertight all-triangle tube from a (s, θ) lattice.

    Winding is chosen so normals point outward by construction (∂θ × ∂s is
    the outward radial direction for a right-handed frame with n × b = t);
    tests assert positive signed volume against closed forms.
    """
    n_s = len(centers)
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # rings[i, j] = centers[i] + r_i * (cosθ_j n_i + sinθ_j b_i)
    rings = (
        centers[:, None, :]
        + radii[:, None, None]
        * (cos_t[None, :, None] * normals[:, None, :] + sin_t[None, :, None] * binormals[:, None, :])
    )
    points = rings.reshape(-1, 3)

    def ring_idx(i: int, j: int) -> int:
        return i * n_theta + (j % n_theta)

    faces: list[int] = []
    n_seg = n_s if periodic_s else n_s - 1
    for i in range(n_seg):
        i1 = (i + 1) % n_s
        for j in range(n_theta):
            a = ring_idx(i, j)
            b = ring_idx(i, j + 1)
            c = ring_idx(i1, j + 1)
            d = ring_idx(i1, j)
            faces += [3, a, b, c, 3, a, c, d]

    if not periodic_s:
        # Cap fans; winding gives outward (−t at s=0, +t at s=end).
        start_center = len(points)
        end_center = start_center + 1
        points = np.vstack([points, centers[0], centers[-1]])
        for j in range(n_theta):
            faces += [3, start_center, ring_idx(0, j + 1), ring_idx(0, j)]
            faces += [3, end_center, ring_idx(n_s - 1, j), ring_idx(n_s - 1, j + 1)]

    return pv.PolyData(points, np.asarray(faces, dtype=np.int64))


def _validate(n_theta: int, n_axial: int, **positives: float) -> None:
    if n_theta < 8:
        raise ValueError(f"n_theta must be >= 8, got {n_theta}")
    if n_axial < 2:
        raise ValueError(f"n_axial must be >= 2, got {n_axial}")
    for name, value in positives.items():
        if not value > 0:
            raise ValueError(f"{name} must be positive, got {value}")


def straight_tube(
    *,
    radius: float = 3.0,
    length: float = 20.0,
    n_theta: int = 64,
    n_axial: int = 41,
    n_truth: int = 501,
) -> tuple[pv.PolyData, AnalyticTruth]:
    """Capped straight tube along +z."""
    _validate(n_theta, n_axial, radius=radius, length=length)
    s = np.linspace(0.0, length, n_axial)
    centers = np.column_stack([np.zeros_like(s), np.zeros_like(s), s])
    n = np.tile([1.0, 0.0, 0.0], (n_axial, 1))
    b = np.tile([0.0, 1.0, 0.0], (n_axial, 1))
    surface = _lattice_tube(centers, n, b, np.full(n_axial, radius), n_theta)

    st = np.linspace(0.0, length, n_truth)
    truth = AnalyticTruth(
        shape="straight_tube",
        params={"radius": radius, "length": length, "n_theta": n_theta, "n_axial": n_axial},
        points=np.column_stack([np.zeros_like(st), np.zeros_like(st), st]),
        tangents=np.tile([0.0, 0.0, 1.0], (n_truth, 1)),
        radii=np.full(n_truth, radius),
        arc_lengths=st,
        length=length,
        closed=False,
        surface_area=2.0 * math.pi * radius * length + 2.0 * math.pi * radius**2,
        volume=math.pi * radius**2 * length,
    )
    return surface, truth


def curved_tube(
    *,
    radius: float = 2.0,
    bend_radius: float = 10.0,
    angle: float = math.pi / 2.0,
    n_theta: int = 64,
    n_axial: int = 81,
    n_truth: int = 501,
) -> tuple[pv.PolyData, AnalyticTruth]:
    """Capped tube swept along a circular arc in the x–z plane.

    Volume/lateral area are exact by Pappus (centroid path length R·Φ).
    """
    _validate(n_theta, n_axial, radius=radius, bend_radius=bend_radius, angle=angle)
    if radius >= bend_radius:
        raise ValueError(
            f"tube radius ({radius}) must be < bend radius ({bend_radius}); "
            "the inner wall would self-intersect"
        )

    def frames(phi: np.ndarray):
        centers = np.column_stack(
            [bend_radius * np.sin(phi), np.zeros_like(phi), bend_radius * (1.0 - np.cos(phi))]
        )
        tangents = np.column_stack([np.cos(phi), np.zeros_like(phi), np.sin(phi)])
        # In-plane radial direction; b = −y so that n × b = t (outward winding).
        normals = np.column_stack([-np.sin(phi), np.zeros_like(phi), np.cos(phi)])
        binormals = np.column_stack([np.zeros_like(phi), -np.ones_like(phi), np.zeros_like(phi)])
        return centers, tangents, normals, binormals

    phi = np.linspace(0.0, angle, n_axial)
    centers, _, n, b = frames(phi)
    surface = _lattice_tube(centers, n, b, np.full(n_axial, radius), n_theta)

    phi_t = np.linspace(0.0, angle, n_truth)
    centers_t, tangents_t, _, _ = frames(phi_t)
    arc_len = bend_radius * angle
    truth = AnalyticTruth(
        shape="curved_tube",
        params={
            "radius": radius,
            "bend_radius": bend_radius,
            "angle": angle,
            "n_theta": n_theta,
            "n_axial": n_axial,
        },
        points=centers_t,
        tangents=tangents_t,
        radii=np.full(n_truth, radius),
        arc_lengths=bend_radius * phi_t,
        length=arc_len,
        closed=False,
        surface_area=2.0 * math.pi * radius * arc_len + 2.0 * math.pi * radius**2,
        volume=math.pi * radius**2 * arc_len,
    )
    return surface, truth


def tapered_tube(
    *,
    r_start: float = 3.0,
    r_end: float = 1.5,
    length: float = 20.0,
    n_theta: int = 64,
    n_axial: int = 41,
    n_truth: int = 501,
) -> tuple[pv.PolyData, AnalyticTruth]:
    """Capped conical frustum along +z with linearly varying radius."""
    _validate(n_theta, n_axial, r_start=r_start, r_end=r_end, length=length)
    s = np.linspace(0.0, length, n_axial)
    centers = np.column_stack([np.zeros_like(s), np.zeros_like(s), s])
    n = np.tile([1.0, 0.0, 0.0], (n_axial, 1))
    b = np.tile([0.0, 1.0, 0.0], (n_axial, 1))
    radii = r_start + (r_end - r_start) * s / length
    surface = _lattice_tube(centers, n, b, radii, n_theta)

    slant = math.hypot(length, r_start - r_end)
    st = np.linspace(0.0, length, n_truth)
    truth = AnalyticTruth(
        shape="tapered_tube",
        params={
            "r_start": r_start,
            "r_end": r_end,
            "length": length,
            "n_theta": n_theta,
            "n_axial": n_axial,
        },
        points=np.column_stack([np.zeros_like(st), np.zeros_like(st), st]),
        tangents=np.tile([0.0, 0.0, 1.0], (n_truth, 1)),
        radii=r_start + (r_end - r_start) * st / length,
        arc_lengths=st,
        length=length,
        closed=False,
        surface_area=math.pi * (r_start + r_end) * slant
        + math.pi * (r_start**2 + r_end**2),
        volume=math.pi * length * (r_start**2 + r_start * r_end + r_end**2) / 3.0,
    )
    return surface, truth


def constricted_tube(
    *,
    radius: float = 3.0,
    length: float = 20.0,
    depth: float = 0.5,
    sigma: float = 2.0,
    n_theta: int = 64,
    n_axial: int = 81,
    n_truth: int = 2001,
) -> tuple[pv.PolyData, AnalyticTruth]:
    """Straight tube with a Gaussian constriction at mid-length.

    r(s) = radius · (1 − depth · exp(−(s − L/2)² / (2σ²))). Volume truth is a
    dense trapezoid quadrature of π r(s)² on the exact profile.
    """
    _validate(n_theta, n_axial, radius=radius, length=length, sigma=sigma)
    if not 0.0 < depth < 1.0:
        raise ValueError(f"depth must be in (0, 1) for an open lumen, got {depth}")

    def r_of(s: np.ndarray) -> np.ndarray:
        return radius * (1.0 - depth * np.exp(-((s - length / 2.0) ** 2) / (2.0 * sigma**2)))

    s = np.linspace(0.0, length, n_axial)
    centers = np.column_stack([np.zeros_like(s), np.zeros_like(s), s])
    n = np.tile([1.0, 0.0, 0.0], (n_axial, 1))
    b = np.tile([0.0, 1.0, 0.0], (n_axial, 1))
    surface = _lattice_tube(centers, n, b, r_of(s), n_theta)

    st = np.linspace(0.0, length, n_truth)
    volume = float(np.trapezoid(math.pi * r_of(st) ** 2, st))
    truth = AnalyticTruth(
        shape="constricted_tube",
        params={
            "radius": radius,
            "length": length,
            "depth": depth,
            "sigma": sigma,
            "n_theta": n_theta,
            "n_axial": n_axial,
        },
        points=np.column_stack([np.zeros_like(st), np.zeros_like(st), st]),
        tangents=np.tile([0.0, 0.0, 1.0], (n_truth, 1)),
        radii=r_of(st),
        arc_lengths=st,
        length=length,
        closed=False,
        surface_area=None,
        volume=volume,
        notes=["volume is dense numeric quadrature of the exact profile (not mesh-derived)"],
    )
    return surface, truth


def torus(
    *,
    tube_radius: float = 1.5,
    ring_radius: float = 6.0,
    n_theta: int = 48,
    n_axial: int = 96,
    n_truth: int = 501,
) -> tuple[pv.PolyData, AnalyticTruth]:
    """Closed torus (periodic in both directions, no caps)."""
    _validate(n_theta, n_axial, tube_radius=tube_radius, ring_radius=ring_radius)
    if tube_radius >= ring_radius:
        raise ValueError(
            f"tube_radius ({tube_radius}) must be < ring_radius ({ring_radius})"
        )

    def frames(u: np.ndarray):
        centers = np.column_stack(
            [ring_radius * np.cos(u), ring_radius * np.sin(u), np.zeros_like(u)]
        )
        tangents = np.column_stack([-np.sin(u), np.cos(u), np.zeros_like(u)])
        normals = np.column_stack([np.cos(u), np.sin(u), np.zeros_like(u)])
        binormals = np.column_stack(
            [np.zeros_like(u), np.zeros_like(u), -np.ones_like(u)]
        )  # b = t × n so that n × b = t
        return centers, tangents, normals, binormals

    u = np.linspace(0.0, 2.0 * math.pi, n_axial, endpoint=False)
    centers, _, n, b = frames(u)
    surface = _lattice_tube(
        centers, n, b, np.full(n_axial, tube_radius), n_theta, periodic_s=True
    )

    u_t = np.linspace(0.0, 2.0 * math.pi, n_truth)
    centers_t, tangents_t, _, _ = frames(u_t)
    truth = AnalyticTruth(
        shape="torus",
        params={
            "tube_radius": tube_radius,
            "ring_radius": ring_radius,
            "n_theta": n_theta,
            "n_axial": n_axial,
        },
        points=centers_t,
        tangents=tangents_t,
        radii=np.full(n_truth, tube_radius),
        arc_lengths=ring_radius * u_t,
        length=2.0 * math.pi * ring_radius,
        closed=True,
        surface_area=4.0 * math.pi**2 * ring_radius * tube_radius,
        volume=2.0 * math.pi**2 * ring_radius * tube_radius**2,
    )
    return surface, truth


def aneurysm_radius(
    s,
    *,
    radius: float = 10.0,
    sac_radius: float = 25.0,
    sac_center: float = 60.0,
    sac_sigma: float = 15.0,
    flare: float = 4.0,
    flare_length: float = 6.0,
):
    """Exact radius profile of :func:`aneurysm_tube` at arc length ``s``.

    r(s) = radius + (sac_radius − radius)·exp(−(s − sac_center)²/(2σ²))
    + flare·exp(−s/flare_length): a fusiform Gaussian sac on a straight
    calibre, plus a flare that widens the caudal end (s = 0) like the
    approach to an aortic bifurcation.
    """
    s = np.asarray(s, dtype=float)
    sac = (sac_radius - radius) * np.exp(-((s - sac_center) ** 2) / (2.0 * sac_sigma**2))
    return radius + sac + flare * np.exp(-s / flare_length)


def aneurysm_tube(
    *,
    radius: float = 10.0,
    sac_radius: float = 25.0,
    sac_center: float = 60.0,
    sac_sigma: float = 15.0,
    flare: float = 4.0,
    flare_length: float = 6.0,
    length: float = 160.0,
    bend_radius: float = 0.0,
    n_theta: int = 64,
    n_axial: int = 161,
    n_truth: int = 4001,
) -> tuple[pv.PolyData, AnalyticTruth]:
    """Capped tube with a fusiform aneurysm sac, a neck and a flared caudal end.

    Millimetre-scale defaults: a 20 mm calibre with a 50 mm sac centred
    60 mm above the flared (28 mm) caudal end, then a 20 mm neck up to the
    proximal cap at 160 mm — a synthetic infrarenal AAA with KNOWN landmarks
    (radius profile :func:`aneurysm_radius`). ``bend_radius`` > 0 sweeps the
    centreline along a circular arc in the x–z plane (a gently curved aorta;
    0 = straight along +z). s = 0 is the caudal (lowest-z) end either way.

    Volume truth is a dense quadrature of π r(s)² ds — exact for a tube of
    circular sections centred on its centreline (Pappus, section by section).
    """
    _validate(
        n_theta, n_axial, radius=radius, sac_radius=sac_radius,
        sac_sigma=sac_sigma, length=length, flare_length=flare_length,
    )
    if flare < 0:
        raise ValueError(f"flare must be >= 0, got {flare}")
    if not 0.0 <= sac_center <= length:
        raise ValueError(f"sac_center must lie in [0, length], got {sac_center}")
    profile = {
        "radius": radius, "sac_radius": sac_radius, "sac_center": sac_center,
        "sac_sigma": sac_sigma, "flare": flare, "flare_length": flare_length,
    }
    r_peak = float(aneurysm_radius(np.linspace(0.0, length, 4001), **profile).max())
    if bend_radius < 0:
        raise ValueError(f"bend_radius must be >= 0 (0 = straight), got {bend_radius}")
    if bend_radius and r_peak >= bend_radius:
        raise ValueError(
            f"the widest section (radius {r_peak:.4g}) must be < bend_radius "
            f"({bend_radius}); the concave wall would self-intersect"
        )

    def frames(s: np.ndarray):
        if not bend_radius:
            centers = np.column_stack([np.zeros_like(s), np.zeros_like(s), s])
            tangents = np.tile([0.0, 0.0, 1.0], (len(s), 1))
            normals = np.tile([1.0, 0.0, 0.0], (len(s), 1))
            binormals = np.tile([0.0, 1.0, 0.0], (len(s), 1))
            return centers, tangents, normals, binormals
        phi = s / bend_radius
        centers = np.column_stack(
            [bend_radius * (1.0 - np.cos(phi)), np.zeros_like(phi), bend_radius * np.sin(phi)]
        )
        tangents = np.column_stack([np.sin(phi), np.zeros_like(phi), np.cos(phi)])
        # n × b = t (outward winding, as in curved_tube)
        normals = np.column_stack([np.cos(phi), np.zeros_like(phi), -np.sin(phi)])
        binormals = np.tile([0.0, 1.0, 0.0], (len(s), 1))
        return centers, tangents, normals, binormals

    s = np.linspace(0.0, length, n_axial)
    centers, _, n, b = frames(s)
    surface = _lattice_tube(centers, n, b, aneurysm_radius(s, **profile), n_theta)

    st = np.linspace(0.0, length, n_truth)
    centers_t, tangents_t, _, _ = frames(st)
    radii_t = aneurysm_radius(st, **profile)
    truth = AnalyticTruth(
        shape="aneurysm_tube",
        params={**profile, "length": length, "bend_radius": bend_radius,
                "n_theta": n_theta, "n_axial": n_axial},
        points=centers_t,
        tangents=tangents_t,
        radii=radii_t,
        arc_lengths=st,
        length=length,
        closed=False,
        surface_area=None,
        volume=float(np.trapezoid(math.pi * radii_t**2, st)),
        notes=["volume is dense numeric quadrature of the exact profile (not mesh-derived)"],
    )
    return surface, truth


def _smoothstep(x):
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def eccentric_sac_profile(
    z,
    *,
    length: float = 60.0,
    sac_radius: float = 30.0,
    neck_radius: float = 12.0,
    shoulder: tuple[float, float] = (10.0, 45.0),
    bulge: float = 8.0,
    kink: float = 15.0,
    kink_span: tuple[float, float] = (35.0, 50.0),
):
    """Exact sections of :func:`eccentric_sac`: the circle in the plane ``z``
    has centre (cx, cy) and radius r. The sac (radius ``sac_radius``, its
    centre ``bulge`` off the neck axis along +x) narrows over ``shoulder`` to
    the neck (``neck_radius``), which steps ``kink`` sideways along +y over
    ``kink_span`` — smoothstep transitions, constant outside them (so the
    profile extends the surface past its caps, for distance checks)."""
    z = np.asarray(z, dtype=float)
    w = 1.0 - _smoothstep((z - shoulder[0]) / (shoulder[1] - shoulder[0]))
    r = neck_radius + (sac_radius - neck_radius) * w
    cx = bulge * w
    cy = kink * _smoothstep((z - kink_span[0]) / (kink_span[1] - kink_span[0]))
    return cx, cy, r


def eccentric_sac(
    *,
    length: float = 60.0,
    sac_radius: float = 30.0,
    neck_radius: float = 12.0,
    shoulder: tuple[float, float] = (10.0, 45.0),
    bulge: float = 8.0,
    kink: float = 15.0,
    kink_span: tuple[float, float] = (35.0, 50.0),
    z0: float = 0.0,
    z1: float | None = None,
    n_theta: int = 96,
    n_axial: int = 121,
    n_truth: int = 4001,
) -> tuple[pv.PolyData, AnalyticTruth]:
    """A cropped aneurysm segment between two PARALLEL flat caps (D87): a wide
    sac whose centre sits off the neck axis, a shoulder into a narrower neck
    and a sideways kink into it — the shape whose centreline-ray wall folds.

    Every section in a plane z = const is an exact circle (``eccentric_sac_profile``)
    — the sections are NOT square to their centre path — and the caps are
    the planes z = ``z0`` and z = ``z1`` (default ``length``; a range past
    [0, length] extends the constant sac / neck — a reference surface for
    distance checks). Volume truth is
    ∫ π r(z)² dz (Cavalieri: a sheared stack of circles), by dense
    quadrature. ``truth.points`` is the centre path (uniform in arc length)
    and ``truth.radii`` the in-plane section radii. Not in ``GENERATORS``: its
    truth is a stack of z-sections, not a tube square to a centreline.
    """
    z1 = length if z1 is None else z1
    _validate(n_theta, n_axial, length=length, sac_radius=sac_radius,
              neck_radius=neck_radius)
    if not z0 < z1:
        raise ValueError(f"need z0 < z1, got z0={z0}, z1={z1}")
    if n_theta < 48:
        # coarser rings make the flat wall facets of the straight stretches
        # crease >= 7.5 deg, enough to pass as extra flat caps
        raise ValueError(f"n_theta must be >= 48 for eccentric_sac, got {n_theta}")
    profile = {
        "length": length, "sac_radius": sac_radius, "neck_radius": neck_radius,
        "shoulder": tuple(shoulder), "bulge": bulge, "kink": kink,
        "kink_span": tuple(kink_span),
    }
    z = np.linspace(z0, z1, n_axial)
    cx, cy, r = eccentric_sac_profile(z, **profile)
    centers = np.column_stack([cx, cy, z])
    normals = np.tile([1.0, 0.0, 0.0], (n_axial, 1))
    binormals = np.tile([0.0, 1.0, 0.0], (n_axial, 1))
    surface = _lattice_tube(centers, normals, binormals, r, n_theta)

    zd = np.linspace(z0, z1, n_truth)
    cxd, cyd, rd = eccentric_sac_profile(zd, **profile)
    path = np.column_stack([cxd, cyd, zd])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    st = np.linspace(0.0, arc[-1], n_truth)
    pts = np.column_stack([np.interp(st, arc, path[:, k]) for k in range(3)])
    tang = np.gradient(pts, st, axis=0)
    tang /= np.linalg.norm(tang, axis=1)[:, None]
    truth = AnalyticTruth(
        shape="eccentric_sac",
        params={**profile, "z0": z0, "z1": z1, "n_theta": n_theta, "n_axial": n_axial},
        points=pts,
        tangents=tang,
        radii=np.interp(st, arc, rd),
        arc_lengths=st,
        length=float(arc[-1]),
        closed=False,
        surface_area=None,
        volume=float(np.trapezoid(math.pi * rd**2, zd)),
        notes=[
            "sections are circles in the planes z = const (radii in-plane), not "
            "square to the centre path; volume = integral of pi r(z)^2 dz"
        ],
    )
    return surface, truth


GENERATORS: dict[str, Callable[..., tuple[pv.PolyData, AnalyticTruth]]] = {
    "straight_tube": straight_tube,
    "curved_tube": curved_tube,
    "tapered_tube": tapered_tube,
    "constricted_tube": constricted_tube,
    "torus": torus,
    "aneurysm_tube": aneurysm_tube,
}


def generate(shape: str, **params: Any) -> tuple[pv.PolyData, AnalyticTruth]:
    """Generate a synthetic surface + truth by shape name (loud on unknowns)."""
    try:
        gen = GENERATORS[shape]
    except KeyError:
        raise ValueError(
            f"unknown shape {shape!r}; available: {', '.join(sorted(GENERATORS))}"
        ) from None
    return gen(**params)
