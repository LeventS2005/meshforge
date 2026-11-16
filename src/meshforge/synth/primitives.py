"""Geometry library: watertight primitive solids with analytic ground truth.

Same registry/truth pattern as ``generators.py``: each primitive returns
(surface, PrimitiveTruth). Where the discrete mesh admits a closed form
(box, n-gon prism, extrusion), the truth matches the MESH exactly; where it
does not (sphere, revolution), the truth is the continuous closed form and
the mesh converges to it (tolerance stated in tests).

``extrude_loops``/``revolve_loop`` take plain (N, 2) numpy loops — outer CCW,
holes CW, the usual 2-D profile convention — but this module
never imports CAD classes.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pyvista as pv


@dataclass
class PrimitiveTruth:
    """Ground truth paired with a generated primitive surface.

    ``exact`` = True means surface_area/volume are closed forms of the
    DISCRETE mesh (equality up to float roundoff); False means they are the
    continuous solid's values (mesh converges with resolution).
    """

    shape: str
    params: dict[str, Any]
    surface_area: float | None
    volume: float | None
    exact: bool
    n_faces: int  # exact expected triangle count of the mesh
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "params": self.params,
            "surface_area": self.surface_area,
            "volume": self.volume,
            "exact": self.exact,
            "n_faces": self.n_faces,
            "notes": self.notes,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _positive(**values: float) -> None:
    for name, value in values.items():
        if not value > 0:
            raise ValueError(f"{name} must be positive, got {value}")


# ---------------------------------------------------------------------------
# cap triangulation (ear clipping with hole bridging — plain numpy)


def _signed_area_2d(loop: np.ndarray) -> float:
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _point_in_triangle(p, a, b, c, eps: float) -> bool:
    def cross(o, u, v):
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])

    d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)


def _ear_clip(poly_xy: np.ndarray, idx: list[int]) -> list[tuple[int, int, int]]:
    """Triangulate a (weakly) simple CCW polygon given as positions + global
    vertex ids; returns triangles as global id triples."""
    n = len(idx)
    if n < 3:
        raise ValueError("cannot triangulate a polygon with < 3 vertices")
    order = list(range(n))
    scale = float(np.abs(poly_xy).max()) or 1.0
    eps = 1e-12 * scale * scale
    tris: list[tuple[int, int, int]] = []
    guard = 0
    while len(order) > 3:
        guard += 1
        if guard > 4 * n * n:
            raise ValueError(
                "cap triangulation failed (no ear found) — loop is likely "
                "self-intersecting or mis-oriented"
            )
        n_cur = len(order)
        clipped = False
        for k in range(n_cur):
            i_prev, i_cur, i_next = (
                order[(k - 1) % n_cur],
                order[k],
                order[(k + 1) % n_cur],
            )
            a, b, c = poly_xy[i_prev], poly_xy[i_cur], poly_xy[i_next]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= eps:  # reflex or collinear — not an ear
                continue
            ear = True
            for m in order:
                if m in (i_prev, i_cur, i_next):
                    continue
                # Skip duplicated bridge vertices sitting exactly on a corner.
                if (
                    np.array_equal(poly_xy[m], a)
                    or np.array_equal(poly_xy[m], b)
                    or np.array_equal(poly_xy[m], c)
                ):
                    continue
                if _point_in_triangle(poly_xy[m], a, b, c, eps):
                    ear = False
                    break
            if not ear:
                continue
            tris.append((idx[i_prev], idx[i_cur], idx[i_next]))
            del order[k]
            clipped = True
            break
        if not clipped:
            raise ValueError(
                "cap triangulation failed (no ear found) — loop is likely "
                "self-intersecting or mis-oriented"
            )
    tris.append((idx[order[0]], idx[order[1]], idx[order[2]]))
    return tris


def _bridge_holes(
    outer: np.ndarray,
    outer_ids: list[int],
    holes: list[np.ndarray],
    hole_ids: list[list[int]],
) -> tuple[np.ndarray, list[int]]:
    """Merge holes into the outer boundary with zero-width bridges (shared
    vertex ids, so the surface stays watertight). Classic construction:
    connect each hole's rightmost vertex to a visible outer vertex."""
    poly = outer.copy()
    ids = list(outer_ids)
    # Process holes right-to-left so earlier bridges cannot block later ones.
    order = sorted(range(len(holes)), key=lambda h: -holes[h][:, 0].max())
    for h in order:
        hole, hids = holes[h], hole_ids[h]
        m = int(np.argmax(hole[:, 0]))
        hp = hole[m]
        # Candidate outer vertices to the right of the hole vertex, nearest
        # first; take the first that yields a non-crossing bridge.
        cands = [k for k in range(len(poly)) if poly[k][0] >= hp[0]]
        if not cands:
            raise ValueError(
                "hole extends beyond its outer loop — invalid nesting"
            )
        cands.sort(key=lambda k: float(np.hypot(*(poly[k] - hp))))
        best = None
        for k in cands:
            if _bridge_clear(hp, poly[k], poly, hole):
                best = k
                break
        if best is None:  # fall back to nearest; ear clipping will complain
            best = cands[0]
        # Splice: outer[..best] + [best, hole[m..], hole[..m], hole[m], best..]
        rolled = np.vstack([hole[m:], hole[:m]])
        rolled_ids = hids[m:] + hids[:m]
        poly = np.vstack(
            [poly[: best + 1], rolled, rolled[:1], poly[best:]]
        )
        ids = (
            ids[: best + 1]
            + rolled_ids
            + [rolled_ids[0]]
            + ids[best:]
        )
    return poly, ids


def _bridge_clear(p: np.ndarray, q: np.ndarray, *loops: np.ndarray) -> bool:
    """True if segment p-q crosses no edge of the given loops (touching at
    the endpoints themselves is fine)."""
    d = q - p
    for loop in loops:
        a = loop
        b = np.roll(loop, -1, axis=0)
        e = b - a
        denom = d[0] * e[:, 1] - d[1] * e[:, 0]
        rel = a - p
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (rel[:, 0] * e[:, 1] - rel[:, 1] * e[:, 0]) / denom
            s = (rel[:, 0] * d[1] - rel[:, 1] * d[0]) / denom
        hit = (
            np.isfinite(t)
            & np.isfinite(s)
            & (t > 1e-9)
            & (t < 1 - 1e-9)
            & (s > 1e-9)
            & (s < 1 - 1e-9)
        )
        if bool(hit.any()):
            return False
    return True


def _cap_triangles(
    loops_xy: list[np.ndarray], loop_ids: list[list[int]]
) -> list[tuple[int, int, int]]:
    """Triangulate the region (outer CCW + holes CW) into global-id triangles
    wound CCW in the plane."""
    outer, holes = loops_xy[0], loops_xy[1:]
    if _signed_area_2d(outer) <= 0:
        raise ValueError("outer loop must be CCW (positive signed area)")
    for h in holes:
        if _signed_area_2d(h) >= 0:
            raise ValueError("hole loops must be CW (negative signed area)")
    poly, ids = _bridge_holes(outer, loop_ids[0], holes, loop_ids[1:])
    return _ear_clip_positions(poly, ids)


def _ear_clip_positions(poly: np.ndarray, ids: list[int]) -> list[tuple[int, int, int]]:
    tris_local = _ear_clip(poly, list(range(len(poly))))
    return [(ids[a], ids[b], ids[c]) for a, b, c in tris_local]


# ---------------------------------------------------------------------------
# primitives


def box(
    *,
    width: float = 4.0,
    height: float = 3.0,
    depth: float = 2.0,
) -> tuple[pv.PolyData, PrimitiveTruth]:
    """Axis-aligned box centered at the origin; 12 triangles, exact truth."""
    _positive(width=width, height=height, depth=depth)
    w, h, d = width / 2.0, height / 2.0, depth / 2.0
    pts = np.array(
        [
            [-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d],
            [-w, -h, d], [w, -h, d], [w, h, d], [-w, h, d],
        ]
    )
    quads = [
        (0, 3, 2, 1),  # bottom (-z)
        (4, 5, 6, 7),  # top (+z)
        (0, 1, 5, 4),  # -y
        (1, 2, 6, 5),  # +x
        (2, 3, 7, 6),  # +y
        (3, 0, 4, 7),  # -x
    ]
    faces: list[int] = []
    for a, b, c, dd in quads:
        faces += [3, a, b, c, 3, a, c, dd]
    surface = pv.PolyData(pts, np.asarray(faces, dtype=np.int64))
    truth = PrimitiveTruth(
        shape="box",
        params={"width": width, "height": height, "depth": depth},
        surface_area=2.0 * (width * height + width * depth + height * depth),
        volume=width * height * depth,
        exact=True,
        n_faces=12,
    )
    return surface, truth


def solid_cylinder(
    *,
    radius: float = 3.0,
    height: float = 10.0,
    n_theta: int = 64,
) -> tuple[pv.PolyData, PrimitiveTruth]:
    """Capped cylinder along +z (base at z=0). Truth is the EXACT inscribed
    n-gon prism (matches the mesh to roundoff), continuous values in notes."""
    _positive(radius=radius, height=height)
    if n_theta < 8:
        raise ValueError(f"n_theta must be >= 8, got {n_theta}")
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    ring = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    bottom = np.column_stack([ring, np.zeros(n_theta)])
    top = np.column_stack([ring, np.full(n_theta, height)])
    pts = np.vstack([bottom, top, [[0, 0, 0]], [[0, 0, height]]])
    cb, ct = 2 * n_theta, 2 * n_theta + 1

    faces: list[int] = []
    for j in range(n_theta):
        j1 = (j + 1) % n_theta
        faces += [3, j, j1, n_theta + j1, 3, j, n_theta + j1, n_theta + j]  # side
        faces += [3, cb, j1, j]  # bottom cap (outward -z)
        faces += [3, ct, n_theta + j, n_theta + j1]  # top cap (outward +z)
    surface = pv.PolyData(pts, np.asarray(faces, dtype=np.int64))

    apothem_area = 0.5 * n_theta * radius**2 * math.sin(2.0 * math.pi / n_theta)
    side_len = 2.0 * radius * math.sin(math.pi / n_theta)
    truth = PrimitiveTruth(
        shape="solid_cylinder",
        params={"radius": radius, "height": height, "n_theta": n_theta},
        surface_area=2.0 * apothem_area + n_theta * side_len * height,
        volume=apothem_area * height,
        exact=True,
        n_faces=4 * n_theta,
        notes=[
            f"continuous cylinder: volume {math.pi * radius**2 * height:.10g}, "
            f"area {2 * math.pi * radius * (radius + height):.10g}"
        ],
    )
    return surface, truth


def solid_sphere(
    *,
    radius: float = 5.0,
    n_theta: int = 64,
    n_phi: int = 32,
) -> tuple[pv.PolyData, PrimitiveTruth]:
    """UV sphere centered at the origin; truth is the continuous closed form
    (inscribed mesh converges from below, <=1% at default resolution)."""
    _positive(radius=radius)
    if n_theta < 8 or n_phi < 4:
        raise ValueError(f"need n_theta >= 8 and n_phi >= 4, got {n_theta}, {n_phi}")
    phi = np.linspace(0.0, math.pi, n_phi + 1)[1:-1]  # interior rings
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    rings = []
    for p in phi:
        rings.append(
            np.column_stack(
                [
                    radius * math.sin(p) * np.cos(theta),
                    radius * math.sin(p) * np.sin(theta),
                    np.full(n_theta, radius * math.cos(p)),
                ]
            )
        )
    pts = np.vstack(rings + [[[0, 0, radius]], [[0, 0, -radius]]])
    top, bot = (n_phi - 1) * n_theta, (n_phi - 1) * n_theta + 1

    faces: list[int] = []
    for j in range(n_theta):
        j1 = (j + 1) % n_theta
        faces += [3, top, j, j1]  # top fan (ring 0 nearest +z pole)
        last = (n_phi - 2) * n_theta
        faces += [3, bot, last + j1, last + j]  # bottom fan
    for i in range(n_phi - 2):
        r0, r1 = i * n_theta, (i + 1) * n_theta
        for j in range(n_theta):
            j1 = (j + 1) % n_theta
            faces += [3, r0 + j, r1 + j, r1 + j1, 3, r0 + j, r1 + j1, r0 + j1]
    surface = pv.PolyData(pts, np.asarray(faces, dtype=np.int64))
    truth = PrimitiveTruth(
        shape="solid_sphere",
        params={"radius": radius, "n_theta": n_theta, "n_phi": n_phi},
        surface_area=4.0 * math.pi * radius**2,
        volume=4.0 / 3.0 * math.pi * radius**3,
        exact=False,
        n_faces=2 * n_theta * (n_phi - 1),
        notes=["inscribed UV sphere converges to the closed form from below"],
    )
    return surface, truth


def _as_loop(loop: Any, name: str) -> np.ndarray:
    arr = np.asarray(loop, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 3:
        raise ValueError(f"{name} must be an (N>=3, 2) array, got shape {arr.shape}")
    if np.linalg.norm(arr[0] - arr[-1]) < 1e-12 * (np.abs(arr).max() or 1.0):
        arr = arr[:-1]  # implicit closure, like Profile2D
        if len(arr) < 3:
            raise ValueError(f"{name} degenerates to < 3 distinct vertices")
    return arr


def extrude_loops(
    loops: list[np.ndarray] | np.ndarray,
    *,
    distance: float = 1.0,
) -> tuple[pv.PolyData, PrimitiveTruth]:
    """Extrude planar loops (first = outer CCW, rest = holes CW) from z=0 to
    z=distance. Volume is EXACTLY (sum of signed loop areas) x distance."""
    _positive(distance=distance)
    if isinstance(loops, np.ndarray) and loops.ndim == 2:
        loops = [loops]
    loops2 = [_as_loop(lp, f"loop {i}") for i, lp in enumerate(loops)]
    areas = [_signed_area_2d(lp) for lp in loops2]
    if areas[0] <= 0:
        raise ValueError(f"outer loop must be CCW, signed area {areas[0]:.6g}")
    for i, a in enumerate(areas[1:], start=1):
        if a >= 0:
            raise ValueError(f"hole loop {i} must be CW, signed area {a:.6g}")
    net_area = float(sum(areas))
    if net_area <= 0:
        raise ValueError(f"holes swallow the outer loop (net area {net_area:.6g})")

    counts = [len(lp) for lp in loops2]
    offsets = np.concatenate([[0], np.cumsum(counts)])
    n_pts2d = int(offsets[-1])
    all_xy = np.vstack(loops2)
    bottom = np.column_stack([all_xy, np.zeros(n_pts2d)])
    top = np.column_stack([all_xy, np.full(n_pts2d, float(distance))])
    pts = np.vstack([bottom, top])

    faces: list[int] = []
    for li, lp in enumerate(loops2):
        base = int(offsets[li])
        n = len(lp)
        for j in range(n):
            a, b = base + j, base + (j + 1) % n
            # CCW outer: outward side normals; CW holes: normals into the hole
            # cavity, i.e. out of the solid.
            faces += [3, a, b, n_pts2d + b, 3, a, n_pts2d + b, n_pts2d + a]

    loop_ids = [list(range(int(offsets[i]), int(offsets[i + 1]))) for i in range(len(loops2))]
    cap = _cap_triangles(loops2, loop_ids)
    for a, b, c in cap:
        faces += [3, n_pts2d + a, n_pts2d + b, n_pts2d + c]  # top: +z outward
        faces += [3, c, b, a]  # bottom: -z outward
    surface = pv.PolyData(pts, np.asarray(faces, dtype=np.int64))

    perimeter = sum(
        float(np.linalg.norm(np.roll(lp, -1, axis=0) - lp, axis=1).sum())
        for lp in loops2
    )
    n_holes = len(loops2) - 1
    n_cap_tris = n_pts2d + 2 * n_holes - 2
    truth = PrimitiveTruth(
        shape="extrude_loops",
        params={"n_loops": len(loops2), "n_vertices": n_pts2d, "distance": distance},
        surface_area=2.0 * net_area + perimeter * float(distance),
        volume=net_area * float(distance),
        exact=True,
        n_faces=2 * n_pts2d + 2 * n_cap_tris,
    )
    return surface, truth


def plate_with_hole(
    *,
    width: float = 10.0,
    height: float = 6.0,
    thickness: float = 1.0,
    hole_radius: float = 1.5,
    n_hole: int = 32,
) -> tuple[pv.PolyData, PrimitiveTruth]:
    """Rectangular plate with a centered n-gon hole; exact discrete truth."""
    _positive(width=width, height=height, thickness=thickness, hole_radius=hole_radius)
    if n_hole < 8:
        raise ValueError(f"n_hole must be >= 8, got {n_hole}")
    if 2.0 * hole_radius >= min(width, height):
        raise ValueError(
            f"hole diameter {2 * hole_radius} must be smaller than the plate "
            f"({width} x {height})"
        )
    w, h = width / 2.0, height / 2.0
    outer = np.array([[-w, -h], [w, -h], [w, h], [-w, h]])
    t = np.linspace(0.0, 2.0 * math.pi, n_hole, endpoint=False)
    hole = np.column_stack([hole_radius * np.cos(t), hole_radius * np.sin(t)])[::-1]  # CW
    surface, base = extrude_loops([outer, hole], distance=thickness)

    hole_area = 0.5 * n_hole * hole_radius**2 * math.sin(2.0 * math.pi / n_hole)
    hole_side = 2.0 * hole_radius * math.sin(math.pi / n_hole)
    truth = PrimitiveTruth(
        shape="plate_with_hole",
        params={
            "width": width,
            "height": height,
            "thickness": thickness,
            "hole_radius": hole_radius,
            "n_hole": n_hole,
        },
        surface_area=2.0 * (width * height - hole_area)
        + 2.0 * (width + height) * thickness
        + n_hole * hole_side * thickness,
        volume=(width * height - hole_area) * thickness,
        exact=True,
        n_faces=base.n_faces,
    )
    return surface, truth


def revolve_loop(
    loop: np.ndarray | None = None,
    *,
    angle: float = 2.0 * math.pi,
    n_seg: int = 64,
) -> tuple[pv.PolyData, PrimitiveTruth]:
    """Revolve a planar loop (columns: radial x >= 0, axial z) about the z
    axis by ``angle`` (partial angles get flat end caps). Truth is Pappus on
    the exact loop polygon: V = angle * integral(x dA); mesh converges."""
    if loop is None:  # default: square annulus profile, a torus-like ring
        loop = np.array([[2.0, -0.5], [3.0, -0.5], [3.0, 0.5], [2.0, 0.5]])
    lp = _as_loop(loop, "loop")
    if not 0.0 < angle <= 2.0 * math.pi + 1e-12:
        raise ValueError(f"angle must be in (0, 2*pi], got {angle}")
    full = abs(angle - 2.0 * math.pi) <= 1e-12
    if n_seg < 8:
        raise ValueError(f"n_seg must be >= 8, got {n_seg}")
    if lp[:, 0].min() < 0:
        raise ValueError(
            f"loop must lie in the x >= 0 half-plane, min x = {lp[:, 0].min():.6g}"
        )
    area = _signed_area_2d(lp)
    if area <= 0:
        raise ValueError(f"loop must be CCW (positive area), got {area:.6g}")
    scale = float(np.abs(lp).max()) or 1.0
    if bool((lp[:, 0] < 1e-12 * scale).any()):
        raise ValueError(
            "axis-touching loops (x = 0 vertices) are not supported yet; "
            "offset the profile off the axis"
        )

    n_ring = n_seg if full else n_seg + 1
    phis = np.linspace(0.0, angle, n_ring, endpoint=not full)
    nv = len(lp)
    pts = np.concatenate(
        [
            np.column_stack(
                [lp[:, 0] * math.cos(p), lp[:, 0] * math.sin(p), lp[:, 1]]
            )
            for p in phis
        ]
    )
    faces: list[int] = []
    for k in range(n_seg):
        k1 = (k + 1) % n_ring
        for j in range(nv):
            j1 = (j + 1) % nv
            a, b = k * nv + j, k * nv + j1
            c, d = k1 * nv + j1, k1 * nv + j
            # CCW profile in (x, z) + rotation +phi: (d-a) x (c-d) is outward.
            faces += [3, a, d, c, 3, a, c, b]
    if not full:
        ids0 = list(range(nv))
        tris = _ear_clip_positions(lp, ids0)
        last = (n_ring - 1) * nv
        for a, b, c in tris:
            # CCW-in-(x,z) triangles have -y normals: outward at phi=0.
            faces += [3, a, b, c]
            faces += [3, last + c, last + b, last + a]  # end cap: +phi side
    surface = pv.PolyData(pts, np.asarray(faces, dtype=np.int64))

    # Pappus: first moment about the axis; both exact for the polygon loop.
    x, z = lp[:, 0], lp[:, 1]
    xn, zn = np.roll(x, -1), np.roll(z, -1)
    cross = x * zn - xn * z
    m_x = float(np.sum((x + xn) * cross)) / 6.0  # integral of x dA
    edge_len = np.hypot(xn - x, zn - z)
    lateral = float(np.sum(angle * (x + xn) / 2.0 * edge_len))
    truth = PrimitiveTruth(
        shape="revolve_loop",
        params={"n_vertices": nv, "angle": angle, "n_seg": n_seg},
        surface_area=lateral + (0.0 if full else 2.0 * area),
        volume=angle * m_x,
        exact=False,
        n_faces=2 * nv * n_seg + (0 if full else 2 * (nv - 2)),
        notes=["Pappus on the exact loop polygon; chordal mesh converges from below"],
    )
    return surface, truth


PRIMITIVES: dict[str, Callable[..., tuple[pv.PolyData, PrimitiveTruth]]] = {
    "box": box,
    "solid_cylinder": solid_cylinder,
    "solid_sphere": solid_sphere,
    "plate_with_hole": plate_with_hole,
    "extrude_loops": extrude_loops,
    "revolve_loop": revolve_loop,
}


def generate_primitive(shape: str, **params: Any) -> tuple[pv.PolyData, PrimitiveTruth]:
    """Generate a primitive surface + truth by name (loud on unknowns)."""
    try:
        gen = PRIMITIVES[shape]
    except KeyError:
        raise ValueError(
            f"unknown primitive {shape!r}; available: {', '.join(sorted(PRIMITIVES))}"
        ) from None
    return gen(**params)
