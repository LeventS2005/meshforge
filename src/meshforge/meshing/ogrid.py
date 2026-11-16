"""O-grid ("butterfly") block layout for disk-like profiles (V26).

The V6 projection method meshes a disk by snapping a square proxy's rim
onto the circle — the four proxy corners shear their neighbourhoods, and
the quality floor drops with refinement until the gate refuses (sj≈0.05
by target_size≈0.1 on the unit circle, D37). The classic O-grid replaces
that layout with a central square block plus four ring blocks whose outer
edges lie exactly on the true curve: the worst corner is the fixed 135°
spoke/core junction, so min scaled Jacobian is set by the LAYOUT
(measured 0.7071 on a circle), invariant under refinement.

Scope (honest): ONE closed CCW loop, star-shaped about its area centroid.
No holes (an annulus O-grid is a different block graph — the swept mesher
covers annuli), no multi-loop profiles. The result is a ``BlockMesh2D``
whose global rectilinear-lattice fields are EMPTY — an O-grid is not on
the V5 lattice, so elliptic smoothing refuses it loudly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def _signed_area(loop: np.ndarray) -> float:
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


@dataclass
class BlockMesh2D:
    """Structured multi-block quad mesh on the global fine lattice.

    The lattice fields (node_id grid, fine ticks, coarse->fine maps, active
    mask, source loops) let the V6 projection method re-blend block
    interiors by TFI after boundary nodes move.
    """

    points: np.ndarray  # (N, 2)
    quads: np.ndarray  # (M, 4) CCW
    block_of: np.ndarray  # (M,) coarse block id per quad
    n_blocks: int
    edge_groups: dict[str, np.ndarray]  # name -> (k, 2) node-id edges
    node_id: np.ndarray  # (nfx, nfy) global node ids, -1 where unused
    first_fx: np.ndarray  # coarse tick k -> fine index of xs[k]
    first_fy: np.ndarray
    active: np.ndarray  # (n_coarse_x, n_coarse_y) material blocks
    loops: list[np.ndarray] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def quad_scaled_jacobians(points: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Per-quad min corner scaled Jacobian (2-D); negative = inverted."""
    p = points[quads]  # (M, 4, 2)
    sj = np.full(len(quads), np.inf)
    for k in range(4):
        e1 = p[:, (k + 1) % 4] - p[:, k]
        e2 = p[:, (k - 1) % 4] - p[:, k]
        cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        n1 = np.linalg.norm(e1, axis=1)
        n2 = np.linalg.norm(e2, axis=1)
        denom = np.where((n1 > 0) & (n2 > 0), n1 * n2, 1.0)
        corner = np.where((n1 > 0) & (n2 > 0), cross / denom, -1.0)
        sj = np.minimum(sj, corner)
    return sj


def _polygon_centroid(loop: np.ndarray) -> np.ndarray:
    """Area centroid of a simple CCW polygon."""
    x, y = loop[:, 0], loop[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = cross.sum() / 2.0
    return np.array(
        [((x + xn) * cross).sum() / (6.0 * area),
         ((y + yn) * cross).sum() / (6.0 * area)]
    )


def _require_star_shaped(loop: np.ndarray, centroid: np.ndarray) -> None:
    """Refuse loops whose vertices do not wind monotonically about the
    centroid — spokes and rim samples of a non-star-shaped loop would
    cross the boundary."""
    ang = np.arctan2(loop[:, 1] - centroid[1], loop[:, 0] - centroid[0])
    inc = np.diff(np.concatenate([ang, ang[:1]]))
    inc = np.mod(inc, 2.0 * np.pi)  # CCW increments in [0, 2*pi)
    if bool((inc <= 0).any()) or bool((inc >= np.pi).any()) or not math.isclose(
        float(inc.sum()), 2.0 * math.pi, rel_tol=1e-9
    ):
        k = int(np.argmax(inc)) if (inc >= np.pi).any() else int(np.argmin(inc))
        raise ValueError(
            f"loop is not star-shaped about its centroid (vertex {k} winds "
            "backward or degenerately) — the O-grid needs every rim point "
            "visible from the centroid"
        )


def _tfi_fill(grid_pts: np.ndarray) -> None:
    """Fill the interior of a (nu, nv, 2) patch by transfinite
    interpolation from its boundary rows/columns (in place)."""
    nu, nv = grid_pts.shape[:2]
    if nu < 3 or nv < 3:
        return
    u = np.linspace(0.0, 1.0, nu)[:, None, None]
    v = np.linspace(0.0, 1.0, nv)[None, :, None]
    e_w = grid_pts[0][None, :, :]
    e_e = grid_pts[-1][None, :, :]
    e_s = grid_pts[:, 0][:, None, :]
    e_n = grid_pts[:, -1][:, None, :]
    tfi = (
        (1 - u) * e_w
        + u * e_e
        + (1 - v) * e_s
        + v * e_n
        - (
            (1 - u) * (1 - v) * grid_pts[0, 0]
            + u * (1 - v) * grid_pts[-1, 0]
            + (1 - u) * v * grid_pts[0, -1]
            + u * v * grid_pts[-1, -1]
        )
    )
    grid_pts[1:-1, 1:-1] = tfi[1:-1, 1:-1]


def ogrid_disk(
    loops: list[np.ndarray],
    *,
    target_size: float,
    core_fraction: float = 0.5,
    min_sj: float = 0.05,
    tol: float | None = None,
    n_t: int | None = None,
    n_r: int | None = None,
) -> BlockMesh2D:
    """Butterfly-mesh a single star-shaped loop: central square block +
    4 ring blocks with rims exactly on the curve.

    ``loops`` follows the Profile2D convention but must hold EXACTLY one
    CCW loop — holes are refused (disk topology only). Core corners sit
    at ``centroid + core_fraction * (rim(t_q) - centroid)`` for the four
    quarter-arc-length parameters t_q (t_0 = the loop's first vertex).
    Counts: n_t = ceil(quarter arc / target_size) along each quadrant,
    n_r = ceil(max core-edge-to-rim gap / target_size) radially; explicit
    ``n_t``/``n_r`` overrides pin them instead (the swept lumen fill, V47,
    needs an IDENTICAL block topology at every station). Raises loudly on
    inverted quads or min scaled Jacobian < ``min_sj``.
    """
    if not target_size > 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    if n_t is not None and n_t < 2:
        raise ValueError(f"n_t override must be >= 2, got {n_t}")
    if n_r is not None and n_r < 1:
        raise ValueError(f"n_r override must be >= 1, got {n_r}")
    if not 0.0 < core_fraction < 1.0:
        raise ValueError(
            f"core_fraction must be in (0, 1), got {core_fraction}"
        )
    loops = [np.asarray(lp, dtype=float) for lp in loops]
    if len(loops) != 1:
        raise ValueError(
            f"the O-grid disk layout takes exactly one loop (no holes), "
            f"got {len(loops)} — annuli/holed profiles need the swept or "
            "projection meshers"
        )
    loop = loops[0]
    if len(loop) < 4:
        raise ValueError(f"loop needs at least 4 vertices, got {len(loop)}")
    if _signed_area(loop) <= 0:
        raise ValueError("outer loop must be CCW (positive signed area)")
    if tol is None:
        diag = float(np.hypot(*(loop.max(axis=0) - loop.min(axis=0))))
        tol = 1e-6 * diag

    centroid = _polygon_centroid(loop)
    _require_star_shaped(loop, centroid)

    # Arc-length parameterization of the rim.
    seg = np.roll(loop, -1, axis=0) - loop
    seglen = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total_len = float(cum[-1])
    if not total_len > 0:
        raise ValueError("degenerate loop (zero perimeter)")

    def rim_at(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float) % total_len
        k = np.clip(
            np.searchsorted(cum, t, side="right") - 1, 0, len(loop) - 1
        )
        frac = (t - cum[k]) / np.where(seglen[k] == 0.0, 1.0, seglen[k])
        return loop[k] + frac[:, None] * seg[k]

    quarter = total_len / 4.0
    t_corner = np.arange(4) * quarter
    rim_corners = rim_at(t_corner)
    core_corners = centroid + core_fraction * (rim_corners - centroid)

    if n_t is None:
        n_t = max(2, math.ceil(quarter / target_size))
    a_frac = np.linspace(0.0, 1.0, n_t + 1)
    if n_r is None:
        gap = 0.0
        for q in range(4):
            inner = (1 - a_frac)[:, None] * core_corners[q] + a_frac[:, None] * core_corners[(q + 1) % 4]
            outer = rim_at(t_corner[q] + a_frac * quarter)
            gap = max(gap, float(np.linalg.norm(outer - inner, axis=1).max()))
        n_r = max(1, math.ceil(gap / target_size))

    # --- global nodes: core lattice, then 4 spokes, then 4 ring patches.
    n1 = n_t + 1
    n_points = n1 * n1 + 4 * n_r + 4 * (n_t - 1) * n_r
    points = np.zeros((n_points, 2))
    core_id = np.arange(n1 * n1, dtype=np.int64).reshape(n1, n1)
    grid = np.zeros((n1, n1, 2))
    for edge, (c_a, c_b) in enumerate(
        [(0, 1), (1, 2), (2, 3), (3, 0)]
    ):
        line = (1 - a_frac)[:, None] * core_corners[c_a] + a_frac[:, None] * core_corners[c_b]
        if edge == 0:
            grid[:, 0] = line  # C0 -> C1
        elif edge == 1:
            grid[n_t, :] = line  # C1 -> C2
        elif edge == 2:
            grid[::-1, n_t] = line  # C2 -> C3
        else:
            grid[0, ::-1] = line  # C3 -> C0
    _tfi_fill(grid)  # straight edges: TFI == the bilinear tensor grid
    points[: n1 * n1] = grid.reshape(-1, 2)
    next_id = n1 * n1

    # Core boundary rows in ring order (corner q -> corner q+1).
    core_edge = [core_id[:, 0], core_id[n_t, :], core_id[::-1, n_t], core_id[0, ::-1]]

    r_frac = np.linspace(0.0, 1.0, n_r + 1)
    spoke = []
    for q in range(4):
        pts = (1 - r_frac)[1:, None] * core_corners[q] + r_frac[1:, None] * rim_corners[q]
        ids = np.arange(next_id, next_id + n_r, dtype=np.int64)
        points[ids] = pts
        next_id += n_r
        spoke.append(np.concatenate([[core_edge[q][0]], ids]))

    quads = [
        [core_id[i, j], core_id[i + 1, j], core_id[i + 1, j + 1], core_id[i, j + 1]]
        for i in range(n_t)
        for j in range(n_t)
    ]
    block_of = [0] * (n_t * n_t)
    outer_rim = []
    for q in range(4):
        ring = np.zeros((n_t + 1, n_r + 1), dtype=np.int64)
        ring[:, 0] = core_edge[q]
        ring[0, :] = spoke[q]
        ring[n_t, :] = spoke[(q + 1) % 4]
        new_ids = np.arange(
            next_id, next_id + (n_t - 1) * n_r, dtype=np.int64
        ).reshape(n_t - 1, n_r)
        next_id += (n_t - 1) * n_r
        ring[1:n_t, 1:] = new_ids
        # Outer arc nodes lie exactly ON the loop (equal arc length).
        points[ring[1:n_t, n_r]] = rim_at(
            t_corner[q] + a_frac[1:n_t] * quarter
        )
        patch = points[ring].copy()
        _tfi_fill(patch)
        points[ring[1:n_t, 1:n_r]] = patch[1:n_t, 1:n_r]
        # (a, r) is a left-handed frame (a CCW along the rim, r outward):
        # traverse (a,r) -> (a,r+1) -> (a+1,r+1) -> (a+1,r) for CCW quads.
        for a in range(n_t):
            for r in range(n_r):
                quads.append(
                    [ring[a, r], ring[a, r + 1], ring[a + 1, r + 1], ring[a + 1, r]]
                )
            block_of.extend([q + 1] * n_r)
        outer_rim.extend((int(ring[a, n_r]), int(ring[a + 1, n_r])) for a in range(n_t))

    quads_arr = np.asarray(quads, dtype=np.int64)

    sj = quad_scaled_jacobians(points, quads_arr)
    n_inverted = int((sj <= 0).sum())
    if n_inverted:
        raise ValueError(
            f"O-grid produced {n_inverted} inverted quad(s) (worst sj "
            f"{float(sj.min()):.4g}); the loop is too far from a disk — "
            "adjust core_fraction or use the projection mesher"
        )
    if float(sj.min()) < min_sj:
        raise ValueError(
            f"O-grid quality below floor: min scaled Jacobian "
            f"{float(sj.min()):.4g} < {min_sj}; adjust core_fraction"
        )

    notes = [
        f"O-grid: 5 blocks (core + 4 rings), {len(quads_arr)} quads, "
        f"{n_points} nodes, n_t {n_t} per quadrant, n_r {n_r} radial, "
        f"target_size {target_size:.6g}, sj_min {float(sj.min()):.4g}"
    ]
    return BlockMesh2D(
        points=points,
        quads=quads_arr,
        block_of=np.asarray(block_of, dtype=np.int64),
        n_blocks=5,
        edge_groups={
            "outer_rim": np.asarray(outer_rim, dtype=np.int64).reshape(-1, 2),
            "hole_rims": np.zeros((0, 2), dtype=np.int64),
        },
        # An O-grid is not on the V5 global rectilinear lattice: the
        # lattice fields stay EMPTY and lattice consumers (elliptic
        # smoothing, TFI re-blend) refuse loudly.
        node_id=np.full((0, 0), -1, dtype=np.int64),
        first_fx=np.zeros(0, dtype=np.int64),
        first_fy=np.zeros(0, dtype=np.int64),
        active=np.zeros((0, 0), dtype=bool),
        loops=[loop],
        notes=notes,
    )
