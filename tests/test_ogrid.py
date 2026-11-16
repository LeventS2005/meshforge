"""V26 O-grid (butterfly) disk layout: area truths, the quality floor,
exact conformity and loud refusals."""
from __future__ import annotations


import numpy as np
import pytest

from meshforge.meshing import ogrid_disk, quad_scaled_jacobians
from meshforge.meshing.ogrid import _signed_area


def _circle(n: int = 128, r: float = 1.0) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


# A square with a narrow off-center slit cut toward the interior: vertex
# (1, 0.3) winds BACKWARD about the centroid -> not star-shaped.
NONSTAR = np.array(
    [[0, 0], [2, 0], [2, 0.2], [1.0, 0.3], [2, 0.4], [2, 2], [0, 2]],
    dtype=float,
)


def _mesh_area(bm) -> float:
    p = bm.points[bm.quads]
    a1 = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (
        p[:, 2, 0] - p[:, 0, 0]
    ) * (p[:, 1, 1] - p[:, 0, 1])
    a2 = (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 3, 1] - p[:, 0, 1]) - (
        p[:, 3, 0] - p[:, 0, 0]
    ) * (p[:, 2, 1] - p[:, 0, 1])
    return float((a1 + a2).sum() / 2.0)


# ---------------------------------------------------------------------------
# truths


def test_disk_area_converges():
    loop = _circle()
    poly_area = _signed_area(loop)  # the 128-gon truth the rim lies on
    bm_1 = ogrid_disk([loop], target_size=0.1)
    area_1 = _mesh_area(bm_1)
    assert abs(area_1 - np.pi) / np.pi <= 0.005  # measured -0.161%

    bm_2 = ogrid_disk([loop], target_size=0.05)
    area_2 = _mesh_area(bm_2)
    # refinement converges toward the polygon area (-0.040% vs pi)
    assert abs(area_2 - np.pi) < abs(area_1 - np.pi)
    assert abs(area_2 - poly_area) / poly_area <= 1e-3


def test_quality_floor():
    loop = _circle()
    # O-grid sj_min is set by the layout (135 deg spoke/core corner):
    # measured 0.7071 at every target size.
    for ts in (0.15, 0.1, 0.05):
        sj = quad_scaled_jacobians(*_pts_quads(ogrid_disk([loop], target_size=ts)))
        assert float(sj.min()) >= 0.70


def _pts_quads(bm):
    return bm.points, bm.quads


def test_conformity_exact_and_rim_on_curve():
    loop = _circle()
    bm = ogrid_disk([loop], target_size=0.1)

    counts: dict[tuple[int, int], int] = {}
    for quad in bm.quads:
        for e in range(4):
            a, b = int(quad[e]), int(quad[(e + 1) % 4])
            counts[(min(a, b), max(a, b))] = counts.get((min(a, b), max(a, b)), 0) + 1
    boundary = {k for k, c in counts.items() if c == 1}
    assert not any(c > 2 for c in counts.values())
    # every boundary edge is an outer_rim edge, and vice versa
    rim_edges = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in bm.edge_groups["outer_rim"]
    }
    assert boundary == rim_edges
    assert len(bm.edge_groups["hole_rims"]) == 0
    assert bm.n_blocks == 5
    assert set(np.unique(bm.block_of)) == {0, 1, 2, 3, 4}

    # no duplicate points anywhere (shared edges are shared IDS)
    assert len(np.unique(np.round(bm.points, 12), axis=0)) == len(bm.points)

    # rim nodes lie exactly on the input polygon (within chord sagitta)
    rim_nodes = np.unique(bm.edge_groups["outer_rim"])
    r = np.linalg.norm(bm.points[rim_nodes], axis=1)
    sagitta = 1.0 * (1.0 - np.cos(np.pi / 128))
    assert np.all(r <= 1.0 + 1e-9)
    assert np.all(r >= 1.0 - sagitta - 1e-9)


# ---------------------------------------------------------------------------
# loud refusals


def test_refusals_are_loud():
    circle = _circle(64)
    with pytest.raises(ValueError, match="exactly one loop"):
        ogrid_disk([circle, circle[::-1] * 0.3], target_size=0.2)
    with pytest.raises(ValueError, match="CCW"):
        ogrid_disk([circle[::-1].copy()], target_size=0.2)
    with pytest.raises(ValueError, match="star-shaped"):
        ogrid_disk([NONSTAR], target_size=0.2)
    with pytest.raises(ValueError, match="core_fraction"):
        ogrid_disk([circle], target_size=0.2, core_fraction=1.0)
    with pytest.raises(ValueError, match="core_fraction"):
        ogrid_disk([circle], target_size=0.2, core_fraction=0.0)
    with pytest.raises(ValueError, match="target_size"):
        ogrid_disk([circle], target_size=0.0)
    with pytest.raises(ValueError, match="at least 4"):
        ogrid_disk([np.array([[0, 0], [1, 0], [0, 1]], float)], target_size=0.2)
