from __future__ import annotations

import math

import numpy as np
import pytest
import pyvista as pv

from meshforge.geometry import compute_centerline
from meshforge.meshing import get_backend
from meshforge.quality.volume import volume_quality
from meshforge.synth import generate

T_WALL = 0.6
N_THETA = 32
N_AXIAL = 60
N_LAYERS = 4


@pytest.fixture(scope="module")
def straight() -> tuple:
    surface, truth = generate("straight_tube")
    cl = compute_centerline(surface)
    return surface, truth, cl


def test_swept_straight_tube_annulus_volume(straight) -> None:
    surface, truth, cl = straight
    wall = get_backend("swept").mesh_wall(
        surface, T_WALL, centerline=cl, n_theta=N_THETA, n_axial=N_AXIAL, n_layers=N_LAYERS
    )
    r = float(truth.radii[0])
    expected = math.pi * ((r + T_WALL) ** 2 - r**2) * cl.length
    assert wall.quality.n_inverted == 0
    assert wall.quality.total_volume == pytest.approx(expected, rel=0.025)
    assert wall.quality.sj_min > 0.3
    assert set(wall.grid.cell_data["region"]) == {1}


def test_swept_boundary_group_counts_exact(straight) -> None:
    surface, _, cl = straight
    wall = get_backend("swept").mesh_wall(
        surface, T_WALL, centerline=cl, n_theta=N_THETA, n_axial=N_AXIAL, n_layers=N_LAYERS
    )
    b = wall.boundary
    s_minus_1 = b["inner"].n_faces // N_THETA
    assert b["inner"].n_faces == s_minus_1 * N_THETA == (N_AXIAL - 1) * N_THETA
    assert b["outer"].n_faces == b["inner"].n_faces
    assert b["inlet"].n_faces == N_THETA * N_LAYERS
    assert b["outlet"].n_faces == N_THETA * N_LAYERS
    # Groups tile the full boundary exactly.
    total = sum(g.n_faces for g in b.values())
    assert total == wall.grid.extract_surface(algorithm="dataset_surface").n_cells
    # Face nodes reference real hex corners of their parent cells.
    for g in b.values():
        cells = wall.grid.cells.reshape(-1, 9)[:, 1:]
        for cid, fn in zip(g.cell_ids[:5], g.face_nodes[:5]):
            assert set(fn).issubset(set(cells[cid]))


def test_swept_curved_tube_no_inversion() -> None:
    surface, _ = generate("curved_tube")
    cl = compute_centerline(surface)
    wall = get_backend("swept").mesh_wall(surface, 0.4, centerline=cl)
    assert wall.quality.n_inverted == 0
    # V4 cap extension: the wall now covers the near-cap regions, whose
    # in-plane rays skim the cap disks — valid but lower-quality cells
    # (measured sj_min 0.119 at 32x60x4; was >0.2 when the wall stopped a
    # radius short of each cap).
    assert wall.quality.sj_min > 0.1
    r = 2.0
    expected = math.pi * ((r + 0.4) ** 2 - r**2) * cl.length
    assert wall.quality.total_volume == pytest.approx(expected, rel=0.04)


def test_swept_accepts_closed_centerline() -> None:
    # Pre-V9 this raised "closed centerline ... not supported"; closed loops
    # now sweep periodically (full coverage in test_closedloop.py).
    surface, _ = generate("torus")
    wall = get_backend("swept").mesh_wall(
        surface, 0.3, n_theta=12, n_axial=24, n_layers=1
    )
    assert wall.quality.n_inverted == 0
    assert set(wall.boundary) == {"inner", "outer"}


def test_collapsed_hex_counts_as_inverted() -> None:
    # Trap #13: a degenerate cell must hit the inverted count, not the stats.
    pts = np.zeros((8, 3))  # fully collapsed hex
    cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    grid = pv.UnstructuredGrid(cells, np.array([pv.CellType.HEXAHEDRON], np.uint8), pts)
    q = volume_quality(grid)
    assert q.n_cells == 1
    assert q.n_inverted == 1
    assert q.total_volume == 0.0
    assert not math.isinf(q.sj_mean if math.isnan(q.sj_mean) is False else 0.0)


def test_meshing_registry_and_guards(straight) -> None:
    surface, _, cl = straight
    with pytest.raises(ValueError, match="unknown meshing backend"):
        get_backend("magic")
    with pytest.raises(ValueError, match="thickness must be positive"):
        get_backend("swept").mesh_wall(surface, 0.0, centerline=cl)
    with pytest.raises(ValueError, match="lattice too coarse"):
        get_backend("swept").mesh_wall(surface, 0.5, centerline=cl, n_theta=4)
