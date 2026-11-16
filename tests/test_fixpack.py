"""V4 fix pack: centerline cap extension, pv shim."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import pyvista as pv

from meshforge.geometry import compute_centerline
from meshforge.meshing import get_backend
from meshforge.synth import generate

# ---------------------------------------------------------------------------
# fix 1: centerline cap extension (BACKLOG #1)


def test_cap_extension_reaches_flat_caps() -> None:
    surface, truth = generate("straight_tube")  # r=3, L=20
    cl = compute_centerline(surface)  # extend_to_caps default ON

    r = float(truth.radii[0])
    assert cl.length >= truth.length - 2.0 * (0.02 * r + 1.5 * cl.voxel_size)
    assert cl.length <= 1.005 * truth.length
    ext_notes = [n for n in cl.notes if "cap extension" in n]
    assert len(ext_notes) == 2 and all("toward cap" in n for n in ext_notes)
    # Extended points stay on the axis; arc sampling stays uniform.
    assert np.linalg.norm(cl.points[:, :2], axis=1).max() < 1.5 * cl.voxel_size
    steps = np.diff(cl.arc_lengths)
    np.testing.assert_allclose(steps, steps[0], rtol=1e-9)


def test_cap_extension_noop_on_closed_loop() -> None:
    surface, _ = generate("torus")
    cl = compute_centerline(surface)
    assert cl.closed
    assert not any("cap extension" in n for n in cl.notes)


def test_cap_extension_tol_and_standoff_branches() -> None:
    """Trap #3: the kept-endpoint branches must be exercised, not assumed.
    An impossible normal tolerance forces the side-wall branch; an enormous
    standoff forces the within-standoff branch — both keep old endpoints."""
    surface, truth = generate("straight_tube")
    base = compute_centerline(surface, extend_to_caps=False)

    side = compute_centerline(surface, cap_normal_tol=1.01)
    assert side.length == pytest.approx(base.length, rel=1e-6)
    assert sum("side wall" in n for n in side.notes) == 2

    near = compute_centerline(surface, cap_standoff=5.0)
    assert near.length == pytest.approx(base.length, rel=1e-6)
    assert sum("within standoff" in n for n in near.notes) == 2


def test_swept_annulus_volume_vs_full_analytic_length() -> None:
    """The point of the fix: the swept wall now covers the FULL tube, so the
    annulus volume is checked against the full analytic length (pre-V4 the
    wall missed ~2 radii of axial coverage)."""
    surface, truth = generate("straight_tube")
    cl = compute_centerline(surface)
    wall = get_backend("swept").mesh_wall(
        surface, 0.6, centerline=cl, n_theta=24, n_axial=30, n_layers=3
    )
    r = float(truth.radii[0])
    full = math.pi * ((r + 0.6) ** 2 - r**2) * truth.length
    assert wall.quality.n_inverted == 0
    assert wall.quality.total_volume == pytest.approx(full, rel=0.025)


def test_swept_curved_extended_zero_inverted() -> None:
    surface, _ = generate("curved_tube")
    cl = compute_centerline(surface)
    wall = get_backend("swept").mesh_wall(
        surface, 0.4, centerline=cl, n_theta=24, n_axial=40, n_layers=3
    )
    assert wall.quality.n_inverted == 0
    assert wall.quality.sj_min > 0.1


# ---------------------------------------------------------------------------
# fix 3: pyvista shim (BACKLOG #5)


def test_interior_point_mask_matches_deprecated_filter() -> None:
    from meshforge.geometry.compat import interior_point_mask

    sphere = pv.Sphere(radius=5.0)
    # Off-lattice origin: points exactly ON the surface are legitimate
    # tie-breaks where ray-cast and signed-distance methods may disagree.
    grid = pv.ImageData(
        dimensions=(15, 15, 15), spacing=(1.0,) * 3, origin=(-7.3, -7.15, -6.9)
    )
    mask = interior_point_mask(grid, sphere)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = grid.select_enclosed_points(sphere, tolerance=0.0, check_surface=True)
    legacy = np.asarray(old["SelectedPoints"]).astype(bool)
    assert mask.dtype == bool
    assert np.array_equal(mask, legacy)
    assert 0 < mask.sum() < grid.n_points


def test_centerline_emits_no_deprecation_warning() -> None:
    surface, _ = generate("straight_tube")
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=Warning)
        # Only the pyvista deprecation matters here; numpy/vtk noise is
        # unrelated third-party churn.
        warnings.filterwarnings("default", module=r".*numpy_support.*")
        warnings.filterwarnings("default", category=DeprecationWarning)
        try:
            from pyvista.core.errors import PyVistaDeprecationWarning

            warnings.filterwarnings("error", category=PyVistaDeprecationWarning)
        except ImportError:
            pass
        compute_centerline(surface)
