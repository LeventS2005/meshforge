from __future__ import annotations

import math

import numpy as np
import pytest

from meshforge.synth import GENERATORS, generate

# Every shape must produce a watertight, all-triangle, outward-oriented
# surface whose mesh volume/area track the analytic truth. Tolerances are
# discretization-driven: inscribed-polygon error ~ (2π/n_theta)²/6 plus
# axial chord error, all well under these bounds at default resolutions.
VOLUME_RTOL = {
    "straight_tube": 0.005,
    "curved_tube": 0.01,
    "tapered_tube": 0.005,
    "constricted_tube": 0.01,
    "torus": 0.01,
    "aneurysm_tube": 0.005,
}
AREA_RTOL = 0.01


@pytest.mark.parametrize("shape", sorted(GENERATORS))
def test_generated_surfaces_are_mesh_ready(shape: str) -> None:
    surface, truth = generate(shape)

    assert surface.n_open_edges == 0
    assert surface.is_manifold
    assert surface.volume > 0  # outward orientation, not just |V|

    from meshforge.quality import count_triangles

    assert count_triangles(surface) == surface.n_cells

    if truth.volume is not None:
        assert abs(surface.volume - truth.volume) / truth.volume < VOLUME_RTOL[shape]
    if truth.surface_area is not None:
        assert abs(surface.area - truth.surface_area) / truth.surface_area < AREA_RTOL


@pytest.mark.parametrize("shape", sorted(GENERATORS))
def test_truth_is_selfconsistent(shape: str) -> None:
    _, truth = generate(shape)

    np.testing.assert_allclose(np.linalg.norm(truth.tangents, axis=1), 1.0, atol=1e-12)
    assert np.all(truth.radii > 0)
    assert np.all(np.diff(truth.arc_lengths) > 0)
    assert truth.arc_lengths[0] == 0.0
    assert math.isclose(truth.arc_lengths[-1], truth.length, rel_tol=1e-12)

    # Finite-difference of truth points must match stated tangents/arc length.
    d = np.diff(truth.points, axis=0)
    seg = np.linalg.norm(d, axis=1)
    np.testing.assert_allclose(seg, np.diff(truth.arc_lengths), rtol=5e-4)
    mid_tangents = 0.5 * (truth.tangents[:-1] + truth.tangents[1:])
    cosang = np.einsum("ij,ij->i", d / seg[:, None], mid_tangents / np.linalg.norm(mid_tangents, axis=1)[:, None])
    assert cosang.min() > 0.99999


def test_face_count_exactness() -> None:
    # Capped tube: 2·n_theta·(n_axial−1) side triangles + 2·n_theta cap fans.
    surface, _ = generate("straight_tube", n_theta=32, n_axial=11)
    assert surface.n_cells == 2 * 32 * 10 + 2 * 32
    # Torus: 2·n_theta·n_axial, no caps.
    surface, _ = generate("torus", n_theta=24, n_axial=48)
    assert surface.n_cells == 2 * 24 * 48


def test_constricted_narrows_at_middle() -> None:
    _, truth = generate("constricted_tube", depth=0.4)
    mid = len(truth.radii) // 2
    assert truth.radii[mid] == pytest.approx(3.0 * 0.6, rel=1e-9)
    assert truth.radii[0] == pytest.approx(3.0, rel=1e-4)
    assert np.argmin(truth.radii) == mid


@pytest.mark.parametrize(
    ("shape", "bad", "match"),
    [
        ("curved_tube", {"radius": 12.0}, "self-intersect"),
        ("torus", {"tube_radius": 7.0}, "must be <"),
        ("constricted_tube", {"depth": 1.5}, "open lumen"),
        ("straight_tube", {"radius": -1.0}, "must be positive"),
        ("straight_tube", {"n_theta": 4}, "n_theta"),
    ],
)
def test_degenerate_params_fail_loudly(shape: str, bad: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        generate(shape, **bad)


def test_unknown_shape_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown shape"):
        generate("klein_bottle")
