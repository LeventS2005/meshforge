"""V3 geometry library: watertight primitives with exact/convergent truths."""
from __future__ import annotations

import math

import numpy as np
import pytest

from meshforge.synth import PRIMITIVES, generate_primitive
from meshforge.synth.primitives import (
    box,
    extrude_loops,
    plate_with_hole,
    revolve_loop,
    solid_cylinder,
    solid_sphere,
)


def _assert_mesh_ready(surface) -> None:
    assert surface.n_open_edges == 0
    assert surface.is_manifold
    assert surface.volume > 0  # outward orientation, not just |V|
    from meshforge.quality import count_triangles

    assert count_triangles(surface) == surface.n_cells


# extrude_loops has no default loops (it is the Profile2D bridge) — covered
# by its dedicated tests below.
@pytest.mark.parametrize("shape", [s for s in sorted(PRIMITIVES) if s != "extrude_loops"])
def test_default_primitives_are_mesh_ready(shape: str) -> None:
    surface, truth = generate_primitive(shape)
    _assert_mesh_ready(surface)
    assert surface.n_cells == truth.n_faces  # exact face-count promise


def test_box_exact() -> None:
    surface, truth = box(width=4.0, height=3.0, depth=2.0)
    _assert_mesh_ready(surface)
    assert surface.n_cells == 12
    assert surface.volume == pytest.approx(24.0, rel=1e-12)
    assert surface.area == pytest.approx(2 * (12 + 8 + 6), rel=1e-12)
    assert truth.volume == 24.0 and truth.exact


def test_solid_cylinder_inscribed_exact() -> None:
    surface, truth = solid_cylinder(radius=3.0, height=10.0, n_theta=24)
    _assert_mesh_ready(surface)
    assert surface.n_cells == 4 * 24
    # Truth is the DISCRETE n-gon prism: mesh matches to roundoff...
    assert surface.volume == pytest.approx(truth.volume, rel=1e-9)
    assert surface.area == pytest.approx(truth.surface_area, rel=1e-9)
    # ... and sits below the continuous cylinder (inscribed).
    assert truth.volume < math.pi * 9 * 10
    assert truth.volume == pytest.approx(math.pi * 9 * 10, rel=0.03)


def test_solid_sphere_within_one_percent() -> None:
    surface, truth = solid_sphere()  # radius 5, 64 x 32
    _assert_mesh_ready(surface)
    assert surface.n_cells == truth.n_faces == 2 * 64 * 31
    assert not truth.exact
    assert surface.volume < truth.volume  # inscribed: converges from below
    assert surface.volume == pytest.approx(truth.volume, rel=0.01)
    assert surface.area == pytest.approx(truth.surface_area, rel=0.01)


def test_plate_with_hole_exact() -> None:
    surface, truth = plate_with_hole(
        width=10.0, height=6.0, thickness=1.0, hole_radius=1.5, n_hole=16
    )
    _assert_mesh_ready(surface)
    hole_area = 0.5 * 16 * 1.5**2 * math.sin(2 * math.pi / 16)
    assert truth.volume == pytest.approx((60 - hole_area) * 1.0, rel=1e-12)
    assert surface.volume == pytest.approx(truth.volume, rel=1e-9)
    assert surface.area == pytest.approx(truth.surface_area, rel=1e-9)
    assert surface.n_cells == truth.n_faces
    with pytest.raises(ValueError, match="hole diameter"):
        plate_with_hole(hole_radius=5.0, width=6.0, height=6.0)


def test_extrude_loops_exact_volume_and_counts() -> None:
    outer = np.array([[0, 0], [10, 0], [10, 8], [0, 8]], dtype=float)
    hole = np.array([[3, 3], [3, 6], [7, 6], [7, 3]], dtype=float)  # CW
    surface, truth = extrude_loops([outer, hole], distance=2.0)
    _assert_mesh_ready(surface)
    assert truth.volume == pytest.approx((80 - 12) * 2.0, rel=1e-12)
    assert surface.volume == pytest.approx(truth.volume, rel=1e-9)
    # V=8 vertices, H=1 hole: sides 2*8, caps 2*(8 + 2 - 2)
    assert surface.n_cells == truth.n_faces == 2 * 8 + 2 * 8
    assert surface.area == pytest.approx(truth.surface_area, rel=1e-9)


def test_extrude_loops_two_holes() -> None:
    outer = np.array([[0, 0], [20, 0], [20, 8], [0, 8]], dtype=float)
    h1 = np.array([[2, 2], [2, 6], [6, 6], [6, 2]], dtype=float)
    h2 = np.array([[12, 2], [12, 6], [16, 6], [16, 2]], dtype=float)
    surface, truth = extrude_loops([outer, h1, h2], distance=1.5)
    _assert_mesh_ready(surface)
    assert surface.volume == pytest.approx((160 - 16 - 16) * 1.5, rel=1e-9)
    assert surface.n_cells == truth.n_faces == 2 * 12 + 2 * (12 + 4 - 2)


def test_extrude_loops_orientation_errors_loud() -> None:
    outer_cw = np.array([[0, 0], [0, 8], [10, 8], [10, 0]], dtype=float)
    with pytest.raises(ValueError, match="CCW"):
        extrude_loops([outer_cw], distance=1.0)
    outer = outer_cw[::-1].copy()
    hole_ccw = np.array([[3, 3], [7, 3], [7, 6], [3, 6]], dtype=float)
    with pytest.raises(ValueError, match="CW"):
        extrude_loops([outer, hole_ccw], distance=1.0)
    with pytest.raises(ValueError, match="positive"):
        extrude_loops([outer], distance=0.0)
    with pytest.raises(ValueError, match=r"\(N>=3, 2\)"):
        extrude_loops([np.zeros((2, 2))], distance=1.0)


def test_revolve_loop_full_torus_square() -> None:
    surface, truth = revolve_loop()  # square annulus at x in [2, 3], n_seg=64
    _assert_mesh_ready(surface)
    assert truth.volume == pytest.approx(2 * math.pi * 2.5, rel=1e-12)  # Pappus
    assert surface.volume < truth.volume  # chordal mesh from below
    assert surface.volume == pytest.approx(truth.volume, rel=0.01)
    assert surface.area == pytest.approx(truth.surface_area, rel=0.01)
    assert surface.n_cells == truth.n_faces == 2 * 4 * 64


def test_revolve_loop_partial_angle_capped() -> None:
    lp = np.array([[2.0, -0.5], [3.0, -0.5], [3.0, 0.5], [2.0, 0.5]])
    surface, truth = revolve_loop(lp, angle=math.pi, n_seg=32)
    _assert_mesh_ready(surface)  # caps close the half-ring
    assert truth.volume == pytest.approx(math.pi * 2.5, rel=1e-12)
    assert surface.volume == pytest.approx(truth.volume, rel=0.01)
    assert surface.n_cells == truth.n_faces == 2 * 4 * 32 + 2 * (4 - 2)
    # Pappus lateral + two flat caps of the loop area
    assert truth.surface_area == pytest.approx(math.pi * (2.5 + 3 + 2.5 + 2) + 2 * 1.0, rel=1e-12)


def test_revolve_loop_degenerate_params_loud() -> None:
    lp = np.array([[2.0, -0.5], [3.0, -0.5], [3.0, 0.5], [2.0, 0.5]])
    with pytest.raises(ValueError, match="x >= 0"):
        revolve_loop(np.array([[-1.0, 0.0], [2.0, 0.0], [2.0, 1.0]]))
    with pytest.raises(ValueError, match="CCW"):
        revolve_loop(lp[::-1].copy())
    with pytest.raises(ValueError, match="angle"):
        revolve_loop(lp, angle=0.0)
    with pytest.raises(ValueError, match="angle"):
        revolve_loop(lp, angle=7.0)
    with pytest.raises(ValueError, match="axis-touching"):
        revolve_loop(np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="n_seg"):
        revolve_loop(lp, n_seg=4)


def test_degenerate_primitive_params_loud() -> None:
    with pytest.raises(ValueError, match="width"):
        box(width=0.0)
    with pytest.raises(ValueError, match="n_theta"):
        solid_cylinder(n_theta=4)
    with pytest.raises(ValueError, match="n_phi|n_theta"):
        solid_sphere(n_phi=2)
    with pytest.raises(ValueError, match="unknown primitive"):
        generate_primitive("nope")
