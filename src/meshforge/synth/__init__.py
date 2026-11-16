"""Synthetic watertight surfaces with exact analytic ground truth.

Tube generators (``generators.py``) return (surface, AnalyticTruth) with
centerline truth; primitive solids (``primitives.py``) return
(surface, PrimitiveTruth) with closed-form area/volume. Both registries feed
the CLI ``synth`` command and downstream tests.
"""
from __future__ import annotations

from meshforge.synth.generators import (
    GENERATORS,
    AnalyticTruth,
    constricted_tube,
    curved_tube,
    generate,
    straight_tube,
    tapered_tube,
    torus,
)
from meshforge.synth.primitives import (
    PRIMITIVES,
    PrimitiveTruth,
    box,
    extrude_loops,
    generate_primitive,
    plate_with_hole,
    revolve_loop,
    solid_cylinder,
    solid_sphere,
)

__all__ = [
    "GENERATORS",
    "PRIMITIVES",
    "AnalyticTruth",
    "PrimitiveTruth",
    "box",
    "constricted_tube",
    "curved_tube",
    "extrude_loops",
    "generate",
    "generate_primitive",
    "plate_with_hole",
    "revolve_loop",
    "solid_cylinder",
    "solid_sphere",
    "straight_tube",
    "tapered_tube",
    "torus",
]
