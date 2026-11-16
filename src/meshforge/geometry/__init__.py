"""Centerlines, frames, cross-sections, wall thickness."""
from __future__ import annotations

from meshforge.geometry.centerline import (
    Centerline,
    centerline_from_surface,
    compute_centerline,
    embed_centerline,
)
from meshforge.geometry.frames import parallel_transport_frames
from meshforge.geometry.sections import (
    Constriction,
    Section,
    compute_sections,
    detect_constrictions,
)
from meshforge.geometry.thickness import ThicknessSpec, resolve_thickness

__all__ = [
    "Centerline",
    "Constriction",
    "Section",
    "ThicknessSpec",
    "resolve_thickness",
    "centerline_from_surface",
    "compute_centerline",
    "compute_sections",
    "embed_centerline",
    "detect_constrictions",
    "parallel_transport_frames",
]
