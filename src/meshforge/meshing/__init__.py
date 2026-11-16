"""Volume meshing: the swept wall backend, the lumen lattice and the O-grid disk."""
from __future__ import annotations

from meshforge.meshing.base import (
    BoundaryGroup,
    WallMesh,
    get_backend,
    register_backend,
    split_boundary_group,
    valid_group_name,
)
from meshforge.meshing.lattice import WallLattice, build_wall_lattice
from meshforge.meshing.ogrid import ogrid_disk, quad_scaled_jacobians
from meshforge.meshing.swept import SweptBackend

register_backend("swept", SweptBackend)

__all__ = [
    "BoundaryGroup",
    "SweptBackend",
    "WallLattice",
    "WallMesh",
    "build_wall_lattice",
    "get_backend",
    "ogrid_disk",
    "quad_scaled_jacobians",
    "register_backend",
    "split_boundary_group",
    "valid_group_name",
]
