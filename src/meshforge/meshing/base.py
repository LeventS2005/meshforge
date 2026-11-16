from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pyvista as pv

from meshforge.quality.volume import VolumeQuality

# F1 (session 12): boundary/patch group names are emitted verbatim (only
# ``.upper()``-cased) into solver decks (``*ELSET``/``*NSET``/``*SURFACE``,
# NASTRAN/LS-DYNA set names) and OpenFOAM ``boundary``/``boundaryField``
# dictionaries. A name carrying a newline would inject arbitrary solver
# cards / dict entries — a silent fail-open (violates the fail-loudly rule).
# ``valid_group_name`` is the single content check; it is called at every
# site where a group/patch name is MINTED (split_boundary_group, the
# BoundaryGroup constructor as a backstop, the CFD BC spec, and the CLI
# patch-name parses) so a bad name is refused loudly before it can reach a
# writer. All internal group names ("inner"/"outer"/"inlet"/"outlet"/
# "front"/"back"/"outer_rim"/"hole_rims"/"body") satisfy the pattern.
_GROUP_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")


def valid_group_name(name: str) -> str:
    """Return ``name`` unchanged if it is a safe boundary/patch group name,
    else raise loudly. Safe = a leading letter/underscore then up to 63 more
    letters/digits/``_.-`` (<=64 chars total, no whitespace/newlines/control
    chars). A refused name is NEVER written into a deck or dictionary."""
    if not isinstance(name, str) or not _GROUP_NAME_RE.match(name):
        raise ValueError(
            f"invalid boundary-group name {name!r}: must match "
            f"{_GROUP_NAME_RE.pattern} (a letter/_ then letters/digits/_.- , "
            f"<=64 chars, no whitespace/newlines/control chars) — refusing to "
            f"write it into a solver deck or dictionary"
        )
    return name


@dataclass
class BoundaryGroup:
    """Named boundary faces as (parent cell id, face node ids) — node-set
    based so downstream solver export is robust to face re-winding."""

    name: str
    cell_ids: np.ndarray  # (k,)
    face_nodes: np.ndarray  # (k, m) node ids, m=4 (hex quad) or 3 (tet tri)

    def __post_init__(self) -> None:
        # Backstop (F1): every group name, however constructed, is content-
        # validated here — so deck/foam writers that iterate wall.boundary can
        # trust the invariant that names are injection-safe.
        valid_group_name(self.name)

    @property
    def n_faces(self) -> int:
        return len(self.cell_ids)

    def to_polydata(self, points: np.ndarray) -> pv.PolyData:
        m = self.face_nodes.shape[1]
        faces = np.column_stack(
            [np.full(self.n_faces, m, dtype=np.int64), self.face_nodes]
        ).ravel()
        return pv.PolyData(points, faces)


@dataclass
class WallMesh:
    """Tagged wall volume mesh with named boundary groups."""

    grid: pv.UnstructuredGrid  # carries cell_data["region"]
    boundary: dict[str, BoundaryGroup]
    backend: str
    quality: VolumeQuality
    notes: list[str] = field(default_factory=list)
    # D85: the swept wall's thickness guard counts (n_capped / n_floored rays,
    # applied thickness range) — a stress result on a capped or floored wall
    # is not reportable, so the counts travel with the mesh, not only in notes.
    lattice: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        out = {
            "backend": self.backend,
            "n_points": int(self.grid.n_points),
            "n_cells": int(self.grid.n_cells),
            "boundary_faces": {k: g.n_faces for k, g in self.boundary.items()},
            "quality": self.quality.to_dict(),
            "notes": self.notes,
        }
        if self.lattice is not None:
            out["lattice"] = dict(self.lattice)
        return out


_AXES = {"x": 0, "y": 1, "z": 2}


def split_boundary_group(
    wall: WallMesh,
    group: str,
    *,
    axis: str,
    threshold: float,
    name: str,
    side: str = "above",
) -> WallMesh:
    """Move the faces of one boundary group whose CENTROID coordinate is
    above/below ``threshold`` into a new group ``name`` (V30 — e.g. the
    moving lid of a cavity out of the extruded ``outer_rim``).

    Mutates ``wall.boundary`` in place and returns the wall. Loud on an
    unknown group/axis/side, a taken target name, and on splits that
    move nothing or everything (almost certainly a wrong threshold).
    """
    if group not in wall.boundary:
        raise ValueError(
            f"unknown boundary group {group!r}; mesh has "
            f"{sorted(wall.boundary)}"
        )
    if axis not in _AXES:
        raise ValueError(f"axis must be one of {sorted(_AXES)}, got {axis!r}")
    if side not in ("above", "below"):
        raise ValueError(f"side must be 'above' or 'below', got {side!r}")
    # F1: content-validate the minted name BEFORE anything else touches it —
    # this is the crafted-input choke point (a ``.mfwork`` split_patch step
    # replays through here with an attacker-controlled ``name``).
    valid_group_name(name)
    if name in wall.boundary:
        raise ValueError(f"target group {name!r} already exists")

    src = wall.boundary[group]
    centroids = np.asarray(wall.grid.points)[src.face_nodes].mean(axis=1)
    mask = centroids[:, _AXES[axis]] >= threshold
    if side == "below":
        mask = ~mask
    if not mask.any() or mask.all():
        raise ValueError(
            f"splitting {group!r} at {axis}={threshold:g} ({side}) moves "
            f"{int(mask.sum())} of {src.n_faces} faces — a split that "
            f"moves nothing or everything is a wrong threshold"
        )
    wall.boundary[name] = BoundaryGroup(
        name=name,
        cell_ids=src.cell_ids[mask],
        face_nodes=src.face_nodes[mask],
    )
    wall.boundary[group] = BoundaryGroup(
        name=group,
        cell_ids=src.cell_ids[~mask],
        face_nodes=src.face_nodes[~mask],
    )
    wall.notes.append(
        f"split_patch: {int(mask.sum())} face(s) of {group!r} with "
        f"centroid {axis} {'>=' if side == 'above' else '<'} {threshold:g} "
        f"-> {name!r}"
    )
    return wall


_REGISTRY: dict[str, Callable[[], object]] = {}


def register_backend(name: str, factory: Callable[[], object]) -> None:
    _REGISTRY[name] = factory


def get_backend(name: str = "swept"):
    """Meshing backends are registered lazily; heavy deps (gmsh) are only
    imported when the backend actually runs."""
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(
            f"unknown meshing backend {name!r}; available: {', '.join(sorted(_REGISTRY))}"
        ) from None
