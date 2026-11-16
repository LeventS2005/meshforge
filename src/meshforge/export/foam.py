"""OpenFOAM polyMesh export (V14): the first CFD mesh format.

Writes ``<case>/constant/polyMesh/{points,faces,owner,neighbour,boundary}``
from a WallMesh. OpenFOAM's contract, honoured here:

- every internal face appears once, wound so its normal points from the
  owner (lower cell id) to the neighbour;
- internal faces come first, in upper-triangular order (sorted by owner,
  then neighbour); boundary faces follow, grouped per patch;
- boundary faces are wound outward.

Faces are extracted from canonical outward-oriented local faces per cell
(hexahedron + tetra — everything meshforge produces); a face seen twice is
internal and the first sighting is the owner, so both conventions fall out
of cell order. Boundary faces must be tiled by the WallMesh's boundary
groups: an uncovered face is a hard error unless ``default_patch`` collects
the leftovers deliberately (never fail-open — CFD patch BCs depend on this).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

from meshforge.export.neutral import guard_output_path
from meshforge.meshing.base import WallMesh

# Local vertex orders producing OUTWARD normals (right-hand rule), for the
# VTK node numbering meshforge writes everywhere. Wedge faces follow the
# positive-volume convention the V22 prism mesher writes (bottom tri wound
# away from the top — verified against VTK cell volumes).
_LOCAL_FACES = {
    int(pv.CellType.HEXAHEDRON): (
        (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ),
    int(pv.CellType.TETRA): ((0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)),
    int(pv.CellType.WEDGE): (
        (0, 1, 2), (3, 5, 4),
        (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0),
    ),
}

_FOAM_CLASSES = {
    "points": "vectorField",
    "faces": "faceList",
    "owner": "labelList",
    "neighbour": "labelList",
    "boundary": "polyBoundaryMesh",
}


def _patch_type(name: str, patch_types: dict[str, str]) -> str:
    if name in patch_types:
        return patch_types[name]
    return "patch" if ("inlet" in name.lower() or "outlet" in name.lower()) else "wall"


def _foam_file(path: Path, obj: str, body: list[str], *, note: str | None = None) -> None:
    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        f"    class       {_FOAM_CLASSES[obj]};",
    ]
    if note is not None:
        lines.append(f'    note        "{note}";')
    lines += [
        '    location    "constant/polyMesh";',
        f"    object      {obj};",
        "}",
        "",
    ]
    path.write_text("\n".join(lines + body) + "\n")


def _extract_faces(grid: pv.UnstructuredGrid):
    """One pass over all cells: internal (owner, neighbour, verts) with
    owner-outward winding, and boundary (owner, verts) wound outward."""
    celltypes = np.asarray(grid.celltypes)
    conn = np.asarray(grid.cell_connectivity)
    offsets = np.asarray(grid.offset)

    seen: dict[tuple, list] = {}
    for cell in range(grid.n_cells):
        ctype = int(celltypes[cell])
        if ctype not in _LOCAL_FACES:
            raise ValueError(
                f"polyMesh export supports hexahedra, tetrahedra and wedges; "
                f"cell {cell} has VTK type {ctype}"
            )
        nodes = conn[offsets[cell] : offsets[cell + 1]]
        for local in _LOCAL_FACES[ctype]:
            verts = tuple(int(nodes[i]) for i in local)
            key = tuple(sorted(verts))
            hit = seen.get(key)
            if hit is None:
                seen[key] = [cell, verts, None]
            elif hit[2] is not None:
                raise ValueError(
                    f"face {key} is shared by more than two cells — the mesh "
                    f"is non-manifold; refusing to write polyMesh"
                )
            else:
                hit[2] = cell
    internal = [(o, n, v) for o, v, n in seen.values() if n is not None]
    internal.sort(key=lambda f: (f[0], f[1]))  # upper-triangular order
    boundary = {k: (o, v) for k, (o, v, n) in seen.items() if n is None}
    return internal, boundary


def _patch_face_map(
    wall: WallMesh, boundary: dict, default_patch: str | None
) -> tuple[list[str], dict[str, list[tuple]]]:
    """Group the boundary faces (sorted-vertex keys) into named patches
    from the WallMesh boundary groups (winding-robust). Loud on faces
    claimed twice, stale group faces, and uncovered faces (unless
    ``default_patch`` collects them deliberately). Used by the
    polyMesh writer (V14)."""
    # F1: WallMesh boundary-group names are already validated by the
    # BoundaryGroup constructor, but ``default_patch`` is minted at write
    # time and is NOT a BoundaryGroup — content-validate it here so it can
    # never inject a forged patch into the polyMesh patch list.
    if default_patch is not None:
        from meshforge.meshing.base import valid_group_name

        valid_group_name(default_patch)
    face_to_patch: dict[tuple, str] = {}
    for name, group in wall.boundary.items():
        for row in np.asarray(group.face_nodes):
            key = tuple(sorted(int(n) for n in row))
            prev = face_to_patch.get(key)
            if prev is not None and prev != name:
                raise ValueError(
                    f"boundary face {key} is claimed by both {prev!r} and {name!r}"
                )
            face_to_patch[key] = name
    stale = len(face_to_patch) - sum(1 for k in face_to_patch if k in boundary)
    if stale:
        raise ValueError(
            f"{stale} boundary-group face(s) are not on the mesh boundary — "
            f"the groups do not describe this grid"
        )

    patch_names = list(wall.boundary)
    patch_faces: dict[str, list[tuple]] = {name: [] for name in patch_names}
    unmatched = 0
    for key in boundary:
        name = face_to_patch.get(key)
        if name is None:
            if default_patch is None:
                unmatched += 1
                continue
            patch_faces.setdefault(default_patch, []).append(key)
            if default_patch not in patch_names:
                patch_names.append(default_patch)
        else:
            patch_faces[name].append(key)
    if unmatched:
        raise ValueError(
            f"{unmatched} boundary face(s) are covered by no boundary group; "
            f"pass default_patch=... to collect them deliberately"
        )
    return [n for n in patch_names if patch_faces.get(n)], patch_faces


def write_polymesh(
    wall: WallMesh,
    case_dir: str | Path,
    *,
    patch_types: dict[str, str] | None = None,
    default_patch: str | None = None,
    input_path: str | Path | None = None,
) -> Path:
    """Write ``constant/polyMesh`` under ``case_dir``; returns the polyMesh dir."""
    patch_types = dict(patch_types or {})
    internal, boundary = _extract_faces(wall.grid)
    patch_names, patch_faces = _patch_face_map(wall, boundary, default_patch)
    for name in patch_types:
        if name not in patch_names:
            raise ValueError(
                f"patch_types names unknown patch {name!r}; patches: {patch_names}"
            )

    # Assemble the global face list: internal first, then patches in order.
    faces: list[tuple] = [v for _, _, v in internal]
    owner: list[int] = [o for o, _, _ in internal]
    neighbour: list[int] = [n for _, n, _ in internal]
    patches: list[tuple[str, int, int]] = []  # (name, nFaces, startFace)
    for name in patch_names:
        keys = sorted(patch_faces[name], key=lambda k: boundary[k][0])
        patches.append((name, len(keys), len(faces)))
        for key in keys:
            o, verts = boundary[key]
            faces.append(verts)
            owner.append(o)

    polymesh = Path(case_dir) / "constant" / "polyMesh"
    for obj in _FOAM_CLASSES:
        guard_output_path(polymesh / obj, input_paths=[input_path])
    polymesh.mkdir(parents=True, exist_ok=True)

    pts = np.asarray(wall.grid.points, dtype=float)
    _foam_file(
        polymesh / "points", "points",
        [f"{len(pts)}", "("]
        + [f"({p[0]:.16g} {p[1]:.16g} {p[2]:.16g})" for p in pts]
        + [")"],
    )
    _foam_file(
        polymesh / "faces", "faces",
        [f"{len(faces)}", "("]
        + [f"{len(v)}({' '.join(str(i) for i in v)})" for v in faces]
        + [")"],
    )
    note = (
        f"nPoints:{len(pts)} nCells:{wall.grid.n_cells} "
        f"nFaces:{len(faces)} nInternalFaces:{len(internal)}"
    )
    _foam_file(
        polymesh / "owner", "owner",
        [f"{len(owner)}", "("] + [str(o) for o in owner] + [")"], note=note,
    )
    _foam_file(
        polymesh / "neighbour", "neighbour",
        [f"{len(neighbour)}", "("] + [str(n) for n in neighbour] + [")"], note=note,
    )
    body = [f"{len(patches)}", "("]
    for name, n_faces, start in patches:
        body += [
            f"    {name}",
            "    {",
            f"        type            {_patch_type(name, patch_types)};",
            f"        nFaces          {n_faces};",
            f"        startFace       {start};",
            "    }",
        ]
    body.append(")")
    _foam_file(polymesh / "boundary", "boundary", body)
    return polymesh
