from __future__ import annotations

import hashlib
from pathlib import Path

import pyvista as pv

from meshforge.io.paths import normalize_path

# A CT volume is not a surface: "load" is the wrong front door for it, so name
# the right one (the segment step) instead of letting pyvista's own error
# through ("This file was not able to be automatically read by pyvista."
# tells a user nothing).
#
# One table: {suffix: (what it is, Workbench panel, CLI subcommand)}.
_DEDICATED_IMPORTERS: dict[str, tuple[str, str, str]] = {
    ".nii": ("a NIfTI volume", "segment", "segment"),
    ".nii.gz": ("a NIfTI volume", "segment", "segment"),
}

# Formats the full meshforge reads through importers this demo edition does
# not ship. Still refused by SUFFIX, before the file is opened, so the user
# reads a plain sentence instead of a reader error. {suffix: what it is}.
_UNREAD_SUFFIXES: dict[str, str] = {
    ".cso": "a MeVisLab contour file",
    ".dxf": "a 2-D CAD drawing",
    ".dwg": "a 2-D CAD drawing",
    ".step": "a 3-D CAD solid",
    ".stp": "a 3-D CAD solid",
    ".iges": "a 3-D CAD solid",
    ".igs": "a 3-D CAD solid",
    ".brep": "a 3-D CAD solid",
    ".npy": "a voxel volume",
    ".dcm": "a DICOM slice",
}


def file_sha256(path: str | Path) -> str:
    """SHA-256 of the file bytes; used to prove inputs stay byte-for-byte unchanged.

    The existence guard is here rather than in each caller because every
    rule-1 bracket (``checksum -> work -> checksum``) hashes FIRST: without it
    a mistyped path surfaces as a bare ``[Errno 2] No such file or directory``
    from ``open()``, and the caller's own friendly message never runs. A file
    that does not exist has nothing to protect, so checking before the first
    hash cannot weaken the read-only proof.
    """
    path = normalize_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dedicated_importer(path: str | Path) -> tuple[str, str, str, str] | None:
    """``(suffix, description, panel, subcommand)`` if this suffix has its own
    importer, else ``None``.

    Longest suffix wins so ``.nii.gz`` never resolves as a bare ``.nii``.
    """
    name = Path(path).name.lower()
    for suffix in sorted(_DEDICATED_IMPORTERS, key=len, reverse=True):
        if name.endswith(suffix):
            return (suffix, *_DEDICATED_IMPORTERS[suffix])
    return None


def _unread_suffix(path: str | Path) -> tuple[str, str] | None:
    """``(suffix, description)`` if this edition refuses the suffix, else ``None``
    (longest suffix wins, case-insensitive)."""
    name = Path(path).name.lower()
    for suffix in sorted(_UNREAD_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix):
            return suffix, _UNREAD_SUFFIXES[suffix]
    return None


def load_surface(path: str | Path) -> pv.PolyData:
    """Load a triangulated surface (.stl/.ply/.vtp/.obj/...) read-only.

    Rejects non-surface data (volume grids, point clouds) rather than
    guessing a conversion — the caller must know what they loaded. A file
    that is not a surface (a NIfTI CT, contours, CAD, other volumes) is
    refused here by SUFFIX, before the file is opened: a NIfTI CT is pointed
    at the segment step, the rest are named as not read by this edition.
    """
    path = normalize_path(path)
    match = dedicated_importer(path)
    if match is not None:
        suffix, what, _panel, subcommand = match
        raise ValueError(
            f"{suffix} is {what} — load a CT with the segment step "
            f"(CLI: meshforge {subcommand} <file>), not as a surface: {path.name}"
        )
    unread = _unread_suffix(path)
    if unread is not None:
        suffix, what = unread
        raise ValueError(
            f"{suffix} is {what} — this demo edition does not read it (it loads CT "
            f"volumes with the segment step and surfaces such as .stl/.vtp/.ply): "
            f"{path.name}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"surface file not found: {path}")
    mesh = pv.read(path)
    if not isinstance(mesh, pv.PolyData):
        raise TypeError(
            f"expected a surface (PolyData), got {type(mesh).__name__} from {path.name}; "
            "meshforge inspects surfaces, not volume grids"
        )
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError(f"surface is empty (0 points or 0 cells): {path.name}")
    return mesh
