"""Minimal NRRD reader (D83): attached-header 3-D scalar volumes -> RAS mm.

Written here instead of adding SimpleITK or pynrrd: CT frames like the
Zenodo 4D-CTA set (``NRRD0004``, ``encoding: gzip``, ``space:
left-posterior-superior``) need only a few header fields, and every field we
do NOT understand is refused loudly rather than guessed.

Supported: an attached header (data follows the blank line), ``encoding``
raw or gzip, ``dimension: 3``, the common scalar types, little/big endian,
``space`` LPS or RAS (either spelling) with ``space directions`` (negative
and rotated allowed) and ``space origin``. Refused: a detached data file,
any other encoding, non-3-D data, non-spatial axes, missing space
information, non-millimetre ``space units``, line/byte skips, and a header
whose voxel count exceeds the ceiling — checked from the HEADER, before a
single byte is decompressed (the F-IMG-7 pattern).

The returned affine maps voxel index (i, j, k) — i fastest, as stored — to
RAS millimetres (LPS input: x and y negated), the frame nibabel and the
Zenodo reference surfaces use.
"""
from __future__ import annotations

import gzip
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from meshforge.textguard import scrub

#: Upper bound on the text header; a real header is a few hundred bytes.
MAX_HEADER_BYTES = 64 * 1024

_TYPES: dict[str, str] = {}
for _names, _dt in (
    (("signed char", "int8", "int8_t"), "i1"),
    (("uchar", "unsigned char", "uint8", "uint8_t"), "u1"),
    (("short", "short int", "signed short", "signed short int", "int16", "int16_t"), "i2"),
    (("ushort", "unsigned short", "unsigned short int", "uint16", "uint16_t"), "u2"),
    (("int", "signed int", "int32", "int32_t"), "i4"),
    (("uint", "unsigned int", "uint32", "uint32_t"), "u4"),
    (("longlong", "long long", "long long int", "signed long long",
      "signed long long int", "int64", "int64_t"), "i8"),
    (("ulonglong", "unsigned long long", "unsigned long long int",
      "uint64", "uint64_t"), "u8"),
    (("float",), "f4"),
    (("double",), "f8"),
):
    for _n in _names:
        _TYPES[_n] = _dt

_SPACES = {
    "left-posterior-superior": "LPS",
    "lps": "LPS",
    "right-anterior-superior": "RAS",
    "ras": "RAS",
}
_VEC = re.compile(r"\(([^()]*)\)")


@dataclass
class NrrdVolume:
    """A decoded volume: ``array[i, j, k]`` and its voxel -> RAS-mm affine."""

    array: np.ndarray
    affine: np.ndarray  # 4x4, voxel index -> RAS millimetres
    space: str  # "LPS" | "RAS", as declared in the file
    fields: dict[str, str]
    notes: list[str] = field(default_factory=list)


def _fail(path: Path, why: str) -> ValueError:
    return ValueError(scrub(f"{path.name}: {why}"))


def read_nrrd_header(path: str | Path) -> tuple[dict[str, str], int]:
    """Parse the text header only: ``(fields, data_offset)``.

    Field names are lower-cased with inner whitespace normalised; ``#``
    comments and ``key:=value`` pairs are skipped. Nothing past the header is
    read.
    """
    path = Path(path)
    with path.open("rb") as fh:
        head = fh.read(MAX_HEADER_BYTES + 1)
    if not re.match(rb"NRRD000[1-5]\r?\n", head):
        raise _fail(path, "not a NRRD file (missing the NRRD000x magic line)")
    fields: dict[str, str] = {}
    pos = head.index(b"\n") + 1
    while True:
        nl = head.find(b"\n", pos)
        if nl < 0:
            raise _fail(
                path,
                f"no blank line ends the header within {MAX_HEADER_BYTES} bytes "
                f"— a detached or corrupt header is not supported",
            )
        line = head[pos:nl].rstrip(b"\r").decode("latin-1")
        pos = nl + 1
        if line == "":
            return fields, pos
        if line.startswith("#"):
            continue
        colon = line.find(":")
        if colon >= 0 and line.startswith(":=", colon):
            continue  # free-form ``key:=value`` metadata: not needed, ignored
        key, sep, value = line.partition(": ")
        if not sep:
            raise _fail(path, f"malformed header line {line[:80]!r}")
        fields[" ".join(key.lower().split())] = value.strip()


def _vectors(path: Path, raw: str, what: str, n: int) -> np.ndarray:
    if "none" in raw.lower().split():
        raise _fail(path, f"{what} has a 'none' (non-spatial) axis — 3 spatial axes are required")
    vecs = _VEC.findall(raw)
    if len(vecs) != n:
        raise _fail(path, f"{what} needs {n} vector(s) '(x,y,z)', got {raw[:120]!r}")
    try:
        out = np.array([[float(c) for c in v.split(",")] for v in vecs], dtype=float)
    except ValueError as exc:
        raise _fail(path, f"{what} is not numeric: {raw[:120]!r}") from exc
    if out.shape != (n, 3) or not np.isfinite(out).all():
        raise _fail(path, f"{what} must hold finite 3-vectors, got {raw[:120]!r}")
    return out


def read_nrrd(path: str | Path, *, max_voxels: int | None = None) -> NrrdVolume:
    """Read an attached-header 3-D NRRD into a :class:`NrrdVolume`.

    ``max_voxels`` (default ``geometry.centerline.MAX_VOXELS``) is enforced
    from the header ``sizes`` BEFORE any data is read or decompressed.
    Every unsupported or missing field raises ``ValueError`` naming it.
    """
    from meshforge.io.image import check_affine

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"NRRD file not found: {path}")
    if max_voxels is None:
        from meshforge.geometry.centerline import MAX_VOXELS

        max_voxels = MAX_VOXELS
    f, offset = read_nrrd_header(path)

    for key in ("data file", "datafile"):
        if key in f:
            raise _fail(path, "detached data ('data file' field) is not supported — "
                        "save the volume as a single attached .nrrd")
    for key in ("line skip", "lineskip", "byte skip", "byteskip"):
        if key in f and f[key].strip() != "0":
            raise _fail(path, f"'{key}: {f[key]}' is not supported")
    for key in ("type", "dimension", "sizes", "encoding"):
        if key not in f:
            raise _fail(path, f"header lacks the required '{key}' field")

    encoding = f["encoding"].lower()
    if encoding not in ("raw", "gzip", "gz"):
        raise _fail(path, f"encoding {f['encoding']!r} is not supported (raw or gzip only)")
    tname = " ".join(f["type"].lower().split())
    if tname not in _TYPES:
        raise _fail(path, f"scalar type {f['type']!r} is not supported")
    if f["dimension"].strip() != "3":
        raise _fail(path, f"dimension {f['dimension']} — only 3-D scalar volumes are supported")
    try:
        sizes = [int(s) for s in f["sizes"].split()]
    except ValueError as exc:
        raise _fail(path, f"sizes {f['sizes']!r} are not integers") from exc
    if len(sizes) != 3 or min(sizes) < 1:
        raise _fail(path, f"sizes {f['sizes']!r} must be three positive integers")
    if "kinds" in f:
        kinds = [k.lower() for k in f["kinds"].split()]
        if len(kinds) != 3 or any(k not in ("domain", "space") for k in kinds):
            raise _fail(path, f"kinds {f['kinds']!r} — only spatial (domain/space) axes are supported")

    # Spatial frame: required, never defaulted.
    if "space" not in f:
        why = "'space dimension' without 'space'" if "space dimension" in f else "no 'space' field"
        raise _fail(path, f"{why}: the anatomical frame (LPS/RAS) is unknown — refusing to guess")
    space = _SPACES.get(f["space"].lower())
    if space is None:
        raise _fail(path, f"space {f['space']!r} is not supported (LPS or RAS only)")
    for key in ("space directions", "space origin"):
        if key not in f:
            raise _fail(path, f"header lacks '{key}' — voxel positions are unknown")
    directions = _vectors(path, f["space directions"], "space directions", 3)
    origin = _vectors(path, f["space origin"], "space origin", 1)[0]
    notes: list[str] = []
    if "space units" in f:
        units = [u.strip('"') for u in f["space units"].split()]
        if units != ["mm", "mm", "mm"]:
            raise _fail(path, f"space units {f['space units']!r} — only millimetres are supported")
    else:
        notes.append(
            f"NRRD space {f['space']!r} declares no 'space units': millimetres by "
            f"the anatomical-space convention"
        )

    n_vox = int(np.prod(sizes, dtype=np.int64))
    if n_vox > max_voxels:
        raise _fail(
            path,
            f"header declares {sizes[0]}x{sizes[1]}x{sizes[2]} = {n_vox} voxels, "
            f"over the {max_voxels} voxel ceiling — refusing before decompressing",
        )

    dtype = np.dtype(_TYPES[tname])
    if dtype.itemsize > 1:
        endian = f.get("endian", "").lower()
        if endian not in ("little", "big"):
            raise _fail(path, f"a {dtype.itemsize}-byte type needs 'endian: little|big'")
        dtype = dtype.newbyteorder("<" if endian == "little" else ">")
    expected = n_vox * dtype.itemsize

    with path.open("rb") as fh:
        fh.seek(offset)
        if encoding == "raw":
            data = fh.read(expected + 1)
        else:
            try:
                # Bounded read: never more than one byte past what the header
                # declares, however the stream is crafted.
                data = gzip.GzipFile(fileobj=fh, mode="rb").read(expected + 1)
            except (OSError, EOFError, zlib.error) as exc:
                raise _fail(path, f"gzip data is corrupt: {exc}") from exc
    if len(data) < expected:
        raise _fail(path, f"data is truncated: {len(data)} of {expected} bytes")
    if len(data) > expected:
        raise _fail(path, f"data holds more than the {expected} bytes the header declares")

    array = np.frombuffer(data, dtype=dtype).reshape(sizes, order="F")
    array = array.astype(dtype.newbyteorder("="), copy=False)

    affine = np.eye(4)
    affine[:3, :3] = directions.T  # column a = step along index axis a
    affine[:3, 3] = origin
    if space == "LPS":
        affine[:2, :] *= -1.0  # LPS -> RAS: negate x and y (directions + origin)
        notes.append("converted the LPS header to the RAS world frame (x, y negated)")
    try:
        affine = check_affine(affine)
    except ValueError as exc:
        raise _fail(path, f"space directions: {exc}") from exc
    return NrrdVolume(array=array, affine=affine, space=space, fields=f, notes=notes)
