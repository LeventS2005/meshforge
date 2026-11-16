"""Biomedical image ingest (V18): 3-D volumes -> iso-surfaces.

The numpy core (``volume_to_surface``) needs no new dependency; the
medical loaders are lazy — nibabel for NIfTI, pydicom for DICOM series —
and their absence raises with an install hint. ``level`` is always
required: a segmentation threshold is a modelling decision, never a
default.

Orientation (D83): a NIfTI whose affine is rotated or has negative axis
directions (LPS-style data, every RAS volume with flipped x/y) is NOT
resampled. Marching cubes runs in voxel-index space and the vertices are
mapped through the full 4x4 affine (``load_volume_affine`` +
``volume_to_surface(..., affine=A)``); a reflecting affine (det < 0)
reverses the triangle winding, so the faces are flipped back to keep the
normals outward. Singular/degenerate affines are refused loudly.
``mask_to_surface`` is the binary-mask front end (AI segmentations):
largest connected component, bounding-box crop, optional Gaussian blur,
level 0.5.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

_NIFTI_SUFFIXES = (".nii", ".nii.gz")


def check_affine(affine) -> np.ndarray:
    """Validate a 4x4 voxel-index -> world affine; return it as float.

    Refuses (ValueError) a wrong shape, non-finite entries, a bottom row
    other than ``0 0 0 1``, a zero-length voxel axis, and a singular or
    near-singular (coplanar voxel axes) linear part. Rotation, reflection
    and shear are all accepted: the mapping is exact for any non-singular
    linear map.
    """
    a = np.asarray(affine, dtype=float)
    if a.shape != (4, 4):
        raise ValueError(f"affine must be 4x4, got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError("affine contains non-finite values")
    if not np.allclose(a[3], (0.0, 0.0, 0.0, 1.0)):
        raise ValueError(f"affine bottom row must be 0 0 0 1, got {a[3].tolist()}")
    lin = a[:3, :3]
    norms = np.linalg.norm(lin, axis=0)
    if float(norms.min()) <= 0.0:
        raise ValueError(
            f"degenerate affine: voxel axis lengths {norms.tolist()} — an axis "
            f"has zero length"
        )
    rel = abs(float(np.linalg.det(lin))) / float(np.prod(norms))
    if rel < 1e-6:
        raise ValueError(
            f"singular affine (|det| / axis lengths = {rel:.3g}): the voxel axes "
            f"are (nearly) coplanar — refusing to place voxels on a degenerate grid"
        )
    return a


def _signed_volume(surface) -> float:
    p = np.asarray(surface.points, dtype=float)
    tri = surface.faces.reshape(-1, 4)[:, 1:]
    return float(
        np.einsum("ij,ij->i", p[tri[:, 0]], np.cross(p[tri[:, 1]], p[tri[:, 2]])).sum()
        / 6.0
    )


def _flipped(surface):
    if hasattr(surface, "flip_faces"):
        return surface.flip_faces()
    surface.flip_normals()  # older pyvista: in-place
    return surface


def _clean_triangles(surface):
    """Triangulate + merge duplicate points, DELETING the slivers that merge.

    clean() by default demotes a triangle whose vertices merge to a line or
    vertex cell; those make a watertight surface fail the all-triangles
    mesh_ready check (seen on a real 0.48 mm CTA aorta: 665 lines + 79 verts
    among 700 k triangles). Deleting a zero-area sliver changes no edge.
    """
    return surface.triangulate().clean(polys_to_lines=False, lines_to_points=False)


def volume_to_surface(
    array: np.ndarray,
    level: float,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    affine=None,
):
    """Marching-cubes iso-surface of a 3-D scalar volume.

    Index (i, j, k) maps to physical ``origin + index * spacing``, or —
    when ``affine`` (4x4) is given — to ``affine @ (i, j, k, 1)``: the
    contour is taken in index space and its vertices are mapped through the
    affine, so rotated/flipped grids need no resampling (D83). Pass either
    ``affine`` or ``spacing``/``origin``, not both.
    Returns (all-triangle outward-oriented pv.PolyData, provenance dict);
    a leaky (non-watertight) surface is a provenance note, not an error —
    downstream ``check`` gates meshing.
    """
    import pyvista as pv

    array = np.asarray(array)
    if array.ndim != 3 or min(array.shape) < 2:
        raise ValueError(
            f"volume must be 3-D with every axis >= 2, got shape {array.shape}"
        )
    a = None
    if affine is not None:
        if tuple(spacing) != (1.0, 1.0, 1.0) or tuple(origin) != (0.0, 0.0, 0.0):
            raise ValueError("pass either affine or spacing/origin, not both")
        a = check_affine(affine)
    # F4: bound the voxel count exactly as the geometry paths do
    # (centerline.py/sdf.py) — the V18 image ingest otherwise flows an
    # oversized/crafted volume straight into pv.ImageData + marching cubes
    # with no ceiling, a memory-exhaustion DoS. Refuse before scanning it.
    from meshforge.geometry.centerline import MAX_VOXELS

    if int(np.prod(array.shape)) > MAX_VOXELS:
        raise ValueError(
            f"volume {tuple(array.shape)} exceeds {MAX_VOXELS} voxels; "
            f"downsample before ingest"
        )
    if not np.isfinite(array).all():
        raise ValueError("volume contains non-finite values — clean it first")
    if not all(s > 0 for s in spacing):
        raise ValueError(f"spacing must be positive, got {spacing}")
    lo, hi = float(array.min()), float(array.max())
    if not lo < level < hi:
        raise ValueError(
            f"iso level {level} is outside the data range [{lo:.6g}, {hi:.6g}]"
        )

    if a is None:
        grid_spacing, grid_origin = tuple(spacing), tuple(origin)
    else:  # contour in voxel-index space, map through the affine below
        grid_spacing, grid_origin = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
    grid = pv.ImageData(
        dimensions=array.shape, spacing=grid_spacing, origin=grid_origin
    )
    grid.point_data["v"] = array.ravel(order="F")
    surface = _clean_triangles(grid.contour([float(level)], scalars="v"))
    if surface.n_cells == 0:
        raise ValueError(
            f"contour at level {level} produced no triangles — the iso-surface "
            f"may clip the volume bounds"
        )

    # Enforce OUTWARD normals (contour can emit inward — the V11 lesson):
    # the signed volume of a closed outward surface is positive. Done in the
    # contour's own (positively scaled) frame, before any affine mapping.
    signed = _signed_volume(surface)
    if signed < 0:
        surface = _flipped(surface)
        signed = -signed

    if a is not None:
        pts = np.asarray(surface.points, dtype=float)
        surface.points = pts @ a[:3, :3].T + a[:3, 3]
        if float(np.linalg.det(a[:3, :3])) < 0.0:
            # A reflecting affine (e.g. LPS-style negative directions on one
            # axis) reverses every triangle's orientation: flip back so the
            # normals stay outward (D83).
            surface = _flipped(surface)
        signed = _signed_volume(surface)

    notes = []
    watertight = int(surface.n_open_edges) == 0
    if not watertight:
        notes.append(
            f"iso-surface is not watertight ({surface.n_open_edges} open "
            f"edge(s)) — it likely clips the volume bounds; pad the volume "
            f"or repair before meshing"
        )
    elif signed <= 0.0:  # pragma: no cover - invariant, fail loud (rule 6)
        raise RuntimeError(
            f"internal error: the closed iso-surface is inward-oriented after "
            f"the affine mapping (signed volume {signed:.6g})"
        )
    if a is None:
        sp_out = [float(s) for s in spacing]
        or_out = [float(v) for v in origin]
    else:
        sp_out = [float(s) for s in np.linalg.norm(a[:3, :3], axis=0)]
        or_out = [float(v) for v in a[:3, 3]]
    provenance = {
        "shape": [int(s) for s in array.shape],
        "spacing": sp_out,
        "origin": or_out,
        "level": float(level),
        "data_range": [lo, hi],
        "n_points": int(surface.n_points),
        "n_triangles": int(surface.n_cells),
        "enclosed_volume": abs(signed) if watertight else None,
        "watertight": watertight,
        "notes": notes,
    }
    if a is not None:
        provenance["affine"] = a.tolist()
    return surface, provenance


def mask_to_surface(
    mask: np.ndarray,
    affine,
    *,
    sigma: float = 1.0,
    max_voxels: int | None = None,
):
    """Binary segmentation mask -> closed outward surface in the affine's frame.

    1. keep the largest 6-connected component (dropped components/voxels are
       reported in the provenance notes);
    2. crop to its bounding box plus a margin and zero-pad past the volume
       edge, so a full-CT-size mask stays under the voxel ceiling and an
       organ cut by the field of view still closes (flat cap at the edge);
    3. optional Gaussian blur, ``sigma`` in voxels (0 = off), then marching
       cubes at level 0.5 in voxel-index space, mapped through ``affine``;
    4. keep the largest connected piece of the surface (a thin bridge can
       split under the blur — reported).

    An empty mask, a cropped box over ``max_voxels`` (default
    ``geometry.centerline.MAX_VOXELS``) and a mask too thin to survive the
    blur are refused loudly. Returns ``(surface, provenance)``.
    """
    from scipy import ndimage

    from meshforge.geometry.centerline import MAX_VOXELS

    max_voxels = MAX_VOXELS if max_voxels is None else int(max_voxels)
    m = np.asarray(mask)
    if m.ndim != 3:
        raise ValueError(f"mask must be 3-D, got shape {m.shape}")
    m = m.astype(bool, copy=False)
    a = check_affine(affine)
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError(f"smoothing sigma must be >= 0 voxels, got {sigma}")

    def bbox(b: np.ndarray):
        lo, hi = [], []
        for ax in range(3):
            other = tuple(i for i in range(3) if i != ax)
            idx = np.flatnonzero(b.any(axis=other))
            lo.append(int(idx[0]))
            hi.append(int(idx[-1]))
        return np.array(lo), np.array(hi)

    n_in = int(np.count_nonzero(m))
    if n_in == 0:
        raise ValueError("the mask is empty — there is no object to extract")
    notes: list[str] = []
    # Label inside the bounding box of ALL foreground only (cheap on a
    # full-CT mask whose object is small): ndimage.label's int32 labels on
    # the full grid would cost 4 bytes per CT voxel.
    lo0, hi0 = bbox(m)
    box0 = m[lo0[0] : hi0[0] + 1, lo0[1] : hi0[1] + 1, lo0[2] : hi0[2] + 1]
    labels, n_comp = ndimage.label(box0)  # 6-connectivity
    if n_comp > 1:
        sizes = np.bincount(labels.ravel())[1:]
        keep = int(np.argmax(sizes)) + 1
        kept = labels == keep
        n_kept = int(sizes[keep - 1])
        notes.append(
            f"kept the largest connected component ({n_kept} voxels); dropped "
            f"{n_comp - 1} other component(s) totalling {n_in - n_kept} voxels"
        )
    else:
        kept = box0
        n_kept = n_in
    del labels
    lo1, hi1 = bbox(kept)
    lo = lo0 + lo1
    hi = lo0 + hi1
    margin = 2 + int(np.ceil(3.0 * sigma))
    k = kept[
        lo1[0] : hi1[0] + 1, lo1[1] : hi1[1] + 1, lo1[2] : hi1[2] + 1
    ]
    # A fine CT (0.5 mm in-plane) puts a bifurcation-to-arch aorta over the
    # ceiling. Block-average the object by the smallest integer factors that
    # fit — finest axis first — rather than refuse: the occupancy fraction is
    # a better marching-cubes input than the binary mask anyway. Loud note.
    extent = hi - lo + 1
    spacing = np.linalg.norm(a[:3, :3], axis=0)
    f = np.ones(3, dtype=int)
    while int(np.prod(-(-extent // f) + 2 * margin)) > max_voxels:
        ax = int(np.argmin(spacing * f + 1e-9 * np.arange(3)))
        f[ax] += 1
        if f[ax] > 8:
            raise ValueError(
                f"the mask's object spans {tuple(int(v) for v in extent)} voxels; "
                f"even 8x block-averaging leaves it over the {max_voxels} voxel "
                f"ceiling — crop the volume to the region of interest first"
            )
    if np.any(f > 1):
        pad = (-extent) % f
        k = np.pad(k, [(0, int(q)) for q in pad])
        c = k.shape // f
        k = k.reshape(c[0], f[0], c[1], f[1], c[2], f[2]).mean(axis=(1, 3, 5))
        notes.append(
            f"the object's {tuple(int(v) for v in extent)} voxel box is over the "
            f"{max_voxels} voxel ceiling — block-averaged by "
            f"{tuple(int(v) for v in f)} to "
            f"{tuple(round(float(v), 3) for v in spacing * f)} voxel spacing "
            f"before extraction"
        )
    sub_shape = tuple(int(v) for v in np.asarray(k.shape) + 2 * margin)
    sub = np.zeros(sub_shape, dtype=np.float32)
    sub[margin:-margin, margin:-margin, margin:-margin] = k
    touches = [
        ax for ax in range(3) if lo[ax] == 0 or hi[ax] == m.shape[ax] - 1
    ]
    if touches:
        notes.append(
            "the object touches the volume edge along axis "
            + "/".join("ijk"[ax] for ax in touches)
            + " — closed there by a flat cap at the field-of-view boundary"
        )
    if sigma > 0:
        # sigma is in the ORIGINAL voxels: keep its physical size on a
        # block-averaged grid (a 1-voxel blur on 3x blocks would shrink a
        # small vessel by ~20%)
        sub = ndimage.gaussian_filter(
            sub, sigma=tuple(sigma / f), mode="constant", cval=0.0
        )
    peak = float(sub.max())
    if peak <= 0.5:
        raise ValueError(
            f"the mask is too thin for smoothing sigma {sigma:g} voxel(s): the "
            f"blurred mask peaks at {peak:.3f} <= 0.5 — lower the smoothing"
        )
    # sub index c sits at volume index lo + (c - margin) * f + (f - 1) / 2:
    # the centre of its block (f = 1: plain crop offset lo - margin)
    offset = np.eye(4)
    offset[:3, :3] = np.diag(f.astype(float))
    offset[:3, 3] = lo - margin * f + (f - 1) / 2.0
    surface, vprov = volume_to_surface(sub, 0.5, affine=a @ offset)

    import pyvista as pv

    regions = surface.connectivity("all")
    region_ids = np.asarray(regions.point_data.get("RegionId", np.zeros(0)))
    n_regions = int(region_ids.max()) + 1 if region_ids.size else 1
    if n_regions > 1:
        largest = surface.connectivity("largest")
        if not isinstance(largest, pv.PolyData):
            largest = largest.extract_surface()
        largest = _clean_triangles(largest)
        for name in list(largest.point_data.keys()):
            del largest.point_data[name]
        for name in list(largest.cell_data.keys()):
            del largest.cell_data[name]
        notes.append(
            f"the smoothed surface split into {n_regions} pieces; kept the "
            f"largest ({largest.n_cells} of {surface.n_cells} triangles)"
        )
        surface = largest
        signed = _signed_volume(surface)
        watertight = int(surface.n_open_edges) == 0
        vprov.update(
            n_points=int(surface.n_points),
            n_triangles=int(surface.n_cells),
            watertight=watertight,
            enclosed_volume=abs(signed) if watertight else None,
        )
    if not vprov["watertight"]:
        notes.append(
            f"the surface is not watertight ({surface.n_open_edges} open "
            f"edge(s)) — repair before meshing"
        )
    voxel_volume = abs(float(np.linalg.det(a[:3, :3])))
    provenance = {
        "mask_shape": [int(s) for s in m.shape],
        "mask_voxels": n_in,
        "kept_voxels": n_kept,
        "components": int(n_comp),
        "dropped_components": int(max(n_comp - 1, 0)),
        "dropped_voxels": int(n_in - n_kept),
        "mask_volume": n_kept * voxel_volume,
        "crop_lo": [int(v) for v in lo],
        "crop_hi": [int(v) for v in hi],
        "margin": margin,
        "block_average": [int(v) for v in f],
        "sigma": sigma,
        "level": 0.5,
        "surface_pieces": n_regions,
        **{
            key: vprov[key]
            for key in ("n_points", "n_triangles", "watertight", "enclosed_volume")
        },
        "spacing": vprov["spacing"],
        "notes": notes,
    }
    return surface, provenance


def load_volume(path: str | Path):
    """Load a volume by kind: .npy array, .nii/.nii.gz (nibabel), a DICOM series
    directory, or a single multi-frame .dcm file (both pydicom).  Returns
    (array, spacing, origin) with index (i, j, k) at ``origin + index*spacing``."""
    path = Path(path)
    if path.is_dir():
        return _load_dicom_series(path)
    if not path.is_file():
        raise FileNotFoundError(f"volume not found: {path}")
    name = path.name.lower()
    if name.endswith(".npy"):
        array = np.load(path, allow_pickle=False)
        return array, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
    if name.endswith(_NIFTI_SUFFIXES):
        return _load_nifti(path)
    if name.endswith(".dcm"):
        array, spacing, origin, _ = _read_dicom_multiframe(path, pixels=True)
        return array, spacing, origin
    raise ValueError(
        f"unsupported volume input {path.name!r}; supported: .npy, "
        f".nii/.nii.gz, a single multi-frame .dcm, or a DICOM series directory"
    )


def load_volume_affine(path: str | Path):
    """Load a volume with its FULL voxel -> world affine (D83).

    Same inputs as :func:`load_volume`, but returns ``(array, affine, info)``
    so a rotated or flipped NIfTI is mapped exactly instead of refused.
    ``info`` = ``{"length_unit": "mm" | None, "notes": [...]}``: DICOM is
    millimetres by the standard; NIfTI follows ``xyzt_units`` (metres and
    microns are converted to mm, an undeclared unit is assumed mm with a
    note); a ``.npy`` declares nothing (identity affine, unit None).
    """
    path = Path(path)
    if path.is_dir():
        array, spacing, origin = _load_dicom_series(path)
        return array, _diag_affine(spacing, origin), {"length_unit": "mm", "notes": []}
    if not path.is_file():
        raise FileNotFoundError(f"volume not found: {path}")
    name = path.name.lower()
    if name.endswith(".npy"):
        array = np.load(path, allow_pickle=False)
        return array, np.eye(4), {"length_unit": None, "notes": []}
    if name.endswith(_NIFTI_SUFFIXES):
        img = _open_nifti_checked(path)
        affine, notes = nifti_affine_mm(img)
        array = np.asarray(img.get_fdata(), dtype=float)
        return array, affine, {"length_unit": "mm", "notes": notes}
    if name.endswith(".dcm"):
        array, spacing, origin, _ = _read_dicom_multiframe(path, pixels=True)
        return array, _diag_affine(spacing, origin), {"length_unit": "mm", "notes": []}
    raise ValueError(
        f"unsupported volume input {path.name!r}; supported: .npy, "
        f".nii/.nii.gz, a single multi-frame .dcm, or a DICOM series directory"
    )


def override_spacing(affine, spacing: str) -> np.ndarray:
    """Replace an axis-aligned, positive affine's voxel sizes by ``"X,Y,Z"``
    (origin kept) — the ``.npy`` spacing override. A rotated or flipped
    affine is refused: an override there would silently re-orient anatomy."""
    parts = [s.strip() for s in str(spacing).split(",")]
    if len(parts) != 3:
        raise ValueError(f"spacing expects X,Y,Z, got {spacing!r}")
    sp = tuple(float(v) for v in parts)
    if not all(np.isfinite(s) and s > 0 for s in sp):
        raise ValueError(f"spacing must be positive, got {sp}")
    a = np.asarray(affine, dtype=float)
    lin = a[:3, :3]
    if (
        float(np.abs(lin - np.diag(np.diag(lin))).max()) > 1e-9 * max(1.0, float(np.abs(lin).max()))
        or np.any(np.diag(lin) <= 0)
    ):
        raise ValueError(
            "a spacing override applies to axis-aligned volumes (.npy inputs); "
            "this volume's affine is rotated or flipped"
        )
    return _diag_affine(sp, a[:3, 3])


def _diag_affine(spacing, origin) -> np.ndarray:
    a = np.diag([*(float(s) for s in spacing), 1.0])
    a[:3, 3] = [float(v) for v in origin]
    return a


#: NIfTI ``xyzt_units`` spatial code -> factor to millimetres.
_NIFTI_MM = {"mm": 1.0, "meter": 1000.0, "micron": 1e-3}


def nifti_mm_scale(header) -> tuple[float, str | None]:
    """``(factor to mm, note or None)`` from a NIfTI header's ``xyzt_units``.

    mm -> 1, no note. metres / microns -> 1000 / 0.001 with a loud note (the
    caller converts). An undeclared ('unknown') unit -> 1 with a loud note
    that millimetres are ASSUMED — the medical-imaging convention, and what
    TotalSegmentator assumes too.
    """
    xyz = header.get_xyzt_units()[0]
    if xyz == "mm":
        return 1.0, None
    if xyz in _NIFTI_MM:
        f = _NIFTI_MM[xyz]
        return f, f"NIfTI declares spatial unit '{xyz}'; converted to millimetres (x{f:g})"
    if xyz == "unknown":
        return 1.0, (
            "NIfTI declares no spatial unit (xyzt_units 'unknown'): ASSUMING "
            "millimetres, the medical-imaging convention — check the scale if "
            "this is not a medical scan"
        )
    raise ValueError(f"unsupported NIfTI spatial unit {xyz!r}")  # pragma: no cover


def nifti_affine_mm(img) -> tuple[np.ndarray, list[str]]:
    """The image's best affine (nibabel: sform, else qform) in millimetres,
    validated, plus loud notes (unit assumption/conversion, no orientation)."""
    notes: list[str] = []
    affine = np.array(img.affine, dtype=float)
    hdr = img.header
    if int(hdr["sform_code"]) == 0 and int(hdr["qform_code"]) == 0:
        notes.append(
            "NIfTI has no orientation (qform and sform codes are 0): nibabel's "
            "fallback affine is used, so world positions are not anatomical"
        )
    factor, note = nifti_mm_scale(hdr)
    if note:
        notes.append(note)
    affine[:3, :] *= factor
    return check_affine(affine), notes


def _open_nifti_checked(path: Path):
    """nibabel.load + the F-IMG-7 header-shape ceiling (before any data read)."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError(
            "NIfTI input needs nibabel — install with: pip install nibabel"
        ) from exc
    img = nib.load(str(path))
    # Bound the voxel count from the HEADER shape, without allocating: get_fdata()
    # materialises the whole array (and scales it to float64), so an oversized or
    # crafted NIfTI would exhaust memory before volume_to_surface's MAX_VOXELS gate
    # ever runs.  Refuse loud first (F-IMG-7).
    from meshforge.geometry.centerline import MAX_VOXELS

    shape = tuple(int(s) for s in img.header.get_data_shape())
    if int(np.prod(shape)) > MAX_VOXELS:
        raise ValueError(
            f"{path.name} volume {shape} exceeds {MAX_VOXELS} voxels; "
            f"downsample before ingest"
        )
    return img


# Axial patient orientation (row = +x, col = +y, slice normal = +z).  These GE
# cardiac CT reconstructions ship as ONE multi-frame file with axis-aligned axial
# frames; an oblique/rotated orientation is refused rather than silently
# mis-oriented (the same posture as the NIfTI affine check).
_AXIAL_IOP = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def dicom_grid(path: str | Path):
    """Geometry-only ``(dims, spacing, origin)`` of a multi-frame .dcm, WITHOUT
    reading the pixel data — for voxelising a segmentation onto the image grid."""
    _, spacing, origin, dims = _read_dicom_multiframe(Path(path), pixels=False)
    return dims, spacing, origin


def dicom_grid_checked(path: str | Path):
    """Read the .dcm ONCE and derive BOTH the rule-1 checksum and the
    geometry-only grid from the same buffer, so there is no window for the
    file to change between the checksum read and the parse read (the
    F-INFO-8 posture, applied to the ``--dicom`` anchor — closes the D62
    BACKLOG two-read). Returns ``(sha256_hexdigest, (dims, spacing, origin))``;
    the digest equals ``sha256(file bytes)`` (the same value
    ``volume_input_sha256`` computes)."""
    import hashlib

    p = Path(path)
    buf = p.read_bytes()
    sha = hashlib.sha256(buf).hexdigest()
    _, spacing, origin, dims = _read_dicom_multiframe(p, pixels=False, buf=buf)
    return sha, (dims, spacing, origin)


def _read_dicom_multiframe(path: Path, *, pixels: bool, buf: bytes | None = None):
    """Read a single multi-frame DICOM (classic SOP class + NumberOfFrames, no
    functional groups — the GE cardiac CT layout).  Frames march +z from
    ImagePositionPatient; the inter-frame step is SpacingBetweenSlices when
    present (the reconstruction interval — SliceThickness is the slab width and
    can differ), else SliceThickness.  Returns (array|None, spacing, origin,
    dims); ``array`` is (x, y, z) HU when ``pixels`` else None."""
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError(
            "DICOM input needs pydicom — install with: pip install pydicom"
        ) from exc

    # ``buf`` lets a caller (dicom_grid_checked) hand in the already-read
    # bytes so the file is read exactly once — checksum + parse from the
    # same buffer. ``path`` is still used for error messages.
    if buf is None:
        ds = pydicom.dcmread(str(path), stop_before_pixels=not pixels)
    else:
        import io as _io

        ds = pydicom.dcmread(_io.BytesIO(buf), stop_before_pixels=not pixels)
    nf = int(getattr(ds, "NumberOfFrames", 1) or 1)
    if nf < 2:
        raise ValueError(
            f"{path.name} has {nf} frame(s); a single-frame .dcm is not a "
            f"volume — point at the DICOM series directory instead"
        )
    if "PerFrameFunctionalGroupsSequence" in ds:
        raise ValueError(
            f"{path.name} is an enhanced multi-frame DICOM (per-frame functional "
            f"groups) — this loader handles the classic axial layout only"
        )
    iop = np.asarray([float(v) for v in getattr(ds, "ImageOrientationPatient",
                                                _AXIAL_IOP)], dtype=float)
    if iop.shape != (6,) or float(np.abs(iop - _AXIAL_IOP).max()) > 1e-4:
        raise ValueError(
            f"{path.name} has a non-axial ImageOrientationPatient {list(iop)} — "
            f"re-orient to axial (row=+x, col=+y) first; refusing to silently "
            f"mis-orient anatomy"
        )
    rows, cols = int(ds.Rows), int(ds.Columns)
    px_row, px_col = (float(v) for v in ds.PixelSpacing)  # (row=y, col=x)
    z_sp = float(getattr(ds, "SpacingBetweenSlices", 0.0) or 0.0)
    if z_sp <= 0:
        z_sp = float(getattr(ds, "SliceThickness", 0.0) or 0.0)
    if z_sp <= 0:
        raise ValueError(
            f"{path.name} lacks a usable slice spacing (SpacingBetweenSlices / "
            f"SliceThickness) — cannot place frames in space"
        )
    origin = tuple(float(v) for v in ds.ImagePositionPatient)
    spacing = (px_col, px_row, z_sp)  # (x, y, z)
    # PixelSpacing/origin are attacker-controlled header fields: a non-finite or
    # non-positive pixel pitch survives the `spacing <= 0` check on the anchoring
    # path (NaN is False) and voxelises a segmentation onto a degenerate lattice.
    # Validate at the source so dicom_grid itself fails loud (only z was guarded).
    if not all(np.isfinite(s) and s > 0 for s in spacing):
        raise ValueError(
            f"{path.name} has a non-finite or non-positive pixel spacing "
            f"{spacing} — refusing to place voxels on a degenerate grid"
        )
    if not all(np.isfinite(v) for v in origin):
        raise ValueError(
            f"{path.name} has a non-finite ImagePositionPatient {origin}"
        )
    dims = (cols, rows, nf)
    if not pixels:
        # geometry only (for grid-anchoring a segmentation) — nothing allocated,
        # so no voxel ceiling; the ROI crop bounds the actual mask.
        return None, spacing, origin, dims
    from meshforge.geometry.centerline import MAX_VOXELS

    if cols * rows * nf > MAX_VOXELS:
        raise ValueError(
            f"volume {dims} exceeds {MAX_VOXELS} voxels; downsample before ingest"
        )
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    # pixel_array is (frame=z, row=y, col=x); -> (x, y, z) to match the
    # origin + index*spacing convention used by volume_to_surface.
    hu = ds.pixel_array.astype(np.float32) * slope + intercept
    array = np.ascontiguousarray(hu.transpose(2, 1, 0))
    return array, spacing, origin, dims


def _load_nifti(path: Path):
    """``load_volume``'s NIfTI branch: (array, spacing, origin) can only say
    "axis-aligned, positive", so anything else is refused here — use
    :func:`load_volume_affine` (what ``segment`` calls),
    which maps rotated/flipped grids exactly (D83)."""
    img = _open_nifti_checked(path)
    affine = np.asarray(img.affine, dtype=float)
    rot = affine[:3, :3]
    off_diag = rot - np.diag(np.diag(rot))
    scale = max(1.0, float(np.abs(np.diag(rot)).max()))
    if float(np.abs(off_diag).max()) > 1e-6 * scale:
        raise ValueError(
            f"{path.name} has a rotated/sheared affine, which (spacing, origin) "
            f"cannot express — re-orient it first, or load it with "
            f"load_volume_affine; refusing to silently "
            f"mis-orient anatomy"
        )
    if np.any(np.diag(rot) <= 0):
        raise ValueError(
            f"{path.name} has non-positive affine scales {np.diag(rot)} — "
            f"re-orient it first, or load it with load_volume_affine"
        )
    array = np.asarray(img.get_fdata(), dtype=float)
    spacing = tuple(float(v) for v in np.diag(rot))
    origin = tuple(float(v) for v in affine[:3, 3])
    return array, spacing, origin


def _load_dicom_series(dirpath: Path):
    try:
        import pydicom
        from pydicom.errors import InvalidDicomError
    except ImportError as exc:
        raise RuntimeError(
            "DICOM input needs pydicom — install with: pip install pydicom"
        ) from exc
    files = sorted(p for p in dirpath.iterdir() if p.is_file())
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f))
        except InvalidDicomError:
            # A genuine non-DICOM straggler (no 'DICM' header) — skip it.  The
            # except is NARROW on purpose: a file that IS a DICOM yet is corrupt
            # raises a *different* error, which must SURFACE rather than being
            # swallowed into a silently-truncated series (F-IMG-6).
            continue
        if hasattr(ds, "PixelData"):
            slices.append(ds)
    if not slices:
        raise ValueError(f"no DICOM slices with pixel data found in {dirpath}")
    # Bound the stacked voxel count BEFORE reading any pixel data — dcmread is
    # lazy, so nothing large is allocated yet.  The multi-frame loader has this
    # ceiling; the directory-series path did not, so a crafted/oversized series
    # could np.stack straight into memory (a DoS).  Refuse loud once the declared
    # dims are known (F-IMG-6).
    from meshforge.geometry.centerline import MAX_VOXELS

    rows, cols = int(slices[0].Rows), int(slices[0].Columns)
    n_vox = rows * cols * len(slices)
    if n_vox > MAX_VOXELS:
        raise ValueError(
            f"DICOM series in {dirpath} declares {cols}x{rows}x{len(slices)} = "
            f"{n_vox} voxels, exceeding {MAX_VOXELS}; downsample before ingest"
        )
    try:
        slices.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
        zs = np.array([float(ds.ImagePositionPatient[2]) for ds in slices])
    except AttributeError as exc:
        raise ValueError(
            f"DICOM series in {dirpath} lacks ImagePositionPatient — cannot "
            f"order slices safely"
        ) from exc
    gaps = np.diff(zs)
    if len(gaps) == 0:
        raise ValueError(f"DICOM series in {dirpath} has a single slice")
    if gaps.min() <= 0 or (gaps.max() - gaps.min()) > 0.01 * gaps.mean():
        raise ValueError(
            f"DICOM series in {dirpath} has non-uniform slice spacing "
            f"({gaps.min():.6g}..{gaps.max():.6g}) — resample first"
        )
    first = slices[0]
    row_sp, col_sp = (float(v) for v in first.PixelSpacing)
    stack = []
    for ds in slices:
        arr = ds.pixel_array.astype(float)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        stack.append(arr * slope + intercept)
    # pixel_array is (row, col) = (y, x); build (x, y, z).
    array = np.stack(stack, axis=-1).transpose(1, 0, 2)
    spacing = (col_sp, row_sp, float(gaps.mean()))
    origin = tuple(float(v) for v in first.ImagePositionPatient)
    return array, spacing, origin


def volume_input_sha256(path: str | Path) -> str:
    """Checksum for rule-1 bookkeeping: file sha256, or for a DICOM
    directory the sha256 of the sorted per-file digests."""
    from meshforge.io.load import file_sha256

    path = Path(path)
    if path.is_file():
        return file_sha256(path)
    if path.is_dir():
        digest = hashlib.sha256()
        for f in sorted(p for p in path.iterdir() if p.is_file()):
            digest.update(f.name.encode())
            digest.update(bytes.fromhex(file_sha256(f)))
        return digest.hexdigest()
    raise FileNotFoundError(f"volume not found: {path}")
