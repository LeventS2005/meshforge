"""Centreline flow section: a curved cut that follows the vessel (D82).

A CFD result rendered as its boundary shows the no-slip wall (U = 0 there),
which hides the flow. A planar clip does not fix it for real anatomy: a
real aortic-arch centreline left its best-fit plane by 36.6 mm against a
10.6-20 mm radius, so a plane misses most of the lumen (session 44).

This builds the curved-planar-reformation surface instead: at every
centreline point a straight row across the cross-section, along
``w = t x d`` where ``t`` is the tangent and ``d`` the view direction (the
normal of the centreline's best-fit plane). Every row contains the
centreline and is face-on to ``d``, so the section never misses the lumen
and reads flat from the camera wherever the vessel is not pointing at it.

Each row is trimmed to the volume by marching outward from the centreline
and stopping at the first sample outside it - so a row that would cross into
ANOTHER part of the vessel (the descending aorta behind the arch) stops at
the wall it leaves through. Only cells whose four corners survive are kept,
so every point of the section lies inside the volume and samples cleanly.

The result is geometry only (no fields): the display layer samples whichever
volume it is showing onto it, so animation frames reuse one section.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RibbonSection:
    surface: object  # pv.PolyData, quads, no point data
    view_dir: tuple[float, float, float]  # unit; the section faces it
    n_rows: int  # centreline stations that contributed a row
    notes: list[str] = field(default_factory=list)


def centerline_ribbon(
    centerline,
    volume,
    *,
    n_across: int = 81,
    margin: float = 2.0,
) -> RibbonSection:
    """Section surface through *volume* along *centerline* (see module doc).

    ``margin`` scales each station's inscribed radius into the row's
    half-width; it must reach past the wall of non-circular sections (the
    inscribed radius is the SMALLEST distance to it), and the outward march
    trims whatever overshoots. ``n_across`` (odd) samples per row.
    """
    import pyvista as pv

    pts = np.asarray(centerline.points, dtype=float)
    tan = np.asarray(centerline.tangents, dtype=float)
    radii = np.asarray(centerline.radii, dtype=float)
    n = len(pts)
    if n < 2:
        raise ValueError("centreline has fewer than 2 points - no section")
    if n_across < 3 or n_across % 2 == 0:
        raise ValueError(f"n_across must be odd and >= 3, got {n_across}")
    if not margin > 1.0:
        raise ValueError(f"margin must exceed 1 (rows must reach the wall), got {margin}")

    centre = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    d = vt[-1]
    if d[np.argmax(np.abs(d))] < 0.0:  # deterministic sign
        d = -d

    w = np.cross(tan, d)
    w_len = np.linalg.norm(w, axis=1)
    ok = w_len > 0.1  # tangent within ~84 deg of the view direction
    if not ok.any():
        raise ValueError(
            "the centreline runs along its own view direction everywhere - "
            "no face-on section exists"
        )
    w[ok] /= w_len[ok, None]
    notes: list[str] = []
    if not ok.all():
        good = np.flatnonzero(ok)
        nearest = good[np.abs(np.arange(n)[:, None] - good[None, :]).argmin(axis=1)]
        w[~ok] = w[nearest[~ok]]
        notes.append(
            f"{int((~ok).sum())} station(s) point at the camera; their rows "
            f"borrow a neighbour's direction and are seen edge-on"
        )
    for i in range(1, n):  # keep the row direction continuous
        if np.dot(w[i], w[i - 1]) < 0.0:
            w[i] = -w[i]

    s = np.linspace(-1.0, 1.0, n_across)
    half = margin * radii
    grid_pts = pts[:, None, :] + (s[None, :, None] * half[:, None, None]) * w[:, None, :]
    flat = grid_pts.reshape(-1, 3)
    probe = pv.PolyData(flat).sample(volume)
    valid = np.asarray(probe.point_data["vtkValidPointMask"]).astype(bool)
    valid = valid.reshape(n, n_across)

    keep = np.zeros_like(valid)
    mid = n_across // 2
    for i in range(n):
        if not valid[i, mid]:
            continue  # the centreline is outside the volume here
        j = mid
        while j >= 0 and valid[i, j]:
            keep[i, j] = True
            j -= 1
        j = mid + 1
        while j < n_across and valid[i, j]:
            keep[i, j] = True
            j += 1
    n_rows = int(keep[:, mid].sum())
    if n_rows < n:
        notes.append(f"{n - n_rows} centreline point(s) lie outside the volume")

    pairs = [(i, i + 1) for i in range(n - 1)]
    if centerline.closed:
        pairs.append((n - 1, 0))
    faces = []
    for a, b in pairs:
        for j in range(n_across - 1):
            if keep[a, j] and keep[a, j + 1] and keep[b, j + 1] and keep[b, j]:
                faces.append(
                    (4, a * n_across + j, a * n_across + j + 1,
                     b * n_across + j + 1, b * n_across + j)
                )
    if not faces:
        raise ValueError("the centreline does not run inside the volume - no section")
    surface = pv.PolyData(flat, np.asarray(faces, dtype=np.int64).ravel())
    surface = surface.clean()  # drop the trimmed-away points
    return RibbonSection(
        surface=surface,
        view_dir=tuple(float(c) for c in d),
        n_rows=n_rows,
        notes=notes,
    )
