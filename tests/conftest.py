from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

SPHERE_RADIUS = 7.5


# ---------------------------------------------------------------------------
# machine-local data directories
#
# The real clinical inputs live OUTSIDE the repo (rule 1: read-only, never
# copied in) and their absolute paths are patient/site identifying, so no path
# is hard-coded here.  Each directory is resolved at collection time from, in
# order: an environment variable, then a git-ignored one-line pointer file next
# to this conftest, then nothing -> None, and the tests that need real data skip
# exactly as they do on a machine that has none.
# ---------------------------------------------------------------------------
def local_data_dir(env_var: str, pointer_name: str) -> Path | None:
    """Resolve a local data directory, or None when it is not configured.

    ``env_var`` wins; otherwise the first non-empty line of
    ``tests/<pointer_name>`` (git-ignored) is used.  Returns None unless the
    resolved path is an existing directory.
    """
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        pointer = Path(__file__).parent / pointer_name
        if pointer.is_file():
            for line in pointer.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    raw = line.strip()
                    break
    if not raw:
        return None
    d = Path(raw).expanduser()
    return d if d.is_dir() else None


def dicom_dir() -> Path | None:
    """Directory holding the matching real .dcm volumes (env MESHFORGE_DICOM_DIR)."""
    return local_data_dir("MESHFORGE_DICOM_DIR", ".local_dicom_dir")


def first_local_file(directory: Path | None, pattern: str) -> Path | None:
    """First (sorted) match of ``pattern`` in ``directory``; None when absent.

    Case files are addressed by their pseudonymous study code only — the rest
    of each real filename never appears in the repo.
    """
    if directory is None:
        return None
    matches = sorted(p for p in directory.glob(pattern) if p.is_file())
    return matches[0] if matches else None


@pytest.fixture()
def sphere_stl(tmp_path: Path) -> Path:
    """Watertight all-triangle sphere with known analytic area/volume."""
    sphere = pv.Sphere(radius=SPHERE_RADIUS, theta_resolution=60, phi_resolution=60)
    path = tmp_path / "sphere.stl"
    sphere.save(path)
    return path


@pytest.fixture()
def open_plane_ply(tmp_path: Path) -> Path:
    """Open (non-watertight) triangulated plane — volume must be refused."""
    plane = pv.Plane(i_resolution=4, j_resolution=4).triangulate()
    path = tmp_path / "plane.ply"
    plane.save(path)
    return path


@pytest.fixture()
def nonmanifold_vtp(tmp_path: Path) -> Path:
    """Three triangles sharing one edge: closed-edge count is fine per pair,
    but the shared edge is non-manifold — watertightness must be refused."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, -1.0, 0.0],
            [0.5, 0.0, 1.0],
        ]
    )
    faces = np.hstack([[3, 0, 1, 2], [3, 0, 1, 3], [3, 0, 1, 4]])
    mesh = pv.PolyData(points, faces)
    path = tmp_path / "nonmanifold.vtp"
    mesh.save(path)
    return path
