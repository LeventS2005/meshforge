"""V14 OpenFOAM polyMesh export: hand-built exactness, tube invariants,
loud failure branches, CLI e2e; checkMesh-gated validation (trap #16)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from meshforge.export.foam import write_polymesh
from meshforge.meshing.base import BoundaryGroup, WallMesh
from meshforge.quality.volume import volume_quality

# ---------------------------------------------------------------------------
# minimal readers for the ASCII files we write


def _foam_body(path: Path) -> list[str]:
    """Entries between the count line's '(' and the closing ')'."""
    lines = path.read_text().splitlines()
    start = lines.index("(")
    end = len(lines) - 1 - lines[::-1].index(")")
    n = int(lines[start - 1])
    body = lines[start + 1 : end]
    assert len(body) >= n  # boundary bodies span multiple lines per entry
    return body


def _read_labels(path: Path) -> list[int]:
    return [int(x) for x in _foam_body(path)]


def _read_points(path: Path) -> np.ndarray:
    return np.array(
        [[float(v) for v in line.strip("()").split()] for line in _foam_body(path)]
    )


def _read_faces(path: Path) -> list[list[int]]:
    out = []
    for line in _foam_body(path):
        m = re.fullmatch(r"(\d+)\(([\d ]+)\)", line.strip())
        assert m, line
        verts = [int(v) for v in m.group(2).split()]
        assert len(verts) == int(m.group(1))
        out.append(verts)
    return out


def _read_boundary(path: Path) -> dict[str, dict[str, str]]:
    text = path.read_text()
    patches = {}
    for m in re.finditer(r"(\w+)\s*\{([^}]*)\}", text):
        name, body = m.group(1), m.group(2)
        if name == "FoamFile":
            continue
        patches[name] = dict(re.findall(r"(\w+)\s+([^;]+);", body))
    return patches


def _boundary_volume(points: np.ndarray, faces: list[list[int]], n_internal: int) -> float:
    """Divergence theorem over the boundary faces (fan triangulation):
    only correct if every boundary face is wound outward."""
    vol = 0.0
    for verts in faces[n_internal:]:
        p = points[verts]
        for i in range(1, len(verts) - 1):
            vol += np.linalg.det(np.stack([p[0], p[i], p[i + 1]])) / 6.0
    return vol


# ---------------------------------------------------------------------------
# hand-built fixtures


def _cube_wall() -> WallMesh:
    pts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float,
    )
    grid = pv.UnstructuredGrid(
        np.array([8, 0, 1, 2, 3, 4, 5, 6, 7], np.int64),
        np.array([pv.CellType.HEXAHEDRON], np.uint8), pts,
    )
    faces = {
        "bottom": [0, 1, 2, 3], "top": [4, 5, 6, 7], "inlet": [0, 1, 5, 4],
        "right": [1, 2, 6, 5], "outlet": [2, 3, 7, 6], "left": [3, 0, 4, 7],
    }
    groups = {
        name: BoundaryGroup(name, np.array([0]), np.array([nodes]))
        for name, nodes in faces.items()
    }
    return WallMesh(grid=grid, boundary=groups, backend="manual",
                    quality=volume_quality(grid))


def _two_tet_wall(*, drop_group_face: bool = False) -> WallMesh:
    pts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float
    )
    grid = pv.UnstructuredGrid(
        np.array([4, 0, 1, 2, 3, 4, 1, 2, 3, 4], np.int64),
        np.array([pv.CellType.TETRA, pv.CellType.TETRA], np.uint8), pts,
    )
    shell = [[0, 2, 1], [0, 1, 3], [0, 3, 2],  # tet-1 boundary
             [1, 2, 4], [2, 3, 4], [1, 4, 3]]  # tet-2 boundary
    if drop_group_face:
        shell = shell[:-1]
    groups = {"shell": BoundaryGroup("shell", np.zeros(len(shell), int),
                                     np.array(shell))}
    return WallMesh(grid=grid, boundary=groups, backend="manual",
                    quality=volume_quality(grid))


# ---------------------------------------------------------------------------
# exactness on hand-built meshes


def test_single_cube_exact(tmp_path):
    polymesh = write_polymesh(_cube_wall(), tmp_path / "case")
    assert polymesh == tmp_path / "case" / "constant" / "polyMesh"
    pts = _read_points(polymesh / "points")
    faces = _read_faces(polymesh / "faces")
    owner = _read_labels(polymesh / "owner")
    neighbour = _read_labels(polymesh / "neighbour")
    assert pts.shape == (8, 3) and len(faces) == 6
    assert owner == [0] * 6 and neighbour == []
    assert _boundary_volume(pts, faces, 0) == pytest.approx(1.0)

    patches = _read_boundary(polymesh / "boundary")
    assert list(patches) == ["bottom", "top", "inlet", "right", "outlet", "left"]
    starts = [int(p["startFace"]) for p in patches.values()]
    counts = [int(p["nFaces"]) for p in patches.values()]
    assert counts == [1] * 6 and starts == list(range(6))  # contiguous
    assert patches["inlet"]["type"] == "patch"   # heuristic
    assert patches["outlet"]["type"] == "patch"
    assert patches["bottom"]["type"] == "wall"
    # owner header note carries the checkMesh-friendly counts.
    assert 'note        "nPoints:8 nCells:1 nFaces:6 nInternalFaces:0"' in (
        polymesh / "owner").read_text()


def test_two_tets_internal_face_owner_orientation(tmp_path):
    wall = _two_tet_wall()
    polymesh = write_polymesh(wall, tmp_path / "case")
    pts = _read_points(polymesh / "points")
    faces = _read_faces(polymesh / "faces")
    owner = _read_labels(polymesh / "owner")
    neighbour = _read_labels(polymesh / "neighbour")
    assert len(faces) == 7 and neighbour == [1] and owner[0] == 0

    # Internal face normal must point owner -> neighbour.
    f = pts[faces[0]]
    normal = np.cross(f[1] - f[0], f[2] - f[0])
    centroids = [pts[[0, 1, 2, 3]].mean(0), pts[[1, 2, 3, 4]].mean(0)]
    assert float(normal @ (centroids[1] - centroids[0])) > 0
    # Material volume via divergence over the outward boundary faces.
    assert _boundary_volume(pts, faces, 1) == pytest.approx(
        wall.quality.total_volume
    )


def test_patch_type_override_and_unknown_target(tmp_path):
    polymesh = write_polymesh(
        _cube_wall(), tmp_path / "a", patch_types={"bottom": "symmetry"}
    )
    assert _read_boundary(polymesh / "boundary")["bottom"]["type"] == "symmetry"
    with pytest.raises(ValueError, match="unknown patch"):
        write_polymesh(_cube_wall(), tmp_path / "b", patch_types={"nope": "wall"})


# ---------------------------------------------------------------------------
# loud failure branches


def test_uncovered_boundary_face_is_loud_unless_default_patch(tmp_path):
    wall = _two_tet_wall(drop_group_face=True)
    with pytest.raises(ValueError, match="covered by no boundary group"):
        write_polymesh(wall, tmp_path / "case")
    polymesh = write_polymesh(wall, tmp_path / "case2", default_patch="rest")
    patches = _read_boundary(polymesh / "boundary")
    assert patches["rest"]["nFaces"] == "1"
    assert patches["shell"]["nFaces"] == "5"


def test_stale_group_face_and_duplicate_claim_are_loud(tmp_path):
    wall = _cube_wall()
    wall.boundary["bottom"] = BoundaryGroup(
        "bottom", np.array([0, 0]), np.array([[0, 1, 2, 3], [0, 1, 2, 4]])
    )
    with pytest.raises(ValueError, match="not on the mesh boundary"):
        write_polymesh(wall, tmp_path / "case")

    wall2 = _cube_wall()
    wall2.boundary["top"] = BoundaryGroup(
        "top", np.array([0, 0]), np.array([[4, 5, 6, 7], [0, 1, 2, 3]])
    )
    with pytest.raises(ValueError, match="claimed by both"):
        write_polymesh(wall2, tmp_path / "case2")


def test_nonmanifold_face_is_loud(tmp_path):
    """Three coincident tets: a face shared by >2 cells must be refused."""
    pts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
    )
    cells = np.array([4, 0, 1, 2, 3] * 3, np.int64)
    grid = pv.UnstructuredGrid(
        cells, np.array([pv.CellType.TETRA] * 3, np.uint8), pts
    )
    wall = WallMesh(grid=grid, boundary={}, backend="manual",
                    quality=volume_quality(grid))
    with pytest.raises(ValueError, match="non-manifold"):
        write_polymesh(wall, tmp_path / "case")


def test_unsupported_cell_type_is_loud(tmp_path):
    pts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]], dtype=float
    )
    grid = pv.UnstructuredGrid(
        np.array([5, 0, 1, 2, 3, 4], np.int64),
        np.array([pv.CellType.PYRAMID], np.uint8), pts,
    )
    wall = WallMesh(grid=grid, boundary={}, backend="manual",
                    quality=volume_quality(grid))
    with pytest.raises(ValueError, match="hexahedra, tetrahedra and wedges"):
        write_polymesh(wall, tmp_path / "case")


# ---------------------------------------------------------------------------
# real swept wall: global invariants


@pytest.fixture(scope="module")
def tube_wall() -> WallMesh:
    from meshforge.geometry import compute_centerline
    from meshforge.meshing import get_backend
    from meshforge.synth import generate

    surface, _ = generate("straight_tube")
    cl = compute_centerline(surface)
    return get_backend("swept").mesh_wall(
        surface, 1.0, centerline=cl, n_theta=16, n_axial=8, n_layers=3
    )


def test_tube_polymesh_invariants(tmp_path, tube_wall):
    polymesh = write_polymesh(tube_wall, tmp_path / "tube")
    pts = _read_points(polymesh / "points")
    faces = _read_faces(polymesh / "faces")
    owner = _read_labels(polymesh / "owner")
    neighbour = _read_labels(polymesh / "neighbour")
    n_internal = len(neighbour)

    assert len(owner) == len(faces)
    # Face conservation: 6 faces per hex, each internal shared by two.
    assert 6 * tube_wall.grid.n_cells == 2 * n_internal + (len(faces) - n_internal)
    # Upper-triangular ordering + owner < neighbour on every internal face.
    pairs = list(zip(owner[:n_internal], neighbour))
    assert all(o < n for o, n in pairs)
    assert pairs == sorted(pairs)
    # Patch faces = boundary-group faces, patch by patch.
    patches = _read_boundary(polymesh / "boundary")
    assert {k: int(p["nFaces"]) for k, p in patches.items()} == {
        k: g.n_faces for k, g in tube_wall.boundary.items()
    }
    starts = [int(p["startFace"]) for p in patches.values()]
    counts = [int(p["nFaces"]) for p in patches.values()]
    assert starts[0] == n_internal
    for i in range(1, len(starts)):
        assert starts[i] == starts[i - 1] + counts[i - 1]
    assert starts[-1] + counts[-1] == len(faces)
    # Outward orientation everywhere: divergence volume == VTK cell volume.
    assert _boundary_volume(pts, faces, n_internal) == pytest.approx(
        tube_wall.quality.total_volume, rel=0.02
    )
    assert patches["inlet"]["type"] == "patch"
    assert patches["inner"]["type"] == "wall"


def test_session_save_foam_journaled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from meshforge.pipeline import MeshingSession

    session = MeshingSession()
    session.generate_synth("straight_tube")
    session.compute_centerline()
    session.compute_volmesh(thickness=1.0, n_theta=12, n_axial=4, n_layers=2)
    result = session.save("foam", tmp_path / "case")
    polymesh = Path(result.summary["path"])
    for obj in ("points", "faces", "owner", "neighbour", "boundary"):
        assert (polymesh / obj).is_file(), obj
    assert session.journal[-1]["step"] == "save"
    assert session.journal[-1]["params"]["what"] == "foam"


# ---------------------------------------------------------------------------
# checkMesh-gated validation (skips cleanly without OpenFOAM — trap #16)


def test_checkmesh_validates_tube(tmp_path, tube_wall):
    from meshforge.solve import check_mesh, find_foam

    if find_foam() is None:
        pytest.skip("OpenFOAM not available")
    case = tmp_path / "case"
    write_polymesh(tube_wall, case)
    # check_mesh writes the minimal system/ stubs itself and raises unless
    # checkMesh reports "Mesh OK".
    report = check_mesh(case)
    assert report["ok"] is True
    assert (case / "log.checkMesh").is_file()
