"""V23 in-app CFD solve: case-builder exactness + loud validation
(ungated), then OpenFOAM-gated runs (skip cleanly without it, trap #16):
the Hagen-Poiseuille pipe benchmark, an icoFoam smoke run, the V22 wedge
boundary-layer case through checkMesh, and the CLI e2e."""
from __future__ import annotations

import numpy as np
import pytest

from meshforge.solve.foam_case import (
    CfdBC,
    CfdSpec,
    read_patch_names,
    write_case,
    write_case_files,
)
from meshforge.solve.foam_runner import find_foam, run_foam_app, solve_case

# ---------------------------------------------------------------------------
# fixtures: the fluid cylinder (O-grid disk, extruded) used throughout


def _disk_wall(target_size: float, length: float, n_layers: int):
    """Fluid cylinder: the kept O-grid disk extruded along +z into hexes with
    the named groups front (z=0), back (z=length), outer_rim, hole_rims.
    (Test-local input builder; the library's block extruder is not part of
    the demo edition.)"""
    import pyvista as pv

    from meshforge.meshing import BoundaryGroup, WallMesh, ogrid_disk
    from meshforge.quality.volume import volume_quality

    theta = np.linspace(0, 2 * np.pi, 129)[:-1]
    circle = np.column_stack([np.cos(theta), np.sin(theta)])  # R = 1, CCW
    bm = ogrid_disk([circle], target_size=target_size)

    n2d = len(bm.points)
    n_quads = len(bm.quads)
    zs = np.linspace(0.0, length, n_layers + 1)
    points = np.column_stack(
        [
            np.tile(bm.points[:, 0], n_layers + 1),
            np.tile(bm.points[:, 1], n_layers + 1),
            np.repeat(zs, n2d),
        ]
    )
    hexes = np.empty((n_quads * n_layers, 8), dtype=np.int64)
    for k in range(n_layers):
        hexes[k * n_quads : (k + 1) * n_quads] = np.hstack(
            [bm.quads + k * n2d, bm.quads + (k + 1) * n2d]
        )
    cells = np.column_stack([np.full(len(hexes), 8, dtype=np.int64), hexes]).ravel()
    celltypes = np.full(len(hexes), pv.CellType.HEXAHEDRON, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, points)
    grid.cell_data["region"] = np.ones(grid.n_cells, dtype=np.int32)

    edge_owner: dict[tuple[int, int], int] = {}
    for qi, quad in enumerate(bm.quads):
        for e in range(4):
            a, b = int(quad[e]), int(quad[(e + 1) % 4])
            edge_owner[(min(a, b), max(a, b))] = qi

    def rim_group(name: str, edges: np.ndarray) -> BoundaryGroup:
        cell_ids, face_nodes = [], []
        for a, b in edges:
            qi = edge_owner[(min(int(a), int(b)), max(int(a), int(b)))]
            for k in range(n_layers):
                cell_ids.append(k * n_quads + qi)
                face_nodes.append(
                    [
                        int(a) + k * n2d,
                        int(b) + k * n2d,
                        int(b) + (k + 1) * n2d,
                        int(a) + (k + 1) * n2d,
                    ]
                )
        return BoundaryGroup(
            name=name,
            cell_ids=np.asarray(cell_ids, dtype=np.int64),
            face_nodes=np.asarray(face_nodes, dtype=np.int64).reshape(-1, 4),
        )

    boundary = {
        "front": BoundaryGroup(
            name="front",
            cell_ids=np.arange(n_quads, dtype=np.int64),
            face_nodes=bm.quads[:, ::-1].copy(),  # outward -z
        ),
        "back": BoundaryGroup(
            name="back",
            cell_ids=np.arange(
                (n_layers - 1) * n_quads, n_layers * n_quads, dtype=np.int64
            ),
            face_nodes=(bm.quads + n_layers * n2d).copy(),  # outward +z
        ),
        "outer_rim": rim_group("outer_rim", bm.edge_groups["outer_rim"]),
        "hole_rims": rim_group("hole_rims", bm.edge_groups["hole_rims"]),
    }
    return WallMesh(
        grid=grid,
        boundary=boundary,
        backend="ogrid-test",
        quality=volume_quality(grid),
        notes=list(bm.notes),
    )


def _pipe_spec(**overrides) -> CfdSpec:
    base = dict(
        solver="simpleFoam",
        nu=0.1,
        bcs={
            "front": CfdBC("velocity", (0.0, 0.0, 1.0)),
            "back": CfdBC("pressure", 0.0),
        },
        end_time=3000.0,
    )
    base.update(overrides)
    return CfdSpec(**base)


# ---------------------------------------------------------------------------
# spec validation is loud (ungated)


def test_bc_validation_is_loud():
    with pytest.raises(ValueError, match="unknown BC kind"):
        CfdBC("inflow", (0, 0, 1))
    with pytest.raises(ValueError, match=r"\(vx, vy, vz\)"):
        CfdBC("velocity", (1.0, 2.0))
    with pytest.raises(ValueError, match="scalar"):
        CfdBC("pressure")
    with pytest.raises(ValueError, match="takes no value"):
        CfdBC("wall", 1.0)


def test_spec_validation_is_loud():
    with pytest.raises(ValueError, match="unknown solver"):
        _pipe_spec(solver="pimpleFoam")
    with pytest.raises(ValueError, match="viscosity"):
        _pipe_spec(nu=0.0)
    with pytest.raises(ValueError, match="end_time"):
        _pipe_spec(end_time=-1.0)
    with pytest.raises(ValueError, match="delta_t"):
        _pipe_spec(solver="icoFoam")  # transient without a time step
    with pytest.raises(ValueError, match="no pressure BC"):
        CfdSpec(bcs={"front": CfdBC("velocity", (0, 0, 1))})


def test_resolve_defaults_and_refusals():
    spec = _pipe_spec()
    resolved = spec.resolve(["front", "back", "outer_rim"])
    assert resolved["outer_rim"].kind == "wall"  # unnamed -> wall default
    with pytest.raises(ValueError, match="unknown patch"):
        spec.resolve(["front"])  # 'back' from the spec is not on the mesh
    # A patch NAMED inlet/outlet with no BC must never silently become a wall.
    with pytest.raises(ValueError, match="flow boundary"):
        spec.resolve(["front", "back", "my_inlet"])


# ---------------------------------------------------------------------------
# written case files (ungated)


@pytest.fixture(scope="module")
def coarse_case(tmp_path_factory):
    case = tmp_path_factory.mktemp("cfd") / "pipe"
    wall = _disk_wall(0.3, 10.0, 20)
    kinds = write_case(wall, case, _pipe_spec())
    return case, kinds


def test_write_case_files_and_patch_kinds(coarse_case):
    case, kinds = coarse_case
    assert kinds == {"front": "velocity", "back": "pressure", "outer_rim": "wall"}
    for rel in (
        "system/controlDict", "system/fvSchemes", "system/fvSolution",
        "constant/transportProperties", "constant/turbulenceProperties",
        "constant/polyMesh/boundary", "0/U", "0/p", "case.foam",
    ):
        assert (case / rel).is_file(), rel
    control = (case / "system" / "controlDict").read_text()
    assert "application     simpleFoam;" in control
    # flow boundaries exported as type patch, the rim as a wall
    boundary = (case / "constant" / "polyMesh" / "boundary").read_text()
    assert read_patch_names(case) == ["front", "back", "outer_rim"]
    assert boundary.count("type            patch;") == 2
    assert boundary.count("type            wall;") == 1


def test_field_files_carry_dims_and_bcs(coarse_case):
    case, _ = coarse_case
    u = (case / "0" / "U").read_text()
    assert "[0 1 -1 0 0 0 0]" in u
    assert "front { type fixedValue; value uniform (0 0 1); }" in u
    assert "outer_rim { type noSlip; }" in u
    assert "back { type zeroGradient; }" in u
    p = (case / "0" / "p").read_text()
    assert "[0 2 -2 0 0 0 0]" in p
    assert "back { type fixedValue; value uniform 0; }" in p
    nu = (case / "constant" / "transportProperties").read_text()
    assert "nu              [0 2 -1 0 0 0 0] 0.1;" in nu


def test_icofoam_case_files(tmp_path):
    case = tmp_path / "ico"
    wall = _disk_wall(0.3, 10.0, 10)
    write_case(
        wall, case, _pipe_spec(solver="icoFoam", end_time=0.5, delta_t=0.01)
    )
    control = (case / "system" / "controlDict").read_text()
    assert "application     icoFoam;" in control
    assert "deltaT          0.01;" in control
    assert "PISO" in (case / "system" / "fvSolution").read_text()


def test_write_case_files_needs_polymesh(tmp_path):
    with pytest.raises(FileNotFoundError, match="polyMesh boundary"):
        write_case_files(tmp_path / "empty", _pipe_spec())


def test_rebuilt_case_carries_no_stale_results(tmp_path):
    """Stale time dirs from a previous solve would win max(time) at
    read-back (V26 regression: a finer mesh reusing outputs/cfd/session
    read the OLD run's fields) — a rebuilt case must shed them."""
    case = tmp_path / "pipe"
    wall = _disk_wall(0.3, 10.0, 5)
    write_case(wall, case, _pipe_spec())
    (case / "302" / "uniform").mkdir(parents=True)
    (case / "302" / "U").write_text("stale fields")
    (case / "log.simpleFoam").write_text("stale log")

    write_case(wall, case, _pipe_spec())
    assert not (case / "302").exists()
    assert not (case / "log.simpleFoam").exists()
    assert (case / "0" / "U").is_file()  # initial fields rebuilt
    assert (case / "constant" / "polyMesh" / "boundary").is_file()


# ---------------------------------------------------------------------------
# runner plumbing (ungated)


def test_find_foam_env_override(monkeypatch, tmp_path):
    exe = tmp_path / "openfoam"
    exe.write_text("#!/bin/sh\n")
    monkeypatch.setenv("MESHFORGE_OPENFOAM", str(exe))
    assert find_foam() == [str(exe), "--"]
    monkeypatch.setenv("MESHFORGE_OPENFOAM", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="missing file"):
        find_foam()


def test_find_foam_absent(monkeypatch):
    monkeypatch.delenv("MESHFORGE_OPENFOAM", raising=False)
    monkeypatch.setenv("PATH", "")
    assert find_foam() is None


def test_run_foam_app_without_openfoam(monkeypatch, tmp_path):
    monkeypatch.delenv("MESHFORGE_OPENFOAM", raising=False)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(RuntimeError, match="OpenFOAM not found"):
        run_foam_app("checkMesh", tmp_path)
    with pytest.raises(FileNotFoundError, match="case directory"):
        run_foam_app("checkMesh", tmp_path / "nope", foam=[])


def test_run_foam_app_refuses_unknown_applications(tmp_path):
    # A hostile controlDict must not turn the launcher into an arbitrary
    # command runner (it will exec anything on PATH).
    with pytest.raises(ValueError, match="not on the allowed list"):
        run_foam_app("bash", tmp_path, foam=[])


def test_solve_case_needs_controldict(tmp_path):
    case = tmp_path / "bare"
    case.mkdir()
    with pytest.raises(FileNotFoundError, match="controlDict"):
        solve_case(case)


def test_session_cfd_knob_errors():
    from meshforge.pipeline import MeshingSession

    session = MeshingSession()
    with pytest.raises(ValueError, match="already carries its BCs"):
        session.compute_cfd(case="somewhere", inlet="front=0,0,1")
    with pytest.raises(RuntimeError, match="volmesh"):
        session.compute_cfd(inlet="front=0,0,1", outlet="back=0")
    with pytest.raises(ValueError, match="NAME=vx,vy,vz"):
        session.compute_cfd(inlet="front")
    with pytest.raises(ValueError, match="NAME=VALUE"):
        session.compute_cfd(inlet="front=0,0,1", outlet="back")


# ---------------------------------------------------------------------------
# OpenFOAM-gated runs (trap #16: skip cleanly when absent)

needs_foam = pytest.mark.skipif(
    find_foam() is None, reason="OpenFOAM not available"
)


def _run_hagen_poiseuille(wall):
    """Solve the pipe case on the given fluid cylinder and assert the
    exact-solution truths (run on the O-grid pipe)."""
    from meshforge.pipeline import MeshingSession

    session = MeshingSession()
    session.volmesh = wall
    result = session.compute_cfd(
        solver="simpleFoam",
        nu=0.1,
        inlet="front=0,0,1",
        outlet="back=0",
        end_time=3000.0,
    )
    assert result.summary["solver"] == "simpleFoam"
    assert any("converged" in n for n in result.summary["notes"])
    assert session.journal[-1]["step"] == "cfd"

    grid = session.cfd.grid
    assert {"U", "p", "U_mag"} <= set(grid.point_data)

    # centerline velocity on the developed tail vs 2*U_mean
    line = grid.sample_over_line((0, 0, 6.0), (0, 0, 9.5), resolution=100)
    u_center = float(line["U"][:, 2].mean())
    assert abs(u_center - 2.0) / 2.0 < 0.05  # measured -2.3% at ts=0.15

    # radial profile at z=8 vs the parabola u = 2*U_mean*(1 - r^2)
    prof = grid.sample_over_line((-0.999, 0, 8.0), (0.999, 0, 8.0), resolution=100)
    r = prof.points[:, 0]
    u = prof["U"][:, 2]
    ana = 2.0 * (1.0 - r**2)
    ok = np.isfinite(u)
    assert np.abs(u[ok] - ana[ok]).max() < 0.10  # measured 0.06 at ts=0.15

    # kinematic pressure gradient on z in [5, 9] vs -8*nu*U_mean/R^2 = -0.8
    ptail = grid.sample_over_line((0, 0, 5.0), (0, 0, 9.0), resolution=100)
    pz = ptail["p"]
    zz = ptail.points[np.isfinite(pz), 2]
    slope = float(np.polyfit(zz, pz[np.isfinite(pz)], 1)[0])
    assert abs(slope - (-0.8)) / 0.8 < 0.05  # measured -1.2% at ts=0.15


@needs_foam
def test_hagen_poiseuille_ogrid_benchmark():
    """V26: the same exact-solution truths on the O-grid pipe at ts=0.1
    — a resolution the projected disk cannot reach (V6 gate refuses,
    D37) — with sj_min 0.7071 instead of 0.0852."""
    wall = _disk_wall(0.1, 10.0, 50)
    assert wall.quality.sj_min > 0.70
    _run_hagen_poiseuille(wall)


@needs_foam
def test_icofoam_transient_smoke(tmp_path):
    """The transient branch runs and writes time dirs (physics is the
    steady benchmark's job)."""
    case = tmp_path / "ico"
    wall = _disk_wall(0.3, 10.0, 10)
    write_case(
        wall, case,
        _pipe_spec(solver="icoFoam", end_time=0.2, delta_t=0.01,
                   write_interval=0.1),
    )
    result = solve_case(case, timeout=300.0)
    assert result.solver == "icoFoam"
    assert result.latest_time == pytest.approx(0.2)
    assert result.u_max > 1.0  # profile already peaking above the mean
    assert (case / "log.icoFoam").is_file()


@needs_foam
def test_cli_cfd_e2e(tmp_path, capsys):
    """polyMesh-only case dir + --inlet/--outlet flags -> solve + results."""
    from meshforge.cli import main
    from meshforge.export.foam import write_polymesh

    case = tmp_path / "pipe_cli"
    wall = _disk_wall(0.3, 10.0, 20)
    write_polymesh(
        wall, case, patch_types={"front": "patch", "back": "patch"}
    )
    rc = main([
        "cfd", str(case),
        "--nu", "0.1",
        "--inlet", "front=0,0,1",
        "--outlet", "back=0",
        "--end-time", "3000",
        "--outputs", str(tmp_path / "outputs"),
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "cfd[simpleFoam]" in out
    assert (tmp_path / "outputs" / "cfd" / "pipe_cli" / "pipe_cli_results.vtu").is_file()
    assert (tmp_path / "outputs" / "cfd" / "pipe_cli" / "pipe_cli_cfd.json").is_file()

    rc = main(["cfd", str(case), "--checkmesh"])
    assert rc == 0
    assert "Mesh OK" in capsys.readouterr().out


def test_cli_cfd_error_paths(tmp_path, capsys):
    # These fail before OpenFOAM is ever invoked — no gating needed.
    from meshforge.cli import main

    # bad inlet syntax
    rc = main(["cfd", str(tmp_path), "--inlet", "oops", "--outlet", "back=0"])
    assert rc == 1
    assert "NAME=vx,vy,vz" in capsys.readouterr().err
    # no polyMesh in the case dir
    rc = main([
        "cfd", str(tmp_path), "--inlet", "front=0,0,1", "--outlet", "back=0",
    ])
    assert rc == 1
    assert "polyMesh" in capsys.readouterr().err
