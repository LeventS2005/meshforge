"""V13 solve plumbing: frd/dat parsers and the ccx runner (fake executables).

Runner failure branches use fake ccx executables so they run WITHOUT
CalculiX. (The session ``solve`` step and the ``solve`` command are not part
of the demo edition; their cases left with them.)
"""
from __future__ import annotations

import stat
from pathlib import Path

import numpy as np
import pytest

from meshforge.solve import (
    find_ccx,
    frd_to_grid,
    parse_displacements,
    parse_stresses,
    read_frd,
    solve_deck,
    von_mises,
)

# ---------------------------------------------------------------------------
# frd fixture writers (exact CalculiX fixed-width ASCII format)


def _frd_nodes(ids_pts: list[tuple[int, tuple[float, float, float]]]) -> str:
    lines = [f"    2C{len(ids_pts):30d}{1:38d}"]
    for nid, (x, y, z) in ids_pts:
        lines.append(f" -1{nid:10d}{x:12.5E}{y:12.5E}{z:12.5E}")
    lines.append(" -3")
    return "\n".join(lines) + "\n"


def _frd_elements(elems: list[tuple[int, int, list[int]]]) -> str:
    lines = [f"    3C{len(elems):30d}{1:38d}"]
    for eid, etype, conn in elems:
        lines.append(f" -1{eid:10d}{etype:5d}{0:5d}{1:5d}")
        lines.append(" -2" + "".join(f"{n:10d}" for n in conn))
    lines.append(" -3")
    return "\n".join(lines) + "\n"


def _frd_field(name: str, comps: list[str], data: dict[int, list[float]]) -> str:
    lines = [
        f"  100CL       101 1.00000E+00{len(data):10d}                     0    1           1",
        f" -4  {name:<8s}{len(comps):5d}    1",
    ]
    for k, c in enumerate(comps, start=1):
        lines.append(f" -5  {c:<8s}    1    2{k:5d}    0")
    for nid, vals in data.items():
        lines.append(f" -1{nid:10d}" + "".join(f"{v:12.5E}" for v in vals))
    lines.append(" -3")
    return "\n".join(lines) + "\n"


_STRESS_COMPS = ["SXX", "SYY", "SZZ", "SXY", "SYZ", "SZX"]


def _unit_cube_frd(tmp_path: Path, *, shuffle_ids: bool = False) -> Path:
    """One he8 unit cube; DISP = 0.1*(x,y,z); uniaxial SXX=2 everywhere."""
    corners = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]
    ids = [3, 1, 8, 2, 7, 4, 6, 5] if shuffle_ids else list(range(1, 9))
    nodes = list(zip(ids, corners))
    disp = {nid: [0.1 * x, 0.1 * y, 0.1 * z] for nid, (x, y, z) in nodes}
    stress = {nid: [2.0, 0.0, 0.0, 0.0, 0.0, 0.0] for nid in ids}
    text = (
        "    1C  meshforge fake\n"
        + _frd_nodes(nodes)
        + _frd_elements([(1, 1, ids)])
        + _frd_field("DISP", ["D1", "D2", "D3", "ALL"], disp)
        + _frd_field("STRESS", _STRESS_COMPS, stress)
        + " 9999\n"
    )
    path = tmp_path / "cube.frd"
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# frd parser


def test_read_frd_unit_cube(tmp_path):
    model = read_frd(_unit_cube_frd(tmp_path))
    assert np.array_equal(model.node_ids, np.arange(1, 9))
    assert model.points.shape == (8, 3)
    assert model.points[1].tolist() == [1.0, 0.0, 0.0]
    disp = model.fields["DISP"]
    assert disp.values.shape == (8, 3)  # trailing computed 'ALL' carries no data
    assert disp.components == ("D1", "D2", "D3")
    np.testing.assert_allclose(disp.values, 0.1 * model.points, atol=1e-9)
    stress = model.fields["STRESS"]
    assert stress.values.shape == (8, 6)
    assert stress.components == tuple(_STRESS_COMPS)


def test_read_frd_sorts_scrambled_node_ids(tmp_path):
    model = read_frd(_unit_cube_frd(tmp_path, shuffle_ids=True))
    assert np.array_equal(model.node_ids, np.arange(1, 9))
    # DISP = 0.1 * coords must survive the reordering node-for-node.
    np.testing.assert_allclose(model.fields["DISP"].values, 0.1 * model.points, atol=1e-9)
    grid = frd_to_grid(model)
    assert grid.n_cells == 1
    assert float(grid.compute_cell_sizes()["Volume"][0]) == pytest.approx(1.0)


def test_frd_to_grid_fields_and_von_mises(tmp_path):
    grid = frd_to_grid(read_frd(_unit_cube_frd(tmp_path)))
    assert grid.n_points == 8 and grid.n_cells == 1
    for name in ("DISP", "U_mag", "STRESS", "S_mises"):
        assert name in grid.point_data, name
    # Uniaxial SXX=2 -> von Mises = 2 exactly.
    np.testing.assert_allclose(grid.point_data["S_mises"], 2.0, atol=1e-12)
    np.testing.assert_allclose(
        grid.point_data["U_mag"], 0.1 * np.linalg.norm(grid.points, axis=1), atol=1e-9
    )


def test_von_mises_pure_shear_and_missing_component():
    vals = np.array([[0.0, 0.0, 0.0, 3.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        von_mises(vals, tuple(_STRESS_COMPS)), [3.0 * np.sqrt(3.0)]
    )
    with pytest.raises(ValueError, match="missing component"):
        von_mises(vals[:, :5], ("SXX", "SYY", "SZZ", "SXY", "SYZ"))


def test_read_frd_loud_failures(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_frd(tmp_path / "nope.frd")

    empty = tmp_path / "empty.frd"
    empty.write_text("    1C  header only\n 9999\n")
    with pytest.raises(ValueError, match="no nodal block"):
        read_frd(empty)

    te10 = tmp_path / "te10.frd"
    te10.write_text(
        _frd_nodes([(i, (float(i), 0.0, 0.0)) for i in range(1, 11)])
        + _frd_elements([(1, 6, list(range(1, 11)))])
    )
    with pytest.raises(ValueError, match="te10"):
        read_frd(te10)

    truncated = tmp_path / "trunc.frd"
    truncated.write_text(
        _frd_nodes([(i, (float(i), 0.0, 0.0)) for i in range(1, 9)])
        + f"    3C{1:30d}\n -1{1:10d}{1:5d}{0:5d}{1:5d}\n -3\n"
    )
    with pytest.raises(ValueError, match="truncated element connectivity"):
        read_frd(truncated)

    partial = tmp_path / "partial.frd"
    partial.write_text(
        _frd_nodes([(1, (0.0, 0.0, 0.0)), (2, (1.0, 0.0, 0.0))])
        + _frd_field("DISP", ["D1", "D2", "D3"], {1: [0.0, 0.0, 0.0]})
    )
    with pytest.raises(ValueError, match="covers 1 of 2 nodes"):
        read_frd(partial)


def test_frd_to_grid_needs_elements(tmp_path):
    no_elems = tmp_path / "noel.frd"
    no_elems.write_text(_frd_nodes([(1, (0.0, 0.0, 0.0))]))
    with pytest.raises(ValueError, match="no element block"):
        frd_to_grid(read_frd(no_elems))


# ---------------------------------------------------------------------------
# dat parsers (shared with the V2 benchmarks)


def test_dat_parsers_roundtrip_and_absent():
    dat = (
        " displacements (vx,vy,vz) for set INNER and time  0.1000000E+01\n\n"
        "        12  1.500000E-03 -2.000000E-04  0.000000E+00\n\n"
        " stresses (elem, integ.pnt.,sxx,syy,szz,sxy,sxz,syz) for set X\n\n"
        "         3   1  1.0E+00  2.0E+00  3.0E+00  4.0E-01  5.0E-02  6.0E-03\n"
    )
    u = parse_displacements(dat)
    assert set(u) == {12}
    np.testing.assert_allclose(u[12], [1.5e-3, -2.0e-4, 0.0])
    s = parse_stresses(dat)
    assert set(s) == {(3, 1)}
    np.testing.assert_allclose(s[(3, 1)], [1.0, 2.0, 3.0, 0.4, 0.05, 0.006])
    assert parse_displacements("") == {}
    assert parse_stresses("no blocks here\n") == {}


# ---------------------------------------------------------------------------
# runner: find_ccx + fake-ccx failure branches (no CalculiX needed)


def test_find_ccx_env_override(tmp_path, monkeypatch):
    fake = tmp_path / "ccx"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("MESHFORGE_CCX", str(fake))
    assert find_ccx() == str(fake)
    monkeypatch.setenv("MESHFORGE_CCX", str(tmp_path / "missing"))
    assert find_ccx() != str(tmp_path / "missing")  # non-file falls through


def _fake_ccx(tmp_path: Path, body: str) -> str:
    exe = tmp_path / "fake_ccx"
    exe.write_text("#!/bin/sh\n" + body)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


def _dummy_deck(tmp_path: Path) -> Path:
    deck = tmp_path / "job.inp"
    deck.write_text("*HEADING\nfake\n")
    return deck


def test_solve_deck_missing_deck_and_missing_ccx(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="deck not found"):
        solve_deck(tmp_path / "nope.inp", tmp_path / "out", ccx="/bin/sh")
    deck = _dummy_deck(tmp_path)
    monkeypatch.setenv("MESHFORGE_CCX", "")
    monkeypatch.setenv("PATH", str(tmp_path))  # hide any real ccx
    monkeypatch.setattr(
        "meshforge.solve.runner._CCX_FALLBACK", tmp_path / "absent"
    )
    with pytest.raises(RuntimeError, match="not found: set MESHFORGE_CCX"):
        solve_deck(deck, tmp_path / "out")
    with pytest.raises(RuntimeError, match="executable not found"):
        solve_deck(deck, tmp_path / "out", ccx=str(tmp_path / "ghost"))


def test_solve_deck_loud_on_solver_failure(tmp_path):
    deck = _dummy_deck(tmp_path)
    with pytest.raises(RuntimeError, match=r"ccx failed \(exit 3\)"):
        solve_deck(deck, tmp_path / "out", ccx=_fake_ccx(tmp_path, "exit 3\n"))
    with pytest.raises(RuntimeError, match="fails quietly"):
        solve_deck(
            deck, tmp_path / "out",
            ccx=_fake_ccx(tmp_path, 'echo "*ERROR reading the input"\nexit 0\n'),
        )
    with pytest.raises(RuntimeError, match="produced no .frd"):
        solve_deck(
            deck, tmp_path / "out",
            ccx=_fake_ccx(tmp_path, 'echo "Job finished"\nexit 0\n'),
        )
    with pytest.raises(RuntimeError, match="timeout"):
        solve_deck(
            deck, tmp_path / "out",
            ccx=_fake_ccx(tmp_path, "sleep 5\n"), timeout=0.5,
        )


def test_solve_deck_success_with_canned_results(tmp_path):
    canned = _unit_cube_frd(tmp_path)
    deck = _dummy_deck(tmp_path)
    dat = tmp_path / "canned.dat"
    dat.write_text(" displacements (vx,vy,vz) for set S\n\n         1  1.0E-01  0.0E+00  0.0E+00\n")
    exe = _fake_ccx(
        tmp_path, f'cp "{canned}" "$2.frd"\ncp "{dat}" "$2.dat"\nexit 0\n'
    )
    out_dir = tmp_path / "out"
    result = solve_deck(deck, out_dir, ccx=exe)
    assert result.n_nodes == 8
    assert sorted(result.frd.fields) == ["DISP", "STRESS"]
    # DISP = 0.1*coords, farthest corner (1,1,1) -> |U| max = 0.1*sqrt(3).
    assert result.u_max == pytest.approx(0.1 * np.sqrt(3.0))
    assert result.mises_max == pytest.approx(2.0)  # uniaxial SXX=2
    assert result.frd_path == out_dir / "job.frd" and result.frd_path.is_file()
    assert result.dat_path == out_dir / "job.dat" and result.dat_path.is_file()
    assert parse_displacements(result.dat_path.read_text())
    # Scratch ran in a temp dir: nothing landed next to the deck.
    assert not (tmp_path / "job.frd").exists()
