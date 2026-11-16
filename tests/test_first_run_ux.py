"""First-run Workbench usability regressions (session 41).

Every case here comes from a hands-on walkthrough with a real clinical .cso.
None of it touches the meshing math — these lock the *front door*: which
importer a suffix is routed to, what a mistyped path reports, and the
explicit panel order the Workbench lays out.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from meshforge.io.load import dedicated_importer, load_surface
from meshforge.io.paths import normalize_path, normalize_path_str
from meshforge.pipeline import STEP_KNOBS, STEP_ORDER, MeshingSession


# ---- fix 1: extension routing ------------------------------------------
# Demo edition: a NIfTI CT is pointed at the segment step; the other formats
# the full meshforge imports are refused by suffix, named, before any read.
UNREAD = [".cso", ".dxf", ".dwg", ".step", ".stp", ".iges", ".igs", ".brep", ".npy", ".dcm"]


@pytest.mark.parametrize("suffix", UNREAD)
def test_unread_suffix_is_refused_by_name(tmp_path, suffix):
    """"load" on a format this edition does not read says so plainly — not
    pyvista's "not able to be automatically read"."""
    target = tmp_path / f"scan{suffix}"
    target.write_bytes(b"not really a mesh")  # never opened: routing is by suffix
    with pytest.raises(ValueError) as excinfo:
        load_surface(target)
    message = str(excinfo.value)
    assert message.startswith(f"{suffix} is ")
    assert "this demo edition does not read it" in message
    assert "segment step" in message
    assert message.endswith(f": scan{suffix}")
    assert "pyvista" not in message.lower()


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_nifti_suffix_names_the_segment_step(tmp_path, suffix):
    target = tmp_path / f"scan{suffix}"
    target.write_bytes(b"not really a volume")  # never opened: routing is by suffix
    with pytest.raises(ValueError) as excinfo:
        load_surface(target)
    assert str(excinfo.value) == (
        f"{suffix} is a NIfTI volume — load a CT with the segment step "
        f"(CLI: meshforge segment <file>), not as a surface: scan{suffix}"
    )


def test_routing_is_by_suffix_not_by_existence(tmp_path):
    """A .cso that does not exist is still refused by its suffix: the user's
    real mistake is the front door, not the typo."""
    with pytest.raises(ValueError, match="does not read it"):
        load_surface(tmp_path / "never-written.cso")
    with pytest.raises(ValueError, match="segment step"):
        load_surface(tmp_path / "never-written.nii.gz")


def test_nii_gz_beats_bare_nii():
    """Longest suffix wins, so .nii.gz can never resolve as some other rule."""
    assert dedicated_importer("scan.nii.gz") == (".nii.gz", "a NIfTI volume", "segment", "segment")
    assert dedicated_importer("scan.nii") == (".nii", "a NIfTI volume", "segment", "segment")
    assert dedicated_importer("mesh.stl") is None
    assert dedicated_importer("SCAN.NII.GZ")[0] == ".nii.gz"  # case-insensitive
    assert dedicated_importer("MESH.CSO") is None  # refused, but not routed anywhere
    with pytest.raises(ValueError, match=r"^\.cso is a MeVisLab contour file"):
        load_surface("MESH.CSO")  # case-insensitive refusal


def test_unknown_suffix_still_fails_loudly(tmp_path):
    """Routing must not turn into a permissive allow-list: an unknown format
    still errors, it is just no longer misdirected."""
    bogus = tmp_path / "mystery.qqq"
    bogus.write_bytes(b"\x00\x01\x02")
    with pytest.raises(Exception) as excinfo:
        load_surface(bogus)
    assert not isinstance(excinfo.value, FileNotFoundError)


def test_supported_surface_still_loads(tmp_path):
    """The whole point is that .stl/.vtp/.ply are untouched by fix 1."""
    import pyvista as pv

    out = tmp_path / "sphere.stl"
    pv.Sphere().triangulate().save(out)
    mesh = load_surface(out)
    assert mesh.n_points > 0 and mesh.n_cells > 0


# ---- fix 3: path normalisation + a reachable not-found message ----------
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  /a/b.stl  ", "/a/b.stl"),                    # trailing/leading space
        ('"/path/with spaces/f.cso"', "/path/with spaces/f.cso"),  # quotes
        ("'/path/with spaces/f.cso'", "/path/with spaces/f.cso"),  # single quotes
        (r"/path/Example\ CSO\ Files/f.cso", "/path/Example CSO Files/f.cso"),
        (r"/a/b\(1\).stl", "/a/b(1).stl"),               # escaped parens
        ("", ""),                                         # empty stays empty
    ],
)
def test_normalize_path_str(raw, expected):
    assert normalize_path_str(raw) == expected


def test_normalize_path_expands_home():
    assert normalize_path_str("~/x.stl") == str(Path.home() / "x.stl")
    assert normalize_path("~/x.stl") == Path.home() / "x.stl"


def test_normalize_path_is_idempotent():
    raw = r'  "~/Example\ CSO\ Files/f.cso" '
    once = normalize_path_str(raw)
    assert normalize_path_str(once) == once


def test_backslash_before_an_ordinary_char_is_left_alone():
    """We un-escape shell escapes; we do not rewrite arbitrary backslashes."""
    assert normalize_path_str(r"C:\Users\data.stl") == r"C:\Users\data.stl"


@pytest.mark.parametrize(
    "shape",
    [
        "~/definitely-not-here-{}.stl",
        '"/tmp/definitely not here {}.stl"',
        r"/tmp/definitely\ not\ here\ {}.stl",
        "/tmp/definitely-not-here-{}.stl   ",
    ],
)
def test_missing_file_says_surface_file_not_found(shape):
    """The friendly guard in load_surface used to be dead code: file_sha256
    opened the file first and raised a bare [Errno 2]."""
    session = MeshingSession()
    with pytest.raises(FileNotFoundError) as excinfo:
        session.load(shape.format("x"))
    message = str(excinfo.value)
    assert message.startswith("surface file not found: ")
    assert "Errno" not in message


def test_load_accepts_every_common_path_habit(tmp_path):
    """All four rows of the walkthrough's failing table now load."""
    import pyvista as pv

    directory = tmp_path / "Example CSO Files"
    directory.mkdir()
    target = directory / "tube.stl"
    pv.Sphere().triangulate().save(target)

    variants = [
        str(target),
        f'"{target}"',
        str(target).replace(" ", r"\ "),
        f"{target}   ",
    ]
    for raw in variants:
        session = MeshingSession()
        session.load(raw)
        assert session.input_path == target, raw
        assert session.surface.n_points > 0


def test_home_shortcut_loads(tmp_path, monkeypatch):
    import pyvista as pv

    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / "aorta.stl"
    pv.Sphere().triangulate().save(target)
    session = MeshingSession()
    session.load("~/aorta.stl")
    assert session.input_path == target


def test_file_sha256_never_leaks_a_bare_errno():
    """Every rule-1 bracket hashes FIRST, so the CLI used to report
    `[Errno 2] No such file or directory` for a simple typo."""
    from meshforge.io import file_sha256

    with pytest.raises(FileNotFoundError) as excinfo:
        file_sha256("~/definitely/not/here.stl")
    message = str(excinfo.value)
    assert message == f"file not found: {Path.home()}/definitely/not/here.stl"
    assert "Errno" not in message


def test_file_sha256_still_hashes_a_real_file(tmp_path):
    import hashlib

    from meshforge.io import file_sha256

    target = tmp_path / "payload.bin"
    target.write_bytes(b"meshforge")
    assert file_sha256(f'"{target}" ') == hashlib.sha256(b"meshforge").hexdigest()


def test_workfile_paths_are_normalised(tmp_path):
    from meshforge.pipeline import load_workfile

    session = MeshingSession()
    session.generate_synth("straight_tube")
    written = session.save_workfile(f'"{tmp_path / "run.mfwork"}"')
    assert written == tmp_path / "run.mfwork"
    assert load_workfile(f"{written}  ").journal


# ---- fix 2: the explicit panel order -----------------------------------
def test_step_order_covers_step_knobs_exactly():
    assert set(STEP_ORDER) == set(STEP_KNOBS)
    assert len(STEP_ORDER) == len(set(STEP_ORDER))


def test_schema_json_is_served_in_display_order():
    from meshforge.web.schema import schema_json, step_order

    assert list(schema_json()) == list(STEP_ORDER) == step_order()


# ---- fix 8: no internal spec tags in user-facing help -------------------
def test_no_milestone_tags_in_knob_help():
    tagged = [
        f"{step}.{knob}"
        for step, knobs in STEP_KNOBS.items()
        for knob, (_, _, help_text) in knobs.items()
        if re.search(r"\(V\d+", help_text)
    ]
    assert tagged == []


def test_no_milestone_tags_in_cli_help():
    source = Path("src/meshforge/cli.py").read_text()
    assert re.search(r"\(V\d+", source) is None
