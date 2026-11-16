"""Session-12 security fixes (from the session-11 audit,
``docs/security/security-analysis-findings.md``). One block per finding; all ungated.

Threat model: the trust boundary is crafted INPUT FILES (``.mfwork``,
DWG, image volumes, solver outputs) + the ``MESHFORGE_*``/PATH
environment — not the user's own CLI args.
"""
from __future__ import annotations


import numpy as np
import pytest

from meshforge.meshing import split_boundary_group, valid_group_name

# A group name carrying a newline that forges a CalculiX *CLOAD card — the
# canonical F1 injection payload.
EVIL_NAME = "lid\n*CLOAD\n1,1,1.0e30"


def _cavity_wall(n: int = 4):
    """A small swept tube wall (kept backend): the host mesh for the
    group-name tests, with an ``outer`` group to split a ``lid`` out of."""
    from meshforge.geometry import compute_centerline
    from meshforge.meshing import get_backend
    from meshforge.synth import generate

    surface, _ = generate("straight_tube")
    cl = compute_centerline(surface)
    return get_backend("swept").mesh_wall(
        surface, 0.3, centerline=cl, n_theta=12, n_axial=n
    )


# ---------------------------------------------------------------------------
# F1 (MEDIUM) — boundary/patch group-name injection into decks & foam dicts


def test_valid_group_name_accepts_the_internal_names():
    # Every name meshforge mints internally must pass, or the backstop in the
    # BoundaryGroup constructor would break real meshes.
    for good in ("inner", "outer", "inlet", "outlet", "front", "back",
                 "outer_rim", "hole_rims", "body", "lid", "_tmp", "wall.1",
                 "Patch-2", "a" * 64):
        assert valid_group_name(good) == good


def test_valid_group_name_refuses_dangerous_names():
    for bad in (EVIL_NAME, "lid\ninjection", "has space", "", "1leads",
                "a" * 65, "semi;colon", "tab\tchar", "esc\x1b[2J", "eq=uals",
                "com,ma", 42, None):
        with pytest.raises(ValueError, match="invalid boundary-group name"):
            valid_group_name(bad)  # type: ignore[arg-type]


# The OpenFOAM sink is a *dictionary*, not a card deck: its grammar is
# ``key { … }`` with ``;``-terminated entries, ``//``/``#`` directives and
# ``$`` variable expansion. A patch name carrying any of those closes the
# enclosing block early and opens an attacker-chosen one — the dict analogue
# of the ``*CLOAD`` card forgery EVIL_NAME performs on a CalculiX deck.
FOAM_DICT_PAYLOADS = (
    "lid} inlet { type fixedValue; value uniform (9 9 9); }",
    "lid;type zeroGradient;",
    "lid{type",
    "lid}",
    "lid#include",
    "lid$internalField",
    "lid//comment",
    "lid/*block*/",
)


def test_valid_group_name_refuses_openfoam_dict_metacharacters():
    for bad in FOAM_DICT_PAYLOADS:
        with pytest.raises(ValueError, match="invalid boundary-group name"):
            valid_group_name(bad)


def test_valid_group_name_length_ceiling_is_exact():
    # 64 is the documented ceiling: the last accepted length, and the first
    # refused one — an off-by-one here would let a name overrun a fixed-width
    # solver set-name field.
    assert valid_group_name("g" * 64) == "g" * 64
    with pytest.raises(ValueError, match="invalid boundary-group name"):
        valid_group_name("g" * 65)


def test_foam_default_patch_refuses_dict_injection():
    """The one patch name that is NOT a BoundaryGroup (so the constructor
    backstop never sees it): ``default_patch`` is minted at write time and
    lands directly in the polyMesh/FLUENT zone list. It is validated at the
    writer (``export/foam.py`` ``_patch_face_map``)."""
    from meshforge.export.foam import _patch_face_map

    wall = _cavity_wall()
    for bad in (EVIL_NAME,) + FOAM_DICT_PAYLOADS:
        with pytest.raises(ValueError, match="invalid boundary-group name"):
            _patch_face_map(wall, {}, bad)


def test_split_boundary_group_refuses_newline_name():
    # (a) The direct mint site: a newline-bearing name is refused LOUDLY
    # before any group is created (fail-loud).
    wall = _cavity_wall()
    with pytest.raises(ValueError, match="invalid boundary-group name"):
        split_boundary_group(
            wall, "outer", axis="z", threshold=1e-9, name=EVIL_NAME,
            side="below",
        )
    assert EVIL_NAME not in wall.boundary  # nothing was written


def test_boundary_group_constructor_backstop():
    from meshforge.meshing import BoundaryGroup

    with pytest.raises(ValueError, match="invalid boundary-group name"):
        BoundaryGroup(
            name=EVIL_NAME,
            cell_ids=np.array([0], dtype=np.int64),
            face_nodes=np.array([[0, 1, 2, 3]], dtype=np.int64),
        )


def test_cfd_spec_refuses_injected_patch_name():
    # The CFD sink: patch names become boundaryField dict keys. CfdSpec's
    # constructor backstop refuses a forged name.
    from meshforge.solve.foam_case import CfdBC, CfdSpec

    with pytest.raises(ValueError, match="invalid boundary-group name"):
        CfdSpec(
            solver="simpleFoam",
            nu=0.01,
            bcs={EVIL_NAME: CfdBC("pressure", 0.0)},
        )


# ---------------------------------------------------------------------------
# F2 (LOW) — terminal-escape injection via raw tool output in exceptions

# ESC screen-clear + OSC title/clipboard change + BEL + raw CR — the classic
# terminal-injection payload a hostile solver / crafted result file could
# emit, which the CLI would print straight to a terminal.
ANSI_PAYLOAD = "start\x1b[2J\x1b]0;pwned\x07mid\rend\x7fdel"


def test_scrub_neutralises_control_chars_keeps_newline_tab():
    from meshforge.textguard import scrub

    out = scrub(ANSI_PAYLOAD)
    # Every C0/C1 control char is gone (ESC, BEL, CR, DEL)...
    for ctrl in ("\x1b", "\x07", "\r", "\x7f"):
        assert ctrl not in out
    # ...but the readable text survives, and \n / \t are preserved.
    assert "start" in out and "pwned" in out and "end" in out
    assert scrub("a\nb\tc") == "a\nb\tc"
    # A non-str is coerced, not crashed.
    assert scrub(1234) == "1234"


def test_dat_parser_error_message_is_terminal_safe():
    # A crafted .dat whose eigenvalue row carries an ESC sequence: the parse
    # error must not carry the raw control bytes into the printed message.
    from meshforge.solve.dat import parse_eigenfrequencies

    crafted = (
        "         E I G E N V A L U E   O U T P U T\n"
        "     1   2.0   3.0   \x1b[2J\x1b]0;x\x07\n"  # digit-led, only 4 cols
    )
    with pytest.raises(ValueError, match="malformed eigenvalue row") as exc:
        parse_eigenfrequencies(crafted)
    msg = str(exc.value)
    assert "\x1b" not in msg and "\x07" not in msg  # no raw escape bytes


# ---------------------------------------------------------------------------
# F4 (LOW) — image ingest memory-exhaustion via unbounded voxel count


def test_image_ingest_refuses_oversized_volume():
    # The V18 image path now enforces the same MAX_VOXELS ceiling the geometry
    # paths use. A zero-memory broadcast view lets us trip the real 40M limit
    # without allocating a giant array — the guard reads only array.shape and
    # fires before any full-array scan.
    from meshforge.geometry.centerline import MAX_VOXELS
    from meshforge.io.image import volume_to_surface

    n = 350  # 350**3 = 42_875_000 > 40_000_000
    assert n ** 3 > MAX_VOXELS
    oversized = np.broadcast_to(np.zeros((1, 1, 1)), (n, n, n))
    with pytest.raises(ValueError, match=rf"exceeds {MAX_VOXELS} voxels"):
        volume_to_surface(oversized, level=0.5)
