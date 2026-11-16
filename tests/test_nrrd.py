"""D83: the minimal NRRD reader — synthetic files only (gzip + raw, LPS with
negative / rotated directions, RAS), the header-first voxel budget, and every
loud refusal (detached data, unknown encoding, missing space info, ...)."""
from __future__ import annotations

import gzip

import numpy as np
import pytest

from meshforge.io.nrrd import read_nrrd, read_nrrd_header


def _write_nrrd(
    path,
    array: np.ndarray,
    *,
    directions,
    origin=(0.0, 0.0, 0.0),
    space="left-posterior-superior",
    encoding="gzip",
    type_name="short",
    endian="little",
    extra: str = "",
    payload: bytes | None = None,
    sizes=None,
):
    dtype = array.dtype.newbyteorder("<" if endian == "little" else ">")
    raw = np.asarray(array, dtype=dtype).tobytes(order="F")
    if payload is None:
        payload = gzip.compress(raw) if encoding in ("gzip", "gz") else raw
    sizes = sizes or array.shape
    dirs = (
        ""
        if directions is None
        else " ".join("(" + ",".join(f"{c:.10g}" for c in d) + ")" for d in directions)
    )
    lines = [
        "NRRD0004",
        "# Complete NRRD file format specification at:",
        "# http://teem.sourceforge.net/nrrd/format.html",
        f"type: {type_name}",
        "dimension: 3",
        f"space: {space}" if space else "",
        f"sizes: {' '.join(str(s) for s in sizes)}",
        f"space directions: {dirs}" if directions is not None else "",
        "kinds: domain domain domain",
        f"endian: {endian}",
        f"encoding: {encoding}",
        "space origin: (" + ",".join(f"{c:.10g}" for c in origin) + ")" if origin is not None else "",
        "series_description:=synthetic volume",
    ]
    header = "\n".join(ln for ln in lines if ln) + "\n" + extra
    path.write_bytes(header.encode() + b"\n" + payload)
    return path


def _ramp(shape=(5, 4, 3)) -> np.ndarray:
    return np.arange(np.prod(shape), dtype=np.int16).reshape(shape, order="F") - 7


def test_gzip_lps_negative_directions(tmp_path):
    """The Zenodo P01..P10 layout: LPS with negative x/y directions -> the RAS
    affine has POSITIVE x/y steps (two sign flips) and a negated origin."""
    arr = _ramp()
    p = _write_nrrd(
        tmp_path / "ct.nrrd",
        arr,
        directions=[(-1.18, 0, 0), (0, -1.18, 0), (0, 0, 1.0)],
        origin=(16.5, -200.6, 544.8),
    )
    vol = read_nrrd(p)
    np.testing.assert_array_equal(vol.array, arr)
    assert vol.array.dtype == np.int16 and vol.space == "LPS"
    expected = np.array(
        [[1.18, 0, 0, -16.5], [0, 1.18, 0, 200.6], [0, 0, 1.0, 544.8], [0, 0, 0, 1]]
    )
    np.testing.assert_allclose(vol.affine, expected)
    assert any("millimetres" in n for n in vol.notes)
    assert any("RAS" in n for n in vol.notes)
    # voxel (i, j, k) of the file -> RAS world, one spot check
    lps = np.array([16.5, -200.6, 544.8]) + np.array([-1.18 * 2, -1.18 * 3, 1.0 * 1])
    ras = vol.affine @ np.array([2, 3, 1, 1.0])
    np.testing.assert_allclose(ras[:3], lps * np.array([-1, -1, 1]))


def test_raw_ras_big_endian_float(tmp_path):
    arr = (np.arange(60, dtype=np.float32).reshape((3, 4, 5), order="F") / 7.0)
    p = _write_nrrd(
        tmp_path / "v.nrrd",
        arr,
        directions=[(0.5, 0, 0), (0, 0.6, 0), (0, 0, 0.7)],
        origin=(1, 2, 3),
        space="RAS",
        encoding="raw",
        type_name="float",
        endian="big",
        extra="space units: \"mm\" \"mm\" \"mm\"\n",
    )
    vol = read_nrrd(p)
    np.testing.assert_array_equal(vol.array, arr)
    assert vol.array.dtype.isnative
    np.testing.assert_allclose(vol.affine[:3, :3], np.diag([0.5, 0.6, 0.7]))
    np.testing.assert_allclose(vol.affine[:3, 3], [1, 2, 3])
    assert not any("declares no 'space units'" in n for n in vol.notes)


def test_rotated_lps_directions(tmp_path):
    """An oblique grid (30 deg about z) keeps its rotation, LPS -> RAS."""
    t = np.radians(30.0)
    d0 = (0.8 * np.cos(t), 0.8 * np.sin(t), 0.0)
    d1 = (-0.8 * np.sin(t), 0.8 * np.cos(t), 0.0)
    d2 = (0.0, 0.0, 1.5)
    arr = _ramp((4, 4, 4)).astype(np.uint16) + 100
    p = _write_nrrd(
        tmp_path / "rot.nrrd", arr, directions=[d0, d1, d2], origin=(5, 6, 7),
        type_name="unsigned short",
    )
    vol = read_nrrd(p)
    lin_lps = np.array([d0, d1, d2]).T
    np.testing.assert_allclose(vol.affine[:3, :3], np.diag([-1, -1, 1]) @ lin_lps)
    np.testing.assert_allclose(vol.affine[:3, 3], [-5, -6, 7])
    np.testing.assert_array_equal(vol.array, arr)


def test_header_parse_skips_comments_and_key_values(tmp_path):
    p = _write_nrrd(tmp_path / "c.nrrd", _ramp(), directions=np.eye(3))
    fields, offset = read_nrrd_header(p)
    assert fields["type"] == "short" and "series_description" not in fields
    assert fields["space directions"].startswith("(1,0,0)")
    assert p.read_bytes()[offset - 2 : offset] == b"\n\n"


def test_voxel_budget_refused_from_header_before_decompressing(tmp_path, monkeypatch):
    """F-IMG-7 pattern: a header declaring a huge grid over a tiny payload is
    refused from ``sizes`` alone — gzip is never opened."""
    import meshforge.io.nrrd as nrrd_mod

    p = _write_nrrd(
        tmp_path / "bomb.nrrd",
        _ramp(),
        directions=np.eye(3),
        sizes=(4000, 4000, 4000),
        payload=gzip.compress(b"\x00" * 64),
    )

    class _Sentinel:
        def __init__(self, *a, **k):
            raise AssertionError("gzip opened before the voxel ceiling was checked")

    monkeypatch.setattr(nrrd_mod.gzip, "GzipFile", _Sentinel)
    with pytest.raises(ValueError, match="voxel ceiling"):
        read_nrrd(p)
    with pytest.raises(ValueError, match="voxel ceiling"):
        read_nrrd(_write_nrrd(tmp_path / "small.nrrd", _ramp(), directions=np.eye(3)),
                  max_voxels=10)


@pytest.mark.parametrize(
    "extra, overrides, match",
    [
        ("data file: ct.raw.gz\n", {}, "detached"),
        ("", {"encoding": "bzip2"}, "encoding 'bzip2' is not supported"),
        ("", {"encoding": "ascii"}, "not supported"),
        ("", {"space": None}, "no 'space' field"),
        ("", {"space": "scanner-xyz"}, "LPS or RAS only"),
        ("", {"directions": None}, "lacks 'space directions'"),
        ("", {"origin": None}, "lacks 'space origin'"),
        ("space units: \"cm\" \"cm\" \"cm\"\n", {}, "only millimetres"),
        ("line skip: 2\n", {}, "line skip"),
        ("", {"type_name": "block"}, "scalar type 'block'"),
    ],
)
def test_loud_refusals(tmp_path, extra, overrides, match):
    kw = {"directions": np.eye(3), "extra": extra}
    if "directions" in overrides:
        kw["directions"] = overrides["directions"]
    for key in ("encoding", "space", "origin", "type_name"):
        if key in overrides:
            kw[key] = overrides[key]
    p = _write_nrrd(tmp_path / "bad.nrrd", _ramp(), **kw)
    with pytest.raises(ValueError, match=match):
        read_nrrd(p)


def test_structural_refusals(tmp_path):
    good = _write_nrrd(tmp_path / "g.nrrd", _ramp(), directions=np.eye(3))
    text = good.read_bytes()
    # dimension 4
    (tmp_path / "d4.nrrd").write_bytes(text.replace(b"dimension: 3", b"dimension: 4"))
    with pytest.raises(ValueError, match="only 3-D"):
        read_nrrd(tmp_path / "d4.nrrd")
    # a non-spatial axis kind
    (tmp_path / "k.nrrd").write_bytes(
        text.replace(b"kinds: domain domain domain", b"kinds: list domain domain")
    )
    with pytest.raises(ValueError, match="spatial"):
        read_nrrd(tmp_path / "k.nrrd")
    # 'none' direction (non-spatial axis)
    (tmp_path / "n.nrrd").write_bytes(text.replace(b"(1,0,0)", b"none"))
    with pytest.raises(ValueError, match="none"):
        read_nrrd(tmp_path / "n.nrrd")
    # singular directions
    sing = _write_nrrd(
        tmp_path / "s.nrrd", _ramp(), directions=[(1, 0, 0), (2, 0, 0), (0, 0, 1)]
    )
    with pytest.raises(ValueError, match="singular"):
        read_nrrd(sing)
    # bad magic
    (tmp_path / "m.nrrd").write_bytes(b"NOTNRRD\n" + text)
    with pytest.raises(ValueError, match="magic"):
        read_nrrd(tmp_path / "m.nrrd")
    # no blank line (header never ends)
    (tmp_path / "h.nrrd").write_bytes(b"NRRD0004\ntype: short\n")
    with pytest.raises(ValueError, match="blank line"):
        read_nrrd(tmp_path / "h.nrrd")
    # 2-byte type without endian
    (tmp_path / "e.nrrd").write_bytes(text.replace(b"endian: little\n", b""))
    with pytest.raises(ValueError, match="endian"):
        read_nrrd(tmp_path / "e.nrrd")
    with pytest.raises(FileNotFoundError):
        read_nrrd(tmp_path / "ghost.nrrd")


def test_truncated_trailing_and_corrupt_data(tmp_path):
    arr = _ramp()
    raw = np.asarray(arr, dtype="<i2").tobytes(order="F")
    short = _write_nrrd(tmp_path / "t.nrrd", arr, directions=np.eye(3),
                        encoding="raw", payload=raw[:-2])
    with pytest.raises(ValueError, match="truncated"):
        read_nrrd(short)
    long = _write_nrrd(tmp_path / "l.nrrd", arr, directions=np.eye(3),
                       encoding="raw", payload=raw + b"\x00\x00")
    with pytest.raises(ValueError, match="more than"):
        read_nrrd(long)
    gz_short = _write_nrrd(tmp_path / "gt.nrrd", arr, directions=np.eye(3),
                           payload=gzip.compress(raw[:-4]))
    with pytest.raises(ValueError, match="truncated"):
        read_nrrd(gz_short)
    corrupt = _write_nrrd(tmp_path / "gc.nrrd", arr, directions=np.eye(3),
                          payload=b"\x1f\x8b\x08\x00garbage-not-deflate")
    with pytest.raises(ValueError, match="corrupt|truncated"):
        read_nrrd(corrupt)
