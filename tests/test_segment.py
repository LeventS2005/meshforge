"""D83 (M1): the ``segment`` step with a FAKE TotalSegmentator.

The fake is a tiny Python script written into tmp_path, made executable and
pointed to via MESHFORGE_TOTALSEG. It records its argv, thresholds the staged
CT (a bright synthetic tube) into ``aorta.nii.gz`` with the input's affine,
and can be told (env FAKE_TS_MODE) to fail, return an empty mask, write
nothing, leave the heart out of the field (``noheart``), or hang. It writes
one mask per requested roi: ``aorta.nii.gz`` is the thresholded tube, any
other roi (the ``heart`` context) a small blob beside it.
"""
from __future__ import annotations

import gzip
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from meshforge.io import segment as segmod  # noqa: E402
from meshforge.io.segment import (  # noqa: E402
    EMPTY_MASK_MESSAGE,
    find_totalseg,
    segment_ct,
)

FAKE_TS = '''#!{python}
import json, os, sys, time
args = sys.argv[1:]
record = os.environ.get("FAKE_TS_RECORD")
if record:
    with open(record, "a") as fh:
        fh.write(json.dumps(args) + "\\n")
if args == ["--version"]:
    print("2.99.0-fake")
    sys.exit(0)
mode = os.environ.get("FAKE_TS_MODE", "ok")
if mode == "fail":
    sys.stderr.write("loading weights...\\nRuntimeError: fake model weights missing\\n")
    sys.exit(1)
if mode == "hang":
    time.sleep(60)
src = args[args.index("-i") + 1]
out = args[args.index("-o") + 1]
os.makedirs(out, exist_ok=True)
if mode == "nomask":
    sys.exit(0)
import nibabel as nib
import numpy as np
img = nib.load(src)
data = np.asanyarray(img.dataobj)
mask = (data > 100).astype(np.uint8)
if mode == "empty":
    mask[:] = 0
if mode == "speck":  # the M0 finding: a few voxels, nowhere near an aorta
    mask[:] = 0
    mask[2:5, 2:5, 2:5] = 1
rois = args[args.index("--roi_subset") + 1:]
rois = rois[:rois.index("--fast")] if "--fast" in rois else rois[:rois.index("--device")]
for roi in rois:
    m = mask
    if roi != "aorta":  # a context organ: a small blob above the tube, or nothing
        m = np.zeros_like(mask)
        if mode != "noheart":
            m[36:46, 2:12, 20:28] = 1  # outside the tube's disk (centre (24, 21), r 9)
    nib.save(nib.Nifti1Image(m, img.affine, img.header), os.path.join(out, roi + ".nii.gz"))
    print("fake TotalSegmentator: wrote " + roi + ".nii.gz")
'''


@pytest.fixture
def fake_ts(tmp_path, monkeypatch):
    if " " in sys.executable:
        pytest.skip("a shebang line cannot carry a path with spaces")
    exe = tmp_path / "bin" / "TotalSegmentator"
    exe.parent.mkdir()
    exe.write_text(FAKE_TS.format(python=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    record = tmp_path / "argv.jsonl"
    monkeypatch.setenv("MESHFORGE_TOTALSEG", str(exe))
    monkeypatch.setenv("FAKE_TS_RECORD", str(record))
    monkeypatch.delenv("FAKE_TS_MODE", raising=False)

    def calls():
        if not record.is_file():
            return []
        return [json.loads(line) for line in record.read_text().splitlines()]

    return exe, calls


# ---------------------------------------------------------------------------
# synthetic CT: a bright tube (HU 300) along the scanner z axis, air elsewhere
# ---------------------------------------------------------------------------
SHAPE = (48, 44, 30)
SPACING = 0.8
SLICE = 1.5
TUBE_R_VOX = 9.0
TUBE_C = (24.0, 21.0)  # (i, j) of the axis


def _tube_ct() -> np.ndarray:
    ii, jj = np.indices(SHAPE[:2])
    disk = (ii - TUBE_C[0]) ** 2 + (jj - TUBE_C[1]) ** 2 <= TUBE_R_VOX**2
    ct = np.full(SHAPE, -100, dtype=np.int16)
    ct[disk] = 300
    return ct


def _write_lps_nrrd(path: Path, ct: np.ndarray, origin_lps=(16.5, -200.6, 544.8)):
    """The Zenodo P01..P10 header layout: LPS, negative x/y directions, gzip."""
    raw = np.asarray(ct, dtype="<i2").tobytes(order="F")
    header = (
        "NRRD0004\n"
        "type: short\n"
        "dimension: 3\n"
        "space: left-posterior-superior\n"
        f"sizes: {SHAPE[0]} {SHAPE[1]} {SHAPE[2]}\n"
        f"space directions: (-{SPACING},0,0) (0,-{SPACING},0) (0,0,{SLICE})\n"
        "kinds: domain domain domain\n"
        "endian: little\n"
        "encoding: gzip\n"
        f"space origin: ({origin_lps[0]},{origin_lps[1]},{origin_lps[2]})\n\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header.encode() + gzip.compress(raw))
    return path


def _expected_axis_ras(origin_lps=(16.5, -200.6, 544.8)) -> np.ndarray:
    """RAS (x, y) of the tube axis: LPS position negated in x and y."""
    lps = np.array(origin_lps[:2]) + np.array([-SPACING * TUBE_C[0], -SPACING * TUBE_C[1]])
    return -lps


def test_segment_nrrd_end_to_end_with_the_fake(tmp_path, fake_ts):
    exe, calls = fake_ts
    ct_path = _write_lps_nrrd(tmp_path / "dir with spaces" / "70%.nrrd", _tube_ct())
    before = ct_path.read_bytes()
    mask_out = tmp_path / "keep" / "mask.nii.gz"
    seg = segment_ct(ct_path, device="cpu", fast=True, smoothing=1.0, keep_mask=mask_out)

    # argument passing: one --version probe, the vessel call, then the heart's
    # own call (never one call for both: see the context test)
    argv = calls()
    assert argv[0] == ["--version"] and argv.count(["--version"]) == 1
    run, context_run = argv[1], argv[2]
    assert run[run.index("--roi_subset") + 1 : run.index("--fast")] == ["aorta"]
    assert context_run[context_run.index("--roi_subset") + 1 : context_run.index("--fast")] == ["heart"]
    assert context_run[context_run.index("-o") + 1] != run[run.index("-o") + 1]
    assert "--fast" in run and run[run.index("--device") + 1] == "cpu"
    staged_in, staged_out = run[run.index("-i") + 1], run[run.index("-o") + 1]
    assert " " not in staged_in and " " not in staged_out  # staged, space-free
    assert staged_in.endswith(".nii.gz") and not Path(staged_in).exists()  # cleaned
    assert "--task" not in run and "-l" not in run and "--license_number" not in run

    # the CT is untouched (rule 1) and nothing was written beside it
    assert ct_path.read_bytes() == before
    assert sorted(p.name for p in ct_path.parent.iterdir()) == ["70%.nrrd"]

    s = seg.summary
    assert s["totalseg"]["version"] == "2.99.0-fake"
    # the vessel call is what it always was; the heart (D108 addendum 7) is a call apart
    assert s["totalseg"]["flags"] == ["--roi_subset", "aorta", "--fast", "--device", "cpu"]
    assert s["totalseg"]["rois"] == ["aorta"] and s["totalseg"]["context_rois"] == ["heart"]
    assert s["totalseg"]["context_flags"] == ["--roi_subset", "heart", "--fast", "--device", "cpu"]
    assert s["totalseg"]["context_wall_time_s"] >= 0
    assert "--- context heart ---" in seg.log
    assert s["totalseg"]["task"] == "total" and s["totalseg"]["wall_time_s"] >= 0
    assert s["ct"]["format"] == "nrrd" and s["ct"]["shape"] == list(SHAPE)
    assert s["length_unit"] == "mm" and "RAS" in s["surface"]["frame"]
    assert s["mask"]["dropped_components"] == 0
    assert s["surface"]["watertight"] is True and seg.surface.n_open_edges == 0
    assert any("LPS" in n for n in s["notes"])

    # the surface lands where the tube is, in RAS millimetres
    pts = np.asarray(seg.surface.points)
    axis = _expected_axis_ras()
    mid = pts[np.abs(pts[:, 2] - (544.8 + SLICE * 15)) < 10]
    np.testing.assert_allclose(mid[:, :2].mean(axis=0), axis, atol=0.05)
    radius = np.linalg.norm(mid[:, :2] - axis, axis=1).mean()
    assert radius == pytest.approx(TUBE_R_VOX * SPACING, rel=0.05)
    assert seg.surface.bounds[4] == pytest.approx(544.8 - SLICE / 2, abs=0.05)
    assert seg.surface.bounds[5] == pytest.approx(544.8 + SLICE * (SHAPE[2] - 0.5), abs=0.05)
    assert float(seg.surface.volume) > 0  # outward

    kept = nib.load(str(mask_out))
    assert np.asanyarray(kept.dataobj).sum() == s["mask"]["kept_voxels"]
    np.testing.assert_allclose(kept.affine, seg.ct_affine, atol=1e-3)
    assert np.asanyarray(kept.dataobj)[36:46, 2:12, 20:28].sum() == 0  # the heart was NOT unioned in


def test_context_heart_is_its_own_surface_never_unioned_and_out_of_field_is_a_note(
    tmp_path, fake_ts, monkeypatch
):
    """D108 addendum 7: ``context_rois`` get a TotalSegmentator call of their
    own (``--roi_subset`` crops the CT around the requested classes before
    the 1.5 mm model, so one call for vessel + heart would change the
    vessel's own outline) and each is its own surface (``context_surfaces``,
    summary ``context``), never part of the vessel; a heart outside the
    field is a note, not an error; only the whole heart of the free task may
    be asked for."""
    exe, calls = fake_ts
    ct = _write_lps_nrrd(tmp_path / "ct.nrrd", _tube_ct())
    seg = segment_ct(ct, device="cpu")
    runs = [c for c in calls() if "-i" in c]
    assert [c[c.index("--roi_subset") + 1 : c.index("--device")] for c in runs] == [["aorta"], ["heart"]]
    heart = seg.context_surfaces["heart"]
    assert set(seg.context_surfaces) == {"heart"} and heart.n_open_edges == 0
    assert float(heart.volume) > 0 and heart.n_points < seg.surface.n_points
    # the blob lies beside the tube: at higher i, so (LPS x negated) at higher RAS x
    assert heart.bounds[0] > seg.surface.bounds[1]
    c = seg.summary["context"]["heart"]
    assert c["watertight"] is True and c["mask_volume_ml"] == pytest.approx(0.768, abs=0.01)
    assert "context only" in c["role"] and "not part of the vessel" in c["role"]
    assert not any("heart not in field" in n for n in seg.notes)
    # the vessel's own numbers do not move with the context asked for
    alone = segment_ct(ct, device="cpu", context_rois=())
    assert len([c for c in calls() if "-i" in c]) == 3  # one more call, the vessel alone
    assert alone.context_surfaces == {} and "context" not in alone.summary
    assert alone.summary["totalseg"]["flags"] == ["--roi_subset", "aorta", "--device", "cpu"]
    assert alone.summary["totalseg"]["context_flags"] == [] and "--- context" not in alone.log
    assert alone.summary["totalseg"]["context_wall_time_s"] is None
    assert alone.summary["mask"]["kept_voxels"] == seg.summary["mask"]["kept_voxels"]
    assert alone.surface.n_points == seg.surface.n_points
    # out of the field of view: skipped with a note, the vessel unaffected
    monkeypatch.setenv("FAKE_TS_MODE", "noheart")
    out = segment_ct(ct, device="cpu")
    assert out.context_surfaces == {} and out.summary["context"] == {}
    assert "heart not in field" in out.notes and "heart not in field" in out.summary["notes"]
    assert out.surface.n_points == seg.surface.n_points
    monkeypatch.delenv("FAKE_TS_MODE")
    # the heart is context, never the vessel; chambers are a licensed task
    with pytest.raises(ValueError, match="context"):
        segment_ct(ct, device="cpu", rois=("aorta", "heart"))
    with pytest.raises(ValueError, match="not allowed"):
        segment_ct(ct, device="cpu", context_rois=("heartchambers_highres",))
    with pytest.raises(ValueError, match="not allowed"):
        segment_ct(ct, device="cpu", context_rois=("heart", "heart"))
    assert segmod.CONTEXT_ROIS == ("heart",) and "heart" in segmod.ALLOWED_ROIS


def test_segment_nifti_input_with_negative_ras_axes(tmp_path, fake_ts):
    """A NIfTI CT whose RAS affine has negative x/y (the Zenodo P11..P20 frame
    once converted) is copied, segmented and mapped exactly."""
    a = np.diag([-SPACING, -SPACING, SLICE, 1.0])
    a[:3, 3] = [40.0, 30.0, -500.0]
    img = nib.Nifti1Image(_tube_ct(), a)
    img.header.set_xyzt_units("mm", "sec")
    ct = tmp_path / "ct.nii.gz"
    nib.save(img, str(ct))
    seg = segment_ct(ct, device="mps", fast=False)
    assert seg.summary["ct"]["format"] == "nifti"
    assert "--fast" not in seg.summary["totalseg"]["flags"]
    pts = np.asarray(seg.surface.points)
    mid = pts[np.abs(pts[:, 2] - (-500 + SLICE * 15)) < 10]
    axis = a[:2, :2] @ np.array(TUBE_C) + a[:2, 3]
    np.testing.assert_allclose(mid[:, :2].mean(axis=0), axis, atol=0.05)
    assert float(seg.surface.volume) > 0 and seg.surface.n_open_edges == 0


def test_missing_executable_is_loud_and_names_the_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("MESHFORGE_TOTALSEG", str(tmp_path / "nope" / "TotalSegmentator"))
    with pytest.raises(RuntimeError, match="MESHFORGE_TOTALSEG"):
        find_totalseg()
    # nothing configured anywhere: env unset, PATH empty, no known env
    monkeypatch.delenv("MESHFORGE_TOTALSEG")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(segmod, "_TOTALSEG_FALLBACK", tmp_path / "no" / "TotalSegmentator")
    with pytest.raises(RuntimeError) as exc:
        find_totalseg()
    message = str(exc.value)
    assert "MESHFORGE_TOTALSEG" in message
    assert "conda create -n totalseg --solver=libmamba python=3.11" in message
    assert "pip install TotalSegmentator" in message
    ct = _write_lps_nrrd(tmp_path / "ct.nrrd", _tube_ct())
    with pytest.raises(RuntimeError, match="MESHFORGE_TOTALSEG"):
        segment_ct(ct)  # fails fast, before staging the CT


def test_failing_executable_reports_its_stderr_tail(tmp_path, fake_ts, monkeypatch):
    monkeypatch.setenv("FAKE_TS_MODE", "fail")
    ct = _write_lps_nrrd(tmp_path / "ct.nrrd", _tube_ct())
    with pytest.raises(RuntimeError) as exc:
        segment_ct(ct, device="cpu")
    message = str(exc.value)
    assert "exit code 1" in message
    assert "fake model weights missing" in message


def test_empty_mask_is_refused_loudly(tmp_path, fake_ts, monkeypatch):
    monkeypatch.setenv("FAKE_TS_MODE", "empty")
    ct = _write_lps_nrrd(tmp_path / "ct.nrrd", _tube_ct())
    with pytest.raises(ValueError) as exc:
        segment_ct(ct, device="cpu")
    assert str(exc.value) == EMPTY_MASK_MESSAGE


def test_a_fragment_is_never_passed_on_as_the_aorta(tmp_path, fake_ts, monkeypatch):
    monkeypatch.setenv("FAKE_TS_MODE", "speck")
    ct = _write_lps_nrrd(tmp_path / "ct.nrrd", _tube_ct())
    with pytest.raises(ValueError, match=r"holds only 0\.0259 mL \(27 voxels\) — implausibly small"):
        segment_ct(ct, device="cpu")


def test_missing_mask_and_timeout_are_loud(tmp_path, fake_ts, monkeypatch):
    ct = _write_lps_nrrd(tmp_path / "ct.nrrd", _tube_ct())
    monkeypatch.setenv("FAKE_TS_MODE", "nomask")
    with pytest.raises(RuntimeError, match=r"wrote no mask for \['aorta'\]"):
        segment_ct(ct, device="cpu")
    monkeypatch.setenv("FAKE_TS_MODE", "hang")
    with pytest.raises(RuntimeError, match="timeout"):
        segment_ct(ct, device="cpu", timeout=1.0)


def test_argument_validation(tmp_path, fake_ts):
    ct = _write_lps_nrrd(tmp_path / "ct.nrrd", _tube_ct())
    with pytest.raises(ValueError, match="device must be one of mps/cpu"):
        segment_ct(ct, device="cuda")
    with pytest.raises(ValueError, match="not allowed"):
        segment_ct(ct, rois=("aorta", "brain"))
    with pytest.raises(ValueError, match="timeout"):
        segment_ct(ct, timeout=0)
    with pytest.raises(ValueError, match="smoothing"):
        segment_ct(ct, smoothing=-1)
    with pytest.raises(FileNotFoundError):
        segment_ct(tmp_path / "ghost.nrrd")
    bad = tmp_path / "ct.mha"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported CT input"):
        segment_ct(bad)
    m = nib.Nifti1Image(_tube_ct(), np.eye(4))
    m.header.set_xyzt_units("meter", "sec")
    nib.save(m, str(tmp_path / "metres.nii"))
    with pytest.raises(ValueError, match="TotalSegmentator reads voxel spacing as millimetres"):
        segment_ct(tmp_path / "metres.nii")


def test_session_segment_step_journal_and_replay(tmp_path, fake_ts, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from meshforge.pipeline import STEP_KNOBS, STEP_ORDER, MeshingSession, load_workfile, replay

    assert STEP_ORDER[0] == "segment"  # the demo starts from a CT: segment leads the panels
    assert set(STEP_KNOBS["segment"]) == {
        "path", "device", "fast", "smoothing", "timeout", "lumen_out", "lumen_hu",  # D85
        "mask_out",  # D92: the AI mask for the sac step
    }
    ct = _write_lps_nrrd(tmp_path / "ct.nrrd", _tube_ct())
    s = MeshingSession()
    with pytest.raises(ValueError, match="segment needs a CT volume path"):
        s.compute_segment()
    r = s.compute_segment(path=f'"{ct}"', device="cpu", fast=1)  # quoted, like a paste
    assert r.summary["segment"]["surface"]["watertight"] is True
    assert "TotalSegmentator 2.99.0-fake" in r.text and "fast on cpu" in r.text
    assert "roi aorta," in r.text and "context heart:" in r.text  # the heart, apart
    assert set(s.context) == {"heart"} and s.context["heart"] is not s.outline
    assert s.length_unit == "mm"
    assert s.cfd_presets()["speed"][0]["unit"] == "mm/s"
    rec = s.journal[-1]
    assert rec["step"] == "segment" and rec["params"]["fast"] == 1
    assert rec["input_sha256"] == s.input_sha256
    wf = s.save_workfile(tmp_path / "seg.mfwork")
    fresh = load_workfile(wf)
    out = replay(fresh)
    assert [x.step for x in out] == ["segment"]
    assert fresh.surface.n_cells == s.surface.n_cells
    assert set(fresh.context) == {"heart"}  # the replay brings the heart back too
    s.generate_synth("straight_tube")
    assert s.outline is None and s.context == {}  # cleared with the outline


def test_web_segment_endpoint(tmp_path, fake_ts):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from meshforge.web.app import create_app

    ct = _write_lps_nrrd(tmp_path / "ct.nrrd", _tube_ct())
    client = TestClient(create_app())
    schema = client.get("/schema").json()
    assert schema["segment"]["device"]["default"] == "mps"
    r = client.post("/step/segment", json={"path": str(ct), "device": "cpu"})
    assert r.status_code == 200, r.text
    bad = client.post("/step/segment", json={"path": str(ct), "license": "x"})
    assert bad.status_code == 422  # unknown knob refused (extra="forbid")


def test_cli_segment_error_is_loud(tmp_path, fake_ts, monkeypatch, capsys):
    from meshforge.cli import main

    monkeypatch.setenv("FAKE_TS_MODE", "empty")
    ct = _write_lps_nrrd(tmp_path / "ct.nrrd", _tube_ct())
    rc = main(["segment", str(ct), "--outputs", str(tmp_path / "o")])
    assert rc == 1
    assert EMPTY_MASK_MESSAGE in capsys.readouterr().err


def test_ct_ceiling_admits_a_full_body_cta():
    # A chest-to-pelvis CTA at 0.6 mm is ~1100-1200 slices of 512 x 512; the
    # first ceiling (1024 slices) refused a real public AAA scan outright.
    from meshforge.io.segment import MAX_CT_VOXELS

    assert MAX_CT_VOXELS >= 512 * 512 * 1800


# ---------------------------------------------------------------------------
# D85: the contrast blood lumen = AI mask AND CT >= 150 HU (a threshold)
# ---------------------------------------------------------------------------
def _lumen_ct() -> np.ndarray:
    """A 120 HU thrombus-like rim (inside the fake's > 100 HU 'aorta'), a
    300 HU contrast core of radius 5 voxels, and a detached 800 HU
    calcification blob in the rim."""
    ii, jj = np.indices(SHAPE[:2])
    r2 = (ii - TUBE_C[0]) ** 2 + (jj - TUBE_C[1]) ** 2
    ct = np.full(SHAPE, -100, dtype=np.int16)
    ct[r2 <= TUBE_R_VOX**2] = 120
    ct[r2 <= 25] = 300
    ct[31:33, 20:22, 10:13] = 800  # r 7..8.1 voxels: in the rim, not touching the core
    return ct


def test_ct_lumen_is_the_bright_core_of_the_mask_and_drops_detached_calcium(tmp_path):
    from meshforge.io.segment import ct_lumen_surface, write_ct_nifti

    ct = _lumen_ct()
    aff = np.diag([SPACING, SPACING, SLICE, 1.0])
    path = tmp_path / "ct dir" / "ct.nii.gz"
    path.parent.mkdir()
    write_ct_nifti(ct, aff, path)
    mask = ct > 100
    surf, summ = ct_lumen_surface(path, mask, aff, SHAPE, aff)
    core = int((ct == 300).sum())
    assert summ["kept_voxels"] == core
    assert summ["dropped_components"] == 1 and summ["dropped_voxels"] == 12
    assert summ["above_threshold_voxels"] == core + 12
    assert summ["dice_vs_ai_mask"] == pytest.approx(2 * core / (int(mask.sum()) + core), abs=1e-4)
    assert summ["fraction_of_mask"] == pytest.approx(core / int(mask.sum()), abs=1e-4)
    assert "THRESHOLD" in summ["method"] and "not an AI segmentation" in summ["method"]
    assert surf.n_open_edges == 0
    # the core cylinder (radius 5 voxels, open to the volume ends) inside the rim
    assert summ["surface"]["enclosed_volume_ml"] == pytest.approx(
        core * SPACING * SPACING * SLICE / 1000.0, rel=0.15)
    with pytest.raises(ValueError, match="reaches 5000 HU"):
        ct_lumen_surface(path, mask, aff, SHAPE, aff, lumen_hu=5000.0)
    with pytest.raises(ValueError, match="not on the CT grid"):
        ct_lumen_surface(path, mask[:-1], aff, SHAPE, aff)
    with pytest.raises(ValueError, match="not on the CT grid"):
        ct_lumen_surface(path, mask, aff * 1.1, SHAPE, aff)


def test_cli_segment_writes_the_threshold_lumen(tmp_path, fake_ts, capsys):
    from meshforge.cli import main
    from meshforge.io.segment import write_ct_nifti

    ct_path = tmp_path / "ct.nii.gz"
    write_ct_nifti(_lumen_ct(), np.diag([SPACING, SPACING, SLICE, 1.0]), ct_path)
    out = tmp_path / "out" / "aorta.vtp"
    lumen = tmp_path / "out" / "lumen.vtp"
    rc = main(["segment", str(ct_path), "--out", str(out), "--fast", "--device", "cpu",
               "--lumen-out", str(lumen)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert lumen.is_file() and out.is_file()
    assert "CT THRESHOLD inside the AI label" in printed
    assert "dropped 1 piece(s) / 12 voxels" in printed
    summary = json.loads((tmp_path / "out" / "aorta_segment.json").read_text())
    assert summary["lumen"]["lumen_hu"] == 150.0
    assert summary["lumen"]["dropped_voxels"] == 12
