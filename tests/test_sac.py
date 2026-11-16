"""D86 (M2): outer aneurysm sac recovery from a CT + lumen-only mask.

Synthetic phantoms only (no real data in the default run): a bright
eccentric lumen (300 HU) inside a thrombus disk (40 HU) inside fat
(-100 HU), radius varying along z, Gaussian noise 10 HU (seeded). The
analytic sac edge along a ray from origin ``o`` in direction ``u`` is the
far root of |o + t u - c|^2 = R^2.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from meshforge.geometry import sac as sacmod
from meshforge.geometry.sac import (
    BRIGHT,
    NO_FAT,
    RESOLVED,
    recover_sac,
    recover_sac_from_files,
    sac_surface,
)

SP, DZ = 0.6, 1.0
SHAPE = (170, 130, 24)
SAC_C = (40.0, 39.0)  # sac centre, world mm (identity-like affine)
LUM_OFF = 7.0  # lumen centre = sac centre + (7, 0): an eccentric lumen
R_LUMEN = 10.0


def r_sac(z):
    return 22.0 + 3.0 * np.sin(np.asarray(z, dtype=float) / SHAPE[2] * np.pi)


def _grid_xy(shape=SHAPE, sp=SP):
    return np.meshgrid(np.arange(shape[0]) * sp, np.arange(shape[1]) * sp, indexing="ij")


def _phantom(*, noise=10.0, seed=0, shape=SHAPE):
    """(ct, lumen, affine) with world = (i*SP, j*SP, k*DZ)."""
    x, y = _grid_xy(shape)
    ct = np.full(shape, -100.0, dtype=np.float32)
    lumen = np.zeros(shape, dtype=bool)
    lc = (SAC_C[0] + LUM_OFF, SAC_C[1])
    for k in range(shape[2]):
        sac = (x - SAC_C[0]) ** 2 + (y - SAC_C[1]) ** 2 <= r_sac(k * DZ) ** 2
        lum = (x - lc[0]) ** 2 + (y - lc[1]) ** 2 <= R_LUMEN**2
        ct[:, :, k][sac] = 40.0
        ct[:, :, k][lum] = 300.0
        lumen[:, :, k] = lum
    if noise:
        ct += np.random.default_rng(seed).normal(0.0, noise, shape).astype(np.float32)
    return ct, lumen, np.diag([SP, SP, DZ, 1.0])


def _world_origin(res, s):
    o = res.origins[s]
    return res.affine[:3, :3] @ np.array([o[0], o[1], res.slices[s]]) + res.affine[:3, 3]


def _ray_dirs(res):
    return np.cos(res.angles)[:, None] * res.basis[0] + np.sin(res.angles)[:, None] * res.basis[1]


def _analytic_radius(res, s, centre_xy, radius):
    o = _world_origin(res, s)[:2]
    u = _ray_dirs(res)[:, :2]
    d = o - np.asarray(centre_xy)
    b = u @ d
    return -b + np.sqrt(b * b - (d @ d - radius**2))


def _sector(theta_deg, centre_deg, half_deg):
    return np.abs((theta_deg - centre_deg + 180.0) % 360.0 - 180.0) <= half_deg


# ---------------------------------------------------------------------------
# the method
# ---------------------------------------------------------------------------
def test_eccentric_phantom_radius_thickness_and_vertex_field():
    ct, lumen, aff = _phantom()
    res = recover_sac(ct, lumen, aff, roi_mm=0)
    assert res.summary["resolved_fraction"] == 1.0
    assert res.summary["ray_status_counts"]["resolved"] == res.status.size
    err, terr = [], []
    for s in range(len(res.slices)):
        z = res.slices[s] * DZ
        truth = _analytic_radius(res, s, SAC_C, r_sac(z))
        err.append(res.radius[s] - truth)
        terr.append(res.thickness[s] - (truth - _analytic_radius(res, s, (SAC_C[0] + LUM_OFF, SAC_C[1]), R_LUMEN)))
    err, terr = np.abs(np.concatenate(err)), np.abs(np.concatenate(terr))
    assert err.max() <= SP, err.max()  # within one voxel everywhere
    assert np.median(err) < 0.25
    # two pixelated boundaries (lumen and sac): within two voxels, ~half a voxel typical
    assert terr.max() <= 2 * SP and np.median(terr) < 0.5, (terr.max(), np.median(terr))
    # thin side (toward the lumen offset) vs thick side, as the geometry says
    th = res.thickness.mean(axis=0)
    assert th[0] < 8.0 and th[len(th) // 2] > 18.0
    # the sac contains the lumen; volume matches the analytic stack
    assert not np.any(lumen & ~res.mask)
    vox = SP * SP * DZ
    analytic_ml = sum(np.pi * r_sac(k * DZ) ** 2 * DZ for k in range(SHAPE[2])) / 1000
    assert res.summary["sac_volume_ml"] == pytest.approx(analytic_ml, rel=0.02)
    assert res.summary["sac_volume_ml"] == pytest.approx(np.count_nonzero(res.mask) * vox / 1000, abs=1e-3)
    # diameters: the sac is a circle of diameter 2 r_sac(z)
    md = res.summary["max_diameter"]
    assert md["max_caliper"]["value_mm"] == pytest.approx(2 * 25.0, abs=1.0)
    assert md["area_equivalent"]["value_mm"] == pytest.approx(2 * 25.0, abs=1.0)
    # vertex field = analytic thrombus + wall thickness along the ray through it
    surf, prov = sac_surface(res)
    assert prov["watertight"]
    p = np.asarray(surf.points)
    keep = (p[:, 2] > 2.0) & (p[:, 2] < (SHAPE[2] - 3) * DZ)
    p = p[keep]
    field = np.asarray(surf.point_data["ilt_thickness_mm"])[keep]
    lc = np.array([SAC_C[0] + LUM_OFF, SAC_C[1]])
    u = (p[:, :2] - lc) / np.linalg.norm(p[:, :2] - lc, axis=1)[:, None]
    d = lc - np.array(SAC_C)
    b = u @ d
    t_sac = -b + np.sqrt(b * b - (d @ d - r_sac(p[:, 2]) ** 2))
    ferr = np.abs(field - (t_sac - R_LUMEN))
    assert np.median(ferr) < 0.5 and np.percentile(ferr, 95) < 1.0
    assert set(np.unique(surf.point_data["sac_resolved"])) == {1}


def test_organ_without_fat_plane_is_unresolved_and_interpolated():
    """A 40 HU organ abuts the sac over +-32.5 deg (about the lumen centre)
    with no fat plane: exactly those 13 of 72 rays are unresolved."""
    ct, lumen, aff = _phantom()
    x, y = _grid_xy()
    lc = (SAC_C[0] + LUM_OFF, SAC_C[1])
    theta = np.degrees(np.arctan2(y - lc[1], x - lc[0]))
    rho = np.hypot(x - lc[0], y - lc[1])
    organ = _sector(theta, 0.0, 32.5) & (rho < 50.0)
    for k in range(SHAPE[2]):
        sl = ct[:, :, k]
        outside = (x - SAC_C[0]) ** 2 + (y - SAC_C[1]) ** 2 > r_sac(k * DZ) ** 2
        sl[organ & outside] = 40.0 + (sl[organ & outside] + 100.0)  # keep the noise
    res = recover_sac(ct, lumen, aff, roi_mm=0, max_thickness_mm=25.0)
    S = len(res.slices)
    counts = res.summary["ray_status_counts"]
    assert counts["no_fat"] == 13 * S, counts
    assert counts["resolved"] == 59 * S, counts
    assert res.summary["resolved_fraction"] == pytest.approx(59 / 72, abs=1e-4)
    assert res.summary["resolved_fraction_per_slice"]["min"] == pytest.approx(59 / 72, abs=1e-3)
    sector = _sector(np.degrees(res.angles), 0.0, 32.5)
    assert np.all(res.status[:, sector] == NO_FAT)
    for s in range(S):  # filled from the neighbours: still on the circle
        truth = _analytic_radius(res, s, SAC_C, r_sac(res.slices[s] * DZ))
        assert np.abs(res.radius[s, sector] - truth[sector]).max() < 1.5
    surf, _ = sac_surface(res)
    flag = np.asarray(surf.point_data["sac_resolved"])
    assert 0 < np.count_nonzero(flag == 0) < flag.size


def test_bone_and_enhancing_organ_stop_rays_but_calcification_does_not():
    ct, lumen, aff = _phantom()
    x, y = _grid_xy()
    ang = np.degrees(np.arctan2(y - SAC_C[1], x - SAC_C[0]))
    rr = np.hypot(x - SAC_C[0], y - SAC_C[1])
    rng = np.random.default_rng(1)
    regions = {}
    for k in range(SHAPE[2]):
        R = r_sac(k * DZ)
        bone = _sector(ang, -90.0, 25.0) & (rr > R) & (rr < R + 15.0)
        gut = _sector(ang, 90.0, 25.0) & (rr > R) & (rr < R + 12.0)
        calc = _sector(ang, 180.0, 12.0) & (rr > R - 1.0) & (rr <= R)  # 1 mm plaque
        noise = rng.normal(0.0, 10.0, bone.shape).astype(np.float32)
        ct[:, :, k][bone] = (400.0 + noise)[bone]
        ct[:, :, k][gut] = (110.0 + noise)[gut]  # duodenum-like, enhancing
        ct[:, :, k][calc] = (700.0 + noise)[calc]
        regions[k] = (bone, gut)
    res = recover_sac(ct, lumen, aff, roi_mm=0)
    for s in range(len(res.slices)):
        truth = _analytic_radius(res, s, SAC_C, r_sac(res.slices[s] * DZ))
        # where each ray leaves the sac, as an angle about the sac centre
        exit_xy = _world_origin(res, s)[:2] + truth[:, None] * _ray_dirs(res)[:, :2]
        deg = np.degrees(np.arctan2(exit_xy[:, 1] - SAC_C[1], exit_xy[:, 0] - SAC_C[0]))
        for centre in (-90.0, 90.0):
            assert np.all(res.status[s, _sector(deg, centre, 18.0)] == BRIGHT)
        assert np.all(res.status[s, _sector(deg, 180.0, 8.0)] == RESOLVED)  # walked through
        assert np.all(res.radius[s] <= truth + 1.0)  # never into bone or gut
        assert np.abs(res.radius[s] - truth).max() < 1.5
        bone, gut = regions[int(res.slices[s])]
        inside = res.mask[:, :, res.slices[s]]
        assert np.count_nonzero(inside & bone) <= 0.01 * np.count_nonzero(bone)
        assert np.count_nonzero(inside & gut) <= 0.03 * np.count_nonzero(gut)


def test_flipped_axes_decreasing_z_and_roi_from_the_caudal_end():
    """LPS-like negative x/y axes and z decreasing with k: the world-frame
    result is unchanged, and roi_mm counts from the LOWEST world z."""
    ct, lumen, _ = _phantom()
    ni, nj, nk = SHAPE
    aff = np.diag([-SP, -SP, -DZ, 1.0])
    aff[:3, 3] = [(ni - 1) * SP, (nj - 1) * SP, (nk - 1) * DZ]
    # flip the arrays so every voxel keeps its world position
    ct_f, lum_f = ct[::-1, ::-1, ::-1].copy(), lumen[::-1, ::-1, ::-1].copy()
    res = recover_sac(ct_f, lum_f, aff, roi_mm=10.0)
    assert len(res.slices) == 11  # z 0..10 mm inclusive at 1 mm slices
    assert res.slices.min() == nk - 11 and res.slices.max() == nk - 1
    assert res.heights_mm.min() == pytest.approx(0.0) and res.heights_mm.max() == pytest.approx(10.0)
    for s in range(len(res.slices)):
        z = (aff @ [0, 0, res.slices[s], 1])[2]
        truth = _analytic_radius(res, s, SAC_C, r_sac(z))
        assert np.abs(res.radius[s] - truth).max() <= SP
    # no slice outside the ROI is in the mask
    assert not res.mask[:, :, : nk - 11].any()


# ---------------------------------------------------------------------------
# refusals (rule 6)
# ---------------------------------------------------------------------------
def test_refuses_an_empty_lumen():
    ct, lumen, aff = _phantom(noise=0)
    with pytest.raises(ValueError, match="lumen mask is empty"):
        recover_sac(ct, np.zeros_like(lumen), aff)


def test_refuses_when_too_few_rays_find_fat():
    ct, lumen, aff = _phantom(noise=0)
    x, y = _grid_xy()
    ang = np.degrees(np.arctan2(y - SAC_C[1], x - SAC_C[0]))
    for k in range(SHAPE[2]):  # soft tissue (40 HU) everywhere but a 90 deg window
        sl = ct[:, :, k]
        sl[(sl < 0) & ~_sector(ang, 180.0, 45.0)] = 40.0
    with pytest.raises(ValueError, match=r"found a fat plane \(need >= 50%\)"):
        recover_sac(ct, lumen, aff, roi_mm=0)
    res = recover_sac(ct, lumen, aff, roi_mm=0, min_resolved=0.2)  # the gate is the knob
    assert 0.2 <= res.summary["resolved_fraction"] < 0.5


def test_refuses_a_lumen_touching_the_field_of_view_edge():
    ct, lumen, aff = _phantom(noise=0)
    i_c = int((SAC_C[0] + LUM_OFF) / SP)
    lumen[0:i_c, 62:68, 5:9] = True  # a channel from the lumen out to i = 0
    ct[0:i_c, 62:68, 5:9] = 300.0
    with pytest.raises(ValueError, match="in-plane field-of-view edge"):
        recover_sac(ct, lumen, aff, roi_mm=0)


def test_refuses_a_non_axial_grid_and_bad_parameters():
    ct, lumen, aff = _phantom(noise=0)
    c, s_ = np.cos(np.radians(30)), np.sin(np.radians(30))
    rot = np.eye(4)
    rot[1:3, 1:3] = [[c, -s_], [s_, c]]
    with pytest.raises(ValueError, match="off axial"):
        recover_sac(ct, lumen, rot @ aff)
    with pytest.raises(ValueError, match="fat_hu"):
        recover_sac(ct, lumen, aff, fat_hu=200.0)
    with pytest.raises(ValueError, match="n_rays"):
        recover_sac(ct, lumen, aff, n_rays=8)
    with pytest.raises(ValueError, match="min_resolved"):
        recover_sac(ct, lumen, aff, min_resolved=0.0)
    with pytest.raises(ValueError, match="same 3-D grid"):
        recover_sac(ct, lumen[:-1], aff)


# ---------------------------------------------------------------------------
# files, session step, CLI
# ---------------------------------------------------------------------------
def _write_case(dirpath, *, centred=False, mask_shape=None):
    """CT (int16) + lumen mask (uint8) NIfTI with LPS-like negative x/y axes."""
    from meshforge.io.segment import write_ct_nifti

    ct, lumen, _ = _phantom()
    ni, nj, nk = SHAPE
    aff = np.diag([-SP, -SP, DZ, 1.0])
    # world (x, y) = -(i, j) * SP + origin; put the sac centre where asked
    cx, cy = (0.0, 0.0) if centred else (12.0, -30.0)
    aff[:3, 3] = [cx + SAC_C[0], cy + SAC_C[1], -12.0]
    dirpath.mkdir(parents=True, exist_ok=True)
    ct_p, m_p = dirpath / "ct scan.nii.gz", dirpath / "lumen mask.nii.gz"
    write_ct_nifti(np.rint(ct).astype(np.int16), aff, ct_p)
    m = lumen.astype(np.uint8)
    if mask_shape is not None:
        m = np.zeros(mask_shape, np.uint8)
    write_ct_nifti(m, aff, m_p)
    return ct_p, m_p, aff


def test_from_files_reads_only_and_reports(tmp_path):
    from meshforge.io.load import file_sha256
    from meshforge.quality.check import check_surface

    ct_p, m_p, aff = _write_case(tmp_path / "in put")
    before = (file_sha256(ct_p), file_sha256(m_p))
    run = recover_sac_from_files(ct_p, m_p)
    assert (file_sha256(ct_p), file_sha256(m_p)) == before
    assert run.input_sha256 == {"ct": before[0], "lumen_mask": before[1]}
    assert run.grid_shape == SHAPE and np.allclose(run.grid_affine, aff)
    s = run.summary
    assert s["resolved_fraction"] == 1.0 and s["surface"]["watertight"]
    assert set(s["surface"]["point_data"]) == {"ilt_thickness_mm", "sac_resolved"}
    assert {"ilt_thickness_mm", "sac_resolved"} <= set(run.surface.point_data.keys())
    assert check_surface(run.surface, path=str(ct_p), sha256=before[0]).mesh_ready
    json.dumps(s)  # plain JSON types throughout
    # the surface sits where the phantom's sac is in the world (x, y flipped)
    c = np.asarray(run.surface.points)[:, :2].mean(axis=0)
    assert np.allclose(c, [12.0, -30.0], atol=1.5)


def test_from_files_refuses_a_mismatched_or_empty_mask(tmp_path):
    ct_p, m_p, _ = _write_case(tmp_path / "a", mask_shape=(SHAPE[0], SHAPE[1], SHAPE[2] + 1))
    with pytest.raises(ValueError, match="does not match the CT grid"):
        recover_sac_from_files(ct_p, m_p)
    ct_p, m_p, _ = _write_case(tmp_path / "b", mask_shape=SHAPE)
    with pytest.raises(ValueError, match="is empty"):
        recover_sac_from_files(ct_p, m_p)
    with pytest.raises(FileNotFoundError, match="lumen mask not found"):
        recover_sac_from_files(ct_p, tmp_path / "nope.nii.gz")


def test_session_sac_step_journal_and_replay(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from meshforge.pipeline import STEP_KNOBS, STEP_ORDER, MeshingSession, load_workfile, replay

    assert STEP_ORDER.index("sac") == STEP_ORDER.index("segment") + 1
    assert set(STEP_KNOBS["sac"]) == {
        "path", "lumen_mask", "lumen_hu", "fat_hu", "max_thickness_mm", "n_rays",
        "roi_mm", "min_resolved", "smoothing",
    }
    ct_p, m_p, _ = _write_case(tmp_path / "case")
    s = MeshingSession()
    with pytest.raises(ValueError, match="sac needs a CT volume path and a lumen mask"):
        s.compute_sac(path=str(ct_p))
    r = s.compute_sac(path=f'"{ct_p}"', lumen_mask=str(m_p), n_rays=48)
    assert r.summary["sac"]["n_rays"] == 48
    assert "resolved (fat plane found): 100.0% of rays" in r.text
    assert s.length_unit == "mm" and s.surface is not None
    rec = s.journal[-1]
    assert rec["step"] == "sac" and rec["params"]["n_rays"] == 48
    wf = s.save_workfile(tmp_path / "sac.mfwork")
    fresh = load_workfile(wf)
    out = replay(fresh)
    assert [x.step for x in out] == ["sac"]
    assert fresh.surface.n_cells == s.surface.n_cells


def test_cli_sac_refusal_is_loud(tmp_path, capsys):
    from meshforge.cli import main

    ct_p, m_p, _ = _write_case(tmp_path / "x", mask_shape=SHAPE)  # empty mask
    rc = main(["sac", str(ct_p), "--lumen-mask", str(m_p), "--outputs", str(tmp_path / "o")])
    assert rc == 1
    assert "is empty" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_constants_are_reported():
    ct, lumen, aff = _phantom(noise=0)
    res = recover_sac(ct, lumen, aff, roi_mm=0)
    for key in ("fat_run_mm", "min_wall_mm", "bright_run_mm", "enhanced_hu",
                "slope_mm_per_deg", "outlier_mm", "fourier_order"):
        assert key in res.params
    assert res.params["enhanced_hu"] == sacmod.ENHANCED_HU
