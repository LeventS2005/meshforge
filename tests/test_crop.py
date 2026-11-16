"""D85 (M2): crop an aorta surface to its infrarenal aneurysm segment.

Ground truth is the synthetic ``aneurysm_tube`` (straight and gently curved):
its exact radius profile fixes every landmark, the cut planes and the cropped
volume analytically.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from meshforge.geometry.crop import (
    CropError,
    crop_aaa,
    crop_with_planes,
    cut_and_cap,
    planes_from_report,
    _ear_clip,
)
from meshforge.synth.generators import aneurysm_radius, aneurysm_tube

PROFILE = dict(radius=10.0, sac_radius=25.0, sac_center=60.0, sac_sigma=15.0,
               flare=4.0, flare_length=6.0)


def _s_of(truth, point) -> float:
    """Arc length of the truth centreline sample nearest ``point``."""
    return float(truth.arc_lengths[np.argmin(np.linalg.norm(truth.points - point, axis=1))])


def _tangent_at(truth, point) -> np.ndarray:
    return truth.tangents[np.argmin(np.linalg.norm(truth.points - point, axis=1))]


def _analytic_landmarks():
    s = np.linspace(0.0, 160.0, 160001)
    d = 2.0 * aneurysm_radius(s, **PROFILE)
    flare_top = float(s[np.argmin(np.where(s < 60.0, d, np.inf))])
    d_ref = 2.0 * PROFILE["radius"]  # the narrowest section above the sac
    thr = max(1.2 * d_ref, d_ref + 2.0)
    above = s > 60.0
    neck = float(s[above][np.argmax(d[above] <= thr)])
    return {"flare_top": flare_top, "sac": 60.0, "neck": neck, "d_sac": float(d.max()),
            "thr": thr, "s": s, "d": d}


@pytest.fixture(scope="module", params=[0.0, 300.0], ids=["straight", "curved"])
def cropped(request):
    surface, truth = aneurysm_tube(bend_radius=request.param)
    inner, _ = aneurysm_tube(bend_radius=request.param, radius=7.0, sac_radius=16.0,
                             flare=2.0)
    res = crop_aaa(surface, lumen=inner)
    return surface, truth, res


def _landmark_point(res, h):
    cl = res.centerline
    return cl.points[int(np.argmin(np.abs(cl.arc_lengths - h)))]


def test_landmarks_match_the_analytic_profile(cropped):
    _, truth, res = cropped
    lm, ref = res.landmarks, _analytic_landmarks()
    assert _s_of(truth, res.centerline.points[0]) == pytest.approx(0.0, abs=2.0)  # bifurcation
    assert _s_of(truth, _landmark_point(res, lm.h_flare_top)) == pytest.approx(
        ref["flare_top"], abs=3.0)
    assert _s_of(truth, _landmark_point(res, lm.h_sac)) == pytest.approx(ref["sac"], abs=3.0)
    assert _s_of(truth, _landmark_point(res, lm.h_neck_start)) == pytest.approx(
        ref["neck"], abs=2.0)
    assert lm.d_sac == pytest.approx(ref["d_sac"], rel=0.02)
    assert lm.d_ref == pytest.approx(20.0, rel=0.02)
    assert lm.threshold == pytest.approx(ref["thr"], rel=0.02)
    # neck diameter: median of the exact profile over the first 10 mm of the neck
    sel = (ref["s"] >= ref["neck"]) & (ref["s"] <= ref["neck"] + 10.0)
    assert lm.d_neck == pytest.approx(float(np.median(ref["d"][sel])), rel=0.03)
    assert lm.sac_neck_ratio == pytest.approx(lm.d_sac / lm.d_neck, rel=1e-9)
    assert lm.neck_open_ended  # the neck runs to the proximal cap


def test_planes_sit_where_the_rules_put_them(cropped):
    _, truth, res = cropped
    lm, ref = res.landmarks, _analytic_landmarks()
    distal, proximal = res.planes
    assert _s_of(truth, distal.origin) == pytest.approx(ref["flare_top"] + 10.0, abs=3.0)
    assert _s_of(truth, proximal.origin) == pytest.approx(ref["neck"] + 25.0, abs=2.5)
    assert distal.height == pytest.approx(lm.h_flare_top + 10.0, abs=0.5)


def test_caps_are_planar_square_and_watertight(cropped):
    _, truth, res = cropped
    out = res.surface
    assert out.n_open_edges == 0 and out.is_manifold
    assert all((np.asarray(out.faces).reshape(-1, 4)[:, 0] == 3))
    assert out.volume > 0  # outward
    for pl in res.planes:
        cap = res.caps[pl.name]
        assert cap.planarity <= 1e-6
        # square to the true vessel axis
        cos = abs(float(_tangent_at(truth, pl.origin) @ pl.normal))
        assert math.degrees(math.acos(min(1.0, cos))) <= 1.0
        # every surface vertex on this cap's plane lies on it to 1e-6
        d = (np.asarray(out.points) - pl.origin) @ pl.normal
        assert (d <= 1e-6).all()  # nothing beyond the cut
        assert cap.n_rim >= 12 and cap.crease > 45.0


def test_embedded_centreline_leaves_each_cap_along_its_normal(cropped):
    from meshforge.geometry import centerline_from_surface

    _, _, res = cropped
    cl = centerline_from_surface(res.surface)
    assert cl is not None
    distal, proximal = res.planes
    for end, pl, sign in ((0, distal, -1.0), (-1, proximal, 1.0)):
        cos = float(cl.tangents[end] @ (sign * pl.normal))
        assert math.degrees(math.acos(min(1.0, cos))) <= 1.0
        gap = float(-((cl.points[end] - pl.origin) @ pl.normal))
        assert 0.0 < gap < 1.0  # just inside the cap


def test_cropped_volume_matches_the_analytic_integral(cropped):
    _, truth, res = cropped
    s_d, s_p = (_s_of(truth, pl.origin) for pl in res.planes)
    s = np.linspace(s_d, s_p, 20001)
    exact = float(np.trapezoid(math.pi * aneurysm_radius(s, **PROFILE) ** 2, s))
    assert res.surface.volume == pytest.approx(exact, rel=0.01)


def test_the_lumen_shares_the_two_cap_planes(cropped):
    _, _, res = cropped
    lum = res.lumen
    assert lum is not None and lum.n_open_edges == 0 and lum.volume > 0
    assert lum.volume < res.surface.volume
    pts = np.asarray(lum.points)
    for pl in res.planes:
        on = np.abs((pts - pl.origin) @ pl.normal) <= 1e-6
        assert on.sum() >= 12  # its cap lies on the very same plane
        assert res.lumen_caps[pl.name].planarity <= 1e-6
        assert (((pts - pl.origin) @ pl.normal) <= 1e-6).all()


def test_report_is_json_and_reuses_its_planes(cropped):
    surface, _, res = cropped
    rep = json.loads(json.dumps(res.report()))
    assert rep["landmarks"]["sac_neck_ratio"] == pytest.approx(res.landmarks.sac_neck_ratio,
                                                               rel=1e-3)
    assert set(rep["planes"]) == {"distal", "proximal"}
    again, caps, _ = crop_with_planes(surface, planes_from_report(rep))
    assert again.volume == pytest.approx(res.surface.volume, rel=1e-6)
    assert caps["distal"].planarity <= 1e-6


def test_refuses_a_tube_without_a_sac():
    from meshforge.synth.generators import straight_tube

    surface, _ = straight_tube(radius=10.0, length=160.0, n_axial=81)
    with pytest.raises(CropError, match=r"no sac: .*h mm:d mm"):
        crop_aaa(surface)


def test_refuses_a_sac_with_no_neck_above_it():
    surface, _ = aneurysm_tube(sac_center=160.0)  # the surface ends in the sac
    with pytest.raises(CropError, match=r"no neck: .*h mm:d mm"):
        crop_aaa(surface)


def test_input_guards():
    surface, _ = aneurysm_tube()
    with pytest.raises(ValueError, match="both"):
        crop_aaa(surface, distal_height_mm=20.0)
    with pytest.raises(ValueError, match="manual heights"):
        crop_aaa(surface, distal_height_mm=120.0, proximal_height_mm=40.0)
    with pytest.raises(ValueError, match="neck_extension_mm"):
        crop_aaa(surface, neck_extension_mm=-1.0)
    with pytest.raises(CropError, match="open edges"):
        crop_aaa(surface.extract_cells(range(10, surface.n_cells)).extract_surface(
            algorithm="dataset_surface"))


def test_a_cut_through_an_existing_flat_cap_is_refused():
    from meshforge.geometry.crop import CutPlane

    surface, _ = aneurysm_tube()
    # a plane tilted 45 degrees, 3 mm above the flat caudal cap: it runs through it
    n = np.array([0.0, math.sin(math.pi / 4), -math.cos(math.pi / 4)])
    planes = [CutPlane("distal", np.array([0.0, 0.0, 3.0]), n, 3.0),
              CutPlane("proximal", np.array([0.0, 0.0, 150.0]), np.array([0.0, 0.0, 1.0]), 150.0)]
    with pytest.raises(CropError, match="flat end cap the input already had"):
        cut_and_cap(surface, planes, np.array([0.0, 0.0, 60.0]))


def test_ear_clip_triangulates_a_crescent_into_positive_triangles():
    t1 = np.linspace(-math.pi / 2, math.pi / 2, 120)
    t2 = np.linspace(math.pi / 2, -math.pi / 2, 120)[1:-1]
    outer = np.column_stack([20 * np.cos(t1), 20 * np.sin(t1)])
    inner = np.column_stack([12 * np.cos(t2) + 6, 12 * np.sin(t2)])
    poly = np.vstack([outer, inner])
    tri = _ear_clip(poly)
    assert tri is not None and tri.shape == (len(poly) - 2, 3)
    p = poly[tri]
    area = 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                  - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
    x, y = poly[:, 0], poly[:, 1]
    exact = 0.5 * float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum())
    assert (area > 0).all()
    assert area.sum() == pytest.approx(exact, rel=1e-12)
    assert _ear_clip(poly[::-1]) is None  # clockwise: refused, not flipped silently


def test_aneurysm_tube_truth():
    surface, truth = aneurysm_tube()
    assert truth.radii.max() == pytest.approx(25.0, rel=1e-3)
    assert surface.volume == pytest.approx(truth.volume, rel=0.005)
    with pytest.raises(ValueError, match="bend_radius"):
        aneurysm_tube(bend_radius=20.0)


# ---------------------------------------------------------------------------
# service layer + CLI
# ---------------------------------------------------------------------------
def test_session_crop_step_journal_replay_and_lumen(tmp_path):
    from meshforge.pipeline import STEP_KNOBS, STEP_ORDER, MeshingSession, load_workfile, replay

    # segment -> sac -> crop: the crop cuts the lumen and the recovered sac
    # with the same two planes
    assert STEP_ORDER.index("crop") == STEP_ORDER.index("sac") + 1
    assert STEP_KNOBS["crop"]["neck_extension_mm"][1] == 25.0
    inner, _ = aneurysm_tube(radius=7.0, sac_radius=16.0, flare=2.0)
    inner_path = tmp_path / "lumen in.vtp"
    inner.save(inner_path)
    s = MeshingSession()
    s.generate_synth("aneurysm_tube")
    with pytest.raises(ValueError, match="both lumen"):
        s.compute_crop(lumen=str(inner_path))
    out_lumen = tmp_path / "out" / "lumen_crop.vtp"
    r = s.compute_crop(lumen=f'"{inner_path}"', lumen_out=str(out_lumen))
    crop = r.summary["crop"]
    assert crop["landmarks"]["sac_neck_ratio"] > 2.0 and "profile" not in crop
    assert "millimetres assumed" in crop["notes"][-1]
    assert "landmarks   : sac" in r.text and "distal cap" in r.text
    assert out_lumen.is_file()
    assert s.surface.n_open_edges == 0 and s.centerline is None
    assert s.journal[-1]["step"] == "crop"
    vol = s.surface.volume
    wf = s.save_workfile(tmp_path / "crop.mfwork")
    fresh = load_workfile(wf)
    assert [x.step for x in replay(fresh)] == ["generate_synth", "crop"]
    assert fresh.surface.volume == pytest.approx(vol, rel=1e-9)
    s.length_unit = "m"
    with pytest.raises(ValueError, match="millimetres"):
        s.compute_crop()


def test_a_second_crop_recuts_the_uncropped_surface(tmp_path):
    """D93: the copilot crops, then re-crops with parallel caps. Cutting the
    first crop again trimmed the s1364 sac from 134.8 to 93.7 mL."""
    from meshforge.pipeline import MeshingSession, load_workfile, replay

    s = MeshingSession()
    s.generate_synth("aneurysm_tube")
    first = s.compute_crop()
    assert not any("re-cropped" in n for n in first.summary["crop"]["notes"])
    v1 = s.surface.volume
    again = s.compute_crop()
    assert any("re-cropped from the uncropped" in n for n in again.summary["crop"]["notes"])
    assert s.surface.volume == pytest.approx(v1, rel=1e-9)
    # replay runs the same two crops and lands on the same surface
    fresh = load_workfile(s.save_workfile(tmp_path / "recrop.mfwork"))
    replay(fresh)
    assert fresh.surface.volume == pytest.approx(v1, rel=1e-9)
    # a surface replaced since the crop is cropped as it is
    s.generate_synth("aneurysm_tube")
    fresh_crop = s.compute_crop()
    assert not any("re-cropped" in n for n in fresh_crop.summary["crop"]["notes"])


def test_volmesh_reports_capped_and_floored_rays(tmp_path):
    from meshforge.meshing import get_backend
    from meshforge.synth import generate

    surface, _ = generate("curved_tube", radius=2.0, bend_radius=4.0)
    wall = get_backend("swept").mesh_wall(surface, 1.5, n_theta=16, n_axial=30)
    lat = wall.summary()["lattice"]
    assert lat["n_capped"] > 0 and lat["n_floored"] > 0 and lat["curvature_cap"] == 0.5
    assert lat["n_rays"] == 16 * 30 and lat["thickness_min"] < 1.5
    loose = get_backend("swept").mesh_wall(surface, 1.5, n_theta=16, n_axial=30,
                                           curvature_cap=0.9)
    assert loose.lattice["n_capped"] < lat["n_capped"]


def test_cli_crop_then_volmesh_and_planes_reuse(tmp_path, capsys):
    from meshforge.cli import main

    outer, _ = aneurysm_tube()
    inner, _ = aneurysm_tube(radius=7.0, sac_radius=16.0, flare=2.0)
    src, lum = tmp_path / "aorta.vtp", tmp_path / "lumen.vtp"
    outer.save(src)
    inner.save(lum)
    out = tmp_path / "crop" / "aaa.vtp"
    rc = main(["crop", str(src), "--out", str(out), "--lumen", str(lum)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    report = out.with_name("aaa_crop.json")
    assert out.is_file() and out.with_name("aaa_lumen.vtp").is_file() and report.is_file()
    assert "sac/neck" in printed and "mesh_ready=True" in printed
    rep = json.loads(report.read_text())
    assert rep["check"]["mesh_ready"] is True and len(rep["profile"]["h_mm"]) > 100

    rc = main(["volmesh", str(out), "--thickness", "1.5", "--curvature-cap", "0.9",
               "--out", str(tmp_path / "wall.vtu")])
    printed = capsys.readouterr().out
    assert rc == 0 and "wall rays   : capped=0  floored=0" in printed

    again = tmp_path / "again.vtp"
    rc = main(["crop", str(src), "--out", str(again), "--planes-from", str(report)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    import pyvista as pv

    assert pv.read(again).volume == pytest.approx(pv.read(out).volume, rel=1e-6)
    assert "planes reused" in printed

    from meshforge.synth.generators import straight_tube

    plain, _ = straight_tube(radius=10.0, length=160.0, n_axial=81)
    plain.save(tmp_path / "plain.vtp")
    rc = main(["crop", str(tmp_path / "plain.vtp"), "--outputs", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 1 and "no sac" in captured.err


def test_a_distal_margin_reaching_the_sac_is_held_halfway():
    surface, _ = aneurysm_tube()
    res = crop_aaa(surface, distal_margin_mm=60.0)
    lm = res.landmarks
    assert res.planes[0].height == pytest.approx(0.5 * (lm.h_flare_top + lm.h_sac), abs=0.5)
    assert any("halfway between the flare top and the sac" in n for n in res.notes)
