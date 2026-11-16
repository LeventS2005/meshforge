"""D92 (M5): the Claude copilot, with a mocked client and a scripted backend.

No network, no API key, no heavy compute: the client replays scripted
responses and the backend returns canned step results. The loop, the gate
recovery, the numbers guard, the request shape and the report are tested.
"""
from __future__ import annotations

import builtins
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from meshforge.copilot import agent as ag
from meshforge.copilot.backend import HttpBackend, StepOutcome
from meshforge.copilot.guard import check_numbers, numbers_in
import re

from meshforge.copilot.report import (
    REPORT_HTML_MARKER,
    render_report,
    shown_path,
    shown_paths,
    twin_card,
    tools_from_journal,
    twin_card_html,
    twin_from_journal,
)
from meshforge.copilot.tools import TOOLS, PipelineTools


# --- fakes ---------------------------------------------------------------------


def _use(name, args, i):
    return NS(type="tool_use", name=name, input=args, id=f"tu_{i}")


def _resp(*blocks, stop="tool_use"):
    return NS(content=list(blocks), stop_reason=stop, model="claude-opus-5-5",
              usage=NS(input_tokens=100, output_tokens=20))


def _text(t):
    return NS(type="text", text=t)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        # snapshot the history as sent (the real SDK serialises at call time)
        self.calls.append({**kw, "messages": list(kw["messages"])})
        return self.responses.pop(0)


class FakeBackend:
    """Canned steps shaped like the real s1364 run (D93): the centreline sweep
    builds a clean wall that is 1.30 mm thick along the normal (gate FAIL);
    the plane sweep refuses caps that are not parallel, and on parallel caps
    builds the 1.50 mm wall that passes."""

    name = "fake"

    def __init__(self, planes_thickness=1.5014, centreline_thickness=1.3013, pws_thickness=None):
        self.steps = []
        self.parallel = False
        self.thickness = {"planes": planes_thickness, "centreline": centreline_thickness}
        # sweep -> the wall under the PWS's top 1 % (default: the wall's median; None = unreported)
        self.pws_thickness = pws_thickness or {}
        self.sweep = None

    def run(self, step, knobs):
        self.steps.append((step, dict(knobs)))
        # the session's real shapes: payload nested under the step name
        if step == "segment":
            return StepOutcome(True, {"segment": {"totalseg": {"wall_time_s": 49.5},
                                                  "mask_volume_ml": 332.6}}, "")
        if step == "sac":
            return StepOutcome(True, {"sac": {"max_diameter": {"max_caliper": {"value_mm": 61.75},
                                                               "area_equivalent": {"value_mm": 52.17}},
                                              "sac_volume_ml": 161.8, "resolved_fraction": 0.661,
                                              "thickness_mm": {"all_rays": {"median": 3.17, "max": 25.52}}}}, "")
        if step == "crop":
            self.parallel = bool(knobs["parallel_caps"])
            return StepOutcome(True, {"crop": {"landmarks": {"sac_d_equiv_mm": 47.6,
                                                             "neck_d_equiv_mm": 20.07,
                                                             "sac_neck_ratio": 2.37}}}, "")
        if step == "volmesh":
            sweep = "centreline" if knobs["sweep"] == "auto" else knobs["sweep"]
            if sweep == "planes" and not self.parallel:
                return StepOutcome(False, {}, "", "ValueError: the end caps are 35.20 deg from "
                                   "parallel (the plane sweep needs <= 2); crop with parallel caps")
            self.sweep = sweep
            return StepOutcome(True, {"n_cells": 7552, "quality": {"n_inverted": 0},
                                      "lattice": {"n_capped": 0, "n_floored": 0, "sweep": sweep,
                                                  "thickness_normal_median": self.thickness[sweep]}},
                               "")
        if step == "aaa_stress":
            h = self.pws_thickness.get(self.sweep, self.thickness[self.sweep])
            pk = {} if h is None else {"top1pct_wall_thickness_mm": {"median": h, "min": 0.92 * h}}
            return StepOutcome(True, {"refused": [],
                                      "systolic": {"von_mises_p99_mpa": 0.278243, "peak": pk},
                                      "whatif": {"von_mises_p99_mpa": 0.238494, "peak": pk},
                                      "whatif_ratio": {"pws": 0.857143},
                                      "thrombus": {"modelled": True}}, "")
        if step == "tissue_index":
            p = knobs["pws_mpa"]
            return StepOutcome(True, {"index": {"p10": p / 1.299, "p50": p / 0.650, "p90": p / 0.308},
                                      "strength_mpa": {"p10": 0.308, "p50": 0.650, "p90": 1.299},
                                      "n_aneurysms": 7, "n_strips": 13}, "")
        return StepOutcome(False, {}, "", f"unexpected step {step}")


GOOD_SUMMARY = (
    "The aneurysm sac measures 61.8 mm at its widest. The first wall mesh was only 1.3 mm thick "
    "along the surface normal (-13.2%), so it failed its quality gate and was rebuilt after "
    "re-cropping with parallel caps. With the thrombus modelled, the "
    "peak wall stress is 278.2 kPa at 140 mmHg and 238.5 kPa at 120 mmHg, a drop of 14.3%. The "
    "investigational index is 0.43 at P50 (0.21 to 0.90), from open strength data on 7 "
    "aneurysms and 13 strips - an investigational index, not a diagnosis."
)


def _script(final=GOOD_SUMMARY):
    seq = [("segment_aorta", {"ct_path": "ct.nii.gz"}), ("crop_aneurysm", {"parallel_caps": False}),
           ("mesh_wall", {"sweep": "auto"}), ("crop_aneurysm", {"parallel_caps": True}),
           ("mesh_wall", {"sweep": "planes"}),
           ("wall_stress", {"systolic_mmhg": 140, "whatif_mmhg": 120}), ("strength_index", {})]
    return [_resp(_use(n, a, i)) for i, (n, a) in enumerate(seq)] + [_resp(_text(final), stop="end_turn")]


def _run(tmp_path, responses, request="Assess this AAA at 140 and 120 mmHg."):
    client = FakeClient(responses)
    tools = PipelineTools(FakeBackend(), tmp_path / "work")
    run = ag.run_copilot(request, tools=tools, client=client, log=lambda m: None)
    return run, client, tools


# --- the loop ------------------------------------------------------------------


def test_full_run_recovers_from_a_failed_gate(tmp_path):
    run, client, tools = _run(tmp_path, _script())
    assert run.stop == "done" and run.complete
    failed = [c for c in run.calls if not c["ok"]]
    assert [c["tool"] for c in failed] == ["mesh_wall"]
    assert "quality gate" in failed[0]["error"] or "parallel" in failed[0]["error"]
    # the failed call reached Claude as an error tool_result, then the recovery ran
    tr = [m for m in run.transcript if m.get("tool_results")]
    assert any(r.get("is_error") for m in tr for r in m["tool_results"])
    assert run.results["mesh_wall"]["gate"] == "PASS"
    assert run.results["wall_stress"]["pws_systolic_kpa"] == 278.2
    seg = run.results["segment_aorta"]
    assert seg["ai_seconds"] == 49.5 and seg["sac_max_diameter_mm"] == 61.8  # nested read
    assert run.summary_text == GOOD_SUMMARY and "traced" in run.guard_note
    assert tools.backend.steps[0] == ("segment", {"path": "ct.nii.gz",
                                                  "lumen_out": str(tmp_path / "work" / "lumen.vtp"),
                                                  "mask_out": str(tmp_path / "work" / "aorta_mask.nii.gz")})


def test_request_shape_follows_the_api_guidance(tmp_path):
    _, client, _ = _run(tmp_path, _script())
    kw = client.calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"]["effort"] == "high"
    assert kw["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in kw["betas"]
    assert "tool_choice" not in kw  # forced tool use is a 400 on this model
    assert {t["name"] for t in kw["tools"]} == {
        "segment_aorta", "crop_aneurysm", "mesh_wall", "wall_stress", "strength_index"}
    assert all(t["strict"] and t["input_schema"]["additionalProperties"] is False for t in TOOLS)
    # append-only history: every request extends the previous one
    for a, b in zip(client.calls, client.calls[1:]):
        assert b["messages"][: len(a["messages"])] == a["messages"]


def test_stress_is_refused_before_a_wall_passes_the_gate(tmp_path):
    tools = PipelineTools(FakeBackend(), tmp_path)
    r = tools.run("wall_stress", {"systolic_mmhg": 140, "whatif_mmhg": 120})
    assert not r.ok and "quality gate" in r.error


def test_the_gate_fails_a_wall_thinner_than_its_nominal_thickness(tmp_path):
    """D93: on s1364 the centreline-swept wall was clean (0/0/0) but 1.30 mm
    thick along the normal, and read 382 vs 279 kPa."""
    tools = PipelineTools(FakeBackend(), tmp_path)
    tools.run("segment_aorta", {"ct_path": "ct.nii.gz"})
    tools.run("crop_aneurysm", {"parallel_caps": False})
    r = tools.run("mesh_wall", {"sweep": "auto"})
    assert not r.ok and not tools.gate_passed and r.payload["gate"] == "FAIL"
    assert r.payload["inverted_cells"] == 0 and r.payload["wall_thickness_measured_mm"] == 1.3
    assert r.payload["wall_thickness_off_percent"] == -13.2
    assert "1.30 mm thick along the surface normal, -13% off the 1.5 mm" in r.error
    assert "parallel_caps=true" in r.error and "sweep 'planes'" in r.error
    tools.run("crop_aneurysm", {"parallel_caps": True})
    ok = tools.run("mesh_wall", {"sweep": "planes"})
    assert ok.ok and tools.gate_passed and ok.payload["wall_thickness_measured_mm"] == 1.5
    # 4.9% off passes, 5.1% fails
    for t, passes in ((1.5 * 1.049, True), (1.5 * 0.949, False)):
        tl = PipelineTools(FakeBackend(planes_thickness=t), tmp_path)
        tl.run("segment_aorta", {"ct_path": "ct.nii.gz"})
        tl.run("crop_aneurysm", {"parallel_caps": True})
        assert tl.run("mesh_wall", {"sweep": "planes"}).ok is passes


def test_a_thin_plane_swept_wall_gets_no_false_recovery_hint(tmp_path):
    tools = PipelineTools(FakeBackend(planes_thickness=1.2), tmp_path)
    tools.run("segment_aorta", {"ct_path": "ct.nii.gz"})
    tools.run("crop_aneurysm", {"parallel_caps": True})
    r = tools.run("mesh_wall", {"sweep": "planes"})
    assert not r.ok and "No setting is known" in r.error and "parallel_caps" not in r.error


def test_an_unmeasured_wall_thickness_fails_the_gate(tmp_path):
    tools = PipelineTools(FakeBackend(planes_thickness=None), tmp_path)
    tools.run("segment_aorta", {"ct_path": "ct.nii.gz"})
    tools.run("crop_aneurysm", {"parallel_caps": True})
    r = tools.run("mesh_wall", {"sweep": "planes"})
    assert not r.ok and "could not be measured" in r.error


# --- D96: the gate under the peak ----------------------------------------------

#: Zenodo P04 (D94): the wall's median is -3.3 %, so the median gate passes, but
#: the peak sits on 0.80 mm columns and its tension read 1.62 x the reference
P04_LIKE = {"centreline_thickness": 1.45, "pws_thickness": {"centreline": 0.80}}
STRESS = ("wall_stress", {"systolic_mmhg": 140, "whatif_mmhg": 120})


def _meshed(tmp_path, parallel, sweep, **kw):
    tools = PipelineTools(FakeBackend(**kw), tmp_path)
    tools.run("segment_aorta", {"ct_path": "ct.nii.gz"})
    tools.run("crop_aneurysm", {"parallel_caps": parallel})
    return tools, tools.run("mesh_wall", {"sweep": sweep})


def test_the_gate_refuses_a_peak_on_a_locally_thin_wall(tmp_path):
    tools, mesh = _meshed(tmp_path, False, "auto", **P04_LIKE)
    assert mesh.ok and mesh.payload["wall_thickness_off_percent"] == -3.3
    r = tools.run(*STRESS)
    assert not r.ok and not tools.gate_passed and tools.pws == {}
    assert r.payload == {"pws_wall_thickness_mm": 0.8, "pws_wall_thickness_off_percent": -46.7,
                         "pws_wall_thickness_min_mm": 0.74, "sweep_used": "centreline",
                         "gate": "FAIL"}  # and no stress number
    assert ("quality gate FAILED under the peak: the peak wall stress sits on wall 0.80 mm thick "
            "along the surface normal (median of its top-1% region; thinnest 0.74 mm), -47% off "
            "the 1.5 mm it should be (limit 5%)") in r.error
    assert "parallel_caps=true" in r.error and "sweep 'planes'" in r.error
    # no stress and no index until the wall is meshed again
    assert "no wall has passed" in tools.run(*STRESS).error
    assert "run wall_stress first" in tools.run("strength_index", {}).error


def test_a_thin_patch_under_a_plane_swept_peak_gets_no_false_recovery_hint(tmp_path):
    tools, _ = _meshed(tmp_path, True, "planes", pws_thickness={"planes": 1.2})
    r = tools.run(*STRESS)
    assert not r.ok and "No setting is known" in r.error and "parallel_caps" not in r.error


def test_the_gate_under_the_peak_limit_and_payload(tmp_path):
    for h, passes in ((1.5 * 1.049, True), (1.5 * 0.951, True),
                      (1.5 * 1.051, False), (1.5 * 0.949, False)):
        tools, _ = _meshed(tmp_path, True, "planes", pws_thickness={"planes": h})
        assert tools.run(*STRESS).ok is passes, h
    # the demo's wall: s1364's plane sweep, 1.501 mm under the peak (D94)
    tools, _ = _meshed(tmp_path, True, "planes")
    r = tools.run(*STRESS)
    assert r.ok and tools.gate_passed and r.payload["pws_systolic_kpa"] == 278.2
    assert r.payload["pws_wall_thickness_mm"] == 1.5
    assert r.payload["pws_wall_thickness_off_percent"] == 0.1


def test_the_gate_under_the_peak_checks_both_pressures():
    from meshforge.copilot.tools import pws_wall_check

    ok = {"top1pct_wall_thickness_mm": {"median": 1.5, "min": 1.4}}
    thin = {"top1pct_wall_thickness_mm": {"median": 0.8, "min": 0.7}}
    for sys_pk, wif_pk, which in ((thin, ok, "systolic"), (ok, thin, "whatif")):
        fields, err = pws_wall_check({"systolic": {"peak": sys_pk}, "whatif": {"peak": wif_pk}},
                                     "planes")
        assert "on wall 0.80 mm thick" in err and fields["gate"] == "FAIL", which
    fields, err = pws_wall_check({"systolic": {"peak": ok}, "whatif": {"peak": ok}}, "planes")
    assert err == "" and fields == {"pws_wall_thickness_mm": 1.5, "pws_wall_thickness_off_percent": 0.0}


def test_an_unreported_thickness_under_the_peak_fails_loudly(tmp_path):
    tools, _ = _meshed(tmp_path, True, "planes", pws_thickness={"planes": None})
    r = tools.run(*STRESS)
    assert not r.ok and "not reported" in r.error and not tools.gate_passed and tools.pws == {}


P04_SUMMARY = (
    "The first wall passed its overall thickness check, but its peak stress sat on wall only "
    "0.8 mm thick, so it was rebuilt after re-cropping with parallel caps. The peak wall stress "
    "is 278.2 kPa at 140 mmHg and 238.5 kPa at 120 mmHg - an investigational result."
)


def test_full_run_recovers_from_a_thin_patch_under_the_peak(tmp_path):
    seq = [("segment_aorta", {"ct_path": "ct.nii.gz"}), ("crop_aneurysm", {"parallel_caps": False}),
           ("mesh_wall", {"sweep": "auto"}), STRESS, ("crop_aneurysm", {"parallel_caps": True}),
           ("mesh_wall", {"sweep": "planes"}), STRESS, ("strength_index", {})]
    client = FakeClient([_resp(_use(n, a, i)) for i, (n, a) in enumerate(seq)]
                        + [_resp(_text(P04_SUMMARY), stop="end_turn")])
    tools = PipelineTools(FakeBackend(**P04_LIKE), tmp_path / "work")
    run = ag.run_copilot("Assess this AAA.", tools=tools, client=client, log=lambda m: None)
    assert run.stop == "done" and run.complete
    assert [c["tool"] for c in run.calls if not c["ok"]] == ["wall_stress"]
    assert run.results["mesh_wall"]["sweep_used"] == "planes"
    assert run.results["wall_stress"]["pws_wall_thickness_mm"] == 1.5
    assert run.summary_text == P04_SUMMARY and "traced" in run.guard_note
    md, _ = render_report(request="r", ct_name="ct.nii.gz", credit="CC BY 4.0", results=run.results,
                          calls=run.calls, summary_text=run.summary_text,
                          guard_note=run.guard_note, model="claude-opus-5-5")
    assert ("FAILED → quality gate FAILED under the peak: the peak wall stress sits on wall "
            "0.80 mm thick") in md


# a new crop or a new mesh is a new wall: the old wall's stress goes with it


def test_a_re_crop_drops_the_old_walls_stress(tmp_path):
    tools, _ = _meshed(tmp_path, True, "planes")
    assert tools.run(*STRESS).ok and tools.pws
    assert tools.run("crop_aneurysm", {"parallel_caps": False}).ok
    assert not tools.gate_passed and tools.pws == {}
    assert tools.run("strength_index", {}).error == "run wall_stress first"
    mesh = tools.run("mesh_wall", {"sweep": "auto"})  # the 1.30 mm centreline wall
    assert not mesh.ok and mesh.payload["gate"] == "FAIL"
    assert tools.run("strength_index", {}).error == "run wall_stress first"
    assert tools.run(*STRESS).error == "no wall has passed the quality gate yet"
    assert [s for s, _ in tools.backend.steps].count("tissue_index") == 0


def test_a_new_wall_that_passes_still_needs_its_own_stress(tmp_path):
    tools, _ = _meshed(tmp_path, True, "planes")
    assert tools.run(*STRESS).ok
    assert tools.run("crop_aneurysm", {"parallel_caps": True}).ok and tools.pws == {}
    assert tools.run("mesh_wall", {"sweep": "planes"}).ok and tools.gate_passed
    assert tools.run("strength_index", {}).error == "run wall_stress first"
    assert tools.run(*STRESS).ok and tools.run("strength_index", {}).ok


def test_a_re_mesh_drops_the_old_walls_stress(tmp_path):
    for sweep, passes in (("planes", True), ("auto", False)):
        tools, _ = _meshed(tmp_path, True, "planes")
        assert tools.run(*STRESS).ok
        assert tools.run("mesh_wall", {"sweep": sweep}).ok is passes and tools.pws == {}
        assert tools.run("strength_index", {}).error == "run wall_stress first", sweep


RESTRESS_SUMMARY = (
    "The aneurysm was re-cropped and its wall meshed again, so the index waited for that "
    "wall's own stress. The peak wall stress is 278.2 kPa at 140 mmHg and 238.5 kPa at "
    "120 mmHg. The investigational index is 0.43 at P50, from open strength data on 7 "
    "aneurysms and 13 strips - an investigational index, not a diagnosis."
)


def test_full_run_stresses_a_re_cropped_wall_before_its_index(tmp_path):
    seq = [("segment_aorta", {"ct_path": "ct.nii.gz"}), ("crop_aneurysm", {"parallel_caps": True}),
           ("mesh_wall", {"sweep": "planes"}), STRESS, ("crop_aneurysm", {"parallel_caps": True}),
           ("mesh_wall", {"sweep": "planes"}), ("strength_index", {}), STRESS,
           ("strength_index", {})]
    client = FakeClient([_resp(_use(n, a, i)) for i, (n, a) in enumerate(seq)]
                        + [_resp(_text(RESTRESS_SUMMARY), stop="end_turn")])
    tools = PipelineTools(FakeBackend(), tmp_path / "work")
    run = ag.run_copilot("Assess this AAA.", tools=tools, client=client, log=lambda m: None)
    assert run.stop == "done" and run.complete
    failed = [(c["tool"], c["error"]) for c in run.calls if not c["ok"]]
    assert failed == [("strength_index", "run wall_stress first")]
    steps = [s for s, _ in tools.backend.steps]  # the index ran once, on the new wall's stress
    assert steps.count("tissue_index") == 2
    assert steps[-3:] == ["aaa_stress", "tissue_index", "tissue_index"]
    assert run.results["wall_stress"]["pws_systolic_kpa"] == 278.2
    assert run.results["strength_index"]["systolic_140mmhg"]["index_p50"] == 0.43
    assert run.summary_text == RESTRESS_SUMMARY and "traced" in run.guard_note


def test_guard_traces_numbers_from_a_failed_gate(tmp_path):
    run, _, _ = _run(tmp_path, _script())
    assert "1.3 mm" in run.summary_text and "traced" in run.guard_note


def test_guard_asks_once_then_accepts_a_corrected_summary(tmp_path):
    bad = GOOD_SUMMARY + " Rupture risk is 12% per year."
    responses = _script(final=bad) + [_resp(_text(GOOD_SUMMARY), stop="end_turn")]
    run, client, _ = _run(tmp_path, responses)
    assert run.summary_text == GOOD_SUMMARY and "after one correction" in run.guard_note
    assert "12" in client.calls[-1]["messages"][-1]["content"]


def test_guard_withholds_a_summary_that_stays_untraceable(tmp_path):
    bad = GOOD_SUMMARY + " Rupture risk is 12% per year."
    responses = _script(final=bad) + [_resp(_text(bad), stop="end_turn")]
    run, _, _ = _run(tmp_path, responses)
    assert run.summary_text is None and "withheld" in run.guard_note and "12" in run.guard_note


def test_a_refusal_stops_the_run_loudly(tmp_path):
    run, _, _ = _run(tmp_path, [_resp(_text(""), stop="refusal")])
    assert run.stop == "refusal" and not run.complete and run.summary_text is None


def test_missing_sdk_is_a_readable_error(monkeypatch, tmp_path):
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "anthropic":
            raise ImportError("no anthropic")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(RuntimeError, match="Anthropic SDK"):
        ag.run_copilot("x", tools=PipelineTools(FakeBackend(), tmp_path))


# --- guard ---------------------------------------------------------------------


def test_numbers_guard_rules():
    src = [{"pws_systolic_kpa": 278.2, "index_p50": 0.43, "change": -14.3, "n": 7}]
    assert numbers_in("P10 P50 s1364 C3D10 at 140 mmHg") == ["140"]
    assert check_numbers("278 kPa, 0.43, drops 14.3%, 7 aneurysms", src) == []
    assert check_numbers("280 kPa", src) == ["280"]  # not a rounding of 278.2
    assert check_numbers("at 120 mmHg", src, allowed_text="use 120 mmHg") == []


# --- report / twin card --------------------------------------------------------


def test_report_has_the_twin_card_and_only_tool_numbers(tmp_path):
    run, _, _ = _run(tmp_path, _script())
    md, page = render_report(request="r", ct_name="ct.nii.gz", credit="CC BY 4.0",
                             results=run.results, calls=run.calls, summary_text=run.summary_text,
                             guard_note=run.guard_note, model="claude-opus-5-5")
    assert "Your twin" in page and "aorta: module 1" in page and "needs a second scan" in page
    assert "7 aneurysms / 13 strips" in page
    assert "278.2 kPa at 140" in md and "FAILED" in md  # the gate failure is on the record
    # the whole first sentence of the failure (not cut at the point in 1.30)
    assert "FAILED → quality gate FAILED: the wall is 1.30 mm thick along the surface normal" in md
    assert "1.5 mm thick along the surface normal — gate PASS" in md
    assert "investigational index, not a diagnosis" in md
    sources = [c["payload"] for c in run.calls if c["ok"]]
    results_section = md.split("## Results")[1].split("## Pipeline")[0]
    assert check_numbers(results_section, sources) == []


def test_report_shows_a_scan_path_by_its_file_name_only():
    calls = [{"tool": "segment_aorta", "input": {"ct_path": "/Users/someone/data/s1364/ct.nii.gz"},
              "ok": True, "seconds": 50.4, "steps": [], "payload": {}, "error": None}]
    md, page = render_report(request="r", ct_name="ct.nii.gz", credit="CC BY 4.0", results={},
                             calls=calls, summary_text=None, guard_note="g", model="m")
    assert "{'ct_path': 'ct.nii.gz'}" in md and "/Users/" not in md and "/Users/" not in page


def test_twin_card_marks_unfilled_slots():
    card = twin_card_html({})
    assert card.count('class="off"') == 7  # nothing run yet: every slot waits



# --- M6: the twin card in the Workbench, and report.html as a real page ------------


class JournalingBackend(FakeBackend):
    """FakeBackend that also keeps the session's journal: one record per step
    that ran, with the summary the step returned (MeshingSession._record)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.journal = []

    def run(self, step, knobs):
        out = super().run(step, knobs)
        if out.ok:
            self.journal.append({"step": step, "params": dict(knobs), "summary": out.summary})
        return out


def _journaled_run(tmp_path):
    client = FakeClient(_script())
    tools = PipelineTools(JournalingBackend(), tmp_path / "work")
    run = ag.run_copilot("Assess this AAA.", tools=tools, client=client, log=lambda m: None)
    return run, tools.backend.journal


def _slot(card, key):
    return next(s for s in card["slots"] if s["key"] == key)


def test_workbench_twin_quotes_exactly_the_copilots_numbers(tmp_path):
    """The Workbench panel replays the session journal through the copilot's own
    tool code: same slots, same values as the report's twin card."""
    run, journal = _journaled_run(tmp_path)
    wb = twin_from_journal(journal)
    rep = twin_card(run.results, None, run.calls)
    assert wb["slots"] == rep["slots"] and wb["lit"] and wb["scan"] == "ct.nii.gz"
    replayed, _, _ = tools_from_journal(journal)  # the same calls, so the same report
    assert [(c["tool"], c["input"], c["ok"], c["payload"]) for c in replayed.calls] == \
        [(c["tool"], c["input"], c["ok"], c["payload"]) for c in run.calls]
    assert replayed.results == run.results
    assert _slot(wb, "stress")["values"] == ["278.2 kPa peak at 140 mmHg"]
    assert _slot(wb, "whatif")["values"] == ["238.5 kPa at 120 mmHg (-14.3 %)", "index P50 0.37"]
    assert _slot(wb, "strength")["values"] == ["index P50 0.43 (P10 0.21 – P90 0.9) at 140 mmHg"]
    assert "7 aneurysms / 13 strips" in _slot(wb, "strength")["detail"]
    assert _slot(wb, "flow")["state"] == "pending" and _slot(wb, "growth")["state"] == "next"
    # never an invented number: every value line traces to a tool payload
    values = " ".join(v for s in wb["slots"] for v in s["values"])
    assert check_numbers(values, [c["payload"] for c in run.calls]) == []


def test_workbench_twin_waits_before_a_run_and_never_mixes_stale_results(tmp_path):
    empty = twin_from_journal([])
    assert not empty["lit"] and empty["scan"] is None
    assert [s["state"] for s in empty["slots"]] == ["pending"] * 6 + ["next"]
    assert all(s["values"] == [] for s in empty["slots"])
    _, journal = _journaled_run(tmp_path)
    steps = [r["step"] for r in journal]
    # mid-run, just after the failed gate: the wall slot says why, stress waits
    mid = twin_from_journal(journal[: steps.index("volmesh") + 1])
    assert _slot(mid, "wall")["note"] == "last wall gate FAIL: 1.3 mm along the normal"
    assert _slot(mid, "stress")["state"] == "pending"
    # a re-crop after the stress makes the stress, the what-if and the index stale
    recrop = twin_from_journal(journal + [journal[steps.index("crop")]])
    assert _slot(recrop, "shape")["state"] == "filled"
    assert {_slot(recrop, k)["state"] for k in ("stress", "whatif", "strength")} == {"pending"}
    # a new source empties the card
    fresh = twin_from_journal(journal + [{"step": "generate_synth", "params": {}, "summary": {}}])
    assert not fresh["lit"] and fresh["scan"] is None


def test_workbench_twin_takes_an_index_only_for_its_own_stress(tmp_path):
    _, journal = _journaled_run(tmp_path)
    for rec in journal:
        if rec["step"] == "tissue_index":
            rec["params"]["pws_mpa"] *= 1.01  # an index run on some other stress
    assert _slot(twin_from_journal(journal), "strength")["state"] == "pending"


def test_report_html_is_a_real_page_with_traced_numbers(tmp_path):
    run, _, _ = _run(tmp_path, _script())
    flow = {"q_lmin": 1.0, "sac_backflow_volume_fraction": 0.1445}
    credit = "TotalSegmentator dataset s1364 (Wasserthal et al.), CC BY 4.0, doi 10.5281/zenodo.10047292"
    md, page = render_report(request="Assess this AAA.", ct_name="ct.nii.gz", credit=credit,
                             results=run.results, calls=run.calls, summary_text=run.summary_text,
                             guard_note=run.guard_note, model="claude-opus-5-5", flow=flow)
    assert "## " not in page and "|---|" not in page and "**" not in page  # no raw markdown
    assert REPORT_HTML_MARKER in page  # only this renderer stamps it
    assert "<h1>" in page and '<table class="grid">' in page and "<script" not in page.lower()
    assert "http://" not in page and "https://" not in page  # no external fetch
    for must in ("Your twin", "aorta: module 1", "needs a second scan", credit, run.guard_note,
                 "Investigational index</b>: peak wall stress divided by",
                 "not a diagnosis and not a rupture prediction", "@media print"):
        assert must in page, must
    assert '<tr class="fail">' in page  # the failed gate stays on the record
    assert "14% of the sac volume flows backward" in page
    # every number in the twin card, the headline tiles and the results traces to a tool
    sources = [c["payload"] for c in run.calls if c["payload"]]
    sources.append({"q_lmin": 1.0, "backflow_percent": 14.45})

    def text(fragment):
        return re.sub(r"<svg.*?</svg>", " ", fragment, flags=re.S)

    shown = text(page.split('<section class="twin')[1].split('<section id="pipeline"')[0])
    shown = re.sub(r"<[^>]+>", " ", shown)
    assert "278.2" in shown and "0.43" in shown and "7552" in shown
    assert check_numbers(shown, sources, allowed_text="aorta: module 1") == []


def test_report_html_escapes_text_from_tools_claude_and_the_command_line(tmp_path):
    run, _, _ = _run(tmp_path, _script())
    calls = [dict(c) for c in run.calls]
    calls[2] = {**calls[2], "error": "<img src=x onerror=alert(1)> gate FAILED. More."}
    _, page = render_report(request="<b>assess</b>", ct_name="<i>ct</i>.nii.gz",
                            credit="CC BY <u>4.0</u>", results=run.results, calls=calls,
                            summary_text="The sac is 61.8 mm.\n\n<script>alert(1)</script>",
                            guard_note="numbers guard: <ok>", model="m")
    lowered = page.lower()
    for raw in ("<script", "<img", "<b>assess", "<i>ct", "<u>4.0", "<ok>"):
        assert raw not in lowered, raw
    for escaped in ("&lt;script&gt;alert(1)&lt;/script&gt;", "&lt;img src=x onerror=alert(1)&gt;",
                    "&lt;b&gt;assess&lt;/b&gt;", "&lt;ok&gt;"):
        assert escaped in page, escaped
    assert "<p>The sac is 61.8 mm.</p>" in page


def test_report_hides_an_absolute_replay_path():
    md, page = render_report(request="r", ct_name="ct.nii.gz", credit="c", results={}, calls=[],
                             summary_text=None, guard_note="g", model="m",
                             replay_hint="meshforge replay /Users/someone/My Runs (1)/session.mfwork")
    assert "`meshforge replay session.mfwork`" in md and "meshforge replay session.mfwork" in page
    assert "/Users/" not in md + page and "someone" not in md + page
    md2, _ = render_report(request="r", ct_name="ct.nii.gz", credit="c", results={}, calls=[],
                           summary_text=None, guard_note="g", model="m",
                           replay_hint="meshforge replay outputs/m5/run/session.mfwork")
    assert "`meshforge replay outputs/m5/run/session.mfwork`" in md2  # relative stays as is




# --- M6: no local folder on screen or in the transcript -----------------------------


def test_shown_paths_cut_local_folders(tmp_path):
    ct = "/Users/someone/My Scans (2)/s1364/ct.nii.gz"
    record = {"request": f"Assess this AAA.\nCT: {ct}",
              "calls": [{"input": {"ct_path": ct}, "error": "read /home/other/x/lumen.vtp: bad"}],
              "work": [str(tmp_path / "run" / "lumen.vtp"), "outputs/m5/run/lumen.vtp", 3, None]}
    out = shown_paths(record, [ct, tmp_path / "run"], cwd=tmp_path)
    flat = repr(out)
    assert "/Users/" not in flat and "someone" not in flat and "/home/" not in flat
    assert out["request"].endswith("CT: ct.nii.gz") and out["calls"][0]["input"]["ct_path"] == "ct.nii.gz"
    assert out["calls"][0]["error"] == "read ~/x/lumen.vtp: bad"
    assert out["work"] == ["run/lumen.vtp", "outputs/m5/run/lumen.vtp", 3, None]  # inside cwd: relative
    assert shown_path(tmp_path / "run" / "report.html", cwd=tmp_path) == "run/report.html"
    assert shown_path("/elsewhere/run/report.html", cwd=tmp_path) == "report.html"
    assert shown_path("outputs/x/report.html", cwd=tmp_path) == "outputs/x/report.html"


def test_report_hides_a_home_folder_inside_a_tool_error():
    calls = [{"tool": "segment_aorta", "input": {"ct_path": "/Users/someone/s/ct.nii.gz"}, "ok": False,
              "seconds": 1.0, "steps": [], "payload": {},
              "error": "segment failed: FileNotFoundError: no CT at /Users/someone/s/ct.nii.gz. More."}]
    md, page = render_report(request="r", ct_name="ct.nii.gz", credit="c", results={}, calls=calls,
                             summary_text=None, guard_note="g", model="m")
    assert "no CT at ~/s/ct.nii.gz" in md and "no CT at ~/s/ct.nii.gz" in page
    assert "someone" not in md + page


def test_copilot_cli_prints_and_records_no_local_folder(tmp_path, monkeypatch, capsys):
    """The CLI prints the report/transcript/replay paths relative to the cwd
    (or by file name) and writes a transcript that names the CT by its file
    name; the work file it replays from is not touched."""
    import meshforge.copilot as cp
    from meshforge.cli import main

    ct = "/Users/someone/scans/s1364/ct.nii.gz"

    class FakeSessionBackend:
        name = "in-process session"

        def __init__(self):
            self.session = NS(save_workfile=lambda p: (Path(p).write_text("{}"), Path(p))[1])

    def fake_run(request, *, tools, model, effort, max_turns, log):
        assert request.endswith(f"CT: {ct}")  # Claude still gets the real path
        return ag.CopilotRun(
            request=request, model=model, stop="done", summary_text=None, guard_note="g",
            calls=[{"tool": "segment_aorta", "input": {"ct_path": ct}, "ok": False, "seconds": 0.1,
                    "steps": ["segment"], "payload": {}, "error": f"segment failed: no file {ct}"}],
            results={}, transcript=[{"role": "user", "content": request}], usage={})

    monkeypatch.setattr(cp, "run_copilot", fake_run)
    monkeypatch.setattr(cp, "SessionBackend", FakeSessionBackend)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    for out_dir, shown in ((cwd / "run", "run/"), (tmp_path / "elsewhere", "")):
        rc = main(["copilot", "Assess this AAA.", "--ct", ct, "--out-dir", str(out_dir)])
        printed = capsys.readouterr().out
        assert rc == 1  # no strength index in the fake run
        assert f"report      : {shown}report.html" in printed
        assert f"transcript  : {shown}transcript.json" in printed
        assert f"replay      : meshforge replay {shown}session.mfwork" in printed
        assert str(tmp_path) not in printed and "/Users/" not in printed
        record = (out_dir / "transcript.json").read_text()
        assert "someone" not in record and str(tmp_path) not in record
        assert json.loads(record)["calls"][0]["input"]["ct_path"] == "ct.nii.gz"
        for name in ("report.md", "report.html"):
            assert "someone" not in (out_dir / name).read_text()


# --- HTTP backend --------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - http.server API
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/step/crop":
            out, code = {"step": "crop", "summary": {"echo": body}, "text": "cropped"}, 200
        else:
            out, code = {"detail": "crop: no neck above the sac"}, 400
        data = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_http_backend_posts_steps_and_reports_400s():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        b = HttpBackend(f"http://127.0.0.1:{srv.server_port}")
        ok = b.run("crop", {"parallel_caps": 1})
        assert ok.ok and ok.summary == {"echo": {"parallel_caps": 1}} and ok.text == "cropped"
        bad = b.run("volmesh", {"sweep": "auto"})
        assert not bad.ok and "400" in bad.error and "no neck" in bad.error
    finally:
        srv.shutdown()


def test_http_backend_unreachable_is_loud():
    with pytest.raises(RuntimeError, match="not reachable"):
        HttpBackend("http://127.0.0.1:9", timeout=2).run("crop", {})


# --- D97: the attached flow metrics live in the session (twin panel == report) -------


S1364_FLOW = {"q_lmin": 1.0, "umax_mm_s": 123.74993133544922,
              "sac_backflow_volume_fraction": 0.14450368879518044,
              "sac_mean_speed_mm_s": 29.72977666215793, "neck_mean_speed_mm_s": 57.14323100196482}
FLOW_LINES = ["14% of the sac volume flows backward", "inflow 1 L/min"]


def _flow_record(metrics=S1364_FLOW):
    """The record the session itself journals (MeshingSession.attach_flow_metrics)."""
    from meshforge.pipeline import MeshingSession

    session = MeshingSession()
    session.attach_flow_metrics(dict(metrics))
    return session.journal[-1]


def test_attached_flow_metrics_fill_the_twin_exactly_as_the_report(tmp_path):
    run, journal = _journaled_run(tmp_path)
    rec = _flow_record()
    assert rec["step"] == "flow_metrics" and rec["params"] == {"metrics": S1364_FLOW}
    assert _slot(twin_from_journal(journal), "flow")["note"] == "not run here"
    wb = twin_from_journal(journal + [rec])
    rep = twin_card(run.results, S1364_FLOW, run.calls)
    assert wb["slots"] == rep["slots"]  # every slot, flow included, byte for byte
    flow = _slot(wb, "flow")
    assert flow["state"] == "filled" and flow["values"] == FLOW_LINES and flow["note"] == ""
    _, got, _ = tools_from_journal(journal + [rec])
    assert got == S1364_FLOW
    md, page = render_report(request="r", ct_name="ct.nii.gz", credit="c", results=run.results,
                             calls=run.calls, summary_text=None, guard_note="g", model="m",
                             flow=S1364_FLOW)
    assert twin_card_html(run.results, S1364_FLOW, run.calls) in page
    for line in FLOW_LINES:
        assert line in page
    # an attached record between the two index runs splits no batch
    i = [r["step"] for r in journal].index("tissue_index") + 1
    assert twin_from_journal(journal[:i] + [rec] + journal[i:])["slots"] == rep["slots"]


def test_attached_flow_survives_a_recrop_and_a_new_scan_clears_it(tmp_path):
    _, journal = _journaled_run(tmp_path)
    steps = [r["step"] for r in journal]
    rec = _flow_record()
    # a property of the scan's lumen, not of the crop: a re-crop keeps it
    recrop = twin_from_journal(journal + [rec, journal[steps.index("crop")]])
    assert _slot(recrop, "flow")["values"] == FLOW_LINES
    assert _slot(recrop, "stress")["state"] == "pending"  # the stale stress still goes
    # so do a re-run wall and a session CFD run (no AAA metrics in its summary)
    again = journal + [rec, journal[steps.index("volmesh")], {"step": "cfd", "params": {}, "summary": {}}]
    assert _slot(twin_from_journal(again), "flow")["values"] == FLOW_LINES
    # a new segment (or any new source) empties the card, flow included
    for new in (journal[steps.index("segment")], {"step": "generate_synth", "params": {}, "summary": {}}):
        card = twin_from_journal(journal + [rec, new])
        assert _slot(card, "flow")["state"] == "pending" and _slot(card, "flow")["note"] == "not run here"
        assert not card["lit"]
    # a hand-edited record that the session would refuse changes nothing, and never raises
    bad = {"step": "flow_metrics", "params": {"metrics": {"q_lmin": "1"}}, "summary": {}}
    assert _slot(twin_from_journal(journal + [bad]), "flow")["note"] == "not run here"
    assert _slot(twin_from_journal(journal + [rec, bad]), "flow")["values"] == FLOW_LINES


def test_copilot_cli_journals_the_flow_metrics_before_the_report(tmp_path, monkeypatch, capsys):
    import meshforge.copilot as cp
    from meshforge.cli import main
    from meshforge.copilot.backend import SessionBackend

    made = []

    class RecordingBackend(SessionBackend):
        def __init__(self):
            super().__init__()
            made.append(self)

    def fake_run(request, *, tools, model, effort, max_turns, log):
        return ag.CopilotRun(request=request, model=model, stop="done", summary_text=None,
                             guard_note="g", calls=[], results={}, transcript=[], usage={})

    real_render = cp.render_report

    def render_after_attach(**kw):
        assert made[0].session.journal[-1]["step"] == "flow_metrics"  # journaled first
        return real_render(**kw)

    monkeypatch.setattr(cp, "run_copilot", fake_run)
    monkeypatch.setattr(cp, "SessionBackend", RecordingBackend)
    monkeypatch.setattr(cp, "render_report", render_after_attach)
    monkeypatch.chdir(tmp_path)
    fm = tmp_path / "s1364_flow_metrics.json"
    fm.write_text(json.dumps(S1364_FLOW, indent=1))
    rc = main(["copilot", "Assess this AAA.", "--ct", "ct.nii.gz", "--flow-metrics", str(fm),
               "--out-dir", "run"])
    printed = capsys.readouterr().out
    assert rc == 1  # the fake run never finishes; the flow attach itself is fine
    assert "flow        : 5 pre-computed CFD metrics journaled in the session" in printed
    saved = json.loads((tmp_path / "run" / "session.mfwork").read_text())["journal"]
    assert saved[-1]["step"] == "flow_metrics" and saved[-1]["params"]["metrics"] == S1364_FLOW
    page = (tmp_path / "run" / "report.html").read_text()
    panel = _slot(twin_from_journal(saved), "flow")
    assert panel == _slot(twin_card({}, S1364_FLOW), "flow") and panel["values"] == FLOW_LINES
    for line in FLOW_LINES:
        assert line in page


def test_copilot_cli_refuses_a_bad_flow_metrics_file_before_any_call(tmp_path, monkeypatch, capsys):
    import meshforge.copilot as cp
    from meshforge.cli import main

    def never(*a, **kw):
        raise AssertionError("the copilot ran on a bad --flow-metrics file")

    monkeypatch.setattr(cp, "run_copilot", never)
    bad = tmp_path / "flow.json"
    for content, why in ((json.dumps({"q_lmin": "1.0"}), "q_lmin must be a number"),
                         (json.dumps({"sac_backflow_volume_fraction": 14.45}), "fraction in [0, 1]"),
                         ("{not json", "Expecting property name")):
        bad.write_text(content)
        rc = main(["copilot", "r", "--flow-metrics", str(bad), "--out-dir", str(tmp_path / "run")])
        err = capsys.readouterr().err
        assert rc == 1 and f"error: --flow-metrics flow.json: " in err and why in err
    rc = main(["copilot", "r", "--flow-metrics", str(tmp_path / "gone.json"),
               "--out-dir", str(tmp_path / "run")])
    err = capsys.readouterr().err
    assert rc == 1 and "gone.json: No such file or directory" in err and str(tmp_path) not in err
    assert not (tmp_path / "run").exists()  # no run folder for a refused start


class _FlowHandler(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):  # noqa: N802 - http.server API
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append((self.path, body))
        if self.path == "/twin/flow" and isinstance(body.get("metrics", {}).get("q_lmin"), float):
            out, code = {"step": "flow_metrics", "summary": {"metrics": body["metrics"]},
                         "text": "flow metrics: attached"}, 200
        else:
            out, code = {"detail": "flow metrics: q_lmin must be a number, got str"}, 400
        data = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_http_backend_attaches_flow_metrics_over_the_twin_route():
    srv = HTTPServer(("127.0.0.1", 0), _FlowHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        b = HttpBackend(f"http://127.0.0.1:{srv.server_port}")
        ok = b.attach_flow_metrics(S1364_FLOW)
        assert ok.ok and ok.summary == {"metrics": S1364_FLOW}
        assert _FlowHandler.seen[-1] == ("/twin/flow", {"metrics": S1364_FLOW})
        bad = b.attach_flow_metrics({"q_lmin": "1"})
        assert not bad.ok and "HTTP 400" in bad.error and "must be a number" in bad.error
    finally:
        srv.shutdown()


def test_session_backend_attach_is_loud_on_a_bad_set():
    from meshforge.copilot.backend import SessionBackend

    b = SessionBackend()
    out = b.attach_flow_metrics({"umax_mm_s": 12.0})
    assert not out.ok and "none of the twin card's metrics" in out.error and b.session.journal == []
    assert b.attach_flow_metrics(S1364_FLOW).ok and b.session.journal[-1]["step"] == "flow_metrics"


# --- D112: one command runs the demo; the plain report; the live flow case ------


def test_summary_prompt_asks_for_three_plain_paragraphs():
    for label in ("What the scan shows:", "What lowering the pressure does:", "How much to trust this:"):
        assert label in ag.SYSTEM
    assert "no tool names" in ag.SYSTEM and "P10 or P90" in ag.SYSTEM
    assert "Do not estimate a rupture probability or a time to rupture" in ag.SYSTEM


def test_report_opens_with_the_plain_words_then_the_steps_in_plain_names(tmp_path):
    run, _, _ = _run(tmp_path, _script(
        "What the scan shows: The sac is 61.8 mm at its widest; the peak wall stress is 278.2 kPa at 140 mmHg.\n\n"
        "What lowering the pressure does: At 120 mmHg it is 238.5 kPa, a drop of 14.3%.\n\n"
        "How much to trust this: The investigational index is 0.43 (band 0.21 to 0.9), from 7 aneurysms "
        "and 13 strips; not a diagnosis."))
    md, page = render_report(request="r", ct_name="ct.nii.gz", credit="CC BY 4.0", results=run.results,
                             calls=run.calls, summary_text=run.summary_text, guard_note=run.guard_note,
                             model="m", flow_note="the scan's blood-lumen case re-solved live: OpenFOAM simpleFoam, 34323 cells, 15 s")
    # reading order: plain words, headline numbers, twin, steps, then the fold
    order = [page.index(s) for s in ('id="summary"', 'id="headline"', '<section class="twin',
                                     'id="steps"', '<details class="more"', 'id="results"', 'id="pipeline"')]
    assert order == sorted(order)
    assert "<p><b>What the scan shows:</b> The sac is 61.8 mm" in page   # the label in bold
    assert "<p><b>How much to trust this:</b>" in page
    steps = page[page.index('id="steps"'):page.index("<details")]
    for name in ("Find the aorta on the scan", "Cut out the aneurysm", "Build the wall",
                 "Compute the wall stress", "Compare with wall strength", "Blood flow (CFD)"):
        assert name in steps, name
    assert steps.count('<li class="fail">') == 1 and "gate FAIL — the quality check stopped the run here" in steps
    assert "parallel cuts" in steps and "7552 blocks · 0 inverted" in steps
    assert "index 0.43 (band 0.21 to 0.9) at 140 mmHg" in steps and "open data: 7 aneurysms, 13 strips" in steps
    assert "OpenFOAM simpleFoam, 34323 cells, 15 s" in steps
    assert "- Blood flow: the scan's blood-lumen case re-solved live" in md
    # every number between the twin card and the fold traces to a tool (no seconds, no tolerances)
    shown = re.sub(r"<svg.*?</svg>", " ", page.split('<section class="twin')[1].split("<details")[0], flags=re.S)
    shown = re.sub(r"<[^>]+>", " ", shown)
    sources = [c["payload"] for c in run.calls if c["payload"]] + [{"cells": 34323, "s": 15}]
    assert check_numbers(shown, sources, allowed_text="aorta: module 1") == []
    assert "our lab will narrow" in page and "narrows" not in page


def test_launch_helpers_start_reset_open_and_stage(tmp_path, monkeypatch):
    from meshforge.copilot import launch

    # port_of: loud without a port or off the loopback
    assert launch.port_of("http://127.0.0.1:8765") == 8765
    with pytest.raises(ValueError):
        launch.port_of("http://127.0.0.1")
    with pytest.raises(ValueError):
        launch.port_of("http://10.0.0.5:8765")

    # start_workbench: spawns `python -m meshforge.web.server --port N` detached, waits for /health
    spawned = []

    class FakeProc:
        pid = 4242

        def __init__(self, argv, **kw):
            spawned.append((argv, kw))

        def poll(self):
            return None

        def terminate(self):
            spawned.append("terminated")

    answers = iter([False, False, True])
    monkeypatch.setattr(launch, "health", lambda url, timeout=1.5: next(answers))
    proc, secs = launch.start_workbench("http://127.0.0.1:8777", tmp_path / "wb.log",
                                        popen=FakeProc, sleep=lambda s: None)
    argv, kw = spawned[0]
    assert argv[1:] == ["-m", "meshforge.web.server", "--port", "8777"]
    assert kw["start_new_session"] is True and kw["stdin"] is not None
    assert proc.pid == 4242 and secs >= 0

    # a Workbench that never answers is stopped and reported with its log
    (tmp_path / "wb2.log").write_text("Address already in use\n")
    monkeypatch.setattr(launch, "health", lambda url, timeout=1.5: False)
    spawned.clear()
    with pytest.raises(RuntimeError) as exc:
        launch.start_workbench("http://127.0.0.1:8777", tmp_path / "wb2.log", wait_s=0.01,
                               popen=FakeProc, sleep=lambda s: None)
    assert "did not answer" in str(exc.value) and "Address already in use" in str(exc.value)
    assert "terminated" in spawned

    # open_window / open_file: a Chrome app window on a Mac with Chrome, else the default browser
    ran, opened = [], []
    monkeypatch.setattr(launch, "chrome_available", lambda: True)
    note = launch.open_window("http://127.0.0.1:8777", run=lambda argv, **kw: ran.append(argv))
    assert note == "Chrome app window"
    assert ran[0][:4] == ["open", "-na", "Google Chrome", "--args"]
    assert ran[0][4] == "--app=http://127.0.0.1:8777/app/"
    monkeypatch.setattr(launch, "chrome_available", lambda: False)
    note = launch.open_file(tmp_path / "report.html", browser=NS(open=lambda u: opened.append(u)))
    assert note == "default browser" and opened[0].startswith("file://") and opened[0].endswith("/report.html")

    # stage_case: a copy without earlier results or logs; loud on a folder that is no case
    case = tmp_path / "case"
    for d in ("0", "295", "constant/polyMesh", "system"):
        (case / d).mkdir(parents=True)
    (case / "system" / "controlDict").write_text("application simpleFoam;")
    (case / "0" / "U").write_text("u")
    (case / "295" / "U").write_text("old")
    (case / "log.simpleFoam").write_text("old log")
    (case / "case.foam").write_text("")
    staged = launch.stage_case(case, tmp_path / "work" / "flow_case")
    assert (staged / "0" / "U").is_file() and (staged / "system" / "controlDict").is_file()
    assert not (staged / "295").exists() and not (staged / "log.simpleFoam").exists()
    assert (case / "295" / "U").read_text() == "old"  # the source is untouched
    with pytest.raises(ValueError):
        launch.stage_case(tmp_path / "work", tmp_path / "x")  # no controlDict


def test_copilot_cli_runs_the_flow_case_then_attaches_the_metrics(tmp_path, monkeypatch, capsys):
    """--flow-case: after a complete run the session's own cfd step solves a
    fresh copy of the case (journaled `cfd`), THEN the metrics are attached,
    so the twin reads both; the report names it; a failed solve is loud and
    fails the exit code; nothing opens without a terminal."""
    import meshforge.copilot as cp
    from meshforge.cli import main
    from meshforge.copilot.backend import SessionBackend

    made = []

    class RecordingBackend(SessionBackend):
        def __init__(self):
            super().__init__()
            made.append(self)
            self.ran = []

        def run(self, step, knobs):
            self.ran.append((step, dict(knobs)))
            if step == "cfd":
                case = Path(knobs["case"])
                assert case.is_dir() and not (case / "295").exists()  # the staged copy, results dropped
                self.session.journal.append({"step": "cfd", "params": knobs, "summary": {}})
                return StepOutcome(True, {"solver": "simpleFoam", "n_cells": 34323}, "")
            return super().run(step, knobs)

    run_obj = ag.CopilotRun(request="r", model="m", stop="done", summary_text=None, guard_note="g",
                            calls=[], results={"strength_index": {"n_aneurysms": 7}}, transcript=[], usage={})
    monkeypatch.setattr(cp, "run_copilot", lambda request, **kw: run_obj)
    monkeypatch.setattr(cp, "SessionBackend", RecordingBackend)
    monkeypatch.chdir(tmp_path)
    case = tmp_path / "flow_case"
    for d in ("0", "295", "constant/polyMesh", "system"):
        (case / d).mkdir(parents=True)
    (case / "system" / "controlDict").write_text("application simpleFoam;")
    fm = tmp_path / "flow_metrics.json"
    fm.write_text(json.dumps(S1364_FLOW))
    rc = main(["copilot", "Assess this AAA.", "--ct", "ct.nii.gz", "--flow-metrics", str(fm),
               "--flow-case", str(case), "--out-dir", "run"])
    printed = capsys.readouterr().out
    assert rc == 0
    assert made[0].ran[0][0] == "cfd" and made[0].session.journal[-1]["step"] == "flow_metrics"
    assert "flow        : the scan's pre-built lumen case re-solved live: OpenFOAM simpleFoam, 34323 cells" in printed
    assert "flow        : 5 pre-computed CFD metrics journaled in the session" in printed
    assert "opened      :" not in printed  # stdout is no terminal here
    page = (tmp_path / "run" / "report.html").read_text()
    assert "Blood flow (CFD)" in page and "re-solved live" in page
    assert "- Blood flow: the scan's pre-built lumen case re-solved live" in (tmp_path / "run" / "report.md").read_text()

    # a folder that is no case is refused before any API call
    rc = main(["copilot", "Assess this AAA.", "--ct", "ct.nii.gz", "--flow-case", str(tmp_path / "nope"),
               "--out-dir", "run2"])
    err = capsys.readouterr().err
    assert rc == 1 and "not an OpenFOAM case" in err


def test_copilot_cli_starts_the_workbench_when_none_answers(tmp_path, monkeypatch, capsys):
    """--server with nothing listening: the CLI starts a Workbench on that
    port, opens its window in Demo mode, and says so; --reset empties one
    that already answers; --no-launch leaves the old loud error."""
    import meshforge.copilot as cp
    from meshforge.cli import main
    from meshforge.copilot import launch

    events = []

    class FakeHttpBackend:
        def __init__(self, url):
            self.name = f"Workbench at {url}"

    run_obj = ag.CopilotRun(request="r", model="m", stop="done", summary_text=None, guard_note="g",
                            calls=[], results={}, transcript=[], usage={})
    monkeypatch.setattr(cp, "run_copilot", lambda request, **kw: run_obj)
    monkeypatch.setattr(cp, "HttpBackend", FakeHttpBackend)
    monkeypatch.setattr(launch, "health", lambda url, timeout=1.5: events.append("health") and False)
    monkeypatch.setattr(launch, "start_workbench", lambda url, log: (events.append(("start", url, Path(log).name)), (NS(pid=77), 1.25))[1])
    monkeypatch.setattr(launch, "open_window", lambda url: (events.append(("window", url)), "Chrome app window")[1])
    monkeypatch.setattr(launch, "reset_workbench", lambda url: events.append(("reset", url)))
    monkeypatch.chdir(tmp_path)
    rc = main(["copilot", "Assess this AAA.", "--ct", "ct.nii.gz", "--server", "http://127.0.0.1:8799",
               "--out-dir", "run"])
    out = capsys.readouterr().out
    assert rc == 1  # the fake run never finishes
    assert ("start", "http://127.0.0.1:8799", "workbench.log") in events and ("window", "http://127.0.0.1:8799") in events
    assert "workbench   : started at http://127.0.0.1:8799 in 1.2 s (pid 77, log run/workbench.log); window opened in the Chrome app window" in out
    assert "workbench   : still running at http://127.0.0.1:8799 (pid 77; stop with: kill 77)" in out

    # already up + --reset: emptied, no start, no window
    events.clear()
    monkeypatch.setattr(launch, "health", lambda url, timeout=1.5: True)
    main(["copilot", "Assess this AAA.", "--ct", "ct.nii.gz", "--server", "http://127.0.0.1:8799",
          "--reset", "--out-dir", "run2"])
    out = capsys.readouterr().out
    assert events == [("reset", "http://127.0.0.1:8799")]
    assert "workbench   : running at http://127.0.0.1:8799; session emptied for this run" in out

    # --no-launch with nothing listening: no start, no window
    events.clear()
    monkeypatch.setattr(launch, "health", lambda url, timeout=1.5: False)
    main(["copilot", "Assess this AAA.", "--ct", "ct.nii.gz", "--server", "http://127.0.0.1:8799",
          "--no-launch", "--out-dir", "run3"])
    assert not any(isinstance(e, tuple) and e[0] in ("start", "window") for e in events)


def test_a_missing_key_is_a_readable_error_not_a_traceback(tmp_path):
    """The SDK raises a bare TypeError at the first call when no key is set;
    the copilot re-words it (the CLI prints `error: …` and the pid of the
    Workbench it started)."""
    class NoKeyClient:
        def __init__(self):
            self.beta = NS(messages=NS(create=self._create))

        def _create(self, **kw):
            raise TypeError('"Could not resolve authentication method. Expected one of api_key, '
                            'auth_token, or credentials to be set."')

    tools = PipelineTools(FakeBackend(), tmp_path / "work")
    with pytest.raises(RuntimeError) as exc:
        ag.run_copilot("Assess this AAA.", tools=tools, client=NoKeyClient(), log=lambda m: None)
    assert "ANTHROPIC_API_KEY" in str(exc.value) and "read -s" in str(exc.value)

    class OtherError(Exception):
        pass

    class BrokenClient(NoKeyClient):
        def _create(self, **kw):
            raise OtherError("boom")

    with pytest.raises(OtherError):  # anything else still surfaces as itself
        ag.run_copilot("Assess this AAA.", tools=tools, client=BrokenClient(), log=lambda m: None)


def test_live_flow_case_is_built_from_the_crop_lumen_with_the_neck_as_inflow(tmp_path):
    """D112 addendum: build_lumen_case fills the cropped lumen with the
    session's own lumen step, takes the higher cap as the neck (inflow at
    1 L/min over its area, blood viscosity), the other as the pressure
    outlet, and writes a complete simpleFoam case; the source is untouched."""
    pytest.importorskip("gmsh")
    import pyvista as pv

    from meshforge.copilot import flow
    from meshforge.pipeline import MeshingSession

    helper = MeshingSession()
    helper.generate_synth("straight_tube")
    src = tmp_path / "crop_lumen.vtp"
    helper.surface.save(src)
    before = src.read_bytes()
    built = flow.build_lumen_case(src, tmp_path / "case")
    case = Path(built["case"])
    for must in ("system/controlDict", "constant/polyMesh/points", "0/U", "0/p", "constant/transportProperties"):
        assert (case / must).is_file(), must
    assert built["n_cells"] > 0 and built["q_lmin"] == 1.0 and built["nu_mm2_s"] == 3.3
    assert built["inflow_patch"] != built["outlet_patch"]
    # the inflow cap is the one higher on z, and the speed is Q over its area
    vol = pv.read(case / "constant/polyMesh/points") if False else None  # the points file is OpenFOAM text
    assert abs(built["inflow_speed_mm_s"] * built["inflow_area_mm2"] - 1e6 / 60) < 0.02 * 1e6 / 60
    u = (case / "0" / "U").read_text()
    assert built["inflow_patch"] in u and "fixedValue" in u
    p_file = (case / "0" / "p").read_text()
    assert built["outlet_patch"] in p_file
    assert src.read_bytes() == before
    assert "nu" in (case / "constant" / "transportProperties").read_text()
    # building again replaces the case
    built2 = flow.build_lumen_case(src, tmp_path / "case")
    assert built2["inflow_patch"] == built["inflow_patch"]
    with pytest.raises(FileNotFoundError):
        flow.build_lumen_case(tmp_path / "missing.vtp", tmp_path / "case2")


def test_copilot_cli_solves_the_live_flow_first_and_falls_back_to_the_case(tmp_path, monkeypatch, capsys):
    import meshforge.copilot as cp
    from meshforge.cli import main
    from meshforge.copilot import flow as liveflow
    from meshforge.copilot.backend import SessionBackend

    made = []

    class RecordingBackend(SessionBackend):
        def __init__(self):
            super().__init__()
            made.append(self)
            self.cases = []

        def run(self, step, knobs):
            if step == "cfd":
                self.cases.append(Path(knobs["case"]).name)
                if Path(knobs["case"]).name == "flow_case_live" and FAIL_LIVE:
                    return StepOutcome(False, {}, "", "simpleFoam diverged")
                self.session.journal.append({"step": "cfd", "params": knobs, "summary": {}})
                return StepOutcome(True, {"solver": "simpleFoam", "n_cells": 1234}, "")
            return super().run(step, knobs)

    def fake_build(crop_lumen, case_dir, **kw):
        Path(case_dir).mkdir(parents=True, exist_ok=True)
        return {"case": str(case_dir), "n_cells": 1234, "fill": "tet", "inflow_patch": "outlet",
                "inflow_area_mm2": 372.1, "inflow_speed_mm_s": 44.79, "outlet_patch": "inlet",
                "q_lmin": 1.0, "nu_mm2_s": 3.3, "notes": []}

    run_obj = ag.CopilotRun(request="r", model="m", stop="done", summary_text=None, guard_note="g",
                            calls=[], results={"strength_index": {"n_aneurysms": 7, "n_strips": 13}},
                            transcript=[], usage={})
    monkeypatch.setattr(cp, "run_copilot", lambda request, **kw: run_obj)
    monkeypatch.setattr(cp, "SessionBackend", RecordingBackend)
    monkeypatch.setattr(liveflow, "build_lumen_case", fake_build)
    monkeypatch.chdir(tmp_path)
    case = tmp_path / "flow_case"
    for d in ("0", "constant/polyMesh", "system"):
        (case / d).mkdir(parents=True)
    (case / "system" / "controlDict").write_text("application simpleFoam;")
    fm = tmp_path / "flow_metrics.json"
    fm.write_text(json.dumps(S1364_FLOW))

    FAIL_LIVE = False
    rc = main(["copilot", "Assess this AAA.", "--ct", "ct.nii.gz", "--flow-metrics", str(fm),
               "--flow-live", "--flow-case", str(case), "--out-dir", "run"])
    out = capsys.readouterr().out
    assert rc == 0 and made[-1].cases == ["flow_case_live"]  # the live case only
    assert "flow        : this crop's blood channel filled: 1234 cells, inflow at the neck 44.79 mm/s for 1 L/min" in out
    assert "flow        : this crop's blood channel solved live: OpenFOAM simpleFoam, 1234 cells" in out
    assert made[-1].session.journal[-1]["step"] == "flow_metrics"  # metrics after the cfd
    assert "blood channel solved live: OpenFOAM simpleFoam, 1234 cells" in (tmp_path / "run" / "report.html").read_text()

    FAIL_LIVE = True
    rc = main(["copilot", "Assess this AAA.", "--ct", "ct.nii.gz", "--flow-live", "--flow-case", str(case),
               "--out-dir", "run2"])
    cap = capsys.readouterr()
    assert rc == 0 and made[-1].cases == ["flow_case_live", "flow_case"]  # the fallback carried it
    assert "the live flow case did not run: simpleFoam diverged" in cap.err
    assert "the scan's pre-built lumen case re-solved live" in cap.out

    rc = main(["copilot", "Assess this AAA.", "--ct", "ct.nii.gz", "--flow-live", "--out-dir", "run3"])
    cap = capsys.readouterr()
    assert rc == 1 and "the live flow case did not run" in cap.err  # no fallback: loud, exit 1


def test_report_strips_claudes_own_bold_marks_from_the_plain_words():
    _, page = render_report(request="r", ct_name="ct.nii.gz", credit="c", results={}, calls=[],
                            summary_text="**What the scan shows:** The sac is wide.\n\n__How much to trust this:__ Little.",
                            guard_note="g", model="m")
    assert "<p><b>What the scan shows:</b> The sac is wide.</p>" in page
    assert "<p><b>How much to trust this:</b> Little.</p>" in page and "**" not in page
