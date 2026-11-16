"""The copilot's five tools, and the executor that maps each onto pipeline steps.

Each tool returns a small JSON payload of the numbers that matter, rounded
the way the report prints them. Those payloads are the ONLY numbers the
copilot may quote (see :mod:`meshforge.copilot.guard`).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WALL_THICKNESS_MM = 1.5
#: the gate's limit on |measured normal thickness - nominal| / nominal (D93)
THICKNESS_TOLERANCE = 0.05


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


TOOLS: list[dict[str, Any]] = [
    _tool(
        "segment_aorta",
        "Outline the aorta on a CT scan with the AI model (TotalSegmentator), keep the "
        "contrast-filled blood lumen, then recover the outer aneurysm sac (lumen + "
        "intraluminal thrombus + wall) from the CT around it. Always the first step.",
        {"ct_path": {"type": "string", "description": "the CT file (.nii.gz or .nrrd)"}},
        ["ct_path"],
    ),
    _tool(
        "crop_aneurysm",
        "Cut the infrarenal aneurysm out of the sac with two flat caps; the neck, sac and "
        "bifurcation landmarks are found automatically and the blood lumen is cut with the "
        "same planes. parallel_caps=true squares both cuts to one common axis, which the "
        "wall mesher needs on a bent aorta.",
        {"parallel_caps": {"type": "boolean"}},
        ["parallel_caps"],
    ),
    _tool(
        "mesh_wall",
        "Build the 1.5 mm hexahedral wall on the cropped sac and run its quality gate "
        "(0 inverted cells, 0 capped and 0 floored wall rays, and the wall's thickness "
        "measured along the surface normal within 5% of 1.5 mm). sweep 'auto' picks the "
        "method; 'centreline' and 'planes' force one. A failed gate returns an error "
        "that says what to change.",
        {"sweep": {"type": "string", "enum": ["auto", "centreline", "planes"]}},
        ["sweep"],
    ),
    _tool(
        "wall_stress",
        "Wall stress under blood pressure (CalculiX finite elements, stiff-linear on the "
        "imaged geometry, intraluminal thrombus modelled) at a systolic and a what-if "
        "pressure. Returns the peak wall stress (99th percentile) at both. Needs a wall "
        "that passed the gate. The gate is checked again under the peak: if the peak sits on "
        "wall more than 5% off 1.5 mm thick along the surface normal, the call fails with an "
        "error that says what to change.",
        {
            "systolic_mmhg": {"type": "number", "description": "e.g. 140"},
            "whatif_mmhg": {"type": "number", "description": "e.g. 120"},
        },
        ["systolic_mmhg", "whatif_mmhg"],
    ),
    _tool(
        "strength_index",
        "Divide the peak wall stress by lab-measured aneurysm-wall failure strength (open "
        "tensile data) and return the investigational index band P10/P50/P90 at both "
        "pressures, with how many aneurysms and strips the strength comes from.",
        {},
        [],
    ),
]


@dataclass
class ToolResult:
    ok: bool
    payload: dict[str, Any]
    seconds: float
    steps: list[str] = field(default_factory=list)
    error: str = ""

    def content(self) -> str:
        """What Claude reads back: the payload as JSON, or the error."""
        return self.error if not self.ok else json.dumps(self.payload, indent=1)


def _g(d: Any, *path, default=None):
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


def _r(x, nd=1):
    return None if x is None else round(float(x), nd)


def _thickness_hint(sweep) -> str:
    """What the copilot can change when the wall is not 1.5 mm along the normal."""
    if sweep == "planes":
        return " No setting is known to fix a plane-swept wall."
    return (f" The {sweep} sweep's rays leave the sac's shoulders at an angle. Re-crop with "
            f"parallel_caps=true and mesh again with sweep 'planes'.")


def pws_wall_check(summary: dict, sweep) -> tuple[dict[str, Any], str]:
    """The gate's local half (D96), after the stress step. The median gate sees
    the whole wall, so a thin patch can pass it and still carry the peak (Zenodo
    P04: median -3.3 %, the peak on 0.80 mm columns). The wall under the PWS
    region (median of its top 1 %) must be within ``THICKNESS_TOLERANCE`` of
    nominal too, at both pressures. Returns (payload fields, error); an empty
    error clears the stress."""
    out: dict[str, Any] = {}
    for key in ("systolic", "whatif"):
        h = _g(summary, key, "peak", "top1pct_wall_thickness_mm") or {}
        med, low = h.get("median"), h.get("min")
        if med is None or low is None:
            return ({"sweep_used": sweep, "gate": "FAIL"},
                    f"quality gate FAILED: the wall's thickness under the peak wall stress "
                    f"({key}) was not reported, so a peak on a locally thin wall cannot be "
                    f"ruled out and the stress is not reportable.")
        off = float(med) / WALL_THICKNESS_MM - 1.0
        fields = {"pws_wall_thickness_mm": _r(med, 2),
                  "pws_wall_thickness_off_percent": _r(100.0 * off)}
        if abs(off) > THICKNESS_TOLERANCE:
            return ({**fields, "pws_wall_thickness_min_mm": _r(low, 2), "sweep_used": sweep,
                     "gate": "FAIL"},
                    f"quality gate FAILED under the peak: the peak wall stress sits on wall "
                    f"{med:.2f} mm thick along the surface normal (median of its top-1% region; "
                    f"thinnest {low:.2f} mm), {100 * off:+.0f}% off the {WALL_THICKNESS_MM:g} mm "
                    f"it should be (limit {100 * THICKNESS_TOLERANCE:.0f}%), so it would be "
                    f"misstated. The whole wall's median passed: this is a local patch."
                    + _thickness_hint(sweep))
        out = out or fields  # the systolic one is the PWS's
    return out, ""


class PipelineTools:
    """Executes the copilot's tools on a backend; keeps the run's state and a
    record of every call for the report."""

    def __init__(self, backend, workdir: str | Path):
        self.backend = backend
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.ct_path: str | None = None
        self.gate_passed = False
        self.pws: dict[str, float] = {}
        self.calls: list[dict[str, Any]] = []
        self.results: dict[str, dict[str, Any]] = {}

    # paths the steps share
    @property
    def lumen(self) -> str:
        return str(self.workdir / "lumen.vtp")

    @property
    def mask(self) -> str:
        return str(self.workdir / "aorta_mask.nii.gz")

    @property
    def crop_lumen(self) -> str:
        return str(self.workdir / "crop_lumen.vtp")

    def run(self, name: str, args: dict[str, Any]) -> ToolResult:
        t0 = time.perf_counter()
        handler = getattr(self, f"_t_{name}", None)
        if handler is None:
            res = ToolResult(False, {}, 0.0, error=f"unknown tool {name!r}")
        else:
            try:
                res = handler(**args)
            except TypeError as exc:
                res = ToolResult(False, {}, 0.0, error=f"bad arguments for {name}: {exc}")
        res.seconds = round(time.perf_counter() - t0, 1)
        if res.ok:
            self.results[name] = res.payload
        self.calls.append({"tool": name, "input": args, "ok": res.ok, "seconds": res.seconds,
                           "steps": res.steps, "payload": res.payload, "error": res.error})
        return res

    def _step(self, step: str, knobs: dict[str, Any], steps: list[str]):
        steps.append(step)
        return self.backend.run(step, knobs)

    def _t_segment_aorta(self, ct_path: str) -> ToolResult:
        steps: list[str] = []
        seg = self._step("segment", {"path": ct_path, "lumen_out": self.lumen,
                                     "mask_out": self.mask}, steps)
        if not seg.ok:
            return ToolResult(False, {}, 0, steps, f"segment failed: {seg.error}")
        self.ct_path = ct_path
        sac = self._step("sac", {"path": ct_path, "lumen_mask": self.mask}, steps)
        if not sac.ok:
            return ToolResult(False, {}, 0, steps, f"sac recovery failed: {sac.error}")
        # session steps nest their payload under the step name (the CLI JSON is flat)
        g = seg.summary.get("segment", seg.summary)
        s = sac.summary.get("sac", sac.summary)
        payload = {
            "ai_model": "TotalSegmentator (total task, aorta)",
            "ai_seconds": _r(_g(g, "totalseg", "wall_time_s")),
            "aorta_mask_ml": _r(_g(g, "mask_volume_ml")),
            "sac_max_diameter_mm": _r(_g(s, "max_diameter", "max_caliper", "value_mm")),
            "sac_area_equivalent_diameter_mm": _r(_g(s, "max_diameter", "area_equivalent", "value_mm")),
            "sac_volume_ml": _r(_g(s, "sac_volume_ml")),
            "thrombus_plus_wall_thickness_mm_median": _r(_g(s, "thickness_mm", "all_rays", "median")),
            "thrombus_plus_wall_thickness_mm_max": _r(_g(s, "thickness_mm", "all_rays", "max")),
            "sac_edge_measured_fraction": _r(_g(s, "resolved_fraction"), 2),
        }
        return ToolResult(True, payload, 0, steps)

    def _t_crop_aneurysm(self, parallel_caps: bool) -> ToolResult:
        steps: list[str] = []
        if self.ct_path is None:
            return ToolResult(False, {}, 0, steps, "run segment_aorta first")
        crop = self._step("crop", {"lumen": self.lumen, "lumen_out": self.crop_lumen,
                                   "parallel_caps": int(bool(parallel_caps))}, steps)
        if not crop.ok:
            return ToolResult(False, {}, 0, steps, f"crop failed: {crop.error}")
        self.gate_passed, self.pws = False, {}  # a new crop is a new wall: no gate, no stress yet
        lm = _g(crop.summary, "crop", "landmarks") or {}
        payload = {
            "sac_diameter_mm": _r(lm.get("sac_d_equiv_mm")),
            "neck_diameter_mm": _r(lm.get("neck_d_equiv_mm")),
            "sac_to_neck_ratio": _r(lm.get("sac_neck_ratio"), 2),
            "parallel_caps": bool(parallel_caps),
        }
        return ToolResult(True, payload, 0, steps)

    def _t_mesh_wall(self, sweep: str) -> ToolResult:
        steps: list[str] = []
        self.gate_passed, self.pws = False, {}  # a new wall: no gate, no stress yet
        vol = self._step("volmesh", {"thickness": WALL_THICKNESS_MM, "curvature_cap": 0.9,
                                     "sweep": sweep}, steps)
        if not vol.ok:
            return ToolResult(False, {}, 0, steps, f"wall mesh failed: {vol.error}")
        s = vol.summary
        inverted = int(_g(s, "quality", "n_inverted", default=_g(s, "n_inverted", default=-1)))
        capped = _g(s, "lattice", "n_capped")
        floored = _g(s, "lattice", "n_floored")
        measured = _g(s, "lattice", "thickness_normal_median")
        off = None if measured is None else measured / WALL_THICKNESS_MM - 1.0
        cells_ok = inverted == 0 and capped == 0 and floored == 0
        thickness_ok = off is not None and abs(off) <= THICKNESS_TOLERANCE
        passed = cells_ok and thickness_ok
        self.gate_passed = passed
        payload = {
            "hex_cells": _g(s, "n_cells"),
            "inverted_cells": inverted,
            "capped_rays": capped,
            "floored_rays": floored,
            "wall_thickness_measured_mm": _r(measured, 2),
            "wall_thickness_off_percent": _r(None if off is None else 100.0 * off),
            "sweep_used": _g(s, "lattice", "sweep"),
            "gate": "PASS" if passed else "FAIL",
        }
        if not cells_ok:
            return ToolResult(
                False, payload, 0, steps,
                f"quality gate FAILED (inverted {inverted}, capped {capped}, floored "
                f"{floored}; sweep {payload['sweep_used']}). A stress result on this wall "
                f"would not be reportable. If the caps are not parallel, re-crop with "
                f"parallel_caps=true and mesh again with sweep 'auto'.",
            )
        if measured is None:
            return ToolResult(
                False, payload, 0, steps,
                "quality gate FAILED: the wall's thickness along the surface normal could not "
                "be measured, so its stress would not be reportable.",
            )
        if not thickness_ok:
            what = (f"quality gate FAILED: the wall is {measured:.2f} mm thick along the surface "
                    f"normal, {100 * off:+.0f}% off the {WALL_THICKNESS_MM:g} mm it should be "
                    f"(limit {100 * THICKNESS_TOLERANCE:.0f}%), so its stress would be misstated.")
            return ToolResult(False, payload, 0, steps, what + _thickness_hint(payload["sweep_used"]))
        return ToolResult(True, payload, 0, steps)

    def _t_wall_stress(self, systolic_mmhg: float, whatif_mmhg: float) -> ToolResult:
        steps: list[str] = []
        if not self.gate_passed:
            return ToolResult(False, {}, 0, steps, "no wall has passed the quality gate yet")
        st = self._step("aaa_stress", {"systolic_mmhg": float(systolic_mmhg),
                                       "whatif_mmhg": float(whatif_mmhg),
                                       "lumen": self.crop_lumen}, steps)
        if not st.ok:
            return ToolResult(False, {}, 0, steps, f"wall stress failed: {st.error}")
        s = st.summary
        if s.get("refused"):
            return ToolResult(False, {}, 0, steps, "wall stress REFUSED: " + "; ".join(s["refused"]))
        local, err = pws_wall_check(s, _g(self.results, "mesh_wall", "sweep_used"))
        if err:  # the wall fails its gate after all: no PWS, mesh again
            self.gate_passed, self.pws = False, {}
            return ToolResult(False, local, 0, steps, err)
        sys_mpa = _g(s, "systolic", "von_mises_p99_mpa")
        wif_mpa = _g(s, "whatif", "von_mises_p99_mpa")
        self.pws = {"systolic": float(sys_mpa), "whatif": float(wif_mpa),
                    "systolic_mmhg": float(systolic_mmhg), "whatif_mmhg": float(whatif_mmhg)}
        payload = {
            "systolic_mmhg": _r(systolic_mmhg, 0),
            "whatif_mmhg": _r(whatif_mmhg, 0),
            "pws_systolic_kpa": _r(1000 * sys_mpa),
            "pws_whatif_kpa": _r(1000 * wif_mpa),
            "whatif_over_systolic": _r(_g(s, "whatif_ratio", "pws"), 3),
            "pws_change_percent": _r(100.0 * (wif_mpa / sys_mpa - 1.0)),
            "thrombus_modelled": bool(_g(s, "thrombus", "modelled")),
            "wall_thickness_mm": WALL_THICKNESS_MM,
            **local,
        }
        return ToolResult(True, payload, 0, steps)

    def _t_strength_index(self) -> ToolResult:
        steps: list[str] = []
        if not self.pws:
            return ToolResult(False, {}, 0, steps, "run wall_stress first")
        out: dict[str, Any] = {}
        for key in ("systolic", "whatif"):
            ti = self._step("tissue_index", {"pws_mpa": self.pws[key]}, steps)
            if not ti.ok:
                return ToolResult(False, {}, 0, steps, f"strength index failed: {ti.error}")
            s = ti.summary
            idx = _g(s, "index") or {}
            out[f"{key}_{int(self.pws[f'{key}_mmhg'])}mmhg"] = {
                "index_p10": _r(idx.get("p10"), 2), "index_p50": _r(idx.get("p50"), 2),
                "index_p90": _r(idx.get("p90"), 2),
            }
            out["strength_mpa_p10_p50_p90"] = [
                _r(_g(s, "strength_mpa", q), 2) for q in ("p10", "p50", "p90")]
            out["n_aneurysms"] = _g(s, "n_aneurysms")
            out["n_strips"] = _g(s, "n_strips")
        out["caveat"] = "investigational index, not a diagnosis"
        return ToolResult(True, out, 0, steps)
