#!/usr/bin/env python
"""The demo without an API key: the copilot's own path, played by a script.

    python scripts/demo_run.py <ct> [--server URL] [--out DIR]

Runs the demo's six steps on a Workbench through the copilot's own tools
(``meshforge.copilot.tools.PipelineTools`` over HTTP), in the order the
copilot runs them: Load scan, Crop aneurysm, Mesh wall, Wall stress, Strength
index, Blood flow. It starts with the standard settings and changes one only
when a step's own error names the change, so the wall's quality gate fails
and recovers here exactly as it does for the copilot. No language model is
called: there is no written summary, and every number printed is a tool's
output.

  <ct>       the CT volume (.nii.gz or .nrrd), for example the demo scan
  --server   the Workbench to run on (default http://127.0.0.1:8765). When
             nothing answers there, one is started on that port and its
             window opens, as ``meshforge copilot`` does; it keeps running
             afterwards so the results stay on screen.
  --out      the run folder (default outputs/demo_run/<timestamp>): the
             steps' shared files, the flow case and demo_run.json

Needs what the demo needs: meshforge with the meshing, image, copilot, web
and viewport extras, and TotalSegmentator, CalculiX and OpenFOAM where
meshforge can find them (see docs/DEMO_GUIDE.md). About ten minutes on a
laptop. Exit code 0 only when every step ran.

Research software: not a medical device, not for diagnosis or treatment.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

SYSTOLIC_MMHG = 140
WHATIF_MMHG = 120
#: how many times a failed wall gate may be answered by re-meshing
GATE_RETRIES = 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="demo_run.py",
        description="Play the demo's six steps on a Workbench with the copilot's own tools, "
        "without a language model or an API key. Research software, not a medical device.",
    )
    parser.add_argument("ct", type=Path, help="the CT volume (.nii.gz or .nrrd)")
    parser.add_argument("--server", default="http://127.0.0.1:8765",
                        help="the Workbench to run on; started there (with its window) when "
                        "nothing answers (default http://127.0.0.1:8765)")
    parser.add_argument("--out", type=Path, default=None,
                        help="the run folder (default outputs/demo_run/<timestamp>)")
    args = parser.parse_args(argv)

    ct = args.ct.expanduser().resolve()
    if not ct.is_file():
        print(f"error: no CT file at {args.ct}", file=sys.stderr)
        return 1
    out = (args.out or Path("outputs") / "demo_run" / time.strftime("%Y%m%d-%H%M%S"))
    out = out.expanduser().resolve()  # the Workbench writes here too: it needs the full path
    out.mkdir(parents=True, exist_ok=True)

    from meshforge.copilot import HttpBackend, launch
    from meshforge.copilot import flow as liveflow
    from meshforge.copilot.tools import PipelineTools

    workbench = None
    if launch.health(args.server):
        print(f"workbench   : running at {args.server}", flush=True)
    else:
        try:
            workbench, secs = launch.start_workbench(args.server, out / "workbench.log")
        except (RuntimeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        where = launch.open_window(args.server)
        print(f"workbench   : started at {args.server} in {secs:.1f} s (pid {workbench.pid}); "
              f"window opened in the {where}", flush=True)

    backend = HttpBackend(args.server)
    tools = PipelineTools(backend, out / "work")
    t0 = time.perf_counter()

    def call(name: str, tool_args: dict):
        res = tools.run(name, tool_args)
        shown = json.dumps(res.payload) if res.ok else res.error
        print(f"{name:15s} {'ok    ' if res.ok else 'FAILED'} {res.seconds:7.1f} s  "
              f"{json.dumps(tool_args)}\n                {shown}", flush=True)
        return res

    def finish(code: int, flow_case: dict | None = None, cfd_summary: dict | None = None) -> int:
        record = {"ct": ct.name, "server": args.server, "calls": tools.calls,
                  "flow_case": flow_case, "cfd": cfd_summary}
        (out / "demo_run.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
        print(f"total       : {time.perf_counter() - t0:.0f} s on the {backend.name}; "
              f"record in {out / 'demo_run.json'}")
        if workbench is not None:
            print(f"workbench   : still running at {args.server} (pid {workbench.pid}; "
                  f"stop with: kill {workbench.pid})")
        return code

    try:
        # 1 Load scan, 2 Crop aneurysm
        if not call("segment_aorta", {"ct_path": str(ct)}).ok:
            return finish(1)
        if not call("crop_aneurysm", {"parallel_caps": False}).ok:
            return finish(1)
        # 3 Mesh wall: follow what a failed gate's error names, nothing else
        wall = call("mesh_wall", {"sweep": "auto"})
        for _ in range(GATE_RETRIES):
            if wall.ok:
                break
            if "parallel_caps=true" in wall.error:
                if not call("crop_aneurysm", {"parallel_caps": True}).ok:
                    return finish(1)
            wall = call("mesh_wall", {"sweep": "planes" if "sweep 'planes'" in wall.error else "auto"})
        if not wall.ok:
            print("error: no wall passed the quality gate; no stress is reported", file=sys.stderr)
            return finish(1)
        # 4 Wall stress, 5 Strength index
        if not call("wall_stress", {"systolic_mmhg": SYSTOLIC_MMHG, "whatif_mmhg": WHATIF_MMHG}).ok:
            return finish(1)
        if not call("strength_index", {}).ok:
            return finish(1)
        # 6 Blood flow: a case built from THIS crop's own lumen, solved by the
        # Workbench's cfd step (the copilot's --flow-live)
        t1 = time.perf_counter()
        try:
            built = liveflow.build_lumen_case(tools.crop_lumen, out / "work" / "flow_case_live")
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"{'flow_case':15s} FAILED  the live flow case could not be built: {exc}", flush=True)
            return finish(1)
        print(f"{'flow_case':15s} ok     {time.perf_counter() - t1:7.1f} s  this crop's blood lumen: "
              f"{built['n_cells']} cells ({built['fill']} fill), inflow at the neck "
              f"{built['inflow_speed_mm_s']} mm/s for {built['q_lmin']:g} L/min", flush=True)
        t1 = time.perf_counter()
        cfd = backend.run("cfd", {"case": built["case"]})
        secs = time.perf_counter() - t1
        if not cfd.ok:
            print(f"{'blood_flow':15s} FAILED {secs:7.1f} s  {cfd.error}", flush=True)
            return finish(1, built)
        notes = "; ".join(cfd.summary.get("notes") or []) or "no solver note"
        print(f"{'blood_flow':15s} ok     {secs:7.1f} s  OpenFOAM {cfd.summary.get('solver')}, "
              f"{cfd.summary.get('n_cells')} cells; {notes}", flush=True)
        if not _converged(cfd.summary.get("notes") or []):
            print(f"{'blood_flow':15s} FAILED  simpleFoam did not converge: {notes}", flush=True)
            return finish(1, built, cfd.summary)
    except Exception as exc:  # the Workbench went away, or a reply could not be read
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return finish(1)

    print(_results(tools, built, cfd.summary), flush=True)
    return finish(0, built, cfd.summary)


def _converged(notes) -> list[str]:
    return [n for n in notes if re.search(r"\bconverged in \d+ iterations", n)]


def _results(tools, built: dict, cfd_summary: dict) -> str:
    """The run's headline numbers, each read from a tool output."""
    seg = tools.results["segment_aorta"]
    crop = tools.results["crop_aneurysm"]
    mesh = tools.results["mesh_wall"]
    stress = tools.results["wall_stress"]
    index = tools.results["strength_index"]
    failed = [c["payload"] for c in tools.calls if c["tool"] == "mesh_wall" and not c["ok"]]
    gate = "".join(f"FAIL at {p.get('wall_thickness_measured_mm')} mm, then " for p in failed)
    gate += (f"PASS at {mesh['wall_thickness_measured_mm']} mm ({mesh['hex_cells']} hexahedra, "
             f"{mesh['inverted_cells']} inverted, sweep {mesh['sweep_used']})")
    sys_key = f"systolic_{SYSTOLIC_MMHG}mmhg"
    wif_key = f"whatif_{WHATIF_MMHG}mmhg"

    def band(b: dict) -> str:
        return f"{b['index_p50']} ({b['index_p10']} to {b['index_p90']})"

    converged = _converged(cfd_summary.get("notes") or [])
    lines = [
        "results     : every number below is a tool output",
        f"  sac maximum diameter   {seg['sac_max_diameter_mm']} mm",
        f"  cropped sac / neck     {crop['sac_diameter_mm']} / {crop['neck_diameter_mm']} mm",
        f"  wall quality gate      {gate}",
        f"  peak wall stress       {stress['pws_systolic_kpa']} kPa at {SYSTOLIC_MMHG} mmHg, "
        f"{stress['pws_whatif_kpa']} kPa at {WHATIF_MMHG} mmHg"
        + (" (thrombus modelled)" if stress.get("thrombus_modelled") else ""),
        f"  strength index P50     {band(index[sys_key])} at {SYSTOLIC_MMHG} mmHg, "
        f"{band(index[wif_key])} at {WHATIF_MMHG} mmHg; {index['caveat']}",
        f"  blood flow             {built['n_cells']} cells, {cfd_summary.get('solver')}: "
        + (converged[0] if converged else "not converged, see the notes above"),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
