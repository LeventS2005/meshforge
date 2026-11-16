"""The copilot's report: deterministic, built only from tool outputs.

It opens with the **twin card**, a personal twin that grows one module at a
time. The aorta is module 1, lit; heart and kidneys are dimmed "next". The
data layers are slots that are either filled (shape, strength, stress, flow,
blood-pressure what-if) or waiting (growth needs a second scan). The prose
summary Claude writes is included only if it passed the numbers guard.

The same card is the Workbench's twin panel (M6): :func:`twin_from_journal`
replays a session's journaled step summaries through the copilot's own tool
code, so the panel shows exactly the numbers the copilot would quote. The
flow slot reads the session's attached ``flow_metrics`` record (the
copilot's ``--flow-metrics``, journaled by the session), so the panel and
the report print the same flow lines too.
"""

from __future__ import annotations

import html
import os
import re
import tempfile
from pathlib import Path, PurePath
from typing import Any

from meshforge.copilot.backend import StepOutcome
from meshforge.copilot.tools import PipelineTools

CAVEATS = (
    "public CC-BY scan; no patient data",
    "investigational index, not a diagnosis and not a rupture prediction",
    "wall thickness assumed 1.5 mm",
)

#: stamped on every page this renderer writes; the escaped-markdown page of
#: M5 lacks it (both carry a twin card, so the card alone tells nothing)
REPORT_HTML_MARKER = "<!-- meshforge report.html v2 -->"

#: the copilot's tools in pipeline order; a tool that runs again makes every
#: later tool's result stale
TWIN_CHAIN = ("segment_aorta", "crop_aneurysm", "mesh_wall", "wall_stress", "strength_index")

#: The twin's body outline. Classes only (no colours), so the report and the
#: Workbench each theme it; the Workbench's index.html carries this exact
#: markup (a test keeps the two identical).
BODY_SVG = (
    '<svg class="tw-svg" viewBox="0 0 140 300" role="img" '
    'aria-label="body outline: the aorta is module 1; heart and kidneys come next">'
    '<circle class="tw-body" cx="70" cy="24" r="16"/>'
    '<path class="tw-body" d="M76 38 L78 48 Q96 50 106 58 L112 96 L117 150 Q118 160 112 160 '
    'L109 152 L104 104 L100 74 L99 112 Q96 136 100 160 L97 226 L95 284 Q95 292 86 292 L80 292 '
    'L79 230 L73 170 Q70 166 67 170 L61 230 L60 292 L54 292 Q45 292 45 284 L43 226 L40 160 '
    'Q44 136 41 112 L40 74 L36 104 L31 152 L28 160 Q22 160 23 150 L28 96 L34 58 Q44 50 62 48 '
    'L64 38 Z"/>'
    '<path class="tw-next" d="M81 86 C77 79 66 81 66 89 C66 97 75 102 81 108 C87 102 96 97 96 89 '
    'C96 81 85 79 81 86 Z"/>'
    '<ellipse class="tw-next" cx="55" cy="126" rx="5.5" ry="9" transform="rotate(-12 55 126)"/>'
    '<ellipse class="tw-next" cx="88" cy="123" rx="5.5" ry="9" transform="rotate(12 88 123)"/>'
    '<g class="tw-aorta">'
    '<path d="M71 94 C69 78 71 66 77 66 C82 66 81 74 79 82 C77 92 74 104 74 132 L74 150"/>'
    '<path d="M74 150 L66 172 M74 150 L82 172"/>'
    '<ellipse cx="74" cy="140" rx="5.5" ry="9"/>'
    "</g></svg>"
)


# --- the twin card ------------------------------------------------------------


def _n(v: Any) -> str:
    """A tool number as the card prints it (140.0 -> 140, None -> —)."""
    if v is None:
        return "—"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _band(idx: dict | None, prefix: str) -> dict | None:
    if not idx:
        return None
    return next((v for k, v in idx.items() if k.startswith(prefix) and isinstance(v, dict)), None)


def _last_failed(calls: list[dict] | None, tool: str) -> dict | None:
    """The payload of ``tool``'s last call if that call failed (a failed gate
    still measured the wall), else None."""
    for c in reversed(calls or []):
        if c.get("tool") == tool:
            return None if c.get("ok") else (c.get("payload") or None)
    return None


def twin_card(results: dict[str, dict], flow: dict | None = None,
              calls: list[dict] | None = None) -> dict[str, Any]:
    """The twin card as data: modules and data-layer slots, each slot
    ``filled`` / ``pending`` / ``next``, with value lines taken only from the
    tool payloads (and the flow metrics, when given)."""
    seg, wall = results.get("segment_aorta"), results.get("mesh_wall")
    st, idx = results.get("wall_stress"), results.get("strength_index")
    sys_band, wif_band = _band(idx, "systolic_"), _band(idx, "whatif_")

    def slot(key, label, detail, filled, values=(), note=""):
        return {"key": key, "label": label, "detail": detail,
                "state": "filled" if filled else "pending",
                "values": [v for v in values if v] if filled else [],
                "note": note if filled else (note or "not run yet")}

    shape = slot("shape", "Shape", "CT → AI outline (TotalSegmentator)", bool(seg), [
        seg and seg.get("sac_max_diameter_mm") is not None
        and f"sac {_n(seg['sac_max_diameter_mm'])} mm at its widest",
        seg and seg.get("sac_volume_ml") is not None and f"sac volume {_n(seg['sac_volume_ml'])} mL",
    ])
    failed_wall = None if wall else _last_failed(calls, "mesh_wall")
    wall_note = ""
    if failed_wall and failed_wall.get("wall_thickness_measured_mm") is not None:
        wall_note = (f"last wall gate FAIL: {_n(failed_wall['wall_thickness_measured_mm'])} mm "
                     f"along the normal")
    walls = slot("wall", "Wall + thrombus", "recovered from the CT", bool(seg), [
        seg and seg.get("thrombus_plus_wall_thickness_mm_max") is not None
        and f"thrombus + wall up to {_n(seg['thrombus_plus_wall_thickness_mm_max'])} mm",
        wall and f"wall {_n(wall.get('wall_thickness_measured_mm'))} mm · gate {wall.get('gate')}",
    ], wall_note)
    stress = slot("stress", "Stress", "finite elements under blood pressure", bool(st), [
        st and f"{_n(st.get('pws_systolic_kpa'))} kPa peak at {_n(st.get('systolic_mmhg'))} mmHg",
    ], "thrombus modelled" if st and st.get("thrombus_modelled") else "")
    n = (f"open data, {_n(idx.get('n_aneurysms'))} aneurysms / {_n(idx.get('n_strips'))} strips"
         if idx and idx.get("n_aneurysms") is not None else "")
    strength = slot("strength", "Strength", "lab v0 — " + n if n else "lab v0", bool(idx), [
        sys_band and (f"index P50 {_n(sys_band['index_p50'])} (P10 {_n(sys_band['index_p10'])} – "
                      f"P90 {_n(sys_band['index_p90'])})"
                      + (f" at {_n(st['systolic_mmhg'])} mmHg" if st else "")),
    ], "wide band: our lab will narrow it")
    bf = (flow or {}).get("sac_backflow_volume_fraction")
    q = (flow or {}).get("q_lmin")
    flows = slot("flow", "Blood flow", "CFD (steady, resting flow)", flow is not None, [
        bf is not None and f"{100 * float(bf):.0f}% of the sac volume flows backward",
        q is not None and f"inflow {_n(q)} L/min",
    ], "" if flow is not None else "not run here")
    whatif = slot("whatif", "BP what-if", "lower pressure → lower stress", bool(st), [
        st and (f"{_n(st.get('pws_whatif_kpa'))} kPa at {_n(st.get('whatif_mmhg'))} mmHg "
                f"({_n(st.get('pws_change_percent'))} %)"),
        wif_band and f"index P50 {_n(wif_band['index_p50'])}",
    ])
    growth = {"key": "growth", "label": "Growth", "detail": "needs a second scan",
              "state": "next", "values": [], "note": "next"}
    slots = [shape, walls, stress, strength, flows, whatif, growth]
    lit = any(s["state"] == "filled" for s in slots)
    return {
        "title": "Your twin · v0",
        "lit": lit,
        "modules": [{"name": "aorta", "state": "module 1", "lit": lit},
                    {"name": "heart", "state": "next", "lit": False},
                    {"name": "kidneys", "state": "next", "lit": False}],
        "slots": slots,
    }


def twin_card_html(results: dict[str, dict], flow: dict | None = None,
                   calls: list[dict] | None = None) -> str:
    card = twin_card(results, flow, calls)
    e = html.escape
    rows = []
    for s in card["slots"]:
        on = s["state"] == "filled"
        vals = "".join(f'<span class="val">{e(v)}</span>' for v in s["values"])
        rows.append(
            f'<li class="{"on" if on else "off"}"><span class="dot">{"●" if on else "○"}</span>'
            f'<div class="lab"><b>{e(s["label"])}</b><span class="det">{e(s["detail"])}</span>'
            f'{vals}<span class="note">{e(s["note"])}</span></div></li>'
        )
    mods = "".join(
        f'<li class="{"lit" if m["lit"] else "dim"}">{"●" if m["lit"] else "○"} '
        f'{e(m["name"])}: {e(m["state"])}</li>' for m in card["modules"]
    )
    return (
        f'<section class="twin{" is-lit" if card["lit"] else ""}">'
        f'<div class="twin-fig">{BODY_SVG}<ul class="mods">{mods}</ul></div>'
        f'<div class="twin-main"><h2>{e(card["title"])}</h2>'
        f'<p class="twin-sub">a personal twin that grows one module at a time</p>'
        f'<ul class="slots">{"".join(rows)}</ul></div></section>'
    )


# --- the twin card from a Workbench session -------------------------------------


class _JournalBackend:
    """Serves journaled step summaries back to the copilot's tool code: each
    tool call is handed the records it may use, and takes them by step name
    (and, for the strength index, by the stress it was run with)."""

    name = "session journal"

    def __init__(self) -> None:
        self.pending: list[dict] = []

    def run(self, step: str, knobs: dict[str, Any]) -> StepOutcome:
        want = knobs.get("pws_mpa")
        for i, rec in enumerate(self.pending):
            if rec.get("step") != step:
                continue
            if want is not None:
                got = (rec.get("params") or {}).get("pws_mpa")
                if got is None or abs(float(got) - float(want)) > 1e-4 * abs(float(want)):
                    continue
            del self.pending[i]
            return StepOutcome(True, rec.get("summary") or {}, "")
        return StepOutcome(False, {}, "", f"{step} is not in the session journal")


_SOURCE_STEPS = ("load", "generate_synth")


def twin_from_journal(journal: list[dict]) -> dict[str, Any]:
    """The twin card of a session, from its journal (step summaries only;
    nothing is recomputed or written). A new source empties the card; a step
    that lands again drops every later tool's result, so the card never mixes
    a new crop with an old stress."""
    tools, flow, scan = tools_from_journal(journal)
    card = twin_card(tools.results, flow, tools.calls)
    card["scan"] = scan
    card["n_steps"] = len(journal)
    return card


def _attached_flow(metrics: Any) -> dict | None:
    """A ``flow_metrics`` record's numbers, or None when they would not pass
    the session's own check (a hand-edited journal leaves the slot as it
    was, never a 500)."""
    from meshforge.pipeline import check_flow_metrics

    try:
        return check_flow_metrics(metrics)
    except ValueError:
        return None


def tools_from_journal(journal: list[dict]) -> tuple[PipelineTools, dict | None, str | None]:
    """Replay a session's journal through the copilot's tool code: (the tools
    with their ``results`` and ``calls``, the flow metrics, the scan's file
    name). The calls carry no timings (0.0 s).

    The flow metrics are the last attached ``flow_metrics`` record's (the
    copilot's ``--flow-metrics``, so the panel and the report print the same
    lines), else ``{}`` if the session itself ran CFD, else None. They belong
    to the scan's lumen, not to the crop: a re-run crop, wall or stress keeps
    them, a new source or segment empties them with the rest of the card,
    and a later session CFD run (no AAA metrics in its summary) does not
    blank them."""
    # nothing is written: the tools only pass paths to the journal backend,
    # but PipelineTools wants an existing work folder
    def fresh() -> PipelineTools:
        return PipelineTools(_JournalBackend(), tempfile.gettempdir())

    tools, seg_rec, flow, scan = fresh(), None, None, None
    batch: list[dict] = []  # consecutive tissue_index records (systolic, what-if)

    def call(tool: str, args: dict[str, Any], recs: list[dict]) -> None:
        for later in TWIN_CHAIN[TWIN_CHAIN.index(tool):]:
            tools.results.pop(later, None)
        tools.backend.pending = list(recs)
        try:
            tools.run(tool, args)
        except Exception:  # noqa: BLE001 - an odd summary leaves the slot pending, never a 500
            tools.results.pop(tool, None)

    def flush() -> None:
        if batch:
            call("strength_index", {}, batch)
            batch.clear()

    for rec in journal:
        step, params = rec.get("step"), rec.get("params") or {}
        if step not in ("tissue_index", "flow_metrics"):  # attached numbers split no batch
            flush()
        if step in _SOURCE_STEPS or str(step).startswith("import_") or step == "segment":
            tools, seg_rec, flow, scan = fresh(), None, None, None
            if step == "segment":
                seg_rec = rec
                scan = PurePath(str(params.get("path", ""))).name or None
        elif step == "sac" and seg_rec is not None:
            call("segment_aorta", {"ct_path": str(seg_rec["params"].get("path", ""))},
                 [seg_rec, rec])
        elif step == "crop":
            call("crop_aneurysm", {"parallel_caps": bool(params.get("parallel_caps"))}, [rec])
        elif step == "volmesh":
            call("mesh_wall", {"sweep": params.get("sweep") or "auto"}, [rec])
        elif step == "aaa_stress":
            call("wall_stress", {"systolic_mmhg": params.get("systolic_mmhg"),
                                 "whatif_mmhg": params.get("whatif_mmhg")}, [rec])
        elif step == "tissue_index":
            batch.append(rec)
        elif step == "cfd":
            if flow is None:  # a CFD run in this session; its summary has no AAA flow metrics
                flow = {}
        elif step == "flow_metrics":
            flow = _attached_flow(params.get("metrics")) or flow
    flush()
    return tools, flow, scan


# --- the report -------------------------------------------------------------------


def _fmt(v: Any) -> str:
    return "—" if v is None else str(v)


def _shown_input(args: dict[str, Any]) -> dict[str, Any]:
    """A tool's input as the report shows it: a path by its file name only,
    so no local folder (or user name) ends up on screen."""
    return {k: PurePath(v).name if k.endswith("_path") and isinstance(v, str) else v
            for k, v in args.items()}


_ABS_TAIL = re.compile(r"(^|\s)((?:/|~/|[A-Za-z]:[\\/]).*)$")
_HOME = re.compile(r"/(?:Users|home)/[^/\s\"']+")


def shown_path(path: str | Path, cwd: str | Path | None = None) -> str:
    """A path as the copilot prints it: relative to the working folder when it
    lies inside it, else its file name, so no local folder (or user name)
    reaches the screen."""
    p, base = Path(path), Path(cwd if cwd is not None else os.getcwd())
    try:
        return str((p if p.is_absolute() else base / p).resolve().relative_to(base.resolve()))
    except ValueError:
        return p.name


def shown_paths(obj: Any, paths=(), cwd: str | Path | None = None) -> Any:
    """``obj`` (JSON-like) with its local paths cut: each of ``paths`` (as
    given and resolved) becomes its :func:`shown_path`, and any other
    ``/Users/<name>`` or ``/home/<name>`` prefix becomes ``~``. For records
    people read (the transcript); replay never reads them (the work file
    keeps the real paths)."""
    subs: dict[str, str] = {}
    for p in paths:
        if p is None:
            continue
        for form in {str(p), str(Path(p).expanduser().resolve())}:
            if Path(form).is_absolute():
                subs[form] = shown_path(p, cwd)
    order = sorted(subs, key=len, reverse=True)

    def text(t: str) -> str:
        for k in order:
            t = t.replace(k, subs[k])
        return _HOME.sub("~", t)

    def walk(o: Any) -> Any:
        if isinstance(o, str):
            return text(o)
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [walk(v) for v in o]
        return o

    return walk(obj)


def _shown_hint(hint: str) -> str:
    """The replay hint with its work-file path cut to the file name: an
    absolute run folder would put a local folder (and user name) on screen."""
    return _ABS_TAIL.sub(lambda m: m.group(1) + PurePath(m.group(2)).name, hint)


def _first_sentence(text: str, limit: int = 220) -> str:
    """Up to the first full stop that ends a sentence (not the one in 1.30),
    with a /Users/<name> prefix shown as ~ (a step's error can name a file)."""
    s = re.split(r"\.(?:\s|$)", _HOME.sub("~", text), maxsplit=1)[0]
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _results_rows(seg, crop, wall, st, idx, bands) -> list[tuple[str, str]]:
    """(label, value) rows of the Results section, from the tool payloads."""
    rows = [
        ("Sac", f"maximum diameter {_n(seg.get('sac_max_diameter_mm'))} mm (area-equivalent "
                f"{_n(seg.get('sac_area_equivalent_diameter_mm'))} mm); thrombus + wall up to "
                f"{_n(seg.get('thrombus_plus_wall_thickness_mm_max'))} mm"),
        ("Crop", f"sac {_n(crop.get('sac_diameter_mm'))} mm vs neck "
                 f"{_n(crop.get('neck_diameter_mm'))} mm (ratio {_n(crop.get('sac_to_neck_ratio'))})"),
        ("Wall mesh", f"{_n(wall.get('hex_cells'))} hex · {_n(wall.get('inverted_cells'))} inverted · "
                      f"{_n(wall.get('capped_rays'))} capped · {_n(wall.get('floored_rays'))} floored · "
                      f"{_n(wall.get('wall_thickness_measured_mm'))} mm thick along the surface "
                      f"normal · gate {_n(wall.get('gate'))}"),
        ("Peak wall stress", f"{_n(st.get('pws_systolic_kpa'))} kPa at {_n(st.get('systolic_mmhg'))} "
                             f"mmHg · {_n(st.get('pws_whatif_kpa'))} kPa at "
                             f"{_n(st.get('whatif_mmhg'))} mmHg ({_n(st.get('pws_change_percent'))} %)"),
    ]
    for k, b in bands.items():
        pressure, _, mmhg = k.partition("_")
        pressure = {"whatif": "what-if"}.get(pressure, pressure)
        rows.append((f"Index, {pressure} {mmhg.replace('mmhg', ' mmHg')}",
                     f"P10 {_n(b['index_p10'])} · P50 {_n(b['index_p50'])} · P90 {_n(b['index_p90'])}"))
    if idx:
        rows.append(("Strength", f"open tensile data from {_n(idx.get('n_aneurysms'))} aneurysms, "
                                 f"{_n(idx.get('n_strips'))} strips — our lab will narrow this band"))
    return rows


_CSS = """
:root{color-scheme:light;--ink:#1a1f26;--muted:#5d6672;--faint:#8a939e;--rule:#e2e6eb;
--card:#f7f8fa;--red:#c8312b;--red-soft:#fbeceb;--ok:#2e7d4f;--bad:#b3261e}
*{box-sizing:border-box}
body{margin:0;background:#fff;color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"SF Pro Text","Helvetica Neue","Segoe UI",Roboto,Arial,sans-serif;
font-variant-numeric:tabular-nums;-webkit-font-smoothing:antialiased}
.page{max-width:920px;margin:0 auto;padding:36px 28px 48px}
.eyebrow{margin:0;color:var(--faint);font-size:12px;letter-spacing:.12em;text-transform:uppercase}
h1{margin:4px 0 14px;font-size:30px;line-height:1.15;letter-spacing:-.01em}
h2{margin:0 0 10px;font-size:13px;letter-spacing:.11em;text-transform:uppercase;color:var(--muted)}
section{margin:0 0 28px}
.meta{display:grid;grid-template-columns:max-content 1fr;gap:3px 14px;margin:0;font-size:14px}
.meta dt{color:var(--faint)}.meta dd{margin:0}
.twin{display:flex;gap:26px;align-items:flex-start;padding:20px 22px;border:1px solid var(--rule);
border-radius:14px;background:linear-gradient(180deg,#fbfcfd,var(--card))}
.twin h2{color:var(--ink);font-size:19px;letter-spacing:0;text-transform:none;margin:2px 0 0}
.twin-sub{margin:2px 0 12px;color:var(--muted);font-size:13px}
.twin-fig{flex:0 0 132px;text-align:center}
.tw-svg{width:118px;height:auto;display:block;margin:0 auto}
.tw-body{fill:#eef1f4;stroke:#b8c0ca;stroke-width:1.4;stroke-linejoin:round}
.tw-next{fill:none;stroke:#aab3be;stroke-width:1.2;stroke-dasharray:3 2}
.tw-aorta path{fill:none;stroke:#aab3be;stroke-width:3.2;stroke-linecap:round;stroke-linejoin:round}
.tw-aorta ellipse{fill:#aab3be}
.is-lit .tw-aorta path{stroke:var(--red)}.is-lit .tw-aorta ellipse{fill:var(--red)}
.mods{list-style:none;margin:10px 0 0;padding:0;font-size:12.5px;color:var(--faint);text-align:left}
.mods li{margin:2px 0}.mods .lit{color:var(--red);font-weight:600}
.twin-main{flex:1;min-width:0}
.slots{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:1fr 1fr;gap:0 22px}
.slots li{display:flex;gap:9px;padding:8px 0;border-top:1px solid var(--rule)}
.slots .dot{flex:0 0 12px;color:var(--faint);line-height:21px}.slots li.on .dot{color:var(--red)}
.slots .lab{min-width:0}.slots .lab b{display:block;font-size:14.5px}
.slots .det{display:block;color:var(--muted);font-size:13px}
.slots .val{display:block;font-weight:600;font-size:14px}
.slots li.off .lab b,.slots li.off .det{color:var(--faint)}
.note{display:block;font-size:12.5px;color:var(--faint)}.note:empty{display:none}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.kpi{border:1px solid var(--rule);border-radius:12px;padding:14px 16px}
.kpi .k{font-size:12px;color:var(--muted);letter-spacing:.06em;text-transform:uppercase}
.kpi .v{font-size:28px;font-weight:650;letter-spacing:-.01em;margin:4px 0 2px}
.kpi .v small{font-size:15px;font-weight:500;color:var(--muted)}
.kpi .s{font-size:13px;color:var(--muted)}
.kpi.idx{background:var(--red-soft);border-color:#f2cfcc}
table.grid{border-collapse:collapse;width:100%;font-size:14px}
table.grid th{text-align:left;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
font-weight:600;border-bottom:1.5px solid var(--ink);padding:6px 8px}
table.grid td{border-bottom:1px solid var(--rule);padding:7px 8px;vertical-align:top}
table.grid td.l{color:var(--muted);white-space:nowrap;width:1%}
table.grid td.num{text-align:right;white-space:nowrap}
table.grid code{font:12.5px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;color:#3b4350}
tr.fail td{background:#fdf3f2}.fail .res{color:var(--bad);font-weight:600}
.ok{color:var(--ok);font-weight:600}
.guard{display:inline-block;margin:8px 0 0;padding:4px 10px;border-radius:999px;font-size:12.5px;
background:#eef6f1;color:var(--ok)}.guard.warn{background:#fdf3f2;color:var(--bad)}
.guard.neutral{background:var(--card);color:var(--muted)}
.summary p{margin:0 0 8px;max-width:68ch}.summary .none{color:var(--muted);font-style:italic}
.caveat{border-left:3px solid var(--red);padding:8px 14px;background:var(--red-soft);border-radius:0 8px 8px 0}
.caveats ul{margin:0;padding-left:18px}.caveats li{margin:2px 0}
.repro code{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--card);padding:3px 7px;border-radius:5px}
.lede{margin:0 0 6px;color:var(--muted);font-size:15px}
.plain{border:1px solid var(--rule);border-radius:14px;padding:18px 22px 16px;background:#fff}
.plain p{margin:0 0 11px;font-size:16.5px;line-height:1.6;max-width:70ch}.plain p:last-of-type{margin-bottom:0}
.plain p b{color:var(--red)}.plain .none{color:var(--muted);font-style:italic}
.steps{list-style:none;margin:0;padding:0;counter-reset:step}
.steps li{display:grid;grid-template-columns:30px 1fr;gap:12px;padding:10px 0;border-top:1px solid var(--rule);counter-increment:step}
.steps .n{width:26px;height:26px;border-radius:13px;background:var(--ink);color:#fff;font-size:13px;font-weight:700;
display:flex;align-items:center;justify-content:center;margin-top:1px}
.steps .n::before{content:counter(step)}
.steps li.fail .n{background:var(--bad)}.steps li.flow .n{background:#2a6fb0}
.steps .t{font-weight:650;font-size:15px}.steps .d{color:var(--muted);font-size:13px}
.steps .r{margin-top:3px;font-size:14px}.steps li.fail .r{color:var(--bad)}
details.more{border-top:1px solid var(--rule);padding-top:10px}
details.more summary{cursor:pointer;font-size:13px;letter-spacing:.11em;text-transform:uppercase;color:var(--muted);font-weight:600;margin-bottom:12px}
details.more h2{margin-top:14px}
footer{border-top:1px solid var(--rule);padding-top:12px;color:var(--muted);font-size:12.5px}
@media (max-width:640px){.page{padding:22px 16px 36px}.twin{flex-direction:column;align-items:center}
.twin-fig{flex:none}.kpis,.slots{grid-template-columns:1fr}h1{font-size:25px}
table.grid td.l{white-space:normal}.pipeline{overflow-x:auto}}
@media print{@page{margin:14mm}body{font-size:10.5pt}.page{max-width:none;padding:0}
.twin,.kpi,table.grid tr,section{break-inside:avoid}.twin{background:none}
.kpi.idx,tr.fail td,.caveat,.guard{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
"""


def _kpi(label: str, value: str, unit: str, sub: str, cls: str = "") -> str:
    e = html.escape
    return (f'<div class="kpi {cls}"><div class="k">{e(label)}</div>'
            f'<div class="v">{e(value)} <small>{e(unit)}</small></div><div class="s">{e(sub)}</div></div>')


def _input_text(args: dict[str, Any]) -> str:
    def shown(v: Any) -> str:
        return str(v).lower() if isinstance(v, bool) else _n(v) if isinstance(v, float) else str(v)

    return ", ".join(f"{k}: {shown(v)}" for k, v in _shown_input(args).items()) or "—"


def render_report(*, request: str, ct_name: str, credit: str, results: dict[str, dict],
                  calls: list[dict], summary_text: str | None, guard_note: str,
                  model: str, flow: dict | None = None,
                  replay_hint: str = "", flow_note: str = "") -> tuple[str, str]:
    """(markdown, html) of the run, from the tool outputs only. ``flow_note``
    (D112) says how the session's own blood-flow result was made, when the
    run solved one."""
    seg = results.get("segment_aorta", {})
    crop = results.get("crop_aneurysm", {})
    wall = results.get("mesh_wall", {})
    st = results.get("wall_stress", {})
    idx = results.get("strength_index", {})
    bands = {k: v for k, v in idx.items() if k.endswith("mmhg")}
    replay_hint = _shown_hint(replay_hint)

    md = [f"# Aorta twin — investigational biomechanics report", "",
          f"**Request:** {request}", f"**Scan:** {ct_name} ({credit})", f"**Copilot:** {model}", ""]
    md += ["## Results", "",
           f"- Sac: maximum diameter {_fmt(seg.get('sac_max_diameter_mm'))} mm "
           f"(area-equivalent {_fmt(seg.get('sac_area_equivalent_diameter_mm'))} mm), "
           f"thrombus + wall up to {_fmt(seg.get('thrombus_plus_wall_thickness_mm_max'))} mm",
           f"- Crop: sac {_fmt(crop.get('sac_diameter_mm'))} mm vs neck "
           f"{_fmt(crop.get('neck_diameter_mm'))} mm (ratio {_fmt(crop.get('sac_to_neck_ratio'))})",
           f"- Wall mesh: {_fmt(wall.get('hex_cells'))} hex, {_fmt(wall.get('inverted_cells'))} inverted, "
           f"{_fmt(wall.get('capped_rays'))} capped, {_fmt(wall.get('floored_rays'))} floored, "
           f"{_fmt(wall.get('wall_thickness_measured_mm'))} mm thick along the surface normal — "
           f"gate {_fmt(wall.get('gate'))}",
           f"- Peak wall stress: {_fmt(st.get('pws_systolic_kpa'))} kPa at "
           f"{_fmt(st.get('systolic_mmhg'))} mmHg, {_fmt(st.get('pws_whatif_kpa'))} kPa at "
           f"{_fmt(st.get('whatif_mmhg'))} mmHg ({_fmt(st.get('pws_change_percent'))} %)"]
    for k, b in bands.items():
        md.append(f"- Index ({k.replace('_', ' ')}): P10 {b['index_p10']} · P50 {b['index_p50']} · "
                  f"P90 {b['index_p90']}")
    if idx:
        md.append(f"- Strength: open tensile data from {_fmt(idx.get('n_aneurysms'))} aneurysms, "
                  f"{_fmt(idx.get('n_strips'))} strips — our lab will narrow this band")
    if flow_note:
        md.append(f"- Blood flow: {flow_note}")
    md += ["", "## Pipeline (every action journaled)", "",
           "| # | tool | input | result | s |", "|---|---|---|---|---|"]
    for i, c in enumerate(calls, 1):
        res = "ok" if c["ok"] else "FAILED → " + _first_sentence(c["error"])
        md.append(f"| {i} | {c['tool']} | `{_shown_input(c['input'])}` | {res} | {c['seconds']} |")
    md += ["", "## Copilot summary", "", summary_text or f"_({guard_note})_", "",
           "## Caveats", ""] + [f"- {c}" for c in CAVEATS]
    if replay_hint:
        md += ["", "## Reproduce", "", f"`{replay_hint}`"]
    markdown = "\n".join(md) + "\n"
    return markdown, _render_html(
        request=request, ct_name=ct_name, credit=credit, results=results, calls=calls,
        summary_text=summary_text, guard_note=guard_note, model=model, flow=flow,
        replay_hint=replay_hint, rows=_results_rows(seg, crop, wall, st, idx, bands),
        flow_note=flow_note,
    )


#: The pipeline's tools as the report names them for a reader who does not
#: know the tools: (title, what the step does).
STEP_NAMES = {
    "segment_aorta": ("Find the aorta on the scan",
                      "the AI model outlines the aorta on the CT; the aneurysm sac is recovered around it"),
    "crop_aneurysm": ("Cut out the aneurysm", "two flat cuts: below the kidney arteries and above the fork"),
    "mesh_wall": ("Build the wall", "a 1.5 mm wall of small blocks; a quality check measures it"),
    "wall_stress": ("Compute the wall stress", "finite elements under the two blood pressures"),
    "strength_index": ("Compare with wall strength",
                       "the stress against open lab measurements of aneurysm-wall strength"),
}


def _band_mmhg(key: str) -> str:
    """``systolic_140mmhg`` -> ``140``."""
    return key.rsplit("_", 1)[-1].replace("mmhg", "")


def _step_line(c: dict) -> str:
    """One plain line for a tool call, from its payload only (so every number
    traces to a tool); a failed quality gate says what the check found."""
    p, tool, ok = c.get("payload") or {}, c.get("tool"), bool(c.get("ok"))
    if tool == "segment_aorta" and ok:
        return (f"sac {_n(p.get('sac_max_diameter_mm'))} mm at its widest · thrombus + wall up to "
                f"{_n(p.get('thrombus_plus_wall_thickness_mm_max'))} mm · AI {_n(p.get('ai_seconds'))} s")
    if tool == "crop_aneurysm" and ok:
        return (f"sac {_n(p.get('sac_diameter_mm'))} mm, neck {_n(p.get('neck_diameter_mm'))} mm "
                f"(ratio {_n(p.get('sac_to_neck_ratio'))})" + (" · parallel cuts" if p.get("parallel_caps") else ""))
    if tool == "mesh_wall" and p:
        if ok:
            return (f"{_n(p.get('hex_cells'))} blocks · {_n(p.get('inverted_cells'))} inverted · "
                    f"{_n(p.get('wall_thickness_measured_mm'))} mm thick · gate {_n(p.get('gate'))}")
        return (f"{_n(p.get('wall_thickness_measured_mm'))} mm thick ({_n(p.get('wall_thickness_off_percent'))} % "
                f"off) · gate {_n(p.get('gate'))} — the quality check stopped the run here: a wall this far "
                f"off would misstate the stress, so the aneurysm was cut again and the wall rebuilt")
    if tool == "wall_stress" and ok:
        return (f"{_n(p.get('pws_systolic_kpa'))} kPa at {_n(p.get('systolic_mmhg'))} mmHg · "
                f"{_n(p.get('pws_whatif_kpa'))} kPa at {_n(p.get('whatif_mmhg'))} mmHg "
                f"({_n(p.get('pws_change_percent'))} %)" + (" · thrombus modelled" if p.get("thrombus_modelled") else ""))
    if tool == "strength_index" and ok:
        parts = []
        for key, b in p.items():
            if isinstance(b, dict) and key.endswith("mmhg"):
                band = (f" (band {_n(b.get('index_p10'))} to {_n(b.get('index_p90'))})"
                        if key.startswith("systolic") else "")
                parts.append(f"index {_n(b.get('index_p50'))}{band} at {_band_mmhg(key)} mmHg")
        if p.get("n_aneurysms") is not None:
            parts.append(f"open data: {_n(p.get('n_aneurysms'))} aneurysms, {_n(p.get('n_strips'))} strips")
        return " · ".join(parts)
    if not ok:
        return "failed: " + _first_sentence(c.get("error") or "")
    return ""


def _steps_html(calls: list[dict], flow_note: str) -> str:
    e = html.escape
    rows = []
    for c in calls:
        title, detail = STEP_NAMES.get(c.get("tool"), (str(c.get("tool")), ""))
        line = _step_line(c)
        rows.append(
            f'<li class="{"" if c.get("ok") else "fail"}"><span class="n" aria-hidden="true"></span><div>'
            f'<div class="t">{e(title)}</div><div class="d">{e(detail)}</div>'
            + (f'<div class="r">{e(line)}</div>' if line else "") + "</div></li>"
        )
    if flow_note:
        rows.append(
            '<li class="flow"><span class="n" aria-hidden="true"></span><div>'
            '<div class="t">Blood flow (CFD)</div><div class="d">steady, resting flow through the lumen</div>'
            f'<div class="r">{e(flow_note)}</div></div></li>'
        )
    return f'<ol class="steps">{"".join(rows)}</ol>'


_LABEL = re.compile(r"^([^:\d\n]{3,48}):\s*(\S.*)$", re.S)


def _plain_html(summary_text: str | None, guard_note: str) -> str:
    """The copilot's summary as readable paragraphs: a paragraph that opens
    with a short label and a colon gets the label in bold."""
    e = html.escape
    if not summary_text:
        paras = ['<p class="none">No copilot summary in this report.</p>']
    else:
        chunks = [c.strip() for c in re.split(r"\n\s*\n", summary_text) if c.strip()]
        if len(chunks) == 1 and "\n" in chunks[0]:
            chunks = [c.strip() for c in chunks[0].split("\n") if c.strip()]
        paras = []
        for chunk in chunks:
            chunk = re.sub(r"\*\*|__", "", chunk).strip()  # Claude's own bold marks: the label is bold already
            m = _LABEL.match(chunk)
            if m:
                paras.append(f"<p><b>{e(m.group(1).strip())}:</b> {e(m.group(2).strip())}</p>")
            else:
                paras.append(f"<p>{e(chunk)}</p>")
    guard_cls = ("" if summary_text else
                 " warn" if any(w in guard_note for w in ("withheld", "declined")) else " neutral")
    return ("".join(paras) + f'<span class="guard{guard_cls}">{e(guard_note)}</span>')


def _render_html(*, request, ct_name, credit, results, calls, summary_text, guard_note, model,
                 flow, replay_hint, rows, flow_note="") -> str:
    """The report as a real page, in reading order: the plain-words summary,
    the three headline numbers, the twin card, the steps in plain names, then
    every number and every call behind a fold. Every string from a tool,
    Claude or the command line goes through html.escape; no script, no
    external fetch."""
    e = html.escape
    st = results.get("wall_stress", {})
    idx = results.get("strength_index", {})
    sys_band = _band(idx, "systolic_")

    kpis = ""
    if st:
        kpis = (
            _kpi("Peak wall stress", _n(st.get("pws_systolic_kpa")), "kPa",
                 f"at {_n(st.get('systolic_mmhg'))} mmHg"
                 + (", thrombus modelled" if st.get("thrombus_modelled") else ""))
            + _kpi("Blood-pressure what-if", _n(st.get("pws_whatif_kpa")), "kPa",
                   f"at {_n(st.get('whatif_mmhg'))} mmHg ({_n(st.get('pws_change_percent'))} %)")
        )
        if sys_band:
            kpis += _kpi("Investigational index", _n(sys_band["index_p50"]), "P50",
                         f"P10 {_n(sys_band['index_p10'])} – P90 {_n(sys_band['index_p90'])}; "
                         f"not a diagnosis", "idx")
        kpis = f'<section class="kpis" id="headline">{kpis}</section>'

    result_rows = "".join(f'<tr><td class="l">{e(k)}</td><td>{e(v)}</td></tr>' for k, v in rows)
    pipe_rows = []
    for i, c in enumerate(calls, 1):
        res = ('<span class="ok">ok</span>' if c["ok"]
               else "FAILED → " + e(_first_sentence(c.get("error") or "")))
        pipe_rows.append(
            f'<tr class="{"" if c["ok"] else "fail"}"><td class="num">{i}</td>'
            f'<td><code>{e(str(c["tool"]))}</code></td><td><code>{e(_input_text(c["input"]))}</code></td>'
            f'<td class="res">{res}</td><td class="num">{e(_n(c.get("seconds")))}</td></tr>'
        )
    caveats = "".join(f"<li>{e(c)}</li>" for c in CAVEATS)
    repro = (f'<section class="repro" id="reproduce"><h2>Reproduce</h2><code>{e(replay_hint)}</code>'
             f"</section>" if replay_hint else "")
    return (
        f"<!doctype html>{REPORT_HTML_MARKER}<html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Aorta twin report — {e(ct_name)}</title><style>{_CSS}</style></head><body>"
        "<main class='page'>"
        "<header><p class='eyebrow'>meshforge · investigational biomechanics report</p>"
        "<h1>Aorta twin</h1>"
        f"<p class='lede'>{e(request)}</p>"
        "<dl class='meta'>"
        f"<dt>Scan</dt><dd>{e(ct_name)} — {e(credit)}</dd>"
        f"<dt>Copilot</dt><dd>{e(model)}</dd></dl></header>"
        f'<section class="plain" id="summary"><h2>In plain words</h2>{_plain_html(summary_text, guard_note)}</section>'
        f"{kpis}"
        f"<section>{twin_card_html(results, flow, calls)}</section>"
        f'<section id="steps"><h2>What the pipeline did</h2>{_steps_html(calls, flow_note)}</section>'
        '<details class="more" id="details"><summary>Every number and every call</summary>'
        f'<section id="results"><h2>Results</h2><table class="grid">{result_rows}</table></section>'
        '<section id="pipeline" class="pipeline"><h2>Pipeline · every action journaled</h2>'
        '<table class="grid"><thead><tr><th>#</th><th>tool</th><th>input</th><th>result</th>'
        f'<th>s</th></tr></thead><tbody>{"".join(pipe_rows)}</tbody></table></section>'
        "</details>"
        '<section class="caveats" id="caveats"><h2>Caveats</h2><div class="caveat">'
        "<b>Investigational index</b>: peak wall stress divided by lab-measured wall strength. "
        "It is not a diagnosis and not a rupture prediction.</div>"
        f'<ul style="margin-top:10px">{caveats}</ul></section>'
        f"{repro}"
        f"<footer>Data: {e(credit)}. Every number on this page comes from a pipeline tool output"
        + (" (blood flow: the attached CFD metrics)" if flow is not None else "")
        + (". The copilot's summary passed the numbers guard." if summary_text else ".")
        + "</footer>"
        "</main></body></html>"
    )
