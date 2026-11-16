"""Static guards for the Workbench frontend (the demo page).

The page lives entirely in ``web/static``. These tests read the shipped
assets as text and assert the *contracts* a browser walkthrough confirmed: no
browser, no JS engine, no DOM. That buys a drift guard, not a rendering
proof.

The page is always the demo layout: six tabs, the patient atlas and the twin
card, the layer switch, the turntable and the reveal. There is no switch to
another layout, no source box, no Quick start, no history table and no native
file picker; ``test_the_page_is_always_the_demo_layout`` pins that.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from meshforge.pipeline import STEP_ORDER


def _static_dir() -> Path:
    import meshforge.web as web_pkg

    return Path(web_pkg.__file__).parent / "static"


@pytest.fixture(scope="module")
def app_js() -> str:
    return (_static_dir() / "app.js").read_text()


@pytest.fixture(scope="module")
def index_html() -> str:
    return (_static_dir() / "index.html").read_text()


@pytest.fixture(scope="module")
def styles_css() -> str:
    return (_static_dir() / "styles.css").read_text()


# ---- fix 5: no double-submit, and the busy state always unwinds ---------
# Every button that fires a request must go through withBusy(), which
# disables + relabels it for the duration and restores in `finally`.

#: Buttons deliberately NOT wrapped: they fire no long request.
_UNBUSIED_BUTTONS = {
    "#btn-reset",  # /session/reset is a dict clear — returns before a relabel would paint
    "#vp-play",    # client-side frame timer, no request at all
    "#fp-pick",    # arms/cancels a viewport pick and relabels itself; no request on click
    "#fp-swap",    # swaps two client-side ends; the Q/Re preview it triggers is a read-only POST
}


def test_with_busy_disables_relabels_and_restores(app_js: str) -> None:
    body = _function_body(app_js, "async function withBusy")
    assert "btn.disabled = true" in body
    assert 'btn.classList.add("busy")' in body
    assert "btn.textContent = label" in body
    # The restore must be in `finally` — an exception mid-request must not
    # strand the button disabled and reading "Running…" forever.
    finally_block = body.split("finally", 1)[1]
    assert "btn.textContent = prevText" in finally_block
    assert 'btn.classList.remove("busy")' in finally_block
    assert "btn.disabled =" in finally_block


def test_busy_restore_is_state_derived_never_a_blind_false(app_js: str) -> None:
    # #btn-save tracks journal_len via refreshState(); restoring it to a flat
    # `false` would re-enable Save on an empty session. withBusy takes a
    # callback for exactly that, and the Save wiring must pass one.
    body = _function_body(app_js, "async function withBusy")
    assert 'typeof restoreDisabled === "function" ? restoreDisabled()' in body
    save_wiring = _slice_between(app_js, '$("#btn-save").addEventListener', '$("#btn-wload")')
    assert "lastJournalLen" in save_wiring, "Save's busy restore dropped its state callback"


@pytest.mark.parametrize("button", ["#btn-save", "#btn-wload"])
def test_every_request_button_is_wrapped(app_js: str, button: str) -> None:
    wiring = _slice_between(app_js, f'$("{button}").addEventListener', "});\n  $(")
    assert "withBusy(" in wiring, f"{button} can be double-clicked into concurrent POSTs"


def test_step_run_buttons_are_wrapped(app_js: str) -> None:
    # The per-step Run button is built in buildStepPanel for every panel,
    # so one assertion covers every generated panel.
    panel = _function_body(app_js, "function buildStepPanel")
    assert 'withBusy(btn, "Running…"' in panel


def test_no_unwrapped_button_slipped_in(app_js: str) -> None:
    # $("#x") and wrap.querySelector("#x") — the cfd flow picker wires its
    # buttons through the latter, which a $(...)-only scan would never see.
    wired = set(re.findall(
        r'(?:\$|\.querySelector)\("(#[\w-]+)"\)\.addEventListener\("click"', app_js
    ))
    unwrapped = {b for b in wired if b not in _UNBUSIED_BUTTONS}
    for button in sorted(unwrapped):
        wiring = _slice_between(app_js, f'$("{button}").addEventListener', "});\n  $(")
        assert "withBusy(" in wiring, (
            f"{button} fires on click without a busy state — wrap it in withBusy() "
            f"or add it to _UNBUSIED_BUTTONS with a reason"
        )


def test_busy_class_is_actually_styled(styles_css: str) -> None:
    # withBusy's class is cosmetic-only if nothing paints it.
    assert "button.run.busy" in styles_css


# ---- fix 7: collapsible panels, and errors where the click was ---------


def test_step_panels_are_collapsible_details(app_js: str) -> None:
    panel = _function_body(app_js, "function buildStepPanel")
    assert 'createElement("details")' in panel
    assert 'createElement("summary")' in panel
    # Closed at start, always: six tabs, the whole path at a glance (a click
    # still opens a tab's knobs). Nothing is stored between loads.
    assert "box.open = false;" in panel


def test_errors_echo_inline_under_their_own_panel(app_js: str, styles_css: str) -> None:
    body = _function_body(app_js, "function panelErrorFor")
    assert 'querySelector(":scope > .panel-error")' in body
    assert "box.appendChild(el)" in body, "the error must land inside the panel, not in #log only"
    assert ".panel-error" in styles_css
    # …and clear on that panel's next success, so a fixed error can't linger.
    assert "clearErrorEl(panelErrorFor(box))" in app_js


def test_workfile_errors_have_slots_too(index_html: str, app_js: str) -> None:
    for slot in ("save-error", "wload-error"):
        assert f'id="{slot}" class="panel-error" hidden' in index_html
        assert f'setError($("#{slot}"), err)' in app_js


def test_summary_chevron_replaced_the_floated_legend(styles_css: str) -> None:
    assert ".step-box > summary" in styles_css
    assert '.step-box[open] > summary::before { content: "▾"; }' in styles_css
    assert "legend" not in styles_css, "the pre-fix floated legend rule is dead — drop it"


# ---- fix 2 (frontend half): panels laid out in the served order ---------


def test_panels_follow_catalog_step_order_and_complain_when_it_is_short(app_js: str) -> None:
    body = _function_body(app_js, "async function init")
    assert "catalog.step_order" in body
    assert "Array.isArray(catalog.step_order)" in body, "a missing/short step_order must not blank the UI"
    # Rule 6: a step /schema has but step_order forgot is still rendered —
    # loudly, at the end — never silently dropped.
    assert "WARNING" in body and "missing from /catalog step_order" in body


# ---- helpers ------------------------------------------------------------


def _function_body(source: str, signature: str) -> str:
    """Text from ``signature`` to the next top-level ``\\n}``.

    Crude on purpose: these assets are hand-written and flat, and a real JS
    parser is not a dependency this suite should grow.
    """
    start = source.index(signature)
    end = source.index("\n}", start)
    return source[start:end]


def _slice_between(source: str, start_marker: str, end_marker: str) -> str:
    start = source.index(start_marker)
    try:
        end = source.index(end_marker, start + len(start_marker))
    except ValueError:
        end = len(source)
    return source[start:end]


# ---- M6: the twin card panel ---------------------------------------------


def test_twin_outline_is_the_reports_own_svg(index_html: str) -> None:
    # one body outline for the report header and the Workbench panel
    from meshforge.copilot.report import BODY_SVG

    assert BODY_SVG in index_html
    for cls in ("tw-body", "tw-next", "tw-aorta"):
        assert f'class="{cls}"' in BODY_SVG


def test_twin_panel_paints_server_data_as_text_only(app_js: str, index_html: str) -> None:
    for slot in ('id="twin"', 'id="twin-slots"', 'id="twin-modules"', 'id="twin-scan"'):
        assert slot in index_html, slot
    body = _function_body(app_js, "function renderTwin")
    # innerHTML only ever clears a list; every server string goes through textContent
    assert re.findall(r'innerHTML\s*=\s*([^;]+);', body) == ['""', '""']
    assert body.count(".textContent =") >= 5
    assert 'dataset.lit = twin.lit ? "1" : "0"' in body
    refresh = _function_body(app_js, "async function refreshTwin")
    assert 'apiGet("/twin")' in refresh and "renderTwin(twin)" in refresh


def test_twin_panel_refreshes_on_every_session_change(app_js: str) -> None:
    # a step, another client's change (the copilot over HTTP), a work-file
    # load, a reset, and the first paint all repaint the twin
    for fn in ("async function afterStep", "function startAutoRefresh",
               "async function loadWorkfile", "async function resetSession",
               "async function init()"):
        assert "await refreshTwin()" in _function_body(app_js, fn), fn


def test_twin_lit_state_is_styled(styles_css: str) -> None:
    assert '.twin[data-lit="1"] .tw-aorta path' in styles_css
    assert ".slot.filled .slot-dot" in styles_css and ".slot.next" in styles_css



def test_no_local_folder_reaches_the_screen(app_js: str) -> None:
    """M6: the Workbench is recorded. The summary and the log show an
    absolute path value by its file name and any /Users/<name> as "~";
    display only (the journal keeps the real paths)."""
    shown = _function_body(app_js, "function shownJson")
    assert "ABS_PATH.test(v) ? shortPath(v) : v" in shown and "shownText(" in shown
    assert "shownText(text)" in _function_body(app_js, "function log")
    assert "shownJson(result.summary, 2)" in _function_body(app_js, "function showSummary")
    # the journal is fetched for the tabs' marks only; nothing of it is printed
    journal = _function_body(app_js, "async function refreshJournal")
    assert "JSON.stringify" not in journal and "textContent" not in journal
    # the atlas rows and the status bar go through the same display helpers
    assert "shownText(s.description" in app_js
    assert "st.input_path ? shortPath(st.input_path)" in app_js


# ---- the demo layout (D108): always on --------------------------------------
# index.html sets ``<body class="demo">`` once; the layout is CSS under that
# class plus app.js functions that paint the tabs, the atlas and the running
# step. These guards keep the contract: the AAA path's steps as numbered tabs,
# a reveal, the running step, the patient atlas, and nothing else on the page.

_DEMO_STEP_RE = re.compile(r'\["(?P<step>[a-z_0-9]+)",\s*"(?P<title>[^"]+)",\s*"(?P<hint>[^"]+)"\]')


def _demo_steps(app_js: str) -> list[tuple[str, str, str]]:
    start = app_js.index("const DEMO_STEPS = [")
    end = app_js.index("];", start)
    return [(m.group("step"), m.group("title"), m.group("hint")) for m in _DEMO_STEP_RE.finditer(app_js[start:end])]


def test_the_page_is_always_the_demo_layout(index_html: str, app_js: str, styles_css: str) -> None:
    """One layout: the class is static, nothing can switch it off, and the
    panels of the general Workbench (the Demo switch, the source box, the
    Quick start, the history table, the native Browse pickers) are gone from
    the markup, the script and the stylesheet alike."""
    assert index_html.count('<body class="demo">') == 1
    assert "document.body.classList" not in app_js, "something still switches the page's own class"
    # nothing is stored or read back: no preference can change the layout
    for gone in ("localStorage", "searchParams", "history.replaceState", "meshforge.demo"):
        assert gone not in app_js, gone
    # the removed pieces, by their ids, functions and classes
    for gone in ('id="demo-switch"', 'id="demo-toggle"', 'id="help"', "Quick start", 'id="source-box"',
                 'id="synth-shape"', 'id="btn-synth"', 'id="load-path"', 'id="btn-load"',
                 'id="history-panel"', 'id="history"'):
        assert gone not in index_html, gone
    for gone in ("applyDemo", "readDemoPref", "isDemo", "initDemoMode", "removeAtlasPanel",
                 "BROWSE_KNOBS", "addBrowseButton", "browseInto", "bridgeAvailable", "meshforgeBridge",
                 "generateSynth", "loadSurface", "/source/synth", "/source/load",
                 "refreshHistory", "#history", "#source-box", "#help", "#demo-toggle",
                 "loadPanelOpenPref", "savePanelOpenPref"):
        assert gone not in app_js, gone
    for gone in (".demo-switch", ".help", ".history", ".browse", "#help", "#source-box", "#history-panel",
                 "body:not(.demo)"):
        assert gone not in styles_css.replace(".knob-help", ""), gone
    # the six tabs and the right-hand panels are what is left
    for kept in ('id="step-panels"', 'id="twin-panel"', '<template id="atlas-svg">', 'id="log"',
                 'id="btn-save"', 'id="btn-wload"', 'id="btn-reset"', 'id="vp-layer-ctl"', 'id="vp-busy"'):
        assert kept in index_html, kept


def test_demo_tabs_are_the_aaa_path_in_run_order(app_js: str) -> None:
    """Four to seven tabs, real steps, in the order the copilot runs them —
    which is also their STEP_ORDER (DOM) order, so no reordering is needed."""
    tabs = _demo_steps(app_js)
    steps = [t[0] for t in tabs]
    assert steps == ["segment", "crop", "volmesh", "aaa_stress", "tissue_index", "cfd"]
    assert 4 <= len(steps) <= 7
    places = [STEP_ORDER.index(s) for s in steps]  # KeyError-free: every one is a step
    assert places == sorted(places), "the tabs must sit in STEP_ORDER, the panels' DOM order"
    titles = [t[1] for t in tabs]
    assert titles[0].startswith("Load"), "the first tab is the scan going in"
    assert any("FEA" in t for t in titles) and any("CFD" in t for t in titles)
    assert all(t[1] and t[2] for t in tabs), "every tab has a title and a hint"
    # the sub-steps without a tab of their own map onto one that exists
    aliases = re.search(r"const DEMO_STEP_OF = \{([^}]*)\};", app_js).group(1)
    for alias, target in re.findall(r'([a-z_]+):\s*"([a-z_]+)"', aliases):
        assert target in steps, (alias, target)
    assert "sac" in aliases and "flow_metrics" in aliases


def test_demo_layout_shows_only_the_tabs(styles_css: str) -> None:
    rules = {
        "body.demo #step-panels > details.step-box { display: none; }",
        'body.demo #step-panels > details.step-box[data-demo="1"] { display: block; }',
        'body.demo #step-panels > details.step-box[data-demo="1"] > summary .step-name { display: none; }',
    }
    for rule in rules:
        assert rule in styles_css, rule
    # no rule hides a panel without the body.demo guard
    for line in styles_css.splitlines():
        if "details.step-box" in line and "display: none" in line:
            assert line.startswith("body.demo "), line
    # the tab is laid out under body.demo, the class index.html always sets
    assert ".demo-tab { display: none; }" in styles_css
    assert "body.demo .demo-tab {" in styles_css


def test_demo_tabs_are_numbered_closed_and_marked_from_the_journal(app_js: str, styles_css: str) -> None:
    assert 'box.dataset.demo = "1"' in app_js
    assert 'box.dataset.demoState = "waiting"' in app_js
    assert 'tab.className = "demo-tab"' in app_js and 'num.textContent = String(i + 1)' in app_js
    assert "decorateDemoPanels();" in app_js and "initDemoLayout();" in app_js
    # the tabs start closed (the path at a glance) and are painted at once
    assert "box.open = false;" in _function_body(app_js, "function buildStepPanel")
    start = app_js[app_js.index("function initDemoLayout()"):app_js.index("async function init()")]
    assert "paintDemoTabs();" in start
    # done = in the journal (truncation upstream puts later tabs back to waiting)
    assert "noteDemoJournal(journal);" in _function_body(app_js, "async function refreshJournal")
    for fn in ("async function afterStep", "function startAutoRefresh", "async function loadWorkfile",
               "async function resetSession", "async function init()"):
        assert "await refreshJournal()" in _function_body(app_js, fn), fn
    paint = app_js[app_js.index("function paintDemoTabs()"):]
    paint = paint[: paint.index("\n}\n") + 3]
    assert '"running"' in paint and '"done"' in paint and '"waiting"' in paint
    assert "demoDoneSteps.has(step)" in paint
    # running = GET /state `running`, through the alias map, on change and on clear
    busy = app_js[app_js.index("function updateBusyLine(st)"):app_js.index("function initDemoLayout()")]
    assert busy.count("paintDemoTabs();") >= 2
    for state in ("running", "done", "failed"):
        assert f'details.step-box[data-demo-state="{state}"]' in styles_css, state


def test_demo_mesh_wall_tab_shows_the_gate_failure_in_the_reports_own_words(app_js: str) -> None:
    """While the copilot's last mesh_wall call failed its gate, the journal
    still holds that wall's volmesh record; the tab must not read as a pass.
    The twin's wall slot (GET /twin, from copilot/report.py) says so in a
    fixed note; app.js matches that note's start, so pin the two together."""
    import inspect

    from meshforge.copilot import report

    assert 'const WALL_GATE_FAIL_NOTE = "last wall gate FAIL";' in app_js
    assert 'f"last wall gate FAIL: {' in inspect.getsource(report)
    assert "noteDemoTwin(twin);" in app_js[app_js.index("function renderTwin(twin)"):]
    note = app_js[app_js.index("function noteDemoTwin(twin)"):]
    note = note[: note.index("\n}\n") + 3]
    assert 's.key === "wall"' in note and "startsWith(WALL_GATE_FAIL_NOTE)" in note
    paint = app_js[app_js.index("function paintDemoTabs()"):]
    paint = paint[: paint.index("\n}\n") + 3]
    assert 'step === "volmesh" && demoWallFailed ? "failed"' in paint
    # running wins over failed (a re-mesh in progress), failed over done
    assert paint.index('"running"') < paint.index('"failed"') < paint.index('"done"')


def test_busy_line_follows_the_running_step(app_js: str, index_html: str) -> None:
    assert 'id="vp-busy"' in index_html
    busy = app_js[app_js.index("function updateBusyLine(st)"):]
    assert "if (!st || !st.running) {" in busy[:200]
    assert "updateBusyLine(st);" in app_js                  # every /state poll
    assert "st.running" in app_js and "running_for" in app_js


def test_step_reveal_draws_a_new_artifact_in_from_the_top_in_demo_mode(
    app_js: str, index_html: str, styles_css: str
) -> None:
    """D112: the D108 curtain is gone; a NEW artifact is drawn in cell by cell
    from the top down (the polygons sorted by their highest z, the
    connectivity grown as a prefix) over 2.4 s, the previous artifact a faint
    shell meanwhile. Demo mode only, never under reduced motion; the same
    artifact served again (a journal-only step) is not replayed; a field or
    frame swap, a new load or a clear cancels a running sweep; the whole
    state stands at the end. viewport.js still asks for ``body.demo``, the
    class index.html always sets."""
    vp_js = (_static_dir() / "viewport.js").read_text()
    assert '<body class="demo">' in index_html
    assert "vp-reveal" not in index_html and ".vp-reveal" not in styles_css and "playReveal" not in app_js
    assert "var SWEEP_MS = 2400;" in vp_js
    assert "function planSweep(state)" in vp_js and "function sweepState(plan, p)" in vp_js
    on = vp_js[vp_js.index("function sweepOn()"):vp_js.index("var TYPED")]
    assert "demoOn() && !reducedMotion()" in on
    plan = vp_js[vp_js.index("function planSweep(state)"):vp_js.index("function sweepState")]
    assert "pts[3 * conn[i + j] + 2]" in plan                 # the cell's highest z
    assert "cells.sort(function (a, b) { return b[2] - a[2]; });" in plan  # top first
    state = vp_js[vp_js.index("function sweepState(plan, p)"):vp_js.index("function sweepStop")]
    assert "plan.ordered.subarray(0, plan.offsets[k])" in state  # a prefix: nothing invented
    assert "Object.assign({}, plan.state, { mesh: mesh })" in state  # points, field, range untouched
    stop = vp_js[vp_js.index("function sweepStop(finish)"):vp_js.index("function sweepStep")]
    assert "items[i].set(items[i].plan.state)" in stop        # the whole state at the end
    assert "function load(payload, resetCamera, sweep)" in vp_js
    load = vp_js[vp_js.index("function load(payload, resetCamera, sweep)"):vp_js.index("function clear()")]
    assert load.index("sweepStop(false);") < load.index("S.mesh.value = payload.state;")
    assert "var animate = !!sweep && sweepOn();" in load
    assert "{ opacity: 0 }" in load and load.index("{ opacity: 0 }") < load.index("v.resetCamera();")
    assert load.index("v.resetCamera();") < load.index("begin();")  # framed on the whole mesh first
    clear = vp_js[vp_js.index("function clear()"):vp_js.index("function setOverlay")]
    assert "sweepStop(false);" in clear
    assert '<vtk-geometry-representation v-if="base" :property="baseProperty">' in vp_js
    # app.js passes the sweep only for a NEW artifact framed anew
    assert "window.MeshforgeViewport.load(payload, resetCamera, resetCamera && changed);" in app_js
    assert "function sameArtifact(a, b)" in app_js
    same = app_js[app_js.index("function sameArtifact(a, b)"):]
    assert "a.label === b.label && a.kind === b.kind" in same[:600]


def test_layer_switch_shows_stress_or_flow_once_both_are_held(app_js: str, index_html: str, styles_css: str) -> None:
    """D112: two buttons over the canvas switch the view between the held
    wall-stress and CFD results (GET /geometry?artifact=stress|cfd, nothing
    computed), once the session holds both; the chosen layer is drawn in from
    the top down with the camera kept."""
    assert 'id="vp-layer-ctl"' in index_html and 'data-layer="stress"' in index_html and 'data-layer="cfd"' in index_html
    assert '<div class="vp-ctl vp-layer" id="vp-layer-ctl" hidden' in index_html  # hidden until both are held
    assert "function showLayer(layer)" in app_js and "function paintLayerButtons()" in app_js
    show = app_js[app_js.index("async function showLayer(layer)"):app_js.index("function paintLayerButtons()")]
    assert "`/geometry?artifact=${encodeURIComponent(layer)}`" in show
    assert "window.MeshforgeViewport.load(payload, false, true);" in show  # camera kept, drawn in
    paint = app_js[app_js.index("function paintLayerButtons()"):app_js.index("// ---- flow section") if "// ---- flow section" in app_js else None]
    assert "ctl.hidden = !(held.stress && held.cfd);" in paint
    assert "vpHeld = st.artifacts || {};" in app_js            # from every /state refresh
    assert "initLayerSwitch();" in app_js
    # the flow is seen inside the wall: the stress result stays as a translucent shell
    assert 'const wall = await fetch("/geometry?artifact=stress");' in show
    assert "window.MeshforgeViewport.setShell(shell);" in show
    vp_js = (_static_dir() / "viewport.js").read_text()
    assert "function setShell(state)" in vp_js and "setShell: setShell," in vp_js
    assert '<vtk-geometry-representation v-if="shell" :property="shellProperty">' in vp_js
    assert "var SHELL_PROPERTY = { color: [0.74, 0.76, 0.8], opacity: 0.22, specular: 0.0 };" in vp_js
    load = vp_js[vp_js.index("function load(payload, resetCamera, sweep)"):vp_js.index("// The layer switch's shell")]
    assert "var base = S.shell && S.shell.value ? null : previous;" in load  # no second ghost of the wall
    refresh = app_js[app_js.index("async function refreshViewport(resetCamera)"):app_js.index("function showPayload")]
    assert "window.MeshforgeViewport.setShell(null);" in refresh          # the current artifact again: no shell


def test_atlas_panel_is_a_template_with_fifteen_regions_six_chips_and_real_rows_only(
    index_html: str, app_js: str, styles_css: str
) -> None:
    """D108 addendum 6: the patient atlas replaces the coverage illustration.
    The body map is exactly the 15 regions of meshforge.atlas.REGIONS (same
    ids, same names, in order), each focusable and labelled; six modality
    chips with All on; every study row comes from GET /atlas (nothing about
    the demo scan is written into the page); the empty state names what
    WOULD list and invents nothing; hover / focus / selected rules exist only
    under body.demo, the class index.html always sets."""
    from meshforge import atlas as atlas_mod

    assert '<template id="atlas-svg">' in index_html
    t0 = index_html.index('<template id="atlas-svg">')
    tpl = index_html[t0:index_html.index("</template>", t0)]
    regions = re.findall(
        r'<path class="atlas-region" data-region="([a-z_]+)" tabindex="0" aria-label="([^"]+)" role="button"'
        r' data-cx="\d+" data-cy="\d+" d="M[^"]+Z"/>',
        tpl,
    )
    assert len(regions) == 15 and tpl.count('class="atlas-region"') == 15
    assert [rid for rid, _ in regions] == [rid for rid, _ in atlas_mod.REGIONS]
    assert dict(regions) == dict(atlas_mod.REGIONS)  # the aria-labels are the catalogue's names
    assert len(atlas_mod.REGIONS) == 15 and len({rid for rid, _ in atlas_mod.REGIONS}) == 15
    # sides are the patient's: every right-side region is drawn left of the figure's
    # centre line (the viewBox's half width), every left-side one right of it
    vb = re.search(r'class="atlas-svg" viewBox="0 0 (\d+) (\d+)"', tpl)
    mid = int(vb.group(1)) / 2
    for rid, cx in re.findall(r'data-region="([a-z_]+)"[^>]*data-cx="(\d+)"', tpl):
        if rid.endswith("_r"):
            assert int(cx) < mid, rid
        elif rid.endswith("_l"):
            assert int(cx) > mid, rid
        else:
            assert int(cx) == mid, rid  # head, chest, torso sit on the centre line
    # every region has its own closed path; the figure is drawn at 1 unit = 1 px
    assert ".atlas .atlas-svg { display: block; width: 170px; height: 330px;" in styles_css
    assert vb.group(1) == "170" and vb.group(2) == "330"
    assert "the patient's right is on the left" in tpl
    chips = re.findall(r'<button type="button" class="atlas-chip( on)?" data-modality="([^"]+)">([^<]+)</button>', tpl)
    assert [m for _, m, _ in chips] == ["All", "CT", "MRI", "X-ray", "US", "Angio"]
    assert [m for _, m, lab in chips if lab != m] == [] and [m for on, m, _ in chips if on] == ["All"]
    assert [m for _, m, _ in chips][1:] == list(atlas_mod.MODALITIES)
    assert 'id="atlas-hover"' in tpl and 'id="atlas-studies"' in tpl and 'id="atlas-scope"' in tpl
    assert "atlas · regions without studies are dim · sides are the patient's" in tpl
    # real rows only: the page and the script know no study, no scan, no patient
    atlas_js = app_js[app_js.index("// ---- patient atlas"):app_js.index("// The result reveal lives in viewport.js")]
    for banned in ("abdomen + pelvis", "s1364", "1.5 mm", "prior", "TotalSegmentator", "date withheld"):
        assert banned not in tpl, banned
        assert banned not in atlas_js, banned
    assert 'apiGet("/atlas")' in app_js
    assert 'loaded.textContent = "loaded ✓";' in app_js and "if (s.loaded) {" in app_js
    assert "would list here" in app_js  # the empty state
    assert "regions without studies are dim · sides are the patient's" in app_js
    # cloned above the twin once, when the page starts (never twice)
    start = app_js[app_js.index("function initDemoLayout()"):app_js.index("async function init()")]
    assert "buildAtlasPanel();" in start
    assert 'if ($("#atlas-panel")) return;' in _function_body(app_js, "function buildAtlasPanel")
    assert "twin.parentElement.insertBefore(node, twin)" in app_js
    # refreshed with the twin on every session change; a no-op until the panel is in the DOM
    assert app_js.count("await refreshAtlas();") == app_js.count("await refreshTwin();") >= 4
    refresh = app_js[app_js.index("async function refreshAtlas()"):]
    assert 'if (!$("#atlas-panel")) return;' in refresh[:120]
    # states: has / none by data attribute; hover, focus and selected under body.demo only
    assert '.atlas .atlas-region[data-has="1"]' in styles_css
    assert "body.demo .atlas .atlas-region:hover" in styles_css
    assert "body.demo .atlas .atlas-region:focus-visible" in styles_css
    assert "body.demo .atlas .atlas-region.selected { stroke: #ff5d55; stroke-width: 2px;" in styles_css
    for line in styles_css.splitlines():
        if ".atlas-region" in line and (":hover" in line or ".selected" in line or ":focus-visible" in line):
            assert line.startswith("body.demo "), line
    # keyboard: Enter / Space select, Esc clears; click again clears
    assert 'e.key === "Enter" || e.key === " "' in app_js
    assert 'e.key === "Escape" && atlasSelected !== null' in app_js
    assert "selectAtlasRegion(atlasSelected === id ? null : id)" in app_js


def test_demo_mode_claims_nothing(index_html: str, app_js: str) -> None:
    demo_text = index_html[index_html.index('<body class="demo">'):]
    for banned in ("rupture probability", "time to rupture", "diagnos", "predict"):
        # "not a diagnosis" is allowed in the twin foot; the demo additions add no claim
        assert banned not in app_js[app_js.index("const DEMO_STEPS"):app_js.index("async function init()")]
    assert "regions without studies are dim" in demo_text  # the atlas: real studies only, nothing invented
    assert "no patient loaded" in demo_text


def test_viewport_draws_the_ghost_outline_behind_the_mesh(app_js: str) -> None:
    """GET /geometry's ``ghost`` list (the whole AI outline, and the patient's
    own heart from the same run) is drawn as faint translucent representations
    behind the mesh and the QC overlay, one per entry, each in the fixed look
    of its kind (the aorta grey, the heart a warmer tint at the same opacity —
    never coloured by data); load() takes the list from the main payload only,
    clear() drops it."""
    vp_js = (_static_dir() / "viewport.js").read_text()
    assert "var GHOST_PROPERTY = { color: [0.78, 0.8, 0.84], opacity: 0.16" in vp_js
    assert "var HEART_GHOST_PROPERTY = { color: [0.88, 0.74, 0.72], opacity: 0.16" in vp_js
    assert "var GHOST_PROPERTIES = { aorta: GHOST_PROPERTY, heart: HEART_GHOST_PROPERTY };" in vp_js
    assert '<vtk-geometry-representation v-for="g in ghosts" :key="g.kind" :property="g.property">' in vp_js
    assert '<vtk-mesh :state="g.state"></vtk-mesh>' in vp_js
    assert vp_js.index('v-for="g in ghosts"') > vp_js.index('v-if="overlay"')  # drawn after, behind by opacity
    assert 'if (S.ghosts && "ghost" in payload) S.ghosts.value = ghostList(payload.ghost);' in vp_js
    assert "property: GHOST_PROPERTIES[kind] || GHOST_PROPERTY" in vp_js
    assert "if (S.ready && S.ghosts) S.ghosts.value = [];" in vp_js  # clear()
    assert "setGhosts: setGhosts," in vp_js
    assert "color-map-preset" not in vp_js[vp_js.index('v-for="g in ghosts"'):vp_js.index("</vtk-view>")]



def test_viewport_turntable_spins_only_in_demo_mode_after_idle(app_js: str) -> None:
    """D111: in Demo mode a view left alone for 3 s spins slowly about the
    camera's up axis (one turn in 45 s) and stops on any input to the canvas
    (press, scroll, touch, key; hovering does not stop it). Pure camera motion
    through vue_vtk's getCamera/setCamera; never outside Demo mode, never
    while a pick is armed, never without a mesh, never under reduced motion,
    paused while the tab is hidden. The page is always in Demo mode (the
    static body class), so mounting the viewport arms it; app.js has no switch
    to re-arm or stop it."""
    vp_js = (_static_dir() / "viewport.js").read_text()
    assert "var IDLE_MS = 3000;" in vp_js
    assert "var SPIN_DEG_PER_S = 8;" in vp_js
    assert 'return document.body.classList.contains("demo");' in vp_js  # demoOn()
    assert '"(prefers-reduced-motion: reduce)"' in vp_js
    spin = vp_js[vp_js.index("function spinStart()"):vp_js.index("function spinStop()")]
    assert "!demoOn() || reducedMotion()" in spin
    assert "if (!(S.mesh && S.mesh.value)) return;" in spin
    assert "if (S.pickCb) return;" in spin
    step = vp_js[vp_js.index("function spinStep(ts)"):vp_js.index("function spinStart()")]
    assert "!hasMesh || !demoOn()" in step and "spinStop();" in step
    assert "Math.min(ts - R.last, 100)" in step  # a stalled tab does not jump
    assert "view.setCamera({ position: [fp[0] + rx, fp[1] + ry, fp[2] + rz], focalPoint: fp, viewUp: up });" in vp_js
    assert 'var evs = ["pointerdown", "wheel", "touchstart", "keydown"];' in vp_js
    assert "pointermove" not in vp_js[vp_js.index("function watchIdle"):vp_js.index("// ---- flow section")]
    assert 'document.addEventListener("visibilitychange"' in vp_js
    assert "watchIdle(host); // D111" in vp_js
    arm = vp_js[vp_js.index("function armPick(cb, onMiss)"):vp_js.index("function onPick(e)")]
    assert arm.count("spinStop();") == 1 and arm.count("idleDisarm();") == 1 and arm.count("idleArm();") == 1
    assert "v.resetCamera();\n          idleArm();" in vp_js  # a newly framed artifact starts the clock
    clear = vp_js[vp_js.index("function clear()"):vp_js.index("function setOverlay")]
    assert "spinStop();" in clear and "idleDisarm();" in clear
    assert "wake: wake," in vp_js
    assert "MeshforgeViewport.wake" not in app_js  # no switch: nothing to re-arm
