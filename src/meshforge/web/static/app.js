// meshforge Workbench frontend: the demo page.
//
// Rule 4: NO meshing/geometry/solve logic here. This module only calls the
// local JSON API and renders its responses. Endpoints used:
//   GET  /schema  /catalog  /journal  /state  /twin  /atlas  /cfd/presets
//   GET  /geometry  /geometry/frame/<i>
//   POST /step/<name>
//   POST /workfile/save  /workfile/load  /session/reset
//   POST /cfd/pick  /cfd/inflow   (D80 flow-ends picker, read-only)
// Plain ES module, no build step.

const $ = (sel) => document.querySelector(sel);

// ---- tiny fetch helper -------------------------------------------------
async function apiGet(path) {
  const res = await fetch(path);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(detailOf(data, res));
  return data;
}
async function apiPost(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(detailOf(data, res));
  return data;
}
function detailOf(data, res) {
  // FastAPI: 400 -> {detail: "<message>"}; 422 -> {detail: [ {..} ]}.
  if (data && data.detail !== undefined) {
    return typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
  }
  return `HTTP ${res.status} ${res.statusText}`;
}

// ---- on-screen paths (M6) ---------------------------------------------
// The Workbench is recorded for the demo: a local folder (and the user name
// in it) must not reach the screen. Display only — the journal, work files
// and requests keep the real paths. An absolute path value is shown by its
// file name; any other text has its /Users/<name> (or /home/<name>) prefix
// shown as "~".
const ABS_PATH = /^(\/|~\/|[A-Za-z]:[\\/])/;
const HOME_PREFIX = /\/(?:Users|home)\/[^\/\s"']+/g;
function shownText(text) {
  return String(text).replace(HOME_PREFIX, "~");
}
function shownJson(obj, indent) {
  return shownText(JSON.stringify(obj, (k, v) =>
    typeof v === "string" && ABS_PATH.test(v) ? shortPath(v) : v, indent));
}

// ---- log + summary -----------------------------------------------------
function log(text, isErr) {
  const el = $("#log");
  const line = document.createElement("span");
  if (isErr) line.className = "err";
  line.textContent = (isErr ? "ERROR: " : "") + shownText(text).replace(/\s+$/, "") + "\n";
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}
function showSummary(result) {
  if (result && result.summary !== undefined) {
    $("#summary-view").textContent = shownJson(result.summary, 2);
  }
}

// After every action: surface the result text, then refresh journal + state.
async function afterStep(result) {
  if (result && result.text) log(result.text);
  showSummary(result);
  await refreshJournal();
  await refreshState();
  await refreshTwin();
  await refreshAtlas();
  viewportField = null; // a new artifact -> let the server pick the default field
  await refreshViewport(true);
  // Any new mesh or source invalidates a picked inflow end; a cfd run keeps it
  // (so the next run only needs a new speed).
  if (!result || result.step !== "cfd") resetFlowPicker();
  await refreshPresets();
}
function onError(err) {
  log(String(err && err.message ? err.message : err), true);
}

// ---- viewport (V35 increment 3, pure client-side vtk.js) ---------------
let viewportUp = false; // did the vtk.js canvas mount?
let viewportField = null; // current color-by field (null = server default)
let vpFrames = 0; // CFD frame count of the current artifact
let vpTimes = []; // per-frame times
let vpPlaying = null; // setInterval handle while animating
let vpOverlay = null; // cached QC-overlay state so the toggle needs no refetch

function initViewport() {
  viewportUp = !!(
    window.MeshforgeViewport &&
    window.MeshforgeViewport.available() &&
    window.MeshforgeViewport.mount("vtk-canvas")
  );
  if (!viewportUp) {
    $("#vp-fallback").hidden = false;
    return;
  }
  $("#vp-field").addEventListener("change", (e) => {
    viewportField = e.target.value || null;
    refreshViewport(false).catch(onError);
  });
  $("#vp-frame").addEventListener("input", (e) => {
    stopPlay();
    loadFrame(Number(e.target.value)).catch(onError);
  });
  $("#vp-play").addEventListener("click", togglePlay);
  $("#vp-overlay").addEventListener("change", (e) => {
    window.MeshforgeViewport.setOverlay(e.target.checked ? vpOverlay : null);
  });
  $("#vp-section").addEventListener("change", (e) => {
    window.MeshforgeViewport.setSectionVisible(e.target.checked);
    $("#vp-wall-ctl").hidden = !e.target.checked;
    applyColourRange();
  });
  $("#vp-wall").addEventListener("change", (e) => {
    window.MeshforgeViewport.setWallVisible(e.target.checked);
  });
  $("#vp-fit").addEventListener("change", () => applyColourRange());
}

// Fetch the current artifact's geometry and (re)paint the canvas. resetCamera
// frames a NEW artifact; a field swap keeps the camera put. A NEW artifact
// (not the same one served again after a journal-only step) is drawn in from
// the top down (D112, viewport.js).
async function refreshViewport(resetCamera) {
  if (!viewportUp) return;
  vpLayer = null; // the session's current artifact again
  window.MeshforgeViewport.setShell(null);
  const url = "/geometry" + (viewportField ? `?field=${encodeURIComponent(viewportField)}` : "");
  const res = await fetch(url);
  if (res.status === 204) return clearViewport(); // nothing rendered yet
  if (!res.ok) return; // viewport hiccup — don't spam the log
  const payload = await res.json();
  const changed = !sameArtifact(vpPayload, payload);
  window.MeshforgeViewport.load(payload, resetCamera, resetCamera && changed);
  showPayload(payload);
}
// Paint the toolbar for a payload (fields, colour bar, frames, overlay, section).
function showPayload(payload) {
  populateFields(payload.fields || [], payload.field);
  updateColorbar(payload);
  setupFrames(payload);
  setupOverlay(payload);
  setupSection(payload);
  vpPayload = payload;
  applyColourRange();
  paintLayerButtons();
}
// The same artifact served again (a journal-only step bumped the revision):
// same label, kind and point count. Its reveal is not replayed.
function sameArtifact(a, b) {
  if (!a || !b) return false;
  const pts = (p) => (p.state && p.state.mesh && p.state.mesh.points && p.state.mesh.points.shape) || null;
  return a.label === b.label && a.kind === b.kind && JSON.stringify(pts(a)) === JSON.stringify(pts(b));
}

// Colour range: the full field range, or (fit colours) the flow inside the
// section — both computed by the server; the page only chooses which.
let vpPayload = null;
function applyColourRange() {
  if (!vpPayload) return;
  const fit = $("#vp-fit").checked && vpPayload.section_range && $("#vp-section").checked;
  const range = fit ? vpPayload.section_range : vpPayload.data_range;
  if (!range) return;
  window.MeshforgeViewport.setRange(range);
  updateColorbar({ field: vpPayload.field, data_range: range });
}

function clearViewport() {
  if (viewportUp) window.MeshforgeViewport.clear();
  $("#vp-field-ctl").hidden = true;
  $("#vp-colorbar").hidden = true;
  $("#vp-anim").hidden = true;
  $("#vp-overlay-ctl").hidden = true;
  $("#vp-section-ctl").hidden = true;
  $("#vp-wall-ctl").hidden = true;
  $("#vp-fit-ctl").hidden = true;
  vpPayload = null;
  vpOverlay = null;
  vpLayer = null;
  paintLayerButtons();
  stopPlay();
}

// ---- layer switch (D112) --------------------------------------------------
// Once the session holds a wall-stress result and a CFD result, two buttons
// over the canvas switch the view between them (GET
// /geometry?artifact=stress|cfd — the server serialises a result it already
// holds; nothing is computed). The camera stays put and the chosen layer is
// drawn in from the top down. null = the session's current artifact.
let vpLayer = null;
let vpHeld = {}; // GET /state artifacts: which results the session holds
const LAYER_LABEL = { stress: "wall stress (investigational)", cfd: "cfd results" };
function initLayerSwitch() {
  for (const btn of document.querySelectorAll("#vp-layer-ctl .vp-layer-btn")) {
    btn.addEventListener("click", () => showLayer(btn.dataset.layer).catch(onError));
  }
}
async function showLayer(layer) {
  if (!viewportUp || !layer) return;
  const res = await fetch(`/geometry?artifact=${encodeURIComponent(layer)}`);
  if (!res.ok) {
    log(`layer ${layer}: ${res.status === 404 ? "no such result in the session yet" : "unavailable"}`, true);
    return;
  }
  const payload = await res.json();
  // the flow is seen inside the wall: the stress result stays as a
  // translucent shell around it (the whole-aorta ghosts stay as always)
  let shell = null;
  if (layer === "cfd") {
    const wall = await fetch("/geometry?artifact=stress");
    if (wall.ok) shell = (await wall.json()).state || null;
  }
  vpLayer = layer;
  viewportField = null;
  window.MeshforgeViewport.setShell(shell);
  window.MeshforgeViewport.load(payload, false, true);
  showPayload(payload);
}
function paintLayerButtons() {
  const ctl = $("#vp-layer-ctl");
  if (!ctl) return;
  const held = { stress: Boolean(vpHeld.aaa_stress), cfd: Boolean(vpHeld.cfd) };
  ctl.hidden = !(held.stress && held.cfd);
  const shown = vpLayer || (vpPayload && Object.keys(LAYER_LABEL).find((k) => LAYER_LABEL[k] === vpPayload.label)) || null;
  for (const btn of ctl.querySelectorAll(".vp-layer-btn")) {
    btn.disabled = !held[btn.dataset.layer];
    btn.classList.toggle("on", btn.dataset.layer === shown);
    btn.setAttribute("aria-pressed", btn.dataset.layer === shown ? "true" : "false");
  }
}

// Flow section (D82): a CFD result carries a section along the vessel
// centreline; the toggle shows it (wall translucent) or the plain wall.
function setupSection(payload) {
  $("#vp-section-ctl").hidden = !payload.section;
  $("#vp-fit-ctl").hidden = !payload.section_range;
  $("#vp-wall-ctl").hidden = !(payload.section && $("#vp-section").checked);
  window.MeshforgeViewport.setWallVisible($("#vp-wall").checked);
  window.MeshforgeViewport.setSectionVisible($("#vp-section").checked);
}

// Overlay lines a step sends with its artifact (the crop step's segment
// centreline): cache the serialized polylines and show the toggle only when
// the artifact carries one.
function setupOverlay(payload) {
  const ctl = $("#vp-overlay-ctl");
  vpOverlay = (payload.overlay && payload.overlay.state) || null;
  if (!vpOverlay) {
    ctl.hidden = true;
    window.MeshforgeViewport.setOverlay(null);
    return;
  }
  $("#vp-overlay-name").textContent = (payload.overlay && payload.overlay.label) || "overlay";
  ctl.hidden = false;
  window.MeshforgeViewport.setOverlay($("#vp-overlay").checked ? vpOverlay : null);
}

function populateFields(fields, current) {
  const sel = $("#vp-field");
  const ctl = $("#vp-field-ctl");
  if (!fields.length) {
    ctl.hidden = true;
    sel.innerHTML = "";
    sel.dataset.fields = "";
    return;
  }
  const want = fields.join("");
  if (sel.dataset.fields !== want) {
    sel.innerHTML = "";
    for (const f of fields) {
      const o = document.createElement("option");
      o.value = f;
      o.textContent = f;
      sel.appendChild(o);
    }
    sel.dataset.fields = want;
  }
  if (current) sel.value = current;
  ctl.hidden = false;
}

function updateColorbar(payload) {
  const cb = $("#vp-colorbar");
  if (!payload.field || !payload.data_range) {
    cb.hidden = true;
    return;
  }
  $("#vp-cb-min").textContent = fmtNum(payload.data_range[0]);
  $("#vp-cb-max").textContent = fmtNum(payload.data_range[1]);
  $("#vp-cb-name").textContent = payload.field;
  cb.hidden = false;
}
function fmtNum(x) {
  if (x === null || x === undefined) return "";
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e4)) return x.toExponential(2);
  return String(Math.round(x * 1000) / 1000);
}

function setupFrames(payload) {
  vpFrames = payload.n_frames || 0;
  vpTimes = payload.times || [];
  const anim = $("#vp-anim");
  if (vpFrames > 1) {
    const slider = $("#vp-frame");
    slider.max = String(vpFrames - 1);
    slider.value = "0";
    updateTimeLabel(0);
    anim.hidden = false;
  } else {
    anim.hidden = true;
    stopPlay();
  }
}
function updateTimeLabel(i) {
  const t = vpTimes[i];
  const tt = t !== undefined ? `t=${fmtNum(t)}` : "";
  $("#vp-time").textContent = `${tt}  (${i + 1}/${vpFrames})`;
}

// Load one CFD frame; keep the camera so the animation doesn't jump.
async function loadFrame(i) {
  if (!viewportUp || vpFrames < 1) return;
  const url = `/geometry/frame/${i}` + (viewportField ? `?field=${encodeURIComponent(viewportField)}` : "");
  const res = await fetch(url);
  if (!res.ok) return;
  const payload = await res.json();
  window.MeshforgeViewport.load(payload, false);
  $("#vp-frame").value = String(i);
  updateTimeLabel(i);
}

function togglePlay() {
  if (vpPlaying) return stopPlay();
  if (vpFrames < 2) return;
  $("#vp-play").textContent = "⏸";
  vpPlaying = setInterval(() => {
    const next = (Number($("#vp-frame").value) + 1) % vpFrames;
    loadFrame(next).catch(onError);
  }, 250);
}
function stopPlay() {
  if (vpPlaying) {
    clearInterval(vpPlaying);
    vpPlaying = null;
  }
  $("#vp-play").textContent = "▶";
}

// ---- twin card (M6) -----------------------------------------------------
// The copilot report's header as a live panel. GET /twin builds it from the
// session's journaled step outputs on the server; this only paints it. Every
// string goes in through textContent (no HTML from the server is trusted).
async function refreshTwin() {
  let twin;
  try {
    twin = await apiGet("/twin");
  } catch (err) {
    return; // a twin hiccup must not spam the log; the next refresh retries
  }
  renderTwin(twin);
}

function renderTwin(twin) {
  noteDemoTwin(twin); // the Mesh wall tab's gate mark
  $("#twin").dataset.lit = twin.lit ? "1" : "0";
  $("#twin-scan").textContent = twin.scan || "no scan yet";
  const mods = $("#twin-modules");
  mods.innerHTML = "";
  for (const m of twin.modules || []) {
    const li = document.createElement("li");
    li.className = "mod" + (m.lit ? " lit" : "");
    li.textContent = `${m.lit ? "●" : "○"} ${m.name} · ${m.state}`;
    mods.appendChild(li);
  }
  const list = $("#twin-slots");
  list.innerHTML = "";
  for (const s of twin.slots || []) {
    const li = document.createElement("li");
    li.className = `slot ${s.state}`;
    li.dataset.slot = s.key;
    const dot = document.createElement("span");
    dot.className = "slot-dot";
    dot.textContent = s.state === "filled" ? "●" : s.state === "next" ? "◌" : "○";
    const body = document.createElement("div");
    body.className = "slot-body";
    const head = document.createElement("div");
    head.className = "slot-head";
    const label = document.createElement("b");
    label.textContent = s.label;
    head.appendChild(label);
    if (s.note) {
      const note = document.createElement("span");
      note.className = "slot-note";
      note.textContent = s.note;
      head.appendChild(note);
    }
    const detail = document.createElement("span");
    detail.className = "slot-detail";
    detail.textContent = s.detail;
    body.append(head, detail);
    for (const v of s.values || []) {
      const val = document.createElement("span");
      val.className = "slot-value";
      val.textContent = v;
      body.appendChild(val);
    }
    li.append(dot, body);
    list.appendChild(li);
  }
}

// ---- journal + state ---------------------------------------------------
// The journal is what marks a tab done (noteDemoJournal); the page shows no
// history table.
async function refreshJournal() {
  const { journal } = await apiGet("/journal");
  noteDemoJournal(journal);
}
let lastJournalLen = 0; // last-seen journal length — #btn-save's disabled rule

async function refreshState() {
  const st = await apiGet("/state");
  lastRev = st.rev; // this client is now current with the session
  $("#state-input").textContent = st.input_path ? shortPath(st.input_path) : "no source";
  vpHeld = st.artifacts || {};
  paintLayerButtons();
  const box = $("#state-artifacts");
  box.innerHTML = "";
  for (const [name, present] of Object.entries(st.artifacts || {})) {
    const chip = document.createElement("span");
    chip.className = "chip" + (present ? " on" : "");
    chip.textContent = (present ? "✓ " : "· ") + name;
    box.appendChild(chip);
  }
  // "Save work file" is enabled only on a non-empty journal.
  lastJournalLen = st.journal_len;
  $("#btn-save").disabled = !(lastJournalLen > 0);
}
function shortPath(p) {
  const parts = String(p).split("/");
  return parts[parts.length - 1] || p;
}

// ---- auto-refresh ------------------------------------------------------
// The server bumps a monotonic session revision on every state-changing call
// (steps, reset, work-file load — from ANY client). Polling it keeps this
// window current when the session changes underneath it, e.g. the copilot
// (or another browser tab) driving the same server; no manual reload needed.
let lastRev = null; // last /state rev this client rendered
let pollBusy = false;

function startAutoRefresh() {
  setInterval(async () => {
    if (pollBusy) return;
    pollBusy = true;
    try {
      const st = await apiGet("/state");
      updateBusyLine(st);
      if (lastRev !== null && st.rev !== lastRev) {
        log("session changed — resyncing view");
        await refreshJournal();
        await refreshState(); // also updates lastRev
        await refreshTwin();
        await refreshAtlas();
        viewportField = null;
        await refreshViewport(true);
        resetFlowPicker();
        await refreshPresets();
      } else if (lastRev === null) {
        lastRev = st.rev;
      }
    } catch (err) {
      // server briefly unreachable — keep polling quietly
    } finally {
      pollBusy = false;
    }
  }, 2000);
}

// ---- busy state + inline panel errors (FIX 5 / FIX 7) -------------------
// A long step (segment / aaa_stress can run minutes) must disable its own
// button and show a running state so a second click can't fire a concurrent
// duplicate request; re-enabling always happens in a `finally` so an error
// can never strand the button disabled (rule 6: fail loudly, not fail-shut).
async function withBusy(btn, label, fn, restoreDisabled) {
  const prevText = btn.textContent;
  btn.disabled = true;
  btn.classList.add("busy");
  btn.textContent = label;
  try {
    return await fn();
  } finally {
    btn.textContent = prevText;
    btn.classList.remove("busy");
    // Restore the STATE-DERIVED disabled value (e.g. #btn-save tracks
    // journal_len via refreshState) — never a blind `false`, which would
    // fight whatever the last state refresh already decided.
    btn.disabled = typeof restoreDisabled === "function" ? restoreDisabled() : !!restoreDisabled;
  }
}

// Errors surface next to the button that caused them, not only in the
// far-right #log (~1000px away). Cleared on that panel's next successful run.
function panelErrorFor(box) {
  let el = box.querySelector(":scope > .panel-error");
  if (!el) {
    el = document.createElement("p");
    el.className = "panel-error";
    el.hidden = true;
    box.appendChild(el);
  }
  return el;
}
function setError(el, err) {
  el.textContent = String(err && err.message ? err.message : err);
  el.hidden = false;
}
function clearErrorEl(el) {
  el.hidden = true;
}

// ---- build the step panels from GET /schema ----------------------------
function buildStepPanel(step, knobs) {
  const box = document.createElement("details");
  box.className = "step-box";
  box.dataset.step = step;
  // Closed at start: six lines, the whole path at a glance. A click still
  // opens a tab's knobs.
  box.open = false;
  const summary = document.createElement("summary");
  summary.textContent = step;
  box.appendChild(summary);
  if (step === "cfd") box.appendChild(buildFlowPicker(box));

  for (const [knob, spec] of Object.entries(knobs)) {
    const label = document.createElement("label");
    label.className = "knob";
    const name = document.createElement("span");
    name.className = "knob-name";
    name.textContent = knob;
    label.appendChild(name);
    const input = makeInput(step, knob, spec);
    label.appendChild(input);
    box.appendChild(label);
    if (spec.help) {
      const help = document.createElement("p");
      help.className = "knob-help";
      help.textContent = spec.help;
      box.appendChild(help);
    }
  }

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "run";
  btn.textContent = "Run " + step;
  btn.addEventListener("click", () => {
    withBusy(btn, "Running…", async () => {
      try {
        await runStep(step, box);
        clearErrorEl(panelErrorFor(box));
      } catch (err) {
        onError(err);
        setError(panelErrorFor(box), err);
      }
    }).catch(onError);
  });
  box.appendChild(btn);
  return box;
}

function makeInput(step, knob, spec) {
  const input = document.createElement("input");
  input.dataset.knob = knob;
  input.dataset.type = spec.type;
  if (spec.type === "bool") {
    input.type = "checkbox";
    input.checked = spec.default === true;
  } else if (spec.type === "int" || spec.type === "float") {
    input.type = "number";
    input.step = spec.type === "int" ? "1" : "any";
    if (spec.default !== null && spec.default !== undefined) input.value = spec.default;
  } else {
    input.type = "text";
    if (spec.default !== null && spec.default !== undefined) input.value = spec.default;
  }
  return input;
}

// Collect a step's knobs into a POST body. Empty text/number inputs are
// omitted (drop_unset parity: unset -> the compute step's own default);
// checkboxes always send their boolean.
function gatherKnobs(box) {
  const body = {};
  for (const input of box.querySelectorAll("input[data-knob], select[data-knob]")) {
    const knob = input.dataset.knob;
    const type = input.dataset.type;
    if (type === "bool") {
      body[knob] = input.checked;
    } else if (input.value !== "") {
      body[knob] = type === "int" || type === "float" ? Number(input.value) : input.value;
    }
  }
  return body;
}

async function runStep(step, box) {
  const result = await apiPost(`/step/${step}`, gatherKnobs(box));
  await afterStep(result);
}

// ---- cfd flow-ends picker (D80) ----------------------------------------
// Click near an end of the vessel in the viewport -> the server says which
// flat end cap that is (inflow) and which is the other (outflow). The picker
// then only WRITES the cfd panel's own knobs in their plain scalar form —
//   inlet = <inflow>=<speed>   outlet = <outflow>=0   nu = <preset>
// so the recorded step and a replay are identical to the click.
// Q and Re come from the server (POST /cfd/inflow); nothing is computed here.
let fpState = null; // {inflow, outflow} patch geometry from POST /cfd/pick
let fpPresets = null; // GET /cfd/presets payload, or null (unit unknown)
let fpPreviewTimer = null;

function buildFlowPicker(box) {
  const wrap = document.createElement("div");
  wrap.className = "flow-picker";
  wrap.innerHTML = `
    <div class="fp-head">flow ends</div>
    <div class="fp-row">
      <button type="button" id="fp-pick" class="run">Pick inflow end</button>
      <button type="button" id="fp-swap" class="run" disabled title="exchange inflow and outflow">Swap</button>
    </div>
    <p id="fp-status" class="fp-status">run lumen (or volmesh), then click Pick and click the vessel near its inflow end</p>
    <label class="knob fp-speed"><span class="knob-name">speed</span>
      <input type="number" id="fp-speed" step="any" min="0" placeholder="inflow speed" />
      <span id="fp-speed-unit" class="fp-unit"></span></label>
    <label class="knob"><span class="knob-name">speed preset</span>
      <select id="fp-speed-preset"></select></label>
    <label class="knob"><span class="knob-name">fluid</span>
      <select id="fp-nu-preset"></select></label>
    <p id="fp-presets-note" class="fp-note" hidden></p>
    <p id="fp-preview" class="fp-preview"></p>`;
  const knobInput = (name) => box.querySelector(`input[data-knob="${name}"]`);
  wrap.querySelector("#fp-pick").addEventListener("click", () => togglePick(wrap));
  wrap.querySelector("#fp-swap").addEventListener("click", () => {
    if (!fpState) return;
    fpState = { inflow: fpState.outflow, outflow: fpState.inflow };
    applyFlowEnds(knobInput);
  });
  wrap.querySelector("#fp-speed").addEventListener("input", () => applyFlowEnds(knobInput));
  wrap.querySelector("#fp-speed-preset").addEventListener("change", (e) => {
    const p = fpPresets && fpPresets.speed.find((x) => x.id === e.target.value);
    if (!p) return;
    wrap.querySelector("#fp-speed").value = p.value;
    applyFlowEnds(knobInput);
  });
  wrap.querySelector("#fp-nu-preset").addEventListener("change", (e) => {
    const p = fpPresets && fpPresets.nu.find((x) => x.id === e.target.value);
    if (!p) return;
    knobInput("nu").value = p.value;
    applyFlowEnds(knobInput);
  });
  return wrap;
}

function togglePick(wrap) {
  const btn = wrap.querySelector("#fp-pick");
  const status = wrap.querySelector("#fp-status");
  if (btn.dataset.armed === "1") { // second click cancels
    window.MeshforgeViewport.disarmPick();
    btn.dataset.armed = "";
    btn.textContent = "Pick inflow end";
    status.textContent = "pick cancelled";
    return;
  }
  if (!viewportUp) return log("pick: the 3-D viewport is unavailable", true);
  const armed = window.MeshforgeViewport.armPick(
    (point) => onFlowPicked(wrap, point).catch((err) => {
      status.textContent = String(err.message || err);
      status.classList.add("err");
    }),
    () => { status.textContent = "nothing under the cursor — click ON the vessel, near its inflow end"; },
  );
  if (!armed) return;
  btn.dataset.armed = "1";
  btn.textContent = "Cancel pick";
  status.classList.remove("err");
  status.textContent = "click the vessel near its INFLOW end (drag still rotates)";
}

async function onFlowPicked(wrap, point) {
  const btn = wrap.querySelector("#fp-pick");
  btn.dataset.armed = "";
  btn.textContent = "Pick inflow end";
  const ends = await apiPost("/cfd/pick", { point });
  fpState = { inflow: ends.inflow, outflow: ends.outflow };
  const box = wrap.closest(".step-box"); // by class: the panel's tag is not a contract
  applyFlowEnds((name) => box.querySelector(`input[data-knob="${name}"]`));
}

// Write the cfd knobs from the picker state, then refresh the Q/Re preview.
function applyFlowEnds(knobInput) {
  const status = $("#fp-status");
  $("#fp-swap").disabled = !fpState;
  if (!fpState) return;
  status.classList.remove("err");
  const a = fpState.inflow;
  status.textContent =
    `inflow: '${a.name}' (area ${fmtNum(a.area)}, D_eq ${fmtNum(a.d_eq)})` +
    `  →  outflow: '${fpState.outflow.name}' (p = 0)`;
  const speed = Number($("#fp-speed").value);
  if (speed > 0) {
    knobInput("inlet").value = `${a.name}=${speed}`;
    knobInput("outlet").value = `${fpState.outflow.name}=0`;
  }
  clearTimeout(fpPreviewTimer);
  fpPreviewTimer = setTimeout(() => previewInflow(knobInput).catch(() => {}), 250);
}

async function previewInflow(knobInput) {
  const out = $("#fp-preview");
  const speed = Number($("#fp-speed").value);
  const nu = Number(knobInput("nu").value);
  if (!fpState || !(speed > 0) || !(nu > 0)) {
    out.textContent = "";
    return;
  }
  try {
    const rep = await apiPost("/cfd/inflow", { patch: fpState.inflow.name, speed, nu });
    out.classList.remove("err");
    out.textContent = (rep.notes || []).join("\n");
  } catch (err) {
    out.classList.add("err");
    out.textContent = String(err.message || err);
  }
}

function resetFlowPicker() {
  fpState = null;
  if (viewportUp) window.MeshforgeViewport.disarmPick();
  const btn = $("#fp-pick");
  if (!btn) return;
  btn.dataset.armed = "";
  btn.textContent = "Pick inflow end";
  $("#fp-swap").disabled = true;
  $("#fp-preview").textContent = "";
  const status = $("#fp-status");
  status.classList.remove("err");
  status.textContent = "run lumen (or volmesh), then click Pick and click the vessel near its inflow end";
}

// Presets follow the SOURCE's length unit; an unknown unit is refused by the
// server and the dropdowns say why instead of offering wrong-unit numbers.
async function refreshPresets() {
  const speedSel = $("#fp-speed-preset");
  const nuSel = $("#fp-nu-preset");
  if (!speedSel || !nuSel) return;
  const note = $("#fp-presets-note");
  const fill = (sel, items, unit) => {
    sel.innerHTML = "";
    const none = document.createElement("option");
    none.value = "";
    none.textContent = items.length ? "— choose —" : "— unavailable —";
    sel.appendChild(none);
    for (const p of items) {
      const o = document.createElement("option");
      o.value = p.id;
      o.textContent = `${p.label}: ${fmtNum(p.value)} ${unit || p.unit}`;
      sel.appendChild(o);
    }
    sel.disabled = !items.length;
  };
  try {
    fpPresets = await apiGet("/cfd/presets");
    fill(speedSel, fpPresets.speed);
    fill(nuSel, fpPresets.nu);
    $("#fp-speed-unit").textContent = `${fpPresets.length_unit}/${fpPresets.time_unit}`;
    note.hidden = true;
  } catch (err) {
    fpPresets = null;
    fill(speedSel, []);
    fill(nuSel, []);
    $("#fp-speed-unit").textContent = "model units/s";
    note.textContent = "presets: " + String(err.message || err);
    note.hidden = false;
  }
}

// ---- work-file actions ---------------------------------------------------
async function saveWorkfile() {
  const path = $("#save-path").value.trim();
  if (!path) throw new Error("save work file: enter a path first");
  const { path: written } = await apiPost("/workfile/save", { path });
  log(`work file saved -> ${written}`);
}
async function loadWorkfile() {
  const path = $("#wload-path").value.trim();
  if (!path) throw new Error("load work file: enter a path first");
  const replay = $("#wload-replay").checked;
  const res = await apiPost("/workfile/load", { path, replay });
  for (const r of res.results || []) log(r.text || `${r.step}: ok`);
  if (res.results && res.results.length) showSummary(res.results[res.results.length - 1]);
  await refreshJournal();
  await refreshState();
  await refreshTwin();
  await refreshAtlas();
  viewportField = null;
  await refreshViewport(true);
  resetFlowPicker();
  await refreshPresets();
}
async function resetSession() {
  await apiPost("/session/reset", {});
  $("#summary-view").textContent = "run a step to see its summary";
  log("session reset");
  viewportField = null;
  clearViewport();
  resetFlowPicker();
  await refreshJournal();
  await refreshState();
  await refreshTwin();
  await refreshAtlas();
  await refreshPresets();
}

// ---- the demo layout (D108) -----------------------------------------------
// The page is always the demo: the aneurysm path's six steps as numbered tabs
// on the left, in the order the copilot runs them, each marked waiting /
// running / done (or failed, when the wall gate refused the mesh); a result
// drawn in from the top down in the viewport (D112, viewport.js); the running
// step with its elapsed seconds; the stress / flow layer switch over the
// canvas once both results exist (D112); and the patient atlas above the
// twin. index.html sets <body class="demo"> once; styles.css lays the tabs
// out under that class, and viewport.js reads it for the idle turntable and
// the reveal. Nothing here calls a compute route or changes what any step
// does (rule 4): it only shows the same results.
// [step, tab title, one-line hint] — the step's own panel (its knobs, its Run
// button) is the tab; the hint is for a viewer who does not know the step.
const DEMO_STEPS = [
  ["segment", "Load scan", "CT in · AI outline of the aorta"],
  ["crop", "Crop aneurysm", "neck → sac → above the bifurcation"],
  ["volmesh", "Mesh wall", "1.5 mm hex wall · quality gate"],
  ["aaa_stress", "Wall stress · FEA", "finite elements at 140 / 120 mmHg"],
  ["tissue_index", "Strength index", "stress against lab strength · investigational"],
  ["cfd", "Blood flow · CFD", "steady resting flow in the lumen"],
];
// Steps that run inside a tab's turn and have no tab of their own (the sac
// recovery runs right after the AI outline), and journal records that fill a
// tab without being its step (the copilot's pre-computed flow metrics).
const DEMO_STEP_OF = { sac: "segment", flow_metrics: "cfd" };
const DEMO_VERBS = {
  segment: "contouring the aorta (AI outline)",
  sac: "recovering the sac from the CT",
  crop: "cropping the aneurysm",
  volmesh: "meshing the wall",
  aaa_stress: "solving the wall stress (finite elements)",
  tissue_index: "computing the investigational index",
  cfd: "solving the flow (CFD)",
};
// Tag the demo steps' panels and give each its tab: number, title, hint and a
// state mark. The panels keep their STEP_ORDER places (the run order); the
// CSS hides the untagged ones (the steps without a tab) and shows the tab in
// place of the step's name. The knobs and the Run button are untouched
// underneath.
function decorateDemoPanels() {
  const host = $("#step-panels");
  DEMO_STEPS.forEach(([step, title, hint], i) => {
    const box = host.querySelector(`details.step-box[data-step="${step}"]`);
    if (!box) return;
    box.dataset.demo = "1";
    box.dataset.demoState = "waiting";
    const summary = box.querySelector("summary");
    const raw = document.createElement("span");
    raw.className = "step-name";
    raw.textContent = summary.textContent; // the step's own name (hidden by the CSS)
    const tab = document.createElement("span");
    tab.className = "demo-tab";
    const num = document.createElement("span");
    num.className = "demo-num";
    num.textContent = String(i + 1);
    const lab = document.createElement("span");
    lab.className = "demo-title";
    lab.textContent = title;
    const sub = document.createElement("span");
    sub.className = "demo-hint";
    sub.textContent = hint;
    const mark = document.createElement("span");
    mark.className = "demo-mark";
    mark.setAttribute("aria-hidden", "true");
    tab.append(num, lab, sub, mark);
    summary.textContent = "";
    summary.append(raw, tab);
  });
}
// Each tab's state: "running" = the step holding the session lock right now
// (GET /state `running`, also when the copilot drives the server); "done" = a
// record for it is in the journal (so a re-run upstream, which truncates the
// journal, puts the tabs after it back to waiting — the journal is the truth);
// else "waiting". Painted on every journal refresh and every /state poll.
// "failed" (the Mesh wall tab only): the twin's wall slot carries the note
// "last wall gate FAIL: … mm along the normal" while the copilot's last
// mesh_wall call failed its quality gate (GET /twin, computed by the report
// code from the same journal). The volmesh record IS in the journal then — the
// step ran, the gate refused the wall — so "done" alone would read as a pass.
// The wording is the report's own (copilot/report.py); the asset test pins
// the two together.
const WALL_GATE_FAIL_NOTE = "last wall gate FAIL";
let demoDoneSteps = new Set();
let demoWallFailed = false;
function noteDemoJournal(journal) {
  demoDoneSteps = new Set((journal || []).map((rec) => DEMO_STEP_OF[rec.step] || rec.step));
  paintDemoTabs();
}
function noteDemoTwin(twin) {
  const wall = (twin.slots || []).find((s) => s.key === "wall");
  demoWallFailed = Boolean(wall && String(wall.note || "").startsWith(WALL_GATE_FAIL_NOTE));
  paintDemoTabs();
}
function paintDemoTabs() {
  const running = busyStep ? DEMO_STEP_OF[busyStep] || busyStep : null;
  for (const box of document.querySelectorAll('#step-panels details.step-box[data-demo="1"]')) {
    const step = box.dataset.step;
    box.dataset.demoState =
      step === running ? "running"
      : step === "volmesh" && demoWallFailed ? "failed"
      : demoDoneSteps.has(step) ? "done"
      : "waiting";
  }
}
// ---- patient atlas (D108 addendum 6) --------------------------------------
// One patient record, every study filed by body region and modality, the
// way every surgeon would land on the same page. Cloned once, when the page
// starts, from the inert <template> in index.html, above the twin. Data: GET
// /atlas — the 15-region
// catalogue and the studies the session REALLY holds (meshforge.atlas: today
// at most the CT the segment step loaded, its coverage declared by checksum).
// This paints that and nothing else: no earlier study, no other patient, no
// date (withheld), and `loaded ✓` is the server's own word for the session's
// source. The modality chip filters the body map and the list alike; a
// region with studies is lit and carries a count badge; hover (or keyboard
// focus) names the region and its studies on the line under the figure; a
// click selects it and filters the list; click again, or Esc, clears.
const SVG_NS = "http://www.w3.org/2000/svg";
const ATLAS_IDLE_HINT = "hover a region";
let atlasData = null; // last GET /atlas payload
let atlasModality = "All"; // the chip that is on
let atlasSelected = null; // selected region id, or null
let atlasHovered = null; // region under the pointer / keyboard focus, or null
function buildAtlasPanel() {
  if ($("#atlas-panel")) return;
  const tpl = $("#atlas-svg");
  const twin = $("#twin-panel");
  if (!tpl || !twin) return;
  const node = tpl.content.firstElementChild.cloneNode(true);
  twin.parentElement.insertBefore(node, twin);
  for (const chip of node.querySelectorAll(".atlas-chip")) {
    chip.addEventListener("click", () => {
      atlasModality = chip.dataset.modality || "All";
      paintAtlas();
    });
  }
  for (const region of node.querySelectorAll(".atlas-region")) {
    const id = region.dataset.region;
    const enter = () => {
      atlasHovered = id;
      paintAtlasHover();
    };
    const leave = () => {
      if (atlasHovered === id) atlasHovered = null;
      paintAtlasHover();
    };
    region.addEventListener("mouseenter", enter);
    region.addEventListener("mouseleave", leave);
    region.addEventListener("focus", enter);
    region.addEventListener("blur", leave);
    region.addEventListener("click", () => selectAtlasRegion(atlasSelected === id ? null : id));
    region.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        selectAtlasRegion(atlasSelected === id ? null : id);
      }
    });
  }
  document.addEventListener("keydown", atlasEscape);
  paintAtlas(); // the catalogue's own labels until /atlas answers
  refreshAtlas().catch(() => {});
}
function atlasEscape(e) {
  if (e.key === "Escape" && atlasSelected !== null) selectAtlasRegion(null);
}
function selectAtlasRegion(id) {
  atlasSelected = id;
  paintAtlas();
}
// Fetched with the twin on every session change; a no-op until the panel is
// in the DOM.
async function refreshAtlas() {
  if (!$("#atlas-panel")) return;
  try {
    atlasData = await apiGet("/atlas");
  } catch (err) {
    return; // a hiccup must not spam the log; the next refresh retries
  }
  paintAtlas();
}
// The studies on file, under the chip, in one region (or all when null).
function atlasStudies(regionId) {
  const all = (atlasData && atlasData.studies) || [];
  return all.filter(
    (s) =>
      (atlasModality === "All" || s.modality === atlasModality) &&
      (!regionId || (s.regions || []).includes(regionId))
  );
}
function atlasRegionName(id) {
  const r = ((atlasData && atlasData.regions) || []).find((x) => x.id === id);
  if (r) return r.name;
  const el = document.querySelector(`.atlas-region[data-region="${id}"]`);
  return el ? el.getAttribute("aria-label") : id;
}
function paintAtlas() {
  const panel = $("#atlas-panel");
  if (!panel) return;
  panel.dataset.modality = atlasModality;
  for (const chip of panel.querySelectorAll(".atlas-chip")) {
    chip.classList.toggle("on", (chip.dataset.modality || "All") === atlasModality);
  }
  $("#atlas-patient").textContent = (atlasData && atlasData.patient) || "no patient loaded";
  const badges = panel.querySelector(".atlas-badges");
  badges.innerHTML = "";
  for (const region of panel.querySelectorAll(".atlas-region")) {
    const id = region.dataset.region;
    const n = atlasStudies(id).length;
    region.dataset.has = n > 0 ? "1" : "0";
    region.classList.toggle("selected", atlasSelected === id);
    region.setAttribute("aria-pressed", atlasSelected === id ? "true" : "false");
    if (n > 0) {
      const g = document.createElementNS(SVG_NS, "g");
      g.setAttribute("class", "atlas-badge");
      const c = document.createElementNS(SVG_NS, "circle");
      c.setAttribute("cx", region.dataset.cx);
      c.setAttribute("cy", region.dataset.cy);
      c.setAttribute("r", "5.5"); // an 11 px badge (1 unit = 1 px)
      const t = document.createElementNS(SVG_NS, "text");
      t.setAttribute("x", region.dataset.cx);
      t.setAttribute("y", region.dataset.cy);
      t.textContent = String(n);
      g.append(c, t);
      badges.appendChild(g);
    }
  }
  paintAtlasHover();
  $("#atlas-scope").textContent = atlasSelected ? atlasRegionName(atlasSelected) : "all regions";
  const list = $("#atlas-studies");
  list.innerHTML = "";
  const rows = atlasStudies(atlasSelected);
  if (!rows.length) {
    const li = document.createElement("li");
    li.className = "atlas-empty";
    const what = atlasModality === "All" ? "studies" : `${atlasModality} studies`;
    const where = atlasSelected ? atlasRegionName(atlasSelected) : "this patient";
    li.textContent = `no ${what} on file for ${where} — X-ray, CT, MRI, US would list here`;
    list.appendChild(li);
  }
  for (const s of rows) {
    const li = document.createElement("li");
    const tag = document.createElement("span");
    tag.className = "atlas-tag";
    tag.textContent = s.modality;
    const desc = document.createElement("span");
    desc.className = "atlas-desc";
    desc.textContent = shownText(s.description || "");
    const date = document.createElement("span");
    date.className = "atlas-date";
    date.textContent = s.date || ""; // the server withholds dates (no PHI); nothing is made up here
    const status = document.createElement("span");
    status.className = "atlas-status";
    status.textContent = s.status || "";
    li.append(tag, desc, date, status);
    if (s.loaded) {
      const loaded = document.createElement("span");
      loaded.className = "atlas-loaded";
      loaded.textContent = "loaded ✓";
      li.appendChild(loaded);
    }
    list.appendChild(li);
  }
  const n = atlasData ? Number(atlasData.n_patients) || 0 : 0;
  $("#atlas-caption").textContent =
    `atlas · ${n} patient${n === 1 ? "" : "s"} on file · regions without studies are dim · sides are the patient's`;
}
// The line under the figure: the hovered (or focused) region, else the
// selected one, else the idle hint.
function paintAtlasHover() {
  const el = $("#atlas-hover");
  if (!el) return;
  const id = atlasHovered || atlasSelected;
  paintAtlasFigureLabel(id);
  if (!id) {
    el.textContent = ATLAS_IDLE_HINT;
    return;
  }
  const name = atlasRegionName(id);
  const rows = atlasStudies(id);
  const mod = atlasModality === "All" ? "" : `${atlasModality} `;
  if (!rows.length) {
    el.textContent = `${name} · no ${mod}studies on file`;
    return;
  }
  const current = rows.find((s) => s.status === "current") || rows[0];
  const count =
    rows.length === 1 ? `1 ${rows[0].modality} study` : `${rows.length} ${mod}studies`;
  el.textContent = `${name} · ${count} · ${current.status}: ${current.modality} ${shownText(current.description || "")}`;
}
// The hovered (or selected) region's name ON the figure: a dark pill with
// white text centred on the region's badge point, kept inside the figure's
// width, so the eye on the figure reads the name without leaving it. The
// line under the chips says the same, with the studies.
const ATLAS_FIG_W = 170;
function paintAtlasFigureLabel(id) {
  const layer = document.querySelector("#atlas-panel .atlas-labels");
  if (!layer) return;
  layer.innerHTML = "";
  if (!id) return;
  const region = document.querySelector(`.atlas-region[data-region="${id}"]`);
  if (!region) return;
  const g = document.createElementNS(SVG_NS, "g");
  g.setAttribute("class", "atlas-label");
  const rect = document.createElementNS(SVG_NS, "rect");
  const text = document.createElementNS(SVG_NS, "text");
  text.textContent = atlasRegionName(id);
  g.append(rect, text);
  layer.appendChild(g);
  const w = (text.getComputedTextLength ? text.getComputedTextLength() : 7 * text.textContent.length) + 12;
  const h = 18;
  const cx = Math.min(Math.max(Number(region.dataset.cx) || 0, w / 2 + 1), ATLAS_FIG_W - w / 2 - 1);
  const cy = Number(region.dataset.cy) || 0;
  rect.setAttribute("x", String(cx - w / 2));
  rect.setAttribute("y", String(cy - h / 2));
  rect.setAttribute("width", String(w));
  rect.setAttribute("height", String(h));
  rect.setAttribute("rx", "4");
  text.setAttribute("x", String(cx));
  text.setAttribute("y", String(cy));
}
// The result reveal lives in viewport.js (D112): a NEW artifact is drawn in
// from the top down, cell by cell (refreshViewport passes the sweep flag).
// The step running on the server right now (GET /state `running`, set while
// the step holds the session lock — also when the copilot drives it over HTTP)
// with its elapsed seconds, ticking locally between polls.
let busyTicker = null;
let busyStart = null;
let busyStep = null;
function updateBusyLine(st) {
  if (!st || !st.running) {
    hideBusyLine();
    return;
  }
  if (busyStep !== st.running) {
    busyStep = st.running;
    busyStart = Date.now() - 1000 * (Number(st.running_for) || 0);
    paintDemoTabs();
  }
  if (!busyTicker) busyTicker = setInterval(tickBusyLine, 250);
  tickBusyLine();
}
function tickBusyLine() {
  const el = $("#vp-busy");
  if (!el || !busyStep) return;
  const secs = Math.max(0, Math.round((Date.now() - busyStart) / 1000));
  el.textContent = `${DEMO_VERBS[busyStep] || busyStep + " running"}… ${secs} s`;
  el.hidden = false;
}
function hideBusyLine() {
  const el = $("#vp-busy");
  if (el) el.hidden = true;
  if (busyTicker) clearInterval(busyTicker);
  busyTicker = null;
  const wasRunning = busyStep !== null;
  busyStep = null;
  busyStart = null;
  if (wasRunning) paintDemoTabs();
}
// The page's own start: the atlas above the twin, the tabs painted at once.
function initDemoLayout() {
  buildAtlasPanel();
  paintDemoTabs();
}

// ---- init --------------------------------------------------------------
async function init() {
  // Panel order from /catalog (rule-4-safe: served by the API, not hardcoded
  // here), fetched first so the panel builder below can use
  // catalog.step_order.
  let catalog = null;
  try {
    catalog = await apiGet("/catalog");
  } catch (err) {
    onError(err);
  }

  // Step panels from the STEP_KNOBS schema, laid out in the server's
  // explicit STEP_ORDER (/catalog.step_order -> pipeline.STEP_ORDER), the
  // order the steps run in. Falls back to the schema's own key order
  // (also STEP_ORDER-sorted server-side) if the catalog fetch failed or came
  // back empty. A step present in /schema but absent from step_order is
  // still appended — never silently dropped (rule 6) — with a loud log.
  try {
    const schema = await apiGet("/schema");
    const host = $("#step-panels");
    const order =
      catalog && Array.isArray(catalog.step_order) && catalog.step_order.length
        ? catalog.step_order
        : Object.keys(schema);
    const placed = new Set();
    for (const step of order) {
      if (!(step in schema)) continue; // step_order named a step /schema doesn't have
      host.appendChild(buildStepPanel(step, schema[step]));
      placed.add(step);
    }
    for (const step of Object.keys(schema)) {
      if (placed.has(step)) continue;
      log(`WARNING: step "${step}" is missing from /catalog step_order — appended at the end of the panel list`, true);
      host.appendChild(buildStepPanel(step, schema[step]));
    }
  } catch (err) {
    onError(err);
  }

  decorateDemoPanels(); // tags the six demo steps and builds their tabs
  initDemoLayout();

  $("#btn-save").addEventListener("click", () => {
    withBusy(
      $("#btn-save"),
      "Saving…",
      async () => {
        try {
          await saveWorkfile();
          clearErrorEl($("#save-error"));
        } catch (err) {
          onError(err);
          setError($("#save-error"), err);
        }
      },
      () => !(lastJournalLen > 0) // state-derived — see refreshState()
    ).catch(onError);
  });
  $("#btn-wload").addEventListener("click", () => {
    withBusy($("#btn-wload"), "Loading…", async () => {
      try {
        await loadWorkfile();
        clearErrorEl($("#wload-error"));
      } catch (err) {
        onError(err);
        setError($("#wload-error"), err);
      }
    }).catch(onError);
  });
  $("#btn-reset").addEventListener("click", () => resetSession().catch(onError));

  initViewport();
  initLayerSwitch();
  await refreshPresets();
  await refreshJournal();
  await refreshState();
  await refreshTwin(); // pending slots before a run; filled if a session is in progress
  await refreshAtlas(); // the patient atlas
  await refreshViewport(true); // paints if a session was already in progress
  startAutoRefresh(); // keep the view current on external session changes
  log("Workbench ready." + (viewportUp ? "" : " (3-D viewport unavailable)"));
}

init();
