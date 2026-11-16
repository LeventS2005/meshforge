// meshforge viewport (V35 increment 3) — PURE client-side vtk.js.
//
// Renders a `vtk_mesh` state fetched from GET /geometry with **no WebSocket and
// no server-side GL**: Vue + trame-vtk's `vue_vtk` bundle, both served locally
// from /vendor. Rule 4: no meshing logic — it only paints a mesh the backend
// already serialized. Exposes `window.MeshforgeViewport`; app.js drives it.

(function () {
  "use strict";

  var S = {
    app: null, viewRef: null, mesh: null, range: null, preset: null, overlay: null, ghosts: null, ready: false,
    // D82 flow section: the server-sampled section state, whether to show it,
    // and the wall's look while it shows (translucent, so the flow reads).
    section: null, sectionState: null, showSection: true, showWall: false, meshProperty: null, wallVisible: null,
    // D112 step reveal: the previous artifact, drawn as a faint shell while
    // the new one sweeps in; and the layer switch's persistent shell (the
    // wall around the flow), kept until the layer changes.
    base: null, shell: null,
    // D80 pick mode: armed for ONE click, then disarmed.
    pickModes: null, pickCb: null, pickMiss: null, downXY: null, upXY: null,
  };

  // Fixed look for the QC overlay (raw source contours): amber unlit lines so
  // they read over any surface colormap.
  var OVERLAY_PROPERTY = { color: [1.0, 0.72, 0.2], lineWidth: 2, lighting: false };
  // The ghosts: the whole AI outline of the aorta (arch and all), drawn faint
  // and translucent behind the artifact cut from it, so a cropped segment, its
  // wall or its stress field is seen in place; and the patient's own heart
  // from the same AI run, above the arch, in a slightly warmer tint at the
  // same opacity. Context only, never coloured by data (fieldless states).
  var GHOST_PROPERTY = { color: [0.78, 0.8, 0.84], opacity: 0.16, specular: 0.0 };
  var HEART_GHOST_PROPERTY = { color: [0.88, 0.74, 0.72], opacity: 0.16, specular: 0.0 };
  var GHOST_PROPERTIES = { aorta: GHOST_PROPERTY, heart: HEART_GHOST_PROPERTY };
  // The wall around a shown flow section: a faint shell for context.
  var WALL_UNDER_SECTION = { opacity: 0.15 };
  var WALL_OPAQUE = { opacity: 1.0 };
  // A press-release that moved further than this (CSS px) was a rotate-drag,
  // not a pick — the browser still fires "click" after a drag.
  var PICK_SLOP_PX = 5;
  // D111 demo-mode turntable: left alone for IDLE_MS, the view spins slowly
  // about the camera's up axis through the focal point (one turn in 45 s),
  // and stops the moment the canvas is pressed, scrolled, touched or typed
  // on; hovering does not stop it. Demo mode only (body.demo), never while a
  // pick is armed, never without a mesh, never when the OS asks for reduced
  // motion. Pure camera motion: the mesh and its data are untouched.
  var IDLE_MS = 3000;
  var SPIN_DEG_PER_S = 8;
  var R = { timer: null, raf: null, last: 0, spinning: false };
  // D112 demo-mode step reveal: a NEW artifact (and a layer switch) is drawn
  // in from the top down over SWEEP_MS. The cells of the serialized surface
  // are sorted by their highest z (the patient's superior end) and the
  // connectivity grows frame by frame as a prefix of that order; the points,
  // the colour field and the range are untouched, so what stands at the end
  // is exactly the state the server sent. The artifact shown before stays as
  // a faint grey shell under the sweep and leaves when it completes. Demo
  // mode only, never under reduced motion; a field or frame swap, a new
  // load or a clear cancels it.
  var SWEEP_MS = 2400;
  var BASE_PROPERTY = { color: [0.6, 0.62, 0.66], opacity: 0.28, specular: 0.0 };
  var SHELL_PROPERTY = { color: [0.74, 0.76, 0.8], opacity: 0.22, specular: 0.0 };
  var W = { raf: null, start: 0, items: null, done: null };

  function available() {
    return !!(window.Vue && window.vue_vtk);
  }

  // Build the Vue app once, mounted on #<elId>. Reactive refs let us swap the
  // mesh (field / animation frame) while the camera is preserved.
  function mount(elId) {
    if (!available()) return false;
    var host = document.getElementById(elId);
    if (!host) return false;
    var empty = document.getElementById("vp-empty");
    if (empty) empty.remove(); // Vue takes over the host element

    var Vue = window.Vue;
    S.mesh = Vue.ref(null);
    S.range = Vue.ref([0, 1]);
    S.preset = Vue.ref("Cool to Warm");
    S.overlay = Vue.ref(null);
    S.ghosts = Vue.ref([]);
    S.section = Vue.ref(null);
    S.base = Vue.ref(null);
    S.shell = Vue.ref(null);
    S.meshProperty = Vue.ref(WALL_OPAQUE);
    S.wallVisible = Vue.ref(true);
    S.pickModes = Vue.ref([]);

    // Remember where a press started and ended so onPick can tell a click
    // from a rotate-drag (capture phase: runs before vtk.js sees the event).
    host.addEventListener("pointerdown", function (e) { S.downXY = [e.clientX, e.clientY]; }, true);
    host.addEventListener("pointerup", function (e) { S.upXY = [e.clientX, e.clientY]; }, true);

    S.app = Vue.createApp({
      setup: function () {
        var view = Vue.ref(null);
        S.viewRef = view;
        return {
          view: view,
          mesh: S.mesh,
          range: S.range,
          preset: S.preset,
          overlay: S.overlay,
          overlayProperty: OVERLAY_PROPERTY,
          ghosts: S.ghosts,
          section: S.section,
          base: S.base,
          baseProperty: BASE_PROPERTY,
          shell: S.shell,
          shellProperty: SHELL_PROPERTY,
          meshProperty: S.meshProperty,
          wallVisible: S.wallVisible,
          pickModes: S.pickModes,
          onPick: onPick,
        };
      },
      template:
        '<vtk-view ref="view" :background="[0.063,0.063,0.071]" style="width:100%;height:100%"' +
        '          :picking-modes="pickModes" @click="onPick">' +
        '  <vtk-geometry-representation v-if="mesh && wallVisible" :color-map-preset="preset" :color-data-range="range" :property="meshProperty">' +
        '    <vtk-mesh :state="mesh"></vtk-mesh>' +
        '  </vtk-geometry-representation>' +
        '  <vtk-geometry-representation v-if="section" :color-map-preset="preset" :color-data-range="range">' +
        '    <vtk-mesh :state="section"></vtk-mesh>' +
        '  </vtk-geometry-representation>' +
        '  <vtk-geometry-representation v-if="base" :property="baseProperty">' +
        '    <vtk-mesh :state="base"></vtk-mesh>' +
        '  </vtk-geometry-representation>' +
        '  <vtk-geometry-representation v-if="shell" :property="shellProperty">' +
        '    <vtk-mesh :state="shell"></vtk-mesh>' +
        '  </vtk-geometry-representation>' +
        '  <vtk-geometry-representation v-if="overlay" :property="overlayProperty">' +
        '    <vtk-mesh :state="overlay"></vtk-mesh>' +
        '  </vtk-geometry-representation>' +
        '  <vtk-geometry-representation v-for="g in ghosts" :key="g.kind" :property="g.property">' +
        '    <vtk-mesh :state="g.state"></vtk-mesh>' +
        '  </vtk-geometry-representation>' +
        "</vtk-view>",
    });
    S.app.config.errorHandler = function (err) {
      console.error("[viewport]", err);
    };
    S.app.use(window.vue_vtk);
    S.app.mount("#" + elId);
    S.ready = true;
    watchIdle(host); // D111: the demo-mode turntable
    return true;
  }

  // A view direction no principal axis lies along: 50.7 degrees off x and y,
  // 63.6 off z. Straight tubes run along z, blocks and plates along x/y, so any
  // axis-aligned direction renders SOME common artifact as a flat outline.
  var THREE_QUARTER = [0.634, -0.634, 0.444]; // unit (1, -1, 0.7)

  // Point the camera three-quarters at the scene, without moving it closer or
  // further: the offset keeps its current length, so this is a direction-only
  // change and the resetCamera that follows sets the distance that frames the
  // mesh. vue_vtk's camera is a PLAIN OBJECT read with getCamera() and written
  // back with setCamera() — it has no vtk.js setter methods on it. Every step
  // is guarded: without those two accessors the camera keeps the direction it
  // had, resetCamera still frames the mesh, and nothing errors.
  function aimThreeQuarter(view) {
    aimAlong(view, THREE_QUARTER);
  }

  // Point the camera from direction `dir` (unit) at the focal point, same
  // distance; view-up is +z projected off `dir` (+y when `dir` is ~vertical).
  // A flow section (D82) is aimed this way so it is seen face-on.
  function aimAlong(view, dir) {
    try {
      if (!view.getCamera || !view.setCamera) return;
      var cam = view.getCamera();
      var fp = cam && cam.focalPoint;
      var pos = cam && cam.position;
      if (!fp || !pos) return;
      var dx = pos[0] - fp[0], dy = pos[1] - fp[1], dz = pos[2] - fp[2];
      var dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
      cam.position = [fp[0] + dir[0] * dist, fp[1] + dir[1] * dist, fp[2] + dir[2] * dist];
      var up = Math.abs(dir[2]) > 0.95 ? [0, 1, 0] : [0, 0, 1];
      var k = up[0] * dir[0] + up[1] * dir[1] + up[2] * dir[2];
      cam.viewUp = [up[0] - k * dir[0], up[1] - k * dir[1], up[2] - k * dir[2]];
      view.setCamera(cam);
    } catch (err) {
      console.warn("[viewport] camera aim unavailable", err);
    }
  }

  // ---- demo-mode turntable (D111) -----------------------------------------
  function demoOn() {
    return document.body.classList.contains("demo");
  }
  function reducedMotion() {
    try {
      return !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
    } catch (err) {
      return false;
    }
  }
  // Rotate the camera position about the view-up axis through the focal
  // point (a turntable), keeping distance and up; vue_vtk's setCamera repaints.
  function orbit(view, deg) {
    var cam = view.getCamera();
    var fp = cam && cam.focalPoint, pos = cam && cam.position, up = cam && cam.viewUp;
    if (!fp || !pos || !up) return false;
    var ul = Math.sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2]) || 1;
    var ux = up[0] / ul, uy = up[1] / ul, uz = up[2] / ul;
    var vx = pos[0] - fp[0], vy = pos[1] - fp[1], vz = pos[2] - fp[2];
    var a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
    var dot = ux * vx + uy * vy + uz * vz;
    var cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
    var rx = vx * c + cx * s + ux * dot * (1 - c);
    var ry = vy * c + cy * s + uy * dot * (1 - c);
    var rz = vz * c + cz * s + uz * dot * (1 - c);
    view.setCamera({ position: [fp[0] + rx, fp[1] + ry, fp[2] + rz], focalPoint: fp, viewUp: up });
    return true;
  }
  function spinStep(ts) {
    if (!R.spinning) return;
    var v = S.viewRef && S.viewRef.value;
    var hasMesh = !!(S.mesh && S.mesh.value);
    if (!v || !v.getCamera || !v.setCamera || !hasMesh || !demoOn()) {
      spinStop();
      return;
    }
    var dt = R.last ? Math.min(ts - R.last, 100) / 1000 : 0; // a stalled tab does not jump
    R.last = ts;
    try {
      if (dt > 0 && !orbit(v, SPIN_DEG_PER_S * dt)) {
        spinStop();
        return;
      }
    } catch (err) {
      console.warn("[viewport] turntable stopped", err);
      spinStop();
      return;
    }
    R.raf = window.requestAnimationFrame(spinStep);
  }
  function spinStart() {
    R.timer = null;
    if (R.spinning || !S.ready || !demoOn() || reducedMotion()) return;
    if (!(S.mesh && S.mesh.value)) return;
    if (S.pickCb) return; // a pick is armed: the view holds still for the click
    R.spinning = true;
    R.last = 0;
    R.raf = window.requestAnimationFrame(spinStep);
  }
  function spinStop() {
    R.spinning = false;
    if (R.raf) window.cancelAnimationFrame(R.raf);
    R.raf = null;
  }
  function idleDisarm() {
    if (R.timer) window.clearTimeout(R.timer);
    R.timer = null;
  }
  // (Re)start the idle clock: the spin begins IDLE_MS from now if nothing
  // touches the canvas (demo mode only; a no-op otherwise).
  function idleArm() {
    idleDisarm();
    if (!S.ready || !demoOn()) return;
    R.timer = window.setTimeout(spinStart, IDLE_MS);
  }
  // Any input on the canvas: stop, and start the clock again.
  function touched() {
    spinStop();
    idleArm();
  }
  // app.js calls this when demo mode is switched (on: arm; off: stop).
  function wake() {
    spinStop();
    idleArm();
  }
  function watchIdle(host) {
    var evs = ["pointerdown", "wheel", "touchstart", "keydown"];
    for (var i = 0; i < evs.length; i++) host.addEventListener(evs[i], touched, true);
    document.addEventListener("visibilitychange", function () {
      if (document.hidden) {
        spinStop();
        idleDisarm();
      } else {
        idleArm();
      }
    });
  }

  // ---- demo-mode step reveal (D112) -----------------------------------------
  function sweepOn() {
    return S.ready && demoOn() && !reducedMotion();
  }
  var TYPED = {
    int8: Int8Array, uint8: Uint8Array, int16: Int16Array, uint16: Uint16Array,
    int32: Int32Array, uint32: Uint32Array, float32: Float32Array, float64: Float64Array,
  };
  // A serialized array as vue_vtk's vtk-mesh takes it ({bvals (base64),
  // dtype, shape}, a plain array, or a typed array) -> a typed array, or
  // null when it is none of those.
  function typedArray(raw) {
    if (!raw) return null;
    if (ArrayBuffer.isView(raw)) return raw;
    if (Array.isArray(raw)) return Float64Array.from(raw);
    if (raw.bvals && raw.dtype && TYPED[raw.dtype]) {
      var bin = window.atob(raw.bvals);
      var bytes = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      return new TYPED[raw.dtype](bytes.buffer);
    }
    return null;
  }
  // Plan one state's top-down order: its polygons sorted by their highest z,
  // the connectivity rebuilt in that order, and the cumulative lengths so the
  // first k cells are one subarray. Null when the state has no polygons.
  function planSweep(state) {
    var mesh = state && state.mesh;
    if (!mesh || !mesh.polys || !mesh.points) return null;
    var pts = typedArray(mesh.points), conn = typedArray(mesh.polys);
    if (!pts || !conn || !conn.length) return null;
    var cells = []; // [start, length, top z]
    var i = 0;
    while (i < conn.length) {
      var n = conn[i];
      if (!(n > 0) || i + 1 + n > conn.length) return null; // not a plain connectivity array
      var top = -Infinity;
      for (var j = 1; j <= n; j++) {
        var z = pts[3 * conn[i + j] + 2];
        if (z > top) top = z;
      }
      cells.push([i, n + 1, top]);
      i += n + 1;
    }
    if (cells.length < 2) return null;
    cells.sort(function (a, b) { return b[2] - a[2]; });
    var ordered = new Uint32Array(conn.length);
    var offsets = new Uint32Array(cells.length + 1);
    var keys = new Float64Array(cells.length);
    var o = 0;
    for (var c = 0; c < cells.length; c++) {
      var start = cells[c][0], len = cells[c][1];
      for (var k = 0; k < len; k++) ordered[o + k] = conn[start + k];
      o += len;
      offsets[c + 1] = o;
      keys[c] = cells[c][2];
    }
    return { state: state, ordered: ordered, offsets: offsets, keys: keys, zTop: keys[0], zBottom: keys[cells.length - 1] };
  }
  // The planned state with only the cells whose top lies above the sweep
  // front at progress p (0 = the first cell, 1 = every cell).
  function sweepState(plan, p) {
    var k = plan.keys.length;
    if (p < 1) {
      var front = plan.zTop - p * (plan.zTop - plan.zBottom);
      var lo = 0, hi = plan.keys.length; // the first index whose key is below the front
      while (lo < hi) {
        var mid = (lo + hi) >> 1;
        if (plan.keys[mid] >= front) lo = mid + 1;
        else hi = mid;
      }
      k = Math.max(1, lo);
    }
    var mesh = Object.assign({}, plan.state.mesh, { polys: plan.ordered.subarray(0, plan.offsets[k]) });
    return Object.assign({}, plan.state, { mesh: mesh });
  }
  // Stop a running sweep; with `finish` every item shows its whole state.
  function sweepStop(finish) {
    if (W.raf) window.cancelAnimationFrame(W.raf);
    W.raf = null;
    var items = W.items, done = W.done;
    W.items = null;
    W.done = null;
    W.start = 0;
    if (items && finish) for (var i = 0; i < items.length; i++) items[i].set(items[i].plan.state);
    if (S.ready && S.base) S.base.value = null;
    if (done && finish) done();
  }
  function sweepStep(ts) {
    if (!W.items) return;
    if (!W.start) W.start = ts;
    var t = Math.min(1, (ts - W.start) / SWEEP_MS);
    var p = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; // ease in, ease out
    try {
      for (var i = 0; i < W.items.length; i++) W.items[i].set(sweepState(W.items[i].plan, p));
    } catch (err) {
      console.warn("[viewport] reveal stopped", err);
      sweepStop(true);
      return;
    }
    if (t >= 1) {
      sweepStop(true);
      return;
    }
    W.raf = window.requestAnimationFrame(sweepStep);
  }
  // items: [{plan, set(state)}]; base: the previous main state, drawn as a
  // fieldless grey shell meanwhile; done: called when the sweep completes.
  function sweepStart(items, base, done) {
    sweepStop(false);
    if (!items.length) {
      if (done) done();
      return false;
    }
    W.items = items;
    W.done = done || null;
    W.start = 0;
    if (S.base) S.base.value = base && base.mesh ? { mesh: base.mesh } : null;
    for (var i = 0; i < items.length; i++) items[i].set(sweepState(items[i].plan, 0));
    W.raf = window.requestAnimationFrame(sweepStep);
    return true;
  }
  // The items a loaded payload sweeps: the main mesh, the ghosts when they
  // arrive with it (the first artifact brings the AI outline), and the flow
  // section when one shows.
  function sweepItems(payload, ghostsNew) {
    var items = [];
    var main = planSweep(payload.state);
    if (main) items.push({ plan: main, set: function (st) { S.mesh.value = st; } });
    if (ghostsNew && S.ghosts.value.length) {
      S.ghosts.value.forEach(function (g, i) {
        var plan = planSweep(g.state);
        if (!plan) return;
        items.push({
          plan: plan,
          set: function (st) {
            var list = S.ghosts.value.slice();
            list[i] = Object.assign({}, list[i], { state: st });
            S.ghosts.value = list;
          },
        });
      });
    }
    if (S.sectionState && S.section.value) {
      var sp = planSweep(S.sectionState);
      if (sp) items.push({ plan: sp, set: function (st) { S.section.value = st; } });
    }
    return items;
  }

  // ---- flow section (D82) -------------------------------------------------
  function applySection() {
    var on = !!(S.showSection && S.sectionState);
    S.section.value = on ? S.sectionState : null;
    S.meshProperty.value = on ? WALL_UNDER_SECTION : WALL_OPAQUE;
    // With the flow map showing, the wall shell is optional (off by default:
    // even translucent it hides the colours behind it).
    S.wallVisible.value = !on || S.showWall;
  }
  function setRange(r) {
    if (S.ready && r && r.length === 2) S.range.value = r.slice();
  }
  function setWallVisible(on) {
    S.showWall = !!on;
    if (S.ready) applySection();
  }
  function setSectionVisible(on) {
    S.showSection = !!on;
    if (S.ready) applySection();
  }

  // ---- pick mode (D80) ----------------------------------------------------
  // armPick(cb, onMiss): the NEXT click on the mesh calls cb([x, y, z]) and
  // disarms; a click on empty background calls onMiss and stays armed; a
  // rotate-drag is ignored. disarmPick() cancels.
  function armPick(cb, onMiss) {
    if (!S.ready) return false;
    S.pickCb = cb;
    S.pickMiss = onMiss || null;
    S.pickModes.value = ["click"];
    spinStop();
    idleDisarm();
    return true;
  }
  function disarmPick() {
    S.pickCb = null;
    S.pickMiss = null;
    if (S.ready) S.pickModes.value = [];
    idleArm();
  }
  function onPick(e) {
    if (!S.pickCb) return;
    if (S.downXY && S.upXY) {
      var mx = S.upXY[0] - S.downXY[0], my = S.upXY[1] - S.downXY[1];
      if (Math.sqrt(mx * mx + my * my) > PICK_SLOP_PX) return; // a drag
    }
    var pos = e && e.worldPosition;
    if (!pos || pos.length !== 3) {
      if (S.pickMiss) S.pickMiss();
      return;
    }
    var cb = S.pickCb;
    disarmPick();
    cb([Number(pos[0]), Number(pos[1]), Number(pos[2])]);
  }

  // Show a serialized mesh. `resetCamera` frames it (used on a new artifact,
  // NOT on a field/frame swap so the view stays put during animation).
  // `sweep` (D112) draws it in from the top down in Demo mode; the previous
  // artifact stays as a faint shell meanwhile. The camera is framed on the
  // WHOLE mesh first (drawn at opacity 0 for that one flush), so the sweep
  // never frames its first cells.
  function load(payload, resetCamera, sweep) {
    if (!S.ready || !payload || !payload.state) return;
    sweepStop(false);
    var previous = S.mesh.value;
    var hadGhosts = !!(S.ghosts && S.ghosts.value && S.ghosts.value.length);
    S.mesh.value = payload.state;
    // The ghosts ride on the main payload only (a field or frame swap keeps them).
    if (S.ghosts && "ghost" in payload) S.ghosts.value = ghostList(payload.ghost);
    S.sectionState = payload.section || null;
    applySection();
    if (payload.data_range && payload.data_range.length === 2) {
      S.range.value = payload.data_range.slice();
    }
    var animate = !!sweep && sweepOn();
    if (animate && resetCamera) S.meshProperty.value = Object.assign({}, S.meshProperty.value, { opacity: 0 });
    var begin = function () {
      if (!animate) return;
      applySection(); // the wall's own look again (opacity back)
      var ghostsNew = !hadGhosts && !!(S.ghosts.value && S.ghosts.value.length);
      // with a shell standing (the wall around the flow) the previous
      // artifact is already on screen as that shell: no second ghost of it
      var base = S.shell && S.shell.value ? null : previous;
      sweepStart(sweepItems(payload, ghostsNew), base, function () { applySection(); });
    };
    if (resetCamera) {
      var dir = payload.view_dir && payload.view_dir.length === 3 ? payload.view_dir : null;
      // Let Vue flush the new mesh into the scene, then frame + repaint it.
      setTimeout(function () {
        var v = S.viewRef && S.viewRef.value;
        if (v && v.resetCamera) {
          // resetCamera re-frames along the camera's CURRENT direction, and the
          // default direction is straight down +z — so a straight tube opened as
          // a flat annulus. Aim first (face-on to a flow section when there is
          // one, else three-quarters); resetCamera then recentres on the mesh
          // bounds and keeps that direction.
          if (dir) aimAlong(v, dir);
          else aimThreeQuarter(v);
          v.resetCamera();
          idleArm();
        }
        begin();
      }, 60);
    } else {
      begin();
    }
  }

  // The layer switch's shell (D112): a serialized state drawn as a
  // translucent fieldless surface around the main mesh — the wall around
  // the flow — or null to drop it. Kept across sweeps until changed.
  function setShell(state) {
    if (!S.ready || !S.shell) return;
    S.shell.value = state && state.mesh ? { mesh: state.mesh } : null;
  }

  function clear() {
    sweepStop(false);
    if (S.ready && S.shell) S.shell.value = null;
    if (S.ready && S.mesh) S.mesh.value = null;
    if (S.ready && S.overlay) S.overlay.value = null;
    if (S.ready && S.ghosts) S.ghosts.value = [];
    S.sectionState = null;
    if (S.ready) applySection();
    disarmPick();
    spinStop();
    idleDisarm();
  }

  // Show/hide the QC overlay (a serialized polyline state, or null). Kept
  // separate from load() so app.js can toggle it without refetching the mesh.
  function setOverlay(state) {
    if (S.ready && S.overlay) S.overlay.value = state || null;
  }
  // GET /geometry's `ghost`: a list of {state, label, kind} (null = none; a
  // single object is taken as a list of one) -> the representations to draw,
  // each with the fixed look of its kind (unknown kinds get the aorta grey).
  function ghostList(raw) {
    var items = Array.isArray(raw) ? raw : raw ? [raw] : [];
    var out = [];
    for (var i = 0; i < items.length; i++) {
      var g = items[i];
      if (!g || !g.state) continue;
      var kind = g.kind || "aorta";
      out.push({ kind: kind, state: g.state, property: GHOST_PROPERTIES[kind] || GHOST_PROPERTY });
    }
    return out;
  }
  // Show/hide the ghosts (the payload's list, or null).
  function setGhosts(list) {
    if (S.ready && S.ghosts) S.ghosts.value = ghostList(list);
  }

  window.MeshforgeViewport = {
    mount: mount,
    load: load,
    clear: clear,
    setOverlay: setOverlay,
    setGhosts: setGhosts,
    setShell: setShell,
    setSectionVisible: setSectionVisible,
    setWallVisible: setWallVisible,
    setRange: setRange,
    armPick: armPick,
    disarmPick: disarmPick,
    wake: wake,
    available: available,
  };
})();
