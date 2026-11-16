"""D80/D82 over HTTP + the shipped frontend wiring (no browser).

The picker routes are read-only queries; the frontend turns their answers
into the cfd knobs' plain scalar form, so the journal never depends on them.
The static checks read the assets as text, like test_web_api.py does.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from meshforge.web.app import create_app  # noqa: E402


def _static(name: str) -> str:
    import meshforge.web as web_pkg

    return (Path(web_pkg.__file__).parent / "static" / name).read_text()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


def _drive_to_lumen(client: TestClient) -> None:
    assert client.post("/source/synth", json={"shape": "straight_tube"}).status_code == 200
    assert client.post("/step/centerline", json={}).status_code == 200
    r = client.post("/step/lumen", json={"n_axial": 12, "n_theta": 16})
    assert r.status_code == 200, r.text


# ---- routes ---------------------------------------------------------------


def test_presets_route_follows_the_source_unit(client: TestClient, tmp_path: Path) -> None:
    assert client.post("/source/synth", json={"shape": "straight_tube"}).status_code == 200
    r = client.get("/cfd/presets")
    assert r.status_code == 400 and "length unit" in r.json()["detail"]
    # a millimetre source (segment/sac set this); the route reads the session's unit
    client.app.state.holder.session.length_unit = "mm"
    body = client.get("/cfd/presets").json()
    assert body["length_unit"] == "mm"
    assert {p["id"]: p["value"] for p in body["speed"]} == {"aorta_mean": 150.0, "aorta_peak": 1000.0}
    assert body["nu"][0]["value"] == 3.3


def test_pick_route_names_inflow_and_outflow(client: TestClient) -> None:
    r = client.post("/cfd/pick", json={"point": [0.0, 0.0, 0.0]})
    assert r.status_code == 400 and "volmesh" in r.json()["detail"]
    _drive_to_lumen(client)
    cl = client.app.state.holder.session.centerline
    body = client.post("/cfd/pick", json={"point": [float(v) for v in cl.points[0]]}).json()
    assert body["inflow"]["name"] == "inlet" and body["outflow"]["name"] == "outlet"
    body = client.post("/cfd/pick", json={"point": [float(v) for v in cl.points[-1]]}).json()
    assert body["inflow"]["name"] == "outlet"
    assert client.post("/cfd/pick", json={"point": [1.0, 2.0]}).status_code == 400


def test_inflow_preview_route_reports_q_and_re(client: TestClient) -> None:
    _drive_to_lumen(client)
    body = client.post("/cfd/inflow", json={"patch": "inlet", "speed": 2.0, "nu": 0.5}).json()
    assert body["flow_rate"] == pytest.approx(2.0 * body["area"])
    assert body["reynolds"] == pytest.approx(2.0 * body["d_eq"] / 0.5)
    assert body["notes"] and "Re " in body["notes"][0]
    for bad in ({"patch": "inlet", "speed": -1.0, "nu": 0.5},
                {"patch": "inlet", "speed": 1.0, "nu": 0.0},
                {"patch": "wall", "speed": 1.0, "nu": 0.5}):
        assert client.post("/cfd/inflow", json=bad).status_code == 400


def test_picker_routes_are_read_only(client: TestClient) -> None:
    _drive_to_lumen(client)
    before = client.get("/state").json()
    journal = client.get("/journal").json()["journal"]
    cl = client.app.state.holder.session.centerline
    client.post("/cfd/pick", json={"point": [float(v) for v in cl.points[0]]})
    client.post("/cfd/inflow", json={"patch": "inlet", "speed": 1.0, "nu": 1.0})
    client.get("/cfd/presets")
    assert client.get("/state").json()["rev"] == before["rev"]
    assert client.get("/journal").json()["journal"] == journal


def test_geometry_payload_carries_the_flow_section(client: TestClient) -> None:
    pytest.importorskip("trame_vtk")
    from meshforge.geometry.ribbon import centerline_ribbon

    _drive_to_lumen(client)
    s = client.app.state.holder.session
    grid = s.volmesh.grid.copy()
    # A no-slip-like field: 0 on every boundary point, 1 inside. The boundary
    # surface alone spans [0, 0] - exactly the "uniform blue wall".
    on_boundary = grid.extract_surface(algorithm="dataset_surface")["vtkOriginalPointIds"]
    u = np.ones(grid.n_points)
    u[on_boundary] = 0.0
    grid.point_data["U_mag"] = u
    sec = centerline_ribbon(s.centerline, s.volmesh.grid)
    holder = client.app.state.holder
    holder.last_render = {"mesh": grid, "kind": "volume"}
    plain = client.get("/geometry?field=U_mag").json()
    assert "section" not in plain
    assert plain["data_range"] == pytest.approx([0.0, 0.0])
    holder.last_render = {
        "mesh": grid, "kind": "volume", "label": "cfd results",
        "section": sec.surface, "view_dir": list(sec.view_dir),
    }
    body = client.get("/geometry?field=U_mag").json()
    assert body["section"] and "mesh" in body["section"]
    assert body["view_dir"] == pytest.approx(list(sec.view_dir))
    # the colour range now spans the interior the section shows
    assert body["data_range"][0] == pytest.approx(0.0)
    assert body["data_range"][1] > 0.9


# ---- shipped frontend wiring (text-level, no JS engine) -------------------


def test_frontend_calls_the_picker_routes_that_exist() -> None:
    js = _static("app.js")
    routes = {r.path for r in create_app().routes}
    for path in ("/cfd/pick", "/cfd/inflow", "/cfd/presets"):
        assert path in js, f"frontend never calls {path}"
        assert path in routes, f"{path} is not a real route"


def test_picker_writes_the_scalar_knob_form_only() -> None:
    js = _static("app.js")
    # the picker writes the cfd panel's OWN knobs, in the replayable form
    assert 'knobInput("inlet").value = `${a.name}=${speed}`' in js
    assert 'knobInput("outlet").value = `${fpState.outflow.name}=0`' in js
    assert 'knobInput("nu").value = p.value' in js
    # swap exchanges the two ends; nothing else
    assert "fpState = { inflow: fpState.outflow, outflow: fpState.inflow }" in js
    # its own controls carry no data-knob, so gatherKnobs never posts them
    start = js.index("wrap.innerHTML = `")
    template = js[start:js.index("`;", start)]
    assert "<input" in template and "data-knob" not in template
    # Q and Re come from the server, never computed in the page (rule 4)
    assert "reynolds" not in js and "d_eq /" not in js
    # a new mesh or source resets a picked end; a cfd run keeps it
    assert 'if (!result || result.step !== "cfd") resetFlowPicker();' in js
    # the pick finds its panel by class: a step box need not be a <fieldset>
    # (a collapsible <details> is pending), and a tag lookup would null out
    assert 'wrap.closest(".step-box")' in js and 'closest("fieldset")' not in js
    for forbidden in ("meshforge.geometry", "meshforge.meshing", "offset_wall"):
        assert forbidden not in js


def test_workbench_assets_are_revalidated(client: TestClient) -> None:
    # Without Cache-Control a browser keeps running a stale app.js against a
    # newer server's routes after an upgrade (seen live with this picker).
    for path in ("/app/app.js", "/app/viewport.js", "/app/index.html"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert r.headers.get("cache-control") == "no-cache", path


def test_viewport_pick_mode_and_section_wiring() -> None:
    vp = _static("viewport.js")
    assert ':picking-modes="pickModes" @click="onPick"' in vp
    assert "PICK_SLOP_PX" in vp  # a rotate-drag is not a pick
    assert "disarmPick();\n    cb(" in vp  # one pick, then disarmed
    assert '<vtk-geometry-representation v-if="section"' in vp
    assert ':property="meshProperty"' in vp and "WALL_UNDER_SECTION = { opacity: 0.15 }" in vp
    assert "if (dir) aimAlong(v, dir);" in vp  # face-on to the section
    for name in ("setSectionVisible", "armPick", "disarmPick"):
        assert f"{name}: {name}" in vp
    html = _static("index.html")
    assert 'id="vp-section"' in html and 'id="vp-section-ctl" hidden' in html
    for forbidden in ("meshforge.geometry", "meshforge.meshing", "offset_wall"):
        assert forbidden not in vp
