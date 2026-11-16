"""Headless HTTP tests for the V35 web layer (increment 1, backend only).

No browser: everything drives the FastAPI app through ``TestClient``. The
last test is the rule-4 guard — a static scan proving the web package holds
no meshing/geometry/solve logic and only calls ``MeshingSession``.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from meshforge.pipeline import STEP_KNOBS  # noqa: E402
from meshforge.web.app import create_app  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    # A fresh app => a fresh in-process MeshingSession per test.
    return TestClient(create_app())


def _drive_to_volmesh(client: TestClient) -> None:
    assert client.post("/source/synth", json={"shape": "straight_tube"}).status_code == 200
    assert client.post("/step/centerline", json={}).status_code == 200


def test_compute_routes_are_serialised_but_state_polls_are_not(client, monkeypatch) -> None:
    """D93: a step and the viewport's GET /geometry ran VTK on two threads at
    once and VTK 9.6.2's wrapper map crashed the Workbench. Compute routes
    share one lock; the /state poll the UI runs every second does not."""
    import threading

    assert client.post("/source/synth", json={"shape": "straight_tube"}).status_code == 200
    session = client.app.state.holder.session
    started, release = threading.Event(), threading.Event()
    real = session.compute_centerline

    def slow(**kw):
        started.set()
        assert release.wait(20)
        return real(**kw)

    monkeypatch.setattr(session, "compute_centerline", slow)
    step = threading.Thread(target=lambda: client.post("/step/centerline", json={}))
    step.start()
    assert started.wait(20)
    assert client.get("/state").status_code == 200  # not held up by the step
    assert client.get("/twin").status_code == 200  # nor is the twin panel (M6)
    assert client.get("/atlas").status_code == 200  # nor the patient atlas (D108 add. 6)
    got: list[int] = []
    geo = threading.Thread(target=lambda: got.append(client.get("/geometry").status_code))
    geo.start()
    geo.join(1.0)
    assert geo.is_alive() and not got  # waits for the step to finish
    release.set()
    step.join(20)
    geo.join(20)
    assert got == [200]


def test_full_pipeline_over_http(client: TestClient) -> None:
    _drive_to_volmesh(client)
    r = client.post("/step/volmesh", json={"n_axial": 30})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["step"] == "volmesh"
    # Clean synthetic geometry: zero inverted cells (definition of done).
    assert body["summary"]["quality"]["n_inverted"] == 0
    assert body["summary"]["n_cells"] > 0
    assert "volmesh[swept]" in body["text"]


def test_journal_mirrors_calls(client: TestClient) -> None:
    _drive_to_volmesh(client)
    client.post("/step/volmesh", json={"n_axial": 30})
    steps = [rec["step"] for rec in client.get("/journal").json()["journal"]]
    assert steps == ["generate_synth", "centerline", "volmesh"]
    state = client.get("/state").json()
    assert state["artifacts"]["volmesh"] is True
    assert state["journal_len"] == 3


def test_schema_covers_every_step(client: TestClient) -> None:
    schema = client.get("/schema").json()
    assert set(schema) == set(STEP_KNOBS)
    # The schema is the form contract: type/default/help per knob.
    thickness = schema["volmesh"]["thickness"]
    assert thickness["type"] == "float"
    assert thickness["default"] == 0.5
    assert thickness["help"]  # non-empty help text
    # A "compute-a-default" knob advertises a null default.
    assert schema["centerline"]["voxel_size"]["default"] is None


def test_health_lists_steps(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["ok"] is True
    assert set(body["steps"]) == set(STEP_KNOBS)


def test_missing_prerequisite_is_loud_400(client: TestClient) -> None:
    r = client.post("/step/centerline", json={})  # no surface yet
    assert r.status_code == 400
    # Same message the CLI prints (fail-loud, rule 6).
    assert "no surface available" in r.json()["detail"]
    assert "generate_synth" in r.json()["detail"]


def test_bad_parameter_type_is_422(client: TestClient) -> None:
    client.post("/source/synth", json={"shape": "straight_tube"})
    r = client.post("/step/centerline", json={"n_samples": "not_a_number"})
    assert r.status_code == 422


def test_unknown_knob_is_422(client: TestClient) -> None:
    client.post("/source/synth", json={"shape": "straight_tube"})
    r = client.post("/step/centerline", json={"bogus_knob": 1})
    assert r.status_code == 422


def test_unset_knob_uses_step_knobs_default(client: TestClient) -> None:
    # An unset knob with a real STEP_KNOBS default is SENT with that default
    # (front-end parity), not omitted: the journal records what the step got.
    client.post("/source/synth", json={"shape": "straight_tube"})
    client.post("/step/centerline", json={})
    r = client.post("/step/volmesh", json={"n_axial": 30})  # thickness unset -> 0.5
    assert r.status_code == 200, r.text
    params = client.get("/journal").json()["journal"][-1]["params"]
    assert params["thickness"] == STEP_KNOBS["volmesh"]["thickness"][1] == 0.5
    assert params["n_theta"] == STEP_KNOBS["volmesh"]["n_theta"][1]
    assert "max_tilt" not in params  # a None default is the compute-a-default sentinel: dropped


def test_workfile_save_reset_load_roundtrip(client: TestClient, tmp_path: Path) -> None:
    _drive_to_volmesh(client)
    client.post("/step/volmesh", json={"n_axial": 30})

    out = tmp_path / "session.mfwork"
    r = client.post("/workfile/save", json={"path": str(out)})
    assert r.status_code == 200, r.text
    assert Path(r.json()["path"]).is_file()

    # Reset wipes state.
    client.post("/session/reset")
    assert client.get("/state").json()["artifacts"]["volmesh"] is False

    # Load replays the journal into a fresh session.
    r = client.post("/workfile/load", json={"path": str(out)})
    assert r.status_code == 200, r.text
    replayed = r.json()
    assert [rec["step"] for rec in replayed["journal"]] == [
        "generate_synth",
        "centerline",
        "volmesh",
    ]
    vol = [res for res in replayed["results"] if res["step"] == "volmesh"][0]
    assert vol["summary"]["quality"]["n_inverted"] == 0
    assert client.get("/state").json()["artifacts"]["volmesh"] is True


def test_workfile_save_empty_journal_is_loud_400(client: TestClient, tmp_path: Path) -> None:
    r = client.post("/workfile/save", json={"path": str(tmp_path / "empty.mfwork")})
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_write_path_guard_refuses_overwriting_input(client: TestClient, tmp_path: Path) -> None:
    # The work-file path is client-supplied -> it must reuse guard_output_path,
    # on the path actually written (the .mfwork suffix is forced first).
    assert client.post("/source/synth", json={"shape": "straight_tube"}).status_code == 200
    src = tmp_path / "in.mfwork"
    src.write_text("{}")
    client.app.state.holder.session.input_path = src  # stand in for a source read from it

    r = client.post("/workfile/save", json={"path": str(tmp_path / "in")})
    assert r.status_code == 400
    assert "refusing to overwrite input file" in r.json()["detail"]
    assert src.read_text() == "{}"


# ---- increment 2: the Workbench frontend -------------------------------


def _static_dir() -> Path:
    import meshforge.web as web_pkg

    return Path(web_pkg.__file__).parent / "static"


def test_catalog_lists_synth_shapes(client: TestClient) -> None:
    # No dropdown any more: the catalog is the test hook's (/source/synth)
    # shape list plus the panel order. The web layer can't import synth
    # (rule 4), so it exposes the catalog by asking the session.
    from meshforge.synth import GENERATORS

    body = client.get("/catalog").json()
    assert body["shapes"] == sorted(GENERATORS)
    assert "straight_tube" in body["shapes"]


def test_workbench_served_not_placeholder(client: TestClient) -> None:
    # GET /app/ now serves the real Workbench, not the increment-1 placeholder.
    r = client.get("/app/")
    assert r.status_code == 200, r.text
    html = r.text
    assert "this is a placeholder" not in html.lower()
    assert 'src="app.js"' in html
    # The 3-pane structure is present.
    for marker in ("pane-left", "pane-center", "pane-right", "step-panels", "twin-panel"):
        assert marker in html, marker


def test_page_is_static_and_local(client: TestClient) -> None:
    # The page stays self-contained: 127.0.0.1-only, no external
    # navigation/fetch (D53 posture) — no CDN/font/link the app fetches.
    html = client.get("/app/").text
    assert "http://" not in html and "https://" not in html


def test_frontend_endpoints_all_exist_on_app() -> None:
    """Every endpoint the frontend calls is a real route (catches a typo'd
    path without a browser), and the frontend leaks no meshing logic."""
    js = (_static_dir() / "app.js").read_text()
    route_paths = {r.path for r in create_app().routes}
    for endpoint in (
        "/schema",
        "/catalog",
        "/journal",
        "/state",
        "/workfile/save",
        "/workfile/load",
        "/session/reset",
        "/twin",
        "/atlas",
    ):
        assert endpoint in js, f"frontend never calls {endpoint}"
        assert endpoint in route_paths, f"{endpoint} is not a real route"
    # The two source routes stay as test hooks; no button calls them.
    for hook in ("/source/synth", "/source/load"):
        assert hook in route_paths, f"{hook} is not a real route"
        assert hook not in js, f"the frontend still calls {hook}"
    # The per-step family is dynamic (/step/{step}).
    assert "/step/" in js
    assert any(p.startswith("/step/") for p in route_paths)
    # Rule 4, mirrored for the static frontend: no meshing internals.
    for forbidden in ("meshforge.geometry", "meshforge.meshing", "offset_wall"):
        assert forbidden not in js


def test_web_layer_holds_no_meshing_logic() -> None:
    """Rule 4: src/meshforge/web/ imports no meshing/geometry/solve internals;
    it only calls MeshingSession (+ the shared guard_output_path, sanctioned).
    AST-based (V43): only real imports/name references count — a token in a
    docstring or comment no longer trips the scan (the scanner itself is
    exercised in test_rule4_scan.py)."""
    from rule4_scan import assert_holds_no_meshing_logic

    import meshforge.web as web_pkg

    web_dir = Path(web_pkg.__file__).parent
    py_files = sorted(web_dir.glob("*.py"))
    assert py_files, "no web modules found to scan"
    assert_holds_no_meshing_logic(*py_files)
    combined = "".join(path.read_text() for path in py_files)
    # Positive contract: the compute entrypoint is MeshingSession, and the
    # one sanctioned export touchpoint is the shared output guard.
    assert "MeshingSession" in combined
    assert "guard_output_path" in combined


# ---- increment 3: the 3-D viewport geometry endpoint -------------------


def test_geometry_empty_before_source(client: TestClient) -> None:
    # Nothing rendered yet -> a loud, empty 204 (never a 500).
    r = client.get("/geometry")
    assert r.status_code == 204
    assert r.content == b""


def test_geometry_after_source(client: TestClient) -> None:
    assert client.post("/source/synth", json={"shape": "straight_tube"}).status_code == 200
    body = client.get("/geometry").json()
    assert body["kind"] == "surface"
    assert body["n_frames"] == 0
    # The serialized vtk.js mesh state is present and well-formed.
    assert isinstance(body["state"], dict) and "mesh" in body["state"]


def test_geometry_volmesh_has_region_field(client: TestClient) -> None:
    _drive_to_volmesh(client)
    assert client.post("/step/volmesh", json={"n_axial": 30}).status_code == 200
    body = client.get("/geometry").json()
    assert body["kind"] == "volume"
    # extract_surface exposes the cell field the volmesh carries.
    assert "region" in body["fields"]
    assert body["field"] == "region"  # server picks the first field by default
    assert body["data_range"] and len(body["data_range"]) == 2


def test_geometry_cell_field_is_served_as_point_data_with_a_matching_label(client: TestClient) -> None:
    """Two viewport defects found on the demo path (27 Oct 2026), both in the
    display layer. (1) vue-vtk's mesh component (trame-vtk 2.11) colours by
    POINT data only: a cell-only field such as the wall mesh's ``region`` threw
    in vtk.js's mapScalars and killed the renderer (a black canvas from the
    wall mesh on until a reload), so a cell-only field is served as a point
    version. (2) trame-vtk compacts an integer field to the smallest dtype
    that holds its values but labels ``type`` from the source dtype (an
    all-ones int32 shipped one byte per cell under "Int32Array"), so the label
    must name the TypedArray of the bytes actually sent."""
    import base64

    import numpy as np

    from meshforge.web.viewport import _JS_TYPE_OF_DTYPE

    _drive_to_volmesh(client)
    assert client.post("/step/volmesh", json={"n_axial": 30}).status_code == 200
    body = client.get("/geometry", params={"field": "region"}).json()
    assert "region" in body["fields"]
    field = body["state"]["field"]
    assert field["name"] == "region" and field["location"] == "PointData"
    dtype = field["values"]["dtype"]
    assert field["type"] == _JS_TYPE_OF_DTYPE[dtype], (field["type"], dtype)
    n_values = len(base64.b64decode(field["values"]["bvals"])) // np.dtype(dtype).itemsize
    n_points = body["state"]["mesh"]["points"]["shape"][0]
    assert n_values == n_points, (n_values, n_points)
    assert body["data_range"] == [1.0, 1.0]  # the averaged constant is still 1


def test_viewport_display_copy_leaves_the_session_mesh_alone_and_labels_every_dtype() -> None:
    import base64

    import numpy as np
    import pyvista as pv

    from meshforge.web import viewport

    m = pv.Sphere(theta_resolution=8, phi_resolution=8)
    for dt in (np.uint8, np.int32, np.int64, np.float32, np.float64):
        m.cell_data["region"] = np.ones(m.n_cells, dtype=dt)
        f = viewport.serialize(m, "surface", field="region")["state"]["field"]
        assert f["location"] == "PointData", dt
        item = np.dtype(f["values"]["dtype"]).itemsize
        assert f["type"] == viewport._JS_TYPE_OF_DTYPE[f["values"]["dtype"]], dt
        assert len(base64.b64decode(f["values"]["bvals"])) // item == m.n_points, dt
        assert "region" not in m.point_data  # the display copy never touches the session's mesh
    # a point field is served as it is (no averaging, dtype kept)
    m.point_data["v"] = np.full(m.n_points, 0.5)
    f = viewport.serialize(m, "surface", field="v")["state"]["field"]
    assert f["location"] == "PointData" and f["type"] == "Float64Array"


def test_geometry_ghost_is_the_ai_outline_behind_artifacts_cut_from_it(client: TestClient) -> None:
    """The segment step keeps the whole AI outline on the session (``outline``)
    after the sac/crop steps replace the working surface; GET /geometry sends
    it in the ``ghost`` list behind every artifact that is not the outline
    itself, serialized once per outline. The context organs of the same run
    (``session.context``: the patient's own heart, D108 addendum 7) are
    further entries, drawn whenever the session holds them — behind the
    outline itself too. A session with neither (synth, a file) sends
    ``ghost: null``."""
    import pyvista as pv

    assert client.post("/source/synth", json={"shape": "straight_tube"}).status_code == 200
    session = client.app.state.holder.session
    assert session.outline is None and session.context == {}  # a synth source is not an AI outline
    assert client.get("/geometry").json()["ghost"] is None
    # stand in for the segment step: the current surface IS the outline
    session.outline = session.surface
    assert client.get("/geometry").json()["ghost"] is None  # not behind itself
    heart = pv.Sphere(radius=20.0, center=(0.0, 0.0, 80.0))
    session.context = {"heart": heart}
    [h] = client.get("/geometry").json()["ghost"]  # the heart shows even then
    assert h["kind"] == "heart" and h["label"] == "AI outline · heart"
    assert "mesh" in h["state"] and "field" not in h["state"]
    assert client.post("/step/centerline", json={}).status_code == 200  # the same source, cut further
    assert client.post("/step/volmesh", json={"n_axial": 30}).status_code == 200
    body = client.get("/geometry").json()
    aorta, heart_entry = body["ghost"]
    assert aorta["label"] == "AI outline · aorta" and aorta["kind"] == "aorta"
    assert heart_entry["kind"] == "heart"
    assert "mesh" in aorta["state"] and "points" in aorta["state"]["mesh"]
    assert "field" not in aorta["state"]  # one faint colour, never a colormap
    holder = client.app.state.holder
    assert holder.ghosts["aorta"][0] is session.outline and holder.ghosts["heart"][0] is heart
    first = holder.ghosts["aorta"][1]
    client.get("/geometry", params={"field": "region"})
    assert holder.ghosts["aorta"][1] is first  # serialized once per outline
    # no context (the heart was out of field): the outline alone, no stale cache
    session.context = {}
    [only] = client.get("/geometry").json()["ghost"]
    assert only["kind"] == "aorta" and set(holder.ghosts) == {"aorta"}
    # a new source drops the outline, the context and the ghosts with them
    assert client.post("/source/synth", json={"shape": "straight_tube"}).status_code == 200
    assert client.app.state.holder.session.outline is None
    assert client.get("/geometry").json()["ghost"] is None and holder.ghosts == {}


def test_atlas_lists_only_the_scan_the_session_loaded(client: TestClient, tmp_path: Path) -> None:
    """D108 addendum 6: GET /atlas is the 15-region catalogue, the modalities
    and the studies the session REALLY holds — none on a fresh session, none
    for a synthetic tube or an STL (not scans), one CT once the segment step
    loaded one (the AI outline it keeps is the marker), its body coverage
    declared only for the public demo CT, by checksum; any other CT is listed
    with its coverage unassigned. No date, no prior, no folder, no second
    patient, ever."""
    import json

    from meshforge import atlas as atlas_mod

    body = client.get("/atlas").json()
    assert [r["id"] for r in body["regions"]] == [rid for rid, _ in atlas_mod.REGIONS]
    assert [r["name"] for r in body["regions"]] == [name for _, name in atlas_mod.REGIONS]
    assert len(body["regions"]) == 15
    assert body["modalities"] == ["CT", "MRI", "X-ray", "US", "Angio"]
    assert body["studies"] == [] and body["patient"] is None and body["n_patients"] == 0
    assert client.post("/source/synth", json={"shape": "straight_tube"}).status_code == 200
    assert client.get("/atlas").json()["studies"] == []  # a synthetic tube is not a scan
    session = client.app.state.holder.session
    # what the segment step leaves behind: the outline it keeps, the CT's path and checksum
    session.outline = session.surface
    session.input_path = tmp_path / "ct.nii.gz"
    session.input_sha256 = "0" * 64
    body = client.get("/atlas").json()
    [study] = body["studies"]
    assert study["modality"] == "CT" and study["loaded"] is True and study["status"] == "current"
    assert study["date"] == "date withheld" and study["patient"] is None
    assert study["regions"] == [] and study["coverage"] == "unassigned"
    assert study["file"] == "ct.nii.gz" and str(tmp_path) not in json.dumps(body)  # no folder on screen
    assert body["n_patients"] == 1 and "unassigned" in body["patient"]
    # the public demo CT, by checksum: chest + torso, subject s1364, with its credit
    session.input_sha256 = atlas_mod.DEMO_CT_SHA256
    body = client.get("/atlas").json()
    [study] = body["studies"]
    assert study["regions"] == ["chest", "torso"] and study["coverage"] == "declared"
    assert study["description"] == "abdomen + pelvis + lower chest · 1.5 mm" and study["patient"] == "s1364"
    assert body["patient"] == "1 patient on file · s1364 · de-identified (TotalSegmentator v2, CC BY 4.0)"
    assert {r for s_ in body["studies"] for r in s_["regions"]} <= {r["id"] for r in body["regions"]}
    # a new source drops the outline and the study with it
    assert client.post("/source/synth", json={"shape": "straight_tube"}).status_code == 200
    assert client.get("/atlas").json() == {**body, "studies": [], "patient": None, "n_patients": 0}


def test_viewport_ghost_is_decimated_fieldless_and_leaves_the_outline_alone() -> None:
    import numpy as np
    import pyvista as pv

    from meshforge.web import viewport

    m = pv.Sphere(theta_resolution=80, phi_resolution=80)  # 12k triangles
    m.point_data["v"] = np.full(m.n_points, 0.5)
    n_cells, n_arrays = m.n_cells, len(m.array_names)
    state = viewport.serialize_ghost(m)
    assert "field" not in state
    polys = state["mesh"]["polys"]["shape"][0]
    assert polys < 4 * n_cells * 0.6  # decimated (4 ints per triangle)
    assert m.n_cells == n_cells and len(m.array_names) == n_arrays  # untouched


def test_geometry_artifact_serves_a_held_result_or_says_why_not(client: TestClient) -> None:
    """D112: GET /geometry?artifact=stress|cfd serialises a result the session
    HOLDS (display only) so the demo's layer switch can show either; 404
    when the session has no such result, 400 for an unknown name; the CFD
    layer opens on the pressure when the grid carries one."""
    from types import SimpleNamespace as NS

    import pyvista as pv

    assert client.get("/geometry?artifact=stress").status_code == 404
    assert client.get("/geometry?artifact=cfd").status_code == 404
    r = client.get("/geometry?artifact=bogus")
    assert r.status_code == 400 and "cfd" in r.json()["detail"] and "stress" in r.json()["detail"]
    state = client.get("/state").json()["artifacts"]
    assert state["aaa_stress"] is False and state["cfd"] is False
    holder = client.app.state.holder
    grid = pv.ImageData(dimensions=(6, 6, 6)).cast_to_unstructured_grid()
    grid.point_data["U"] = [[1.0, 0.0, 0.0]] * grid.n_points
    grid.point_data["p"] = list(range(grid.n_points))
    holder.session.cfd = NS(grid=grid)
    holder.session.aaa_stress = NS(grid=pv.Sphere().delaunay_3d())
    body = client.get("/geometry?artifact=cfd").json()
    assert body["label"] == "cfd results" and body["kind"] == "volume"
    assert body["field"] == "p" and set(body["fields"]) >= {"U", "p"}   # the layer's own default
    assert client.get("/geometry?artifact=cfd&field=U").json()["field"] == "U"  # a field still wins
    stress = client.get("/geometry?artifact=stress").json()
    assert stress["label"] == "wall stress (investigational)"
    state = client.get("/state").json()["artifacts"]
    assert state["aaa_stress"] is True and state["cfd"] is True
    assert holder.last_render is None  # nothing rendered "last": the held views leave it alone
    # the CURRENT artifact, when it is a CFD result without a section, opens on p too
    holder.last_render = {"mesh": grid, "kind": "volume", "label": "cfd results"}
    assert client.get("/geometry").json()["field"] == "p"
    holder.last_render = {"mesh": grid, "kind": "volume", "label": "cfd results", "section": None}
    assert client.get("/geometry?field=U").json()["field"] == "U"
    # the colour range of a CFD seen from outside is the 2nd-98th percentile, not min/max
    import numpy as np

    from meshforge.web import viewport

    grid.point_data["p"] = np.r_[np.linspace(0.0, 10.0, grid.n_points - 1), 1000.0]  # one stagnation spike
    body = client.get("/geometry").json()
    assert body["field"] == "p" and body["range_note"] == "2nd–98th percentile"
    assert body["data_range"][1] < 100.0 and body["data_range"][0] >= 0.0
    assert viewport.robust_range(grid, "volume", "U") is None  # a flat field: no range
    assert viewport.robust_range(grid, "volume", "nope") is None
    holder.last_render = None


def test_geometry_unknown_field_is_loud_400(client: TestClient) -> None:
    assert client.post("/source/synth", json={"shape": "straight_tube"}).status_code == 200
    _drive_to_volmesh(client)
    client.post("/step/volmesh", json={"n_axial": 30})
    r = client.get("/geometry", params={"field": "not_a_field"})
    assert r.status_code == 400
    assert "unknown field" in r.json()["detail"]


def test_geometry_frame_endpoint_bounds() -> None:
    # Inject a tiny synthetic 2-frame series (no OpenFOAM solve in CI): the
    # frame endpoint's index handling is what we exercise, headless.
    import pyvista as pv

    app = create_app()
    client = TestClient(app)
    m0 = pv.Sphere()
    m0["f"] = m0.points[:, 0]
    m1 = pv.Sphere()
    m1["f"] = m1.points[:, 1]
    app.state.holder.last_render = {
        "mesh": m0,
        "kind": "surface",
        "label": "cfd",
        "frames": [(0.0, m0), (1.0, m1)],
    }
    meta = client.get("/geometry").json()
    assert meta["n_frames"] == 2 and meta["times"] == [0.0, 1.0]

    f0 = client.get("/geometry/frame/0").json()
    assert f0["index"] == 0 and f0["time"] == 0.0 and "mesh" in f0["state"]
    assert client.get("/geometry/frame/1").status_code == 200
    # out of range -> loud 400, no IndexError/traceback leak.
    over = client.get("/geometry/frame/5")
    assert over.status_code == 400 and "out of range" in over.json()["detail"]


def test_geometry_frame_404_without_series(client: TestClient) -> None:
    assert client.post("/source/synth", json={"shape": "straight_tube"}).status_code == 200
    r = client.get("/geometry/frame/0")
    assert r.status_code == 404


def test_vendor_assets_served_and_locked(client: TestClient) -> None:
    # The two viewport vendor files are served read-only from the installed
    # trame packages; any other name is refused (the dict lookup makes path
    # traversal structurally impossible).
    for name in ("vue.global.js", "trame-vtk.js"):
        r = client.get(f"/vendor/{name}")
        assert r.status_code == 200, name
        assert "javascript" in r.headers["content-type"]
        assert len(r.content) > 0
    assert client.get("/vendor/bogus.js").status_code == 404
    assert client.get("/vendor/app.py").status_code == 404


def test_viewport_module_is_rule4_clean() -> None:
    # The display module is scanned by the rule-4 guard (globs web/*.py); assert
    # it exists and calls the serializer, not any meshing internal.
    import meshforge.web as web_pkg

    vp = Path(web_pkg.__file__).parent / "viewport.py"
    assert vp.is_file(), "web/viewport.py missing"
    src = vp.read_text()
    assert "vtk_mesh" in src  # the display serializer it delegates to
    for token in ("offset_wall", "mesh_wall(", "get_backend", "meshforge.synth"):
        assert token not in src


def test_frontend_viewport_wiring() -> None:
    # The Workbench JS drives the geometry endpoint and the viewport glue holds
    # no meshing logic (rule-4 mirror for the new static files).
    app_js = (_static_dir() / "app.js").read_text()
    assert "/geometry" in app_js
    assert "/geometry" in {r.path for r in create_app().routes} or any(
        r.path.startswith("/geometry") for r in create_app().routes
    )
    vp_js = (_static_dir() / "viewport.js").read_text()
    for forbidden in ("meshforge.geometry", "meshforge.meshing", "offset_wall"):
        assert forbidden not in vp_js


# ---- session-34 item 1: contour QC overlay + auto-refresh ----------------


def test_state_rev_bumps_on_every_session_change(client: TestClient, tmp_path: Path) -> None:
    """The monotonic /state rev is the frontend's auto-refresh signal: every
    state-changing call bumps it; read-only calls (and a work-file SAVE,
    which mutates nothing) do not."""
    r0 = client.get("/state").json()["rev"]
    assert client.post("/source/synth", json={"shape": "straight_tube"}).status_code == 200
    r1 = client.get("/state").json()["rev"]
    assert r1 > r0
    assert client.post("/step/centerline", json={}).status_code == 200
    r2 = client.get("/state").json()["rev"]
    assert r2 > r1
    # reads do not bump
    client.get("/journal")
    client.get("/geometry")
    assert client.get("/state").json()["rev"] == r2
    # a work-file save mutates nothing -> no bump
    wf = tmp_path / "s.mfwork"
    assert client.post("/workfile/save", json={"path": str(wf)}).status_code == 200
    assert client.get("/state").json()["rev"] == r2
    # reset and work-file load both swap the session -> both bump
    assert client.post("/session/reset", json={}).status_code == 200
    r3 = client.get("/state").json()["rev"]
    assert r3 > r2
    assert client.post(
        "/workfile/load", json={"path": str(wf), "replay": True}
    ).status_code == 200
    assert client.get("/state").json()["rev"] > r3


def test_frontend_autorefresh_and_overlay_wiring() -> None:
    """Session-34 item 1: the Workbench polls the session revision and paints
    the QC overlay client-side; the glue stays meshing-free (rule-4 mirror)."""
    app_js = (_static_dir() / "app.js").read_text()
    assert "startAutoRefresh" in app_js
    assert 'apiGet("/state")' in app_js
    assert "setupOverlay" in app_js
    vp_js = (_static_dir() / "viewport.js").read_text()
    assert "setOverlay" in vp_js
    html = (_static_dir() / "index.html").read_text()
    assert "vp-overlay" in html
    for js in (app_js, vp_js):
        for forbidden in ("meshforge.geometry", "meshforge.meshing", "offset_wall"):
            assert forbidden not in js


# ---- session 39: a fresh clone must be told what to install ---------------


def _block_imports(monkeypatch, *prefixes):
    """Make `import <prefix>[...]` raise ImportError, as it does on a machine
    where only the BASE deps are installed."""
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            raise ImportError(f"no module named {name}")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)


def test_serve_without_the_web_extra_names_the_fix(monkeypatch) -> None:
    """`conda env create -f environment.yml` installs the base deps only, and
    `-e .` pulls no extras — so a fresh clone reaches serve() with no FastAPI.
    That must name the install, not raise a bare ModuleNotFoundError."""
    import sys

    from meshforge.web import server

    for mod in [m for m in sys.modules if m.startswith(("uvicorn", "fastapi"))]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.delitem(sys.modules, "meshforge.web.app", raising=False)
    _block_imports(monkeypatch, "uvicorn", "fastapi")

    with pytest.raises(RuntimeError) as exc:
        server.serve(port=1)
    msg = str(exc.value)
    assert "[web] extra" in msg
    assert "pip install -e '.[web]'" in msg
    assert "trame-vtk" in msg  # the viewport is a separate, conda-forge install
    assert isinstance(exc.value.__cause__, ImportError)  # chained, not swallowed


# ---- M6: the twin card panel (the copilot report's header, live) --------


def _copilot_journal(pws=(0.282481, 0.242127), whatif_pws=None) -> list[dict]:
    """A canned session journal with the step summaries' real shapes (the
    s1364 copilot path, after the gate recovery)."""
    sys_mpa, wif_mpa = pws
    idx = lambda p: {"p10": p / 1.299236, "p50": p / 0.6503, "p90": p / 0.3081114}  # noqa: E731
    ti = lambda p: {"step": "tissue_index", "params": {"pws_mpa": p},  # noqa: E731
                    "summary": {"index": idx(p), "n_aneurysms": 7, "n_strips": 13,
                                "strength_mpa": {"p10": 0.308, "p50": 0.65, "p90": 1.299}}}
    return [
        {"step": "segment", "params": {"path": "/somewhere/private/s1364/ct.nii.gz"},
         "summary": {"segment": {"totalseg": {"wall_time_s": 25.4}, "mask_volume_ml": 332.5}}},
        {"step": "sac", "params": {"path": "/somewhere/private/s1364/ct.nii.gz"},
         "summary": {"sac": {"max_diameter": {"max_caliper": {"value_mm": 61.75},
                                              "area_equivalent": {"value_mm": 52.17}},
                             "sac_volume_ml": 161.8, "resolved_fraction": 0.661,
                             "thickness_mm": {"all_rays": {"median": 3.17, "max": 25.52}}}}},
        {"step": "crop", "params": {"parallel_caps": 1},
         "summary": {"crop": {"landmarks": {"sac_d_equiv_mm": 51.8, "neck_d_equiv_mm": 24.5,
                                            "sac_neck_ratio": 2.11}}}},
        {"step": "volmesh", "params": {"sweep": "planes", "thickness": 1.5},
         "summary": {"n_cells": 7552, "quality": {"n_inverted": 0},
                     "lattice": {"n_capped": 0, "n_floored": 0, "sweep": "planes",
                                 "thickness_normal_median": 1.5014}}},
        {"step": "aaa_stress", "params": {"systolic_mmhg": 140.0, "whatif_mmhg": 120.0},
         "summary": {"refused": [], "systolic": {"von_mises_p99_mpa": sys_mpa, "peak": PEAK_OK},
                     "whatif": {"von_mises_p99_mpa": wif_mpa, "peak": PEAK_OK},
                     "whatif_ratio": {"pws": 0.857}, "thrombus": {"modelled": True}}},
        ti(sys_mpa), ti(whatif_pws if whatif_pws is not None else wif_mpa),
    ]


#: D96: a real aaa_stress summary reports the wall under the peak (median, min of
#: its top-1% region) per pressure; the twin replay runs the same gate as the copilot
PEAK_OK = {"top1pct_wall_thickness_mm": {"median": 1.5, "min": 1.4}}


def _slots(twin: dict) -> dict[str, dict]:
    return {s["key"]: s for s in twin["slots"]}


def test_twin_waits_before_a_run_then_fills_from_the_session(client: TestClient, tmp_path: Path) -> None:
    """Before any step every slot waits (no numbers); a loaded journal fills
    them from its own step outputs (no replay, nothing recomputed); a reset
    empties the card again."""
    import json

    before = client.get("/twin")
    assert before.status_code == 200
    twin = before.json()
    assert twin["lit"] is False and twin["scan"] is None and twin["title"] == "Your twin · v0"
    assert [s["state"] for s in twin["slots"]] == ["pending"] * 6 + ["next"]
    assert all(not s["values"] for s in twin["slots"])
    assert [m["name"] for m in twin["modules"]] == ["aorta", "heart", "kidneys"]

    wf = tmp_path / "copilot.mfwork"
    wf.write_text(json.dumps({"version": 1, "journal": _copilot_journal()}))
    r = client.post("/workfile/load", json={"path": str(wf), "replay": False})
    assert r.status_code == 200, r.text
    twin = client.get("/twin").json()
    slots = _slots(twin)
    assert twin["lit"] is True and twin["scan"] == "ct.nii.gz"  # the file name only
    assert "/somewhere" not in json.dumps(twin)
    assert slots["shape"]["values"] == ["sac 61.8 mm at its widest", "sac volume 161.8 mL"]
    assert slots["wall"]["values"][-1] == "wall 1.5 mm · gate PASS"
    assert slots["stress"]["values"] == ["282.5 kPa peak at 140 mmHg"]
    assert slots["whatif"]["values"] == ["242.1 kPa at 120 mmHg (-14.3 %)", "index P50 0.37"]
    assert slots["strength"]["values"] == ["index P50 0.43 (P10 0.22 – P90 0.92) at 140 mmHg"]
    assert slots["flow"]["state"] == "pending" and slots["growth"]["state"] == "next"

    assert client.post("/session/reset", json={}).status_code == 200
    assert client.get("/twin").json()["lit"] is False


def test_twin_never_fails_the_workbench_on_an_odd_summary(client: TestClient) -> None:
    """A zero stress (or an index run on another stress) leaves its slot
    waiting; the panel itself never turns into a 500."""
    holder = client.app.state.holder
    holder.session.journal = _copilot_journal(pws=(0.0, 0.0))
    r = client.get("/twin")
    assert r.status_code == 200
    assert _slots(r.json())["stress"]["state"] == "pending"
    holder.session.journal = _copilot_journal(whatif_pws=0.2)
    slots = _slots(client.get("/twin").json())
    assert slots["stress"]["state"] == "filled" and slots["strength"]["state"] == "pending"


# ---- D97: the copilot's flow metrics, journaled in the session ------------

S1364_FLOW = {"q_lmin": 1.0, "umax_mm_s": 123.74993133544922,
              "sac_backflow_volume_fraction": 0.14450368879518044,
              "sac_mean_speed_mm_s": 29.72977666215793, "neck_mean_speed_mm_s": 57.14323100196482}
FLOW_LINES = ["14% of the sac volume flows backward", "inflow 1 L/min"]


def test_twin_flow_route_fills_the_panel_with_the_reports_lines(client: TestClient, tmp_path: Path) -> None:
    """POST /twin/flow journals the metrics as their own record (nothing is
    computed) and bumps the revision, so the panel repaints with the exact
    lines the copilot's report prints for the same metrics."""
    import json

    from meshforge.copilot.report import render_report, tools_from_journal, twin_card

    wf = tmp_path / "copilot.mfwork"
    wf.write_text(json.dumps({"version": 1, "journal": _copilot_journal()}))
    assert client.post("/workfile/load", json={"path": str(wf), "replay": False}).status_code == 200
    assert _slots(client.get("/twin").json())["flow"]["note"] == "not run here"
    rev = client.get("/state").json()["rev"]

    r = client.post("/twin/flow", json={"metrics": S1364_FLOW})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["step"] == "flow_metrics" and body["summary"]["metrics"] == S1364_FLOW
    assert "nothing computed" in body["text"]
    assert client.get("/state").json()["rev"] > rev  # the panel's repaint signal
    journal = client.get("/journal").json()["journal"]
    assert journal[-1]["step"] == "flow_metrics" and journal[-1]["params"] == {"metrics": S1364_FLOW}

    twin = client.get("/twin").json()
    flow = _slots(twin)["flow"]
    assert flow["state"] == "filled" and flow["values"] == FLOW_LINES and flow["note"] == ""
    tools, _, _ = tools_from_journal(_copilot_journal())
    assert twin["slots"] == twin_card(tools.results, S1364_FLOW, tools.calls)["slots"]
    _, page = render_report(request="r", ct_name="ct.nii.gz", credit="c", results=tools.results,
                            calls=tools.calls, summary_text=None, guard_note="g", model="m",
                            flow=S1364_FLOW)
    for line in FLOW_LINES:
        assert line in page
    assert client.get("/state").json()["artifacts"]["cfd"] is False  # nothing was run


@pytest.mark.parametrize("body", [
    {"metrics": {"q_lmin": "1.0"}},  # numbers only, never coerced
    {"metrics": {"q_lmin": True}},
    {"metrics": {"q_lmin": None, "sac_backflow_volume_fraction": 0.1}},
    {"metrics": {"q_lmin": [1.0]}},
    {"metrics": {}},
    {"metrics": {"umax_mm_s": 123.7}},  # none of the card's metrics
    {"metrics": {"sac_backflow_volume_fraction": 14.45}},  # a percent, not a fraction
    {"metrics": {"q_lmin": -1.0}},
    {"metrics": {"q_lmin": 1.0, "../x": 2.0}},
    {"metrics": {"q_lmin": 1.0, **{f"m{i}": 1.0 for i in range(40)}}},  # bounded size
    {"metrics": [1.0, 0.14]},
    {"q_lmin": 1.0},
    {},
])
def test_twin_flow_refuses_a_malformed_payload_and_changes_nothing(client: TestClient, body) -> None:
    assert client.post("/source/synth", json={"shape": "straight_tube"}).status_code == 200
    before = (client.get("/journal").json(), client.get("/state").json()["rev"])
    r = client.post("/twin/flow", json=body)
    assert 400 <= r.status_code < 500, r.text
    assert (client.get("/journal").json(), client.get("/state").json()["rev"]) == before
    assert _slots(client.get("/twin").json())["flow"]["state"] == "pending"


def test_twin_flow_refuses_a_non_finite_number(client: TestClient) -> None:
    r = client.post("/twin/flow", content=b'{"metrics": {"q_lmin": NaN}}',
                    headers={"Content-Type": "application/json"})
    assert 400 <= r.status_code < 500 and client.get("/journal").json()["journal"] == []


def test_twin_flow_survives_a_workfile_round_trip_with_replay_on_and_off(
    client: TestClient, tmp_path: Path
) -> None:
    """The work file keeps the record; a replay re-records it (no-op, nothing
    recomputed, no crash); a load without replay shows it straight away."""
    import json

    assert client.post("/source/synth", json={"shape": "straight_tube"}).status_code == 200
    assert client.post("/twin/flow", json={"metrics": S1364_FLOW}).status_code == 200
    wf = tmp_path / "flow.mfwork"
    assert client.post("/workfile/save", json={"path": str(wf)}).status_code == 200
    saved = json.loads(wf.read_text())["journal"]
    assert [rec["step"] for rec in saved] == ["generate_synth", "flow_metrics"]

    for replay in (True, False):
        assert client.post("/session/reset", json={}).status_code == 200
        assert _slots(client.get("/twin").json())["flow"]["state"] == "pending"
        r = client.post("/workfile/load", json={"path": str(wf), "replay": replay})
        assert r.status_code == 200, r.text
        res = r.json()
        assert [rec["step"] for rec in res["journal"]] == ["generate_synth", "flow_metrics"]
        assert res["journal"][-1]["params"] == {"metrics": S1364_FLOW}
        if replay:
            assert [x["step"] for x in res["results"]] == ["generate_synth", "flow_metrics"]
            assert "nothing computed" in res["results"][-1]["text"]
            assert client.get("/state").json()["artifacts"]["surface"] is True
        else:
            assert res["results"] == []
        assert _slots(client.get("/twin").json())["flow"]["values"] == FLOW_LINES

    # a work file whose record the session would refuse fails loudly on load
    bad = tmp_path / "bad.mfwork"
    saved[-1]["params"]["metrics"]["q_lmin"] = "1"
    bad.write_text(json.dumps({"version": 1, "journal": saved}))
    for replay in (True, False):
        r = client.post("/workfile/load", json={"path": str(bad), "replay": replay})
        assert r.status_code == 400 and "q_lmin must be a number" in r.json()["detail"]


def test_twin_flow_is_serialised_like_every_journal_writer(client, monkeypatch) -> None:
    """Mirror of the compute-vs-poll test: POST /twin/flow writes the journal,
    so it waits for a running step; GET /twin (read-only) does not."""
    import threading

    assert client.post("/source/synth", json={"shape": "straight_tube"}).status_code == 200
    session = client.app.state.holder.session
    started, release = threading.Event(), threading.Event()
    real = session.compute_centerline

    def slow(**kw):
        started.set()
        assert release.wait(20)
        return real(**kw)

    monkeypatch.setattr(session, "compute_centerline", slow)
    step = threading.Thread(target=lambda: client.post("/step/centerline", json={}))
    step.start()
    assert started.wait(20)
    got: list[int] = []
    flow = threading.Thread(
        target=lambda: got.append(client.post("/twin/flow", json={"metrics": S1364_FLOW}).status_code)
    )
    flow.start()
    flow.join(1.0)
    assert flow.is_alive() and not got  # waits for the step to finish
    assert client.get("/twin").status_code == 200  # the poll is not held up
    assert [rec["step"] for rec in session.journal] == ["generate_synth"]
    release.set()
    step.join(20)
    flow.join(20)
    assert got == [200]
    assert [rec["step"] for rec in session.journal] == ["generate_synth", "centerline", "flow_metrics"]


def test_state_reports_the_running_step_while_it_holds_the_lock(client, monkeypatch) -> None:
    """D108: GET /state carries ``running`` (a STEP_KNOBS name) and
    ``running_for`` (seconds) while a step runs under the session lock, and
    both are None otherwise — so a Workbench driven from outside (the copilot
    over HTTP) can show "meshing… 12 s"."""
    from meshforge.pipeline import MeshingSession

    holder = client.app.state.holder
    assert client.get("/state").json()["running"] is None
    assert client.get("/state").json()["running_for"] is None

    seen: dict[str, object] = {}

    def fake_centerline(self, **params):
        seen["running"] = holder.running
        seen["since_is_set"] = holder.running_since is not None
        raise ValueError("stop here (no artifact needed)")

    monkeypatch.setattr(MeshingSession, "compute_centerline", fake_centerline)
    r = client.post("/step/centerline", json={})
    assert r.status_code == 400  # the domain error is mapped, loudly
    assert seen == {"running": "centerline", "since_is_set": True}
    # cleared in `finally`, even on the error path
    st = client.get("/state").json()
    assert st["running"] is None and st["running_for"] is None
