"""Display bridge for the web viewport (V35 increment 3).

Turns a session artifact's **already-computed** pyvista mesh (handed over in a
``StepResult.render_hints``) into the vtk.js state the browser canvas renders.

**Rule 4 — this is a DISPLAY concern, not meshing.** It imports none of the
core mesh-building modules (geometry / meshing / synth / solve) and *recomputes
nothing*: the mesh arrives fully built. The derivations it performs are
render-prep, never stored back on the session: ``extract_surface`` on a
*volume* grid (the visible boundary of a solid), and — when the session handed
over a flow section (D82, built in ``geometry/ribbon.py``) — probing the
displayed volume's values onto that already-built surface, so the viewport can
show the flow inside a vessel instead of its no-slip wall. ``trame_vtk`` is
**lazy-imported inside the functions** so importing the base web layer never
pulls trame (matches the CLI's discipline).

The client renders this state fully **client-side** (pure vtk.js via trame-vtk's
``vue_vtk`` bundle); there is **no WebSocket** and no server-side GL — the server
only serializes.
"""
from __future__ import annotations

from typing import Any

# vtk bookkeeping arrays created by extract_surface / the probe — not user fields.
_INTERNAL_ARRAYS = frozenset(
    {"vtkOriginalPointIds", "vtkOriginalCellIds", "vtkValidPointMask", "vtkGhostType"}
)


def _renderable(mesh: Any, kind: str) -> Any:
    """The visible surface to display. A *volume* grid shows its boundary
    (render-prep, not meshing); everything else renders as-is."""
    if kind == "volume":
        return mesh.extract_surface(algorithm="dataset_surface")
    return mesh


def available_fields(mesh: Any) -> list[str]:
    """Colour-by-field candidates on a mesh (point + cell arrays), minus the
    vtk internal bookkeeping arrays. De-duplicated, order-preserving — a field
    (e.g. CFD ``U``/``p``) can live in BOTH point and cell data."""
    seen: list[str] = []
    for name in list(mesh.point_data.keys()) + list(mesh.cell_data.keys()):
        if name not in _INTERNAL_ARRAYS and name not in seen:
            seen.append(name)
    return seen


# numpy dtype of the bytes actually shipped → the JS TypedArray that reads
# them. trame-vtk 2.11's ``vtk_mesh`` compacts an integer field to the
# smallest dtype that holds its values (an all-ones int32 ``region`` ships as
# one byte per cell) but labels ``type`` from the SOURCE dtype (Int32Array),
# so the browser read a quarter of the cells and vtk.js crashed its renderer
# (the canvas stayed black from the wall mesh on until a reload). Floats are
# shipped as-is and were never affected. Display passthrough only.
_JS_TYPE_OF_DTYPE = {
    "int8": "Int8Array", "uint8": "Uint8Array",
    "int16": "Int16Array", "uint16": "Uint16Array",
    "int32": "Int32Array", "uint32": "Uint32Array",
    "float32": "Float32Array", "float64": "Float64Array",
}


def _label_field_by_its_bytes(state: Any) -> Any:
    """Make a serialized state's ``field.type`` name the TypedArray of the
    bytes it carries (``field.values.dtype``). Returns the same state."""
    field = state.get("field") if isinstance(state, dict) else None
    values = field.get("values") if isinstance(field, dict) else None
    dtype = values.get("dtype") if isinstance(values, dict) else None
    js = _JS_TYPE_OF_DTYPE.get(str(dtype)) if dtype is not None else None
    if js is not None:
        field["type"] = js
    return state


def _point_field_for_display(surf: Any, field: str | None) -> Any:
    """vue-vtk's mesh component (trame-vtk 2.11) colours by POINT data only: a
    cell-only field (the wall mesh's ``region``, the stress mesh's end-zone
    mask, the cropped lumen's cap ids) threw inside vtk.js's ``mapScalars``
    and killed the renderer — the canvas stayed black from that artifact on
    until a page reload. Such a field gets a point version (cell → point
    average, float32) on a shallow display copy; the session's mesh and every
    point field are untouched."""
    if field is None or field in surf.point_data or field not in surf.cell_data:
        return surf
    import numpy as np

    shown = surf.copy(deep=False)  # its own attribute containers, shared arrays
    averaged = surf.cell_data_to_point_data(pass_cell_data=False)
    shown.point_data[field] = np.asarray(averaged.point_data[field], dtype=np.float32)
    return shown


def _state_range(state: Any) -> list[float] | None:
    field_meta = state.get("field") if isinstance(state, dict) else None
    rng = field_meta.get("dataRange") if field_meta else None
    return [float(rng[0]), float(rng[1])] if rng is not None else None


def section_view(mesh: Any, section: Any) -> Any:
    """The displayed volume's values on the session-built flow section.

    Every section point lies inside the volume by construction (D82), so the
    probe is total; the probe's own bookkeeping arrays are dropped."""
    probed = section.sample(mesh)
    for name in _INTERNAL_ARRAYS:
        if name in probed.point_data:
            del probed.point_data[name]
    return probed


def serialize(
    mesh: Any,
    kind: str = "surface",
    field: str | None = None,
    section: Any = None,
) -> dict[str, Any]:
    """``pyvista`` mesh → ``{kind, field, fields, data_range, state}`` for the
    vtk.js canvas (+ ``section`` when a flow section rides along).

    ``field`` selects the scalar coloured; ``None`` picks the first available
    field (or none, for a fieldless mesh). An unknown ``field`` raises
    ``ValueError`` (→ a loud HTTP 400 at the endpoint), never a silent miscolour.
    With a ``section`` the colour range spans the section too: the interior
    flow can exceed anything on the boundary.
    """
    surf = _renderable(mesh, kind)
    fields = available_fields(surf)
    if field is not None and field not in fields:
        raise ValueError(
            f"unknown field {field!r} for this artifact; available: {fields or '(none)'}"
        )
    if field is None and fields:
        field = fields[0]
    surf = _point_field_for_display(surf, field)

    # Lazy import: keeps `import meshforge.web.app` (and the CLI) trame-free.
    from trame_vtk.modules.vtk import vtk_mesh

    state = _label_field_by_its_bytes(vtk_mesh(surf, field_to_keep=field))
    data_range = _state_range(state)
    payload = {
        "kind": kind,
        "field": field,
        "fields": fields,
        "data_range": data_range,
        "state": state,
    }
    if section is not None and field is not None:
        probed = section_view(mesh, section)
        if field in probed.point_data:
            sec_state = _label_field_by_its_bytes(vtk_mesh(probed, field_to_keep=field))
            sec_range = _state_range(sec_state)
            if data_range and sec_range:
                payload["data_range"] = [
                    min(data_range[0], sec_range[0]),
                    max(data_range[1], sec_range[1]),
                ]
            payload["section"] = sec_state
            # A robust range of the flow INSIDE (2nd-98th percentile): a small
            # fast inflow otherwise pins the scale and paints the map one colour.
            import numpy as np

            vals = np.asarray(probed.point_data[field], dtype=float)
            if vals.ndim == 2:
                vals = np.linalg.norm(vals, axis=1)
            lo, hi = np.percentile(vals, [2.0, 98.0])
            if hi > lo:
                payload["section_range"] = [float(lo), float(hi)]
    return payload


def robust_range(mesh: Any, kind: str, field: str) -> list[float] | None:
    """The 2nd-98th percentile of ``field`` on the rendered surface (vector
    fields by magnitude), or None when it is flat or absent. Display only:
    a colour range, never a result."""
    import numpy as np

    surf = _point_field_for_display(_renderable(mesh, kind), field)
    data = surf.point_data if field in surf.point_data else surf.cell_data
    if field not in data:
        return None
    vals = np.asarray(data[field], dtype=float)
    if vals.ndim == 2:
        vals = np.linalg.norm(vals, axis=1)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return None
    lo, hi = np.percentile(vals, [2.0, 98.0])
    return [float(lo), float(hi)] if hi > lo else None


def serialize_ghost(mesh: Any, target_reduction: float = 0.6) -> dict[str, Any]:
    """The ghost outline → its bare vtk.js state: the triangulated surface
    decimated for transport (it is the whole aorta, the session's largest
    surface) with every field dropped — the canvas paints it in one faint
    translucent colour behind the artifact. Display passthrough only; the
    session's outline is not touched (decimate returns a new mesh)."""
    from trame_vtk.modules.vtk import vtk_mesh

    tri = mesh.triangulate() if mesh.n_cells else mesh
    ghost = tri.decimate(target_reduction) if tri.n_cells > 2000 else tri.copy(deep=False)
    ghost.clear_data()
    return vtk_mesh(ghost)


def serialize_overlay(mesh: Any) -> dict[str, Any]:
    """A QC overlay (e.g. the raw CSO contour polylines) → its bare vtk.js
    state.  Display passthrough only: no colour fields, no ranges — the canvas
    paints it in a fixed colour over the main artifact."""
    from trame_vtk.modules.vtk import vtk_mesh

    return vtk_mesh(mesh)
