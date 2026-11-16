"""CalculiX ASCII .frd result reader.

Parses the nodal block (``2C``), the element block (``3C``) and every nodal
result block (``100CL`` / ``-4`` / ``-5`` / ``-1``) into numpy arrays. Fields
are fixed-width: ``-1`` data lines carry an I10 id followed by E12.5 values,
with ``-2`` continuation lines for fields wider than six components.

Only element types meshforge writes are mapped (he8 -> VTK hexahedron,
te4 -> VTK tetra); anything else fails loudly with the frd type name.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from meshforge.textguard import scrub

# frd element type code -> (name, n_nodes, vtk cell type). frd node order for
# he8/te4 equals the Abaqus/VTK order meshforge writes (validated by the
# ccx-gated Lame benchmark reading back what write_deck produced).
_FRD_ELEMENT_TYPES = {
    1: ("he8", 8, 12),   # VTK_HEXAHEDRON
    3: ("te4", 4, 10),   # VTK_TETRA
}
_KNOWN_UNSUPPORTED = {
    2: "pe6", 4: "he20", 5: "pe15", 6: "te10",
    7: "tr3", 8: "tr6", 9: "qu4", 10: "qu8", 11: "be2", 12: "be3",
}


@dataclass
class FrdField:
    """One nodal result block: (n_nodes, n_components) values."""

    name: str
    components: tuple[str, ...]
    values: np.ndarray  # aligned with FrdModel.node_ids


@dataclass
class FrdModel:
    node_ids: np.ndarray            # (N,) 1-based solver ids, ascending
    points: np.ndarray              # (N, 3)
    cell_types: np.ndarray | None   # VTK cell types, None when no 3C block
    cells: np.ndarray | None        # VTK-style [n, i0..in-1, ...] 0-based
    fields: dict[str, FrdField] = field(default_factory=dict)
    # V31: EVERY parsed block per field, [(100CL value, packed values)] in
    # file order — transient steps carry the step time there (probe: token
    # 3 of the 100CL line). ``fields`` keeps the last block, unchanged.
    history: dict[str, list[tuple[float, np.ndarray]]] = field(
        default_factory=dict
    )
    notes: list[str] = field(default_factory=list)


def _values_from(line: str, start: int) -> list[float]:
    """Split a fixed-width frd line into 12-char floats from ``start``."""
    out = []
    for i in range(start, len(line.rstrip("\n")), 12):
        chunk = line[i : i + 12].strip()
        if not chunk:
            break
        out.append(float(chunk))
    return out


def read_frd(path: str | Path, *, elements: bool = True) -> FrdModel:
    """Parse a ccx .frd. ``elements=False`` skips the element block (3C)
    unread — for a caller that needs only nodal fields of a deck whose
    element type has no VTK mapping here (D88: the AAA C3D20R wall); the
    model then carries ``cells=None`` and a note saying so."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"frd result file not found: {path}")
    lines = path.read_text().splitlines()

    node_ids: list[int] = []
    coords: list[list[float]] = []
    elem_types: list[int] = []
    elem_conn: list[list[int]] = []
    fields: dict[str, FrdField] = {}
    # (name, 100CL value, raw {node id: values}) per block, in file order.
    history_raw: list[tuple[str, float, dict[int, list[float]]]] = []
    current_time: float | None = None
    notes: list[str] = []

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        key = line[:6].strip()
        if key == "2C":  # nodal coordinates
            i += 1
            while i < n and not lines[i].startswith(" -3"):
                row = lines[i]
                if row.startswith(" -1"):
                    node_ids.append(int(row[3:13]))
                    coords.append(_values_from(row, 13)[:3])
                i += 1
        elif key == "3C" and not elements:  # skipped unread (D88)
            i += 1
            while i < n and not lines[i].startswith(" -3"):
                i += 1
            notes.append("element block skipped (elements=False): nodal fields only")
        elif key == "3C":  # elements
            i += 1
            while i < n and not lines[i].startswith(" -3"):
                row = lines[i]
                if row.startswith(" -1"):
                    etype = int(row[13:18])
                    if etype not in _FRD_ELEMENT_TYPES:
                        name = _KNOWN_UNSUPPORTED.get(etype, f"type {etype}")
                        raise ValueError(
                            f"unsupported frd element type {etype} ({name}) in "
                            f"{path.name}; meshforge reads he8 and te4 only"
                        )
                    elem_types.append(etype)
                    n_nodes = _FRD_ELEMENT_TYPES[etype][1]
                    conn: list[int] = []
                    while len(conn) < n_nodes:
                        i += 1
                        if i >= n or not lines[i].startswith(" -2"):
                            raise ValueError(
                                f"truncated element connectivity in {path.name} "
                                f"(element {int(row[3:13])})"
                            )
                        conn.extend(
                            int(lines[i][j : j + 10])
                            for j in range(3, 3 + 10 * min(10, n_nodes - len(conn)), 10)
                            if lines[i][j : j + 10].strip()
                        )
                    elem_conn.append(conn[:n_nodes])
                i += 1
        elif line.startswith("  100CL"):  # result-set header carrying the time
            tokens = line.split()
            try:
                current_time = float(tokens[2])
            except (IndexError, ValueError) as exc:
                raise ValueError(
                    f"malformed 100CL header in {path.name}: {scrub(repr(line))} "
                    f"(expected the time/value as whitespace token 3)"
                ) from exc
        elif line.startswith(" -4"):  # nodal result block inside 100CL
            name = line[5:13].strip()
            ncomp_declared = int(line[13:18])
            comps: list[str] = []
            i += 1
            while i < n and lines[i].startswith(" -5"):
                comps.append(lines[i][5:13].strip())
                i += 1
            data: dict[int, list[float]] = {}
            while i < n and not lines[i].startswith(" -3"):
                row = lines[i]
                if row.startswith(" -1"):
                    nid = int(row[3:13])
                    vals = _values_from(row, 13)
                    while i + 1 < n and lines[i + 1].startswith(" -2"):
                        i += 1
                        vals.extend(_values_from(lines[i], 3))
                    data[nid] = vals
                i += 1
            if not data:
                raise ValueError(
                    f"result block {name!r} in {path.name} contains no data lines"
                )
            ncomp = len(next(iter(data.values())))
            # frd declares computed extras (e.g. DISP "ALL") that carry no
            # data column; trust the data width, keep matching names.
            comps = comps[:ncomp] if len(comps) >= ncomp else [
                f"C{k + 1}" for k in range(ncomp)
            ]
            if current_time is None:
                raise ValueError(
                    f"result block {name!r} in {path.name} has no preceding "
                    f"100CL header — no time/value to file it under"
                )
            history_raw.append((name, current_time, data))
            fields[name] = FrdField(
                name=name, components=tuple(comps), values=data  # packed below
            )
            _ = ncomp_declared
        i += 1

    if not node_ids:
        raise ValueError(f"no nodal block (2C) found in {path}")
    ids = np.asarray(node_ids, dtype=np.int64)
    order = np.argsort(ids)
    ids = ids[order]
    points = np.asarray(coords, dtype=float)[order]

    # Pack every block aligned with the sorted node ids; missing nodes are
    # loud. ``fields`` receives the last packed block per name (V13
    # behavior, unchanged); ``history`` keeps them all with their times.
    history: dict[str, list[tuple[float, np.ndarray]]] = {}
    for fname, t, data in history_raw:
        try:
            packed = [data[int(k)] for k in ids]
        except KeyError:
            packed = None
        if packed is None or len(data) != len(ids):
            raise ValueError(
                f"field {fname!r} covers {len(data)} of {len(ids)} nodes in {path.name}"
            )
        history.setdefault(fname, []).append(
            (t, np.asarray(packed, dtype=float))
        )
    for fname, f in fields.items():
        f.values = history[fname][-1][1]
        if len(history[fname]) > 1:
            notes.append(
                f"field {fname!r} repeats {len(history[fname])}x (multi-step "
                f"frd); fields keeps the last block, history keeps all"
            )

    cell_types = cells = None
    if elem_conn:
        id_to_index = {int(v): j for j, v in enumerate(ids)}
        try:
            zero_based = [[id_to_index[nid] for nid in conn] for conn in elem_conn]
        except KeyError as exc:
            raise ValueError(
                f"element references unknown node id {exc} in {path.name}"
            ) from exc
        cell_types = np.asarray(
            [_FRD_ELEMENT_TYPES[t][2] for t in elem_types], dtype=np.uint8
        )
        cells = np.concatenate(
            [np.concatenate(([len(c)], c)) for c in zero_based]
        ).astype(np.int64)

    return FrdModel(
        node_ids=ids, points=points, cell_types=cell_types, cells=cells,
        fields=fields, history=history, notes=notes,
    )


def von_mises(values: np.ndarray, components: tuple[str, ...]) -> np.ndarray:
    """Von Mises stress from a 6-component nodal tensor, by component NAME
    (frd order is SXX SYY SZZ SXY SYZ SZX — never assume, look it up)."""
    comp = {name.upper(): values[:, k] for k, name in enumerate(components)}
    try:
        sxx, syy, szz = comp["SXX"], comp["SYY"], comp["SZZ"]
        sxy, syz, szx = comp["SXY"], comp["SYZ"], comp["SZX"]
    except KeyError as exc:
        raise ValueError(
            f"stress tensor is missing component {exc}; have {list(comp)}"
        ) from exc
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy**2 + syz**2 + szx**2)
    )


def temperature_times(model: FrdModel) -> tuple[float, ...]:
    """The NDTEMP snapshot times recorded in a heat solve's frd (one per
    increment on a transient solve, V46). Loud when the frd carries none."""
    if "NDTEMP" not in model.fields:
        raise ValueError(
            "frd carries no temperature field (NDTEMP) — not a heat solve "
            "result; run analysis='heat' first and chain ITS .frd"
        )
    return tuple(float(t) for t, _vals in model.history["NDTEMP"])


def nodal_temperatures(
    model: FrdModel, *, time: float | None = None
) -> tuple[tuple[int, float], ...]:
    """(0-based node id, T) pairs from a temperature block of a heat solve's
    frd — the shape ``DeckSpec.nodal_temperatures`` takes (V45).

    ``time=None`` takes the LAST block (the V45 behavior). A given ``time``
    must match one of the frd's OWN recorded NDTEMP times (V46 — a transient
    history's increment times are irregular under adaptive stepping, so the
    caller can never assume k*dt); an unknown time refuses loudly listing
    the available times. NO interpolation, NO nearest-match: a
    silently-nearest snapshot would be silently wrong stress (rule 6).

    Fails loud when the frd carries no NDTEMP field (not a heat result) or a
    non-finite temperature (corrupt data must never become a load, rule 6).
    """
    if "NDTEMP" not in model.fields:
        raise ValueError(
            "frd carries no temperature field (NDTEMP) — not a heat solve "
            "result; run analysis='heat' first and chain ITS .frd"
        )
    if time is None:
        values = model.fields["NDTEMP"].values
    else:
        history = model.history["NDTEMP"]
        tol = 1e-9 * max(1.0, abs(float(time)))
        hits = [vals for t, vals in history if abs(float(t) - float(time)) <= tol]
        if not hits:
            avail = [f"{float(t):g}" for t, _ in history]
            shown = (
                ", ".join(avail)
                if len(avail) <= 12
                else ", ".join(avail[:6]) + f", … ({len(avail)} total) …, "
                + ", ".join(avail[-3:])
            )
            raise ValueError(
                f"frd has no NDTEMP snapshot at time {float(time):g} — "
                f"available times: {shown}. Pick one of the frd's own "
                f"recorded times exactly (no interpolation/nearest: a "
                f"silently-nearest snapshot would be silently wrong stress)"
            )
        values = hits[-1]
    temps = np.asarray(values[:, 0], dtype=float)
    if not np.all(np.isfinite(temps)):
        raise ValueError(
            "non-finite temperature (NaN/inf) in the frd NDTEMP field — "
            "refusing to build a thermal load from corrupt data"
        )
    return tuple(
        (int(nid) - 1, float(t)) for nid, t in zip(model.node_ids, temps)
    )


def frd_to_grid(model: FrdModel):
    """Build a pv.UnstructuredGrid from the element block, with every parsed
    field attached as point data (+ derived S_mises when stress is present)."""
    import pyvista as pv

    if model.cells is None:
        raise ValueError(
            "frd file has no element block (3C) — cannot rebuild a grid; "
            "solve with the mesh in-session to attach results directly"
        )
    grid = pv.UnstructuredGrid(model.cells, model.cell_types, model.points)
    attach_fields(grid, model)
    return grid


def attach_fields(grid, model: FrdModel) -> list[str]:
    """Attach parsed nodal fields to ``grid.point_data``; returns the names."""
    written = []
    for name, f in model.fields.items():
        arr = f.values[:, 0] if f.values.shape[1] == 1 else f.values
        grid.point_data[name] = arr
        written.append(name)
        if name.upper() == "DISP":
            grid.point_data["U_mag"] = np.linalg.norm(f.values[:, :3], axis=1)
            written.append("U_mag")
        if name.upper() == "STRESS" and f.values.shape[1] == 6:
            grid.point_data["S_mises"] = von_mises(f.values, f.components)
            written.append("S_mises")
    return written
