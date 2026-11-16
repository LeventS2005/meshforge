"""Intuitive flow boundaries for session-built CFD cases (D80).

The ``cfd`` step's ``inlet`` knob used to take only a velocity VECTOR
(``NAME=vx,vy,vz``): the user had to work out the patch's inward normal by
hand, and a sign slip blew fluid OUT of the "inlet" without complaint. This
module holds the core of the friendlier forms; front-ends only reach it
through ``MeshingSession`` (rule 4):

* :func:`patch_geometry` - area, centroid and INWARD unit normal of one
  boundary patch, measured on the mesh. Every face is oriented by its parent
  cell's centroid, never by its winding (``BoundaryGroup`` is node-set based;
  face order is not a contract).
* :func:`resolve_inlet_speed` - the scalar knob form ``NAME=SPEED``: speed x
  inward normal as ONE uniform vector, i.e. exactly the ``fixedValue`` the
  vector form writes. Refused on a patch that is not a flat cap - a curved
  patch has no single inflow direction.
* :func:`inflow_report` / :func:`inflow_notes` - area, equivalent diameter,
  volumetric flow and Reynolds number for whatever velocity was prescribed.
  Session 44's live demo prescribed a plausible 160 mm/s that carried 3.9 %
  of a resting cardiac output; Q makes that visible at a glance.
* :func:`flow_ends` - the Workbench picker: the flat cap nearest a picked
  point is the inflow, the other cap the outflow.
* :func:`cardiac_presets` - blood viscosity and aortic speeds in the model's
  length unit. An unknown unit is refused, never guessed: a millimetre model
  fed SI values is off by 10^3 in speed and 10^6 in viscosity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# A patch whose face vector-areas sum to at least this fraction of its total
# area is a flat cap: a straight cut through a vessel scores 1.0, a whole tube
# wall ~0. Below it the mean normal is not a flow direction.
MIN_CAP_PLANARITY = 0.9

# Above this pipe-flow transition Reynolds number a steady laminar solution
# may not exist (session 44: Re ~ 6300 stalled simpleFoam for 2000 iterations).
LAMINAR_RE_LIMIT = 2300.0


@dataclass(frozen=True)
class PatchGeometry:
    """Measured geometry of one boundary patch."""

    name: str
    n_faces: int
    area: float
    centroid: tuple[float, float, float]
    normal_in: tuple[float, float, float]  # unit, pointing INTO the domain
    planarity: float  # |sum of face vector areas| / sum of face areas

    @property
    def d_eq(self) -> float:
        """Equivalent diameter: the circle of the same area."""
        return 2.0 * math.sqrt(self.area / math.pi)

    @property
    def is_cap(self) -> bool:
        return self.planarity >= MIN_CAP_PLANARITY

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_faces": self.n_faces,
            "area": self.area,
            "d_eq": self.d_eq,
            "centroid": list(self.centroid),
            "normal_in": list(self.normal_in),
            "planarity": self.planarity,
        }


def patch_geometry(wall, name: str) -> PatchGeometry:
    """Area, centroid, inward normal and planarity of boundary patch *name*.

    Each face's Newell vector area is flipped, if needed, to point away from
    its parent cell's centroid (outward); the patch's outward normal is the
    normalised sum, and the inward normal its negation.
    """
    import numpy as np

    if name not in wall.boundary:
        raise ValueError(
            f"no boundary patch {name!r} on this mesh; patches: "
            f"{sorted(wall.boundary)}"
        )
    group = wall.boundary[name]
    if group.n_faces == 0:
        raise ValueError(f"boundary patch {name!r} has no faces")
    points = np.asarray(wall.grid.points, dtype=float)
    poly = points[np.asarray(group.face_nodes)]  # (k, m, 3)
    vec = 0.5 * np.cross(poly, np.roll(poly, -1, axis=1)).sum(axis=1)
    centres = poly.mean(axis=1)
    cells = np.asarray(wall.grid.cell_centers().points)[np.asarray(group.cell_ids)]
    side = np.einsum("ij,ij->i", vec, centres - cells)
    if np.any(side == 0.0):
        raise ValueError(
            f"boundary patch {name!r} has a degenerate face (zero area, or "
            f"its parent cell centroid lies in the face plane)"
        )
    vec = vec * np.sign(side)[:, None]
    areas = np.linalg.norm(vec, axis=1)
    area = float(areas.sum())
    total = vec.sum(axis=0)
    norm = float(np.linalg.norm(total))
    normal_in = tuple(float(c) for c in -total / norm) if norm > 0.0 else (0.0, 0.0, 0.0)
    centroid = tuple(float(c) for c in (centres * areas[:, None]).sum(axis=0) / area)
    return PatchGeometry(
        name=name,
        n_faces=int(group.n_faces),
        area=area,
        centroid=centroid,
        normal_in=normal_in,
        planarity=norm / area,
    )


def check_inlet_speed(speed: float, name: str) -> float:
    """Loud unless *speed* is a positive finite inflow speed."""
    if not (math.isfinite(speed) and speed > 0.0):
        raise ValueError(
            f"inlet speed must be a positive finite number - it flows INTO the "
            f"domain along {name!r}'s inward normal - got {speed!r}; to reverse "
            f"the flow, swap the inlet and outlet patches instead"
        )
    return float(speed)


def resolve_inlet_speed(
    wall, name: str, speed: float
) -> tuple[tuple[float, float, float], PatchGeometry]:
    """``NAME=SPEED`` -> (velocity vector, patch geometry): speed along the
    patch's inward normal, uniform over the patch."""
    speed = check_inlet_speed(speed, name)
    geo = patch_geometry(wall, name)
    if not geo.is_cap:
        raise ValueError(
            f"patch {name!r} is not a flat end cap (planarity "
            f"{geo.planarity:.3f} < {MIN_CAP_PLANARITY}): it has no single "
            f"inflow direction - give the inlet as a vector NAME=vx,vy,vz"
        )
    # Components below 1e-12 of the speed are round-off of an axis-aligned
    # normal; zero them so the written case reads (0 0 1), not (1e-17 ...).
    vector = tuple(
        0.0 if abs(c) < 1e-12 else speed * c for c in geo.normal_in
    )
    return vector, geo


# metres per model length unit
_METRES_PER_UNIT = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}


def inflow_report(
    geo: PatchGeometry,
    velocity,
    nu: float,
    length_unit: str | None = None,
) -> dict[str, Any]:
    """What a prescribed inlet velocity amounts to on its patch."""
    u_n = float(sum(v * n for v, n in zip(velocity, geo.normal_in)))
    q = u_n * geo.area
    report: dict[str, Any] = {
        "patch": geo.name,
        "velocity": [float(v) for v in velocity],
        "normal_in": list(geo.normal_in),
        "normal_speed": u_n,
        "area": geo.area,
        "d_eq": geo.d_eq,
        "planarity": geo.planarity,
        "flow_rate": q,
        "reynolds": abs(u_n) * geo.d_eq / nu,
    }
    if length_unit in _METRES_PER_UNIT:
        # model unit^3/s -> m^3/s -> L/min
        report["flow_rate_l_per_min"] = q * _METRES_PER_UNIT[length_unit] ** 3 * 6e4
    return report


def inflow_notes(report: dict[str, Any], *, laminar: bool = True) -> list[str]:
    """Human-readable lines for a CFD result's notes."""
    n = [round(c, 4) + 0.0 for c in report["normal_in"]]  # no "-0.0000"
    q = f"Q {report['flow_rate']:.4g}"
    if "flow_rate_l_per_min" in report:
        q += f" ({report['flow_rate_l_per_min']:.3g} L/min)"
    notes = [
        f"inflow {report['patch']!r}: U.n {report['normal_speed']:.4g} along "
        f"inward n ({n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f}); area "
        f"{report['area']:.4g}, D_eq {report['d_eq']:.4g} -> {q}; "
        f"Re {report['reynolds']:.0f} (U.n x D_eq / nu)"
    ]
    if report["normal_speed"] <= 0.0:
        notes.append(
            f"WARNING: the inlet velocity on {report['patch']!r} does not point "
            f"into the domain (U.n = {report['normal_speed']:.4g}) - check its "
            f"sign, or give a scalar speed NAME=SPEED"
        )
    if report["planarity"] < MIN_CAP_PLANARITY:
        notes.append(
            f"patch {report['patch']!r} is not flat (planarity "
            f"{report['planarity']:.3f}): Q and Re use its mean normal"
        )
    if laminar and report["reynolds"] > LAMINAR_RE_LIMIT:
        notes.append(
            f"Re {report['reynolds']:.0f} > {LAMINAR_RE_LIMIT:.0f} (pipe-flow "
            f"transition): a steady laminar solution may not exist - expect "
            f"stalled residuals"
        )
    return notes


def flow_ends(wall, point) -> dict[str, Any]:
    """The flat end cap nearest *point* is the inflow, the other the outflow.

    The picker handles exactly two flat caps (a tube); anything else is
    refused so the user sets the knobs by hand rather than getting a guess.
    """
    import numpy as np

    p = np.asarray(point, dtype=float).reshape(-1)
    if p.shape != (3,) or not np.all(np.isfinite(p)):
        raise ValueError(f"picked point must be three finite numbers, got {point!r}")
    caps = [g for g in (patch_geometry(wall, n) for n in wall.boundary) if g.is_cap]
    if len(caps) != 2:
        raise ValueError(
            f"the inflow picker needs exactly two flat end caps; this mesh has "
            f"{len(caps)} ({[g.name for g in caps]}) - set the inlet and outlet "
            f"knobs by hand"
        )
    dist = [float(np.linalg.norm(p - np.asarray(g.centroid))) for g in caps]
    i = int(np.argmin(dist))
    return {
        "inflow": caps[i].to_dict(),
        "outflow": caps[1 - i].to_dict(),
        "pick_distance": dist[i],
    }


# Preset values in SI; the provenance and citations are DECISIONS D80.
#   blood: Newtonian, mu 3.5 mPa.s / rho 1060 kg/m^3 = 3.30e-6 m^2/s - the
#     usual large-artery CFD values (shear > ~100 1/s, where blood's
#     shear-thinning is weak).
#   aorta, cycle-mean at rest: ~90 mL/beat x 70 bpm through a 30 mm aorta.
#   ascending aorta, peak systole: 4D-flow MRI 1.01 +/- 0.31 m/s (n = 116).
_NU_PRESETS = (
    ("blood", "blood (Newtonian, mu 3.5 mPa.s, rho 1060 kg/m^3)", 3.3e-6),
)
_SPEED_PRESETS = (
    ("aorta_mean", "aorta - cycle-mean at rest", 0.15),
    ("aorta_peak", "ascending aorta - peak systole", 1.0),
)


def _sig(x: float) -> float:
    """Strip binary round-off from a unit conversion (0.15/1e-3 = 149.99...)."""
    return float(f"{x:.12g}")


def cardiac_presets(length_unit: str | None) -> dict[str, Any]:
    """Blood nu and aortic inlet speeds converted to *length_unit* (time: s)."""
    if length_unit is None:
        raise ValueError(
            "cardiac presets need the model's length unit, and this source "
            "declares none (STL/VTK and synthetic geometry carry no unit; CSO "
            "and DICOM are millimetres) - type nu and the inlet speed in your "
            "model's own units"
        )
    if length_unit not in _METRES_PER_UNIT:
        raise ValueError(
            f"no cardiac presets for length unit {length_unit!r}; known: "
            f"{sorted(_METRES_PER_UNIT)}"
        )
    m = _METRES_PER_UNIT[length_unit]
    return {
        "length_unit": length_unit,
        "time_unit": "s",
        "nu": [
            {"id": i, "label": label, "value": _sig(si / m**2),
             "unit": f"{length_unit}^2/s", "si": si}
            for i, label, si in _NU_PRESETS
        ],
        "speed": [
            {"id": i, "label": label, "value": _sig(si / m),
             "unit": f"{length_unit}/s", "si": si}
            for i, label, si in _SPEED_PRESETS
        ],
    }
