"""The blood flow of THIS crop, solved live (D112 addendum).

After the copilot's strength index, the CLI builds an OpenFOAM case from the
cropped blood lumen the crop step wrote (``work/crop_lumen.vtp``): a helper
in-process session fills the lumen with tetrahedra (the same ``lumen`` step
the Workbench offers), the cap higher on the body axis is the neck and takes
the inflow (a resting 1 L/min spread over its area, along its inward
normal), the other cap is the pressure outlet, and the walls are no-slip.
The Workbench's own ``cfd`` step then solves the written case, so the flow
result sits in the same session as the wall stress, on the same crop, and
the journal carries it.

Nothing here invents a number: the mesh is the crop's lumen, the inflow is a
stated resting value, the viscosity is blood's.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

#: resting flow through the infrarenal aorta, litres per minute
Q_LMIN = 1.0
#: kinematic viscosity of blood, mm²/s (3.3e-6 m²/s; the session's meshes are in mm)
NU_BLOOD_MM2_S = 3.3


def build_lumen_case(crop_lumen: str | Path, case_dir: str | Path, *,
                     q_lmin: float = Q_LMIN, nu: float = NU_BLOOD_MM2_S) -> dict[str, Any]:
    """Write a complete steady laminar simpleFoam case for the cropped lumen
    surface at ``crop_lumen`` into ``case_dir`` (replaced if it exists).
    Returns what was built: cells, the fill method, the inflow patch, its
    area and the speed that gives ``q_lmin``. Loud when the surface has no
    two flat caps."""
    from meshforge.pipeline import MeshingSession
    from meshforge.solve import CfdBC, CfdSpec, write_case
    from meshforge.solve.inflow import patch_geometry, resolve_inlet_speed

    crop_lumen, case_dir = Path(crop_lumen), Path(case_dir)
    if not crop_lumen.is_file():
        raise FileNotFoundError(f"no cropped lumen at {crop_lumen}")
    helper = MeshingSession()  # its own journal; the Workbench's session is untouched
    helper.load(crop_lumen)
    # The cropped lumen as the crop step writes it is watertight but carries
    # the segmentation's small folds; gmsh refuses its parametrisation
    # ("wrong topology of boundary mesh"). The session's own repair step
    # first, then the fill. The station-ring tet fill comes first: on the
    # demo scan it gives 6,748 clean cells and simpleFoam converges in ~140
    # iterations (3 s), where the swept O-grid (9,450 cells, pinned
    # topology) ran to endTime unconverged. Then the automatic choice, then
    # the surface tet fill.
    helper.compute_repair()
    errors: list[str] = []
    filled = None
    for method in ("tet", "auto", "surface-tet"):
        try:
            filled = helper.compute_lumen(method=method)
            break
        except Exception as exc:  # noqa: BLE001 - gmsh raises bare Exceptions
            errors.append(f"{method}: {exc}")
    if filled is None:
        raise RuntimeError("the lumen could not be filled — " + "; ".join(errors))
    vol = helper.volmesh
    summary = filled.summary
    caps = sorted(name for name in vol.boundary if name != "wall")
    if len(caps) != 2:
        raise ValueError(
            f"the lumen fill gave {len(caps)} end cap(s) ({', '.join(caps) or 'none'}); "
            "a flow needs exactly two"
        )
    geo = {name: patch_geometry(vol, name) for name in caps}
    neck = max(caps, key=lambda n: geo[n].centroid[2])  # superior on the body axis (RAS z)
    outlet = next(n for n in caps if n != neck)
    area = geo[neck].area
    if not area > 0.0:
        raise ValueError(f"the neck cap {neck!r} has no area")
    q_mm3_s = float(q_lmin) * 1e6 / 60.0
    speed = q_mm3_s / area
    velocity, _ = resolve_inlet_speed(vol, neck, speed)
    spec = CfdSpec(
        solver="simpleFoam", nu=float(nu),
        bcs={neck: CfdBC("velocity", velocity), outlet: CfdBC("pressure", 0.0)},
    )
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    write_case(vol, case_dir, spec, input_path=crop_lumen)
    return {
        "case": str(case_dir),
        "n_cells": summary.get("n_cells"),
        "fill": summary.get("backend"),
        "inflow_patch": neck,
        "inflow_area_mm2": round(area, 1),
        "inflow_speed_mm_s": round(speed, 2),
        "outlet_patch": outlet,
        "q_lmin": float(q_lmin),
        "nu_mm2_s": float(nu),
        "notes": list(summary.get("notes") or []),
    }
