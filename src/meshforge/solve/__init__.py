"""In-app solve: CalculiX runner + .dat/.frd read-back (V13) and the
OpenFOAM case builder/runner/read-back (V23)."""
from meshforge.solve.dat import parse_displacements, parse_stresses
from meshforge.solve.foam_case import CfdBC, CfdSpec, write_case, write_case_files
from meshforge.solve.foam_runner import (
    CfdResult,
    check_mesh,
    find_foam,
    read_case,
    read_case_series,
    run_foam_app,
    solve_case,
)
from meshforge.solve.frd import (
    FrdField,
    FrdModel,
    attach_fields,
    frd_to_grid,
    nodal_temperatures,
    temperature_times,
    read_frd,
    von_mises,
)
from meshforge.solve.runner import SolveResult, find_ccx, solve_deck

__all__ = [
    "CfdBC",
    "CfdResult",
    "CfdSpec",
    "FrdField",
    "FrdModel",
    "SolveResult",
    "attach_fields",
    "check_mesh",
    "find_ccx",
    "find_foam",
    "frd_to_grid",
    "nodal_temperatures",
    "temperature_times",
    "parse_displacements",
    "parse_stresses",
    "read_case",
    "read_case_series",
    "read_frd",
    "run_foam_app",
    "solve_case",
    "solve_deck",
    "von_mises",
]
