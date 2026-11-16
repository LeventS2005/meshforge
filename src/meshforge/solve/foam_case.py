"""OpenFOAM case builder (V23): polyMesh + system/0 files from a small
declarative BC spec.

The mesh side reuses V14's ``write_polymesh``; this module adds what a
solver needs on top: ``system/{controlDict,fvSchemes,fvSolution}``,
``constant/{transportProperties,turbulenceProperties}``, ``0/{U,p}`` and
the empty ``case.foam`` stub pyvista's OpenFOAMReader keys on.

Laminar incompressible only (simpleFoam steady / icoFoam transient).
Every patch of the mesh must end up with a boundary condition: patches
not named in the spec default to no-slip walls, but a patch whose NAME
says inlet/outlet with no explicit BC is refused — silently turning an
intended inlet into a wall would fail-open into wrong physics.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

_SOLVERS = ("simpleFoam", "icoFoam")
_BC_KINDS = ("velocity", "pressure", "wall", "moving_wall", "empty")
_TURBULENCE_MODELS = ("laminar", "kOmegaSST")


def _require_multiple(value: float, unit: float, name: str, unit_name: str) -> None:
    """Loud unless value is an integer multiple of unit (rel tol 1e-6)."""
    ratio = value / unit
    if abs(ratio - round(ratio)) > 1e-6 * max(1.0, abs(ratio)):
        raise ValueError(
            f"{name} ({value:g}) is not an integer multiple of {unit_name} "
            f"({unit:g}): the run would step past the write/end boundary "
            f"and the final state would never be written (V29 probe: "
            f"dt=0.03, writeInterval=0.2, endTime=0.5 last wrote t=0.39)"
        )


@dataclass(frozen=True)
class CfdBC:
    """One patch BC: velocity inlet, pressure outlet, no-slip wall,
    tangentially moving wall (V30 — lid-driven cavities), or the
    OpenFOAM 2-D ``empty`` direction (V30)."""

    kind: str
    value: tuple[float, float, float] | float | None = None

    def __post_init__(self) -> None:
        if self.kind not in _BC_KINDS:
            raise ValueError(f"unknown BC kind {self.kind!r}; use {_BC_KINDS}")
        if self.kind in ("velocity", "moving_wall"):
            if self.value is None or len(tuple(self.value)) != 3:
                raise ValueError(f"{self.kind} BC needs a (vx, vy, vz) value")
            object.__setattr__(
                self, "value", tuple(float(v) for v in self.value)
            )
        elif self.kind == "pressure":
            if self.value is None:
                raise ValueError("pressure BC needs a scalar value (kinematic p)")
            object.__setattr__(self, "value", float(self.value))
        elif self.value is not None:
            raise ValueError(f"{self.kind} BC takes no value")


@dataclass(frozen=True)
class CfdSpec:
    """Declarative incompressible case description: laminar (V23) or
    steady kOmegaSST RANS (V27, simpleFoam only).

    Turbulent inlet scales are uniform estimates from the FIRST velocity
    BC: k = 1.5·(|U|·I)², omega = √k/(0.09^0.25·l) with I =
    ``turbulence_intensity`` and l = ``turbulence_length`` (REQUIRED for
    turbulent cases — e.g. 0.07·D for a pipe; never guessed from the
    mesh)."""

    solver: str = "simpleFoam"
    nu: float = 1e-3
    bcs: dict[str, CfdBC] = field(default_factory=dict)
    end_time: float = 2000.0
    delta_t: float | None = None
    write_interval: float | None = None
    turbulence: str = "laminar"
    turbulence_intensity: float = 0.05
    turbulence_length: float | None = None
    closed_cavity: bool = False
    p_ref_cell: int = 0
    p_ref_value: float = 0.0

    def __post_init__(self) -> None:
        # F1 backstop: patch names become dict keys in constant/polyMesh/
        # boundary and 0/{U,p,k,omega,nut} boundaryField — content-validate
        # every one so a newline can never inject a forged dictionary entry.
        from meshforge.meshing.base import valid_group_name

        for patch_name in self.bcs:
            valid_group_name(patch_name)
        if self.solver not in _SOLVERS:
            raise ValueError(
                f"unknown solver {self.solver!r}; V23 supports {_SOLVERS}"
            )
        if not self.nu > 0.0:
            raise ValueError(f"kinematic viscosity must be positive, got {self.nu}")
        if self.end_time <= 0.0:
            raise ValueError(f"end_time must be positive, got {self.end_time}")
        if self.write_interval is not None and self.write_interval <= 0.0:
            raise ValueError(
                f"write_interval must be positive, got {self.write_interval}"
            )
        if self.solver == "icoFoam":
            if self.delta_t is None or self.delta_t <= 0.0:
                raise ValueError(
                    "icoFoam is transient: a positive delta_t is required"
                )
            # Probe-verified (V29): with a fixed dt, OpenFOAM's runTime
            # write control only writes when a step lands near an interval
            # boundary — non-commensurate knobs step PAST end_time and the
            # end state is computed but NEVER WRITTEN (dt=0.03/wi=0.2/
            # end=0.5 last wrote t=0.39); read-back then silently serves
            # an earlier time. Refuse the combination outright.
            _require_multiple(self.end_time, self.delta_t, "end_time", "delta_t")
            if self.write_interval is not None:
                _require_multiple(
                    self.write_interval, self.delta_t,
                    "write_interval", "delta_t",
                )
                _require_multiple(
                    self.end_time, self.write_interval,
                    "end_time", "write_interval",
                )
        elif self.write_interval is not None:
            # simpleFoam writes on a timeStep (iteration) control — the
            # interval must be a whole number of iterations.
            if abs(self.write_interval - round(self.write_interval)) > 1e-9:
                raise ValueError(
                    f"simpleFoam writes per iteration: write_interval must "
                    f"be a whole number, got {self.write_interval}"
                )
        if self.closed_cavity:
            # V30: the pressure level is pinned by pRefCell/pRefValue, and
            # the boundary must actually be closed and actually driven.
            if any(bc.kind == "pressure" for bc in self.bcs.values()):
                raise ValueError(
                    "closed_cavity with a pressure BC is a contradiction — "
                    "the reference level is pRefCell/pRefValue's job; drop "
                    "the pressure BC or the closed_cavity flag"
                )
            if any(bc.kind == "velocity" for bc in self.bcs.values()):
                raise ValueError(
                    "closed_cavity with a velocity inlet: a sealed domain "
                    "cannot absorb net inflow — tangentially driven lids "
                    "are the moving_wall BC kind"
                )
            if not any(bc.kind == "moving_wall" for bc in self.bcs.values()):
                raise ValueError(
                    "closed_cavity with nothing driving the flow — give "
                    "one patch a moving_wall BC (e.g. the lid)"
                )
            if self.p_ref_cell < 0:
                raise ValueError(
                    f"p_ref_cell must be a non-negative cell index, got "
                    f"{self.p_ref_cell}"
                )
        elif not any(bc.kind == "pressure" for bc in self.bcs.values()):
            raise ValueError(
                "no pressure BC anywhere: the incompressible pressure level "
                "would be undefined — give one patch a pressure BC, or set "
                "closed_cavity (pRefCell/pRefValue) for a sealed domain"
            )
        if self.turbulence not in _TURBULENCE_MODELS:
            raise ValueError(
                f"unknown turbulence model {self.turbulence!r}; "
                f"supported: {_TURBULENCE_MODELS}"
            )
        if self.turbulence != "laminar":
            if self.solver != "simpleFoam":
                raise ValueError(
                    "turbulence needs simpleFoam — the icoFoam branch is "
                    "laminar-only (V23 scope)"
                )
            if not 0.0 < self.turbulence_intensity < 1.0:
                raise ValueError(
                    f"turbulence_intensity must be in (0, 1), got "
                    f"{self.turbulence_intensity}"
                )
            if self.turbulence_length is None or self.turbulence_length <= 0.0:
                raise ValueError(
                    "turbulent cases need a positive turbulence_length "
                    "(mixing length for the inlet omega — e.g. 0.07*D for "
                    "a pipe); it is never guessed from the mesh"
                )
            if not any(bc.kind == "velocity" for bc in self.bcs.values()):
                raise ValueError(
                    "turbulence needs at least one velocity BC — the inlet "
                    "k/omega scales come from the first velocity magnitude"
                )
            if not self.turbulence_inlet()[0] > 0.0:
                raise ValueError(
                    "turbulence inlet k is zero — the first velocity BC "
                    "has zero magnitude, so k/omega cannot be scaled"
                )

    def turbulence_inlet(self) -> tuple[float, float]:
        """(k, omega) inlet/initial values from the first velocity BC."""
        u = next(
            bc.value for bc in self.bcs.values() if bc.kind == "velocity"
        )
        u_mag = float(sum(c * c for c in u) ** 0.5)
        k = 1.5 * (u_mag * self.turbulence_intensity) ** 2
        omega = k**0.5 / (0.09**0.25 * self.turbulence_length)
        return k, omega

    def resolve(self, patch_names: list[str]) -> dict[str, CfdBC]:
        """Full patch -> BC map: unnamed patches default to walls, except
        patches whose NAME promises a flow boundary — those are refused."""
        unknown = set(self.bcs) - set(patch_names)
        if unknown:
            raise ValueError(
                f"BC spec names unknown patch(es) {sorted(unknown)}; "
                f"mesh patches: {patch_names}"
            )
        resolved: dict[str, CfdBC] = {}
        for name in patch_names:
            bc = self.bcs.get(name)
            if bc is None:
                if "inlet" in name.lower() or "outlet" in name.lower():
                    raise ValueError(
                        f"patch {name!r} looks like a flow boundary but has "
                        f"no BC — defaulting it to a wall would silently "
                        f"close the domain; give it a velocity/pressure BC"
                    )
                bc = CfdBC("wall")
            resolved[name] = bc
        return resolved


def _foam_dict(obj: str, cls: str, location: str, body: str) -> str:
    return (
        "FoamFile\n{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {cls};\n"
        f'    location    "{location}";\n'
        f"    object      {obj};\n"
        "}\n" + body
    )


def read_patch_names(case_dir: str | Path) -> list[str]:
    """Patch names, in file order, from ``constant/polyMesh/boundary``."""
    boundary = Path(case_dir) / "constant" / "polyMesh" / "boundary"
    if not boundary.is_file():
        raise FileNotFoundError(
            f"no polyMesh boundary file at {boundary} — give a complete "
            f"OpenFOAM case, or export the mesh first with save('foam', ...)"
        )
    names = [
        m.group(1)
        for m in re.finditer(r"(\w+)\s*\{[^}]*\}", boundary.read_text())
        if m.group(1) != "FoamFile"
    ]
    if not names:
        raise ValueError(f"no patches found in {boundary}")
    return names


def _control_dict(spec: CfdSpec) -> str:
    if spec.solver == "simpleFoam":
        delta_t = 1.0
        write = spec.write_interval if spec.write_interval else spec.end_time
        write_control = "timeStep"
    else:
        delta_t = spec.delta_t
        write = spec.write_interval if spec.write_interval else spec.end_time
        write_control = "runTime"
    return _foam_dict(
        "controlDict", "dictionary", "system",
        f"""
application     {spec.solver};
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {spec.end_time:g};
deltaT          {delta_t:g};
writeControl    {write_control};
writeInterval   {write:g};
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
""",
    )


_FV_SCHEMES_SIMPLE = """
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss linearUpwind grad(U);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
"""

_FV_SCHEMES_ICO = """
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }
divSchemes      { default none; div(phi,U) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
"""

_FV_SOLUTION_SIMPLE = """
solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.01; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 1;
__PREF__
    residualControl { p 1e-5; U 1e-6; }
}
relaxationFactors { fields { p 0.3; } equations { U 0.7; } }
"""

_FV_SOLUTION_ICO = """
solvers
{
    p      { solver PCG; preconditioner DIC; tolerance 1e-7; relTol 0.05; }
    pFinal { solver PCG; preconditioner DIC; tolerance 1e-7; relTol 0; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0; }
}
PISO
{
    nCorrectors 2;
    nNonOrthogonalCorrectors 1;
__PREF__
}
"""

# kOmegaSST steady templates, exactly as probe-verified (V27): upwinded
# k/omega transport, wallDist for the SST blending functions, SIMPLEC
# with k/omega relaxation (converged in 150 iterations on the pipe).
_FV_SCHEMES_TURB = """
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss linearUpwind grad(U);
    div(phi,k) bounded Gauss upwind;
    div(phi,omega) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
wallDist        { method meshWave; }
"""

_FV_SOLUTION_TURB = """
solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.01; }
    "(U|k|omega)"
    { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent yes;
__PREF__
    residualControl { p 1e-5; U 1e-6; "(k|omega)" 1e-6; }
}
relaxationFactors { equations { U 0.9; ".*" 0.7; } }
"""


def _bc_entry(bc: CfdBC, *, for_field: str) -> str:
    if bc.kind == "empty":
        return "{ type empty; }"  # 2-D direction: every field is empty
    if for_field == "U":
        if bc.kind in ("velocity", "moving_wall"):
            vx, vy, vz = bc.value
            return f"{{ type fixedValue; value uniform ({vx:g} {vy:g} {vz:g}); }}"
        if bc.kind == "pressure":
            return "{ type zeroGradient; }"
        return "{ type noSlip; }"
    if bc.kind == "pressure":
        return f"{{ type fixedValue; value uniform {bc.value:g}; }}"
    return "{ type zeroGradient; }"


def _field_file(
    obj: str, cls: str, dims: str, internal: str, entries: dict[str, str]
) -> str:
    body = f"\ndimensions      {dims};\ninternalField   {internal};\n"
    body += "boundaryField\n{\n"
    for name, entry in entries.items():
        body += f"    {name} {entry}\n"
    body += "}\n"
    return _foam_dict(obj, cls, "0", body)


def write_case_files(case_dir: str | Path, spec: CfdSpec) -> dict[str, str]:
    """Write system/constant/0 files for an existing polyMesh; returns the
    resolved patch -> BC-kind map (for summaries).

    A REBUILT case must carry no results from a previous life: stale time
    directories would win max(time) at read-back — silently wrong fields
    on the same mesh, a reader error on a different one (found when a
    finer mesh reused ``outputs/cfd/session``, V26). Time dirs (0/ is
    rebuilt below) and old ``log.*`` files are removed; ``constant/`` and
    ``system/`` are rewritten in place. User-supplied complete cases run
    through ``solve_case`` untouched — restart semantics stay theirs.
    """
    case = Path(case_dir)
    resolved = spec.resolve(read_patch_names(case))

    for entry in case.iterdir():
        if entry.is_dir():
            try:
                float(entry.name)
            except ValueError:
                continue
            shutil.rmtree(entry)
        elif entry.name.startswith("log."):
            entry.unlink()

    turbulent = spec.turbulence != "laminar"
    system = case / "system"
    system.mkdir(parents=True, exist_ok=True)
    (system / "controlDict").write_text(_control_dict(spec))
    if turbulent:
        schemes, solution = _FV_SCHEMES_TURB, _FV_SOLUTION_TURB
    elif spec.solver == "simpleFoam":
        schemes, solution = _FV_SCHEMES_SIMPLE, _FV_SOLUTION_SIMPLE
    else:
        schemes, solution = _FV_SCHEMES_ICO, _FV_SOLUTION_ICO
    # V30: a closed cavity pins the pressure level through pRefCell/
    # pRefValue; open cases drop the token line (byte-identical output).
    pref = (
        f"    pRefCell        {spec.p_ref_cell};\n"
        f"    pRefValue       {spec.p_ref_value:g};\n"
        if spec.closed_cavity
        else ""
    )
    solution = solution.replace("__PREF__\n", pref)
    (system / "fvSchemes").write_text(_foam_dict("fvSchemes", "dictionary", "system", schemes))
    (system / "fvSolution").write_text(_foam_dict("fvSolution", "dictionary", "system", solution))

    const = case / "constant"
    const.mkdir(parents=True, exist_ok=True)
    (const / "transportProperties").write_text(
        _foam_dict(
            "transportProperties", "dictionary", "constant",
            f"\ntransportModel  Newtonian;\nnu              [0 2 -1 0 0 0 0] {spec.nu:g};\n",
        )
    )
    if turbulent:
        turb_body = (
            "\nsimulationType  RAS;\n"
            "RAS\n{\n"
            f"    RASModel    {spec.turbulence};\n"
            "    turbulence  on;\n"
            "    printCoeffs on;\n"
            "}\n"
        )
    else:
        turb_body = "\nsimulationType  laminar;\n"
    (const / "turbulenceProperties").write_text(
        _foam_dict("turbulenceProperties", "dictionary", "constant", turb_body)
    )

    # Initial guess: the first velocity BC seeds the interior (helps SIMPLE
    # converge); pressure starts at 0 everywhere.
    seed = next(
        (bc.value for bc in resolved.values() if bc.kind == "velocity"),
        (0.0, 0.0, 0.0),
    )
    zero = case / "0"
    zero.mkdir(parents=True, exist_ok=True)
    (zero / "U").write_text(
        _field_file(
            "U", "volVectorField", "[0 1 -1 0 0 0 0]",
            f"uniform ({seed[0]:g} {seed[1]:g} {seed[2]:g})",
            {n: _bc_entry(bc, for_field="U") for n, bc in resolved.items()},
        )
    )
    (zero / "p").write_text(
        _field_file(
            "p", "volScalarField", "[0 2 -2 0 0 0 0]", "uniform 0",
            {n: _bc_entry(bc, for_field="p") for n, bc in resolved.items()},
        )
    )
    if turbulent:
        k_in, omega_in = spec.turbulence_inlet()

        def turb_entries(inlet: str, outlet: str, wall: str) -> dict[str, str]:
            by_kind = {
                "velocity": inlet, "pressure": outlet, "wall": wall,
                "moving_wall": wall, "empty": "{ type empty; }",
            }
            return {name: by_kind[bc.kind] for name, bc in resolved.items()}

        (zero / "k").write_text(
            _field_file(
                "k", "volScalarField", "[0 2 -2 0 0 0 0]",
                f"uniform {k_in:g}",
                turb_entries(
                    f"{{ type fixedValue; value uniform {k_in:g}; }}",
                    "{ type zeroGradient; }",
                    f"{{ type kqRWallFunction; value uniform {k_in:g}; }}",
                ),
            )
        )
        (zero / "omega").write_text(
            _field_file(
                "omega", "volScalarField", "[0 0 -1 0 0 0 0]",
                f"uniform {omega_in:g}",
                turb_entries(
                    f"{{ type fixedValue; value uniform {omega_in:g}; }}",
                    "{ type zeroGradient; }",
                    f"{{ type omegaWallFunction; value uniform {omega_in:g}; }}",
                ),
            )
        )
        # nut walls get the ALL-y+ Spalding law — probe-verified: the
        # log-layer nutkWallFunction is -21% on friction at the O-grid
        # pipe's buffer-layer y+ (~16); Spalding is -0.7% (V27_SPEC).
        (zero / "nut").write_text(
            _field_file(
                "nut", "volScalarField", "[0 2 -1 0 0 0 0]", "uniform 0",
                turb_entries(
                    "{ type calculated; value uniform 0; }",
                    "{ type calculated; value uniform 0; }",
                    "{ type nutUSpaldingWallFunction; value uniform 0; }",
                ),
            )
        )
    (case / "case.foam").touch()  # pyvista OpenFOAMReader entry point
    return {name: bc.kind for name, bc in resolved.items()}


def write_case(
    wall,
    case_dir: str | Path,
    spec: CfdSpec,
    *,
    input_path: str | Path | None = None,
) -> dict[str, str]:
    """polyMesh (V14) + solver files in one call. Patch types follow the
    BCs: flow boundaries become ``patch``, everything else ``wall``."""
    from meshforge.export.foam import write_polymesh

    patch_types = {}
    for name, bc in spec.bcs.items():
        if bc.kind in ("velocity", "pressure"):
            patch_types[name] = "patch"
        elif bc.kind == "moving_wall":
            patch_types[name] = "wall"
        elif bc.kind == "empty":
            patch_types[name] = "empty"
    write_polymesh(
        wall, case_dir, patch_types=patch_types, input_path=input_path
    )
    return write_case_files(case_dir, spec)
