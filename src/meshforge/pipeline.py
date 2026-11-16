"""Front-end-agnostic service layer: MeshingSession.

Every front-end (the CLI and the Workbench) drives ONLY this class. Each step
returns a StepResult carrying a machine summary, human-readable text, and
neutral render hints — fully testable without a display. Paths are accepted as parameters
but files are read only inside the step that consumes them (trap #7).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Declarative knob specs: (type, default, help). The Workbench generates its
# parameter widgets from these, so new knobs appear with no front-end changes.
#
# The ``help`` strings are USER-FACING (Workbench knob captions and CLI
# --help). Keep internal spec/milestone ids ("V38", "V47", D-numbers) OUT of
# them — they mean nothing to a user. Provenance belongs in the section
# comments below or in docstrings.
STEP_KNOBS: dict[str, dict[str, tuple[type, Any, str]]] = {
    "centerline": {
        "voxel_size": (float, None, "voxel size (default: bbox diagonal / 150)"),
        "smoothing": (float, None, "RMS smoothing in physical units (default: 1 voxel)"),
        "n_samples": (int, 200, "uniform arc-length samples"),
    },
    "volmesh": {
        "backend": (str, "swept", "swept (hexahedral sweep; the only wall backend in this edition)"),
        "thickness": (float, 0.5, "wall thickness; 0.5 suits the synthetic tube — a mm-scale scan wants ~1.5"),
        "n_theta": (int, 32, "azimuthal divisions"),
        "n_axial": (int, 60, "axial stations"),
        "n_layers": (int, 4, "radial layers (swept)"),
        "layer_ratio": (float, 1.0, "radial grading: outer/inner layer thickness"),
        "curvature_cap": (float, 0.5, "swept: fraction of the fold radius 1/|κ| a wall ray may reach on a concave bend (0.9 for an aneurysm sac)"),
        # D87: station layout of the swept wall (default unchanged).
        "sweep": (str, "centreline", "swept stations: centreline (rays square to the centreline) | planes (sections parallel to two flat parallel end caps — for a short, wide, eccentric segment such as a cropped sac) | auto (centreline unless its wall inverts or is capped/floored, then planes)"),
        "side": (str, "outward", "planes: grow the wall outward of the surface, or inward (the surface is the outer wall, e.g. a recovered sac)"),
        "max_tilt": (float, None, "planes: largest surface tilt to the station planes, degrees, before the wall is thinned (default 80)"),
        "section_smoothing": (float, 0.0, "planes: Gaussian smoothing of each section along its length (length units, 0 = exact) for a surface rougher than the wall thickness; how far the nodes move is reported"),
    },
    # V47: fill the blood volume (lumen) instead of the wall; the result
    # takes the session volmesh slot so cfd/save work unchanged.
    "lumen": {
        "method": (str, "auto", "auto (O-grid hexes, then tet, then surface-tet on failure) | ogrid | tet | surface-tet"),
        "n_theta": (int, 32, "rays per station (ring resolution)"),
        "n_axial": (int, 40, "stations along the centerline"),
        "core_fraction": (float, 0.5, "O-grid core size fraction of centroid-to-rim"),
        "smooth": (int, 4, "guarded smoothing iterations, sj_min can never regress (0 = off)"),
        "mesh_size": (float, None, "tet-fallback target size (default: 1.5x median ring edge)"),
    },
    # D83 (M1): AI segmentation — TotalSegmentator (own conda env, run as a
    # subprocess) outlines the aorta on a CT; the mask becomes the source surface.
    "segment": {
        "path": (str, "", "CT volume .nrrd / .nii(.gz) — the aorta is outlined by TotalSegmentator"),
        "device": (str, "mps", "mps (Apple GPU) | cpu"),
        "fast": (int, 0, "1 = TotalSegmentator's faster 3 mm model"),
        "smoothing": (float, 1.0, "Gaussian blur of the mask in voxels before the surface (0 = off)"),
        "timeout": (float, 1800.0, "TotalSegmentator wall-clock limit in seconds"),
        "lumen_out": (str, "", "also write the contrast blood lumen to this .vtp file (CT >= lumen_hu inside the AI outline — a threshold, not AI)"),
        "lumen_hu": (float, 150.0, "HU threshold for the blood lumen (calcification is bright too: detached pieces are dropped)"),
        "mask_out": (str, "", "also write the AI aorta mask to this .nii.gz file (the sac step's lumen_mask)"),
    },
    # D85 (M2): crop the aorta to its infrarenal aneurysm segment — two cut
    # planes square to the centreline, flat caps, landmark report.
    "crop": {
        "neck_extension_mm": (float, 25.0, "proximal cut this far above the neck start (the flow/clamp extension), held inside the neck"),
        "distal_margin_mm": (float, 10.0, "distal cut this far above the top of the bifurcation flare"),
        "distal_height_mm": (float, None, "manual distal cut, mm above the caudal end (set both heights or neither)"),
        "proximal_height_mm": (float, None, "manual proximal cut, mm above the caudal end"),
        "lumen": (str, "", "blood-lumen surface file to cut with the same two planes"),
        "lumen_out": (str, "", "write the cut lumen surface to this .vtp file"),
        "parallel_caps": (int, 0, "1 = both cuts square to the chord between them: parallel caps, which the plane-swept wall needs on a bent aorta (landmarks unchanged)"),
    },
    # D86 (M2): outer aneurysm sac (lumen + thrombus + wall) recovered from the
    # CT around a lumen-only mask, by axial rays to the first fat plane.
    "sac": {
        "path": (str, "", "CT volume .nrrd / .nii(.gz) in HU — the scan the lumen mask was drawn on"),
        "lumen_mask": (str, "", "lumen mask file on the same grid (.nii/.nii.gz/.nrrd), e.g. an AI aorta mask"),
        "lumen_hu": (float, 150.0, "contrast threshold (HU): mask voxels at or above it are the lumen"),
        "fat_hu": (float, -30.0, "fat / air below this HU marks the outer sac edge"),
        "max_thickness_mm": (float, 40.0, "longest thrombus + wall searched beyond the lumen (mm)"),
        "n_rays": (int, 72, "rays per axial slice"),
        "roi_mm": (float, 150.0, "height above the lumen's lower end to process (mm; 0 = whole mask)"),
        "min_resolved": (float, 0.5, "refuse when fewer rays than this fraction find a fat plane"),
        "smoothing": (float, 1.0, "Gaussian blur of the sac mask in voxels before the surface (0 = off)"),
    },
    "repair": {
        "merge_tol": (float, None, "vertex merge tolerance (default 1e-9 x diag)"),
        "max_hole_edges": (int, 16, "fill holes up to this many edges"),
    },
    # D88 (M3): AAA wall stress under blood pressure — stiff-linear solve on
    # the loaded geometry, peak wall stress, wall tension, a pressure what-if.
    "aaa_stress": {
        "systolic_mmhg": (float, 140.0, "systolic blood pressure (mmHg) on the lumen side of the wall"),
        "whatif_mmhg": (float, 120.0, "what-if blood pressure (mmHg), solved separately"),
        "E": (float, 100.0, "wall Young's modulus in MPa (stiff-linear on the loaded geometry: the stress barely depends on it)"),
        "nu": (float, 0.49, "wall Poisson's ratio (near-incompressible tissue)"),
        "end_zone": (str, "neck", "left out of the metrics next to each clamped end: neck (one neck diameter from the crop report, else each end's own diameter) | bending (3 shell bending lengths) | a length in mm"),
        "crop_report": (str, "", "crop report JSON file with the neck diameter (empty: this session's crop step, if any)"),
        "lumen": (str, "", "blood-lumen surface file cut by the same two planes (crop's lumen_out): models the intraluminal thrombus between it and the wall as a tied solid (empty: wall only)"),
        "sac": (str, "", "the sac surface file the wall was built from (required with lumen): the thrombus fills the space between it and the lumen"),
        "E_ilt": (float, 5.0, "thrombus Young's modulus in MPa (the Zenodo AAA FE model's value; used with lumen)"),
        "ccx": (str, "", "CalculiX executable (empty: MESHFORGE_CCX / PATH / conda env)"),
        "timeout": (float, 600.0, "solver wall-clock limit per solve in seconds"),
    },
    # V23: in-app CFD solve — OpenFOAM on a session-built (or external)
    # case, U/p read back via pyvista.
    "cfd": {
        "case": (str, "", "existing complete case dir to run (empty: build one from the session volmesh)"),
        "solver": (str, "simpleFoam", "simpleFoam (steady) | icoFoam (transient)"),
        "nu": (float, 1e-3, "kinematic viscosity"),
        "inlet": (str, "", "velocity inlet: NAME=SPEED flows in along the patch's inward normal (e.g. inlet=150), or NAME=vx,vy,vz"),
        "outlet": (str, "", "pressure outlet NAME=value, e.g. back=0"),
        "end_time": (float, 2000.0, "endTime (steady: max iterations)"),
        "delta_t": (float, 0.0, "time step (icoFoam only; 0 invalid there)"),
        "write_interval": (float, 0.0, "write interval (0: final result only)"),
        "timeout": (float, 600.0, "solver wall-clock limit in seconds"),
        "turbulence": (str, "laminar", "laminar | kOmegaSST (steady RANS)"),
        "intensity": (float, 0.05, "inlet turbulence intensity I (kOmegaSST)"),
        "turb_length": (float, 0.0, "inlet mixing length, e.g. 0.07*D (REQUIRED for kOmegaSST)"),
        "series": (int, 0, "1 = read ALL written time steps back as frames (needs write_interval < end_time)"),
        "moving_wall": (str, "", "tangentially moving wall NAME=vx,vy,vz, e.g. lid=1,0,0"),
        "empty": (str, "", "comma-separated 2-D 'empty' patches, e.g. front,back"),
        "closed_cavity": (int, 0, "1 = sealed domain: pressure level pinned by pRefCell/pRefValue"),
        "p_ref_cell": (int, 0, "reference cell index (closed_cavity)"),
        "p_ref_value": (float, 0.0, "reference kinematic pressure (closed_cavity)"),
    },
    # D89 (M4): strength index band (peak wall stress / lab-measured wall
    # strength), tissue library v0 (open USP AAA strip tests).
    "tissue_index": {
        "pws_mpa": (float, None, "peak wall stress in MPa (Cauchy, from the wall-stress solve); or use from_summary"),
        "from_summary": (str, "", "wall-stress summary .json file holding a pws_mpa value (instead of pws_mpa)"),
        "selection": (str, "aaa", "strength strips: aaa (rupture-border aneurysm wall) | control (non-dilated aorta) | all"),
        "method": (str, "empirical", "band: empirical (strip quantiles) | lognormal (fitted)"),
    },
}

# Panel/display order for front-ends (Workbench builds its left pane from this).
# STEP_KNOBS is keyed by name everywhere else (cli.py, web/app.py), so the
# order a user sees is stated ONCE, here, instead of riding on dict insertion
# order. The CT steps come first because a real run starts from a scan:
# they produce the surface every processing step below them consumes.
STEP_ORDER: tuple[str, ...] = (
    "segment",
    "sac",
    "crop",
    "centerline",
    "volmesh",
    "lumen",
    "repair",
    "aaa_stress",
    "tissue_index",
    "cfd",
)

# Fail loudly if a step is ever added to STEP_KNOBS without a display slot,
# rather than letting it silently vanish from (or duplicate in) the Workbench.
_missing = set(STEP_KNOBS) - set(STEP_ORDER)
_extra = set(STEP_ORDER) - set(STEP_KNOBS)
if _missing or _extra:  # pragma: no cover - import-time contract check
    raise RuntimeError(
        f"STEP_ORDER is out of sync with STEP_KNOBS: missing {sorted(_missing)}, "
        f"unknown {sorted(_extra)}"
    )
if len(STEP_ORDER) != len(set(STEP_ORDER)):  # pragma: no cover
    raise RuntimeError("STEP_ORDER contains a duplicate step")


def sac_text(summ: dict[str, Any]) -> str:
    """Human-readable sac summary (step text and CLI share it)."""
    md, th = summ["max_diameter"], summ["thickness_mm"]
    counts = ", ".join(f"{k} {v}" for k, v in summ["ray_status_counts"].items())
    rps = summ["resolved_fraction_per_slice"]
    text = (
        f"sac: {summ['n_slices']} axial slices x {summ['n_rays']} rays, h "
        f"{summ['h_range_mm'][0]:g}..{summ['h_range_mm'][1]:g} mm above the lumen's "
        f"lower end\n"
        f"  resolved (fat plane found): {summ['resolved_fraction']:.1%} of rays; per "
        f"slice min {rps['min']:.0%} / median {rps['median']:.0%}; {counts}\n"
        f"  sac {summ['sac_volume_ml']:.1f} mL, lumen {summ['lumen_volume_ml']:.1f} mL; "
        f"max diameter {md['max_caliper']['value_mm']:.1f} mm (caliper, h "
        f"{md['max_caliper']['h_mm']:g}), short axis {md['min_caliper']['value_mm']:.1f} "
        f"mm, area-equivalent {md['area_equivalent']['value_mm']:.1f} mm\n"
        f"  thrombus + wall thickness: median {th['all_rays']['median']:.1f}, p90 "
        f"{th['all_rays']['p90']:.1f}, max {th['all_rays']['max']:.1f} mm\n"
    )
    for note in summ["notes"]:
        text += f"  note: {note}\n"
    return text


def _clean_path(raw: str | Path) -> str:
    r"""Normalise one user-entered path (``~``, surrounding quotes, ``\ ``
    drag-into-terminal escapes, stray whitespace) — see ``io/paths.py``.

    Applied at EVERY path entry point in this service layer, so the CLI
    and the Workbench both accept the same path habits and a
    missing file reports the loader's own friendly message instead of a raw
    ``[Errno 2]``. Lazy import: ``meshforge.io`` pulls pyvista, and this
    module must stay cheap to import (rule 5).
    """
    from meshforge.io.paths import normalize_path_str

    return normalize_path_str(raw)


def _parse_inlet_spec(inlet: str):
    """``NAME=vx,vy,vz`` -> (name, (vx, vy, vz))."""
    from meshforge.meshing import valid_group_name

    name, sep, raw = inlet.partition("=")
    parts = raw.split(",")
    if not sep or not name.strip() or len(parts) != 3:
        raise ValueError(f"inlet expects NAME=vx,vy,vz, got {inlet!r}")
    try:
        return valid_group_name(name.strip()), tuple(float(v) for v in parts)
    except ValueError as exc:
        if "invalid boundary-group name" in str(exc):
            raise
        raise ValueError(f"inlet velocity {raw!r} is not three numbers") from exc


def _parse_inlet_knob(inlet: str):
    """The cfd ``inlet`` knob: ``NAME=SPEED`` -> (name, speed) or
    ``NAME=vx,vy,vz`` -> (name, (vx, vy, vz)). The scalar is resolved
    against the mesh later (``solve/inflow.py``); only its syntax and sign
    are checked here, before any mesh is required."""
    from meshforge.meshing import valid_group_name
    from meshforge.solve.inflow import check_inlet_speed

    name, sep, raw = inlet.partition("=")
    parts = raw.split(",")
    if not sep or not name.strip() or len(parts) not in (1, 3):
        raise ValueError(
            f"inlet expects NAME=SPEED or NAME=vx,vy,vz, got {inlet!r}"
        )
    if len(parts) == 3:
        return _parse_inlet_spec(inlet)
    name = valid_group_name(name.strip())
    try:
        speed = float(parts[0])
    except ValueError as exc:
        raise ValueError(f"inlet speed {raw!r} is not a number") from exc
    return name, check_inlet_speed(speed, name)


def _parse_outlet_spec(outlet: str):
    """``NAME=value`` -> (name, p)."""
    from meshforge.meshing import valid_group_name

    name, sep, raw = outlet.partition("=")
    if not sep or not name.strip():
        raise ValueError(f"outlet expects NAME=VALUE, got {outlet!r}")
    valid_group_name(name.strip())
    try:
        return name.strip(), float(raw)
    except ValueError as exc:
        raise ValueError(f"outlet pressure {raw!r} is not a number") from exc


@dataclass
class StepResult:
    step: str
    summary: dict[str, Any]
    text: str
    render_hints: dict[str, Any] = field(default_factory=dict)


def crop_text(rep: dict[str, Any]) -> str:
    """Human-readable lines of a crop report (``geometry.crop.CropResult.report``)
    — shared by the crop step and the CLI."""
    lines = []
    lm = rep.get("landmarks")
    if lm:
        lines.append(
            f"landmarks   : sac h {lm['sac_height_mm']:.1f} mm, d_equiv "
            f"{lm['sac_d_equiv_mm']:.1f} mm (max diameter {lm['sac_max_diameter_mm']:.1f}); "
            f"neck from h {lm['neck_start_height_mm']:.1f} mm, d {lm['neck_d_equiv_mm']:.1f} "
            f"mm, length {'>= ' if lm['neck_open_ended'] else ''}{lm['neck_length_mm']:.1f} mm; "
            f"sac/neck {lm['sac_neck_ratio']:.2f}; flare top h {lm['flare_top_height_mm']:.1f} mm"
        )
    else:
        lines.append("landmarks   : not found (manual heights or planes)")
    for name in ("distal", "proximal"):
        pl, cap = rep["planes"][name], rep["caps"][name]
        lines.append(
            f"{name + ' cap':<12}: h {pl['height_mm']:.1f} mm, d_equiv {cap['d_equiv_mm']:.1f} mm, "
            f"{cap['rim_vertices']} rim vertices, planar to {cap['planarity_mm']:.1e} mm, rim "
            f"crease >= {cap['rim_crease_min_deg']:.1f} deg, normal "
            f"{[round(v, 4) for v in pl['normal_out']]}"
        )
    srf = rep["surface"]
    lines.append(
        f"cropped     : {srf['n_points']} pts, {srf['n_triangles']} tris, watertight "
        f"{srf['watertight']}, {srf['volume_ml']:.2f} mL"
    )
    if rep.get("lumen"):
        lum = rep["lumen"]
        lines.append(
            f"lumen       : {lum['n_triangles']} tris, watertight {lum['watertight']}, "
            f"{lum['volume_ml']:.2f} mL, same two planes"
        )
    lines += [f"note        : {n}" for n in rep.get("notes", [])]
    return "\n".join(lines) + "\n"


def _recut_parallel(surface, lumen_mesh, res, rep):
    """D92: re-cut a crop with both planes square to the chord between the two
    cut origins (parallel caps). The landmark rules square each cut to its own
    local axis; on a bent aorta those caps can be 35 deg apart, so the
    parallel-plane wall sweep cannot apply and the centreline sweep folds."""
    from types import SimpleNamespace

    import numpy as np

    from meshforge.geometry.crop import CutPlane, crop_with_planes, planes_from_report

    planes = planes_from_report(rep)
    chord = planes[1].origin - planes[0].origin
    norm = float(np.linalg.norm(chord))
    if not norm > 0:
        raise ValueError("parallel caps: the two cut origins coincide")
    c = chord / norm
    tilt = [float(np.degrees(np.arccos(np.clip(abs(pl.normal @ c), -1, 1)))) for pl in planes]
    new = [CutPlane(name=pl.name, origin=pl.origin, normal=(-c if k == 0 else c), height=pl.height)
           for k, pl in enumerate(planes)]
    cut, caps, note = crop_with_planes(surface, new, what="surface")
    lumen_cut = None
    if lumen_mesh is not None:
        lumen_cut, _, lnote = crop_with_planes(lumen_mesh, new, what="lumen")
    rep = dict(rep)
    rep["planes"] = {pl.name: pl.to_dict() for pl in new}
    rep["caps"] = {k: v.to_dict() for k, v in caps.items()}
    rep["notes"] = list(rep.get("notes", [])) + [
        f"parallel caps: both cuts squared to the chord between them "
        f"({norm:.1f} mm; the landmark cuts were {tilt[0]:.1f} / {tilt[1]:.1f} deg off it)",
        note,
    ]
    return SimpleNamespace(surface=cut, lumen=lumen_cut), rep


class MeshingSession:
    """Holds pipeline state: surface → centerline → volmesh → wall stress / flow."""

    def __init__(self) -> None:
        self.input_path: Path | None = None
        self.input_sha256: str | None = None
        self.surface = None
        self.truth = None  # AnalyticTruth when the surface is synthetic
        self.centerline = None
        self.volmesh = None
        self.cfd = None  # CfdResult from compute_cfd (V23)
        self.aaa_stress = None  # AaaStressResult from compute_aaa_stress (D88)
        # The AI outline of the whole aorta from the segment step. The sac and
        # crop steps replace ``surface`` with the piece they work on; this
        # stays so a front-end can draw the whole vessel behind that piece
        # (the Workbench's ghost). Cleared by every other source step.
        self.outline = None
        # D108 addendum 7: context organs from the same TotalSegmentator run
        # (``heart``: its own surface, never part of the vessel), drawn as
        # ghosts beside the outline. Filled by ``segment``, cleared with
        # ``outline`` wherever that is cleared.
        self.context: dict[str, Any] = {}
        # D93: the surface the last crop cut, and what it produced; a crop of
        # that unchanged output re-crops the source instead
        self._crop_source = None
        self._crop_output = None
        # Length unit of the current source, or None when it declares none
        # (STL/VTK/synthetic). Set by each source step; segment and sac are
        # millimetres by definition. Gates the cardiac CFD presets (D80).
        self.length_unit: str | None = None
        # V10 journal: ordered records of every step call — step name,
        # JSON-safe params (incl. paths + input checksums), result summary.
        self.journal: list[dict[str, Any]] = []

    def _record(
        self, step: str, params: dict[str, Any], result: "StepResult", **extra: Any
    ) -> None:
        self.journal.append(
            {"step": step, "params": params, "summary": result.summary, **extra}
        )

    # -- sources ---------------------------------------------------------
    def load(self, path: str | Path) -> StepResult:
        from meshforge.io import file_sha256, load_surface, normalize_path
        from meshforge.quality import inspect_surface

        path = normalize_path(path)
        # load_surface FIRST: it owns the friendly "surface file not found" and
        # the wrong-front-door suffix routing. Hashing first would have opened
        # the file and raised a bare [Errno 2] before either guard could speak.
        surface = load_surface(path)
        self.input_sha256 = file_sha256(path)
        self.surface = surface
        self.input_path = path
        self.truth = None
        self.outline = None  # not an AI outline
        self.context = {}
        self._invalidate_downstream()
        self.length_unit = None  # STL/VTK/... carry no unit
        report = inspect_surface(self.surface, path=str(path), sha256=self.input_sha256)
        result = StepResult(
            step="load",
            summary=report.to_dict(),
            text=report.to_text(),
            render_hints={"mesh": self.surface, "kind": "surface", "label": path.name},
        )
        self._record(
            "load", {"path": str(path)}, result, input_sha256=self.input_sha256
        )
        return result

    def generate_synth(self, shape: str, **params: Any) -> StepResult:
        from meshforge.synth import generate

        self.surface, self.truth = generate(shape, **params)
        self.input_path = None
        self.outline = None  # not an AI outline
        self.context = {}
        self.input_sha256 = None
        self._invalidate_downstream()
        self.length_unit = None  # synthetic shapes are unitless
        text = (
            f"synth {shape}: {self.surface.n_points} pts, {self.surface.n_cells} tris, "
            f"length {self.truth.length:.6g}, closed {self.truth.closed}\n"
        )
        result = StepResult(
            step="generate_synth",
            summary={"shape": shape, "params": params, "length": self.truth.length},
            text=text,
            render_hints={"mesh": self.surface, "kind": "surface", "label": shape},
        )
        self._record("generate_synth", {"shape": shape, **params}, result)
        return result

    @staticmethod
    def available_shapes() -> list[str]:
        """The synthetic-source shape names ``generate_synth`` accepts.

        Read-only catalog (the sorted ``synth.GENERATORS`` keys), used by the
        web layer's ``GET /catalog``. Lazy-imported so this
        module stays cheap to import."""
        from meshforge.synth import GENERATORS

        return sorted(GENERATORS)

    # -- pipeline steps ---------------------------------------------------
    def compute_centerline(self, **knobs: Any) -> StepResult:
        surface = self._require_surface()
        from meshforge.geometry import compute_centerline

        self.centerline = compute_centerline(surface, **knobs)
        s = self.centerline.summary()
        text = (
            f"centerline: {s['n_samples']} samples, length {s['length']:.6g}, "
            f"radius {s['radius_min']:.4g}..{s['radius_max']:.4g}, closed {s['closed']}\n"
        )
        import pyvista as pv

        poly = pv.lines_from_points(self.centerline.points)
        result = StepResult(
            step="centerline",
            summary=s,
            text=text,
            render_hints={"mesh": poly, "kind": "polyline", "label": "centerline"},
        )
        self._record("centerline", dict(knobs), result)
        return result

    def compute_volmesh(self, *, backend: str = "swept", thickness: float = 0.5, **knobs: Any) -> StepResult:
        surface = self._require_surface()
        from meshforge.meshing import get_backend

        recorded = {"backend": backend, "thickness": thickness}
        recorded.update({k: v for k, v in knobs.items() if k != "centerline"})
        if self.centerline is not None:
            knobs.setdefault("centerline", self.centerline)
        self.volmesh = get_backend(backend).mesh_wall(surface, thickness, **knobs)
        from meshforge.solve.aaa_metrics import annotate_wall_thickness

        annotate_wall_thickness(self.volmesh)  # D93: what the wall really is, along the normal
        s = self.volmesh.summary()
        q = s["quality"]
        text = (
            f"volmesh[{backend}]: {s['n_cells']} cells, inverted {q['n_inverted']}, "
            f"sj_min {q['sj_min']:.4g}, volume {q['total_volume']:.6g}\n"
            f"  boundary: {s['boundary_faces']}\n"
        )
        if "lattice" in s:
            lat = s["lattice"]
            text += (
                f"  wall rays: capped {lat['n_capped']}, floored {lat['n_floored']} of "
                f"{lat['n_rays']} (curvature_cap {lat['curvature_cap']:g}); thickness "
                f"{lat['thickness_min']:.4g}..{lat['thickness_max']:.4g}"
                + (f", along the surface normal median {lat['thickness_normal_median']:.4g}"
                   if lat.get("thickness_normal_median") is not None else "")
                + "\n"
            )
            if lat.get("sweep") == "planes":
                text += (
                    f"  plane sweep ({lat['side']}): capped by tilt {lat['n_capped_tilt']}, "
                    f"by concavity {lat['n_capped_concave']}; tilt max "
                    f"{lat['tilt_max_deg']:.1f} deg; nodes off the surface max "
                    f"{lat['surface_offset_max']:.3g} (section_smoothing "
                    f"{lat['section_smoothing']:g})\n"
                )
        result = StepResult(
            step="volmesh",
            summary=s,
            text=text,
            render_hints={"mesh": self.volmesh.grid, "kind": "volume", "label": f"wall ({backend})"},
        )
        self._record("volmesh", recorded, result)
        return result

    def compute_lumen(self, *, method: str = "auto", **knobs: Any) -> StepResult:
        surface = self._require_surface()
        from meshforge.meshing.lumen import fill_lumen

        recorded = {"method": method}
        recorded.update({k: v for k, v in knobs.items() if k != "centerline"})
        if self.centerline is not None:
            knobs.setdefault("centerline", self.centerline)
        # The lumen mesh takes the volmesh slot: it IS the session's volume
        # mesh (region-tagged, wall/inlet/outlet groups), so CFD and save
        # work on it unchanged (V47).
        self.volmesh = fill_lumen(surface, method=method, **knobs)
        s = self.volmesh.summary()
        q = s["quality"]
        text = (
            f"lumen[{s['backend']}]: {s['n_cells']} cells, "
            f"inverted {q['n_inverted']}, sj_min {q['sj_min']:.4g}, "
            f"volume {q['total_volume']:.6g}\n"
            f"  boundary: {s['boundary_faces']}\n"
        )
        for note in s["notes"]:
            text += f"  note: {note}\n"
        result = StepResult(
            step="lumen",
            summary=s,
            text=text,
            render_hints={
                "mesh": self.volmesh.grid,
                "kind": "volume",
                "label": f"lumen ({s['backend']})",
            },
        )
        self._record("lumen", recorded, result)
        return result

    def compute_segment(
        self,
        *,
        path: str = "",
        device: str = "mps",
        fast: int = 0,
        smoothing: float = 1.0,
        timeout: float = 1800.0,
        lumen_out: str = "",
        lumen_hu: float = 150.0,
        mask_out: str = "",
    ) -> StepResult:
        """D83 (M1): CT (.nrrd/.nii/.nii.gz) -> TotalSegmentator aorta mask ->
        watertight surface as the session source, in RAS millimetres.

        D85: ``lumen_out`` also writes the contrast blood lumen (CT >=
        ``lumen_hu`` inside the AI mask, largest piece — a threshold, not AI)
        to that .vtp; the session source stays the AI outline.

        All work lives in ``io/segment.py`` (rule 4); TotalSegmentator runs in
        its own env as a subprocess, staged in a temp dir — nothing is written
        next to the CT or into outputs/ by this step.
        """
        path = _clean_path(path)
        if not path:
            raise ValueError("segment needs a CT volume path (.nrrd or .nii/.nii.gz)")
        from meshforge.io.segment import segment_ct
        from meshforge.quality import inspect_surface

        lumen_out = _clean_path(lumen_out)
        if lumen_out:
            from meshforge.export import guard_output_path

            guard_output_path(lumen_out, input_paths=[path])
        mask_out = _clean_path(mask_out)
        if mask_out:
            from meshforge.export import guard_output_path

            guard_output_path(mask_out, input_paths=[path])
            Path(mask_out).parent.mkdir(parents=True, exist_ok=True)
        seg = segment_ct(
            path,
            device=device,
            fast=bool(fast),
            smoothing=float(smoothing),
            timeout=float(timeout),
            lumen_hu=float(lumen_hu) if lumen_out else None,
            keep_mask=mask_out or None,
        )
        if lumen_out:
            Path(lumen_out).parent.mkdir(parents=True, exist_ok=True)
            seg.lumen_surface.save(lumen_out)
        self.surface = seg.surface
        self.outline = seg.surface  # the whole vessel, kept through sac/crop
        self.context = dict(seg.context_surfaces)  # the heart, as a ghost
        self.input_path = Path(path)
        self.input_sha256 = seg.input_sha256
        self.truth = None
        self._invalidate_downstream()
        self.length_unit = "mm"  # NRRD anatomical space / NIfTI mm (D83)
        summ = seg.summary
        ts, mask, surf = summ["totalseg"], summ["mask"], summ["surface"]
        report = inspect_surface(self.surface, path=str(path), sha256=self.input_sha256)
        text = (
            f"segment: TotalSegmentator {ts['version'] or '(version unknown)'} "
            f"task total, roi {'+'.join(ts['rois'])}, "
            f"{'fast' if ts['fast'] else 'full'} on {ts['device']}, "
            f"{ts['wall_time_s']:.1f} s\n"
            f"  mask: {mask['kept_voxels']} voxels kept "
            f"({summ['mask_volume_ml']:.1f} mL); dropped "
            f"{mask['dropped_components']} component(s) / "
            f"{mask['dropped_voxels']} voxels; smoothing sigma {mask['sigma']:g} vox\n"
            f"  surface: {surf['n_points']} pts, {surf['n_triangles']} tris, "
            f"watertight {surf['watertight']}, frame {surf['frame']}\n"
        )
        for roi, c in summ.get("context", {}).items():
            text += (
                f"  context {roi}: {c['n_points']} pts, {c['n_triangles']} tris, "
                f"{c['mask_volume_ml']:.1f} mL — a ghost in the viewport, not part of "
                f"the vessel\n"
            )
        for note in summ["notes"]:
            text += f"  note: {note}\n"
        if lumen_out:
            lum = summ["lumen"]
            text += (
                f"  lumen: {lum['kept_voxels']} voxels >= {lum['lumen_hu']:g} HU kept "
                f"({lum['lumen_volume_ml']:.1f} mL, {100 * lum['fraction_of_mask']:.1f}% of "
                f"the AI mask, Dice {lum['dice_vs_ai_mask']:.4f}); dropped "
                f"{lum['dropped_components']} piece(s) / {lum['dropped_voxels']} voxels -> "
                f"{lumen_out}\n  lumen: {lum['method']}\n"
            )
        text += report.to_text()
        result = StepResult(
            step="segment",
            summary={**report.to_dict(), "segment": summ},
            text=text,
            render_hints={
                "mesh": self.surface, "kind": "surface", "label": Path(path).name
            },
        )
        self._record(
            "segment",
            {
                "path": path,
                "device": device,
                "fast": int(fast),
                "smoothing": float(smoothing),
                "timeout": float(timeout),
                "lumen_out": lumen_out,
                "lumen_hu": float(lumen_hu),
                "mask_out": mask_out,
            },
            result,
            input_sha256=self.input_sha256,
        )
        return result

    def compute_crop(
        self,
        *,
        neck_extension_mm: float = 25.0,
        distal_margin_mm: float = 10.0,
        distal_height_mm: float | None = None,
        proximal_height_mm: float | None = None,
        lumen: str = "",
        lumen_out: str = "",
        parallel_caps: int = 0,
    ) -> StepResult:
        """D85 (M2): crop the session surface (an aorta in millimetres, +z
        superior) to its infrarenal aneurysm segment with two flat caps square
        to the centreline; an optional lumen surface is cut by the SAME planes
        and written to ``lumen_out``. All work lives in ``geometry/crop.py``.

        A crop of an unchanged crop starts again from the surface the first
        crop cut (D93): the landmarks need the whole aorta, and cutting a
        crop trims it further (on s1364 the sac went 134.8 -> 93.7 mL)."""
        surface = self._require_surface()
        recrop = self._crop_output is not None and surface is self._crop_output
        if recrop:
            surface = self._crop_source
        if self.length_unit not in (None, "mm"):
            raise ValueError(
                f"crop works in millimetres; this surface is in {self.length_unit!r}"
            )
        lumen, lumen_out = _clean_path(lumen), _clean_path(lumen_out)
        if bool(lumen) != bool(lumen_out):
            raise ValueError(
                "crop: give both lumen (the surface to cut) and lumen_out (where the cut "
                "lumen is written), or neither"
            )
        from meshforge.geometry.crop import crop_aaa
        from meshforge.quality import inspect_surface

        lumen_mesh = None
        if lumen:
            from meshforge.export import guard_output_path
            from meshforge.io import load_surface

            guard_output_path(lumen_out, input_paths=[self.input_path, lumen])
            lumen_mesh = load_surface(lumen)
        res = crop_aaa(
            surface,
            neck_extension_mm=float(neck_extension_mm),
            distal_margin_mm=float(distal_margin_mm),
            distal_height_mm=None if distal_height_mm is None else float(distal_height_mm),
            proximal_height_mm=None if proximal_height_mm is None else float(proximal_height_mm),
            lumen=lumen_mesh,
        )
        rep = res.report()
        rep.pop("profile", None)  # every section: in the CLI's JSON, not the journal
        if parallel_caps:
            res, rep = _recut_parallel(surface, lumen_mesh, res, rep)
        if self.length_unit is None:
            rep["notes"].append("the surface carries no length unit: millimetres assumed")
        if recrop:
            rep["notes"].append("re-cropped from the uncropped surface, not from the previous crop")
        self.surface = res.surface
        self._crop_source, self._crop_output = surface, res.surface
        self._invalidate_downstream()
        if lumen_out:
            Path(lumen_out).parent.mkdir(parents=True, exist_ok=True)
            res.lumen.save(lumen_out)
        report = inspect_surface(self.surface, path=str(self.input_path), sha256=self.input_sha256)
        text = crop_text(rep) + report.to_text()
        import pyvista as pv

        from meshforge.geometry import centerline_from_surface

        hints: dict[str, Any] = {"mesh": self.surface, "kind": "surface", "label": "AAA crop"}
        seg_cl = centerline_from_surface(self.surface)
        if seg_cl is not None:
            hints["overlay"] = pv.lines_from_points(seg_cl.points)
            hints["overlay_label"] = "segment centreline"
        result = StepResult(
            step="crop", summary={**report.to_dict(), "crop": rep}, text=text,
            render_hints=hints,
        )
        self._record(
            "crop",
            {
                "neck_extension_mm": float(neck_extension_mm),
                "distal_margin_mm": float(distal_margin_mm),
                "distal_height_mm": distal_height_mm,
                "proximal_height_mm": proximal_height_mm,
                "lumen": lumen,
                "lumen_out": lumen_out,
                "parallel_caps": int(bool(parallel_caps)),
            },
            result,
            input_sha256=self.input_sha256,
        )
        return result

    def compute_sac(
        self,
        *,
        path: str = "",
        lumen_mask: str = "",
        lumen_hu: float = 150.0,
        fat_hu: float = -30.0,
        max_thickness_mm: float = 40.0,
        n_rays: int = 72,
        roi_mm: float = 150.0,
        min_resolved: float = 0.5,
        smoothing: float = 1.0,
    ) -> StepResult:
        """D86 (M2): CT + lumen-only mask -> outer aneurysm sac (lumen +
        thrombus + wall) as the session source surface, RAS millimetres, with
        point fields ``ilt_thickness_mm`` and ``sac_resolved``.

        All work lives in ``geometry/sac.py`` (rule 4); nothing is written.
        """
        path, lumen_mask = _clean_path(path), _clean_path(lumen_mask)
        if not path or not lumen_mask:
            raise ValueError(
                "sac needs a CT volume path and a lumen mask path on the same grid"
            )
        from meshforge.geometry.sac import recover_sac_from_files
        from meshforge.quality import inspect_surface

        run = recover_sac_from_files(
            path,
            lumen_mask,
            lumen_hu=float(lumen_hu),
            fat_hu=float(fat_hu),
            max_thickness_mm=float(max_thickness_mm),
            n_rays=int(n_rays),
            roi_mm=float(roi_mm),
            min_resolved=float(min_resolved),
            smoothing=float(smoothing),
        )
        self.surface = run.surface
        self.input_path = Path(path)
        self.input_sha256 = run.input_sha256["ct"]
        self.truth = None
        self._invalidate_downstream()
        self.length_unit = "mm"
        summ = run.summary
        report = inspect_surface(self.surface, path=str(path), sha256=self.input_sha256)
        text = sac_text(summ)
        text += report.to_text()
        result = StepResult(
            step="sac",
            summary={**report.to_dict(), "sac": summ},
            text=text,
            render_hints={"mesh": self.surface, "kind": "surface", "label": "sac"},
        )
        self._record(
            "sac",
            {
                "path": path,
                "lumen_mask": lumen_mask,
                "lumen_hu": float(lumen_hu),
                "fat_hu": float(fat_hu),
                "max_thickness_mm": float(max_thickness_mm),
                "n_rays": int(n_rays),
                "roi_mm": float(roi_mm),
                "min_resolved": float(min_resolved),
                "smoothing": float(smoothing),
            },
            result,
            input_sha256=self.input_sha256,
        )
        return result

    def compute_repair(
        self, *, merge_tol: float | None = None, max_hole_edges: int = 16
    ) -> StepResult:
        surface = self._require_surface()
        from meshforge.quality import repair_surface

        repaired, report = repair_surface(
            surface, merge_tol=merge_tol, max_hole_edges=max_hole_edges
        )
        self.surface = repaired
        self._invalidate_downstream()
        result = StepResult(
            step="repair",
            summary=report.to_dict(),
            text=report.to_text(),
            render_hints={"mesh": repaired, "kind": "surface", "label": "repaired"},
        )
        self._record(
            "repair",
            {"merge_tol": merge_tol, "max_hole_edges": max_hole_edges},
            result,
        )
        return result

    def compute_aaa_stress(
        self,
        *,
        systolic_mmhg: float = 140.0,
        whatif_mmhg: float = 120.0,
        E: float = 100.0,
        nu: float = 0.49,
        end_zone: str = "neck",
        crop_report: str = "",
        lumen: str = "",
        sac: str = "",
        E_ilt: float = 5.0,
        ccx: str = "",
        timeout: float = 600.0,
    ) -> StepResult:
        """D88 (M3) + D91 (thrombus): AAA wall stress on the session's swept wall — an
        INVESTIGATIONAL metric. Stiff-linear CalculiX solves (pressure on
        ``inner``, ``inlet``/``outlet`` encastre) at ``systolic_mmhg`` and
        ``whatif_mmhg``; peak wall stress (99th percentile of the
        integration-point von Mises outside the end zones), max principal,
        wall tension. A wall with inverted cells or capped/floored wall rays
        is refused (the metrics are None and the text says why). All work
        lives in ``solve/aaa_metrics.py``; decks, results and the
        ``session_aaa_stress.vtu/.json`` go to ``outputs/aaa_stress/session``.
        """
        import json

        from meshforge.solve.aaa_metrics import crop_report_from_journal, run_aaa_stress

        vol = self._require(self.volmesh, "volmesh", "compute_volmesh")
        crop_report, ccx = _clean_path(crop_report), _clean_path(ccx)
        if crop_report:
            path = Path(crop_report)
            if not path.is_file():
                raise ValueError(f"crop report not found: {crop_report}")
            report = json.loads(path.read_text())
        else:
            report = crop_report_from_journal(self.journal)
        lumen, sac = _clean_path(lumen), _clean_path(sac)
        lumen_surface = sac_surface = None
        if lumen or sac:
            from meshforge.io import load_surface

            for name, path in (("lumen", lumen), ("sac", sac)):
                if path and not Path(path).is_file():
                    raise ValueError(f"{name} surface not found: {path}")
            lumen_surface = load_surface(lumen) if lumen else None
            sac_surface = load_surface(sac) if sac else None
        if lumen_surface is not None and sac_surface is None and report is not None:
            # D92: after a crop, the session surface IS the cropped sac the
            # wall was built from - the thrombus fills between it and the lumen
            sac_surface = self.surface
            sac = "(the session's cropped surface)"
        params = {
            "systolic_mmhg": float(systolic_mmhg), "whatif_mmhg": float(whatif_mmhg),
            "E": float(E), "nu": float(nu), "end_zone": str(end_zone),
            "crop_report": crop_report, "lumen": lumen, "sac": sac, "E_ilt": float(E_ilt),
            "ccx": ccx, "timeout": float(timeout),
        }
        out_dir = Path("outputs") / "aaa_stress" / "session"
        res = run_aaa_stress(
            vol, vol.summary(), out_dir=out_dir, stem="session", crop_report=report,
            systolic_mmhg=params["systolic_mmhg"], whatif_mmhg=params["whatif_mmhg"],
            E=params["E"], nu=params["nu"], end_zone=params["end_zone"],
            ccx=ccx or None, timeout=params["timeout"], input_path=self.input_path,
            lumen=lumen_surface, sac=sac_surface, E_ilt=params["E_ilt"],
        )
        self.aaa_stress = res
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "session_aaa_stress.json").write_text(
            json.dumps(res.summary, indent=2) + "\n"
        )
        if res.grid is not None:
            res.grid.save(out_dir / "session_aaa_stress.vtu")
        result = StepResult(
            step="aaa_stress",
            summary=res.summary,
            text=res.to_text(),
            render_hints={
                "mesh": res.grid if res.grid is not None else vol.grid,
                "kind": "volume",
                "label": "wall stress (investigational)",
            },
        )
        self._record("aaa_stress", params, result)
        return result

    def compute_cfd(
        self,
        *,
        case: str = "",
        solver: str = "simpleFoam",
        nu: float = 1e-3,
        inlet: str = "",
        outlet: str = "",
        end_time: float = 2000.0,
        delta_t: float = 0.0,
        write_interval: float = 0.0,
        timeout: float = 600.0,
        turbulence: str = "laminar",
        intensity: float = 0.05,
        turb_length: float = 0.0,
        series: int = 0,
        moving_wall: str = "",
        empty: str = "",
        closed_cavity: int = 0,
        p_ref_cell: int = 0,
        p_ref_value: float = 0.0,
    ) -> StepResult:
        """V23/V27: run OpenFOAM and read U/p (+k/omega/nut) back.

        With ``case`` empty, an incompressible case — laminar or steady
        kOmegaSST RANS per the ``turbulence`` knob — is built from the
        session volmesh under outputs/cfd/session/ (patch BCs from the
        inlet/outlet knobs, every other patch a no-slip wall); otherwise
        the given complete case runs as-is and the BC knobs must stay empty.
        With ``series`` (V29), EVERY written time step is read back as a
        frame — a read-back knob, so it applies to both forms.
        """
        from meshforge.solve import solve_case

        case = _clean_path(case)
        params = {
            "case": case, "solver": solver, "nu": nu, "inlet": inlet,
            "outlet": outlet, "end_time": end_time, "delta_t": delta_t,
            "write_interval": write_interval, "timeout": timeout,
            "turbulence": turbulence, "intensity": intensity,
            "turb_length": turb_length, "series": series,
            "moving_wall": moving_wall, "empty": empty,
            "closed_cavity": closed_cavity, "p_ref_cell": p_ref_cell,
            "p_ref_value": p_ref_value,
        }
        inflow = section = None  # session-built cases only (D80/D82)
        if case:
            if inlet or outlet or moving_wall or empty:
                raise ValueError(
                    "an existing case already carries its BCs — the "
                    "inlet/outlet/moving_wall/empty knobs apply only to "
                    "session-built cases"
                )
            if turbulence != "laminar":
                raise ValueError(
                    "an existing case already carries its turbulence setup "
                    "— the turbulence knobs apply only to session-built cases"
                )
            if closed_cavity:
                raise ValueError(
                    "an existing case already carries its fvSolution — the "
                    "closed_cavity/pRef knobs apply only to session-built "
                    "cases"
                )
            case_dir = Path(case)
            self.cfd = solve_case(case_dir, timeout=timeout, series=bool(series))
        else:
            # Cheap validation first: knob syntax and spec consistency are
            # checked before any mesh is required.
            import dataclasses

            from meshforge.solve import CfdBC, CfdSpec, write_case

            bcs = {}
            inlet_name = inlet_speed = None
            if inlet:
                inlet_name, velocity = _parse_inlet_knob(inlet)
                if isinstance(velocity, float):
                    # D80: a scalar speed along the patch's inward normal,
                    # resolved once the mesh is in hand. A provisional vector
                    # of the same magnitude lets the spec validate now (the
                    # turbulence inlet scales use |U| only).
                    inlet_speed, velocity = velocity, (velocity, 0.0, 0.0)
                bcs[inlet_name] = CfdBC("velocity", velocity)
            if outlet:
                name, p_value = _parse_outlet_spec(outlet)
                bcs[name] = CfdBC("pressure", p_value)
            if moving_wall:
                name, velocity = _parse_inlet_spec(moving_wall)
                bcs[name] = CfdBC("moving_wall", velocity)
            for patch in filter(None, (p.strip() for p in empty.split(","))):
                bcs[patch] = CfdBC("empty")
            spec = CfdSpec(
                solver=solver,
                nu=nu,
                bcs=bcs,
                end_time=end_time,
                delta_t=delta_t or None,
                write_interval=write_interval or None,
                turbulence=turbulence,
                turbulence_intensity=intensity,
                turbulence_length=turb_length or None,
                closed_cavity=bool(closed_cavity),
                p_ref_cell=p_ref_cell,
                p_ref_value=p_ref_value,
            )
            vol = self._require(
                self.volmesh, "volmesh", "compute_lumen or compute_volmesh"
            )
            if vol.quality.n_inverted > 0:
                raise ValueError(
                    f"refusing to run CFD on a mesh with "
                    f"{vol.quality.n_inverted} inverted cell(s)"
                )
            inflow_notes_ = []
            if inlet_name is not None:
                from meshforge.solve.inflow import (
                    inflow_notes,
                    inflow_report,
                    patch_geometry,
                    resolve_inlet_speed,
                )

                geo = None
                if inlet_speed is not None:
                    velocity, geo = resolve_inlet_speed(vol, inlet_name, inlet_speed)
                    spec = dataclasses.replace(
                        spec,
                        bcs={**spec.bcs, inlet_name: CfdBC("velocity", velocity)},
                    )
                else:
                    velocity = spec.bcs[inlet_name].value
                    try:  # the vector form needs no geometry - report if we can
                        geo = patch_geometry(vol, inlet_name)
                    except ValueError as exc:
                        inflow_notes_.append(f"inflow report unavailable: {exc}")
                if geo is not None:
                    inflow = inflow_report(geo, velocity, nu, self.length_unit)
                    inflow_notes_ += inflow_notes(
                        inflow, laminar=turbulence == "laminar"
                    )
            case_dir = Path("outputs") / "cfd" / "session"
            write_case(vol, case_dir, spec, input_path=self.input_path)
            self.cfd = solve_case(
                case_dir, solver=solver, timeout=timeout, series=bool(series)
            )
            self.cfd.notes.extend(inflow_notes_)
            section = self._flow_section(vol, through_flow=inlet_name is not None)

        text = self.cfd.to_text()
        grid = self.cfd.grid
        render_hints = {"mesh": grid, "kind": "volume", "label": "cfd results"}
        if self.cfd.frames is not None:
            render_hints["frames"] = list(
                zip(self.cfd.times, self.cfd.frames)
            )
        if section is not None:
            # D82: the viewport samples the flow onto this centreline section
            # and looks at it face-on, so the inflow is visible, not the wall.
            render_hints["section"] = section.surface
            render_hints["view_dir"] = list(section.view_dir)
        summary = self.cfd.summary()
        if inflow is not None:
            summary["inflow"] = inflow
        result = StepResult(
            step="cfd",
            summary=summary,
            text=text,
            render_hints=render_hints,
        )
        self._record("cfd", params, result)
        return result

    def compute_tissue_index(
        self,
        *,
        pws_mpa: float | None = None,
        from_summary: str = "",
        selection: str = "aaa",
        method: str = "empirical",
    ) -> StepResult:
        """D89 (M4): strength index band (peak wall stress / lab-measured wall strength), P10/P50/P90.

        Strength = Cauchy failure stress of the packaged tissue library v0
        (open USP AAA strip tests, CC BY 4.0), compared like-for-like with the
        Cauchy wall stress of the solve. Investigational, not a diagnosis.
        Exactly one PWS source: ``pws_mpa`` or ``from_summary`` (a wall-stress
        summary JSON with a ``pws_mpa`` key, read only). Session geometry is
        neither needed nor changed; all logic lives in ``tissue/strength.py``.
        """
        from meshforge.tissue.strength import (
            check_pws,
            index_band_report,
            index_band_text,
            pws_from_summary,
        )

        from_summary = _clean_path(from_summary)
        if (pws_mpa is None) == (not from_summary):
            raise ValueError(
                "tissue_index needs exactly one peak wall stress source: pws_mpa (MPa) "
                "or from_summary (a wall-stress summary .json with a pws_mpa key)"
            )
        extra: dict[str, Any] = {}
        if from_summary:
            pws, where, digest = pws_from_summary(from_summary)
            pws_source = f"{Path(from_summary).name}: {where}"
            extra["input_sha256"] = digest
        else:
            pws, pws_source = check_pws(pws_mpa), "entered"
        report = index_band_report(pws, selection=selection, method=method)
        report["pws_source"] = pws_source
        result = StepResult(
            step="tissue_index",
            summary=report,
            text=index_band_text(report, pws_source=pws_source),
        )
        self._record(
            "tissue_index",
            {
                "pws_mpa": None if from_summary else pws,
                "from_summary": from_summary,
                "selection": selection,
                "method": method,
            },
            result,
            **extra,
        )
        return result

    # -- output -----------------------------------------------------------
    def save(self, what: str, out_path: str | Path, *, fmt: str | None = None) -> StepResult:
        from meshforge.export import export_surface, guard_output_path

        out_path = Path(_clean_path(out_path))
        if what == "surface":
            mesh = self._require_surface()
            written = export_surface(mesh, out_path, fmt=fmt, input_path=self.input_path)
        elif what == "volmesh":
            vol = self._require(self.volmesh, "volmesh", "compute_volmesh")
            guard_output_path(out_path, input_paths=[self.input_path])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            vol.grid.save(out_path)
            written = out_path
        elif what == "abaqus":
            from meshforge.export import export_abaqus

            vol = self._require(self.volmesh, "volmesh", "compute_volmesh")
            written = export_abaqus(
                vol, out_path, formulation=fmt or "hybrid", input_path=self.input_path
            )
        elif what == "foam":
            from meshforge.export.foam import write_polymesh

            vol = self._require(self.volmesh, "volmesh", "compute_volmesh")
            written = write_polymesh(vol, out_path, input_path=self.input_path)
        else:
            raise ValueError(
                f"unknown save target {what!r}; "
                f"use surface | volmesh | abaqus | foam"
            )
        text = f"saved {what} -> {written}\n"
        result = StepResult(
            step="save", summary={"what": what, "path": str(written)}, text=text
        )
        self._record(
            "save", {"what": what, "out_path": str(out_path), "fmt": fmt}, result
        )
        return result

    # -- D80: inflow picker + presets (read-only; nothing is journaled) ----
    # The Workbench turns these into the cfd knobs' plain scalar form
    # (inlet=NAME=SPEED, outlet=NAME=0, nu=...), so a replay of the
    # recorded step is identical to the click that produced it.
    def flow_ends(self, point) -> dict[str, Any]:
        """Which flat end cap of the session volume mesh is the inflow, given
        a point picked near it; the other cap is the outflow."""
        from meshforge.solve.inflow import flow_ends

        vol = self._require(self.volmesh, "volmesh", "compute_lumen or compute_volmesh")
        return flow_ends(vol, point)

    def inflow_preview(self, patch: str, speed: float, nu: float) -> dict[str, Any]:
        """Velocity, flow rate and Reynolds number a scalar inlet speed would
        prescribe on *patch* - before anything runs."""
        from meshforge.solve.inflow import inflow_notes, inflow_report, resolve_inlet_speed

        vol = self._require(self.volmesh, "volmesh", "compute_lumen or compute_volmesh")
        if not nu > 0.0:
            raise ValueError(f"kinematic viscosity must be positive, got {nu}")
        velocity, geo = resolve_inlet_speed(vol, patch, speed)
        report = inflow_report(geo, velocity, nu, self.length_unit)
        report["notes"] = inflow_notes(report)
        return report

    def cfd_presets(self) -> dict[str, Any]:
        """Cardiac nu / inlet-speed presets in this source's length unit
        (loud when the source declares no unit)."""
        from meshforge.solve.inflow import cardiac_presets

        return cardiac_presets(self.length_unit)

    # -- M6: attached flow metrics (journaled, never computed) ----------------
    def attach_flow_metrics(self, metrics: dict[str, Any]) -> StepResult:
        """Journal pre-computed steady-CFD metrics of this scan's lumen (the
        copilot's ``--flow-metrics``) as their own ``flow_metrics`` record, so
        the twin card's blood-flow slot reads them from the session. Nothing
        is computed and no mesh is touched; a replay re-records the same
        numbers. The metrics must pass :func:`check_flow_metrics`."""
        metrics = check_flow_metrics(metrics)
        listed = ", ".join(f"{k} {v:g}" for k, v in metrics.items())
        result = StepResult(
            step="flow_metrics",
            summary={"metrics": dict(metrics),
                     "note": "pre-computed CFD metrics, attached; nothing was computed"},
            text=f"flow metrics: {len(metrics)} pre-computed CFD value(s) attached, "
            f"nothing computed ({listed})\n",
        )
        self._record("flow_metrics", {"metrics": dict(metrics)}, result)
        return result

    def _flow_section(self, vol, *, through_flow: bool):
        """D82: the centreline section a CFD result is viewed on, or None.

        A display aid, never a result: failures become a note on the CFD
        text instead of failing a solve that succeeded."""
        if self.centerline is None:
            if through_flow:
                self.cfd.notes.append(
                    "flow section: no session centreline (run centerline) - "
                    "the viewport shows the boundary only"
                )
            return None
        from meshforge.geometry.ribbon import centerline_ribbon

        try:
            section = centerline_ribbon(self.centerline, vol.grid)
        except ValueError as exc:
            self.cfd.notes.append(f"flow section unavailable: {exc}")
            return None
        self.cfd.notes.extend(f"flow section: {n}" for n in section.notes)
        return section

    # -- helpers -----------------------------------------------------------
    def _require_surface(self):
        return self._require(self.surface, "surface", "load or generate_synth")

    @staticmethod
    def _require(value, name: str, producer: str):
        if value is None:
            raise RuntimeError(f"no {name} available — run {producer} first")
        return value

    def _invalidate_downstream(self) -> None:
        self.centerline = None
        self.volmesh = None
        self.cfd = None
        self.aaa_stress = None

    # -- work files (V10) --------------------------------------------------
    def save_workfile(self, path: str | Path) -> Path:
        """Persist the journal as a .mfwork JSON work file."""
        import json

        if not self.journal:
            raise ValueError("nothing to save: the session journal is empty")
        path = Path(_clean_path(path))
        if path.suffix != ".mfwork":
            path = path.with_suffix(".mfwork")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"version": 1, "journal": self.journal}, indent=2) + "\n"
        )
        return path


#: M6: the twin card's flow lines read these two; an attached set must name one
FLOW_METRICS_SHOWN = ("sac_backflow_volume_fraction", "q_lmin")
FLOW_METRICS_MAX = 32


def check_flow_metrics(metrics: Any) -> dict[str, int | float]:
    """``metrics`` as a plain dict if it is a set of flow metrics a session may
    carry, else a loud ValueError: 1 to 32 entries, plain-name keys (letters,
    digits, ``_``; at most 64 characters), finite numbers only (no bool,
    string, list or null; at most 1e12 in size), naming at least one metric
    the twin card shows, with the backflow a fraction in [0, 1] and the
    inflow positive."""
    import math
    import re

    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("flow metrics must be a non-empty JSON object of numbers")
    if len(metrics) > FLOW_METRICS_MAX:
        raise ValueError(f"flow metrics: {len(metrics)} entries, at most {FLOW_METRICS_MAX}")
    out: dict[str, int | float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", key):
            raise ValueError(
                f"flow metrics: key {str(key)[:40]!r} is not a plain name "
                "(letters, digits, _; at most 64 characters)"
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"flow metrics: {key} must be a number, got {type(value).__name__}")
        try:
            finite = math.isfinite(float(value)) and abs(float(value)) <= 1e12
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"flow metrics: {key} must be a finite number (at most 1e12 in size)")
        out[key] = value
    if not any(k in out for k in FLOW_METRICS_SHOWN):
        raise ValueError(
            f"flow metrics: none of the twin card's metrics ({', '.join(FLOW_METRICS_SHOWN)})"
        )
    backflow = out.get("sac_backflow_volume_fraction")
    if backflow is not None and not 0.0 <= backflow <= 1.0:
        raise ValueError(
            f"flow metrics: sac_backflow_volume_fraction is a fraction in [0, 1], got {backflow}"
        )
    if "q_lmin" in out and not out["q_lmin"] > 0.0:
        raise ValueError(f"flow metrics: q_lmin must be positive, got {out['q_lmin']}")
    return out


_WORKFILE_STEPS = (
    "load",
    "generate_synth",
    "centerline",
    "volmesh",
    "lumen",
    "save",
    "segment",
    "sac",
    "crop",
    "repair",
    "aaa_stress",
    "cfd",
    "tissue_index",
    "flow_metrics",
)

def load_workfile(path: str | Path) -> MeshingSession:
    """Load a .mfwork file into a fresh session (journal only, no state);
    run :func:`replay` to recompute. Corrupted files fail loudly."""
    import json

    path = Path(_clean_path(path))
    if not path.is_file():
        raise FileNotFoundError(f"work file not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"corrupted work file (not valid JSON): {path}: {exc}") from exc
    if (
        not isinstance(data, dict)
        or data.get("version") != 1
        or not isinstance(data.get("journal"), list)
        or not data["journal"]
    ):
        raise ValueError(
            f"corrupted work file: {path} needs version=1 and a non-empty journal"
        )
    for i, rec in enumerate(data["journal"]):
        if (
            not isinstance(rec, dict)
            or rec.get("step") not in _WORKFILE_STEPS
            or not isinstance(rec.get("params"), dict)
        ):
            raise ValueError(
                f"corrupted work file: journal[{i}] needs a known step "
                f"({'/'.join(_WORKFILE_STEPS)}) and a params mapping"
            )
        if rec["step"] == "flow_metrics":
            try:
                check_flow_metrics(rec["params"].get("metrics"))
            except ValueError as exc:
                raise ValueError(f"corrupted work file: journal[{i}] {exc}") from exc
    session = MeshingSession()
    session.journal = data["journal"]
    return session


def replay(
    session: MeshingSession, overrides: dict[str, dict[str, Any]] | None = None
) -> list[StepResult]:
    """Re-run the journal, applying ``overrides`` ({step: {param: value}}).

    Only steps from the earliest overridden one onward are re-executed —
    upstream results already held by the session are reused (the existing
    downstream-invalidation semantics). A freshly loaded work-file session
    holds no state, so everything runs.
    """
    if not session.journal:
        raise ValueError("cannot replay: session journal is empty")
    overrides = overrides or {}
    step_names = [rec["step"] for rec in session.journal]
    for name in overrides:
        if name not in step_names:
            raise ValueError(
                f"override targets unknown step {name!r}; journal has {step_names}"
            )

    start = 0
    if overrides and session.surface is not None:
        start = min(step_names.index(n) for n in overrides)
        # A source override invalidates everything anyway.
    if session.surface is None:
        start = 0

    plan = [
        {
            "step": rec["step"],
            "params": {**rec["params"], **overrides.get(rec["step"], {})},
        }
        for rec in session.journal
    ]
    session.journal = session.journal[:start]  # re-run steps re-record

    results: list[StepResult] = []
    for entry in plan[start:]:
        name, params = entry["step"], dict(entry["params"])
        if name == "load":
            results.append(session.load(params["path"]))
        elif name == "generate_synth":
            results.append(session.generate_synth(params.pop("shape"), **params))
        elif name == "centerline":
            results.append(session.compute_centerline(**params))
        elif name == "volmesh":
            results.append(session.compute_volmesh(**params))
        elif name == "lumen":
            results.append(session.compute_lumen(**params))
        elif name == "save":
            results.append(
                session.save(
                    params["what"], params["out_path"], fmt=params.get("fmt")
                )
            )
        elif name == "segment":
            results.append(session.compute_segment(**params))
        elif name == "crop":
            results.append(session.compute_crop(**params))
        elif name == "sac":
            results.append(session.compute_sac(**params))
        elif name == "repair":
            results.append(session.compute_repair(**params))
        elif name == "aaa_stress":
            results.append(session.compute_aaa_stress(**params))
        elif name == "cfd":
            results.append(session.compute_cfd(**params))
        elif name == "tissue_index":
            results.append(session.compute_tissue_index(**params))
        elif name == "flow_metrics":  # attached numbers: re-recorded, never recomputed
            results.append(session.attach_flow_metrics(**params))
        else:  # pragma: no cover - load_workfile validates step names
            raise ValueError(f"unknown journal step {name!r}")
    return results
