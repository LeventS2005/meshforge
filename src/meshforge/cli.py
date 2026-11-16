from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _path_arg(raw: str) -> Path:
    r"""argparse ``type=`` for every path argument.

    Normalises the shell habits a user carries in — ``~``, surrounding quotes,
    ``\ `` drag-into-terminal escapes, stray whitespace — BEFORE anything
    opens or checksums the file, so a mistyped path reports the loader's own
    message instead of a raw ``[Errno 2]``. Same core helper the Workbench and
    the service layer use (``meshforge.io.paths``); lazy-imported so ``--help``
    stays cheap.
    """
    from meshforge.io.paths import normalize_path

    return normalize_path(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="meshforge",
        description="meshforge demo edition: one CT of an aortic aneurysm from scan to wall stress and blood flow. Research software, not a medical device.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser(
        "check",
        help="Surface readiness check: open edges, non-manifold edges, "
        "watertight, all-triangles, mesh_ready.",
    )
    p_check.add_argument("surface", type=_path_arg, help="surface file (.stl/.ply/.vtp/...)")
    p_check.add_argument("--outputs", type=_path_arg, default=Path("outputs"))

    p_repair = sub.add_parser(
        "repair",
        help="Repair a surface: merge duplicate vertices, drop degenerate/"
        "duplicate faces, fill small holes; non-manifold edges are reported, "
        "never auto-fixed.",
    )
    p_repair.add_argument("surface", type=_path_arg, help="surface file (.stl/.ply/.vtp/...)")
    p_repair.add_argument(
        "--merge-tol",
        type=float,
        default=None,
        help="absolute vertex merge tolerance (default: 1e-9 x bbox diagonal)",
    )
    p_repair.add_argument(
        "--max-hole-edges",
        type=int,
        default=16,
        help="fill holes up to this many boundary edges (default 16)",
    )
    p_repair.add_argument("--out", type=_path_arg, default=None, help="repaired surface path")
    p_repair.add_argument("--outputs", type=_path_arg, default=Path("outputs"))

    p_vol = sub.add_parser(
        "volmesh",
        help="Volume-mesh the wall region of a tubular surface with swept hexahedra.",
    )
    p_vol.add_argument("surface", type=_path_arg)
    p_vol.add_argument(
        "--thickness", type=float, default=None, help="wall thickness (scalar)"
    )
    p_vol.add_argument(
        "--outer-surface",
        type=_path_arg,
        default=None,
        help="measured outer surface file (mutually exclusive with --thickness)",
    )
    p_vol.add_argument(
        "--backend", default="swept",
        help="swept (hexahedral sweep; the only wall backend in this edition)",
    )
    p_vol.add_argument("--n-theta", type=int, default=32)
    p_vol.add_argument("--n-axial", type=int, default=60)
    p_vol.add_argument("--n-layers", type=int, default=4, help="radial layers (swept only)")
    p_vol.add_argument(
        "--layer-ratio",
        type=float,
        default=1.0,
        help="radial grading ratio, outer/inner layer thickness (swept only)",
    )
    p_vol.add_argument(
        "--curvature-cap",
        type=float,
        default=0.5,
        help="swept only: fraction of the fold radius 1/|kappa| a wall ray may reach on a "
        "concave bend before it is capped (default 0.5; 0.9 for an aneurysm sac)",
    )
    p_vol.add_argument(
        "--sweep",
        choices=("centreline", "planes", "auto"),
        default="centreline",
        help="swept only: station layout — centreline (default: rays square to the "
        "centreline), planes (sections parallel to two flat parallel end caps; for a "
        "short, wide, eccentric segment such as a cropped aneurysm sac), auto "
        "(centreline unless its wall inverts or is capped/floored, then planes)",
    )
    p_vol.add_argument(
        "--side",
        choices=("outward", "inward"),
        default="outward",
        help="planes only: grow the wall outward of the surface (default) or inward "
        "(the surface is the outer wall, e.g. a recovered sac)",
    )
    p_vol.add_argument(
        "--max-tilt",
        type=float,
        default=None,
        help="planes only: largest surface tilt to the station planes (degrees) before "
        "the wall is thinned and counted as capped (default 80)",
    )
    p_vol.add_argument(
        "--section-smoothing",
        type=float,
        default=0.0,
        help="planes only: Gaussian smoothing of each section along its length "
        "(length units; default 0 = exact) for a surface rougher than the wall; the "
        "distance the nodes move off the surface is reported",
    )
    p_vol.add_argument("--out", type=_path_arg, default=None, help="output .vtu path")
    p_vol.add_argument("--outputs", type=_path_arg, default=Path("outputs"))

    p_lumen = sub.add_parser(
        "lumen",
        help="Fill the blood volume (lumen) of a tubular surface: swept "
        "O-grid hexahedra where the geometry permits, automatic gmsh tet "
        "fallback with a loud fold-number note where it does not.",
    )
    p_lumen.add_argument("surface", type=_path_arg)
    p_lumen.add_argument(
        "--method",
        default="auto",
        help="auto (default: O-grid, then tet, then surface-tet on failure) | ogrid | tet | surface-tet",
    )
    p_lumen.add_argument("--n-theta", type=int, default=32, help="rays per station")
    p_lumen.add_argument(
        "--n-axial", type=int, default=40, help="stations along the centerline"
    )
    p_lumen.add_argument(
        "--core-fraction",
        type=float,
        default=0.5,
        help="O-grid core size fraction of centroid-to-rim",
    )
    p_lumen.add_argument(
        "--smooth",
        type=int,
        default=4,
        help="guarded smoothing iterations, sj_min can never regress (0 = off)",
    )
    p_lumen.add_argument(
        "--mesh-size",
        type=float,
        default=None,
        help="tet-fallback target size (default: 1.5x median ring edge)",
    )
    p_lumen.add_argument("--out", type=_path_arg, default=None, help="output .vtu path")
    p_lumen.add_argument("--outputs", type=_path_arg, default=Path("outputs"))

    p_replay = sub.add_parser(
        "replay",
        help="Re-run a saved .mfwork session work file, optionally overriding "
        "step parameters; only steps downstream of an override recompute.",
    )
    p_replay.add_argument("workfile", type=_path_arg, help=".mfwork file (see --save-workfile)")
    p_replay.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="STEP.KNOB=VALUE",
        help="override a recorded step parameter (repeatable)",
    )
    p_replay.add_argument(
        "--save-workfile",
        type=_path_arg,
        default=None,
        help="write the updated journal back to this .mfwork path",
    )

    p_seg = sub.add_parser(
        "segment",
        help="AI segmentation: CT (.nrrd/.nii/.nii.gz) -> TotalSegmentator "
        "aorta mask -> watertight surface (.vtp, RAS millimetres) + mask, "
        "TotalSegmentator log and summary JSON beside it. Exits 1 when the "
        "surface is not mesh_ready.",
    )
    p_seg.add_argument("ct", type=_path_arg, help="CT volume (.nrrd, .nii, .nii.gz)")
    p_seg.add_argument(
        "--out", type=_path_arg, default=None,
        help="surface .vtp path (default: <outputs>/segment/<ct stem>_aorta.vtp)",
    )
    p_seg.add_argument(
        "--fast", action="store_true",
        help="TotalSegmentator's faster 3 mm model",
    )
    p_seg.add_argument(
        "--device", choices=("mps", "cpu"), default=None,
        help="TotalSegmentator device (default: mps)",
    )
    p_seg.add_argument(
        "--smoothing", type=float, default=1.0,
        help="Gaussian blur of the mask in voxels before the surface (0 = off)",
    )
    p_seg.add_argument(
        "--timeout", type=float, default=1800.0,
        help="TotalSegmentator wall-clock limit in seconds",
    )
    p_seg.add_argument(
        "--lumen-out", type=_path_arg, default=None, metavar="PATH",
        help="also write the contrast blood lumen (.vtp): CT >= --lumen-hu inside the AI "
        "aorta mask, largest piece — a CT threshold, not AI",
    )
    p_seg.add_argument(
        "--lumen-hu", type=float, default=150.0,
        help="HU threshold for --lumen-out (default 150)",
    )
    p_seg.add_argument("--outputs", type=_path_arg, default=Path("outputs"))

    p_crop = sub.add_parser(
        "crop",
        help="Crop an aorta surface (mm, +z superior, e.g. from 'segment') to its "
        "infrarenal aneurysm: landmarks from the centreline diameter profile (sac, "
        "neck, bifurcation flare), two cut planes square to the centreline, exactly "
        "flat caps; --lumen cuts a second surface with the same planes. Writes the "
        "cropped .vtp and a <out>_crop.json landmark report. Exits 1 on a refusal "
        "(no sac / no neck).",
    )
    p_crop.add_argument("surface", type=_path_arg, help="closed aorta surface (.vtp/.stl/...)")
    p_crop.add_argument("--out", type=_path_arg, default=None,
                        help="cropped surface .vtp (default: <outputs>/crop/<stem>_crop.vtp)")
    p_crop.add_argument("--lumen", type=_path_arg, default=None,
                        help="second closed surface (e.g. the blood lumen) cut by the same planes")
    p_crop.add_argument("--lumen-out", type=_path_arg, default=None,
                        help="where the cut --lumen is written (default: <out stem>_lumen.vtp)")
    p_crop.add_argument("--neck-extension", type=float, default=25.0,
                        help="proximal cut, mm above the neck start (default 25)")
    p_crop.add_argument("--distal-margin", type=float, default=10.0,
                        help="distal cut, mm above the bifurcation flare top (default 10)")
    p_crop.add_argument("--distal-height", type=float, default=None,
                        help="manual distal cut, mm above the caudal end (with --proximal-height)")
    p_crop.add_argument("--proximal-height", type=float, default=None,
                        help="manual proximal cut, mm above the caudal end")
    p_crop.add_argument("--planes-from", type=_path_arg, default=None, metavar="CROP_JSON",
                        help="reuse the two planes of an earlier crop's JSON report (no "
                        "landmark rules) — e.g. to cut a later outer-sac surface identically")
    p_crop.add_argument("--outputs", type=_path_arg, default=Path("outputs"))

    p_sac = sub.add_parser(
        "sac",
        help="Outer aneurysm sac from a contrast CT and a lumen-only mask: axial "
        "rays from the lumen to the first fat plane -> lumen + thrombus + wall "
        "surface (.vtp, RAS millimetres, point fields ilt_thickness_mm and "
        "sac_resolved) + sac mask (<out>_mask.nii.gz) + summary JSON (<out>.json). "
        "Exits 1 when too few rays find a fat plane or the surface is not mesh_ready.",
    )
    p_sac.add_argument("ct", type=_path_arg, help="CT volume in HU (.nrrd, .nii, .nii.gz)")
    p_sac.add_argument(
        "--lumen-mask", type=_path_arg, required=True,
        help="lumen mask on the CT grid (.nii, .nii.gz, .nrrd), e.g. a TotalSegmentator aorta mask",
    )
    p_sac.add_argument(
        "--out", type=_path_arg, default=None,
        help="surface .vtp path (default: <outputs>/sac/<ct stem>_sac.vtp)",
    )
    p_sac.add_argument("--lumen-hu", type=float, default=150.0,
                       help="contrast threshold: mask voxels at or above it are the lumen")
    p_sac.add_argument("--fat-hu", type=float, default=-30.0,
                       help="fat / air below this HU marks the outer sac edge")
    p_sac.add_argument("--max-thickness", type=float, default=40.0,
                       help="longest thrombus + wall searched beyond the lumen (mm)")
    p_sac.add_argument("--n-rays", type=int, default=72, help="rays per axial slice")
    p_sac.add_argument("--roi-mm", type=float, default=150.0,
                       help="height above the lumen's lower end to process (mm; 0 = whole mask)")
    p_sac.add_argument("--min-resolved", type=float, default=0.5,
                       help="refuse when fewer rays than this fraction find a fat plane")
    p_sac.add_argument("--smoothing", type=float, default=1.0,
                       help="Gaussian blur of the sac mask in voxels before the surface (0 = off)")
    p_sac.add_argument("--outputs", type=_path_arg, default=Path("outputs"))

    p_tix = sub.add_parser(
        "tissue-index",
        help="Strength index band (investigational): peak wall stress / lab-measured wall "
        "strength (tissue library v0: open USP AAA strip tests, CC BY 4.0), "
        "P10/P50/P90 with n shown. Not a diagnosis.",
    )
    pws_src = p_tix.add_mutually_exclusive_group(required=True)
    pws_src.add_argument("--pws", type=float, metavar="MPA",
                         help="peak wall stress in MPa (Cauchy, from the wall-stress solve)")
    pws_src.add_argument("--from-summary", type=_path_arg, metavar="JSON",
                         help="wall-stress summary JSON holding a pws_mpa value (read only)")
    p_tix.add_argument("--selection", choices=("aaa", "control", "all"), default="aaa",
                       help="strength strips: aaa = rupture-border aneurysm wall (default), "
                       "control = non-dilated aorta of the same donors, all = both")
    p_tix.add_argument("--method", choices=("empirical", "lognormal"), default="empirical",
                       help="band: empirical strip quantiles (default) or a lognormal fit")
    p_tix.add_argument("--out", type=_path_arg, default=None,
                       help="also write the band, sensitivities and provenance as JSON here")

    p_cop = sub.add_parser(
        "copilot",
        help="Claude copilot (investigational): a plain-language request runs segment -> sac "
        "-> crop -> wall mesh (quality gate) -> wall stress -> strength index, recovering from a "
        "failed gate; writes a deterministic report with the twin card, the transcript and a "
        "replayable work file. Needs the Anthropic SDK and an API key in the environment.",
    )
    p_cop.add_argument("request", help="what to do, in plain language")
    p_cop.add_argument("--ct", type=_path_arg, default=None,
                       help="the CT file (.nii.gz/.nrrd); appended to the request")
    p_cop.add_argument("--credit", default="public CC-BY scan",
                       help="the scan's source and licence, printed in the report")
    p_cop.add_argument("--server", default=None,
                       help="run the steps on a live Workbench (e.g. http://127.0.0.1:8765) so "
                       "its viewport follows along; default: in-process")
    p_cop.add_argument("--model", default="claude-opus-5-5")
    p_cop.add_argument("--effort", default="high", choices=("low", "medium", "high", "xhigh", "max"))
    p_cop.add_argument("--max-turns", type=int, default=24)
    p_cop.add_argument("--flow-metrics", type=_path_arg, default=None,
                       help="a pre-computed CFD metrics JSON (numbers only) to fill the twin "
                       "card's flow slot; journaled in the session, so the Workbench panel shows "
                       "it too")
    p_cop.add_argument("--out-dir", type=_path_arg, default=None,
                       help="run folder (default: <outputs>/copilot/<timestamp>)")
    p_cop.add_argument("--outputs", type=_path_arg, default=Path("outputs"))
    # D112: one command runs the whole demo from a terminal
    p_cop.add_argument("--launch", dest="launch", action="store_true", default=True,
                       help="when nothing answers at --server, start a Workbench there and open "
                       "its window (the default)")
    p_cop.add_argument("--no-launch", dest="launch", action="store_false",
                       help="never start a Workbench or open a window")
    p_cop.add_argument("--reset", action="store_true",
                       help="empty a Workbench that is already running at --server before the "
                       "run (a fresh take from grey tabs)")
    p_cop.add_argument("--flow-live", action="store_true",
                       help="after the strength index, build an OpenFOAM case from THIS crop's "
                       "blood lumen (the crop step's lumen, tet-filled; 1 L/min at the neck, "
                       "blood viscosity) and let the session's cfd step solve it (about 20 s), "
                       "so the flow sits on the same crop as the wall stress")
    p_cop.add_argument("--flow-case", type=_path_arg, default=None,
                       help="a complete OpenFOAM case of this scan's blood lumen (system/, "
                       "constant/polyMesh, 0/); the session re-solves a copy of it (the cfd "
                       "step) after the strength index — with --flow-live, only when the live "
                       "build fails")
    p_cop.add_argument("--open", dest="open_report", action="store_true", default=None,
                       help="open report.html when the run ends (default: when stdout is a "
                       "terminal)")
    p_cop.add_argument("--no-open", dest="open_report", action="store_false",
                       help="leave the report closed")

    p_aaa = sub.add_parser(
        "aaa-stress",
        help="AAA wall stress under blood pressure (INVESTIGATIONAL): stiff-linear "
        "CalculiX solves on the loaded geometry of a swept hex wall (pressure on the "
        "lumen side, both ends clamped) at a systolic and a what-if pressure; peak "
        "wall stress (99th percentile von Mises outside the end zones), max "
        "principal, wall tension. Refuses (exit 1) a wall with inverted cells, "
        "capped/floored wall rays or no mesh summary. Writes a results .vtu and a "
        "summary JSON.",
    )
    p_aaa.add_argument(
        "wall", type=_path_arg,
        help="swept wall .vtu from 'volmesh' (its <stem>.summary.json beside it), or a "
        ".mfwork session whose replay ends with that wall",
    )
    p_aaa.add_argument("--summary", type=_path_arg, default=None,
                       help="the wall's volmesh summary JSON (default: <wall stem>.summary.json)")
    p_aaa.add_argument("--crop-report", type=_path_arg, default=None,
                       help="crop report JSON (the neck diameter for the end zones, and "
                       "which cap is distal)")
    p_aaa.add_argument("--systolic", type=float, default=140.0,
                       help="systolic pressure, mmHg (default 140)")
    p_aaa.add_argument("--whatif", type=float, default=120.0,
                       help="what-if pressure, mmHg (default 120)")
    p_aaa.add_argument("--E", type=float, default=100.0,
                       help="wall Young's modulus, MPa (default 100)")
    p_aaa.add_argument("--nu", type=float, default=0.49,
                       help="wall Poisson's ratio (default 0.49)")
    p_aaa.add_argument(
        "--end-zone", default="neck",
        help="excluded next to each clamped end: neck (default: one neck diameter "
        "from the crop report, else each end's own diameter) | bending (3 shell "
        "bending lengths) | a length in mm",
    )
    p_aaa.add_argument(
        "--lumen", type=_path_arg, default=None,
        help="the blood-lumen surface cut with the same two planes (e.g. crop --lumen-out): "
        "models the intraluminal thrombus between it and the wall as a tied solid; needs "
        "--crop-report",
    )
    p_aaa.add_argument(
        "--sac", type=_path_arg, default=None,
        help="the surface the wall was built from (the cropped outer sac): the thrombus "
        "fills the space between it and --lumen; required with --lumen",
    )
    p_aaa.add_argument("--E-ilt", type=float, default=5.0,
                       help="thrombus Young's modulus, MPa (default 5, as in the Zenodo AAA "
                       "FE model)")
    p_aaa.add_argument("--ilt-size", type=float, default=3.0,
                       help="thrombus tet size, mm (default 3)")
    p_aaa.add_argument("--ccx", default=None, help="CalculiX executable")
    p_aaa.add_argument("--timeout", type=float, default=600.0,
                       help="solver limit per solve, seconds")
    p_aaa.add_argument("--out-dir", type=_path_arg, default=None,
                       help="results folder (default: <outputs>/aaa_stress/<wall stem>)")
    p_aaa.add_argument("--outputs", type=_path_arg, default=Path("outputs"))

    p_cfd = sub.add_parser(
        "cfd",
        help="Run OpenFOAM on a case directory and read U/p back: with "
        "--inlet/--outlet the dir needs only constant/polyMesh and the "
        "solver files are written from the flags; "
        "without them the case must already be complete. Persists a "
        "results .vtu + summary JSON under outputs/cfd/<case>/.",
    )
    p_cfd.add_argument("case", type=_path_arg, help="OpenFOAM case directory")
    p_cfd.add_argument(
        "--solver", default="simpleFoam", help="simpleFoam (steady) | icoFoam (transient)"
    )
    p_cfd.add_argument("--nu", type=float, default=1e-3, help="kinematic viscosity")
    p_cfd.add_argument(
        "--inlet",
        default=None,
        metavar="NAME=VX,VY,VZ",
        help="velocity inlet patch, e.g. --inlet inlet=0,0,1",
    )
    p_cfd.add_argument(
        "--outlet",
        default=None,
        metavar="NAME=P",
        help="pressure outlet patch (kinematic p), e.g. --outlet outlet=0",
    )
    p_cfd.add_argument("--end-time", type=float, default=2000.0)
    p_cfd.add_argument(
        "--delta-t", type=float, default=None, help="time step (icoFoam only)"
    )
    p_cfd.add_argument("--write-interval", type=float, default=None)
    p_cfd.add_argument("--timeout", type=float, default=600.0)
    p_cfd.add_argument(
        "--turbulence",
        default="laminar",
        choices=("laminar", "kOmegaSST"),
        help="turbulence model for built cases (steady kOmegaSST RANS)",
    )
    p_cfd.add_argument(
        "--intensity",
        type=float,
        default=0.05,
        help="inlet turbulence intensity I (kOmegaSST; default 0.05)",
    )
    p_cfd.add_argument(
        "--turb-length",
        type=float,
        default=None,
        help="inlet mixing length, e.g. 0.07*D for a pipe (REQUIRED with "
        "--turbulence kOmegaSST)",
    )
    p_cfd.add_argument(
        "--moving-wall",
        default=None,
        metavar="NAME=VX,VY,VZ",
        help="tangentially moving wall patch (closed cavities), "
        "e.g. --moving-wall lid=1,0,0",
    )
    p_cfd.add_argument(
        "--empty",
        default=None,
        metavar="NAMES",
        help="comma-separated patches to treat as the OpenFOAM 2-D "
        "'empty' direction, e.g. --empty front,back",
    )
    p_cfd.add_argument(
        "--closed-cavity",
        action="store_true",
        help="sealed domain: no pressure BC anywhere, pressure level "
        "pinned by pRefCell/pRefValue",
    )
    p_cfd.add_argument(
        "--p-ref-cell", type=int, default=0,
        help="reference cell index (--closed-cavity; default 0)",
    )
    p_cfd.add_argument(
        "--p-ref-value", type=float, default=0.0,
        help="reference kinematic pressure (--closed-cavity; default 0)",
    )
    p_cfd.add_argument(
        "--checkmesh",
        action="store_true",
        help="run checkMesh only (no solve); loud unless the mesh is OK",
    )
    p_cfd.add_argument(
        "--series",
        action="store_true",
        help="read ALL written time steps back and persist one .vtu per "
        "frame (combine with --write-interval below --end-time)",
    )
    p_cfd.add_argument("--outputs", type=_path_arg, default=Path("outputs"))

    p_web = sub.add_parser(
        "web",
        help="Serve the local web app (FastAPI over MeshingSession) on "
        "127.0.0.1:<free port> and print the URL. Deps: the 'web' "
        "extra (pip install .[web]).",
    )
    p_web.add_argument(
        "--port",
        type=int,
        default=None,
        help="port to bind (default: an OS-assigned free port)",
    )

    args = parser.parse_args(argv)
    if args.command == "web":
        return _cmd_web(args)
    if args.command == "cfd":
        return _cmd_cfd(args)
    if args.command == "segment":
        return _cmd_segment(args)
    if args.command == "crop":
        return _cmd_crop(args)
    if args.command == "sac":
        return _cmd_sac(args)
    if args.command == "tissue-index":
        return _cmd_tissue_index(args)
    if args.command == "copilot":
        return _cmd_copilot(args)
    if args.command == "aaa-stress":
        return _cmd_aaa_stress(args)
    if args.command == "replay":
        return _cmd_replay(args)
    if args.command == "repair":
        return _cmd_repair(args)
    if args.command == "volmesh":
        return _cmd_volmesh(args)
    if args.command == "lumen":
        return _cmd_lumen(args)
    if args.command == "check":
        return _cmd_check(args)
    parser.error(f"unknown command: {args.command}")
    return 2


def _cmd_web(args: argparse.Namespace) -> int:
    # Lazy: the web stack (FastAPI/uvicorn) is imported only for this command,
    # so importing meshforge.cli never pulls it (rule 5 / D50).
    from meshforge.web.server import serve

    serve(port=args.port)
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    from meshforge.io import file_sha256, load_surface
    from meshforge.quality import check_surface

    surface: Path = args.surface
    try:
        checksum_before = file_sha256(surface)
        mesh = load_surface(surface)
        report = check_surface(mesh, path=str(surface), sha256=checksum_before)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if file_sha256(surface) != checksum_before:
        print(f"error: input file changed on disk during check: {surface}", file=sys.stderr)
        return 1

    outputs: Path = args.outputs
    outputs.mkdir(parents=True, exist_ok=True)
    json_path = outputs / f"{surface.stem}_check.json"
    txt_path = outputs / f"{surface.stem}_check.txt"
    json_path.write_text(report.to_json() + "\n")
    txt_path.write_text(report.to_text())

    print(report.to_text(), end="")
    print(f"report             : {json_path}")
    print(f"report             : {txt_path}")
    return 0


def _parse_param(item: str) -> tuple[str, float | int]:
    """Parse KEY=VALUE into a numeric generator parameter (no eval)."""
    key, sep, raw = item.partition("=")
    if not sep or not key or not raw:
        raise ValueError(f"expected KEY=VALUE, got {item!r}")
    try:
        return key, int(raw)
    except ValueError:
        pass
    try:
        return key, float(raw)
    except ValueError:
        raise ValueError(f"parameter {key!r} must be numeric, got {raw!r}") from None


def _cmd_replay(args: argparse.Namespace) -> int:
    from meshforge.pipeline import load_workfile, replay

    overrides: dict[str, dict] = {}
    try:
        for item in args.overrides:
            key, sep, raw = item.partition("=")
            step, dot, knob = key.partition(".")
            if not sep or not dot:
                raise ValueError(f"--set expects STEP.KNOB=VALUE, got {item!r}")
            _, value = _parse_param(f"{knob}={raw}")
            overrides.setdefault(step, {})[knob] = value
        session = load_workfile(args.workfile)
        results = replay(session, overrides)
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for result in results:
        print(result.text, end="")
    if args.save_workfile is not None:
        written = session.save_workfile(args.save_workfile)
        print(f"workfile    : {written}")
    print(f"replayed    : {len(results)} step(s) from {args.workfile}")
    return 0


def _cmd_repair(args: argparse.Namespace) -> int:
    from meshforge.export import guard_output_path
    from meshforge.io import file_sha256, load_surface
    from meshforge.quality import repair_surface

    src: Path = args.surface
    out: Path = args.out or (args.outputs / f"{src.stem}_repaired.vtp")
    try:
        checksum_before = file_sha256(src)
        mesh = load_surface(src)
        repaired, report = repair_surface(
            mesh, merge_tol=args.merge_tol, max_hole_edges=args.max_hole_edges
        )
        guard_output_path(out, input_paths=[src])
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if file_sha256(src) != checksum_before:
        print(f"error: input changed on disk during repair: {src}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    repaired.save(out)
    json_path = args.outputs / f"{src.stem}_repair.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.to_json() + "\n")

    print(report.to_text(), end="")
    print(f"repaired    : {out}")
    print(f"report      : {json_path}")
    return 0 if report.mesh_ready else 1


def _cmd_segment(args: argparse.Namespace) -> int:
    import json

    from meshforge.export import guard_output_path
    from meshforge.io.segment import DEFAULT_DEVICE, DEFAULT_FAST, segment_ct
    from meshforge.quality.check import check_surface

    src: Path = args.ct
    stem = src.name
    for suffix in (".nii.gz", ".nrrd", ".nii"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    out: Path = args.out or (args.outputs / "segment" / f"{stem}_aorta.vtp")
    if out.suffix.lower() != ".vtp":
        out = out.with_suffix(".vtp")
    base = out.with_suffix("")
    mask_path = base.parent / f"{base.name}_mask.nii.gz"
    log_path = base.parent / f"{base.name}_totalseg.log"
    json_path = base.parent / f"{base.name}_segment.json"
    device = args.device or DEFAULT_DEVICE
    fast = bool(args.fast or DEFAULT_FAST)

    try:
        for target in (out, mask_path, log_path, json_path, args.lumen_out):
            if target is not None:
                guard_output_path(target, input_paths=[src])
        print(
            f"segment     : {src.name} -> TotalSegmentator (task total, roi aorta, "
            f"{'fast' if fast else 'full'}, device {device}) ...",
            flush=True,
        )
        seg = segment_ct(
            src,
            device=device,
            fast=fast,
            smoothing=args.smoothing,
            timeout=args.timeout,
            keep_mask=mask_path,
            lumen_hu=args.lumen_hu if args.lumen_out is not None else None,
            context_rois=(),  # the CLI writes the vessel; the heart ghost is the Workbench's
        )
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    seg.surface.save(out)
    if args.lumen_out is not None:
        args.lumen_out.parent.mkdir(parents=True, exist_ok=True)
        seg.lumen_surface.save(args.lumen_out)
    log_path.write_text(seg.log)
    summary = dict(seg.summary)
    report = check_surface(seg.surface, path=str(src), sha256=seg.input_sha256)
    summary["check"] = report.to_dict()
    ts, mask, surf, ct = (
        summary["totalseg"], summary["mask"], summary["surface"], summary["ct"]
    )
    print(
        f"ct          : {ct['format']} {ct['shape']} voxels, spacing "
        f"{[round(v, 4) for v in ct['spacing_mm']]} mm"
    )
    print(
        f"totalseg    : {ts['version'] or '(version unknown)'}  task total, roi "
        f"{'+'.join(ts['rois'])}, {'fast' if ts['fast'] else 'full'}, device "
        f"{ts['device']}, wall {ts['wall_time_s']:.1f} s"
    )
    print(
        f"mask        : {mask['kept_voxels']} voxels kept ({summary['mask_volume_ml']:.2f} mL), "
        f"{mask['components']} component(s), dropped {mask['dropped_components']} / "
        f"{mask['dropped_voxels']} voxels, smoothing sigma {mask['sigma']:g} vox"
    )
    print(
        f"surface     : {surf['n_points']} pts, {surf['n_triangles']} tris, "
        f"enclosed {surf['enclosed_volume_ml']} mL, frame {surf['frame']}"
    )
    print(f"check       : watertight={report.watertight}  mesh_ready={report.mesh_ready}")
    for note in summary["notes"]:
        print(f"note        : {note}")
    if args.lumen_out is not None:
        lum = summary["lumen"]
        print(
            f"lumen       : {lum['kept_voxels']} voxels >= {lum['lumen_hu']:g} HU kept "
            f"({lum['lumen_volume_ml']:.2f} mL, {100 * lum['fraction_of_mask']:.1f}% of the AI "
            f"mask, Dice vs mask {lum['dice_vs_ai_mask']:.4f}); dropped "
            f"{lum['dropped_components']} piece(s) / {lum['dropped_voxels']} voxels"
        )
        print(f"lumen       : {lum['method']}")
        for note in lum["notes"][1:]:
            print(f"lumen note  : {note}")

    json_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"surface     : {out}")
    print(f"mask        : {mask_path}")
    print(f"log         : {log_path}")
    print(f"summary     : {json_path}")
    if args.lumen_out is not None:
        print(f"lumen       : {args.lumen_out}")
    if not report.mesh_ready:
        print("error: segmented surface is not mesh_ready — see check report",
              file=sys.stderr)
        return 1
    return 0


def _cmd_crop(args: argparse.Namespace) -> int:
    import json

    from meshforge.export import guard_output_path
    from meshforge.geometry.crop import crop_aaa, crop_with_planes, planes_from_report
    from meshforge.io import file_sha256, load_surface
    from meshforge.pipeline import crop_text
    from meshforge.quality.check import check_surface

    src: Path = args.surface
    out: Path = args.out or (args.outputs / "crop" / f"{src.stem}_crop.vtp")
    if out.suffix.lower() != ".vtp":
        out = out.with_suffix(".vtp")
    json_path = out.with_name(f"{out.stem}_crop.json")
    lumen_out = None
    if args.lumen is not None:
        lumen_out = args.lumen_out or out.with_name(f"{out.stem}_lumen.vtp")
    try:
        if args.lumen_out is not None and args.lumen is None:
            raise ValueError("--lumen-out needs --lumen (the surface to cut)")
        for target in (out, json_path, lumen_out):
            if target is not None:
                guard_output_path(target, input_paths=[src, args.lumen, args.planes_from])
        sha = {p: file_sha256(p) for p in (src, args.lumen) if p is not None}
        surface = load_surface(src)
        lumen = load_surface(args.lumen) if args.lumen is not None else None
        print(f"crop        : {src.name}"
              + (f" + {args.lumen.name} (same planes)" if lumen is not None else ""), flush=True)
        if args.planes_from is not None:
            if args.distal_height is not None or args.proximal_height is not None:
                raise ValueError("--planes-from and manual heights are mutually exclusive")
            planes = planes_from_report(json.loads(args.planes_from.read_text()))
            cut, caps, note = crop_with_planes(surface, planes, what="surface")
            rep = {
                "planes": {pl.name: pl.to_dict() for pl in planes},
                "caps": {k: v.to_dict() for k, v in caps.items()},
                "surface": {"n_points": int(cut.n_points), "n_triangles": int(cut.n_cells),
                            "watertight": int(cut.n_open_edges) == 0,
                            "volume_ml": round(float(cut.volume) / 1000.0, 4)},
                "landmarks": None,
                "notes": [f"planes reused from {args.planes_from.name}", note],
            }
            lumen_cut = None
            if lumen is not None:
                lumen_cut, lcaps, lnote = crop_with_planes(lumen, planes, what="lumen")
                rep["lumen"] = {"n_points": int(lumen_cut.n_points),
                                "n_triangles": int(lumen_cut.n_cells),
                                "watertight": int(lumen_cut.n_open_edges) == 0,
                                "volume_ml": round(float(lumen_cut.volume) / 1000.0, 4),
                                "caps": {k: v.to_dict() for k, v in lcaps.items()}}
                rep["notes"].append(lnote)
        else:
            res = crop_aaa(
                surface,
                neck_extension_mm=args.neck_extension,
                distal_margin_mm=args.distal_margin,
                distal_height_mm=args.distal_height,
                proximal_height_mm=args.proximal_height,
                lumen=lumen,
            )
            cut, lumen_cut, rep = res.surface, res.lumen, res.report()
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for p, h in sha.items():
        if file_sha256(p) != h:
            print(f"error: input changed on disk during the crop: {p}", file=sys.stderr)
            return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    cut.save(out)
    if lumen_cut is not None:
        lumen_cut.save(lumen_out)
    report = check_surface(cut, path=str(src), sha256=sha[src])
    rep["check"] = report.to_dict()
    json_path.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(crop_text(rep), end="")
    print(f"check       : watertight={report.watertight}  mesh_ready={report.mesh_ready}")
    print(f"surface     : {out}")
    if lumen_cut is not None:
        print(f"lumen       : {lumen_out}")
    print(f"report      : {json_path}")
    if not report.mesh_ready:
        print("error: the cropped surface is not mesh_ready", file=sys.stderr)
        return 1
    return 0


def _cmd_sac(args: argparse.Namespace) -> int:
    import json

    from meshforge.export import guard_output_path
    from meshforge.geometry.sac import recover_sac_from_files, write_sac_mask
    from meshforge.pipeline import sac_text
    from meshforge.quality.check import check_surface

    src: Path = args.ct
    stem = src.name
    for suffix in (".nii.gz", ".nrrd", ".nii"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    out: Path = args.out or (args.outputs / "sac" / f"{stem}_sac.vtp")
    if out.suffix.lower() != ".vtp":
        out = out.with_suffix(".vtp")
    base = out.with_suffix("")
    mask_path = base.parent / f"{base.name}_mask.nii.gz"
    json_path = base.parent / f"{base.name}.json"
    try:
        for target in (out, mask_path, json_path):
            guard_output_path(target, input_paths=[src, args.lumen_mask])
        print(f"sac         : {src.name} + lumen {args.lumen_mask.name} -> outer sac ...",
              flush=True)
        run = recover_sac_from_files(
            src,
            args.lumen_mask,
            lumen_hu=args.lumen_hu,
            fat_hu=args.fat_hu,
            max_thickness_mm=args.max_thickness,
            n_rays=args.n_rays,
            roi_mm=args.roi_mm,
            min_resolved=args.min_resolved,
            smoothing=args.smoothing,
        )
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    run.surface.save(out)
    write_sac_mask(run.result, mask_path)
    summary = dict(run.summary)
    report = check_surface(run.surface, path=str(src), sha256=run.input_sha256["ct"])
    summary["check"] = report.to_dict()
    surf = summary["surface"]
    print(sac_text(summary), end="")
    print(
        f"surface     : {surf['n_points']} pts, {surf['n_triangles']} tris, "
        f"enclosed {surf['enclosed_volume_ml']} mL, point data "
        f"{'+'.join(surf['point_data'])}, frame {surf['frame']}"
    )
    print(f"check       : watertight={report.watertight}  mesh_ready={report.mesh_ready}")

    json_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"surface     : {out}")
    print(f"sac mask    : {mask_path}")
    print(f"summary     : {json_path}")
    if not report.mesh_ready:
        print("error: sac surface is not mesh_ready — see check report", file=sys.stderr)
        return 1
    return 0


def _cmd_tissue_index(args: argparse.Namespace) -> int:
    import json

    from meshforge.tissue.strength import (
        StrengthLibraryError,
        index_band_report,
        index_band_text,
        pws_from_summary,
    )

    try:
        if args.from_summary is not None:
            pws, where, digest = pws_from_summary(args.from_summary)
            pws_source = f"{args.from_summary.name}: {where}"
        else:
            pws, pws_source, digest = args.pws, "entered", None
        if args.out is not None and args.from_summary is not None and (
            args.out.resolve() == args.from_summary.resolve()
        ):
            raise ValueError(f"refusing to overwrite input file: {args.from_summary}")
        report = index_band_report(pws, selection=args.selection, method=args.method)
    except (ValueError, FileNotFoundError, StrengthLibraryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    report["pws_source"] = pws_source
    if digest is not None:
        report["pws_summary_sha256"] = digest
    print(index_band_text(report, pws_source=pws_source), end="")
    if args.out is not None:
        out = args.out if args.out.suffix.lower() == ".json" else args.out.with_suffix(".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"summary            : {out}")
    return 0


def _cmd_cfd(args: argparse.Namespace) -> int:
    import json

    from meshforge.export import guard_output_path
    from meshforge.pipeline import _parse_inlet_spec, _parse_outlet_spec
    from meshforge.solve import CfdBC, CfdSpec, check_mesh, solve_case, write_case_files

    case: Path = args.case
    try:
        if args.checkmesh:
            report = check_mesh(case, timeout=args.timeout)
            print(f"checkMesh   : Mesh OK  {report}")
            return 0

        builds_case = bool(
            args.inlet or args.outlet or args.moving_wall or args.empty
        )
        if not builds_case and args.turbulence != "laminar":
            raise ValueError(
                "--turbulence applies only when the solver files are built "
                "here (--inlet/--outlet/--moving-wall); a complete case "
                "already carries its turbulence setup"
            )
        if not builds_case and args.closed_cavity:
            raise ValueError(
                "--closed-cavity applies only when the solver files are "
                "built here; a complete case already carries its fvSolution"
            )
        if builds_case:
            bcs: dict[str, CfdBC] = {}
            if args.inlet:
                _, sep, raw = args.inlet.partition("=")
                if sep and len(raw.split(",")) == 1:
                    raise ValueError(
                        "--inlet NAME=SPEED needs the mesh in memory to find "
                        "the patch's inward normal, and this command only "
                        "has a case directory: give NAME=VX,VY,VZ here, or "
                        "use the speed form in the Workbench or a work file "
                        "run with 'meshforge replay'"
                    )
                name, velocity = _parse_inlet_spec(args.inlet)
                bcs[name] = CfdBC("velocity", velocity)
            if args.outlet:
                name, p_value = _parse_outlet_spec(args.outlet)
                bcs[name] = CfdBC("pressure", p_value)
            if args.moving_wall:
                name, velocity = _parse_inlet_spec(args.moving_wall)
                bcs[name] = CfdBC("moving_wall", velocity)
            for patch in filter(
                None, (p.strip() for p in (args.empty or "").split(","))
            ):
                bcs[patch] = CfdBC("empty")
            spec = CfdSpec(
                solver=args.solver,
                nu=args.nu,
                bcs=bcs,
                end_time=args.end_time,
                delta_t=args.delta_t,
                write_interval=args.write_interval,
                turbulence=args.turbulence,
                turbulence_intensity=args.intensity,
                turbulence_length=args.turb_length,
                closed_cavity=args.closed_cavity,
                p_ref_cell=args.p_ref_cell,
                p_ref_value=args.p_ref_value,
            )
            patch_kinds = write_case_files(case, spec)
            print(f"case        : {case}  patches {patch_kinds}")
        result = solve_case(case, timeout=args.timeout, series=args.series)

        out_dir: Path = args.outputs / "cfd" / case.name
        vtu = out_dir / f"{case.name}_results.vtu"
        guard_output_path(vtu, input_paths=[])
        frame_paths = []
        if result.frames is not None:
            # Index-named (time-named files would depend on float
            # formatting); the JSON summary maps index -> time.
            frame_paths = [
                out_dir / f"{case.name}_t{i:04d}.vtu"
                for i in range(len(result.frames))
            ]
            for path in frame_paths:
                guard_output_path(path, input_paths=[])
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    result.grid.save(vtu)
    for path, frame in zip(frame_paths, result.frames or []):
        frame.save(path)
    summary = result.summary()
    summary["results_vtu"] = str(vtu)
    if frame_paths:
        summary["frame_vtus"] = [str(p) for p in frame_paths]
    summary_path = out_dir / f"{case.name}_cfd.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(result.to_text(), end="")
    print(f"results     : {vtu}")
    if frame_paths:
        print(f"frames      : {frame_paths[0].name} .. {frame_paths[-1].name}")
    print(f"summary     : {summary_path}")
    return 0


def _cmd_copilot(args: argparse.Namespace) -> int:
    import json
    import time as _time

    from meshforge.copilot import HttpBackend, SessionBackend, render_report, run_copilot
    from meshforge.copilot.report import shown_path, shown_paths
    from meshforge.copilot.tools import PipelineTools
    from meshforge.pipeline import check_flow_metrics

    flow = None
    if args.flow_metrics is not None:  # checked before any API call is spent
        try:
            flow = check_flow_metrics(json.loads(args.flow_metrics.read_text()))
        except (OSError, ValueError) as exc:
            # an OSError's own text repeats the full path; its reason does not
            why = (exc.strerror or type(exc).__name__) if isinstance(exc, OSError) else exc
            print(f"error: --flow-metrics {args.flow_metrics.name}: {why}", file=sys.stderr)
            return 1
    flow_case: Path | None = args.flow_case
    if flow_case is not None and not (flow_case / "system" / "controlDict").is_file():
        print(f"error: --flow-case {flow_case.name}: not an OpenFOAM case (no system/controlDict)",
              file=sys.stderr)
        return 1
    out_dir: Path = args.out_dir or (args.outputs / "copilot" / _time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    request = args.request + (f"\nCT: {args.ct}" if args.ct is not None else "")
    # D112: the Workbench this run drives — already up (emptied on --reset), or
    # started here with its window, so one terminal command runs the demo.
    workbench = None
    if args.server:
        from meshforge.copilot import launch

        if launch.health(args.server):
            if args.reset:
                try:
                    launch.reset_workbench(args.server)
                except RuntimeError as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    return 1
                print(f"workbench   : running at {args.server}; session emptied for this run",
                      flush=True)
            else:
                print(f"workbench   : running at {args.server}", flush=True)
        elif args.launch:
            try:
                workbench, secs = launch.start_workbench(args.server, out_dir / "workbench.log")
            except (RuntimeError, ValueError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            where = launch.open_window(args.server)
            print(f"workbench   : started at {args.server} in {secs:.1f} s (pid {workbench.pid}, "
                  f"log {shown_path(out_dir / 'workbench.log')}); window opened in the {where}",
                  flush=True)
    backend = HttpBackend(args.server) if args.server else SessionBackend()
    tools = PipelineTools(backend, out_dir / "work")
    print(f"copilot     : {args.model} (effort {args.effort}) on the {backend.name}", flush=True)
    t0 = _time.perf_counter()
    try:
        run = run_copilot(request, tools=tools, model=args.model, effort=args.effort,
                          max_turns=args.max_turns,
                          log=lambda m: print(m, flush=True))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if workbench is not None:
            print(f"workbench   : still running at {args.server} (pid {workbench.pid}; stop with: "
                  f"kill {workbench.pid})")
        return 1
    flow_error = ""
    flow_note = ""
    if run.complete and (args.flow_live or flow_case is not None):
        # D112: the blood flow in the SAME session as the stress, journaled as
        # `cfd` before the metrics below so the twin reads both. --flow-live
        # builds the case from this crop's own lumen; --flow-case re-solves a
        # given case (the fallback when the live build fails, or alone).
        from meshforge.copilot import launch
        from meshforge.copilot.backend import StepOutcome

        def solve(case_dir: Path, what: str) -> tuple[StepOutcome, float]:
            t1 = _time.perf_counter()
            try:
                cfd = backend.run("cfd", {"case": str(case_dir)})
            except RuntimeError as exc:  # the Workbench is gone
                cfd = StepOutcome(False, {}, "", str(exc))
            secs = _time.perf_counter() - t1
            if cfd.ok:
                solver = cfd.summary.get("solver") or "simpleFoam"
                cells = cfd.summary.get("n_cells")
                return cfd, secs, (f"{what}: OpenFOAM {solver}"
                                   + (f", {cells} cells" if cells else "") + f", {secs:.0f} s")
            return cfd, secs, ""

        cfd = None
        if args.flow_live:
            from meshforge.copilot import flow as liveflow

            try:
                built = liveflow.build_lumen_case(tools.crop_lumen, out_dir / "work" / "flow_case_live")
                print(f"flow        : this crop's blood channel filled: {built['n_cells']} cells, "
                      f"inflow at the neck {built['inflow_speed_mm_s']} mm/s for {built['q_lmin']:g} L/min",
                      flush=True)
                cfd, _secs, flow_note = solve(Path(built["case"]), "this crop's blood channel solved live")
                if not cfd.ok:
                    flow_error = f"the live flow case did not run: {cfd.error}"
            except (OSError, ValueError, RuntimeError) as exc:
                flow_error = f"the live flow case could not be built: {exc}"
            if flow_error:
                print(f"error: {flow_error}", file=sys.stderr)
        if (cfd is None or not cfd.ok) and flow_case is not None:
            try:
                staged = launch.stage_case(flow_case, out_dir / "work" / "flow_case")
                cfd, _secs, flow_note = solve(staged, "the scan's pre-built lumen case re-solved live")
                if cfd.ok:
                    flow_error = ""  # the fallback carried the flow
                else:
                    flow_error = f"the flow case did not run: {cfd.error}"
            except (OSError, ValueError, RuntimeError) as exc:
                flow_error = f"the flow case could not be staged: {exc}"
            if flow_error:
                print(f"error: {flow_error}", file=sys.stderr)
        if flow_note:
            print(f"flow        : {flow_note}", flush=True)
    if flow is not None:
        # the session journals the metrics as its own flow_metrics record
        # BEFORE the report is written (and before session.mfwork is saved),
        # so the Workbench panel and the report print the same flow lines
        try:
            attached = backend.attach_flow_metrics(flow)
            attach_error = "" if attached.ok else attached.error
        except RuntimeError as exc:  # the Workbench is gone
            attach_error = str(exc)
        if attach_error:
            flow_error = "; ".join(filter(None, (flow_error, attach_error)))
            print(f"error: the flow metrics are not in the session (its twin panel will not "
                  f"show them): {attach_error}", file=sys.stderr)
        else:
            print(f"flow        : {len(flow)} pre-computed CFD metrics journaled in the session",
                  flush=True)
    replay = ""
    if isinstance(backend, SessionBackend):
        try:
            wf = backend.session.save_workfile(out_dir / "session.mfwork")
            replay = f"meshforge replay {shown_path(wf)}"  # no local folder on screen
        except ValueError as exc:  # an empty journal (no step landed): nothing to replay
            print(f"replay      : none ({exc})", flush=True)
    md, page = render_report(
        request=args.request, ct_name=args.ct.name if args.ct else "(in the request)",
        credit=args.credit, results=run.results, calls=run.calls,
        summary_text=run.summary_text, guard_note=run.guard_note, model=args.model,
        flow=flow, replay_hint=replay, flow_note=flow_note,
    )
    (out_dir / "report.md").write_text(md)
    (out_dir / "report.html").write_text(page)
    # the transcript names the CT (and the run folder) without the local folder
    # around them; replay reads session.mfwork, which keeps the real paths
    record = json.loads(json.dumps(
        {"request": request, "model": args.model, "stop": run.stop, "usage": run.usage,
         "guard": run.guard_note, "calls": run.calls, "transcript": run.transcript}, default=str))
    (out_dir / "transcript.json").write_text(
        json.dumps(shown_paths(record, [args.ct, out_dir]), indent=1) + "\n")
    print(f"stop        : {run.stop}; {run.guard_note}")
    print(f"tokens      : {run.usage.get('input_tokens', 0)} in, "
          f"{run.usage.get('output_tokens', 0)} out; {_time.perf_counter() - t0:.0f} s")
    if run.summary_text:
        print("summary     :\n" + run.summary_text)
    print(f"report      : {shown_path(out_dir / 'report.html')}")
    print(f"transcript  : {shown_path(out_dir / 'transcript.json')}")
    if replay:
        print(f"replay      : {replay}")
    open_report = args.open_report if args.open_report is not None else sys.stdout.isatty()
    if open_report:
        from meshforge.copilot import launch

        where = launch.open_file(out_dir / "report.html")
        print(f"opened      : report.html in the {where}")
    if workbench is not None:
        print(f"workbench   : still running at {args.server} (pid {workbench.pid}; stop with: "
              f"kill {workbench.pid})")
    return 0 if run.complete and not flow_error else 1


def _cmd_aaa_stress(args: argparse.Namespace) -> int:
    import json

    from meshforge.export import guard_output_path
    from meshforge.io import file_sha256
    from meshforge.solve.aaa_metrics import (
        crop_report_from_journal,
        read_wall,
        run_aaa_stress,
    )

    src: Path = args.wall
    stem = src.stem
    out_dir: Path = args.out_dir or (args.outputs / "aaa_stress" / stem)
    vtu = out_dir / f"{stem}_aaa_stress.vtu"
    json_path = out_dir / f"{stem}_aaa_stress.json"
    try:
        if not src.is_file():
            raise FileNotFoundError(f"wall input not found: {src}")
        inputs = [src, args.crop_report, args.lumen, args.sac]
        if src.suffix.lower() == ".mfwork":
            if args.summary is not None:
                raise ValueError("--summary applies to a wall .vtu, not a .mfwork session")
            from meshforge.pipeline import load_workfile, replay

            session = load_workfile(src)
            print(f"replay      : {src.name} ({len(session.journal)} steps)", flush=True)
            replay(session)
            wall = session._require(session.volmesh, "volmesh", "a volmesh step in the session")
            summary = wall.summary()
            crop = crop_report_from_journal(session.journal)
        else:
            summary_path = args.summary or src.with_suffix(".summary.json")
            inputs.append(summary_path)
            wall, summary = read_wall(src, summary_path)
            crop = None
        if args.crop_report is not None:
            crop = json.loads(args.crop_report.read_text())
        for target in (vtu, json_path):
            guard_output_path(target, input_paths=inputs)
        sha = {p: file_sha256(p) for p in inputs if p is not None and Path(p).is_file()}
        lumen = sac = None
        if args.lumen is not None or args.sac is not None:
            from meshforge.io import load_surface

            lumen = load_surface(args.lumen) if args.lumen is not None else None
            sac = load_surface(args.sac) if args.sac is not None else None
        print(f"aaa-stress  : {src.name}, {args.systolic:g} and {args.whatif:g} mmHg"
              + (f", thrombus between {args.lumen.name} and the wall" if lumen is not None else ""),
              flush=True)
        result = run_aaa_stress(
            wall, summary, out_dir=out_dir, stem=stem, crop_report=crop,
            systolic_mmhg=args.systolic, whatif_mmhg=args.whatif, E=args.E, nu=args.nu,
            end_zone=args.end_zone, ccx=args.ccx, timeout=args.timeout, input_path=src,
            lumen=lumen, sac=sac, E_ilt=args.E_ilt, ilt_size=args.ilt_size,
        )
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for p, h in sha.items():
        if file_sha256(p) != h:
            print(f"error: input changed on disk during the run: {p}", file=sys.stderr)
            return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result.summary, indent=2) + "\n")
    print(result.to_text(), end="")
    if result.grid is not None:
        result.grid.save(vtu)
        print(f"results     : {vtu}")
    print(f"summary     : {json_path}")
    if result.refused:
        print("error: no wall-stress metric reported (REFUSED, see above)", file=sys.stderr)
        return 1
    return 0


def _cmd_volmesh(args: argparse.Namespace) -> int:
    import json

    from meshforge.export import guard_output_path
    from meshforge.io import file_sha256, load_surface
    from meshforge.meshing import get_backend

    surface_path: Path = args.surface
    out: Path = args.out or (args.outputs / f"{surface_path.stem}_wall_{args.backend}.vtu")

    try:
        # Trap #5 precedence: exactly one thickness source, loudly.
        if args.thickness is not None and args.outer_surface is not None:
            raise ValueError(
                "--thickness and --outer-surface are mutually exclusive; "
                "drop --thickness to use the measured outer surface"
            )
        if args.thickness is None and args.outer_surface is None:
            raise ValueError("give either --thickness or --outer-surface")

        checksum_before = file_sha256(surface_path)
        mesh = load_surface(surface_path)
        if args.outer_surface is not None:
            from meshforge.geometry import ThicknessSpec

            t_arg = ThicknessSpec.from_outer_surface(load_surface(args.outer_surface))
        else:
            t_arg = args.thickness
        backend = get_backend(args.backend)
        kwargs: dict = {
            "n_theta": args.n_theta,
            "n_axial": args.n_axial,
        }
        if args.backend == "swept":
            kwargs["n_layers"] = args.n_layers
            kwargs["layer_ratio"] = args.layer_ratio
            kwargs["curvature_cap"] = args.curvature_cap
            kwargs["sweep"] = args.sweep
            kwargs["side"] = args.side
            kwargs["max_tilt"] = args.max_tilt
            kwargs["section_smoothing"] = args.section_smoothing
        elif (args.sweep != "centreline" or args.side != "outward"
              or args.max_tilt is not None or args.section_smoothing):
            raise ValueError("--sweep/--side/--max-tilt/--section-smoothing are swept-only")
        wall = backend.mesh_wall(mesh, t_arg, **kwargs)
        guard_output_path(out, input_paths=[surface_path, args.outer_surface])
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if file_sha256(surface_path) != checksum_before:
        print(f"error: input changed on disk during meshing: {surface_path}", file=sys.stderr)
        return 1

    from meshforge.solve.aaa_metrics import annotate_wall_thickness

    annotate_wall_thickness(wall)  # D93: the thickness along the surface normal
    out.parent.mkdir(parents=True, exist_ok=True)
    wall.grid.save(out)
    summary = wall.summary()
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    q = summary["quality"]
    print(f"backend     : {summary['backend']}")
    print(f"cells       : {summary['n_cells']}  points: {summary['n_points']}")
    print(f"boundary    : {summary['boundary_faces']}")
    print(
        f"quality     : inverted={q['n_inverted']}  "
        f"sj_min={q['sj_min']:.4g}  sj_mean={q['sj_mean']:.4g}  "
        f"volume={q['total_volume']:.6g}"
    )
    if "lattice" in summary:
        lat = summary["lattice"]
        print(
            f"wall rays   : capped={lat['n_capped']}  floored={lat['n_floored']}  of "
            f"{lat['n_rays']} (curvature_cap {lat['curvature_cap']:g}); thickness "
            f"{lat['thickness_min']:.4g}..{lat['thickness_max']:.4g}"
        )
        if lat.get("thickness_normal_median") is not None:
            print(f"wall normal : median thickness {lat['thickness_normal_median']:.4g} along the "
                  f"surface normal")
        if lat.get("sweep") == "planes":
            print(
                f"plane sweep : side={lat['side']}  capped_tilt={lat['n_capped_tilt']}  "
                f"capped_concave={lat['n_capped_concave']}  tilt_max={lat['tilt_max_deg']:.1f} "
                f"deg (bound {lat['max_tilt_deg']:g})  section_smoothing="
                f"{lat['section_smoothing']:g}  nodes_off_surface max="
                f"{lat['surface_offset_max']:.3g} p95={lat['surface_offset_p95']:.3g}"
            )
    for note in summary["notes"]:
        print(f"note        : {note}")
    print(f"mesh        : {out}")
    print(f"summary     : {summary_path}")
    return 0 if q["n_inverted"] == 0 else 1


def _cmd_lumen(args: argparse.Namespace) -> int:
    import json

    from meshforge.export import guard_output_path
    from meshforge.io import file_sha256, load_surface
    from meshforge.meshing.lumen import fill_lumen

    surface_path: Path = args.surface
    out: Path = args.out or (args.outputs / f"{surface_path.stem}_lumen.vtu")

    try:
        checksum_before = file_sha256(surface_path)
        mesh = load_surface(surface_path)
        lumen = fill_lumen(
            mesh,
            method=args.method,
            n_theta=args.n_theta,
            n_axial=args.n_axial,
            core_fraction=args.core_fraction,
            smooth=args.smooth,
            mesh_size=args.mesh_size,
        )
        guard_output_path(out, input_paths=[surface_path])
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if file_sha256(surface_path) != checksum_before:
        print(f"error: input changed on disk during meshing: {surface_path}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    lumen.grid.save(out)
    summary = lumen.summary()
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    q = summary["quality"]
    print(f"backend     : {summary['backend']}")
    print(f"cells       : {summary['n_cells']}  points: {summary['n_points']}")
    print(f"boundary    : {summary['boundary_faces']}")
    print(
        f"quality     : inverted={q['n_inverted']}  "
        f"sj_min={q['sj_min']:.4g}  sj_mean={q['sj_mean']:.4g}  "
        f"volume={q['total_volume']:.6g}"
    )
    for note in summary["notes"]:
        print(f"note        : {note}")
    print(f"mesh        : {out}")
    print(f"summary     : {summary_path}")
    return 0 if q["n_inverted"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
