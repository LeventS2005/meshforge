# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] — 2026-11-16 — demo edition

This repository now holds one thing: the demo in which an AI copilot takes a
CT of an aortic aneurysm from scan to wall stress and blood flow. The general
meshing toolkit of 1.x is no longer part of it; it remains available at the
tag [`v1.0.0`](https://github.com/LeventS2005/meshforge/tree/v1.0.0).
Research software: not a medical device, not for diagnosis or treatment.

### Added

- **Load scan.** `segment`: TotalSegmentator, installed in its own
  environment and run as a separate program, outlines the aorta on a CT
  (`.nii.gz` / `.nrrd`); the contrast blood lumen is a threshold inside that
  outline. `sac`: the outer aneurysm sac (lumen + thrombus + wall) recovered
  from the CT around the lumen.
- **Crop aneurysm.** `crop`: the neck, the sac and the bifurcation found on
  the diameter profile, two flat caps, the lumen cut with the same planes,
  and parallel caps on request.
- **Mesh wall.** A plane-swept hexahedral wall (`volmesh --sweep planes` or
  `auto`) beside the centreline sweep, and a quality gate: no inverted cell,
  no capped or floored wall ray, and the thickness measured along the surface
  normal within 5% of the 1.5 mm asked for. A wall that fails gives no stress.
- **Wall stress.** `aaa-stress`: CalculiX on the imaged geometry at a
  systolic and a what-if blood pressure, with the intraluminal thrombus
  modelled as a tied solid. The peak wall stress is the 99th percentile
  outside the clamped end zones. Investigational.
- **Strength index.** `tissue-index`: the peak wall stress divided by
  lab-measured wall strength (open strip-test data, CC BY 4.0), as a
  P10 / P50 / P90 band with the number of aneurysms and strips shown.
  Investigational, not a diagnosis.
- **Blood flow.** A steady laminar OpenFOAM case built from the crop's own
  blood lumen (`meshforge copilot --flow-live`) and solved by the `cfd` step.
- **The copilot.** `meshforge copilot`: a Claude model plans the steps, calls
  five tools and recovers when the wall fails its gate. The report is built
  from the tool outputs. The model's own summary is shown only when a guard
  finds every number in it in a tool output.
- **The demo page.** The Workbench shows the six steps as tabs with their
  state, results that sweep in from the top, a slow turntable, a wall stress /
  blood flow switch, a patient atlas and a twin card.
- `scripts/demo_run.py`: the same six steps through the copilot's own tools,
  without a language model or an API key.
- `docs/DEMO_GUIDE.md`.

### Changed

- The Workbench is always the demo layout. The Demo switch, the source box,
  the Quick start, the history table and the native file pickers are gone.
- The command line is `copilot`, `web`, and the single-step commands
  `segment`, `sac`, `crop`, `volmesh`, `lumen`, `repair`, `aaa-stress`,
  `tissue-index`, `cfd`, `replay` and `check`.
- The extras are `meshing`, `image`, `copilot`, `web`, `viewport` and `dev`.

### Removed

- The general meshing toolkit of 1.x: CAD import (DWG/DXF, STEP/IGES/BREP),
  image-volume and contour import, multi-block meshing with projection,
  smoothing and grading, boundary layers, the tetrahedral wall backend,
  branching trees, the generic CalculiX deck builder and solve step, the
  MYSTRAN cross-check, the NASTRAN, FLUENT, LS-DYNA and ANSYS writers,
  recipes, the desktop shell and the installers, with their commands, tests
  and documents.

### Fixed

- `GET /geometry` sends `view_dir` again, so a CFD result with a flow section
  opens face-on to it.

## [1.0.0] — 2026-07-24

Initial public release — commit `25a23f2`, "meshforge v1.0 — initial public
release". Version numbering starts here; development before this point was
tracked internally as milestones V1–V47.

The published package metadata lagged this release at `0.1.0` until the
2026-08-25 cleanup brought every source of the version into agreement; the
release itself is this commit.

### Added

- **Pipeline** — CAD/surface/image/CSO ingest → centreline → sections → wall →
  volume mesh, with a session journal, `.mfwork` work files and selective
  replay.
- **Meshing** — multi-block decomposition, projection onto curved profiles,
  O-grid (butterfly) disk layout, elliptic (Winslow) smoothing with a
  do-no-harm quality guard, transition grading, boundary-layer prisms, and a
  swept-hex lumen filler with an automatic tet fallback.
- **Solver decks** — Abaqus/CalculiX `.inp` (statics, modal, buckling, heat,
  transient heat, film convection, thermal stress), NASTRAN SOL 101, OpenFOAM
  `constant/polyMesh`, FLUENT `.msh`. LS-DYNA `.k` and ANSYS `.cdb` are
  **format-level only** — no solver has validated them, and the emitted deck
  headers say so.
- **In-app solve** — CalculiX and OpenFOAM runners with results read back onto
  the session mesh; MYSTRAN as an independent cross-check.
- **Interfaces** — a scriptable CLI, YAML recipes with parameter sweeps, a local
  127.0.0.1 web Workbench with an interactive vtk.js viewport, and a native
  desktop shell over it.
- **Validation** — analytic and published-benchmark ground truth throughout:
  Lamé thick cylinder, Euler buckling, rod spectrum, log T(r), Hagen–Poiseuille,
  Blasius friction, and Ghia et al. (1982) lid-driven cavity.

[2.0.0]: https://github.com/LeventS2005/meshforge/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/LeventS2005/meshforge/releases/tag/v1.0.0
