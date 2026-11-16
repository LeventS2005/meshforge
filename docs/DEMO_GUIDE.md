# meshforge demo guide

Research software: not a medical device, not for diagnosis or treatment.

## 1. What the demo does

An AI copilot takes one routine CT of an abdominal aortic aneurysm from scan
to physics on a laptop. It runs six steps and shows each one as a tab of a
browser app, the Workbench. The language model plans the steps, reads their
results and writes a short summary. It computes no number itself: every
number comes from a tool (TotalSegmentator, the mesher, CalculiX, OpenFOAM),
and a guard checks every number in the summary against the tool outputs.

| Tab | What runs | What you see |
|---|---|---|
| 1 Load scan | `segment` (TotalSegmentator outlines the aorta), then `sac` | the aorta outline, then the outer aneurysm sac |
| 2 Crop aneurysm | `crop` | the aneurysm between two flat caps, neck to just above the bifurcation |
| 3 Mesh wall | `volmesh` (1.5 mm hexahedral wall) and its quality gate | the wall; the tab shows a failed gate in red |
| 4 Wall stress · FEA | `aaa_stress` (CalculiX) at two blood pressures | the wall coloured by stress |
| 5 Strength index | `tissue_index` | the index band in the twin card |
| 6 Blood flow · CFD | `cfd` (OpenFOAM) on the cropped blood lumen | the flow result, with a layer switch to the stress |

## 2. What you need

- meshforge installed as the README describes.
- Three external programs. meshforge looks for each in this order:

| Tool | Used by | Lookup order |
|---|---|---|
| TotalSegmentator | Load scan | `MESHFORGE_TOTALSEG` (path to the executable), then `TotalSegmentator` on `PATH`, then `~/miniconda3/envs/totalseg/bin/TotalSegmentator` |
| CalculiX (`ccx`) | Wall stress | `MESHFORGE_CCX`, then `ccx` on `PATH`, then `~/miniconda3/envs/ccxsolve/bin/ccx` |
| OpenFOAM | Blood flow | `MESHFORGE_OPENFOAM` (the launcher), then `simpleFoam` on `PATH` (a sourced environment), then an `openfoam` / `openfoam2506` / `openfoam2512` / `openfoam2606` launcher on `PATH` |

Install TotalSegmentator in its own environment. It brings PyTorch, which
meshforge itself never imports:

```bash
conda create -n totalseg --solver=libmamba python=3.11
conda run -n totalseg pip install TotalSegmentator
```

The first run downloads the model weights. meshforge never changes
TotalSegmentator's own settings.

CalculiX: `conda create -n ccxsolve -c conda-forge calculix`. OpenFOAM on a
Mac: `brew install gerlero/openfoam/openfoam@2506`.

- The demo scan: `ct.nii.gz`, attached to the GitHub release `v2.0.0`.
  Its SHA-256 is
  `7957f71d5a673486f0d23b403ece22233fdf1a5ac02a2ee4649fb99006f856dd`.
  Check it with `shasum -a 256 ct.nii.gz`.

## 3. Run it with the copilot

The copilot needs the Anthropic SDK (`pip install -e ".[copilot]"` in the clone, or
`anthropic` from conda-forge) and an API key in your own shell. Every run
calls the API, so it costs money.

```bash
read -s ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY
meshforge copilot "Assess this aneurysm at 140 and 120 mmHg" \
    --ct ct.nii.gz --server http://127.0.0.1:8765 --flow-live
```

What happens:

1. If no Workbench answers at that address, the copilot starts one on
   127.0.0.1 and that port, and opens its window (a Chrome app window on a
   Mac with Chrome, else the default browser). The Workbench keeps running
   after the run; the last line prints how to stop it.
2. The copilot calls its five tools in order: `segment_aorta`,
   `crop_aneurysm`, `mesh_wall`, `wall_stress`, `strength_index`. Each tool
   runs one or two Workbench steps, so the tabs fill as the run goes.
3. With `--flow-live`, it then builds a flow case from this crop's own blood
   lumen and the Workbench's `cfd` step solves it.
4. It writes the report and opens it.

Without a pressure in the request, it uses 140 mmHg with a 120 mmHg what-if.

Useful flags (all in `meshforge copilot --help`):

| Flag | What it does |
|---|---|
| `--server URL` | run the steps on a Workbench at that address; the copilot starts one only on 127.0.0.1 or localhost. Without it the steps run in-process and no window opens |
| `--no-launch` | never start a Workbench or open a window |
| `--reset` | empty a Workbench that is already running before the run |
| `--flow-live` | after the strength index, build and solve the flow on this crop's lumen |
| `--flow-case DIR` | re-solve a copy of a complete OpenFOAM case instead; with `--flow-live`, only when the live build fails |
| `--flow-metrics FILE` | a JSON of flow numbers for the twin card's flow slot |
| `--out-dir DIR` | the run folder (default `outputs/copilot/<timestamp>`) |
| `--credit TEXT` | the scan's source and licence, printed in the report |
| `--model`, `--effort`, `--max-turns` | the model (default `claude-opus-5-5`), its effort (default `high`) and the turn limit (default 24) |
| `--open` / `--no-open` | open the report at the end (default: when the output is a terminal) |

The exit code is 0 only when the run reached the strength index and any
requested flow ran.

## 4. Run it without a key

```bash
python scripts/demo_run.py ct.nii.gz [--server URL] [--out DIR]
```

This plays the same path through the copilot's own tools, with no language
model and no API call. The steps, the gate and the numbers are the same;
there is no model summary.

## 5. The page

The Workbench always shows the demo layout. Open it by hand with
`meshforge web --port 8765`; it binds 127.0.0.1 only.

- **Left: the six tabs.** Each tab is the step's own panel, closed. A click
  opens its knobs and its Run button. The mark on a tab reads:
  - waiting (grey);
  - running (red, pulsing) while the step holds the session;
  - done (green ✓) once the step is in the session's record. A re-run of an
    earlier step puts the tabs after it back to waiting;
  - failed (red ✕), on Mesh wall only, while the last wall failed its
    quality gate, until a new wall passes.
- **Middle: the 3-D view.** A new result sweeps in from the top (the
  patient's head end) down, over about 2.4 s. The previous result stays as a
  faint grey shell meanwhile. The points, colours and colour range are the
  ones the server sent; only the drawing order is animated. Left alone for
  3 s, the view turns slowly (one turn in 45 s) and stops when you touch the
  canvas. Nothing animates when the system asks for reduced motion.
- **The running step** shows bottom-left of the view with its elapsed
  seconds, for example "contouring the aorta (AI outline)… 12 s".
- **The layer switch.** Once the session holds both a wall-stress and a
  flow result, two buttons over the view, **wall stress** and **blood
  flow**, switch between them. The flow layer is coloured by pressure on the
  lumen wall (velocity there is zero), with the wall-stress result as a
  translucent shell around it. The switch only shows results the session
  already holds; it computes nothing.
- **Right: the patient atlas.** One record: a front-view body map of 15
  regions (the patient's right on your left, as on a scan), modality chips
  (All, CT, MRI, X-ray, US, Angio) and a study list. It lists only the scan
  this session loaded. The demo scan is recognised by its checksum and lights
  the chest and torso; any other CT is listed with its coverage
  "unassigned". No prior study and no other patient is shown, and the date
  reads "date withheld". The list is a record, not a launcher.
- **Right: the twin card.** Slots for shape, wall + thrombus, stress,
  strength (with n), flow, the blood-pressure what-if, and growth. A slot
  waits ("not run yet"; the flow slot reads "not run here") until its step
  lands, then shows that step's numbers, for example "282.5 kPa peak at
  140 mmHg". The flow slot is marked filled once the flow has run; it lists
  flow numbers only when a set was attached with `--flow-metrics`. The card is rebuilt from the
  session's own record through the copilot's tool code, so it quotes the same
  numbers as the report. After a failed gate the wall slot says so ("last
  wall gate FAIL: 1.3 mm along the normal"). A new crop clears the stress and
  index it made stale. The growth slot stays empty: it needs a second scan.
- **The log** prints each step's result and every refusal with its reason.

## 6. The six steps

### 6.1 Load scan (`segment`, then `sac`)

TotalSegmentator (`total` task) outlines the aorta on the CT. The CT is
copied into a temporary folder, TotalSegmentator runs there as a separate
program, and the folder is deleted; the CT is only read and is checksummed
before and after. The mask keeps its largest connected piece and becomes a
closed surface in RAS millimetres.

The contrast blood lumen is the CT voxels at or above 150 HU inside the AI
outline. This is a threshold inside the AI label, not an AI segmentation.

On a native aneurysm the AI label often traces only the lumen. `sac`
recovers the outer sac (lumen + thrombus + wall): on every axial slice, rays
leave the lumen and stop at the fat plane (below -30 HU), at a bright
structure (another vessel, bone, an enhancing organ), at the scan edge, or
after 40 mm. Rays that find no fat plane are filled from their neighbours.

- **Check:** the step reports the fraction of rays that found a fat plane.
- **Limit:** where the sac touches an organ with no fat plane between them,
  the edge is interpolated, not measured.

### 6.2 Crop aneurysm (`crop`)

The crop finds three landmarks on the vessel's diameter profile: the
bifurcation, the widest section (the sac) and the neck above it. It cuts
10 mm above the bifurcation flare and 25 mm above the neck start, and closes
each opening with an exactly flat cap. The blood lumen is cut with the same
two planes. With `parallel_caps` on, both cuts are square to one common
axis, so the two caps are parallel.

- **Check:** it refuses, with the measured profile, when there is no sac
  (the widest section is less than 1.5 times the neck) or no neck.
- **Limit:** the renal arteries are not in the AI label, so the neck length
  is the non-dilated length above the sac, not a measured renal-to-sac
  distance.

### 6.3 Mesh wall (`volmesh`)

The wall is a structured hexahedral mesh, 1.5 mm thick, grown out from the
cropped sac surface. The 1.5 mm is the copilot's setting; run by hand, the
step's own default thickness is 0.5 (a unitless test value), so set it. Two
sweeps place the cross-sections:

- `centreline`: sections square to the vessel centreline.
- `planes`: sections parallel to the two end caps, evenly spaced from cap
  to cap. It needs caps parallel within 2°. Planes never cross, so the
  sections cannot fold.
- `auto`: the centreline sweep when its wall is clean, else the planes when
  the caps are parallel.

- **Check:** the quality gate (section 7).
- **Limit:** the wall thickness is assumed: 1.5 mm, uniform. A CT does not
  show the wall's real thickness.

### 6.4 Wall stress · FEA (`aaa_stress`)

CalculiX solves the wall under a uniform blood pressure at two pressures
(the systolic and the what-if), each a separate solve. Both ends are
clamped. The CT shows the aorta already under pressure, so the solve is
linear and small-strain on that loaded geometry; the stress is then set by
the geometry and the pressure. Wall E 100 MPa, ν 0.49, quadratic hexahedra
(C3D20R).

The thrombus between the cropped lumen and the sac is modelled as a solid
(quadratic tetrahedra, E 5 MPa) tied to the wall, and the pressure acts on
the lumen. Without it, the wall stress is overstated.

The peak wall stress (PWS) is the 99th percentile of the von Mises stress,
volume-weighted, outside a zone next to each clamped end (one neck diameter
by default).

- **Check:** it refuses any wall with an inverted cell or a capped or
  floored ray, and refuses when the end zones leave less than 10% of the
  wall. The copilot also checks the wall thickness under the peak (section 7).
- **Limit:** one uniform static pressure, an assumed wall thickness, linear
  material. The stress is proportional to the pressure, so the what-if shows
  the direction of the effect, not a benefit measured in this patient.

### 6.5 Strength index (`tissue_index`)

The index is the peak wall stress divided by the failure strength of
aneurysm wall measured in a lab. The strength data are open: uniaxial tests
of 13 strips of aneurysm wall from 7 aneurysms taken at autopsy (Gomes et
al., section 11). Strength is the Cauchy failure stress, the same measure as
the wall stress. The result is a band: P10, P50, P90; P90 uses the weakest
strength (P10).

- **Check:** every result states n (7 aneurysms, 13 strips).
- **Limit:** it is an investigational strength index, not a diagnosis. The
  band is wide because n is small. The tissue is autopsy tissue, and the
  test direction was not recorded.

### 6.6 Blood flow · CFD (`cfd`)

With `--flow-live`, the copilot fills this crop's blood lumen with
tetrahedra (6,748 cells on the demo scan) and writes an OpenFOAM case:

- the cap higher on the body is the neck and takes a resting 1 L/min, spread
  evenly over its area, along its inward normal;
- the other cap is the pressure outlet;
- every other face is a no-slip wall;
- blood's kinematic viscosity, 3.3 mm²/s; steady laminar `simpleFoam`.

The Workbench's own `cfd` step solves the case, so the flow sits in the same
session as the stress, on the same crop.

- **Check:** the terminal and the report say which case ran, with the
  solver, the cells and the seconds. A failed flow with no fallback fails
  the exit code.
- **Limit:** one steady resting flow and a plug inflow; no pulse, no
  compliant wall.

## 7. The quality gate and the recovery

A wall passes the gate only when all of these hold:

- 0 inverted cells;
- 0 capped and 0 floored wall rays (rays the mesher had to limit);
- the wall thickness measured along the surface normal is within 5% of
  1.5 mm (median over the wall).

A wall that fails gives no stress. After the stress solve, the wall under
the peak (the top 1% region) must also be within 5% of 1.5 mm, at both
pressures, or the call fails the same way.

On the demo scan:

1. The copilot starts with the standard settings: caps not forced parallel,
   sweep `auto`.
2. The first wall has no bad cells, but it is 1.30 mm thick along the
   surface normal. On this short, wide sac the centreline sweep's rays meet
   the sac's shoulders at an angle, and the wall comes out thin. A thin wall
   reads a higher stress. The gate fails it, and the Mesh wall tab shows ✕.
3. The error names the fix: re-crop with parallel caps and mesh with the
   plane sweep. The copilot does both.
4. The new wall is 1.50 mm thick, 7,552 hexahedra, 0 inverted: PASS.

The failed wall stays in the record and in the report.

## 8. What the numbers are and are not

The recorded run on the demo scan:

| Quantity | Value |
|---|---|
| sac maximum diameter (Load scan) | 61.8 mm |
| cropped sac / neck diameter | 51.8 mm / 24.5 mm |
| first wall | gate FAIL, 1.30 mm along the normal |
| second wall (parallel caps, plane sweep) | gate PASS, 1.50 mm, 7,552 hexahedra, 0 inverted |
| peak wall stress, thrombus modelled | 282.5 kPa at 140 mmHg; 242.1 kPa at 120 mmHg |
| strength index P50 (P10 to P90) | 0.43 (0.22 to 0.92) at 140 mmHg; 0.37 (0.19 to 0.79) at 120 mmHg |
| live flow case | 6,748 cells, `simpleFoam` converged |

What they are: the outputs of one model on one public scan, with its stated
assumptions.

What they are not:

- not a diagnosis, and not a statement about this person's health;
- not clinically validated;
- not computed by the language model: the model only chooses which tool to
  call and with which setting.

The summary in the report is included only when every number in it matches
a tool output. One correction is allowed; otherwise the summary is withheld
and the report says so. The rest of the report is built from the tool
outputs alone.

## 9. Where files land

The run folder (`--out-dir`, default `outputs/copilot/<timestamp>`):

| File | What it is |
|---|---|
| `report.html`, `report.md` | the report: the summary, the headline numbers, the twin card, the steps, then every number and call |
| `transcript.json` | every request, tool call and result |
| `session.mfwork` | the work file, written when the steps run in-process; `meshforge replay session.mfwork` reproduces them |
| `workbench.log` | the log of a Workbench the copilot started |
| `work/` | the shared step files: `lumen.vtp`, `aorta_mask.nii.gz`, `crop_lumen.vtp`, and the flow case (`flow_case_live/`) |

The Workbench's own steps write under `outputs/` in the folder it was
started from: wall stress in `outputs/aaa_stress/session/`. Nothing is ever
written next to or over an input file.

## 10. Troubleshooting

| Message | Meaning / fix |
|---|---|
| `TotalSegmentator not found: set MESHFORGE_TOTALSEG …` | Install it in its own environment (section 2), or point `MESHFORGE_TOTALSEG` at the executable. |
| `TotalSegmentator failed (exit code N). Last log lines: …` | The tail is TotalSegmentator's own output. Without the Apple GPU, set the Load scan knob `device` to `cpu` in the Workbench; `meshforge copilot` and `scripts/demo_run.py` always use the default `mps`. |
| `TotalSegmentator returned an empty aorta mask …` | The scan does not show the aorta, or it is a tight crop with no body around it. TotalSegmentator needs the full field of view. |
| `CalculiX (ccx) not found: …` | Set `MESHFORGE_CCX`, put `ccx` on `PATH`, or create the `ccxsolve` environment. |
| `OpenFOAM not found: …` | Set `MESHFORGE_OPENFOAM` to the launcher, put the solvers on `PATH`, or install it. |
| `the copilot needs the Anthropic SDK …` | `pip install -e ".[copilot]"` in the clone. |
| `the copilot could not authenticate with the API …` | Set `ANTHROPIC_API_KEY` in this shell. |
| `--server … names no port` / `… on 127.0.0.1 only` | Give a local address with a port, e.g. `http://127.0.0.1:8765`. |
| `the Workbench started on port N did not answer within 60 s` | The error ends with the tail of `workbench.log`, which says why. |
| `quality gate FAILED …` | Expected on the first wall of the demo scan; the copilot re-crops and re-meshes. By hand: `parallel_caps` 1 on Crop aneurysm, then `sweep` `planes` on Mesh wall (with the copilot's `thickness` 1.5 and `curvature_cap` 0.9). |
| `wall stress REFUSED: the end zones (…) leave N% of the wall volume …` | The crop is too short for one neck diameter at each clamped end. Set `end_zone` to `bending` or a length in mm. |
| OpenFOAM fails on a path | OpenFOAM cannot run under paths with spaces; meshforge stages each run in a temporary folder and refuses one with a space. |

## 11. Credits

- **Demo scan:** TotalSegmentator dataset, subject s1364. Wasserthal et al.,
  doi 10.5281/zenodo.10047292, CC BY 4.0.
- **Strength data:** Gomes VC, da Silva ES, da Silva LF, Gomes J, Raghavan
  ML, Silvestre G, Queiroz A, Marques M, Zyngier SP, Chung TK-J. *Dataset on
  cadaveric human abdominal aorta aneurysm biomechanics (uniaxial) and
  histology.* Mendeley Data, version 3 (2020). doi 10.17632/x64srrc39p.3,
  CC BY 4.0. The packaged table is derived from it (units converted, Cauchy
  stress recomputed, flags added); see
  `src/meshforge/data/STRENGTH_V0_SOURCES.md`.
- **Tools called at run time:** TotalSegmentator (aorta outline), gmsh
  (thrombus and lumen tetrahedra), CalculiX (wall stress), OpenFOAM (blood
  flow), vtk.js through trame-vtk (the 3-D view). See `NOTICE`.
