# Contributing

Thanks for looking. meshforge is a research project maintained by one person;
issues and small, focused pull requests are welcome.

This repository holds one demo: an AI copilot takes a CT of an aortic
aneurysm from scan to wall stress, strength index and blood flow. See
[`docs/DEMO_GUIDE.md`](../docs/DEMO_GUIDE.md).

## Environment

Install as the [README](../README.md) describes. The external programs the
demo calls (TotalSegmentator, CalculiX, OpenFOAM) live outside meshforge;
the guide says how meshforge finds them.

## Tests

Run one test file per process:

```bash
python -m pytest -q tests/test_copilot.py
python -m pytest -q tests/test_crop.py
```

Tests that need an external program (TotalSegmentator, CalculiX, OpenFOAM)
or an optional dependency **skip cleanly** when it is absent. That is
deliberate; CI runs the ungated set.

What a change needs before it lands:

- Tests for the new behaviour **and** its failure branches, not just the
  happy path. Ground truth is analytic or a published benchmark wherever one
  exists; tolerances reflect discretization, never hand-tuning to make a
  test pass.
- The test files it touches green, and for anything touching the pipeline, a
  real run end to end with **0 inverted cells**.
- Test fixtures stay **tiny and synthetic**. Never commit clinical data, a
  real filename, or anything patient-derived.

## Architecture rules

These are load-bearing, not style preferences:

1. **User-supplied input files are read-only.** Checksum before and after; a
   writer that could overwrite an input must refuse loudly.
2. **Generated files go only into `outputs/`** (git-ignored) or the run
   folder the user names. Temp files go in a temp dir, never the repo.
3. **The core stays UI-agnostic.** All logic lives in `src/meshforge/`; the
   web layer may only call the service layer (`MeshingSession`). This is
   enforced by an AST scan, see `tests/rule4_scan.py`.
4. **Heavy and optional dependencies are lazy-imported**, so the CLI never
   pays for gmsh, the Anthropic SDK or FastAPI when it does not use them.
5. **Fail loudly.** A bad input raises with a message that says what was
   wrong and what to do; it never silently produces a plausible-looking
   result.
6. **Numbers come from the tools.** The copilot's text may quote only numbers
   that a tool returned; the guard in `src/meshforge/copilot/guard.py`
   checks this.

## Branches and commits

- Work on a feature branch off `main`; never commit directly to `main`.
- One logical change per commit, with a real message: a short imperative
  subject (`fix(web): …`, `docs: …`, `test: …`), then a body explaining
  *why* when it is not obvious.
- Documentation ships in the **same commit** as the code it describes.
- Open a PR against `main`; CI must be green.

## Security

Please report vulnerabilities privately, see [`SECURITY.md`](SECURITY.md).
Do not open a public issue with details, and never attach clinical data.
