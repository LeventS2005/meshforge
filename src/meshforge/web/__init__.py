"""meshforge local web layer (roadmap v7, V35).

A thin, front-end-agnostic FastAPI layer that exposes ``MeshingSession`` 1:1 as
JSON — one endpoint per compute step, request models generated from
``STEP_KNOBS``, journal/workfile endpoints for history. **Rule 4 governs:**
this package holds ZERO meshing/solve/export-writer logic; every endpoint
only CALLS a ``MeshingSession`` step (or ``load_workfile``/``replay``/
``save_workfile``) and returns its ``StepResult``/journal as JSON.

FastAPI/uvicorn are imported inside :mod:`meshforge.web.app` /
:mod:`meshforge.web.server`, which the CLI imports lazily (inside its ``web``
subcommand) — so ``import meshforge.cli`` never pays the web
import cost.
"""
