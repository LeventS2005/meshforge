"""STEP_KNOBS -> JSON schema + per-step pydantic request models.

The single source of truth is ``pipeline.STEP_KNOBS`` (``{step: {knob:
(type, default, help)}}``). Two derived artefacts:

* :func:`schema_json` — the declarative knob table as JSON (types as names),
  served at ``GET /schema``; the frontend generates its panels from this.
* :func:`build_request_models` — one pydantic model per step, every knob
  ``Optional`` and defaulting to ``None`` so an *unset* knob is dropped by the
  handler and the compute step's own default applies (the empty-⇒-omit
  rule).

No meshing/geometry logic lives here — only a projection of STEP_KNOBS.
"""
from __future__ import annotations

from typing import Any, Optional

from meshforge.pipeline import STEP_KNOBS, STEP_ORDER

_TYPE_NAMES = {int: "int", float: "float", str: "str", bool: "bool"}


def step_order() -> list[str]:
    """The steps in front-end display order (``pipeline.STEP_ORDER``).

    Served alongside the schema so the Workbench lays its panels out from an
    explicit list rather than from dict insertion order.
    """
    return list(STEP_ORDER)


def schema_json() -> dict[str, dict[str, dict[str, Any]]]:
    """STEP_KNOBS as JSON: ``{step: {knob: {type, default, help}}}``.

    Emitted in :data:`~meshforge.pipeline.STEP_ORDER` order so a client that
    ignores the explicit order list still gets the intended panel order.
    """
    return {
        step: {
            knob: {
                "type": _TYPE_NAMES.get(typ, typ.__name__),
                "default": default,
                "help": help_text,
            }
            for knob, (typ, default, help_text) in STEP_KNOBS[step].items()
        }
        for step in STEP_ORDER
    }


def build_request_models() -> dict[str, type]:
    """One pydantic model per STEP_KNOBS step.

    Each knob is ``Optional[type]`` **defaulting to its STEP_KNOBS default**
    (help text as its description). This mirrors the Workbench exactly: it
    pre-fills every knob widget with the STEP_KNOBS default and sends it, so
    a ``0.5`` default (e.g. ``volmesh.thickness``) must be sent, not left to the
    compute method's own — different — Python default. :func:`drop_unset`
    then strips only the ``None``/empty-string defaults, which are the
    "compute-a-default" sentinels (e.g. ``centerline.voxel_size``) that the
    compute step fills in itself.
    """
    from pydantic import ConfigDict, Field, create_model

    models: dict[str, type] = {}
    for step, knobs in STEP_KNOBS.items():
        fields: dict[str, tuple[Any, Any]] = {
            knob: (Optional[typ], Field(default=default, description=help_text))
            for knob, (typ, default, help_text) in knobs.items()
        }
        # extra="forbid": an unknown knob is a loud 422, never silently dropped.
        model = create_model(
            f"{step}_request",
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )
        models[step] = model
    return models


def drop_unset(dumped: dict[str, Any]) -> dict[str, Any]:
    """Strip knobs the client left unset (``None`` or empty string).

    The front-end rule (empty ⇒ omit ⇒ compute default applies).
    """
    return {
        key: value
        for key, value in dumped.items()
        if value is not None and not (isinstance(value, str) and value == "")
    }
