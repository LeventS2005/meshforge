"""Where the copilot's pipeline steps run: in-process, or on a live Workbench."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class StepOutcome:
    ok: bool
    summary: dict[str, Any]
    text: str
    error: str = ""


class SessionBackend:
    """Runs steps on an in-process :class:`~meshforge.pipeline.MeshingSession`
    (the same ``compute_<step>`` methods the Workbench and CLI call)."""

    name = "in-process session"

    def __init__(self, session=None):
        if session is None:
            from meshforge.pipeline import MeshingSession

            session = MeshingSession()
        self.session = session

    def run(self, step: str, knobs: dict[str, Any]) -> StepOutcome:
        method = getattr(self.session, f"compute_{step}", None)
        if method is None:
            return StepOutcome(False, {}, "", f"no pipeline step {step!r}")
        try:
            res = method(**knobs)
        except (ValueError, RuntimeError, FileNotFoundError, TypeError) as exc:
            return StepOutcome(False, {}, "", f"{type(exc).__name__}: {exc}")
        return StepOutcome(True, res.summary, res.text)

    def attach_flow_metrics(self, metrics: dict[str, Any]) -> StepOutcome:
        """Journal pre-computed flow metrics in the session as its own
        ``flow_metrics`` record (computes nothing), so the saved work file
        keeps them and the twin card reads them from the journal."""
        try:
            res = self.session.attach_flow_metrics(metrics)
        except ValueError as exc:
            return StepOutcome(False, {}, "", f"{type(exc).__name__}: {exc}")
        return StepOutcome(True, res.summary, res.text)


class HttpBackend:
    """Runs steps on a running Workbench (``meshforge web``) via
    ``POST /step/<name>``, so its viewport shows each artifact as it lands.
    A step's domain error comes back as HTTP 400 with the readable reason."""

    def __init__(self, base_url: str = "http://127.0.0.1:8765", timeout: float = 3600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.name = f"Workbench at {self.base_url}"

    def run(self, step: str, knobs: dict[str, Any]) -> StepOutcome:
        return self._post(f"/step/{step}", knobs)

    def attach_flow_metrics(self, metrics: dict[str, Any]) -> StepOutcome:
        """``POST /twin/flow``: the Workbench's session journals the metrics
        as its own ``flow_metrics`` record, so its twin panel shows them; a
        malformed set comes back as HTTP 400/422 with the reason."""
        return self._post("/twin/flow", {"metrics": metrics})

    def _post(self, path: str, body: dict[str, Any]) -> StepOutcome:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode()).get("detail", "")
            except Exception:  # noqa: BLE001 - any unreadable error body
                detail = ""
            return StepOutcome(False, {}, "", f"HTTP {exc.code}: {detail or exc.reason}")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"the Workbench at {self.base_url} is not reachable ({exc.reason}) — "
                f"start it with `meshforge web` or drop --server"
            ) from None
        return StepOutcome(True, body.get("summary") or {}, body.get("text") or "")
