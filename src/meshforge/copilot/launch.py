"""The copilot's own Workbench: start one when none answers at ``--server``,
open its window, empty a running one on request, stage the blood-lumen
OpenFOAM case it re-solves, and open the finished report. Process plumbing
only: no meshing, no numbers.

Everything here is a thin wrapper over ``subprocess`` / ``urllib`` /
``webbrowser`` so the CLI stays short and the tests can stand each piece in.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from urllib.parse import urlsplit

#: the browser the demo windows use on macOS (an app-mode window: no tabs,
#: no address bar); anywhere else, or without it, the default browser
CHROME_APP = "/Applications/Google Chrome.app"
CHROME_NAME = "Google Chrome"
#: how long a freshly started Workbench may take to answer /health
START_WAIT_S = 60.0


def health(url: str, timeout: float = 1.5) -> bool:
    """True when a Workbench answers ``GET /health`` at ``url``."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=timeout) as resp:
            return resp.status == 200 and bool(json.loads(resp.read().decode()).get("ok"))
    except (urllib.error.URLError, ValueError, OSError):
        return False


def port_of(url: str) -> int:
    """The port a ``--server`` URL names (loud when it names none)."""
    parts = urlsplit(url if "//" in url else f"http://{url}")
    if parts.port is None:
        raise ValueError(f"--server {url!r} names no port (e.g. http://127.0.0.1:8765)")
    if parts.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError(f"--server {url!r}: the copilot starts a Workbench on 127.0.0.1 only")
    return int(parts.port)


def start_workbench(url: str, log_path: str | Path, *, wait_s: float = START_WAIT_S,
                    popen=subprocess.Popen, sleep=time.sleep) -> tuple[subprocess.Popen, float]:
    """Start ``meshforge web`` on the port of ``url`` as its own process group
    (it outlives the copilot: the window stays live after the run), logging
    to ``log_path``, and wait until ``/health`` answers. Returns the process
    and the seconds it took; a Workbench that never answers is stopped and
    reported with the tail of its log."""
    port = port_of(url)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")  # noqa: SIM115 - handed to the child, closed with it
    argv = [sys.executable, "-m", "meshforge.web.server", "--port", str(port)]
    proc = popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                 start_new_session=True)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < wait_s:
        if proc.poll() is not None:
            break
        if health(url, timeout=0.8):
            return proc, time.perf_counter() - t0
        sleep(0.25)
    if proc.poll() is None:
        proc.terminate()
    try:
        tail = log_path.read_text(errors="replace")[-1500:]
    except OSError:
        tail = ""
    raise RuntimeError(
        f"the Workbench started on port {port} did not answer within {wait_s:.0f} s"
        + (f"; its log ends with:\n{tail}" if tail.strip() else "")
    )


def reset_workbench(url: str, timeout: float = 30.0) -> None:
    """``POST /session/reset``: empty a running Workbench's session (a fresh
    take starts from grey tabs). Loud when it fails."""
    req = urllib.request.Request(f"{url.rstrip('/')}/session/reset", data=b"{}",
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}")
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"the Workbench at {url} did not reset: {exc}") from None


def chrome_available() -> bool:
    return platform.system() == "Darwin" and Path(CHROME_APP).is_dir()


def open_url(url: str, *, run=subprocess.run, browser=webbrowser) -> str:
    """Open ``url`` in a Chrome app-mode window (macOS with Chrome), else the
    default browser. Returns a short note of what was opened."""
    if chrome_available():
        run(["open", "-na", CHROME_NAME, "--args", f"--app={url}"], check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "Chrome app window"
    browser.open(url)
    return "default browser"


def open_window(url: str, **kw) -> str:
    """Open the Workbench window (the demo page at ``/app/``)."""
    return open_url(f"{url.rstrip('/')}/app/", **kw)


def open_file(path: str | Path, **kw) -> str:
    """Open a local file (the finished report) the same way."""
    return open_url(Path(path).resolve().as_uri(), **kw)


def _is_result_dir(name: str) -> bool:
    try:
        return float(name) > 0.0
    except ValueError:
        return False


def stage_case(src: str | Path, dest: str | Path) -> Path:
    """Copy an OpenFOAM case to ``dest`` without its earlier results (time
    directories after 0 and ``log.*``), so the session's cfd step solves it
    afresh and the source case stays as it was. Loud on a folder that is no
    case."""
    src, dest = Path(src), Path(dest)
    if not (src / "system" / "controlDict").is_file():
        raise ValueError(f"{src} is not an OpenFOAM case (no system/controlDict)")
    if not (src / "constant" / "polyMesh").is_dir():
        raise ValueError(f"{src} has no mesh (constant/polyMesh)")
    if dest.exists():
        shutil.rmtree(dest)

    def skip(folder, names):
        if Path(folder) != src:
            return []
        return [n for n in names if _is_result_dir(n) or n.startswith("log.")]

    shutil.copytree(src, dest, ignore=skip)
    return dest
