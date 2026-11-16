"""Path-entry normalisation, shared by every front-end (CLI, web).

A user types, pastes or drags a path in; the shell habits that ride along with
it are not part of the filename:

* ``~/Documents/aorta.stl``            — an unexpanded home shortcut
* ``"/path/with spaces/f.cso"``        — surrounding quotes from a copy
* ``/path/Example\\ CSO\\ Files/f.cso``  — drag-into-terminal backslash escapes
* ``/path/f.cso ``                     — a trailing space

Every one of those used to fail with a raw ``[Errno 2]``. Normalising them
**here, in the core** (not in the web layer) means the CLI and the Workbench
behave the same way.

Read-only by construction: this rewrites a STRING and never opens, stats or
touches a file, so rule 1 (user input files stay byte-for-byte unchanged) is
untouched.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

__all__ = ["normalize_path", "normalize_path_str"]

_QUOTES = ("'", '"')

# Characters a shell escapes with a backslash when you drag a file into a
# terminal. A backslash before anything else (e.g. a Windows separator) is
# left exactly as typed — we un-escape, we do not guess.
_ESCAPABLE = set(" \t()[]{}<>|&;'\"`*?$!#~,")
_ESCAPE_RX = re.compile(r"\\(.)", re.DOTALL)


def _unescape(match: "re.Match[str]") -> str:
    char = match.group(1)
    return char if char in _ESCAPABLE else match.group(0)


def normalize_path_str(raw: str | Path) -> str:
    """Clean one user-entered path string; ``""`` stays ``""``.

    Order matters: trim, then unwrap quotes (a quoted path may itself have
    padding inside), then un-escape, then expand ``~``. Idempotent — running
    it twice is the same as running it once.
    """
    text = str(raw).strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in _QUOTES:
        text = text[1:-1].strip()
    text = _ESCAPE_RX.sub(_unescape, text)
    return os.path.expanduser(text)


def normalize_path(raw: str | Path) -> Path:
    """:func:`normalize_path_str` as a :class:`~pathlib.Path`.

    ``""`` becomes ``Path("")`` (i.e. ``.``) exactly as ``Path("")`` always
    did — callers that require a non-empty path keep their own loud guard.
    """
    return Path(normalize_path_str(raw))
