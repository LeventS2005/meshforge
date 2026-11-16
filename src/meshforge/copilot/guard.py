"""Numbers guard: every number in the copilot's prose must come from a tool.

A number in the text matches a tool number when it equals that number
rounded to the precision the text prints (``278`` matches 278.2; ``0.43``
matches 0.428) or is within 0.5 % of it. Digits glued to letters are labels,
not quantities (``P10``, ``s1364``, ``C3D10``), and are ignored. So are the
numbers the user typed in the request.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

_NUMBER = re.compile(r"(?<![A-Za-z_\d.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?![A-Za-z_\d])")
_REL_TOL = 0.005


def numbers_in(text: str) -> list[str]:
    """The numeric tokens of ``text`` as written (labels like P10 excluded)."""
    return [m.group(0) for m in _NUMBER.finditer(text)]


def _flatten(obj: Any) -> Iterable[float]:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _flatten(v)
    elif isinstance(obj, str):
        for tok in numbers_in(obj):
            yield float(tok.replace(",", ""))


def _matches(token: str, value: float) -> bool:
    x = float(token.replace(",", ""))
    decimals = len(token.split(".")[1]) if "." in token else 0
    for v in (value, abs(value)):  # prose says "drops by 14%" for a -14.3 change
        if abs(x - round(v, decimals)) < 0.5 * 10 ** (-decimals) + 1e-12:
            return True
        if v != 0 and abs(x - v) <= _REL_TOL * abs(v):
            return True
    return False


def check_numbers(text: str, sources: Iterable[Any], *, allowed_text: str = "") -> list[str]:
    """The numeric tokens of ``text`` that no source number backs (empty =
    the text passes). ``sources`` are tool payloads (any JSON-like objects);
    ``allowed_text`` (the user's request) contributes numbers verbatim."""
    values = [v for src in sources for v in _flatten(src)]
    values += [float(t.replace(",", "")) for t in numbers_in(allowed_text)]
    return [tok for tok in numbers_in(text) if not any(_matches(tok, v) for v in values)]
