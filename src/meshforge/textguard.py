"""Tiny text-sanitisation helpers shared by the I/O and solver layers.

F2 (session-12 security fix): raw external-tool output and raw file bytes
get embedded in exception messages (ccx/OpenFOAM stdout tails,
``line!r`` from the .frd/.dat parsers, dwg2dxf stderr). Those messages are
printed to a terminal by the CLI, where ANSI/OSC control sequences would be
interpreted — screen clears, cursor moves, title/clipboard changes, or
success-spoofing. A hostile solver binary (via a ``MESHFORGE_*`` override or
first on PATH) or a crafted ``.frd``/``.F06``/``.dat`` is the injection
vector. ``scrub`` strips those control characters before the text is
embedded.
"""
from __future__ import annotations

_REPLACEMENT = "�"


def scrub(text: str) -> str:
    """Return ``text`` with non-printable control characters replaced by
    U+FFFD, keeping ``\\n`` and ``\\t`` (so multi-line tails stay readable).

    Everything else that ``str.isprintable`` rejects — ESC (0x1b), the other
    C0/C1 controls, etc. — is neutralised, so the result is inert when printed
    to a terminal.
    """
    if not isinstance(text, str):
        text = str(text)
    return "".join(
        c if (c == "\n" or c == "\t" or c.isprintable()) else _REPLACEMENT
        for c in text
    )
