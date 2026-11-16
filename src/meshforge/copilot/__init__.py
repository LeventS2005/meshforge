"""Claude copilot over the AAA pipeline (D92, M5).

A plain-language request becomes a sequence of pipeline steps: segment → sac
→ crop → wall mesh (quality gate) → wall stress (+ thrombus) → strength
index. Claude plans the order, calls the tools, and recovers when a quality
gate fails. The numbers come from the tools only. The report is built
deterministically from the tool outputs, and any prose Claude adds must pass
a numbers guard: every number in it has to match a tool output.

The steps run in-process (:class:`SessionBackend`) or on a running Workbench
(:class:`HttpBackend`), so the 3-D view updates live while the copilot works.
The Anthropic SDK is imported lazily; the rest of meshforge never needs it.
"""

from meshforge.copilot.agent import CopilotRun, run_copilot
from meshforge.copilot.backend import HttpBackend, SessionBackend
from meshforge.copilot.guard import check_numbers
from meshforge.copilot.report import render_report

__all__ = [
    "CopilotRun",
    "HttpBackend",
    "SessionBackend",
    "check_numbers",
    "render_report",
    "run_copilot",
]
