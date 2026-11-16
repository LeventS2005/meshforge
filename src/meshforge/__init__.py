"""meshforge, demo edition: one CT of an aortic aneurysm from scan to wall
stress and blood flow. Research software, not a medical device.

UI-agnostic core. All meshing/geometry/solve logic lives here; the web layer
and the copilot only drive the service layer (pipeline.MeshingSession).
"""
from __future__ import annotations

__version__ = "2.0.0"
