"""Patient atlas: the body regions a patient record is filed by, and the
studies the current session really holds (Workbench Demo mode, D108
addendum 6).

The Workbench shows one patient record with every study by body region and
modality, the way every surgeon would land on the same page. This module is
its honest, GUI-agnostic back: the 15-region catalogue, the modality list,
and ``studies_for(session)``, which lists exactly the scans the session
loaded — today at most one, the CT the ``segment`` step outlined. The only
scan whose body coverage is *declared* is the public demo CT, matched by
checksum; any other CT is listed with its coverage ``unassigned``. Nothing
here invents a prior, a date, a second study or a second patient, and dates
are withheld (no PHI).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

#: The 15 body regions of the front-view body map, (id, name), filed by the
#: PATIENT's side (the patient's right is on the viewer's left, as on a scan).
#: The ids are the Workbench's SVG region ids; the asset test pins the two.
REGIONS: tuple[tuple[str, str], ...] = (
    ("head", "Head and neck"),
    ("chest", "Chest"),
    ("torso", "Torso (abdomen and pelvis)"),
    ("shoulder_r", "Right shoulder"),
    ("shoulder_l", "Left shoulder"),
    ("arm_r", "Right arm"),
    ("arm_l", "Left arm"),
    ("hand_r", "Right hand"),
    ("hand_l", "Left hand"),
    ("upper_leg_r", "Right upper leg"),
    ("upper_leg_l", "Left upper leg"),
    ("lower_leg_r", "Right lower leg"),
    ("lower_leg_l", "Left lower leg"),
    ("foot_r", "Right foot"),
    ("foot_l", "Left foot"),
)

#: The modalities a record is filed under (the chips).
MODALITIES: tuple[str, ...] = ("CT", "MRI", "X-ray", "US", "Angio")

#: SHA-256 of the public demo CT: TotalSegmentator v2 subject s1364
#: (Wasserthal et al., CC BY 4.0), ``ct.nii.gz``, 1.5 mm isotropic.
DEMO_CT_SHA256 = "7957f71d5a673486f0d23b403ece22233fdf1a5ac02a2ee4649fb99006f856dd"

#: Scans whose body coverage is declared, by checksum of the file. One entry:
#: the demo CT covers the abdomen, the pelvis and the lower chest.
KNOWN_SCANS: dict[str, dict[str, Any]] = {
    DEMO_CT_SHA256: {
        "patient": "s1364",
        "credit": "de-identified (TotalSegmentator v2, CC BY 4.0)",
        "description": "abdomen + pelvis + lower chest · 1.5 mm",
        "regions": ("chest", "torso"),
    },
}


def studies_for(session: Any) -> list[dict[str, Any]]:
    """The studies ``session`` really holds: at most one — the CT the
    ``segment`` step loaded, whose AI outline the session still keeps (every
    other source step clears it). The study is the session's own source path
    (file name only: no folder reaches a screen) and checksum; the regions it
    covers come from :data:`KNOWN_SCANS`, or stay empty, ``unassigned``."""
    if getattr(session, "outline", None) is None:
        return []  # the source is not an AI-outlined CT (STL, synth, CAD, ...)
    path = getattr(session, "input_path", None)
    if not path:
        return []
    known = KNOWN_SCANS.get(getattr(session, "input_sha256", None) or "")
    name = Path(path).name
    return [
        {
            "modality": "CT",
            "description": known["description"] if known else f"{name} · coverage unassigned",
            "date": "date withheld",
            "status": "current",
            "loaded": True,
            "regions": list(known["regions"]) if known else [],
            "coverage": "declared" if known else "unassigned",
            "file": name,
            "patient": known["patient"] if known else None,
        }
    ]


def atlas_for(session: Any) -> dict[str, Any]:
    """The atlas payload: the catalogue, the modalities, the studies the
    session holds, and the patient line (``None`` while nothing is loaded)."""
    studies = studies_for(session)
    patient = None
    if studies:
        known = KNOWN_SCANS.get(getattr(session, "input_sha256", None) or "")
        patient = (
            f"1 patient on file · {known['patient']} · {known['credit']}"
            if known
            else "1 patient on file · unassigned scan · coverage not declared"
        )
    return {
        "regions": [{"id": rid, "name": name} for rid, name in REGIONS],
        "modalities": list(MODALITIES),
        "n_patients": 1 if studies else 0,
        "patient": patient,
        "studies": studies,
    }
