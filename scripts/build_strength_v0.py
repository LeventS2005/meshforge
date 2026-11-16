"""Build the packaged tissue strength library v0 from the USP AAA dataset.

Source: "Dataset on cadaveric human abdominal aorta aneurysm biomechanics
(uniaxial) and histology", Mendeley Data v3, doi 10.17632/x64srrc39p.3,
CC BY 4.0 (Gomes, da Silva, da Silva, Gomes, Raghavan, Silvestre, Queiroz,
Marques, Zyngier, Chung).

Writes (derived data, CC BY 4.0, attribution in STRENGTH_V0_SOURCES.md):
  src/meshforge/data/strength_v0.csv         one row per tensile strip
  src/meshforge/data/strength_v0_donors.csv  age / sex / shape / Dmax per aneurysm

Run it with an interpreter that has xlrd + openpyxl (they are NOT meshforge
dependencies), isolated, never from inside the data folder:

    python -I scripts/build_strength_v0.py <dataset dir> [--check] [--out-dir DIR]

<dataset dir> holds extracted/, _sha256.txt and _mendeley_dataset_v3.json.
--check rebuilds in memory and fails unless the files on disk are identical.

The downloaded files are untrusted data: nothing in them is executed, every
file is checked against the download-time sha256 list, and anything that does
not match the expected layout, units or column identities stops the build
with a message (fail loudly). The donors' autopsy sample numbers and the
operator name in the xls/TXT headers are read only to check them and are
never written out, and donor ages of 90 and over are written as "90+".
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))  # our own stdlib-only conversions

from meshforge.tissue import mechanics as mech  # noqa: E402

DOI = "10.17632/x64srrc39p.3"
LICENCE = "CC BY 4.0"
STRESS_MEASURE = "cauchy_incompressible"
STRAIN_MEASURE = "engineering_grip"

# File-name site codes (abbreviations doc: borda = border, dir/esq = right/
# left, ant/post; "Ao infra prox/middle/distal"). "abd..." is used in cases E
# and H and is not defined in the doc.
SITES = {
    "bordaant": ("rupture_border", "anterior", "border-ant"),
    "bordapost": ("rupture_border", "posterior", "border-post"),
    "bordadir": ("rupture_border", "right", "border-right"),
    "bordaesq": ("rupture_border", "left", "border-left"),
    "aoinfraprox": ("control", "infrarenal proximal", "control-infra-prox"),
    "aoinfradistal": ("control", "infrarenal distal", "control-infra-distal"),
    "abdprox": ("control", "'abd' proximal", "control-abd-prox"),
    "abdmedia": ("control", "'abd' middle", "control-abd-mid"),
    "abddistal": ("control", "'abd' distal", "control-abd-distal"),
}

XLS_HEADER = [
    "Data Point", "Displacement (mm)", "Load (kN)", "Load (N)", "Strain",
    "Tension (N/cm)", "Stress (N/cm^2)", "Adjusted Load", "Adjusted Tension",
    "Adjusted Stress", "Stretch Ratio",
]

# Three strips exist only as a chart image (no xls, no report). Their values
# were transcribed from the text Series IX prints on the chart ("Failure
# Stress", "Failure Tension", "Ultimate Yield" = strain at failure). The
# sha256 pins the transcription to those exact bytes; a changed image stops
# the build. The second name is a byte-identical duplicate in the record.
CHART_ONLY = [
    {
        "case": "F", "site": "bordadir", "file": "Case F/Case F borda dir.jpg",
        "duplicates": [],
        "sha256": "08850c18cb96e2713a35aa91eeaaa5ad1baf0486cffe8487fbfe3864f15809c5",
        "stress_n_per_cm2": "107.23", "tension_n_per_cm": "10.72", "strain": "0.47",
    },
    {
        "case": "G", "site": "bordaesq", "file": "Case G/CaseG borda esq .jpg",
        "duplicates": ["Case G/CaseG borda esq Curve Fitted.jpg"],
        "sha256": "30692354a2b337ccd518454a2109a72cb341fd545e449b8141f3d7e1ad83505c",
        "stress_n_per_cm2": "53.60", "tension_n_per_cm": "8.20", "strain": "0.25",
    },
    {
        "case": "G", "site": "bordadir", "file": "Case G/CaseGbordadir.jpg",
        "duplicates": ["Case G/CaseGbordadir Curve Fitted.jpg"],
        "sha256": "23925c116d402631715d9d027148bcd995b2048e4de16529b8787149a1af15fb",
        "stress_n_per_cm2": "65.03", "tension_n_per_cm": "9.71", "strain": "0.27",
    },
]

DONORS_FILE = "General info /Donors Demographic information.xlsx"
HISTOLOGY_FILE = (
    "Histological data - collagen and elastic fiber percentage of coverage table/"
    "Data in Brief HISTOLOGY.xlsx"
)
ABBREVIATIONS_FILE = "General info /ABBREVIATIONS - Data in brief.doc"
DONOR_HEADER = (
    "CASE CODE", "Age (YEARS)", "gender", "AORTA type", "Larger transverse diameter (cm)",
)
SHAPES = {"aneurysm - fusiform": "fusiform", "aneurysm - sacular": "saccular"}
# HIPAA Safe Harbor: ages over 89 are identifiers and must be grouped. The
# exact age is read only to validate it; 90 and over is written as "90+".
AGE_TOP_CODE = 90


def age_bucket(age: int) -> str:
    """Donor age for the shipped table: exact below 90, "90+" from 90 up."""
    return "90+" if age >= AGE_TOP_CODE else str(age)

# Facts established by the M0 inspection (data_inspection_m0.md 3.1-3.4). The
# build refuses to write a library that disagrees with any of them.
EXPECTED_STRIPS_PER_CASE = {"A": 4, "B": 4, "D": 4, "E": 5, "F": 1, "G": 2, "H": 4}
EXPECTED_LATE_PEAK = {"E-border-right", "E-control-abd-distal"}

# A late peak: the Cauchy maximum lies this far (strain) past the load
# maximum, i.e. the tissue had already started to fail and the stress kept
# rising only through the stretch factor. Observed gaps: 0.21 and 0.31 in the
# two E strips, <= 0.017 in the other 19.
LATE_PEAK_STRAIN_GAP = 0.10

NUM = r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?"

COLUMNS = [
    "strip_id", "case", "location_class", "site", "site_code", "aneurysm_shape",
    "failure_cauchy_mpa", "failure_engineering_mpa", "failure_strain",
    "failure_tension_n_per_mm", "thickness_mm", "thickness_source", "width_mm",
    "gauge_length_mm", "stress_measure", "strain_measure", "stress_unit",
    "tension_unit", "length_unit", "data_kind", "cauchy_recomputed",
    "reported_failure_stress_n_per_cm2", "reported_failure_tension_n_per_cm",
    "recomputed_minus_reported_n_per_cm2", "load_offset_at_failure_n", "late_peak",
    "first_peak_cauchy_mpa", "first_peak_engineering_mpa", "first_peak_strain",
    "test_direction", "source_file", "source_sheet", "source_cells", "source_report",
    "source_sha256", "notes", "dataset_doi", "licence",
]
DONOR_COLUMNS = [
    "case", "age_years", "sex", "aneurysm_shape", "max_transverse_diameter_cm",
    "n_strips_rupture_border", "n_strips_control", "source_file", "source_sheet",
    "source_cells", "dataset_doi", "licence",
]


class BuildError(RuntimeError):
    """The dataset is not what the build expects. Nothing is written."""


def fail(msg: str) -> None:
    raise BuildError(msg)


def g6(x: float) -> str:
    return format(x, ".6g")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -- integrity ---------------------------------------------------------------
def verify_integrity(root: Path) -> dict[str, str]:
    """Every extracted file matches the download-time sha256 list, and the
    extracted tree holds exactly the listed files (no symlinks)."""
    listing = root / "_sha256.txt"
    if not listing.is_file():
        fail(f"missing {listing}")
    listed: dict[str, str] = {}
    for line in listing.read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"([0-9a-f]{64})  \./(.+)", line)
        if not m:
            fail(f"unexpected line in {listing.name}: {line!r}")
        listed[m.group(2)] = m.group(1)
    extracted = root / "extracted"
    found: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(extracted, followlinks=False):
        for name in list(dirnames) + filenames:
            if os.path.islink(os.path.join(dirpath, name)):
                fail(f"symlink in the dataset tree: {os.path.join(dirpath, name)}")
        for name in filenames:
            p = Path(dirpath) / name
            found["extracted/" + p.relative_to(extracted).as_posix()] = sha256(p)
    want = {k: v for k, v in listed.items() if k.startswith("extracted/")}
    if set(found) != set(want):
        fail(
            "extracted tree differs from the sha256 list: missing "
            f"{sorted(set(want) - set(found))}, extra {sorted(set(found) - set(want))}"
        )
    bad = sorted(k for k in want if found[k] != want[k])
    if bad:
        fail(f"sha256 mismatch (file changed since download): {bad}")
    meta = "_mendeley_dataset_v3.json"
    if meta not in listed or sha256(root / meta) != listed[meta]:
        fail(f"{meta} is missing or changed since download")
    return {k[len("extracted/"):]: v for k, v in found.items()}


def verify_metadata(root: Path) -> None:
    d = json.loads((root / "_mendeley_dataset_v3.json").read_text(encoding="utf-8"))
    doi = (d.get("doi") or {}).get("id")
    lic = (d.get("data_licence") or {}).get("short_name")
    if doi != DOI or d.get("version") != 3 or lic != LICENCE:
        fail(f"dataset metadata: doi {doi!r}, version {d.get('version')!r}, licence {lic!r}")


# -- donors ------------------------------------------------------------------
def read_donors(extracted: Path) -> dict[str, dict]:
    import openpyxl

    def sheet_rows(rel: str, ncols: int) -> list[tuple]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # openpyxl: "Unknown extension"
            wb = openpyxl.load_workbook(extracted / rel, data_only=True, read_only=True)
            if wb.sheetnames != ["Sheet1"]:
                fail(f"{rel}: sheets {wb.sheetnames}, expected ['Sheet1']")
            rows = [
                tuple(r[:ncols]) for r in wb["Sheet1"].iter_rows(values_only=True)
                if any(v is not None for v in r)
            ]
            wb.close()
        return rows

    rows = sheet_rows(DONORS_FILE, 5)
    if rows[0] != DONOR_HEADER or len(rows) != 9:
        fail(f"{DONORS_FILE}: header {rows[0]!r} / {len(rows)} rows, expected 1 + 8")
    histo = sheet_rows(HISTOLOGY_FILE, 5)
    if histo[: len(rows)] != rows:
        fail(f"{HISTOLOGY_FILE}: its first five columns disagree with {DONORS_FILE}")
    donors = {}
    for i, (case, age, sex, shape, dmax) in enumerate(rows[1:], start=2):
        if case not in "ABCDEFGH" or len(case) != 1 or case in donors:
            fail(f"{DONORS_FILE} row {i}: case {case!r}")
        if not isinstance(age, int) or not 18 <= age <= 120:
            fail(f"{DONORS_FILE} row {i}: age {age!r}")
        if sex not in ("male", "female"):
            fail(f"{DONORS_FILE} row {i}: gender {sex!r}")
        if shape not in SHAPES:
            fail(f"{DONORS_FILE} row {i}: aorta type {shape!r}")
        if not isinstance(dmax, (int, float)) or not 1.0 <= float(dmax) <= 15.0:
            fail(f"{DONORS_FILE} row {i}: diameter {dmax!r} cm")
        donors[case] = {
            "case": case, "age_years": age, "sex": sex, "aneurysm_shape": SHAPES[shape],
            "max_transverse_diameter_cm": float(dmax), "row": i,
        }
    return donors


# -- raw-curve strips (xls + TXT report) --------------------------------------
def _cell(sheet, r: int, c: int, pattern: str, what: str, rel: str) -> re.Match:
    raw = sheet.cell_value(r, c)
    m = re.fullmatch(pattern, str(raw).strip())
    if not m:
        fail(f"{rel} Plan1!{chr(65 + c)}{r + 1}: {what} cell is {raw!r}")
    return m


def read_report(path: Path, rel: str) -> dict:
    text = path.read_bytes().decode("latin-1")
    lines = [ln.rstrip() for ln in text.splitlines()]
    joined = "\n".join(lines)
    for need in (
        r"Test Type: Tensile\b", r"Units: SI\b", r"Geometry: Rectangular\b",
        r"Number of specimens: 1\b", r"Extensometer: Disabled\b",
        r"Separate dimension entry: Enabled\b", r"\[bom teste\s*\]",
    ):
        if not re.search(need, joined):
            fail(f"{rel}: report lacks {need!r}")

    def field(label: str, unit: str) -> float:
        m = re.search(rf"^\s*{label}:\s*({NUM}) {unit}\s*$", joined, re.M)
        if not m:
            fail(f"{rel}: report has no '{label}: <number> {unit}' line")
        return float(m.group(1))

    rep = {
        "width": field("Width", "mm"),
        "thickness": field("Thickness", "mm"),
        "gauge": field("Spec gauge len", "mm"),
        "speed": field("Crosshead Speed", "mm/min"),
    }
    m = re.search(r"Number of data points:\s*(\d+)", joined)
    if not m:
        fail(f"{rel}: report has no 'Number of data points'")
    rep["n"] = int(m.group(1))
    data = []
    for ln in lines:
        d = re.fullmatch(rf"\s*(\d+),\s*({NUM}),\s*({NUM})", ln)
        if d:
            data.append((int(d.group(1)), float(d.group(2)), float(d.group(3))))
    if len(data) != rep["n"] or [p for p, _, _ in data] != list(range(1, rep["n"] + 1)):
        fail(f"{rel}: {len(data)} data lines, header says {rep['n']}")
    rep["data"] = data
    return rep


def read_raw_strip(extracted: Path, rel: str, case: str, site: str, hashes: dict) -> dict:
    import xlrd

    wb = xlrd.open_workbook(extracted / rel)
    if wb.sheet_names() != ["Plan1", "Plan2", "Plan3"] or any(
        wb.sheet_by_name(n).nrows for n in ("Plan2", "Plan3")
    ):
        fail(f"{rel}: sheets {wb.sheet_names()} (expected data only in Plan1)")
    s = wb.sheet_by_name("Plan1")
    if s.ncols != 11:
        fail(f"{rel}: Plan1 has {s.ncols} columns, expected 11")

    m = _cell(s, 0, 0, r"Sample ID:\s+\d+([a-z]+)", "sample id", rel)
    if m.group(1) != site + "t":  # the donor number is checked, never kept
        fail(f"{rel}: sample id names site {m.group(1)!r}, file name says {site + 't'!r}")
    speed = float(_cell(s, 0, 3, rf"Crosshead Speed:\s*({NUM})", "crosshead speed", rel).group(1))
    _cell(s, 1, 0, r"Specimen Notes: bom teste", "specimen note", rel)
    width = float(_cell(s, 2, 0, rf"Width:\s*({NUM})", "width", rel).group(1))
    thick = float(_cell(s, 2, 1, rf"Thickness:\s*({NUM})", "thickness", rel).group(1))
    gauge = float(_cell(s, 2, 2, rf"Spec gauge Length:\s*({NUM}) mm", "gauge length", rel).group(1))
    _cell(s, 2, 3, rf"Ext\. gauge Length:\s*{NUM} mm", "ext. gauge length", rel)
    max_load_txt = _cell(s, 3, 0, rf"Maximum Load:\s*({NUM}) N", "maximum load", rel).group(1)
    stress_txt = _cell(s, 3, 1, rf"Failure Stress:\s*({NUM}) N/cm\^2", "failure stress", rel).group(1)
    tension_txt = _cell(s, 3, 2, rf"Failure Tension:\s*({NUM}) N/cm", "failure tension", rel).group(1)
    header = [s.cell_value(6, c) for c in range(11)]
    if header != XLS_HEADER:
        fail(f"{rel}: row 7 header {header!r}")
    rows = []
    for r in range(7, s.nrows):
        vals = [s.cell_value(r, c) for c in range(11)]
        if not all(isinstance(v, float) and math.isfinite(v) for v in vals):
            fail(f"{rel}: Plan1 row {r + 1} is not 11 finite numbers: {vals!r}")
        rows.append(vals)
    n = len(rows)
    if n < 50 or [int(v[0]) for v in rows] != list(range(1, n + 1)):
        fail(f"{rel}: data points are not 1..{n}")

    # The Series IX report beside it: same specimen, same dimensions, same curve.
    stem = rel.rsplit("/", 1)[1].replace(" Specimen 1 Data File.xls", "")
    folder = (extracted / rel).parent
    want = {stem.lower() + ".txt", stem[:-1].lower() + ".txt"}
    reports = sorted(p.name for p in folder.iterdir() if p.name.lower() in want)
    if len(reports) != 1:
        fail(f"{rel}: expected exactly one report {sorted(want)}, found {reports}")
    rep_rel = rel.rsplit("/", 1)[0] + "/" + reports[0]
    rep = read_report(folder / reports[0], rep_rel)
    for key, val in (("width", width), ("thickness", thick), ("gauge", gauge), ("speed", speed)):
        if abs(rep[key] - val) > 1e-5 * max(1.0, abs(val)):
            fail(f"{rel}: {key} {val} vs report {rep_rel} {rep[key]}")
    if rep["n"] != n:
        fail(f"{rel}: {n} rows vs report {rep['n']}")
    for (p, u, kn), row in zip(rep["data"], rows):
        if abs(u - row[1]) > 5e-7 or abs(kn - row[2]) > 5e-7:
            fail(f"{rel}: data point {p} differs from the report ({u}, {kn} vs {row[1]}, {row[2]})")

    # Column identities (data_inspection_m0.md 3.3), recomputed from the raw
    # displacement, load and the entered dimensions.
    lam, sig, ten, eng = [], [], [], []
    peak_f = max(v[3] for v in rows)
    peak_af = max(v[7] for v in rows)
    for v in rows:
        p, u, kn, f, eps, t_col, s_col, af, at_col, as_col, lam_col = v
        strain = u / gauge
        stretch = 1.0 + strain
        checks = [
            ("Load (N) = 1000 x Load (kN)", abs(f - 1000.0 * kn) <= 1e-9),
            ("Strain = displacement / gauge", abs(eps - strain) <= 1e-6),
            ("Stretch Ratio = 1 + Strain", abs(lam_col - (1.0 + eps)) <= 1e-9),
        ]
        if abs(f) > 0.05 * peak_f:
            checks += [
                ("Stress = F lambda / A0", _rel(s_col, mech.mpa_to_n_per_cm2(
                    mech.cauchy_incompressible_mpa(f, stretch, width, thick))) <= 1e-6),
                # the 10x label error: the column holds N/mm, not N/cm
                ("Tension column = F sqrt(lambda) / w0 in N/mm", _rel(
                    t_col, mech.tension_n_per_mm(f, stretch, width)) <= 1e-6),
            ]
        if abs(af) > 0.05 * peak_af:
            checks += [
                ("Adjusted Stress = F_adj lambda / A0", _rel(as_col, mech.mpa_to_n_per_cm2(
                    mech.cauchy_incompressible_mpa(af, stretch, width, thick))) <= 1e-6),
                ("Adjusted Tension = F_adj sqrt(lambda) / w0 in N/cm", _rel(
                    at_col, 10.0 * mech.tension_n_per_mm(af, stretch, width)) <= 1e-6),
            ]
        for name, ok in checks:
            if not ok:
                fail(f"{rel}: data point {int(p)} breaks the identity {name}")
        off = f - af
        if not 0.0 <= off <= 0.1:
            fail(f"{rel}: data point {int(p)} load offset {off} N outside 0..0.1 N")
        lam.append(stretch)
        sig.append(mech.cauchy_incompressible_mpa(af, stretch, width, thick))
        ten.append(mech.tension_n_per_mm(af, stretch, width))
        eng.append(af / (width * thick))

    i = max(range(n), key=lambda k: sig[k])  # failure = maximum Cauchy stress
    j = max(range(n), key=lambda k: ten[k])
    k = max(range(n), key=lambda q: rows[q][7])  # load maximum
    reported = float(stress_txt)
    diff = mech.mpa_to_n_per_cm2(sig[i]) - reported
    if abs(diff) > 0.05 + 1e-9:
        fail(f"{rel}: recomputed failure stress {mech.mpa_to_n_per_cm2(sig[i]):.4f} N/cm^2 "
             f"does not round to the reported {stress_txt}")
    if abs(10.0 * ten[j] - float(tension_txt)) > 0.05 + 1e-9:
        fail(f"{rel}: recomputed failure tension {10 * ten[j]:.4f} N/cm vs reported {tension_txt}")
    if abs(rows[k][7] - float(max_load_txt)) > 0.05 + 1e-9:
        fail(f"{rel}: max adjusted load {rows[k][7]:.4f} N vs reported {max_load_txt}")
    strain_i = lam[i] - 1.0
    offset = rows[i][3] - rows[i][7]
    if offset > 0.02 * rows[i][3]:
        fail(f"{rel}: load offset at failure {offset:.4f} N is > 2% of the load")
    gap = strain_i - (lam[k] - 1.0)
    late = gap > LATE_PEAK_STRAIN_GAP
    if not late and gap > 0.03:
        fail(f"{rel}: stress peak {gap:.3f} strain past the load peak, neither clearly "
             "coincident nor clearly late")
    rec = {
        "failure_cauchy_mpa": sig[i], "failure_engineering_mpa": eng[i],
        "failure_strain": strain_i, "failure_tension_n_per_mm": ten[i],
        "thickness_mm": thick, "thickness_source": "entered_at_test",
        "width_mm": width, "gauge_length_mm": gauge, "data_kind": "raw_curve",
        "cauchy_recomputed": "yes", "reported_failure_stress_n_per_cm2": stress_txt,
        "reported_failure_tension_n_per_cm": tension_txt,
        "recomputed_minus_reported_n_per_cm2": format(diff, ".3g"),
        "load_offset_at_failure_n": format(offset, ".4g"),
        "late_peak": "yes" if late else "no",
        "source_file": rel, "source_sheet": "Plan1",
        "source_cells": (
            f"A3 width, B3 thickness, C3 gauge length, B4 failure stress, C4 failure "
            f"tension; curve rows 8-{n + 7} (B displacement, H adjusted load), "
            f"failure at data point {i + 1}"
        ),
        "source_report": rep_rel, "source_sha256": hashes[rel], "notes": [],
    }
    if late:
        drop = 1.0 - min(sig[k:i + 1]) / sig[k]
        rec.update(
            first_peak_cauchy_mpa=sig[k], first_peak_engineering_mpa=eng[k],
            first_peak_strain=lam[k] - 1.0,
        )
        rec["notes"].append(
            f"late peak: the reported failure stress is the global maximum at strain "
            f"{strain_i:.3f}, after the load peak at strain {lam[k] - 1.0:.3f} "
            f"(Cauchy {sig[k]:.3f} MPa) and a {drop:.0%} stress drop"
        )
    if gauge < 15.0:
        rec["notes"].append(f"short gauge length {gauge:g} mm (most strips about 20 mm)")
    return rec


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


def chart_only_strip(spec: dict, hashes: dict) -> dict:
    rel = spec["file"]
    if hashes.get(rel) != spec["sha256"]:
        fail(f"{rel}: sha256 {hashes.get(rel)} is not the transcribed image {spec['sha256']}")
    for dup in spec["duplicates"]:
        if hashes.get(dup) != spec["sha256"]:
            fail(f"{dup}: expected a byte-identical duplicate of {rel}")
    cauchy = mech.n_per_cm2_to_mpa(float(spec["stress_n_per_cm2"]))
    strain = float(spec["strain"])
    tension = mech.n_per_cm_to_n_per_mm(float(spec["tension_n_per_cm"]))
    return {
        "failure_cauchy_mpa": cauchy,
        "failure_engineering_mpa": mech.engineering_from_cauchy(cauchy, strain),
        "failure_strain": strain, "failure_tension_n_per_mm": tension,
        "thickness_mm": mech.thickness_from_tension_mm(tension, cauchy, strain),
        "thickness_source": "derived_t0_from_tension", "width_mm": None,
        "gauge_length_mm": None, "data_kind": "chart_only",
        "cauchy_recomputed": "no", "reported_failure_stress_n_per_cm2": spec["stress_n_per_cm2"],
        "reported_failure_tension_n_per_cm": spec["tension_n_per_cm"],
        "recomputed_minus_reported_n_per_cm2": None, "load_offset_at_failure_n": None,
        "late_peak": "no", "source_file": rel, "source_sheet": "",
        "source_cells": "chart text: 'Failure Stress', 'Failure Tension', 'Ultimate Yield' (strain)",
        "source_report": "", "source_sha256": spec["sha256"],
        "notes": [
            "chart-only: no raw curve; values transcribed from the chart text (strain to "
            "2 decimals); thickness derived as t0 = T sqrt(1+e) / sigma; no late peak "
            "visible on the chart"
        ],
    }


# -- build -------------------------------------------------------------------
def build(root: Path) -> tuple[str, str, dict]:
    hashes = verify_integrity(root)
    verify_metadata(root)
    extracted = root / "extracted"
    donors = read_donors(extracted)

    accounted = {DONORS_FILE, HISTOLOGY_FILE, ABBREVIATIONS_FILE}
    strips = []
    for rel in sorted(hashes):
        m = re.fullmatch(r"Case ([A-H]) ?/Case([A-H])([a-z]+)t Specimen 1 Data File\.xls", rel)
        if not m:
            continue
        case, case2, site = m.groups()
        if case != case2 or site not in SITES:
            fail(f"{rel}: folder case {case}, file case {case2}, site code {site!r}")
        rec = read_raw_strip(extracted, rel, case, site, hashes)
        folder = rel.rsplit("/", 1)[0]
        stem = rel.rsplit("/", 1)[1].replace(" Specimen 1 Data File.xls", "")
        chart = f"{folder}/{stem} Curve Fitted.jpg"
        if chart not in hashes:
            fail(f"{rel}: no chart {chart}")
        accounted |= {rel, rec["source_report"], chart}
        strips.append((case, site, rec))
    for spec in CHART_ONLY:
        if any(c == spec["case"] and s == spec["site"] for c, s, _ in strips):
            fail(f"chart-only strip {spec['case']}/{spec['site']} also has raw data")
        strips.append((spec["case"], spec["site"], chart_only_strip(spec, hashes)))
        accounted |= {spec["file"], *spec["duplicates"]}
    if accounted != set(hashes):
        fail(f"unaccounted dataset files: {sorted(set(hashes) - accounted)}")

    rows = []
    for case, site, rec in strips:
        cls, site_name, short = SITES[site]
        if case not in donors:
            fail(f"case {case} has strips but no donor row")
        notes = list(rec.pop("notes"))
        if site.startswith("abd"):
            notes.append("'abd' site code is not defined in the dataset's abbreviations document")
        row = {
            "strip_id": f"{case}-{short}", "case": case, "location_class": cls,
            "site": site_name, "site_code": site,
            "aneurysm_shape": donors[case]["aneurysm_shape"],
            "stress_measure": STRESS_MEASURE, "strain_measure": STRAIN_MEASURE,
            "stress_unit": "MPa", "tension_unit": "N/mm", "length_unit": "mm",
            "test_direction": "not recorded", "dataset_doi": DOI, "licence": LICENCE,
            "notes": "; ".join(notes),
            "first_peak_cauchy_mpa": None, "first_peak_engineering_mpa": None,
            "first_peak_strain": None,
        }
        row.update(rec)
        rows.append(row)
    order = {"rupture_border": 0, "control": 1}
    rows.sort(key=lambda r: (r["case"], order[r["location_class"]], r["strip_id"]))

    # The inspection's facts, enforced.
    per_case = {}
    for r in rows:
        per_case[r["case"]] = per_case.get(r["case"], 0) + 1
    if per_case != EXPECTED_STRIPS_PER_CASE:
        fail(f"strips per case {per_case}, expected {EXPECTED_STRIPS_PER_CASE}")
    n_border = sum(r["location_class"] == "rupture_border" for r in rows)
    n_raw = sum(r["data_kind"] == "raw_curve" for r in rows)
    if (len(rows), n_border, n_raw) != (24, 13, 21):
        fail(f"{len(rows)} strips / {n_border} border / {n_raw} raw, expected 24 / 13 / 21")
    late = {r["strip_id"] for r in rows if r["late_peak"] == "yes"}
    if late != EXPECTED_LATE_PEAK:
        fail(f"late-peak strips {sorted(late)}, expected {sorted(EXPECTED_LATE_PEAK)}")
    if len({r["strip_id"] for r in rows}) != len(rows):
        fail("duplicate strip ids")

    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({
            c: ("" if r[c] is None else g6(r[c]) if isinstance(r[c], float) else r[c])
            for c in COLUMNS
        })

    dout = io.StringIO()
    dw = csv.DictWriter(dout, fieldnames=DONOR_COLUMNS, lineterminator="\n")
    dw.writeheader()
    for case in sorted(donors):
        d = donors[case]
        dw.writerow({
            "case": case, "age_years": age_bucket(d["age_years"]), "sex": d["sex"],
            "aneurysm_shape": d["aneurysm_shape"],
            "max_transverse_diameter_cm": g6(d["max_transverse_diameter_cm"]),
            "n_strips_rupture_border": sum(
                r["case"] == case and r["location_class"] == "rupture_border" for r in rows),
            "n_strips_control": sum(
                r["case"] == case and r["location_class"] == "control" for r in rows),
            "source_file": DONORS_FILE, "source_sheet": "Sheet1",
            "source_cells": f"A{d['row']}:E{d['row']}", "dataset_doi": DOI, "licence": LICENCE,
        })
    summary = {
        "n_strips": len(rows), "n_rupture_border": n_border, "n_control": len(rows) - n_border,
        "n_raw_curve": n_raw, "n_chart_only": len(rows) - n_raw,
        "cases_with_strips": sorted(per_case), "late_peak": sorted(late),
        "max_abs_recompute_diff_n_per_cm2": max(
            abs(float(r["recomputed_minus_reported_n_per_cm2"]))
            for r in rows if r["cauchy_recomputed"] == "yes"),
        "n_dataset_files_verified": len(hashes),
    }
    return out.getvalue(), dout.getvalue(), summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", type=Path, help="folder with extracted/, _sha256.txt, metadata JSON")
    ap.add_argument("--out-dir", type=Path, default=REPO / "src" / "meshforge" / "data")
    ap.add_argument("--check", action="store_true",
                    help="rebuild in memory; fail unless the files on disk are identical")
    args = ap.parse_args(argv)
    root = args.dataset.resolve()
    if Path.cwd().resolve() == root or root in Path.cwd().resolve().parents:
        print("error: do not run this from inside the dataset folder", file=sys.stderr)
        return 2
    try:
        lib, donors, summary = build(root)
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    targets = {args.out_dir / "strength_v0.csv": lib,
               args.out_dir / "strength_v0_donors.csv": donors}
    if args.check:
        stale = [p.name for p, text in targets.items()
                 if not p.is_file() or p.read_text(encoding="utf-8") != text]
        if stale:
            print(f"error: out of date: {stale}", file=sys.stderr)
            return 1
        print("check: library files are identical to a fresh build")
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for p, text in targets.items():
            tmp = p.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(p)
            print(f"wrote {p} ({hashlib.sha256(text.encode()).hexdigest()[:16]})")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
