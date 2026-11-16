"""Tissue strength library v0 and the strength index band.

The library is lab-measured failure stress of abdominal aortic aneurysm (AAA)
wall strips (USP AAA dataset, Mendeley Data doi 10.17632/x64srrc39p.3,
CC BY 4.0), parsed with full provenance by ``scripts/build_strength_v0.py``
into the packaged ``meshforge/data/strength_v0.csv``.

The index is peak wall stress (PWS) divided by wall strength. It is an
investigational index, not a diagnosis. Rules:

* **Like with like.** The wall-stress solve (stiff-linear on the loaded,
  imaged geometry) gives Cauchy stress, so strength is the strips' Cauchy
  failure stress (F * lambda / A0, incompressible). Engineering stress is
  available, as a sensitivity only.
* **Selection.** ``aaa`` = rupture-border strips (aneurysm wall, the default);
  ``control`` = non-dilated infrarenal strips of the same aortas (shown as a
  comparison); ``all`` = both.
* **Late peaks.** Two strips report a global maximum reached after the load
  had peaked and the stress had dropped. By default they enter at their first
  failure (the load peak), which is the lower, cautious value.
* **Band.** Strength P10/P50/P90 over the selected strips, either empirical
  (linear interpolation between order statistics, the default) or from a
  lognormal fit. The index band is PWS / strength with the quantiles flipped:
  index P90 = PWS / strength P10 (the weak-wall end), index P10 = PWS /
  strength P90. Both methods obey that identity exactly.

Small n is the point: every result carries n_strips and n_aneurysms.
stdlib csv + numpy only; no file is written here.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any

from meshforge.tissue import mechanics as mech

LIBRARY_RESOURCE = "strength_v0.csv"
DATASET_NAMES = {"10.17632/x64srrc39p.3": "USP AAA, Mendeley Data"}
CAVEAT = "investigational index, not a diagnosis"
LAB_LINE = "our lab will narrow this"
QUANTILES = (0.1, 0.5, 0.9)
#: Implausible for an aortic wall in MPa — almost certainly kPa or Pa.
PWS_MAX_MPA = 5.0
MIN_STRIPS = 3

SELECTIONS = {
    "aaa": ("rupture_border",),
    "control": ("control",),
    "all": ("rupture_border", "control"),
}
SELECTION_LABEL = {
    "aaa": "rupture-border AAA wall",
    "control": "non-dilated infrarenal controls",
    "all": "rupture-border + control",
}
METHODS = ("empirical", "lognormal")
MEASURES = ("cauchy", "engineering")
LATE_PEAK_RULES = ("first_peak", "reported")

KNOWN = {
    "stress_measure": {"cauchy_incompressible"},
    "strain_measure": {"engineering_grip"},
    "stress_unit": {"MPa"},
    "tension_unit": {"N/mm"},
    "length_unit": {"mm"},
    "location_class": {"rupture_border", "control"},
    "data_kind": {"raw_curve", "chart_only"},
    "late_peak": {"yes", "no"},
    "aneurysm_shape": {"fusiform", "saccular"},
}
REQUIRED_COLUMNS = (
    "strip_id", "case", "location_class", "site", "aneurysm_shape",
    "failure_cauchy_mpa", "failure_engineering_mpa", "failure_strain",
    "failure_tension_n_per_mm", "thickness_mm", "stress_measure", "strain_measure",
    "stress_unit", "tension_unit", "length_unit", "data_kind", "late_peak",
    "first_peak_cauchy_mpa", "first_peak_engineering_mpa", "first_peak_strain",
    "source_file", "source_sha256", "dataset_doi", "licence", "notes",
)


class StrengthLibraryError(ValueError):
    """The strength library is missing, empty, or has a row it cannot trust."""


@dataclass(frozen=True)
class StrengthRecord:
    strip_id: str
    case: str
    location_class: str
    site: str
    aneurysm_shape: str
    failure_cauchy_mpa: float
    failure_engineering_mpa: float
    failure_strain: float
    failure_tension_n_per_mm: float
    thickness_mm: float
    data_kind: str
    late_peak: bool
    first_peak_cauchy_mpa: float | None
    first_peak_engineering_mpa: float | None
    first_peak_strain: float | None
    source_file: str
    source_sha256: str
    dataset_doi: str
    licence: str
    notes: str

    def strength_mpa(self, measure: str = "cauchy", late_peak: str = "first_peak") -> float:
        """Failure stress in MPa for one stress measure and late-peak rule."""
        if measure not in MEASURES:
            raise ValueError(f"unknown stress measure {measure!r} (use {'/'.join(MEASURES)})")
        if late_peak not in LATE_PEAK_RULES:
            raise ValueError(f"unknown late-peak rule {late_peak!r} (use {'/'.join(LATE_PEAK_RULES)})")
        if self.late_peak and late_peak == "first_peak":
            v = self.first_peak_cauchy_mpa if measure == "cauchy" else self.first_peak_engineering_mpa
        else:
            v = self.failure_cauchy_mpa if measure == "cauchy" else self.failure_engineering_mpa
        assert v is not None  # load_strength_library guarantees it
        return v


@dataclass(frozen=True)
class StrengthLibrary:
    records: tuple[StrengthRecord, ...]
    source: str
    sha256: str
    dataset_dois: tuple[str, ...] = field(default=())
    licences: tuple[str, ...] = field(default=())

    def select(self, selection: str = "aaa", *, include_chart_only: bool = True) -> list[StrengthRecord]:
        if selection not in SELECTIONS:
            raise ValueError(
                f"unknown selection {selection!r} (use {'/'.join(SELECTIONS)})"
            )
        classes = SELECTIONS[selection]
        return [
            r for r in self.records
            if r.location_class in classes and (include_chart_only or r.data_kind != "chart_only")
        ]


# -- loading -----------------------------------------------------------------
def _num(row: dict, key: str, where: str, *, optional: bool = False) -> float | None:
    raw = (row.get(key) or "").strip()
    if not raw:
        if optional:
            return None
        raise StrengthLibraryError(f"{where}: {key} is empty")
    try:
        v = float(raw)
    except ValueError:
        raise StrengthLibraryError(f"{where}: {key} {raw!r} is not a number") from None
    if not math.isfinite(v) or v <= 0.0:
        raise StrengthLibraryError(f"{where}: {key} {raw!r} must be a finite number > 0")
    return v


def parse_strength_csv(text: str, source: str) -> StrengthLibrary:
    """Validate a strength library CSV (text) into records; refuse anything off."""
    reader = csv.DictReader(io.StringIO(text))
    cols = reader.fieldnames or []
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        raise StrengthLibraryError(f"{source}: missing column(s) {missing}")
    records: list[StrengthRecord] = []
    seen: set[str] = set()
    for i, row in enumerate(reader, start=2):
        sid = (row.get("strip_id") or "").strip()
        where = f"{source} line {i} ({sid or 'no strip_id'})"
        if not sid or sid in seen:
            raise StrengthLibraryError(f"{where}: strip_id empty or duplicated")
        seen.add(sid)
        for key, allowed in KNOWN.items():
            val = (row.get(key) or "").strip()
            if val not in allowed:
                what = "missing" if not val else f"unknown value {val!r}"
                raise StrengthLibraryError(
                    f"{where}: {key} {what} (accepted: {', '.join(sorted(allowed))}) — "
                    "refusing a strength value whose unit or stress measure is not stated"
                )
        for key in ("case", "site", "source_file", "source_sha256", "dataset_doi", "licence"):
            if not (row.get(key) or "").strip():
                raise StrengthLibraryError(f"{where}: {key} is empty (provenance is required)")
        cauchy = _num(row, "failure_cauchy_mpa", where)
        eng = _num(row, "failure_engineering_mpa", where)
        strain = _num(row, "failure_strain", where)
        if strain >= 5.0:
            raise StrengthLibraryError(f"{where}: failure_strain {strain} is not an engineering strain")
        if abs(mech.engineering_from_cauchy(cauchy, strain) - eng) > 1e-4 * eng:
            raise StrengthLibraryError(
                f"{where}: engineering {eng} MPa != Cauchy {cauchy} / (1 + strain {strain}) — "
                "the two stress measures disagree"
            )
        late = row["late_peak"].strip() == "yes"
        fp = [_num(row, k, where, optional=True) for k in (
            "first_peak_cauchy_mpa", "first_peak_engineering_mpa", "first_peak_strain")]
        if late and None in fp:
            raise StrengthLibraryError(f"{where}: a late-peak strip needs its first-peak values")
        if late and not (fp[0] <= cauchy and fp[2] < strain):  # type: ignore[operator]
            raise StrengthLibraryError(f"{where}: first peak must come before and below the reported peak")
        records.append(
            StrengthRecord(
                strip_id=sid, case=row["case"].strip(),
                location_class=row["location_class"].strip(), site=row["site"].strip(),
                aneurysm_shape=row["aneurysm_shape"].strip(),
                failure_cauchy_mpa=cauchy, failure_engineering_mpa=eng,  # type: ignore[arg-type]
                failure_strain=strain,  # type: ignore[arg-type]
                failure_tension_n_per_mm=_num(row, "failure_tension_n_per_mm", where),  # type: ignore[arg-type]
                thickness_mm=_num(row, "thickness_mm", where),  # type: ignore[arg-type]
                data_kind=row["data_kind"].strip(), late_peak=late,
                first_peak_cauchy_mpa=fp[0] if late else None,
                first_peak_engineering_mpa=fp[1] if late else None,
                first_peak_strain=fp[2] if late else None,
                source_file=row["source_file"].strip(), source_sha256=row["source_sha256"].strip(),
                dataset_doi=row["dataset_doi"].strip(), licence=row["licence"].strip(),
                notes=(row.get("notes") or "").strip(),
            )
        )
    if not records:
        raise StrengthLibraryError(f"{source}: the strength library is empty — refusing to guess a strength")
    return StrengthLibrary(
        records=tuple(records),
        source=source,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        dataset_dois=tuple(sorted({r.dataset_doi for r in records})),
        licences=tuple(sorted({r.licence for r in records})),
    )


def load_strength_library(path: str | Path | None = None) -> StrengthLibrary:
    """Load and validate a strength library; default = the packaged v0 CSV."""
    if path is None:
        from importlib import resources

        res = resources.files("meshforge.data").joinpath(LIBRARY_RESOURCE)
        if not res.is_file():
            raise StrengthLibraryError(
                f"packaged strength library meshforge/data/{LIBRARY_RESOURCE} is missing"
            )
        return parse_strength_csv(res.read_text(encoding="utf-8"), f"meshforge/data/{LIBRARY_RESOURCE}")
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"strength library not found: {p}")
    return parse_strength_csv(p.read_text(encoding="utf-8"), str(p))


# -- the band ----------------------------------------------------------------
def check_pws(pws_mpa: Any) -> float:
    if isinstance(pws_mpa, bool) or not isinstance(pws_mpa, (int, float)):
        raise ValueError(f"peak wall stress must be a number in MPa, got {pws_mpa!r}")
    v = float(pws_mpa)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"peak wall stress must be a finite number > 0 MPa, got {pws_mpa!r}")
    if v > PWS_MAX_MPA:
        raise ValueError(
            f"peak wall stress {v:g} MPa is implausible for an aortic wall (> {PWS_MAX_MPA:g} MPa) "
            "— is it in kPa or Pa? Pass MPa."
        )
    return v


def strength_quantiles(values: list[float], method: str = "empirical") -> dict[str, float]:
    """Strength P10/P50/P90 (MPa) of positive values by ``method``."""
    import numpy as np

    if method not in METHODS:
        raise ValueError(f"unknown band method {method!r} (use {'/'.join(METHODS)})")
    v = np.asarray(values, dtype=float)
    if v.size < MIN_STRIPS:
        raise ValueError(f"{v.size} strip(s) is too few for a band (need >= {MIN_STRIPS})")
    if not np.all(np.isfinite(v)) or np.any(v <= 0):
        raise ValueError("strength values must be finite and > 0")
    if method == "empirical":
        q = np.quantile(v, QUANTILES, method="linear")
    else:
        logs = np.log(v)
        mu, sd = float(logs.mean()), float(logs.std(ddof=1))
        q = [math.exp(mu + NormalDist().inv_cdf(p) * sd) for p in QUANTILES]
    return {f"p{round(p * 100)}": float(x) for p, x in zip(QUANTILES, q)}


def rupture_index_band(
    pws_mpa: float,
    library: StrengthLibrary | None = None,
    *,
    selection: str = "aaa",
    method: str = "empirical",
    measure: str = "cauchy",
    late_peak: str = "first_peak",
    include_chart_only: bool = True,
) -> dict[str, Any]:
    """P10/P50/P90 of index = PWS / strength over the selected strips.

    Index Pq = PWS / strength P(100-q): the index P90 (weak-wall end) comes from
    the strength P10 (the weak wall). Returns n_strips, n_aneurysms, the
    method, strength and index quantiles, and the strips behind them.
    """
    pws = check_pws(pws_mpa)
    lib = library if library is not None else load_strength_library()
    recs = lib.select(selection, include_chart_only=include_chart_only)
    if len(recs) < MIN_STRIPS:
        raise ValueError(
            f"selection {selection!r} holds {len(recs)} strip(s) — too few for a band "
            f"(need >= {MIN_STRIPS})"
        )
    values = [r.strength_mpa(measure, late_peak) for r in recs]
    s = strength_quantiles(values, method)
    index = {"p10": pws / s["p90"], "p50": pws / s["p50"], "p90": pws / s["p10"]}
    cases = sorted({r.case for r in recs})
    return {
        "pws_mpa": pws,
        "selection": selection,
        "selection_label": SELECTION_LABEL[selection],
        "method": method,
        "measure": measure,
        "late_peak": late_peak,
        "include_chart_only": include_chart_only,
        "n_strips": len(recs),
        "n_aneurysms": len(cases),
        "cases": cases,
        "strength_mpa": s,
        "strength_range_mpa": [min(values), max(values)],
        "index": index,
        "quantile_rule": "index Pq = PWS / strength P(100-q); index P90 is the weak-wall end",
        "late_peak_strips": [r.strip_id for r in recs if r.late_peak],
        "chart_only_strips": [r.strip_id for r in recs if r.data_kind == "chart_only"],
        "caveat": CAVEAT,
        "library": {
            "source": lib.source, "sha256": lib.sha256,
            "dataset_doi": list(lib.dataset_dois), "licence": list(lib.licences),
        },
    }


def index_band_report(
    pws_mpa: float,
    library: StrengthLibrary | None = None,
    *,
    selection: str = "aaa",
    method: str = "empirical",
) -> dict[str, Any]:
    """The primary band plus the sensitivities and the control comparison.

    Sensitivities (same PWS, one thing changed at a time): the other band
    method; engineering instead of Cauchy strength (mixed measures, shown to
    say why like-for-like matters); late-peak strips at their reported global
    maximum; raw-curve strips only. The comparison is the non-dilated control
    strips (omitted when they are the primary selection).
    """
    lib = library if library is not None else load_strength_library()
    primary = rupture_index_band(pws_mpa, lib, selection=selection, method=method)
    other = METHODS[1 - METHODS.index(method)]
    sens: dict[str, dict[str, Any]] = {}
    for key, kw in (
        (f"method_{other}", {"method": other}),
        ("measure_engineering", {"measure": "engineering"}),
        ("late_peak_reported", {"late_peak": "reported"}),
        ("raw_curve_only", {"include_chart_only": False}),
    ):
        try:
            band = rupture_index_band(pws_mpa, lib, selection=selection, **{"method": method, **kw})
        except ValueError as exc:  # e.g. too few raw strips in a toy library
            sens[key] = {"error": str(exc)}
            continue
        sens[key] = _brief(band)
    report = {**primary, "sensitivity": sens}
    if selection != "control":
        try:
            report["comparison_control"] = _brief(
                rupture_index_band(pws_mpa, lib, selection="control", method=method)
            )
        except ValueError as exc:
            report["comparison_control"] = {"error": str(exc)}
    return report


def _brief(band: dict[str, Any]) -> dict[str, Any]:
    return {k: band[k] for k in ("n_strips", "n_aneurysms", "method", "measure", "strength_mpa", "index")}


def _q3(d: dict[str, float]) -> str:
    return f"{d['p10']:.3f} / {d['p50']:.3f} / {d['p90']:.3f}"


def index_band_text(report: dict[str, Any], *, pws_source: str = "") -> str:
    """Human-readable band (step text and CLI share it)."""
    s, ix = report["strength_mpa"], report["index"]
    src = f"; {pws_source}" if pws_source else ""
    lines = [
        f"tissue index — {report['caveat']}",
        f"  peak wall stress   : {report['pws_mpa']:.3f} MPa (Cauchy{src})",
        f"  open strength data : {report['n_aneurysms']} aneurysms, {report['n_strips']} strips "
        f"({report['selection_label']}) — {LAB_LINE}",
        f"  wall strength      : P10 / P50 / P90 = {_q3(s)} MPa (Cauchy, {report['method']}; "
        f"strips {report['strength_range_mpa'][0]:.3f}..{report['strength_range_mpa'][1]:.3f})",
        f"  strength index     : P10 / P50 / P90 = {_q3(ix)}   (PWS / strength; "
        f"P90 = PWS / strength P10, the weak-wall end)",
    ]
    if report["late_peak_strips"]:
        rule = "the first failure (load peak)" if report["late_peak"] == "first_peak" else "the reported maximum"
        lines.append(f"  late-peak strips   : {', '.join(report['late_peak_strips'])} — entered at {rule}")
    if report["chart_only_strips"]:
        lines.append(f"  chart-only strips  : {', '.join(report['chart_only_strips'])} (no raw curve)")
    labels = {
        "method_lognormal": "lognormal fit",
        "method_empirical": "empirical quantiles",
        "measure_engineering": "engineering strength (mixed measures)",
        "late_peak_reported": "late peaks at reported max",
        "raw_curve_only": "raw-curve strips only",
    }
    lines.append("  sensitivity (index P10 / P50 / P90):")
    for key, band in report.get("sensitivity", {}).items():
        label = labels.get(key, key)
        if "error" in band:
            lines.append(f"    {label:<38}: n/a ({band['error']})")
        else:
            lines.append(
                f"    {label:<38}: {_q3(band['index'])}   "
                f"[{band['n_aneurysms']} aneurysms, {band['n_strips']} strips]"
            )
    comp = report.get("comparison_control")
    if comp:
        if "error" in comp:
            lines.append(f"  comparison, non-dilated controls: n/a ({comp['error']})")
        else:
            lines.append(
                f"  comparison, non-dilated controls ({comp['n_aneurysms']} aneurysms, "
                f"{comp['n_strips']} strips): strength {_q3(comp['strength_mpa'])} MPa, "
                f"index {_q3(comp['index'])}"
            )
    lib = report["library"]
    names = ", ".join(
        f"{DATASET_NAMES.get(d, 'dataset')} doi {d}" for d in lib["dataset_doi"]
    )
    lines.append(
        f"  strength data      : {names}, {', '.join(lib['licence'])} "
        f"({lib['source']}, sha256 {lib['sha256'][:12]})"
    )
    return "\n".join(lines) + "\n"


# -- wall-stress summary input -------------------------------------------------
def pws_from_summary(path: str | Path) -> tuple[float, str, str]:
    """Read ``pws_mpa`` from a wall-stress summary JSON.

    Tolerant of where it sits (top level or nested); loud when it is absent,
    when several different values are found, or when it is not a number.
    Returns (value, JSON location, sha256 of the file). The file is only read.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"wall-stress summary not found: {p}")
    raw = p.read_bytes()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"wall-stress summary {p} is not valid JSON: {exc}") from exc
    found: list[tuple[str, Any]] = []

    def walk(node: Any, where: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{where}.{k}" if where else str(k)
                if k == "pws_mpa":
                    found.append((here, v))
                else:
                    walk(v, here)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{where}[{i}]")

    walk(data, "")
    if not found:
        raise ValueError(
            f"no 'pws_mpa' key in {p} — the wall-stress summary must state the peak wall "
            "stress in MPa as pws_mpa (or pass --pws)"
        )
    values = {json.dumps(v, sort_keys=True) for _, v in found}
    if len(values) > 1:
        raise ValueError(
            f"{p} holds {len(found)} different pws_mpa values ({', '.join(w for w, _ in found)}) "
            "— refusing to pick one; pass --pws"
        )
    where, value = found[0]
    return check_pws(value), where, hashlib.sha256(raw).hexdigest()


def library_summary(library: StrengthLibrary | None = None) -> dict[str, Any]:
    """n and strength quantiles per selection (both measures, both methods)."""
    lib = library if library is not None else load_strength_library()
    out: dict[str, Any] = {"source": lib.source, "sha256": lib.sha256, "n_strips": len(lib.records)}
    for sel in SELECTIONS:
        recs = lib.select(sel)
        entry: dict[str, Any] = {
            "n_strips": len(recs), "n_aneurysms": len({r.case for r in recs}),
        }
        if len(recs) >= MIN_STRIPS:
            for measure in MEASURES:
                vals = [r.strength_mpa(measure) for r in recs]
                for method in METHODS:
                    entry[f"{measure}_{method}"] = strength_quantiles(vals, method)
        out[sel] = entry
    return out
