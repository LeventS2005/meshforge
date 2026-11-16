"""D89 (M4): tissue strength library v0 + strength index band.

Runs on the packaged library (src/meshforge/data/strength_v0.csv, derived
from the open USP AAA dataset, CC BY 4.0) and on toy libraries written here.
The dataset itself is not needed: the parser that built the CSV
(scripts/build_strength_v0.py) needs xlrd/openpyxl, which are not meshforge
dependencies; its ``--check`` mode is the rebuild proof.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from importlib import resources

import pytest

from meshforge.tissue import mechanics as mech
from meshforge.tissue.strength import (
    CAVEAT,
    REQUIRED_COLUMNS,
    StrengthLibraryError,
    check_pws,
    index_band_report,
    index_band_text,
    load_strength_library,
    parse_strength_csv,
    pws_from_summary,
    rupture_index_band,
)

DOI = "10.17632/x64srrc39p.3"


def packaged_text(name: str = "strength_v0.csv") -> str:
    return resources.files("meshforge.data").joinpath(name).read_text(encoding="utf-8")


def toy_row(strip_id: str, case: str, cauchy: float, *, strain: float = 0.25,
            cls: str = "rupture_border", **over) -> dict:
    row = {c: "" for c in REQUIRED_COLUMNS}
    row.update(
        strip_id=strip_id, case=case, location_class=cls, site="left",
        aneurysm_shape="fusiform", failure_cauchy_mpa=repr(cauchy),
        failure_engineering_mpa=repr(cauchy / (1.0 + strain)), failure_strain=repr(strain),
        failure_tension_n_per_mm="1.0", thickness_mm="1.5",
        stress_measure="cauchy_incompressible", strain_measure="engineering_grip",
        stress_unit="MPa", tension_unit="N/mm", length_unit="mm", data_kind="raw_curve",
        late_peak="no", source_file=f"toy/{strip_id}.xls", source_sha256="0" * 64,
        dataset_doi="toy", licence="CC0", notes="",
    )
    row.update(over)
    return row


def toy_csv(rows: list[dict]) -> str:
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=list(REQUIRED_COLUMNS), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return out.getvalue()


@pytest.fixture
def toy3():
    """Three strips, 0.5 / 1.0 / 1.5 MPa Cauchy, from two aneurysms."""
    return parse_strength_csv(
        toy_csv([toy_row("X-a", "X", 0.5), toy_row("X-b", "X", 1.0), toy_row("Y-a", "Y", 1.5)]),
        "toy3",
    )


# -- unit and measure conversions -----------------------------------------------
def test_unit_conversions():
    assert mech.n_per_cm2_to_mpa(58.0) == pytest.approx(0.58, abs=1e-15)
    assert mech.mpa_to_n_per_cm2(0.58) == pytest.approx(58.0, abs=1e-12)
    assert mech.n_per_cm_to_n_per_mm(12.3) == pytest.approx(1.23, abs=1e-15)
    # engineering <-> Cauchy under incompressibility: factor 1 + strain
    assert mech.engineering_from_cauchy(1.2, 0.2) == pytest.approx(1.0, abs=1e-15)
    assert mech.cauchy_from_engineering(1.0, 0.2) == pytest.approx(1.2, abs=1e-15)
    for c, e in ((0.58, 0.22), (1.99, 0.186), (0.27, 0.76)):
        assert mech.cauchy_from_engineering(mech.engineering_from_cauchy(c, e), e) == pytest.approx(c)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            mech.n_per_cm2_to_mpa(bad)
    with pytest.raises(ValueError, match="non-positive stretch"):
        mech.engineering_from_cauchy(1.0, -1.0)
    with pytest.raises(ValueError, match="width"):
        mech.cauchy_incompressible_mpa(1.0, 1.1, 0.0, 1.0)


def test_cauchy_recomputation_formulas():
    # exact synthetic strip: F 10 N, lambda 1.25, w0 5 mm, t0 2 mm
    assert mech.cauchy_incompressible_mpa(10.0, 1.25, 5.0, 2.0) == pytest.approx(1.25)
    assert mech.engineering_from_cauchy(1.25, 0.25) == pytest.approx(1.0)  # F / A0
    t = mech.tension_n_per_mm(10.0, 1.25, 5.0)
    assert t == pytest.approx(2.0 * math.sqrt(1.25))
    assert t == pytest.approx(1.25 * 2.0 / math.sqrt(1.25))  # T = sigma * t0 / sqrt(lambda)
    assert mech.thickness_from_tension_mm(t, 1.25, 0.25) == pytest.approx(2.0)
    # the M0 worked example (D, proximal control): 4.53 x 2.34 mm, adjusted
    # load 5.0372 N at lambda 1.219675 -> 57.96 N/cm^2, reported "58.0"
    sigma = mech.cauchy_incompressible_mpa(5.0372, 1.219675, 4.53, 2.34)
    assert mech.mpa_to_n_per_cm2(sigma) == pytest.approx(57.96, abs=0.005)
    assert round(mech.mpa_to_n_per_cm2(sigma), 1) == 58.0


def test_packaged_rows_obey_the_cauchy_identities():
    """Every strip: engineering = Cauchy / (1+e) and tension = Cauchy t0 /
    sqrt(1+e); raw-curve strips: the recomputed Cauchy rounds to the report."""
    rows = list(csv.DictReader(io.StringIO(packaged_text())))
    n_raw = 0
    for r in rows:
        c, e = float(r["failure_cauchy_mpa"]), float(r["failure_strain"])
        # 6 significant digits per stored value -> combined rounding < 2e-5
        assert float(r["failure_engineering_mpa"]) == pytest.approx(c / (1 + e), rel=2e-5)
        t0 = float(r["thickness_mm"])
        assert float(r["failure_tension_n_per_mm"]) == pytest.approx(
            c * t0 / math.sqrt(1 + e), rel=2e-5), r["strip_id"]
        assert r["stress_measure"] == "cauchy_incompressible"
        if r["data_kind"] == "raw_curve":
            n_raw += 1
            assert r["cauchy_recomputed"] == "yes"
            reported = float(r["reported_failure_stress_n_per_cm2"])
            # stored Cauchy has 6 significant digits: <= 5e-4 N/cm^2 of rounding
            assert abs(100 * c - reported) <= 0.05 + 1e-3, r["strip_id"]
            assert float(r["recomputed_minus_reported_n_per_cm2"]) == pytest.approx(
                100 * c - reported, abs=1e-3)
        else:
            assert r["cauchy_recomputed"] == "no"
            assert 100 * c == pytest.approx(float(r["reported_failure_stress_n_per_cm2"]))
    assert n_raw == 21


# -- the packaged library --------------------------------------------------------
def test_packaged_library_counts_and_provenance():
    lib = load_strength_library()
    assert len(lib.records) == 24
    aaa, ctl = lib.select("aaa"), lib.select("control")
    assert (len(aaa), len({r.case for r in aaa})) == (13, 7)
    assert (len(ctl), len({r.case for r in ctl})) == (11, 5)
    assert {r.case for r in lib.records} == set("ABDEFGH")  # case C has no data
    assert {r.strip_id for r in lib.records if r.late_peak} == {
        "E-border-right", "E-control-abd-distal"}
    for r in lib.records:
        if r.late_peak:
            assert r.first_peak_cauchy_mpa < r.failure_cauchy_mpa
            assert r.first_peak_strain < r.failure_strain
        assert r.dataset_doi == DOI and r.licence == "CC BY 4.0"
        assert len(r.source_sha256) == 64 and r.source_file
    chart = [r for r in lib.records if r.data_kind == "chart_only"]
    assert sorted(r.strip_id for r in chart) == ["F-border-right", "G-border-left", "G-border-right"]
    assert lib.dataset_dois == (DOI,) and lib.licences == ("CC BY 4.0",)


def test_csv_round_trip_and_provenance_columns():
    text = packaged_text()
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    for col in REQUIRED_COLUMNS + (
        "site_code", "source_sheet", "source_cells", "source_report", "test_direction",
        "reported_failure_stress_n_per_cm2", "cauchy_recomputed", "thickness_source",
    ):
        assert col in reader.fieldnames, col
    # write it back unchanged -> byte-identical
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=reader.fieldnames, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    assert out.getvalue() == text
    # parsed records carry exactly the cell values
    lib = parse_strength_csv(text, "round-trip")
    for rec, row in zip(lib.records, rows):
        assert rec.strip_id == row["strip_id"]
        assert rec.failure_cauchy_mpa == float(row["failure_cauchy_mpa"])
        assert rec.source_sha256 == row["source_sha256"]
    assert lib.sha256 == hashlib.sha256(text.encode()).hexdigest()
    # no autopsy sample numbers or operator name leaked into the shipped files
    for name in ("strength_v0.csv", "strength_v0_donors.csv"):
        t = packaged_text(name)
        for leak in ("12880", "12875", "10258", "9646", "9666", "Gina", "Sample ID"):
            assert leak not in t, (name, leak)


def test_donor_table_and_attribution_file():
    donors = list(csv.DictReader(io.StringIO(packaged_text("strength_v0_donors.csv"))))
    assert [d["case"] for d in donors] == list("ABCDEFGH")
    assert set(donors[0]) >= {"age_years", "sex", "aneurysm_shape", "max_transverse_diameter_cm"}
    assert {d["aneurysm_shape"] for d in donors} == {"fusiform", "saccular"}
    c = next(d for d in donors if d["case"] == "C")
    assert (c["n_strips_rupture_border"], c["n_strips_control"]) == ("0", "0")
    sources = packaged_text("STRENGTH_V0_SOURCES.md")
    for must in (DOI, "CC BY 4.0", "Gomes VC", "Raghavan ML", "derived", "endorse", '"90+"'):
        assert must in sources, must


def test_no_shipped_age_over_89_unless_top_coded():
    """HIPAA Safe Harbor: an exact age over 89 is an identifier, so the donor
    table writes 90 and over as "90+"; every other age is an integer <= 89."""
    donors = list(csv.DictReader(io.StringIO(packaged_text("strength_v0_donors.csv"))))
    ages = [d["age_years"] for d in donors]
    for age in ages:
        assert age == "90+" or (age.isdigit() and int(age) <= 89), age
    assert "90+" in ages  # the two oldest donors are top-coded, not dropped
    # no exact age >= 90 anywhere else in the shipped text either
    import re

    sources = packaged_text("STRENGTH_V0_SOURCES.md")
    assert not re.search(r"aged[^.\n]*\b(9\d|1\d\d)\b(?!\+)", sources)  # "90+" is fine


# -- refusals ------------------------------------------------------------------------
def test_refuses_an_empty_library(tmp_path):
    with pytest.raises(StrengthLibraryError, match="library is empty"):
        parse_strength_csv(toy_csv([]), "empty")
    with pytest.raises(StrengthLibraryError, match="missing column"):
        parse_strength_csv("strip_id,case\nA-1,A\n", "thin")
    with pytest.raises(FileNotFoundError, match="strength library not found"):
        load_strength_library(tmp_path / "nope.csv")


@pytest.mark.parametrize(
    "field, value",
    [
        ("stress_unit", ""), ("stress_unit", "N/cm^2"), ("stress_unit", "kPa"),
        ("tension_unit", ""), ("length_unit", "cm"),
        ("stress_measure", ""), ("stress_measure", "engineering"),
        ("strain_measure", "true"),
    ],
)
def test_refuses_unknown_or_missing_units_and_measures(field, value):
    rows = [toy_row("X-a", "X", 0.5), toy_row("X-b", "X", 1.0, **{field: value})]
    with pytest.raises(StrengthLibraryError, match=rf"line 3 \(X-b\): {field}"):
        parse_strength_csv(toy_csv(rows), "bad")


@pytest.mark.parametrize(
    "over, match",
    [
        ({"failure_cauchy_mpa": ""}, "failure_cauchy_mpa is empty"),
        ({"failure_cauchy_mpa": "-0.5"}, "finite number > 0"),
        ({"failure_cauchy_mpa": "abc"}, "not a number"),
        ({"failure_engineering_mpa": "0.5"}, "stress measures disagree"),
        ({"strip_id": "X-a"}, "duplicated"),
        ({"dataset_doi": ""}, "provenance is required"),
        ({"late_peak": "yes"}, "first-peak values"),
        ({"location_class": "thoracic"}, "location_class"),
    ],
)
def test_refuses_bad_rows(over, match):
    over = dict(over)
    sid = over.pop("strip_id", "X-b")
    rows = [toy_row("X-a", "X", 0.5), toy_row(sid, "X", 1.0, **over)]
    with pytest.raises(StrengthLibraryError, match=match):
        parse_strength_csv(toy_csv(rows), "bad")


def test_refuses_bad_pws():
    for bad in (0, -0.2, float("nan"), float("inf"), True, "0.3", None):
        with pytest.raises(ValueError, match="peak wall stress"):
            check_pws(bad)
    with pytest.raises(ValueError, match="kPa or Pa"):
        check_pws(300.0)  # a kPa number passed as MPa
    assert check_pws(0.3) == 0.3


def test_refuses_too_few_strips_and_unknown_options(toy3):
    with pytest.raises(ValueError, match="too few"):
        rupture_index_band(0.3, toy3, selection="control")
    with pytest.raises(ValueError, match="unknown selection"):
        rupture_index_band(0.3, toy3, selection="thoracic")
    with pytest.raises(ValueError, match="unknown band method"):
        rupture_index_band(0.3, toy3, method="bootstrap")
    with pytest.raises(ValueError, match="unknown stress measure"):
        rupture_index_band(0.3, toy3, measure="piola")


# -- the band ------------------------------------------------------------------------
def test_known_ratio_exact_quantiles(toy3):
    """PWS 0.3 MPa vs 0.5 / 1.0 / 1.5 MPa: strength P10 = 0.5 + 0.2*0.5 = 0.6,
    P50 = 1.0, P90 = 1.0 + 0.8*0.5 = 1.4; index = 0.3 / (1.4, 1.0, 0.6)."""
    b = rupture_index_band(0.3, toy3)
    assert (b["n_strips"], b["n_aneurysms"], b["method"]) == (3, 2, "empirical")
    assert b["strength_mpa"] == pytest.approx({"p10": 0.6, "p50": 1.0, "p90": 1.4}, abs=1e-12)
    assert b["index"] == pytest.approx(
        {"p10": 0.3 / 1.4, "p50": 0.3, "p90": 0.5}, abs=1e-12)
    assert b["caveat"] == CAVEAT
    # lognormal: geometric mean at P50, sd of logs with ddof=1, z(0.9)
    logs = [math.log(v) for v in (0.5, 1.0, 1.5)]
    mu = sum(logs) / 3
    sd = math.sqrt(sum((x - mu) ** 2 for x in logs) / 2)
    z = 1.2815515655446004
    ln = rupture_index_band(0.3, toy3, method="lognormal")
    assert ln["strength_mpa"] == pytest.approx(
        {"p10": math.exp(mu - z * sd), "p50": math.exp(mu), "p90": math.exp(mu + z * sd)},
        rel=1e-12)
    assert ln["index"]["p50"] == pytest.approx(0.3 / 0.75 ** (1 / 3), rel=1e-12)


@pytest.mark.parametrize("method", ["empirical", "lognormal"])
def test_quantile_direction_index_p90_is_the_weak_wall(toy3, method):
    b = rupture_index_band(0.3, toy3, method=method)
    s, ix = b["strength_mpa"], b["index"]
    assert s["p10"] < s["p50"] < s["p90"]
    assert ix["p10"] < ix["p50"] < ix["p90"]
    assert ix["p90"] == pytest.approx(0.3 / s["p10"], rel=1e-15)
    assert ix["p10"] == pytest.approx(0.3 / s["p90"], rel=1e-15)
    # the high-risk end: above PWS / the median strength
    assert ix["p90"] > 0.3 / s["p50"]
    # a weaker weakest strip raises the high-risk end
    weaker = parse_strength_csv(toy_csv([
        toy_row("X-a", "X", 0.25), toy_row("X-b", "X", 1.0), toy_row("Y-a", "Y", 1.5)]), "w")
    w = rupture_index_band(0.3, weaker, method=method)
    assert w["index"]["p90"] > ix["p90"]
    if method == "empirical":  # P10 index rides on the two strongest strips only
        assert w["index"]["p10"] == pytest.approx(ix["p10"], rel=1e-15)


def test_index_scales_with_pws(toy3):
    a, b = rupture_index_band(0.2, toy3), rupture_index_band(0.4, toy3)
    for q in ("p10", "p50", "p90"):
        assert b["index"][q] == pytest.approx(2 * a["index"][q], rel=1e-14)


def test_packaged_aaa_band_regression():
    """Library v0 is frozen: pin the AAA (rupture-border) Cauchy band."""
    b = rupture_index_band(0.3)
    assert (b["n_strips"], b["n_aneurysms"]) == (13, 7)
    assert b["strength_mpa"] == pytest.approx(
        {"p10": 0.3081114, "p50": 0.6503, "p90": 1.299236}, rel=1e-6)
    assert b["late_peak_strips"] == ["E-border-right"]
    # engineering strength is lower -> a higher index (the mixed-measure trap)
    e = rupture_index_band(0.3, measure="engineering")
    assert all(e["index"][q] > b["index"][q] for q in ("p10", "p50", "p90"))
    # raw-curve strips only: 10 strips from 5 aneurysms
    raw = rupture_index_band(0.3, include_chart_only=False)
    assert (raw["n_strips"], raw["n_aneurysms"]) == (10, 5)
    # all: 24 strips, 7 aneurysms
    allb = rupture_index_band(0.3, selection="all")
    assert (allb["n_strips"], allb["n_aneurysms"]) == (24, 7)


def test_report_text_shows_n_and_the_caveat():
    rep = index_band_report(0.25)
    txt = index_band_text(rep, pws_source="placeholder")
    assert "open strength data : 7 aneurysms, 13 strips" in txt
    assert "our lab will narrow this" in txt
    assert CAVEAT in txt
    assert DOI in txt and "CC BY 4.0" in txt
    assert set(rep["sensitivity"]) == {
        "method_lognormal", "measure_engineering", "late_peak_reported", "raw_curve_only"}
    assert rep["comparison_control"]["n_strips"] == 11
    assert "comparison_control" not in index_band_report(0.25, selection="control")


# -- wall-stress summary input -----------------------------------------------------
def test_pws_from_summary_is_tolerant_and_loud(tmp_path):
    def write(obj, name="s.json"):
        p = tmp_path / name
        p.write_text(json.dumps(obj))
        return p

    top = write({"pws_mpa": 0.31, "other": 1})
    before = hashlib.sha256(top.read_bytes()).hexdigest()
    v, where, digest = pws_from_summary(top)
    assert (v, where, digest) == (0.31, "pws_mpa", before)
    assert hashlib.sha256(top.read_bytes()).hexdigest() == before  # read only
    nested = write({"case": "x", "results": {"wall": {"pws_mpa": 0.27}}}, "n.json")
    assert pws_from_summary(nested)[:2] == (0.27, "results.wall.pws_mpa")
    same = write({"pws_mpa": 0.2, "runs": [{"pws_mpa": 0.2}]}, "same.json")
    assert pws_from_summary(same)[0] == 0.2
    with pytest.raises(ValueError, match="no 'pws_mpa' key"):
        pws_from_summary(write({"pws": 0.3, "peak_wall_stress": 0.3}, "none.json"))
    with pytest.raises(ValueError, match="different pws_mpa values"):
        pws_from_summary(write({"a": {"pws_mpa": 0.2}, "b": {"pws_mpa": 0.3}}, "two.json"))
    with pytest.raises(ValueError, match="peak wall stress"):
        pws_from_summary(write({"pws_mpa": "0.3"}, "str.json"))
    with pytest.raises(ValueError, match="kPa or Pa"):
        pws_from_summary(write({"pws_mpa": 310.0}, "kpa.json"))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        pws_from_summary(bad)
    with pytest.raises(FileNotFoundError, match="summary not found"):
        pws_from_summary(tmp_path / "missing.json")


# -- pipeline step, work file, CLI, web -------------------------------------------
def test_session_tissue_index_step_journal_and_replay(tmp_path):
    from meshforge.pipeline import STEP_KNOBS, STEP_ORDER, MeshingSession, load_workfile, replay

    # solve -> aaa_stress -> tissue_index: the index reads the wall-stress PWS
    assert STEP_ORDER.index("tissue_index") == STEP_ORDER.index("aaa_stress") + 1
    assert set(STEP_KNOBS["tissue_index"]) == {"pws_mpa", "from_summary", "selection", "method"}
    s = MeshingSession()
    with pytest.raises(ValueError, match="exactly one peak wall stress source"):
        s.compute_tissue_index()
    summ = tmp_path / "aaa_stress.json"
    summ.write_text(json.dumps({"pws_mpa": 0.3}))
    with pytest.raises(ValueError, match="exactly one peak wall stress source"):
        s.compute_tissue_index(pws_mpa=0.3, from_summary=str(summ))
    r = s.compute_tissue_index(pws_mpa=0.3)
    assert (r.summary["n_strips"], r.summary["n_aneurysms"]) == (13, 7)
    assert "open strength data : 7 aneurysms, 13 strips" in r.text and CAVEAT in r.text
    assert s.surface is None  # needs no geometry, changes none
    r2 = s.compute_tissue_index(from_summary=f'"{summ}"', method="lognormal")
    assert r2.summary["pws_mpa"] == 0.3 and r2.summary["method"] == "lognormal"
    rec = s.journal[-1]
    assert rec["step"] == "tissue_index" and rec["params"]["from_summary"] == str(summ)
    assert rec["input_sha256"] == hashlib.sha256(summ.read_bytes()).hexdigest()
    wf = s.save_workfile(tmp_path / "t.mfwork")
    out = replay(load_workfile(wf))
    assert [x.step for x in out] == ["tissue_index", "tissue_index"]
    assert out[0].summary["index"] == r.summary["index"]
    assert out[1].summary["index"] == r2.summary["index"]


def test_cli_tissue_index(tmp_path, capsys):
    from meshforge.cli import main

    out = tmp_path / "m4" / "band.json"
    assert main(["tissue-index", "--pws", "0.3", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "open strength data : 7 aneurysms, 13 strips" in printed and CAVEAT in printed
    saved = json.loads(out.read_text())
    assert saved["index"]["p90"] == pytest.approx(0.3 / saved["strength_mpa"]["p10"])
    summ = tmp_path / "aaa_stress.json"
    summ.write_text(json.dumps({"case": "demo", "pws_mpa": 0.25}))
    assert main(["tissue-index", "--from-summary", str(summ), "--selection", "all"]) == 0
    assert "0.250 MPa (Cauchy; aaa_stress.json: pws_mpa)" in capsys.readouterr().out
    summ.write_text(json.dumps({"case": "demo"}))
    assert main(["tissue-index", "--from-summary", str(summ)]) == 1
    assert "no 'pws_mpa' key" in capsys.readouterr().err
    summ.write_text(json.dumps({"pws_mpa": 0.25}))
    assert main(["tissue-index", "--from-summary", str(summ), "--out", str(summ)]) == 1
    assert "refusing to overwrite input file" in capsys.readouterr().err
    assert main(["tissue-index", "--pws", "250"]) == 1
    with pytest.raises(SystemExit):  # exactly one source
        main(["tissue-index", "--pws", "0.3", "--from-summary", str(summ)])


def test_web_step_tissue_index():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from meshforge.web.app import create_app

    client = TestClient(create_app())
    r = client.post("/step/tissue_index", json={"pws_mpa": 0.3})
    assert r.status_code == 200, r.text
    assert r.json()["summary"]["n_aneurysms"] == 7
    r = client.post("/step/tissue_index", json={})
    assert r.status_code == 400 and "exactly one" in r.json()["detail"]
