# Tissue strength library v0: sources and attribution

`strength_v0.csv` (one row per tensile strip) and `strength_v0_donors.csv`
(one row per aneurysm) are **derived from**:

> Gomes VC, da Silva ES, da Silva LF, Gomes J, Raghavan ML, Silvestre G,
> Queiroz A, Marques M, Zyngier SP, Chung TK-J. *Dataset on cadaveric human
> abdominal aorta aneurysm biomechanics (uniaxial) and histology.* Mendeley
> Data, version 3 (2020). doi:[10.17632/x64srrc39p.3](https://doi.org/10.17632/x64srrc39p.3)

Licence: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## Licence of these two files

They are CC BY 4.0 derived data, redistributed here with attribution as that
licence allows. The Apache-2.0 licence of the meshforge code does **not**
apply to them. If you reuse them, credit the dataset authors as above, link
the licence, and say that the values were changed (see below). Neither the
dataset authors nor the Universidade de São Paulo endorse meshforge or this use.

## What was changed

The dataset ships one Excel file, one test report and one chart per strip.
`scripts/build_strength_v0.py` turned them into a table:

- Stress converted from N/cm² to MPa (÷ 100) and tension from N/cm to N/mm (÷ 10).
- **Cauchy failure stress** recomputed for the 21 strips with raw curves as
  F·λ / (w₀·t₀) from the adjusted load, the displacement and the entered
  dimensions (λ = 1 + displacement / gauge length). This is the dataset's own
  definition, so the reported "Failure Stress" is a Cauchy-type stress under
  incompressibility, not engineering stress. Every recomputed value rounds to
  the reported one (largest difference 0.049 N/cm²; the reports round to 0.1).
- **Engineering failure stress** added as Cauchy / (1 + failure strain).
- Failure point = the maximum Cauchy stress; strain and tension are taken at
  that same data point. The dataset's own failure tension (the maximum
  tension, sometimes at another point) is kept as `reported_failure_tension_n_per_cm`.
- **Late peaks flagged.** In two case-E strips the maximum stress comes after
  the load had peaked and the stress had dropped, so it is not the first
  failure. Their first-failure (load-peak) values are added in `first_peak_*`.
- Three strips (case F ×1, case G ×2) exist only as a chart image. Their
  failure stress, tension and strain were transcribed from the text printed
  on the chart. Their thickness is derived as t₀ = T·√(1+ε) / σ. Each row
  records the sha256 of the image it came from.
- File-name site codes were translated (`borda` = rupture border;
  `dir`/`esq`/`ant`/`post` = right/left/anterior/posterior; `aoinfra` =
  infrarenal aorta). The `abd…` code used in cases E and H is not defined in
  the dataset's abbreviations document. Those strips count as controls.
- The donors' autopsy sample numbers and the test operator's name (both in
  the file headers) are **not** carried over. The donor table keeps only what
  the dataset lists per case: age, sex, aneurysm shape and largest transverse
  diameter.
- Donor ages of 90 and over are top-coded as "90+" (HIPAA Safe Harbor groups
  ages over 89); younger ages are as listed.

## How it was checked

All 71 dataset files were checked against the sha256 list recorded at
download, and every file was accounted for. The parser stops with a message
if any file, unit string, column header or column identity differs from what
it expects. It checks Load (N) = 1000 × Load (kN), strain = displacement /
gauge length, both stress columns, and the tension column, which holds N/mm
despite its N/cm label (10× off). Each test report must match its Excel file
in dimensions, point count and every data point. The donor sheet must match
the histology sheet. The library must hold 24 strips (13 rupture-border, 11
control, 21 with raw curves) from cases A, B, D–H, with exactly the two known
late-peak strips. `--check` rebuilds the library and confirms it is
byte-identical to these files.

## Known limits (read before using the numbers)

- **Small n.** 7 aneurysms have data (case C has none): 13 rupture-border
  strips from 7 aneurysms and 11 control strips from 5. Strips from the same
  aneurysm are not independent.
- Autopsy tissue from donors aged 73 to 90+. Each aorta had been inflated with a
  balloon until it ruptured before the strips were cut, and the border strips
  come from the edge of that rupture.
- The "controls" are non-dilated infrarenal segments of the same aneurysmal
  aortas, not healthy aortas.
- Test direction (circumferential or longitudinal) is not recorded anywhere.
- Thickness was entered per strip. The dataset does not say how it was
  measured or whether it includes thrombus.
- Strain is grip-to-grip (extensometer disabled). Six strips have a short
  9.5–11.8 mm gauge length.
- "Adjusted load" is the raw load minus an offset of 0.006–0.072 N (at most
  0.9 % of the load at failure). The method behind the offset is not documented.
- Two aneurysms (G, H) are saccular, with a largest diameter of 2.7–2.8 cm.
