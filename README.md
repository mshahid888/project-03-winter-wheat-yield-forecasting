# Predicting Winter Wheat Yield Across German Federal States with DWD Climate Data

A small, honest machine-learning project: can winter-wheat yield at the
Bundesland (federal state) level in Germany be forecast from last year's yield
plus climate information that is genuinely available *before* harvest?

**Short answer:** not measurably. Neither trained model demonstrably improves on the persistence baseline on the 2020–2025 holdout. Ridge lands
essentially on top of it and Random Forest a little behind, but on 78 test
points none of the gaps is distinguishable from noise (see *Results*). The
useful finding is the absence of a demonstrable gain, and this project
documents that instead of hiding it.

## Research question

> Can Bundesland-level winter-wheat yield (t/ha) be forecast for year Y using
> only information available by 30 June of year Y?

## Data

| Source | What | Coverage |
|---|---|---|
| Regionaldatenbank Deutschland, table **41241-01-03-4-B** ("Erträge ausgewählter landwirtschaftlicher Feldfrüchte – Jahressumme – regionale Ebenen") | Winter-wheat yield (dt/ha), 13 Bundesländer | 1999–2025 (Berlin, Hamburg, Bremen have no reported winter-wheat yield and are excluded) |
| DWD Climate Data Center, `regional_averages_DE/monthly/` | Monthly mean temperature, precipitation, sunshine duration per Bundesland | 1999–2025 |

Full provenance, download procedure, and license notes: [`data_sources.md`](data_sources.md).

## Method (one paragraph)

The modeling panel has 338 rows (13 states × 26 years, 2000–2025 after building
the lag). Seven features per state-year: harvest `year` (a linear trend term),
previous year's yield, spring (Mar–Jun of the harvest year) temperature /
precipitation / sunshine, and winter (Oct–Dec of the previous year + Jan–Feb)
temperature / precipitation. State identity is *not* a feature. The forecast
cutoff is
**30 June of the harvest year** — no July/August climate, no current-year
production or area figures are used. Hyperparameters are tuned with expanding
year-based cross-validation (never shuffled), and final performance is measured
on a strictly chronological holdout: train 2000–2019, test 2020–2025.
Models: persistence baseline (predict last year's yield), Ridge regression,
Random Forest. See [`methodology.md`](methodology.md) for the full design and
[`limitations.md`](limitations.md) for what this cannot do.

## Results (test set 2020–2025, 78 state-years)

| Model | MAE (t/ha) | RMSE (t/ha) | R² |
|---|---|---|---|
| Persistence baseline | 0.5554 | 0.6765 | 0.3195 |
| Ridge (α=100) | 0.5607 | 0.6899 | 0.2922 |
| Random Forest (500 trees) | 0.5954 | 0.7571 | 0.1477 |

Interpretation: with only state-level seasonal aggregates and one lag, the
climate signal is not strong enough to improve on "last year". The regularised
linear model stays close to the dominant lag signal (CV picked the heaviest
penalty offered, α=100) without beating it, and the tree ensemble does a little
worse.

**How firm is that?** Not very — and the honest framing matters here. On the 78
test points, the paired difference in absolute error between persistence and
Ridge is −0.0053 t/ha (95% bootstrap CI −0.094 to +0.081, paired *t* p=0.91) and
between persistence and Random Forest −0.0400 t/ha (CI −0.136 to +0.052,
p=0.41). So the ranking above is the point estimate, but the data cannot
actually separate the three models. Neither trained model demonstrably improves on the persistence baseline on the 2020–2025 holdout. It would
be overclaiming in the other direction to say the trained models are
*demonstrably worse*.

The largest errors occur in 2025 in the north-west — Nordrhein-Westfalen,
Rheinland-Pfalz and Niedersachsen — where an unusually sunny, dry spring
(spring sunshine ranked 1st of 26 years nationally, precipitation 26th of 26)
coincided with unusually large year-on-year yield increases of roughly +0.9 to
+2.0 t/ha over 2024. These were not record yields: every state finished 2025
below its own historical maximum (Nordrhein-Westfalen came closest at 97.4% of
its record). A lag-anchored model cannot anticipate a jump like that.

## Reproduce

Run all commands **from the repository root** — every script uses relative
paths (`data/`, `outputs/`, `notebooks/`) and will fail elsewhere:

```bash
cd project-03-winter-wheat-yield-forecasting
pip install -r requirements.txt
python3 scripts/build_dataset.py   # builds data/processed/wheat_yield_panel.csv
python3 scripts/analyze.py         # EDA + models -> outputs/figures, outputs/tables
python3 scripts/make_notebook.py   # regenerates AND executes
                                   # notebooks/wheat_yield_forecasting.ipynb
```

The notebook [`notebooks/wheat_yield_forecasting.ipynb`](notebooks/wheat_yield_forecasting.ipynb)
is generated *and executed end-to-end* by `make_notebook.py` (via `nbclient`,
so the saved outputs are a genuine verified run) and walks through the whole
pipeline as a teaching document.

## License & attribution

- **Yield data:** Regionaldatenbank Deutschland (Statistische Ämter des Bundes
  und der Länder), table 41241-01-03-4-B, Erntestatistik. Provided under
  [Datenlizenz Deutschland – Namensnennung – Version 2.0](https://www.govdata.de/dl-de/by-2-0)
  — attribution required: © Statistische Ämter des Bundes und der Länder, Deutschland.
- **Climate data:** Deutscher Wetterdienst (DWD), Climate Data Center
  `regional_averages_DE/monthly/`. Provided under **CC BY 4.0** (DWD CDC Terms
  of Use, May 2024; see https://opendata.dwd.de/climate_environment/CDC/Terms_of_use.pdf)
  — redistribution is permitted with attribution to the Deutscher Wetterdienst
  as the source. The 36 raw monthly files in `data/raw/` are redistributed
  here under that licence; exact retrieval instructions are documented in
  [`data_sources.md`](data_sources.md).
- **Code and text in this repository:** all scripts, docs, figures and tables
  produced by this pipeline.

## Project layout

```
project-03-winter-wheat-yield-forecasting/
├── README.md  methodology.md  limitations.md  data_sources.md
├── LICENSE    requirements.txt
├── scripts/            # build_dataset.py, analyze.py, make_notebook.py
├── data/raw/           # yield CSV + 36 DWD monthly files (as downloaded)
├── data/processed/     # wheat_yield_panel.csv (338 rows)
├── notebooks/          # executed teaching notebook
├── outputs/figures/    # 7 figures
├── outputs/tables/     # model comparison, per-state/year errors, importances
└── docs/               # interview-understanding.md, cv-entry.md
```

## Honesty notes

- No current-year production or harvested area is used — those would leak the answer.
- The CV folds are year-based and expanding, never shuffled; the test set is the
  last six calendar years.
- Negative result reported as-is: Neither trained model demonstrably improves on the persistence baseline on the 2020–2025 holdout.
  The differences between all three models are within noise on this holdout
  (paired-test p between 0.33 and 0.91), so no model is claimed to be better
  *or* demonstrably worse than another.
- `year` is a feature, and the test years (2020–2025) lie outside the training
  range (2000–2019), so it is an extrapolated term. Its measured permutation
  importance for Random Forest is exactly 0 on the test set, because every test
  year falls beyond the last split threshold the trees learned. Documented in
  `limitations.md` rather than quietly dropped.
