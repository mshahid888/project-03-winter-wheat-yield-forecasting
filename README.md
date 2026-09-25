# Predicting Winter Wheat Yield Across German Federal States with DWD Climate Data

A small, honest machine-learning project: can winter-wheat yield at the
Bundesland (federal state) level in Germany be forecast from last year's yield
plus climate information that is genuinely available *before* harvest?

**Short answer:** not measurably. Neither trained model improves on the
persistence baseline on the 2020–2025 holdout: Ridge lands essentially on top
of it and Random Forest a little behind. Significance tests reported below
indicate that none of the gaps is distinguishable from noise on the 78 test
points, but those tests were run outside this repository and cannot be
reproduced from its code (see *Results* and *What is and isn't reproducible*).
The useful finding is the absence of a demonstrated gain, and this project
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
| Ridge (α=100)¹ | 0.5607 | 0.6899 | 0.2922 |
| Random Forest (500 trees)¹ | 0.5954 | 0.7571 | 0.1477 |

¹ Hyperparameters as documented in `methodology.md` §3.3. `scripts/analyze.py`
selects them by year-based cross-validation but only prints its choice; it is
not saved to any output, so the selection itself is not recorded in the
repository. The notebook refits these hard-coded values and reproduces the
table above to four decimals, which is consistent with, but does not prove,
that selection.

Interpretation: with only state-level seasonal aggregates and one lag, the
climate signal is not strong enough to improve on "last year". The regularised
linear model stays close to the dominant lag signal (α=100 is the largest value
in the searched grid) without beating it, and the tested Random Forest
configuration scores a little worse on these point estimates.

**How firm is that?** Not very. The differences in MAE quoted here are exact
differences of the table above. The confidence intervals and p-values are
**not** produced by any code in this repository: they come from an analysis
run outside it that was not committed, and they cannot currently be reproduced
from the repository, because the pipeline does not save the per-observation
predictions they require. With that caveat, the reported paired difference in
absolute error between persistence and Ridge is −0.0053 t/ha (95% bootstrap CI
−0.094 to +0.081, paired *t* p=0.91), and between persistence and Random Forest
−0.0400 t/ha (CI −0.136 to +0.052, p=0.41). On those reported tests, the ranking
above is a point estimate that the data cannot separate. Neither trained model
demonstrably improves on the persistence baseline on the 2020–2025 holdout, and
it would be overclaiming in the other direction to say the trained models are
*demonstrably worse*.

2025 is the hardest test year: it has the highest MAE of the six for all three
models (`outputs/tables/per_year_mae.csv`). The persistence baseline's two
largest test errors are Nordrhein-Westfalen 2025 (1.99 t/ha) and Niedersachsen
2025 (1.45 t/ha), i.e. their year-on-year yield increases; the pipeline does not
save per-state predictions for Ridge or the Random Forest. The 2025 spring was
unusually sunny and dry (spring sunshine ranked 1st of 26 years nationally,
precipitation 26th of 26), and yields rose over 2024 by roughly +0.9 to
+2.0 t/ha in Rheinland-Pfalz, Niedersachsen and Nordrhein-Westfalen. These were
not record yields: every state finished 2025 below its own historical maximum
(Rheinland-Pfalz came closest at 98.3% of its record, Nordrhein-Westfalen at
97.4%). A lag-anchored model cannot anticipate a jump like that. The figures in
this paragraph that are not in `outputs/tables/` were derived by hand from the
committed data, not by a pipeline script (see below).

## What is and isn't reproducible

**Reference platform.** The reference results were generated and verified on
Linux x86_64, where the committed Random Forest result is MAE 0.5954,
RMSE 0.7571, R² 0.1477. Apple Silicon (ARM64) may produce slightly different
numerical Random Forest values.

Every result in this repository falls into one of three categories.

**A. Generated by code in this repository**

- `data/processed/wheat_yield_panel.csv` and the leakage audit
  (`scripts/build_dataset.py`). Re-run on 2026-09-25 with Python 3.14.7,
  pandas 3.0.6 and NumPy 2.5.3: the output was byte-identical to the committed
  file.
- Every file in `outputs/tables/` and `outputs/figures/` (`scripts/analyze.py`):
  model metrics, per-state and per-year MAE, permutation importance,
  correlations. **Not yet re-run:** no environment with scikit-learn was
  available on 2026-09-25, so reproducing these files has not been demonstrated.
- The hyperparameter selection. `analyze.py` performs it, but only prints the
  result; it is not saved to any output.
- The notebook (`scripts/make_notebook.py`), including the per-state 2025
  yield changes and spring-sunshine standard scores it prints. Its markdown text
  (and two code comments) were corrected on 2026-09-25 without re-executing it;
  all saved code outputs are from the original execution.

**B. Derived by hand from committed files, not by a pipeline script**

These can be checked against the committed data, and were spot-checked on
2026-09-25, but no script in the repository produces them:

- the differences in MAE between models (differences of
  `outputs/tables/model_comparison.csv`);
- the 2025 national spring sunshine and precipitation ranks (the `Deutschland`
  column of the committed DWD files), each state's 2025 yield relative to its
  own record, and the national year-on-year jumps quoted in `methodology.md`;
- the leakage-injection check described in `methodology.md` §2.4 (a manual
  test, not a committed one).

**C. Not reproducible from this repository**

- The bootstrap confidence intervals, paired *t*-tests, Wilcoxon tests and
  annual-mean p-values (this README, `methodology.md` §4, `limitations.md` §7).
  They come from an analysis outside the repository that was not committed. No
  code here computes them, scipy is not a dependency, and the per-observation
  predictions they require are not saved. They are reported as documented
  results only.

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
pipeline as a teaching document. Its markdown text was later corrected without
re-execution (see *What is and isn't reproducible*); re-running
`make_notebook.py` regenerates exactly the corrected text.

### Dependencies and versions

`requirements.txt` sets only minimum versions (pandas ≥ 2.0, numpy ≥ 1.24,
matplotlib ≥ 3.6, scikit-learn ≥ 1.4) and leaves jupyter, nbformat and
nbclient unpinned. The notebook also relies on ipykernel and IPython, which
jupyter installs. scikit-learn 1.4 is the minimum the code actually needs:
`analyze.py` imports `sklearn.metrics.root_mean_squared_error`, which was added
in scikit-learn 1.4; every other scikit-learn API used is older. No upper
bounds are set. scipy is not used by any script and is not listed.

**The exact versions that produced the committed results are not recorded in
this repository**, so no exact pins are given. The only recorded interpreter
version is Python 3.14.0, in the executed notebook's metadata. Pins should be
added only from an environment that has actually reproduced the committed
tables.

### Continuous integration

`.github/workflows/reproduce.yml` runs on every push and pull request. It
installs `requirements.txt` on Python 3.14, records the versions pip resolved
(uploaded as a build artifact together with the regenerated outputs), then runs
the three scripts above. The existing checks run as part of the pipeline: the
leakage audit and row-count assertions in `build_dataset.py`, the
chronological CV-fold assertions in `analyze.py`, and end-to-end notebook
execution in `make_notebook.py`. Finally it requires the regenerated
`data/processed/` panel and every table in `outputs/tables/` to be
**byte-identical** to the committed files.

Limits of this check, stated rather than worked around:

- Because dependency versions are not pinned, CI resolves current releases. If
  the comparison fails, the committed results may depend on library versions
  that were not recorded, rather than on a code change. The comparison is kept
  strict on purpose; a failure should be investigated, not loosened.
- `outputs/tables/permutation_importance.csv` stores unrounded floats,
  including values at floating-point noise level (for example `year` for
  Random Forest, 5.551115123125783e-17, which is exactly 2⁻⁵⁴). Its true value
  is zero, because shuffling `year` cannot change any Random Forest test
  prediction. These values are the least likely to match bit-for-bit across
  library versions or platforms, and possibly between runs: `analyze.py` uses
  `n_jobs=-1`, and parallel prediction may sum per-tree outputs in a varying
  order. That is a hypothesis, not yet observed. If this file alone fails the
  comparison with differences at this scale, the difference is numerical
  representation, not a scientific change; deciding how to handle it is left
  until it has actually been observed.
- **The workflow has not been executed yet**, so whether it passes is unknown.
- Figures and the notebook are regenerated but not compared byte-for-byte,
  since image encoding and execution metadata vary between library versions.
- The significance tests reported in `methodology.md`, `limitations.md` and
  above (bootstrap CIs, paired *t*, Wilcoxon, annual-mean tests) are **not**
  computed by any script in this repository, so CI cannot check them.

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
  produced by this pipeline are under the MIT licence (see `LICENSE`). The
  licence scope and the third-party data terms are in `DATA_LICENSES.md`.

## Project layout

```
project-03-winter-wheat-yield-forecasting/
├── README.md  methodology.md  limitations.md  data_sources.md
├── LICENSE    DATA_LICENSES.md    requirements.txt
├── scripts/            # build_dataset.py, analyze.py, make_notebook.py
├── data/raw/           # yield CSV + 36 DWD monthly files (as downloaded)
├── data/processed/     # wheat_yield_panel.csv (338 rows)
├── notebooks/          # executed teaching notebook
├── outputs/figures/    # 7 figures
├── outputs/tables/     # model comparison, per-state/year errors, importances
└── .github/workflows/  # reproduce.yml: CI re-run of the full pipeline
```

## Honesty notes

- No current-year production or harvested area is used — those would leak the answer.
- The CV folds are year-based and expanding, never shuffled; the test set is the
  last six calendar years.
- Negative result reported as-is: Neither trained model demonstrably improves on the persistence baseline on the 2020–2025 holdout.
  According to the externally run significance tests (paired-test p between
  0.33 and 0.91; not reproducible from this repository, see *What is and isn't
  reproducible*), the differences between all three models are within noise on
  this holdout, so no model is claimed to be better *or* demonstrably worse than
  another.
- `year` is a feature, and the test years (2020–2025) lie outside the training
  range (2000–2019), so it is an extrapolated term. Its measured permutation
  importance for Random Forest is zero up to floating-point rounding on the test
  set (stored as 5.55e-17), because every test year falls beyond the last split
  threshold the trees learned. Documented in
  `limitations.md` rather than quietly dropped.
