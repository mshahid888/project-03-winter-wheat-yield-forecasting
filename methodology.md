# Methodology

## 1. Research question

Can Bundesland-level winter-wheat yield (t/ha) in Germany be forecast for
harvest year Y using only information genuinely available by **30 June of
year Y**?

The fixed forecast date matters: any feature computed from data after
30 June would be cheating, because a real forecaster could not have seen it.

## 2. Data

### 2.1 Yield (target)

- **Source:** Regionaldatenbank Deutschland, table **41241-01-03-4-B**,
  "Erträge ausgewählter landwirtschaftlicher Feldfrüchte – Jahressumme –
  regionale Ebenen" (yields of selected agricultural crops, annual totals).
- **Measure:** winter-wheat yield in **dt/ha** (Dezitonne per hectare),
  converted to **t/ha** by dividing by 10 (1 dt = 100 kg).
- **Coverage:** 16 Bundesländer × 27 years (1999–2025), 432 rows, complete —
  every cell numeric. Berlin, Hamburg and Bremen have no reported winter-wheat
  yield (city-states with negligible production), so the modeling panel uses
  the remaining **13 states**.
- No harvested-area or production-quantity measure is available in this table,
  so no area-based features are used. See `data_sources.md` for acquisition
  and license details.

### 2.2 Climate (features)

- **Source:** DWD Climate Data Center, `regional_averages_DE/monthly/`
  (opendata.dwd.de), files `dwd_tm_MM.txt` (mean temperature),
  `dwd_rr_MM.txt` (precipitation), `dwd_sd_MM.txt` (sunshine duration),
  MM = 01–12. All 36 files downloaded 2026-09-21; headers report creation
  2026-09-02.
- Each file carries per-Bundesland monthly series; files use semicolon
  delimiters and decimal points (the yield CSV uses semicolons with comma
  decimals and latin-1 encoding — both parsed explicitly in
  `scripts/build_dataset.py`).

### 2.3 Feature engineering

For each state and harvest year Y (modeling years 2000–2025, since 1999 is
needed for the lag):

| Feature | Definition |
|---|---|
| `year` | Harvest year Y (captures the long-run trend) |
| `yield_lag1_t_ha` | Winter-wheat yield of year Y−1, t/ha |
| `temp_spring_C` | Mean temperature, Mar–Jun of Y |
| `precip_spring_mm` | Total precipitation, Mar–Jun of Y |
| `sun_spring_h` | Total sunshine hours, Mar–Jun of Y |
| `temp_winter_C` | Mean temperature, Oct(Y−1)–Dec(Y−1) + Jan(Y)–Feb(Y) |
| `precip_winter_mm` | Total precipitation, same winter window |

The **spring window ends in June** — the forecast cutoff. The winter window
covers the vernalization/establishment period of winter wheat. No July or
August climate is used (grain filling happens then, but the data would not
exist yet on 30 June).

### 2.4 Leakage audit

`scripts/build_dataset.py` asserts after every build:

1. No climate month later than June of the harvest year enters any feature.
2. No column whose name suggests production, area, or harvest quantity exists.
3. The only yield-derived feature is the lag-1 value.

The script prints the latest climate month used and fails loudly if any
assertion breaks.

## 3. Modeling

### 3.1 Panel structure

338 rows = 13 states × 26 years (2000–2025). No missing values.

### 3.2 Validation design

- **Outer evaluation:** strict chronological split. Train on 2000–2019
  (260 rows), test on 2020–2025 (78 rows). The test years are never touched
  during tuning.
- **Hyperparameter tuning:** expanding, year-based cross-validation inside the
  training period only: train 2000–2004 → validate 2005–2007; train through
  2007 → validate 2008–2010; … through 2016 → validate 2017–2019. Folds are
  constructed from year masks, never from shuffled row positions, so no
  future year ever appears in a training fold.
- Ridge inputs are standardized with the scaler refit inside every training
  fold — implemented as a `StandardScaler → Ridge` scikit-learn `Pipeline`
  passed directly into `GridSearchCV`, so no scaling statistics leak across
  folds; Random Forest uses raw features.

### 3.3 Models

1. **Persistence baseline** — predict `yield_lag1_t_ha`. Any model must beat
   this to justify its existence.
2. **Ridge regression** — linear model with L2 penalty; grid over
   α ∈ {0.1, 1, 10, 100}; selected α = 100.
3. **Random Forest** — grid over n_estimators ∈ {200, 500},
   max_depth ∈ {None, 6, 10}, min_samples_leaf ∈ {1, 4};
   selected: 500 trees, unrestricted depth, leaf size 1.

### 3.4 Metrics

MAE, RMSE and R² on the 2020–2025 holdout, plus per-state and per-year MAE
breakdowns, residual diagnostics, and permutation feature importance.

## 4. Results

Test set (2020–2025, 78 state-years):

| Model | MAE (t/ha) | RMSE (t/ha) | R² |
|---|---|---|---|
| Persistence baseline | 0.5554 | 0.6765 | 0.3195 |
| Ridge (α=100) | 0.5607 | 0.6899 | 0.2922 |
| Random Forest (500 trees) | 0.5954 | 0.7571 | 0.1477 |

Neither trained model beats the naive baseline on the holdout: Ridge is a
touch worse (MAE 0.561 vs 0.555 t/ha) and Random Forest clearly worse
(0.595 vs 0.555). The expanding year-based CV selected heavy regularization
for Ridge (α=100), i.e. the data supports shrinking almost everything toward
the dominant lag signal — and even that cannot improve on the lag itself.

Permutation importance: for Ridge, the dominant feature is the lagged yield
(+0.12 MAE when shuffled); climate features contribute an order of magnitude
less, and spring sunshine is slightly negative (noise). For Random Forest,
the lagged yield dominates even more strongly (+0.15 MAE), with climate
features near zero or slightly negative.

## 5. Why the baseline is so hard to beat

1. **Yield is highly autocorrelated** at state level (R² of persistence ≈ 0.32
   on its own). Technology trend + stable regional conditions make "last year"
   a strong prior.
2. **Seasonal aggregates are coarse.** A Mar–Jun mean temperature hides the
   timing of frost, heat spikes, and drought episodes that actually drive
   yield — the signal that matters lives at daily/weekly resolution.
3. **State means hide heterogeneity.** One number per Bundesland averages over
   very different soils and microclimates.
4. **Regime shifts break lag models.** The worst errors are 2025 in
   Nordrhein-Westfalen, Rheinland-Pfalz and Niedersachsen: an exceptionally
   sunny, dry spring (+2.3 to +2.6 SD sunshine vs. 2000–2024) coincided with
   record yield jumps of +1.2 to +2.0 t/ha vs. 2024. A model anchored on
   last year's yield cannot anticipate a break like that.
