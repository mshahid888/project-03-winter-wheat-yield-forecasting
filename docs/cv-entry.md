# Factual CV entry — Project 3

**Winter Wheat Yield Forecasting Across German Federal States (DWD + Regionaldatenbank)** — 2026

- Built an end-to-end forecasting study in Python (pandas, scikit-learn, matplotlib)
  testing whether Bundesland-level winter-wheat yield can be predicted from
  information available by a fixed 30 June cutoff: assembled a 338-row panel
  (13 states × 26 harvest years, 2000–2025) from the Regionaldatenbank Erntestatistik
  (yield, dt/ha → t/ha) and 36 DWD Climate Data Center monthly regional-average files
  (temperature, precipitation, sunshine), joined into seasonal spring and winter
  windows plus a one-year lagged yield.
- Enforced the forecast cutoff with a data-driven leakage audit: a single feature
  specification drives both construction and verification, and the build reads back
  the actual (year-offset, month) pairs aggregated into every feature and fails if any
  observation post-dates the cutoff — verified by injection testing, not by asserting
  a hardcoded list.
- Evaluated with a strictly chronological design: train 2000–2019 (260 rows), test
  2020–2025 (78 rows), hyperparameters tuned by expanding year-based cross-validation
  built from year masks rather than row positions (the panel is state-major, so
  sklearn's TimeSeriesSplit would fold across states instead of across years). Ridge
  was fitted as a `StandardScaler → Ridge` Pipeline passed directly to `GridSearchCV`,
  so scaling statistics are refit inside each training fold and never leak across folds.
- Reported a negative result rather than tuning toward a positive one: neither Ridge
  (MAE 0.5607, R² 0.2922) nor Random Forest (MAE 0.5954, R² 0.1477) demonstrably
  improves on a persistence baseline (MAE 0.5554, R² 0.3195), and paired testing shows
  none of the differences is distinguishable from noise on 78 test points
  (p = 0.33–0.91) — so the finding is the absence of a demonstrable gain, in either
  direction.
- Diagnosed why with permutation importance, per-state and per-year error breakdowns:
  lagged yield dominates (+0.12 to +0.15 MAE when shuffled) while seasonal climate
  aggregates contribute an order of magnitude less; the worst errors concentrate in
  2025 in the north-west, where the sunniest and driest spring in the 26-year record
  coincided with year-on-year yield increases of +0.9 to +2.0 t/ha that a lag-anchored
  model cannot anticipate.
- Documented the design honestly, including the limitations that weaken it: `year` is
  an extrapolated feature whose measured test-set importance is exactly zero for the
  tree model, spring sunshine carries negative permutation importance, and a six-year
  holdout cannot resolve differences of this size.
