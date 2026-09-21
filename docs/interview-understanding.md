# Interview Understanding Guide

How to talk about this project in a job interview — what was decided, why,
and what you would say when challenged. Read this alongside
`methodology.md` and `limitations.md`.

## The 60-second pitch

"I built a yield-forecasting study on German winter wheat at the federal-state
level, using official yield statistics and DWD climate data. The design
constraint was a fixed forecast date — 30 June of the harvest year — so every
feature had to be knowable by then. I compared a persistence baseline, Ridge,
and Random Forest with strictly chronological validation, including expanding
year-based CV folds built from year masks. The honest result: neither trained
model beats the baseline — Ridge is a touch worse, Random Forest clearly
worse. That negative result is the point — it shows why state-level seasonal
aggregates carry too little signal to beat last year's yield, and I can
explain exactly where the models fail."

## Questions you should be able to answer

### "Why a fixed forecast date?"

Because without one, leakage is almost guaranteed — using July/August weather
to "predict" a harvest that is already in the bins. The cutoff forces every
feature to be genuinely knowable at forecast time. In production this is the
difference between a backtest and a fantasy.

### "Why chronological validation instead of k-fold?"

Yield data is autocorrelated in time and the trend is non-stationary
(varieties and management improve). Shuffled k-fold would let the model train
on 2023 to predict 2010 — information from the future. Expanding year-based
folds mimic how the model would actually be deployed: trained on the past,
judged on the future.

### "Why does the baseline win?"

Three reasons: (1) state-level yield is strongly autocorrelated — trend plus
stable regional conditions make last year a strong prior; (2) seasonal climate
means hide the extreme events that actually move yield; (3) one number per
Bundesland averages over huge heterogeneity. The models have almost no real
signal to work with beyond the lag.

### "Why did Random Forest do worse than Ridge?"

Small panel (338 rows, effectively 26 independent years), and trees with
unrestricted depth memorize state-year idiosyncrasies. On a chronological
holdout with a regime shift (2025), that memorization hurts. Ridge's
regularization keeps it close to the dominant linear signal — the lag.

### "What would you do with more time?"

In priority order: (1) daily/weekly climate features — growing-degree days,
heat-stress days during flowering, dry-spell length; (2) district-level
modeling for spatial resolution; (3) rolling-origin backtesting instead of a
single holdout; (4) prediction intervals. Each is in `limitations.md` with
the reason it was out of scope.

### "How do you know there's no leakage?"

Three checks, automated in the build script: no climate month after June of
the harvest year enters any feature; no production/area/harvest-quantity
columns exist anywhere in the pipeline; the only yield-derived feature is the
previous year's value. Plus the outer test set (2020–2025) was never used for
any tuning decision.

### "The R² is 0.32 — isn't that bad?"

"Yes, and that's reported prominently, not buried. The value of the project
is the experimental discipline: proper temporal validation, a baseline that
earns its place, and a documented negative result with a causal story for the
failures — the 2025 errors line up with an exceptional spring. I'd rather
defend an honest 0.32 than an overfit 0.9."

## Numbers to have at your fingertips

- 338 rows: 13 states × 26 years (2000–2025; 1999 used for the lag).
- Test: 2020–2025 (78 state-years). Train: 2000–2019.
- Persistence MAE 0.555, Ridge MAE 0.561 (α=100), RF MAE 0.595 (500 trees) —
  all in t/ha, test 2020–2025. Baseline R² 0.32; Ridge 0.29; RF 0.15.
- Worst errors: NRW 2025 (RF error 2.36 t/ha), Rheinland-Pfalz 2025 (1.76),
  Niedersachsen 2025 (1.72) — record yield jumps on an exceptionally sunny,
  dry spring.
- Yield range in the data: 3.97–10.48 t/ha; mean ≈ 7.41 t/ha.
- Sources: Regionaldatenbank table 41241-01-03-4-B; DWD CDC regional monthly
  averages (temperature, precipitation, sunshine).

## What NOT to claim

- Do not call it a production forecasting system (no pipeline, no
  monitoring, no retraining policy).
- Do not claim field-level or farm-level relevance.
- Do not claim causal effects of climate on yield.
- Do not present the 2025 error story as proven — it is an observed
  association (exceptional spring coincides with record jumps), offered as
  the most plausible explanation, not a causal finding.
