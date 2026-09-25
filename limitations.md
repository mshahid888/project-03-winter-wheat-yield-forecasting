# Limitations

What this project is, and — more importantly — what it is not.

## Scope limitations

1. **State-level aggregates only.** One yield number per Bundesland per year
   averages over heterogeneous soils, farms, and microclimates. Results say
   nothing about individual fields or farms.
2. **13 of 16 states.** Berlin, Hamburg and Bremen have no reported
   winter-wheat yield in the source table and are excluded. Findings do not
   cover them.
3. **Seasonal climate windows.** Temperature, precipitation and sunshine are
   aggregated over multi-month windows. The timing of extreme events (late
   frost, heatwave during flowering, drought at grain filling) — likely the
   real yield drivers — is invisible at this resolution.
4. **No soil, management, or variety data.** Fertilization, crop protection,
   sowing dates, and cultivar turnover (a major driver of the long-run yield
   trend) are absent. The `year` feature absorbs the trend but explains
   nothing.
5. **No harvested area.** The source table contains no area or production
   measure, so area-based features (e.g. lagged area as a proxy for planting
   intentions) could not be built.

## Methodological limitations

6. **Small effective sample.** 338 rows sounds reasonable, but with 13 states
   the *independent* time dimension is 26 years. The test set is 6 years —
   one unusual year (like 2025) moves the metrics substantially.
7. **Random Forest scores lowest, but not significantly so on the reported
   tests.** The tested configuration (documented as 500 trees, unrestricted
   depth, leaf size 1) has the highest MAE and lowest R² of the three on the
   chronological holdout (R² 0.15 vs 0.32 for persistence). Reported as-is,
   not tuned away. The paired difference in absolute error against persistence
   is −0.040 t/ha (from `outputs/tables/model_comparison.csv`), with a reported
   95% bootstrap CI of [−0.136, +0.052] (p = 0.41). The CI and p-value come from
   an analysis outside this repository and cannot be reproduced from its code
   (see *What is and isn't reproducible* in the README). The repository does not
   record training-set error, so "overfitting" is a plausible reading of the
   ranking, not something established here.

8. **`year` is an extrapolated feature.** The training years are 2000–2019 and
   the test years 2020–2025, so every test-set value of `year` lies outside the
   range the models were fitted on. For Random Forest this is a known failure
   mode: a tree can only split at thresholds it saw in training, so all test
   rows fall into the terminal "latest year" leaf. The evidence is in this
   project's own output — `outputs/tables/permutation_importance.csv` gives the
   Random Forest a permutation importance for `year` of 5.55e-17, i.e. exactly
   zero: shuffling it on the test set changes nothing, because the model cannot
   use it there. The feature still consumed splits during training. It is
   documented here rather than dropped, because removing it would change the
   published results and this release reports the documented method as run.

9. **Spring sunshine has negative permutation importance.** For both models,
   shuffling `sun_spring_h` on the test set slightly *improves* MAE (−0.037
   Ridge, −0.018 Random Forest). That means the learned relationship for
   sunshine is not merely uninformative on unseen years but mildly harmful —
   consistent with a weak signal fitted to noise in the training period.
10. **No spatial modeling.** States are treated as independent rows with a
   shared model. Spatial autocorrelation (neighboring states sharing weather)
   is not modeled; a mixed-effects or explicitly spatial model could be a
   next step.
11. **Single train/test split.** One chronological split (2020–2025 holdout)
    rather than rolling-origin evaluation. The reported metrics are
    conditional on those six years — which, given limitation 6, is the main
    reason the model comparison is underpowered.
12. **No uncertainty quantification.** Point forecasts only; no prediction
    intervals. A forecaster would want those.

13. **The 30 June cutoff is a modelling convention, not an operational one.**
    June climate precedes the winter-wheat harvest, so using it is not
    statistical leakage. But DWD publishes the June monthly regional averages in
    early July, so a forecaster standing on 30 June would not yet have the June
    aggregate in hand. A strictly operational version would end the spring
    window in May, or use partial-June data.

## Interpretation cautions

14. **Correlation, not causation.** The sunshine/yield association in 2025 is
    observational. Nothing here identifies causal effects of climate on yield.
15. **Not a production forecasting system.** No operational data pipeline, no
    monitoring, no retraining policy. It is a retrospective modeling study.
16. **Climate data is post-hoc.** DWD monthly files used here were published
    after the fact; an operational 30-June forecast would rely on
    provisional/nowcast data with its own errors. The study assumes the
    climate inputs are known exactly at the cutoff.

## What would come next (and why it was not done here)

- Daily or weekly climate features (growing-degree days, heat-stress days,
  dry-spell length) instead of seasonal means.
- District-level (Kreis) modeling for finer spatial resolution.
- Rolling-origin backtesting over multiple holdout windows.
- Prediction intervals (quantile regression, conformal prediction).
- Adding soil and management covariates where available.

Each of these is a deliberate scope extension, not an oversight — the project
was kept small on purpose so every claim in it can be defended.
