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
7. **Random Forest overfits the panel structure.** 500 trees with unrestricted
   depth memorize state-year patterns in training; on the chronological
   holdout it performs worse than the baseline (R² 0.15 vs 0.32). Reported
   as-is, not tuned away.
8. **No spatial modeling.** States are treated as independent rows with a
   shared model. Spatial autocorrelation (neighboring states sharing weather)
   is not modeled; a mixed-effects or explicitly spatial model could be a
   next step.
9. **Single train/test split.** One chronological split (2020–2025 holdout)
   rather than rolling-origin evaluation. The reported metrics are
   conditional on those six years.
10. **No uncertainty quantification.** Point forecasts only; no prediction
    intervals. A forecaster would want those.

## Interpretation cautions

11. **Correlation, not causation.** The sunshine/yield association in 2025 is
    observational. Nothing here identifies causal effects of climate on yield.
12. **Not a production forecasting system.** No operational data pipeline, no
    monitoring, no retraining policy. It is a retrospective modeling study.
13. **Climate data is post-hoc.** DWD monthly files used here were published
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
