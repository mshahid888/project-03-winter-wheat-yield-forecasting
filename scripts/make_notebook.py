"""Generate the teaching notebook for Project 3 and execute it."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

def md(text):
    nb.cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    nb.cells.append(nbf.v4.new_code_cell(text))

md("""# Predicting Winter Wheat Yield Across German Federal States
### A pre-harvest forecasting exercise with official yield statistics and DWD climate data

**The one-minute version:** Can we forecast this year's winter wheat yield per German federal state *before harvest*, using only information that is actually known by end of June — last year's yield plus spring and winter weather? I compare three approaches on strictly unseen recent years (2020–2025): a naive persistence baseline ("this year = last year"), Ridge regression, and a Random Forest. The answer on this holdout: neither trained model demonstrably improves on the naive persistence baseline — which is itself a useful finding.

**Data (all real, all open):**
- Yield: Regionaldatenbank Deutschland, table 41241-01-03-4 (Erntestatistik), winter wheat, dt/ha → t/ha, 13 Bundesländer, 1999–2025.
- Climate: DWD open data, monthly Bundesland averages (temperature, precipitation, sunshine), 1881–present.
- Full provenance in `data_sources.md`; build steps in `scripts/build_dataset.py`.
""")

code("""# Suppress a machine-specific matplotlib import warning (multi-version dist-packages)
# so notebook outputs are clean and portable across machines.
import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D", category=UserWarning)
import pandas as pd, numpy as np, matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})
df = pd.read_csv("data/processed/wheat_yield_panel.csv")
print(df.shape, "| years", df.year.min(), "–", df.year.max(), "| states", df.state.nunique())
df.head(3)""")

md("""## 1. Forecasting setup — the 30 June rule

This is framed as a **pre-harvest forecast made on 30 June**. Every feature must be known by then:

| Feature | Definition | Known by 30 June? |
|---|---|---|
| `yield_lag1_t_ha` | Previous harvest year's yield (t/ha), same state | Yes — last year's harvest is long finished |
| `year` | Calendar year of the forecast | Yes |
| `temp_spring_C` | Mean temperature, March–June of harvest year | Yes — June data is in |
| `precip_spring_mm` | Precipitation sum, March–June | Yes |
| `sun_spring_h` | Sunshine sum, March–June | Yes |
| `temp_winter_C` | Mean temperature, Oct (prev. year)–Feb | Yes |
| `precip_winter_mm` | Precipitation sum, Oct (prev. year)–Feb | Yes |

Deliberately excluded (leakage): current-year production and harvested area — production is arithmetically tied to yield, and area is only known after harvest. The source table contains no area column at all, so this is structural, not just a choice. `scripts/build_dataset.py` ends with an automated leakage audit asserting no climate month after June is used.""")

code("""from IPython.display import Image
Image("outputs/figures/train_test_timeline.png")""")

md("""**Evaluation protocol:** strict chronological split — train on 2000–2019, test on the unseen recent years 2020–2025. No shuffling, no peeking: for time-ordered data, random splits leak the future into training.""")

code("""# Yield dynamics: state trajectories + national mean
fig, ax = plt.subplots(figsize=(10, 4.5))
for s, g in df.groupby("state"):
    ax.plot(g.year, g.yield_t_ha, lw=1, alpha=0.5)
nat = df.groupby("year").yield_t_ha.mean()
ax.plot(nat.index, nat.values, lw=2.5, color="black", label="13-state mean")
ax.axvline(2019.5, ls="--", color="red", lw=1, label="train/test split")
ax.set(xlabel="Harvest year", ylabel="Yield (t/ha)",
       title="Winter wheat yield by Bundesland — persistence is visible")
ax.legend(fontsize=8)
plt.show()
print("National mean yield 2000–2019: %.2f t/ha | 2020–2025: %.2f t/ha" %
      (nat.loc[2000:2019].mean(), nat.loc[2020:2025].mean()))""")

md("""## 2. Models

1. **Persistence baseline** — predict this year's yield = last year's yield (per state). Zero parameters; the number every real model must beat.
2. **Ridge regression** — linear model with L2 regularization; features standardized; `alpha=100`.
3. **Random Forest** — the tested configuration: 500 trees, `max_depth=None`, `min_samples_leaf=1`.

**Where these hyperparameters come from:** they are hard-coded in this notebook. `scripts/analyze.py` chooses them by expanding year-based cross-validation on the training period (train 2000–2004 → validate 2005–2007, …, train through 2016 → validate 2017–2019), over α ∈ {0.1, 1, 10, 100} for Ridge and n_estimators ∈ {200, 500} × max_depth ∈ {None, 6, 10} × min_samples_leaf ∈ {1, 4} for the Random Forest. The folds are built from year masks, never shuffled — the panel is state-major, so a naive row-based `TimeSeriesSplit` would fold across states instead of across years. `analyze.py` prints its selection but does not save it, so the repository holds no recorded output showing which values it chose; the values above are the ones documented in `methodology.md`. The metrics printed below match `outputs/tables/model_comparison.csv` (written by `analyze.py`) to four decimals, which is consistent with — but does not prove — `analyze.py` having selected these same values.""")

code("""from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

FEATURES = ["year","yield_lag1_t_ha","temp_spring_C","precip_spring_mm",
            "sun_spring_h","temp_winter_C","precip_winter_mm"]
tr = df[df.year < 2020]; te = df[df.year >= 2020]
Xtr, ytr, Xte, yte = tr[FEATURES], tr.yield_t_ha, te[FEATURES], te.yield_t_ha

pred_base = te["yield_lag1_t_ha"].values                       # persistence
# Scaler fitted inside the pipeline (train data only) — mirrors scripts/analyze.py
ridge = Pipeline([("scaler", StandardScaler()),
                  ("ridge", Ridge(alpha=100.0))]).fit(Xtr, ytr)  # hard-coded; see the Models section
pred_ridge = ridge.predict(Xte)
rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1).fit(Xtr, ytr)  # hard-coded; see the Models section
pred_rf = rf.predict(Xte)

for name, p in [("persistence", pred_base), ("ridge", pred_ridge), ("random_forest", pred_rf)]:
    print(f"{name:13s} MAE={mean_absolute_error(yte,p):.4f}  "
          f"RMSE={root_mean_squared_error(yte,p):.4f}  R²={r2_score(yte,p):.4f}")""")

md("""## 3. Results on unseen years (2020–2025)

The full comparison table (from `scripts/analyze.py`) is saved at `outputs/tables/model_comparison.csv`:""")

code("""print(pd.read_csv("outputs/tables/model_comparison.csv").to_string(index=False))""")

md("""**Reading the results:** on the 2020–2025 holdout, *neither* trained model improves on the naive baseline. As point estimates, Ridge (MAE 0.561 t/ha) is slightly worse than simply predicting last year's yield (0.555), and the tested Random Forest configuration has the highest MAE (0.595) and the lowest R² (0.15) of the three. With only 260 training rows and strong autocorrelation in the target, the regularized linear model stays close to the dominant signal (the lag) without improving on it. This notebook does not measure training-set error, so it does not establish *why* the Random Forest scores lower: overfitting is one possible explanation, not a demonstrated one. Whether the gaps between the three models exceed noise is discussed in `methodology.md` §4; the significance tests reported there come from an analysis outside this repository and cannot be reproduced from its code.""")

code("""from IPython.display import Image
Image("outputs/figures/pred_vs_actual.png")""")

md("""## 4. What actually drives the predictions?""")

code("""from IPython.display import Image
Image("outputs/figures/feature_importance.png")""")

md("""Permutation importance on the *test* set (how much worse MAE gets when a feature is shuffled): last year's yield dominates for both models (~0.12–0.15 t/ha). Each climate feature contributes about 0.02 t/ha or less, and spring sunshine is negative for both models (−0.037 Ridge, −0.018 Random Forest): shuffling it slightly *improves* test MAE. At Bundesland-monthly resolution, weather adds little on top of persistence; a plausible explanation, not tested here, is that the relevant climate signal lives at finer spatial/temporal scales than state-monthly averages.""")

md("""## 5. Error analysis — where does it break?

Per-state test MAE shows geography matters more than the model choice:""")

code("""print(pd.read_csv("outputs/tables/per_state_mae.csv").to_string(index=False))""")

md("""Averaged over the three models, Thüringen (0.26–0.32 t/ha) and Sachsen (0.26–0.48 t/ha) have the lowest test MAE, and Nordrhein-Westfalen (0.93–1.04 t/ha) and Schleswig-Holstein (0.67–0.87 t/ha) the highest — the two states with the highest mean yields over the 2000–2019 training period. And 2025 stands out:""")

code("""print(pd.read_csv("outputs/tables/per_year_mae.csv").to_string(index=False))""")

code("""# What made 2025 special? Yield jumps vs 2024 and spring climate anomalies
d25 = df[df.year == 2025].set_index("state"); hist = df[df.year < 2025]
rows = []
for s in sorted(d25.index):
    chg = d25.loc[s, "yield_t_ha"] - df[(df.state == s) & (df.year == 2024)]["yield_t_ha"].values[0]
    h = hist[hist.state == s]; r = d25.loc[s]
    zsun = (r["sun_spring_h"] - h["sun_spring_h"].mean()) / h["sun_spring_h"].std()
    rows.append((s, round(chg, 2), round(zsun, 1)))
print(pd.DataFrame(rows, columns=["state", "yield_change_2024→2025_t/ha", "spring_sunshine_z"]).to_string(index=False))""")

md("""**The 2025 story (from the table above):** yields rose in 12 of 13 states compared with 2024, by 0.5 t/ha or more in 8 of them (largest: Nordrhein-Westfalen +1.99 t/ha, Niedersachsen +1.45 t/ha); Mecklenburg-Vorpommern fell (−0.28 t/ha). Spring sunshine was 1.7 to 2.6 standard deviations above each state's 2000–2024 mean. 2025 has the highest test MAE of the six test years for all three models (`outputs/tables/per_year_mae.csv`). The persistence baseline under-predicts every state whose yield rose, by construction; the pipeline does not save per-state predictions for Ridge or the Random Forest, so their 2025 errors by state are not shown here. A year this unusual relative to the training period is a known weak point of empirical forecasting, and it is one reason the conclusion is modest.""")

code("""from IPython.display import Image
Image("outputs/figures/residuals.png")""")

md("""## 6. Limitations (see `limitations.md` for the full list)

- **Aggregation level:** Bundesland-monthly climate averages smooth away the extreme events (heat spikes, dry spells) that actually drive yield losses.
- **Small data:** 338 rows / 260 for training — limits model complexity; the Random Forest's lower test scores are consistent with overfitting but do not establish it.
- **No management data:** varieties, sowing dates, fertilizer, plant protection are unobserved and absorbed into the lagged-yield term.
- **No area/production:** the source table has no area measure, so supply-side questions are out of scope.
- **Stationarity assumption:** the 2025 miss shows the model assumes the future looks like the past.

## 7. What I would try next

1. Finer-grained weather (weekly/daily extremes: heat days >30 °C, dry-spell length) instead of seasonal means.
2. Soil and management proxies per state.
3. A hierarchical/mixed-effects model with state-level intercepts instead of one global model.
4. Expanding-window backtesting for a more robust performance estimate.

---
*Notebook generated 2026-09-21. Data: Regionaldatenbank table 41241-01-03-4-B (Datenlizenz Deutschland – Namensnennung – Version 2.0) and DWD CDC open data.*""")

nb_path = "notebooks/wheat_yield_forecasting.ipynb"
# encoding is explicit: the notebook contains non-ASCII text (arrows, umlauts,
# the superscript in R^2), which fails on any platform whose default encoding
# is not UTF-8 (e.g. cp1252 on Windows).
with open(nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("notebook written:", len(nb.cells), "cells")

# Execute the notebook end-to-end (reproducible, not just generated).
# Run from the repository root so the notebook's relative paths resolve.
from nbclient import NotebookClient
nb = nbf.read(nb_path, as_version=4)
NotebookClient(nb, timeout=900, kernel_name="python3").execute()
with open(nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
n_out = sum(len(c.get("outputs", [])) for c in nb.cells)
print("notebook executed end-to-end:", n_out, "cell outputs")
