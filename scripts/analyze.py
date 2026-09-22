"""
Project 3 analysis: EDA + model comparison.

Chronological split (no shuffling — temporal problem):
  train: harvest years 2000..2019  (260 rows)
  test : harvest years 2020..2025  (78 rows, strictly unseen recent years)

Models:
  1. persistence baseline: predict this year's yield = last year's yield (per state)
  2. Ridge regression (StandardScaler -> Ridge as one Pipeline so the scaler is
     refit inside every CV training fold; alpha chosen by expanding year-based
     CV on train — never shuffled, never row-position folds)
  3. Random Forest (hyperparams chosen by expanding year-based CV on train)

Outputs:
  outputs/figures/*.png, outputs/tables/*.csv, metrics printed to stdout.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})

DATA = "data/processed/wheat_yield_panel.csv"
FIG = "outputs/figures"
TAB = "outputs/tables"
FEATURES = ["year", "yield_lag1_t_ha", "temp_spring_C", "precip_spring_mm",
            "sun_spring_h", "temp_winter_C", "precip_winter_mm"]
TARGET = "yield_t_ha"
TEST_START = 2020

df = pd.read_csv(DATA)
train = df[df.year < TEST_START].reset_index(drop=True)
test = df[df.year >= TEST_START].reset_index(drop=True)
X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]
print(f"train: {len(train)} rows ({train.year.min()}..{train.year.max()}), "
      f"test: {len(test)} rows ({test.year.min()}..{test.year.max()})")

# ---------------- EDA ----------------
fig, ax = plt.subplots(figsize=(10, 4.5))
for s, g in df.groupby("state"):
    ax.plot(g.year, g.yield_t_ha, lw=1, alpha=0.55)
nat = df.groupby("year").yield_t_ha.mean()
ax.plot(nat.index, nat.values, lw=2.5, color="black", label="13-state mean")
ax.axvline(TEST_START - 0.5, ls="--", color="red", lw=1, label="train/test split")
ax.set(xlabel="Harvest year", ylabel="Winter wheat yield (t/ha)",
       title="Winter wheat yield by Bundesland, 2000–2025")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(f"{FIG}/yield_timeseries.png"); plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, f in zip(axes.flat, [c for c in FEATURES if c != "year"]):
    ax.scatter(df[f], df.yield_t_ha, s=8, alpha=0.4)
    ax.set(xlabel=f, ylabel="yield (t/ha)")
axes.flat[-1].axis("off")
fig.suptitle("Feature vs target (all years, all states)")
fig.tight_layout(); fig.savefig(f"{FIG}/feature_scatter.png"); plt.close(fig)

corr = df[FEATURES + [TARGET]].corr().round(2)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(corr.columns)), corr.columns, fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
ax.set_title("Feature correlation matrix")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout(); fig.savefig(f"{FIG}/correlation.png"); plt.close(fig)
corr.to_csv(f"{TAB}/correlation.csv")

# ---------------- models ----------------
# Expanding, year-based CV folds built from year masks (never shuffled and
# never from row positions — the panel is state-major, so sklearn's
# TimeSeriesSplit would fold across states instead of across years).
# Each fold trains on all states for years <= train_end and validates on the
# next 3-year block, mimicking real deployment: past -> future.
YEAR_FOLDS = [
    (2000, 2004, 2005, 2007),
    (2000, 2007, 2008, 2010),
    (2000, 2010, 2011, 2013),
    (2000, 2013, 2014, 2016),
    (2000, 2016, 2017, 2019),
]
_years = train["year"].values
cv = [(np.where((_years >= tr0) & (_years <= tr1))[0],
       np.where((_years >= va0) & (_years <= va1))[0])
      for tr0, tr1, va0, va1 in YEAR_FOLDS]
# sanity: folds are chronological and cover the whole training period
for (tri, vai), (tr0, tr1, va0, va1) in zip(cv, YEAR_FOLDS):
    assert _years[tri].max() == tr1 and _years[vai].min() == va0 \
        and _years[vai].max() == va1 and tr1 < va0
print(f"CV folds: {[(f[0], f[1], f[2], f[3]) for f in YEAR_FOLDS]}")

# 1. persistence baseline (no fitting)
pred_base = test["yield_lag1_t_ha"].values

# 2. Ridge with scaling, alpha via CV.
# The scaler lives INSIDE a Pipeline that is passed directly into
# GridSearchCV, so StandardScaler is fitted independently on each CV
# training fold — no scaling statistics can leak from later years into
# an earlier validation fold (and the final refit scales with train-only
# statistics before touching the test set).
ridge_cv = GridSearchCV(
    Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
    {"ridge__alpha": [0.1, 1.0, 10.0, 100.0]},
    cv=cv, scoring="neg_mean_absolute_error")
ridge_cv.fit(X_train, y_train)
ridge = ridge_cv.best_estimator_
pred_ridge = ridge.predict(X_test)
print("ridge best alpha:", {"alpha": ridge_cv.best_params_["ridge__alpha"]})

# 3. Random Forest, hyperparams via CV (small grid — dataset is small)
rf_cv = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    {"n_estimators": [200, 500], "max_depth": [None, 6, 10],
     "min_samples_leaf": [1, 4]},
    cv=cv, scoring="neg_mean_absolute_error")
rf_cv.fit(X_train, y_train)
rf = rf_cv.best_estimator_
pred_rf = rf.predict(X_test)
print("rf best params:", rf_cv.best_params_)

results = pd.DataFrame({
    "model": ["persistence_baseline", "ridge", "random_forest"],
    "MAE": [mean_absolute_error(y_test, p) for p in (pred_base, pred_ridge, pred_rf)],
    "RMSE": [root_mean_squared_error(y_test, p)
             for p in (pred_base, pred_ridge, pred_rf)],
    "R2": [r2_score(y_test, p) for p in (pred_base, pred_ridge, pred_rf)],
}).round(4)
results.to_csv(f"{TAB}/model_comparison.csv", index=False)
print(results.to_string(index=False))

# train/test timeline figure
fig, ax = plt.subplots(figsize=(10, 2.2))
ax.barh(["train"], [TEST_START - 2000], left=[2000], height=0.5, color="steelblue")
ax.barh(["test"], [2025 - TEST_START + 1], left=[TEST_START], height=0.5, color="darkorange")
ax.set(xlim=(1999, 2026), xlabel="Harvest year",
       title="Chronological train/test split (no shuffling)")
for y0, lab in ((2000, "train 2000–2019 (n=260)"), (TEST_START, "test 2020–2025 (n=78)")):
    ax.text(y0 + 0.3, 0, lab, va="center", fontsize=9, color="white" if y0 == 2000 else "black")
fig.tight_layout(); fig.savefig(f"{FIG}/train_test_timeline.png"); plt.close(fig)

# predicted vs actual
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
for ax, p, name in zip(axes, (pred_base, pred_ridge, pred_rf),
                       ("Persistence", "Ridge", "Random Forest")):
    ax.scatter(y_test, p, s=14, alpha=0.6)
    lo, hi = y_test.min(), y_test.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set(xlabel="Actual yield (t/ha)", ylabel="Predicted yield (t/ha)", title=name)
fig.suptitle("Predicted vs actual on unseen test years (2020–2025)")
fig.tight_layout(); fig.savefig(f"{FIG}/pred_vs_actual.png"); plt.close(fig)

# per-state test MAE
err = test[["state", "year"]].copy()
err["actual"] = y_test.values
for name, p in (("baseline", pred_base), ("ridge", pred_ridge), ("rf", pred_rf)):
    err[name] = np.abs(err["actual"] - p)
per_state = err.groupby("state")[["baseline", "ridge", "rf"]].mean().round(3)
per_state.to_csv(f"{TAB}/per_state_mae.csv")
print("\nPer-state test MAE (t/ha):"); print(per_state.to_string())

# per-year test MAE
per_year = err.groupby("year")[["baseline", "ridge", "rf"]].mean().round(3)
per_year.to_csv(f"{TAB}/per_year_mae.csv")
print("\nPer-year test MAE (t/ha):"); print(per_year.to_string())

# feature importance: permutation importance on the TEST set (honest, model-agnostic)
# (the Ridge pipeline accepts the raw test frame — scaling is part of the model)
imp_frames = []
for name, model in (("ridge", ridge), ("random_forest", rf)):
    pi = permutation_importance(model, X_test, y_test.values, n_repeats=20,
                                random_state=42, scoring="neg_mean_absolute_error")
    imp = pd.DataFrame({"feature": FEATURES,
                        "mean_abs_increase_in_MAE": pi.importances_mean,
                        "std": pi.importances_std,
                        "model": name}).sort_values("mean_abs_increase_in_MAE",
                                                   ascending=False)
    imp_frames.append(imp)
imp_all = pd.concat(imp_frames)
imp_all.to_csv(f"{TAB}/permutation_importance.csv", index=False)
print("\nPermutation importance (test set):"); print(imp_all.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, (name, title) in zip(axes, (("ridge", "Ridge"), ("random_forest", "Random Forest"))):
    d = imp_all[imp_all.model == name].sort_values("mean_abs_increase_in_MAE")
    ax.barh(d.feature, d.mean_abs_increase_in_MAE, xerr=d["std"], color="teal", alpha=0.8)
    ax.set(xlabel="Increase in test MAE when shuffled (t/ha)", title=f"{title} — permutation importance")
fig.suptitle("Which features matter on unseen years?")
fig.tight_layout(); fig.savefig(f"{FIG}/feature_importance.png"); plt.close(fig)

# residual analysis: error vs actual yield level
fig, ax = plt.subplots(figsize=(7, 4.5))
resid = y_test.values - pred_rf
ax.scatter(y_test, resid, s=16, alpha=0.6)
ax.axhline(0, color="k", lw=1)
ax.set(xlabel="Actual yield (t/ha)", ylabel="RF residual (actual − predicted, t/ha)",
       title="Random Forest residuals vs yield level (test set)")
fig.tight_layout(); fig.savefig(f"{FIG}/residuals.png"); plt.close(fig)

print("\nWorst test errors (RF), top 8:")
worst = err.assign(rf_err=np.abs(err.actual - pred_rf)).nlargest(8, "rf_err")
print(worst[["state", "year", "actual", "rf_err"]].round(2).to_string(index=False))
print("\nDone. Figures ->", FIG, "| Tables ->", TAB)
