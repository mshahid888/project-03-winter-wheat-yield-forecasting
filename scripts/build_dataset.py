"""
Build the modeling panel for Project 3.

Reads:
  data/raw/41241-01-03-4_winterwheat_bundesland_1999-2025_raw.csv  (yield, dt/ha)
  data/raw/dwd_tm_MM.txt / dwd_rr_MM.txt / dwd_sd_MM.txt           (DWD climate, per month)

Writes:
  data/processed/wheat_yield_panel.csv

Forecast framing:
  - Prediction date: 30 June of the harvest year (pre-harvest / early-season forecast).
  - Every climate feature may only use months whose data is known by 30 June:
      * spring window: March..June of the harvest year
      * winter window: October of the previous year .. February of the harvest year

Leakage audit (see audit_feature_windows below):
  The audit does NOT check a hardcoded list of months. It reads back, for every
  feature actually built, the (calendar-year offset, month) pairs of the climate
  rows that were aggregated into it, and asserts each pair is knowable by the
  cutoff: either it belongs to an earlier calendar year, or it is month <= June
  of the harvest year. Changing FEATURE_SPEC therefore changes what the audit
  tests, and a window that reached past the cutoff would fail even if the
  docstring still claimed otherwise.

Structural leakage rules also asserted:
  - No current-year production, no current-year area (the yield table has no area
    column at all, so this is structural, not just a choice).
  - Only lagged yield (year-1) is used as an autoregressive feature.
"""
import re
import pandas as pd
import numpy as np

RAW = "data/raw"
OUT = "data/processed/wheat_yield_panel.csv"

# 13 modeled states (region codes from the Regionaldatenbank export)
STATES = {
    "01": "Schleswig-Holstein", "03": "Niedersachsen", "05": "Nordrhein-Westfalen",
    "06": "Hessen", "07": "Rheinland-Pfalz", "08": "Baden-Württemberg",
    "09": "Bayern", "10": "Saarland", "12": "Brandenburg",
    "13": "Mecklenburg-Vorpommern", "14": "Sachsen", "15": "Sachsen-Anhalt",
    "16": "Thüringen",
}
# DWD column name -> our state name
DWD_COLS = {
    "Baden-Wuerttemberg": "Baden-Württemberg", "Bayern": "Bayern",
    "Brandenburg": "Brandenburg", "Hessen": "Hessen",
    "Mecklenburg-Vorpommern": "Mecklenburg-Vorpommern", "Niedersachsen": "Niedersachsen",
    "Nordrhein-Westfalen": "Nordrhein-Westfalen", "Rheinland-Pfalz": "Rheinland-Pfalz",
    "Saarland": "Saarland", "Sachsen": "Sachsen", "Sachsen-Anhalt": "Sachsen-Anhalt",
    "Schleswig-Holstein": "Schleswig-Holstein", "Thueringen": "Thüringen",
}

CUTOFF_MONTH = 6  # 30 June forecast date


def load_yield():
    rows = []
    with open(f"{RAW}/41241-01-03-4_winterwheat_bundesland_1999-2025_raw.csv",
              encoding="latin-1") as f:
        for line in f:
            if re.match(r"^\d{4};", line):
                p = line.rstrip("\n").split(";")
                code, year, ww = p[1], int(p[0]), p[3]
                if code in STATES and re.fullmatch(r"\d+,\d+", ww):
                    rows.append((STATES[code], year, float(ww.replace(",", ".")) / 10.0))
    y = pd.DataFrame(rows, columns=["state", "year", "yield_t_ha"])
    assert len(y) == 13 * 27, f"expected 351 rows, got {len(y)}"
    return y


def load_climate(var):
    """Long table: year, month, state, value for one DWD variable."""
    frames = []
    for m in range(1, 13):
        # NOTE: DWD files use "." as decimal separator (unlike the yield CSV).
        df = pd.read_csv(f"{RAW}/dwd_{var}_{m:02d}.txt", sep=";", encoding="latin-1",
                         skiprows=1)
        df = df.rename(columns={"Jahr": "year", "Monat": "month"})
        keep = {"year": "year", "month": "month"}
        keep.update({c: DWD_COLS[c] for c in DWD_COLS if c in df.columns})
        df = df[list(keep)].rename(columns=keep)
        assert set(DWD_COLS.values()) <= set(df.columns), f"missing states in {var} {m:02d}"
        long = df.melt(id_vars=["year", "month"], var_name="state", value_name=var)
        long[var] = pd.to_numeric(long[var], errors="coerce")
        frames.append(long)
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=[var])
    return out


SPRING_MONTHS = [3, 4, 5, 6]
WINTER_MONTHS = [10, 11, 12, 1, 2]

# The single source of truth for feature construction AND for the leakage audit:
# (output column, DWD variable, harvest-year key, months, aggregation).
FEATURE_SPEC = [
    ("temp_spring_C",    "tm", "hyear_spring", SPRING_MONTHS, "mean"),
    ("precip_spring_mm", "rr", "hyear_spring", SPRING_MONTHS, "sum"),
    ("sun_spring_h",     "sd", "hyear_spring", SPRING_MONTHS, "sum"),
    ("temp_winter_C",    "tm", "hyear_winter", WINTER_MONTHS, "mean"),
    ("precip_winter_mm", "rr", "hyear_winter", WINTER_MONTHS, "sum"),
]


def window_stats(clim, var, year_key, months, agg):
    """Aggregate one variable over a month window into harvest-year rows."""
    sub = clim[clim["month"].isin(months)]
    g = sub.groupby(["state", year_key])[var]
    return g.agg(agg).reset_index()


def observed_offsets(clim, year_key, months):
    """The (calendar_year - harvest_year, month) pairs actually aggregated.

    Read back from the climate frame that feeds the aggregation, not from a
    literal: this is what makes the audit a check on the real construction.
    """
    sub = clim[clim["month"].isin(months)]
    return set(zip((sub["year"] - sub[year_key]).astype(int), sub["month"].astype(int)))


def audit_feature_windows(frames):
    """Assert every month that entered a feature is knowable by the cutoff.

    A (offset, month) pair is admissible iff the observation predates the
    forecast date: either offset < 0 (an earlier calendar year, so complete),
    or offset == 0 and month <= CUTOFF_MONTH (this harvest year, up to June).
    Raises AssertionError naming the offending feature and month otherwise.
    """
    audited = []
    for col, var, year_key, months, _agg in FEATURE_SPEC:
        pairs = observed_offsets(frames[var], year_key, months)
        assert pairs, f"{col}: no climate rows contributed - window is empty"
        for off, month in sorted(pairs):
            ok = off < 0 or (off == 0 and month <= CUTOFF_MONTH)
            assert ok, (
                f"LEAK in {col}: month {month:02d} of the harvest year "
                f"(offset {off:+d}) is after the {CUTOFF_MONTH:02d}/30 cutoff")
        latest_same_year = max([m for o, m in pairs if o == 0], default=None)
        audited.append((col, var, sorted(pairs), latest_same_year))
        print(f"  audit {col:<17s} <- {var} months "
              f"{sorted({m for _, m in pairs})} "
              f"offsets {sorted({o for o, _ in pairs})} "
              f"latest harvest-year month {latest_same_year}")
    return audited


def main():
    y = load_yield()
    tm = load_climate("tm")
    rr = load_climate("rr")
    sd = load_climate("sd")

    # Harvest year for each climate row: spring months belong to the same year,
    # winter window Oct(year-1)..Feb(year) belongs to harvest year `year`.
    frames = {"tm": tm, "rr": rr, "sd": sd}
    for df in frames.values():
        df["hyear_spring"] = df["year"]                       # Mar..Jun of harvest year
        df["hyear_winter"] = np.where(df["month"] >= 10, df["year"] + 1, df["year"])

    # ---- leakage audit on the real windows, BEFORE anything is merged ----
    print("leakage audit (reads back the months each feature actually used):")
    audited = audit_feature_windows(frames)

    feats = y[["state", "year"]].drop_duplicates()
    for col, var, year_key, months, agg in FEATURE_SPEC:
        feats = feats.merge(
            window_stats(frames[var], var, year_key, months, agg)
            .rename(columns={var: col, year_key: "year"}),
            on=["state", "year"])

    panel = y.merge(feats, on=["state", "year"], how="left")
    panel = panel.sort_values(["state", "year"])
    panel["yield_lag1_t_ha"] = panel.groupby("state")["yield_t_ha"].shift(1)
    panel = panel.dropna().reset_index(drop=True)  # drops harvest year 1999 (no lag)
    assert len(panel) == 13 * 26, f"expected 338 rows, got {len(panel)}"

    # ---- structural leakage checks on the finished panel ----
    # (the month-window audit already ran on the real aggregation inputs above)
    built_cols = {col for col, *_ in FEATURE_SPEC}
    assert built_cols <= set(panel.columns), \
        f"features declared in FEATURE_SPEC missing from panel: {built_cols - set(panel.columns)}"
    # every climate column in the panel must come from an audited spec entry,
    # so a hand-added feature cannot bypass the window audit
    climate_cols = set(panel.columns) - {"state", "year", "yield_t_ha", "yield_lag1_t_ha"}
    assert climate_cols == built_cols, \
        f"unaudited climate columns in panel: {sorted(climate_cols - built_cols)}"
    # the target itself must not appear among the features: only the lag-1
    # yield may be derived from the yield series
    assert "yield_t_ha" in panel.columns  # target present exactly once
    yield_derived = {c for c in panel.columns if "yield" in c}
    assert yield_derived == {"yield_t_ha", "yield_lag1_t_ha"}, \
        f"unexpected yield-derived columns: {sorted(yield_derived)}"
    feature_cols = ["year", "yield_lag1_t_ha", "temp_spring_C", "precip_spring_mm",
                    "sun_spring_h", "temp_winter_C", "precip_winter_mm"]
    assert "production" not in panel.columns and "area" not in " ".join(panel.columns)
    assert panel[feature_cols].notna().all().all()

    panel[["state", "year", "yield_t_ha", "yield_lag1_t_ha", "temp_spring_C",
             "precip_spring_mm", "sun_spring_h", "temp_winter_C",
             "precip_winter_mm"]].to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(panel)} rows, years {panel.year.min()}..{panel.year.max()}")
    print("features:", feature_cols)
    latest_same_year = max(m for *_, m in audited if m is not None)
    print(f"leakage audit passed: latest harvest-year climate month actually used = "
          f"{latest_same_year:02d} (cutoff {CUTOFF_MONTH:02d}/30); every other month "
          f"came from an earlier calendar year; no production/area columns; "
          f"lag-1 yield only.")


if __name__ == "__main__":
    main()
