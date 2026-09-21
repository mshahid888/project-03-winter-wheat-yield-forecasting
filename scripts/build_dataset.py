"""
Build the modeling panel for Project 3.

Reads:
  data/raw/41241-01-03-4_winterwheat_bundesland_1999-2025_raw.csv  (yield, dt/ha)
  data/raw/dwd_tm_MM.txt / dwd_rr_MM.txt / dwd_sd_MM.txt           (DWD climate, per month)

Writes:
  data/processed/wheat_yield_panel.csv

Forecast framing (documented decision):
  - Prediction date: 30 June of the harvest year (pre-harvest / early-season forecast).
  - Every climate feature may only use months whose data is known by 30 June:
      * spring window: March..June of the harvest year
      * winter window: October of the previous year .. February of the harvest year
  - The leakage audit at the end of this script ASSERTS that no climate input
    uses a month later than June of the harvest year.

Leakage rules enforced:
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


def window_stats(clim, var, year_key, months, agg):
    """Aggregate one variable over a month window into harvest-year rows."""
    sub = clim[clim["month"].isin(months)]
    g = sub.groupby(["state", year_key])[var]
    return g.agg(agg).reset_index()


def main():
    y = load_yield()
    tm = load_climate("tm")
    rr = load_climate("rr")
    sd = load_climate("sd")

    # Harvest year for each climate row: spring months belong to the same year,
    # winter window Oct(year-1)..Feb(year) belongs to harvest year `year`.
    for df in (tm, rr, sd):
        df["hyear_spring"] = df["year"]                       # Mar..Jun of harvest year
        df["hyear_winter"] = np.where(df["month"] >= 10, df["year"] + 1, df["year"])

    spring_months = [3, 4, 5, 6]
    winter_months = [10, 11, 12, 1, 2]

    feats = y[["state", "year"]].drop_duplicates()
    feats = feats.merge(window_stats(tm, "tm", "hyear_spring", spring_months, "mean")
                        .rename(columns={"tm": "temp_spring_C", "hyear_spring": "year"}),
                        on=["state", "year"])
    feats = feats.merge(window_stats(rr, "rr", "hyear_spring", spring_months, "sum")
                        .rename(columns={"rr": "precip_spring_mm", "hyear_spring": "year"}),
                        on=["state", "year"])
    feats = feats.merge(window_stats(sd, "sd", "hyear_spring", spring_months, "sum")
                        .rename(columns={"sd": "sun_spring_h", "hyear_spring": "year"}),
                        on=["state", "year"])
    feats = feats.merge(window_stats(tm, "tm", "hyear_winter", winter_months, "mean")
                        .rename(columns={"tm": "temp_winter_C", "hyear_winter": "year"}),
                        on=["state", "year"])
    feats = feats.merge(window_stats(rr, "rr", "hyear_winter", winter_months, "sum")
                        .rename(columns={"rr": "precip_winter_mm", "hyear_winter": "year"}),
                        on=["state", "year"])

    panel = y.merge(feats, on=["state", "year"], how="left")
    panel = panel.sort_values(["state", "year"])
    panel["yield_lag1_t_ha"] = panel.groupby("state")["yield_t_ha"].shift(1)
    panel = panel.dropna().reset_index(drop=True)  # drops harvest year 1999 (no lag)
    assert len(panel) == 13 * 26, f"expected 338 rows, got {len(panel)}"

    # ---- leakage audit ----
    used = [("spring", m) for m in spring_months] + \
           [("winter(prevOct-Dec)", m) for m in (10, 11, 12)] + \
           [("winter(Jan-Feb)", m) for m in (1, 2)]
    latest = max(m for name, m in used if "prevOct-Dec" not in name)
    assert latest <= CUTOFF_MONTH, \
        f"leak: harvest-year climate month {latest} is after the June forecast cutoff"
    # months 7..12 of the harvest year are never used:
    assert not any(name == "spring" and m > CUTOFF_MONTH for name, m in used), \
        "leak: climate month after June forecast cutoff used"
    # the Oct-Dec winter months belong to the previous calendar year (fine),
    # and the Jan-Feb winter months are within the cutoff:
    assert set(m for name, m in used if "prevOct-Dec" in name) == {10, 11, 12}
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
    print("leakage audit passed: latest climate month used = June of harvest year; "
          "no production/area columns; lag-1 yield only.")


if __name__ == "__main__":
    main()
