# Data sources — Project 3

## 1. Winter wheat yield (target variable)

- **Source:** Regionaldatenbank Deutschland (Statistische Ämter des Bundes und der Länder)
- **Table:** `41241-01-03-4` — "Erträge ausgewählter landwirtschaftlicher Feldfrüchte – Jahressumme"
- **Survey:** Erntestatistik (Ernte- und Betriebsberichterstattung)
- **Table page:** https://www.regionalstatistik.de/genesis/online?operation=table&code=41241-01-03-4
- **Export variant used:** `41241-01-03-4-B`, regional level "regionale Ebenen" (Bundesländer)
- **Measure:** ERT001 "Hektarerträge" in **dt/ha** (converted to t/ha by dividing by 10 for modeling)
- **Crop column used:** `Winterweizen` (first of 10 crop columns; the other 9 are ignored)
- **Temporal coverage in this export:** 1999–2025 (27 years, every year present)
- **Geographic coverage in this export:** all 16 Bundesländer (region codes 01–16), 432 rows = 16 × 27, no duplicates
- **Modeling panel:** 13 states — Baden-Württemberg, Bayern, Brandenburg, Hessen, Mecklenburg-Vorpommern, Niedersachsen, Nordrhein-Westfalen, Rheinland-Pfalz, Saarland, Sachsen, Sachsen-Anhalt, Schleswig-Holstein, Thüringen. Berlin, Hamburg, Bremen excluded: the export contains no winter wheat data for them ("." = no data) in any year, consistent with negligible wheat area.
- **Completeness of the 13-state panel:** 351/351 year–state cells numeric — zero missing values.
- **Plausibility:** Bundesland winter wheat yields range 39.7–104.8 dt/ha (3.97–10.48 t/ha); Germany 2025 = 79.0 dt/ha = 7.90 t/ha, consistent with known German wheat yield levels.
- **Missing-value codes in the raw file:** `-` (not applicable), `/` (value withheld/uncertain), `.` (no data). None occur in the 13-state winter wheat panel.
- **What this table does NOT contain:** no area (Anbaufläche/Erntefläche) measure — the planned optional lagged-area feature was dropped. Yield only.
- **License:** Datenlizenz Deutschland – Namensnennung – Version 2.0 (attribution required; source: "© Statistische Ämter des Bundes und der Länder, Deutschland, 2026", export date 21.09.2026)
- **Raw file:** `data/raw/41241-01-03-4_winterwheat_bundesland_1999-2025_raw.csv` (31,280 bytes, ISO-8859-1/Latin-1, semicolon-delimited, comma decimals)
- **Reproducible download (how the file was obtained):**
  1. Open the table page above in a browser.
  2. In the table configuration, keep the default crops (all 10 included) and the yield measure.
  3. Under "ZEIT AUSWÄHLEN", choose "Alle verfügbaren Zeitangaben" (1999–2025).
  4. Restrict regions to the Bundesländer level if desired (reduces file size; full Kreise-level export exceeds the guest download limit of the portal and requires a free registered account).
  5. Download as CSV. Convert encoding to UTF-8 if needed; decimals use commas.
- **Verification performed on the raw file:** 432 data rows, 13 columns (Jahr;Regionscode;Regionsname;10 crops), 27/27 years present, 351/351 numeric winter-wheat cells for the 13 modelled states, value range plausible.

## 2. Climate data (features)

- **Source:** Deutscher Wetterdienst (DWD) Open Data, CDC regional averages for Germany
- **Base URL:** https://opendata.dwd.de/climate_environment/CDC/regional_averages_DE/monthly/
- **Files used (per calendar month, per Bundesland, 1881–present):** 36 text files
  - `monthly/air_temperature_mean/regional_averages_tm_MM.txt` — mean air temperature (°C), MM = 01…12
  - `monthly/precipitation/regional_averages_rr_MM.txt` — precipitation total (mm), MM = 01…12
  - `monthly/sunshine_duration/regional_averages_sd_MM.txt` — sunshine duration (hours), MM = 01…12
- **Format:** semicolon-delimited text; header row `Jahr;Monat;<Bundesland columns>;Deutschland;`. Individual-Bundesland columns exist for all 13 modeled states (plus combination columns such as Brandenburg/Berlin, which are not used). First data row 1881; no login required.
- **Access date:** 2026-09-21 (files retrieved by direct HTTP download; all 36 returned HTTP 200, created 2026-09-02 per file header).
- **License:** CC BY 4.0 — the DWD CDC "Terms of use for data on the CDC-OpenData area" (status May 2024, https://opendata.dwd.de/climate_environment/CDC/Terms_of_use.pdf) states that the Creative Commons BY 4.0 licence applies. Redistribution of the raw files in this repository is permitted under that licence with attribution to the Deutscher Wetterdienst as the source (verified 2026-09-21). Details: https://www.dwd.de/copyright.
- **Raw files:** `data/raw/dwd_tm_MM.txt`, `data/raw/dwd_rr_MM.txt`, `data/raw/dwd_sd_MM.txt` (MM = 01…12)
- **Feature construction:** monthly values are aggregated to growing-season windows with a strict forecast cutoff of **30 June** of the harvest year (all features known before harvest; see `methodology.md`).

## 3. Derived modeling dataset

- **File:** `data/processed/wheat_yield_panel.csv`
- **Built by:** `scripts/build_dataset.py` (see header comments for the exact transformation steps)
- **Content:** one row per Bundesland × harvest year (2000–2025, after one-year lag), target `yield_t_ha`, features per the feature table in `methodology.md`.
