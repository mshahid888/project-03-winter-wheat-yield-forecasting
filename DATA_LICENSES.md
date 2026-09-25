# Licensing scope and third-party data

This file was split out of `LICENSE` so that `LICENSE` contains only the
standard MIT licence text. Its content is unchanged apart from headings and
references to "the MIT licence above", which now point to `LICENSE`.

## Scope of the MIT licence

The MIT licence in `LICENSE` covers ONLY the material authored in this repository:

  - the source code (scripts/, notebooks/ where present)
  - the documentation and text files written for this project
  - the figures, maps and derived statistics produced by this project's code

It does NOT cover, and the author claims no ownership of, any third-party
dataset used, redistributed, or derived from in this repository. Each such
dataset remains the property of its provider and is governed by that
provider's own licence and attribution requirements, which are recorded in
this repository's data documentation and reproduced below.

## Third-party data

This project redistributes raw input data files under the licences below.

- Winter wheat yield, table 41241-01-03-4-B (Erntestatistik)
  Regionaldatenbank Deutschland, Statistische Aemter des Bundes und der
  Laender. REDISTRIBUTED in this repository at
  data/raw/41241-01-03-4_winterwheat_bundesland_1999-2025_raw.csv.
  Licence: Datenlizenz Deutschland - Namensnennung - Version 2.0
  (https://www.govdata.de/dl-de/by-2-0). Attribution required.
  Attribution: "(c) Statistische Aemter des Bundes und der Laender,
  Deutschland, 2026."

- Monthly regional averages of air temperature, precipitation and sunshine
  duration per Bundesland
  Deutscher Wetterdienst (DWD), Climate Data Center,
  regional_averages_DE/monthly/. REDISTRIBUTED in this repository at
  data/raw/dwd_{tm,rr,sd}_MM.txt (36 files).
  Licence: CC BY 4.0, per the DWD CDC Terms of Use (status May 2024,
  https://opendata.dwd.de/climate_environment/CDC/Terms_of_use.pdf).
  Attribution: "Deutscher Wetterdienst (DWD), Climate Data Center."

Both providers permit redistribution with attribution, which is why these raw
files are included in the repository. The derived panel
data/processed/wheat_yield_panel.csv combines both sources and inherits both
attribution requirements.

Users of the derived outputs in this repository must continue to satisfy the
attribution requirements of the upstream data providers listed above.
