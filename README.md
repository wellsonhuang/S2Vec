# Data

This folder is not tracked in git. This file describes every input, where it comes from, and which script
produces each derived file. Paths are relative to the repository root.

The two large inputs can stay where they are: pass `--layer-dir DIR` to `scripts/data/build_dataset.py`
and `--hrrr-dir DIR` to `scripts/hazards/hrrr_extract.py` (see `reproduce.sh`).

## Layout

```
data/
├── raw/
│   ├── hex_county_state_zip_crosswalk.parquet
│   ├── layers/                      # H3-indexed feature layers (one Parquet per layer)
│   ├── pde/pde_junwei.shp           # Hurricane Harvey building damage points (+ .dbf .shx .prj .cpg)
│   ├── mrms/                        # GaugeCorr_QPE_01H_00.00_YYYYMMDD-HHMMSS.grib2.gz
│   ├── hrrr/MMDD/                   # hrrr.tHHz.wrfsfcf00.grib2
│   └── outage/zcta_hc_beryl_outage.gpkg
├── interim/hrrr_wind_points.parquet
├── Harvey_MaxDepth_results.parquet  # flood depth features (provided as is)
├── houston_harris_dataset.parquet   # integrated features, Harris County
├── texas_dataset.parquet            # integrated features, Texas
├── houston_harris_pre.parquet       # preprocessed features
├── texas_pre.parquet
├── harris_county_h3_pde_avg.parquet
├── harris_county_h3_rain_features.parquet
├── harris_h3l8_beryl_outage.parquet
└── harris_h3l8_wind_features.parquet
```

## Raw inputs

### Feature layers (`raw/layers/`) and crosswalk (`raw/hex_county_state_zip_crosswalk.parquet`)

Multi-source H3 resolution-8 database: HIFLD critical infrastructure, OpenPOI, CMS nursing homes,
building stock, Ookla and Microsoft connectivity, FEMA National Risk Index (hazard, exposure,
vulnerability and community resilience layers), U.S. Census. Each file has a `hex_id` column;
the crosswalk maps every cell to its county, state, ZIP code and
resolution-7/6 parents.

- Source: internal data of our lab; variable definitions follow the
  [Resilitix BigQuery data dictionary (v4)](https://tamucs-my.sharepoint.com/:x:/r/personal/liuchenyue_tamu_edu/_layouts/15/doc2.aspx?sourcedoc=%7B23D51A7A-1A96-44F6-90CA-5EDA54E0E4A4%7D&file=data%20dictionary%20resilitix%20bigquery_v4.xlsx&fromShare=true&action=default&mobileredirect=true).
- Not publicly redistributed.

### Building damage, Hurricane Harvey 2017 (`raw/pde/pde_junwei.shp`)

72,755 building points in WGS 84 (EPSG:4326) with one field, `pde` (percent damage estimate, 0–1).
72,292 points fall inside Harris County cells.

- Source: internal data of our lab. Not publicly redistributed.

### Rainfall, Hurricane Harvey 2017 (`raw/mrms/`)

NOAA Multi-Radar Multi-Sensor, gauge-corrected 1-hour QPE (`GaugeCorr_QPE_01H`), 145 hourly files from
2017-08-25 01:00 UTC to 2017-08-31 00:00 UTC.

- Source: Iowa Environmental Mesonet MTArchive,
  `https://mtarchive.geol.iastate.edu/2017/08/<DD>/mrms/ncep/GaugeCorr_QPE_01H/`
- Public NOAA data.

### Wind, Hurricane Beryl 2024 (`raw/hrrr/`)

NOAA High-Resolution Rapid Refresh surface analyses (`wrfsfcf00`), hourly, one folder per day (`0706` …),
2024-07-06 01:00 UTC to 2024-07-12 00:00 UTC (144 hours; `scripts/hazards/hrrr_extract.py` skips files
outside this window).
Fields used: `gust`, `10u`, `10v`.

- Source: University of Utah HRRR archive,
  `https://home.chpc.utah.edu/~u0553130/Brian_Blaylock/cgi-bin/hrrr_download.cgi?model=hrrr&field=sfc&date=2024-07-DD&link2=grib2`
- Public NOAA data.

### Power outages, Hurricane Beryl 2024 (`raw/outage/zcta_hc_beryl_outage.gpkg`)

140 ZIP Code Tabulation Areas (2020 ZCTA geometry, NAD83 / EPSG:4269) in Harris County. Fields used:

| Field | Meaning |
|---|---|
| `ZCTA5CE20` | ZIP code |
| `T1_total_hours` | total outage hours |
| `T2_peak_avg_pct` | mean peak outage percentage |
| `T3_days_to_restore` | days until power was restored (missing for 47 ZCTAs, see `flag_T3_undefined`) |

The file also contains customer counts, quality flags and `fold_id` / `micro_id`.

- Source: internal data of our lab. Not publicly redistributed.

### Flood depth (`Harvey_MaxDepth_results.parquet`)

Maximum flood depth features for Harris County cells in the floodplain, used as provided.

## Derived files

| File | Script |
|---|---|
| `houston_harris_dataset.parquet`, `texas_dataset.parquet` | `scripts/data/build_dataset.py` (`REGION`) |
| `houston_harris_pre.parquet`, `texas_pre.parquet` | `src/preprocessing.py` (`REGION`) |
| `harris_county_h3_pde_avg.parquet` | `scripts/hazards/pde_labels.py` |
| `harris_county_h3_rain_features.parquet` | `scripts/hazards/mrms_rainfall.py` |
| `harris_h3l8_beryl_outage.parquet` | `scripts/hazards/outage_labels.py` |
| `interim/hrrr_wind_points.parquet` | `scripts/hazards/hrrr_extract.py` |
| `harris_h3l8_wind_features.parquet` | `scripts/hazards/hrrr_wind_features.py` |
