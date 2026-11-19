"""
Task 3 targets: Hurricane Beryl (2024) power outages per H3 resolution-8 cell.

Outage metrics are reported per ZIP code (ZCTA). Each cell receives the mean value of the ZIP codes it
belongs to according to the H3-ZIP crosswalk:
  T1_total_hours, T2_peak_avg_pct, T3_days_to_restore
"""
import geopandas as gpd
import pandas as pd

from common import load_crosswalk

OUTAGE_PATH = "data/raw/outage/zcta_hc_beryl_outage.gpkg"
OUTPUT_PATH = "data/harris_h3l8_beryl_outage.parquet"
TARGETS = ["T1_total_hours", "T2_peak_avg_pct", "T3_days_to_restore"]

if __name__ == "__main__":
    outage = gpd.read_file(OUTAGE_PATH)
    outage["zip"] = outage["ZCTA5CE20"].astype(str).str.zfill(5)

    cw = load_crosswalk().to_pandas()
    cw["zip"] = cw["Zipcode"].fillna("").astype(str).str.split(".").str[0].str.zfill(5)

    merged = cw.merge(outage[["zip"] + TARGETS], on="zip", how="inner")
    out = merged.groupby("hex_id")[TARGETS].mean().reset_index()
    out.to_parquet(OUTPUT_PATH, index=False)
    print(f"{outage['zip'].nunique()} ZIP codes -> {len(out):,} cells; saved {OUTPUT_PATH}")
