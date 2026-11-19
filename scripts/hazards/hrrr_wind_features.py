"""
Task 3, step 2: HRRR wind features per H3 resolution-8 cell (Hurricane Beryl, 2024).

Aggregates the hourly HRRR points over time, then assigns each cell center the values of its nearest
HRRR grid point (KD-tree):
  wx_gust_max     maximum gust (m/s)
  wx_wind_max     maximum sustained wind speed (m/s)
  wx_hours_gt40   number of hours with sustained wind >= 40 mph (17.88 m/s)
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from common import harris_cells, cell_centers

POINTS_PATH = "data/interim/hrrr_wind_points.parquet"
OUTPUT_PATH = "data/harris_h3l8_wind_features.parquet"
THRESHOLD_MS = 40 * 0.44704

if __name__ == "__main__":
    wind = pd.read_parquet(POINTS_PATH)
    wind["is_gt40"] = (wind["wind_speed_ms"] >= THRESHOLD_MS).astype(int)
    pts = wind.groupby(["latitude", "longitude"]).agg(
        wx_gust_max=("gust_ms", "max"),
        wx_wind_max=("wind_speed_ms", "max"),
        wx_hours_gt40=("is_gt40", "sum"),
    ).reset_index()

    hex_ids = harris_cells()
    _, nearest = cKDTree(pts[["latitude", "longitude"]].values).query(np.array(cell_centers(hex_ids)))

    out = pd.DataFrame({"hex_id": hex_ids})
    for c in ["wx_gust_max", "wx_wind_max", "wx_hours_gt40"]:
        out[c] = pts[c].values[nearest]
    out.to_parquet(OUTPUT_PATH, index=False)
    print(f"{len(pts)} HRRR points -> {len(out):,} cells; saved {OUTPUT_PATH}")
