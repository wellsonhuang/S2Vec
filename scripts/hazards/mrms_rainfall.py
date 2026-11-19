"""
Task 2 hazard features: Hurricane Harvey rainfall from NOAA MRMS GaugeCorr_QPE_01H.

Hourly files cover 25 Aug 2017 01:00 UTC to 31 Aug 2017 00:00 UTC. Each cell center is matched to the
MRMS grid pixel that contains it. Features per cell:
  rain_total                total rainfall (mm)
  rain_max_6h/12h/24h       maximum rolling 6-, 12-, 24-hour accumulation (mm)
  hours_gt_10mm             longest run of consecutive hours with rainfall > 10 mm
"""
import glob

import numpy as np
import pandas as pd
import rasterio

from common import harris_cells, cell_centers

MRMS_GLOB   = "data/raw/mrms/GaugeCorr_QPE_01H_00.00_*.grib2.gz"
OUTPUT_PATH = "data/harris_county_h3_rain_features.parquet"
HEAVY_MM    = 10.0


def longest_run_above(series, threshold):
    best = cur = 0
    for v in series > threshold:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def max_rolling_sum(s, window):
    return float(s.rolling(window).sum().max()) if len(s) >= window else float(s.sum())


if __name__ == "__main__":
    hex_ids = harris_cells()
    centers = cell_centers(hex_ids)
    files = sorted(glob.glob(MRMS_GLOB))
    print(f"{len(hex_ids)} cells, {len(files)} hourly MRMS files")

    with rasterio.open(f"/vsigzip/{files[0]}") as src:
        idx = [src.index(lng + 360 if (src.bounds.left > 0 and lng < 0) else lng, lat) for lat, lng in centers]
    rows = np.array([r for r, _ in idx])
    cols = np.array([c for _, c in idx])

    hourly = []
    for f in files:
        with rasterio.open(f"/vsigzip/{f}") as src:
            hourly.append(np.maximum(src.read(1)[rows, cols], 0.0))   # negative values are no-data flags
    rain = np.array(hourly)                                          # (hours, cells)

    records = []
    for i, hid in enumerate(hex_ids):
        series = rain[:, i]
        s = pd.Series(series)
        records.append({
            "hex_id": hid,
            "rain_total": round(float(series.sum()), 2),
            "rain_max_6h": round(max_rolling_sum(s, 6), 2),
            "rain_max_12h": round(max_rolling_sum(s, 12), 2),
            "rain_max_24h": round(max_rolling_sum(s, 24), 2),
            "hours_gt_10mm": longest_run_above(series, HEAVY_MM),
        })

    pd.DataFrame(records).to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}")
