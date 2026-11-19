"""
Task 3, step 1: extract hourly HRRR surface wind for Hurricane Beryl (2024).

Reads HRRR surface analyses (wrfsfcf00) from HRRR_DIR/<MMDD>/hrrr.t<HH>z.wrfsfcf00.grib2 for
2024-07-06 01:00 UTC to 2024-07-12 00:00 UTC (inclusive) and keeps grid points inside a bounding box
around Harris County. For each point and hour it stores the gust and the
10-m wind components; sustained wind speed = sqrt(u^2 + v^2).
"""
import argparse
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pygrib

HRRR_DIR    = "data/raw/hrrr"
OUTPUT_PATH = "data/interim/hrrr_wind_points.parquet"
LAT_MIN, LAT_MAX = 29.4, 30.3
LON_MIN, LON_MAX = -96.0, -94.9
YEAR  = 2024
START = datetime(2024, 7, 6, 1)
END   = datetime(2024, 7, 12, 0)

_ap = argparse.ArgumentParser()
_ap.add_argument("--hrrr-dir", default=HRRR_DIR)
HRRR_DIR = _ap.parse_known_args()[0].hrrr_dir

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    mask = lats = lons = None
    frames = []
    for folder in sorted(glob.glob(os.path.join(HRRR_DIR, "07*"))):
        mmdd = os.path.basename(folder)
        for path in sorted(glob.glob(os.path.join(folder, "hrrr.t*z.wrfsfcf00.grib2"))):
            hour = int(os.path.basename(path).split(".t")[1].split("z.")[0])
            if not START <= datetime(YEAR, int(mmdd[:2]), int(mmdd[2:]), hour) <= END:
                continue
            try:
                grbs = pygrib.open(path)
                gust = grbs.select(shortName="gust")[0]
                if mask is None:
                    la, lo = gust.latlons()
                    mask = (la >= LAT_MIN) & (la <= LAT_MAX) & (lo >= LON_MIN) & (lo <= LON_MAX)
                    lats, lons = la[mask], lo[mask]
                u = grbs.select(shortName="10u")[0].values[mask]
                v = grbs.select(shortName="10v")[0].values[mask]
                frames.append(pd.DataFrame({
                    "date_folder": os.path.basename(folder),
                    "hour_z": os.path.basename(path).split(".t")[1].split("z.")[0],
                    "latitude": lats,
                    "longitude": lons,
                    "gust_ms": gust.values[mask],
                    "u_wind_ms": u,
                    "v_wind_ms": v,
                    "wind_speed_ms": np.sqrt(u ** 2 + v ** 2),
                }))
                grbs.close()
            except Exception as e:
                print(f"Skipping {path}: {e}")

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"{len(frames)} hourly files, {len(lats)} grid points; saved {OUTPUT_PATH}")
