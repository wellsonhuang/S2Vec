"""
Build the cell-level feature table for a study area (Section 3.2, "Data sources and integration").

Input: a folder of H3-indexed layers (one Parquet file per layer, each with a "hex_id" column at
resolution 8) plus the hex-county-state-ZIP crosswalk (data/raw/). Steps:
  1. Select the cells of the study area from the crosswalk.
  2. For each layer: keep rows of the study area, collect multiple records of the same cell into lists
     (group_by hex_id + agg all), and left-join onto the cell table. Column-name collisions get the
     layer name as a suffix.
  3. Write the joined table. List columns are flattened later by src/preprocessing.py.

Layers are joined one at a time and written to disk between steps to keep memory bounded. If the run
is interrupted, running the same command again resumes after the last finished layer.
Requires polars >= 1.25.

Usage (from the repository root):
    python scripts/data/build_dataset.py --region houston_harris|texas [--layer-dir DIR]
"""
import argparse
import gc
import os
import shutil
from pathlib import Path

os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"] = "load_as_storage"

import polars as pl
import polars.selectors as cs

# ==========================================================
# Configuration
# ==========================================================
REGION    = "houston_harris"               # "houston_harris" | "texas"
LAYER_DIR = Path("data/raw/layers")        # one Parquet file per layer
CROSSWALK = Path("data/raw/hex_county_state_zip_crosswalk.parquet")

REGIONS = {
    "houston_harris": {"filter": (pl.col("County") == "Harris County") & (pl.col("State") == "Texas"),
                       "output": Path("data/houston_harris_dataset.parquet")},
    "texas": {"filter": pl.col("State") == "Texas",
              "output": Path("data/texas_dataset.parquet")},
}
KEEP_COLS = ["hex_id", "hex_id_l7", "hex_id_l6"]

_ap = argparse.ArgumentParser()
_ap.add_argument("--region", choices=["houston_harris", "texas"], default=REGION)
_ap.add_argument("--layer-dir", type=Path, default=LAYER_DIR)
_args = _ap.parse_known_args()[0]
REGION, LAYER_DIR = _args.region, _args.layer_dir


if __name__ == "__main__":
    cfg = REGIONS[REGION]
    tmp_dir = Path(f"data/tmp_build_{REGION}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    base = (
        pl.scan_parquet(CROSSWALK)
        .filter(cfg["filter"])
        .select(KEEP_COLS)
        .unique(subset=["hex_id"], keep="first")
        .collect()
    )
    cells = base.get_column("hex_id").to_list()
    print(f"{REGION}: {len(cells):,} cells")

    # master_<k>.parquet = cell table after joining the first k layers; resume from the latest one
    done = sorted(int(f.stem.split("_")[1]) for f in tmp_dir.glob("master_*.parquet"))
    k0 = done[-1] if done else 0
    current = tmp_dir / f"master_{k0}.parquet"
    if k0 == 0:
        base.write_parquet(current)
    else:
        print(f"Resuming after layer {k0}")

    layers = sorted(f for f in LAYER_DIR.glob("*.parquet") if f.name != CROSSWALK.name)
    for i, path in enumerate(layers, start=1):
        if i <= k0:
            continue
        nxt = tmp_dir / f"master_{i}.parquet"
        if "hex_id" not in pl.read_parquet_schema(path):
            print(f"[{i}/{len(layers)}] {path.name}: no hex_id, skipped", flush=True)
            shutil.copy(current, nxt.with_suffix(".tmp"))
        else:
            print(f"[{i}/{len(layers)}] {path.name}", flush=True)
            layer = (
                pl.scan_parquet(path)
                .filter(pl.col("hex_id").is_in(cells))
                .group_by("hex_id")
                .agg(pl.all())
            )
            merged = (
                pl.scan_parquet(current)
                .join(layer, on="hex_id", how="left", coalesce=True, suffix=f"_{path.stem}")
                .drop(cs.contains("__index_level_0__"))
            )
            merged.collect(engine="streaming").write_parquet(nxt.with_suffix(".tmp"))
        nxt.with_suffix(".tmp").rename(nxt)     # only complete files are named master_*.parquet
        current.unlink()
        current = nxt
        gc.collect()

    cfg["output"].parent.mkdir(parents=True, exist_ok=True)
    shutil.move(current, cfg["output"])
    shutil.rmtree(tmp_dir)
    print(f"Saved {cfg['output']}")
