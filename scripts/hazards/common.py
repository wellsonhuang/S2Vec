"""Shared helpers for building hazard features and labels for Harris County, TX."""
import h3
import polars as pl

CROSSWALK_PATH = "data/raw/hex_county_state_zip_crosswalk.parquet"   # H3 res-8 cell -> county, state, ZIP
COUNTY, STATE = "Harris County", "Texas"


def load_crosswalk(path=CROSSWALK_PATH):
    """Crosswalk rows for Harris County, TX (exact match; excludes e.g. Harrison County or Harris County, GA)."""
    return pl.read_parquet(path).filter((pl.col("County") == COUNTY) & (pl.col("State") == STATE))


def harris_cells(path=CROSSWALK_PATH):
    return sorted(load_crosswalk(path)["hex_id"].unique().to_list())


def cell_centers(hex_ids):
    """(lat, lng) of each H3 cell center."""
    return [h3.cell_to_latlng(h) for h in hex_ids]
