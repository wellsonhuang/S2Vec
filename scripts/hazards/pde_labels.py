"""
Task 2 target: mean percent damage estimate (PDE) per H3 resolution-8 cell (Hurricane Harvey, 2017).

Each building point is assigned to the resolution-8 cell that contains it, and the PDE of all points
in a cell is averaged.
"""
import geopandas as gpd
import h3

from common import harris_cells

PDE_PATH    = "data/raw/pde/pde_junwei.shp"        # building points with a "pde" field in [0, 1]
OUTPUT_PATH = "data/harris_county_h3_pde_avg.parquet"
RES = 8

if __name__ == "__main__":
    cells = set(harris_cells())
    gdf = gpd.read_file(PDE_PATH)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    gdf["hex_id"] = [h3.latlng_to_cell(lat, lng, RES) for lat, lng in zip(gdf.geometry.y, gdf.geometry.x)]
    gdf = gdf[gdf["hex_id"].isin(cells)]

    out = gdf.groupby("hex_id", as_index=False)["pde"].mean().rename(columns={"pde": "pde_average"})
    out.to_parquet(OUTPUT_PATH, index=False)
    print(f"{len(gdf):,} points in Harris County -> {len(out):,} of {len(cells):,} cells with PDE; saved {OUTPUT_PATH}")
