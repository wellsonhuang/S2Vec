"""
Feature preprocessing for H3 resolution-8 cells.

Steps:
  1. Drop identifier, geometry, label (SoVI, NRI), string, and temporal columns.
  2. Flatten list columns (numeric lists -> mean, other lists -> first element).
  3. Add missing-value indicators and impute by type
     (counts/rates -> 0, continuous variables -> global mean).
  4. Winsorize every numeric column at its 99.5th percentile.
  5. Apply log(1 + x) to columns with skewness > 1 (ground elevation excluded).
  6. Cast all features to Float32 and write the result to Parquet.
"""
import os

import polars as pl
import polars.selectors as cs

os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"] = "load_as_storage"

# ==========================================================
# Configuration
# ==========================================================
INPUT_PATH = "./data/houston_harris_LIST_dataset.parquet"
OUTPUT_PATH = "./data/houston_pre.parquet"

COLS_TO_DROP = [
    "County", "State", "Zipcode",
    "hex_id_l7", "hex_id_l6",
    "name", "brand", "sovi",
    "highest_hazard_dam_name",
]

# Columns whose names contain these keywords are treated as counts/rates (missing -> 0)
ZERO_KEYWORDS = ["count", "_n", "_km", "toxic", "proxy", "tests", "active", "outage",
                 "damage", "rate", "fraction", "ratio", "hr"]
WINSOR_QUANTILE = 0.995
SKEW_THRESHOLD = 1.0
NO_LOG_PREFIXES = ["avg_grnd_elv"]   # can be negative; never log-transformed

NUMERIC_TYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)


if __name__ == "__main__":
    print(f"Reading {INPUT_PATH}")
    try:
        df = pl.read_parquet(INPUT_PATH)
    except Exception as e:
        raise SystemExit(f"Failed to read input file: {e}")

    # ------------------------------------------------------
    # Step 1: drop identifiers, geometry, labels and non-numeric columns
    # ------------------------------------------------------
    print("Dropping identifier, geometry, SoVI and NRI columns")
    df_cleaned = (
        df.drop(COLS_TO_DROP, strict=False)
          .drop(cs.contains("geometry"))
          .drop(cs.contains("nri"))
    )
    string_and_time_cols = [
        c for c, dt in df_cleaned.schema.items()
        if c != "hex_id" and (dt == pl.String or dt == pl.Categorical or dt.is_temporal())
    ]
    df_cleaned = df_cleaned.drop(string_and_time_cols, strict=False)

    # ------------------------------------------------------
    # Step 2: flatten list columns into scalars
    # ------------------------------------------------------
    print("Flattening list columns")
    scalar_exprs = []
    for col, dtype in df_cleaned.schema.items():
        if col == "hex_id":
            continue
        if isinstance(dtype, pl.List):
            if dtype.inner in NUMERIC_TYPES:
                scalar_exprs.append(pl.col(col).list.mean().alias(col))
            else:
                scalar_exprs.append(pl.col(col).list.first().alias(col))
        else:
            scalar_exprs.append(pl.col(col))
    df_scalar = df_cleaned.select(["hex_id"] + scalar_exprs)

    # ------------------------------------------------------
    # Step 3: missing-value indicators and type-specific imputation
    # ------------------------------------------------------
    all_numeric_cols = [c for c, dt in df_scalar.schema.items() if dt.is_numeric() and c != "hex_id"]
    zero_cols = [
        c for c in all_numeric_cols
        if any(kw in c.lower() for kw in ZERO_KEYWORDS) and not c.lower().startswith("avg_")
    ]
    mean_cols = [c for c in all_numeric_cols if c not in zero_cols]

    print("Adding missing-value indicators for mean-imputed columns")
    missing_exprs = [pl.col(c).is_null().cast(pl.Float32).alias(f"{c}_is_missing") for c in mean_cols]
    if missing_exprs:
        df_scalar = df_scalar.with_columns(missing_exprs)

    print("Imputing missing values (counts/rates -> 0, continuous -> mean)")
    if zero_cols:
        df_scalar = df_scalar.with_columns([pl.col(c).fill_null(0).fill_nan(0) for c in zero_cols])
    if mean_cols:
        df_scalar = df_scalar.with_columns(
            [pl.col(c).fill_null(pl.col(c).mean()).fill_nan(pl.col(c).mean()) for c in mean_cols]
        )

    # ------------------------------------------------------
    # Step 4: winsorization
    # ------------------------------------------------------
    print(f"Winsorizing at the {WINSOR_QUANTILE:.1%} quantile")
    quantiles = df_scalar.select(
        [pl.col(c).quantile(WINSOR_QUANTILE).alias(c) for c in all_numeric_cols]
    ).to_dicts()[0]
    clip_exprs = [pl.col(c).clip(upper_bound=quantiles[c]) for c in all_numeric_cols if quantiles[c] is not None]
    if clip_exprs:
        df_scalar = df_scalar.with_columns(clip_exprs)

    # ------------------------------------------------------
    # Step 5: log(1 + x) for right-skewed columns
    # ------------------------------------------------------
    print("Applying log(1 + x) to right-skewed columns")
    skewness = df_scalar.select([pl.col(c).skew().alias(c) for c in all_numeric_cols])
    log1p_cols = [
        c for c in all_numeric_cols
        if skewness[c][0] is not None and skewness[c][0] > SKEW_THRESHOLD
        and not any(c.lower().startswith(p) for p in NO_LOG_PREFIXES)
    ]
    if log1p_cols:
        df_scalar = df_scalar.with_columns([pl.col(c).log1p() for c in log1p_cols])
    print(f"  log-transformed columns: {len(log1p_cols)}")

    # ------------------------------------------------------
    # Step 6: cast to Float32 and save
    # ------------------------------------------------------
    df_final = df_scalar.with_columns([pl.col(c).cast(pl.Float32) for c in df_scalar.columns if c != "hex_id"])

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    df_final.write_parquet(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH} ({df_final.shape[0]:,} rows x {df_final.shape[1]} columns)")