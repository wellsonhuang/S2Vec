import os
import numpy as np
os.environ['POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR'] = 'load_as_storage'

import polars as pl
import polars.selectors as cs

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    file_path = './data/texas_LIST_dataset.parquet'
    output_file = './data/texas_FLATTENED_FILLED_dataset_final.parquet'
    
    print("Loading raw dataset...")
    
    try:
        df = pl.read_parquet(file_path)
    except Exception as e:
        print(f"Failed to read file: {e}")
        exit()
        
    original_columns = set(df.columns)

    # ---------------------------------------------------------
    # Step 1: Remove basic identifiers, geometry, and all NRI features
    # ---------------------------------------------------------
    cols_to_drop = [
        "County", "State", "Zipcode", 
        "hex_id_l7", "hex_id_l6", 
        "name", "brand", "sovi",
        "highest_hazard_dam_name" # String type outlier feature, drop directly
    ]

    print("Dropping basic labels, geometry, and NRI features...")
    df_cleaned = (
        df.drop(cols_to_drop, strict=False)
          .drop(cs.contains("geometry"))
          .drop(cs.contains("nri"))
    )

    # ---------------------------------------------------------
    # Step 2: List Flattening -> Convert array features to scalar statistics
    # ---------------------------------------------------------
    print("Flattening list arrays into scalar values...")
    numeric_types = (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    )

    scalar_exprs = []
    for col, dtype in df_cleaned.schema.items():
        if col == "hex_id": 
            continue
        if isinstance(dtype, pl.List):
            if dtype.inner in numeric_types:
                scalar_exprs.append(pl.col(col).list.mean())   # Mean for numeric arrays
            else:
                scalar_exprs.append(pl.col(col).list.first())  # First element for non-numeric arrays
        else:
            scalar_exprs.append(pl.col(col))

    df_scalar = df_cleaned.select(["hex_id"] + scalar_exprs)

    # ---------------------------------------------------------
    # Step 3: Feature Stratification and Missing Value Imputation
    # ---------------------------------------------------------
    all_numeric_cols = [c for c, dt in df_scalar.schema.items() if dt.is_numeric() and c != "hex_id"]
    
    # 1. Strict zero-fill list (exclude population, sqft to avoid creating false spatial signals)
    zero_keywords = ["count", "_n", "_km", "toxic", "proxy", "tests", "active", "outage"]
    zero_cols = [
        c for c in all_numeric_cols 
        if any(kw in c.lower() for kw in zero_keywords)
        and not c.lower().startswith("avg_")
    ]
    
    # 2. Mean-fill list (remaining continuous variables; will become 0 after standardization)
    mean_cols = [c for c in all_numeric_cols if c not in zero_cols]

    print("Executing stratified missing value imputation (Zero, Mean)...")

    if zero_cols:
        df_scalar = df_scalar.with_columns(
            [pl.col(c).fill_null(0).fill_nan(0) for c in zero_cols]
        )
        print(f"  [Phase 1/2] Filled {len(zero_cols)} 'physical quantity and facility' features with 0")

    if mean_cols:
        df_scalar = df_scalar.with_columns(
            [pl.col(c).fill_null(pl.col(c).mean()).fill_nan(pl.col(c).mean()) for c in mean_cols]
        )
        print(f"  [Phase 2/2] Filled remaining {len(mean_cols)} 'continuous/population/area' features with global mean")

    # ---------------------------------------------------------
    # Step 4: Log1p Transformation (for heavily right-skewed features)
    # ---------------------------------------------------------
    print("Applying Log1p transformation to highly skewed features...")
    log1p_keywords = zero_keywords + ["population", "sqft", "income", "aaod", "d_final", "meters_out", "avg_val_struct", "damage"]
    log1p_cols = [
        c for c in all_numeric_cols 
        if any(kw in c.lower() for kw in log1p_keywords) and not c.lower().startswith("avg_grnd_elv")
    ]

    if log1p_cols:
        df_scalar = df_scalar.with_columns([pl.col(c).log1p() for c in log1p_cols])
        print(f"  Completed ln(1+x) transformation for {len(log1p_cols)} long-tail features")

    # ---------------------------------------------------------
    # Step 5: Global Z-score Standardization
    # ---------------------------------------------------------
    print("Executing global feature Z-score standardization...")
    feature_cols = [col for col in df_scalar.columns if col != "hex_id"]
    eps = 1e-8
    
    exprs = []
    for col in feature_cols:
        col_expr = pl.col(col)
        mean_expr = col_expr.mean()
        std_expr = col_expr.std()
        
        # Prevent division by zero: if standard deviation is near 0, set feature value to 0
        z_score_expr = pl.when(std_expr < eps).then(0.0).otherwise((col_expr - mean_expr) / std_expr)
        exprs.append(z_score_expr.alias(col))
        
    df_final = df_scalar.with_columns(exprs)
    print("  Z-score standardization complete.")

    # ---------------------------------------------------------
    # Step 6: Export Final Results and Print Audit Report
    # ---------------------------------------------------------
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_final.write_parquet(output_file)

    print("\nData flattening, imputation, log transformation, and standardization completed successfully.")
    print(f"Processed file saved to: {output_file}")
    print(f"Final data dimensions: {df_final.shape[0]:,} rows x {df_final.shape[1]} columns")

    # Calculate residual null values
    total_nulls = df_final.select(cs.numeric().is_null().sum()).sum_horizontal().item(0)
    print(f"Residual null values in numeric features: {total_nulls}")

    # ===== Print Feature Processing Audit Report =====
    print("\n" + "="*80)
    print(" Data Cleaning and Imputation Audit Report")
    print("="*80)
    
    deleted_cols = sorted(list(original_columns - set(df_cleaned.columns)))
    print(f"[Force Dropped] Basic labels/geometry/NRI fields ({len(deleted_cols)} total):")
    print("   " + ", ".join(deleted_cols[:15]) + ("..." if len(deleted_cols) > 15 else ""))
    print("-" * 80)

    print(f"[Strategy 1: Zero-fill] Absolute presence features (e.g., facility counts, damages) ({len(zero_cols)} total):")
    if zero_cols:
        sorted_zero = sorted(zero_cols)
        for i in range(0, len(sorted_zero), 3):
            print("   " + ", ".join(sorted_zero[i:i+3]))
    print("-" * 80)

    print(f"[Strategy 2: Mean-fill] Continuous variables (e.g., population, area metrics) ({len(mean_cols)} total):")
    if mean_cols:
        sorted_mean = sorted(mean_cols)
        for i in range(0, len(sorted_mean), 3):
            print("   " + ", ".join(sorted_mean[i:i+3]))
    print("-" * 80)
    
    print(f"[Strategy 3: Log1p Transformation] Handling long-tail distribution ({len(log1p_cols)} total):")
    if log1p_cols:
        sorted_log1p = sorted(log1p_cols)
        for i in range(0, len(sorted_log1p), 3):
            print("   " + ", ".join(sorted_log1p[i:i+3]))
    print("="*80)