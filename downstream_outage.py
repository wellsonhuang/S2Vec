"""
Downstream evaluation: power outages from Hurricane Beryl (Harris County).

Outage labels are reported per ZIP code and assigned to every H3 resolution-8 cell in that ZIP,
so all cells of a ZIP share the same label. This script evaluates each feature set
(raw features or cell embeddings, always combined with the same wind and flood features)
in two ways:

1. Cell-level evaluation (two-layer MLP, S2Vec protocol)
   - Targets are min-max scaled to [0, 1]; features are standardized.
   - In each split, the hyperparameter configuration with the lowest validation loss is selected;
     early stopping is based on the validation loss.
   - Split modes (SPLIT_MODES):
       "random"    : 60/20/20 random split over cells, repeated NUM_RUNS times.
       "zip_kfold" : grouped K-fold over ZIP codes (N_FOLDS x N_REPEATS runs). Cells of a test ZIP
                     never appear in training, which removes label leakage between neighboring cells.
       "zip_split" : grouped 60/20/20 split over ZIP codes, repeated NUM_RUNS times.
   - Reports cell-level R^2 and MAE (scaled target), plus ZIP-level R^2, computed after averaging
     the cell predictions within each test ZIP.

2. ZIP-level evaluation (optional, ZIP_LEVEL_EVAL = True)
   - Features are averaged within each ZIP (wind features also contribute their maximum),
     giving one sample per ZIP.
   - Models suited to small samples: "ridge" (standardize -> PCA -> RidgeCV) and
     "gbm" (shallow gradient-boosted trees).
   - Repeated K-fold over ZIP codes; reports R^2, MAE and Spearman rank correlation.
   - Baselines: "Mean" (predicts the training mean) and "Wind only" (hazard features only).

All settings are in the configuration section below.
"""
import os
import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings("ignore")
os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"] = "load_as_storage"

# ==========================================================
# Configuration
# ==========================================================
MASTER_PATH = Path("data/houston_harris_final_dataset.parquet")       # raw cell features
WIND_PATH   = Path("data/harris_h3l8_wind_features.parquet")          # HRRR wind features
OUTAGE_PATH = Path("data/harris_h3l8_beryl_outage.parquet")           # outage targets per cell
FLOOD_PATH  = Path("data/Harvey_MaxDepth_results.parquet")            # flood-depth features (floodplain cells only)
ZIP_LOOKUP_PATH    = Path("data/hex_county_state_zip_crosswalk.parquet")  # H3 cell -> ZIP code
ZIP_LOOKUP_ZIP_COL = "Zipcode"
ZIP_LOOKUP_COUNTY  = "Harris"          # keep only rows of this county (exact match); None = no filter

# Feature sets: name -> "MASTER" (raw numeric columns of MASTER_PATH) or an embedding file (emb_* columns).
# Wind and flood features are appended to every set.
FEATURE_SETS = {
    "Dataset only": "MASTER",
    "H3-S2Vec":     "embedding/harris_h3_s2vec.parquet",
    "H3-S2Vec-C":   "embedding/harris_h3_s2vec_c.parquet",
}

TARGETS        = ["T1_total_hours", "T2_peak_avg_pct", "T3_days_to_restore"]
LABEL_KEYWORDS = ["sovi", "nri"]       # columns containing these keywords are never used as features
REGIONS        = ["All cells"]         # also available: "Floodplain only", "No floodplain"
MIN_SAMPLES    = 50
SEED           = 42

# --- Cell-level evaluation ---
SPLIT_MODES        = ["random", "zip_kfold"]   # "zip_split" is also available
NUM_RUNS           = 20                        # runs for "random" and "zip_split"
N_FOLDS, N_REPEATS = 5, 4                      # "zip_kfold": 5 folds x 4 repeats = 20 runs
PARAM_GRID = [
    {"hidden_dim": 64,  "lr": 1e-3, "dropout_rate": 0.1},
    {"hidden_dim": 128, "lr": 1e-3, "dropout_rate": 0.1},
    {"hidden_dim": 128, "lr": 5e-4, "dropout_rate": 0.2},
    {"hidden_dim": 256, "lr": 1e-3, "dropout_rate": 0.1},
]
MAX_EPOCHS   = 100
PATIENCE     = 15
WEIGHT_DECAY = 1e-3

# --- ZIP-level evaluation ---
ZIP_LEVEL_EVAL         = True
ZIP_MODELS             = ["ridge", "gbm"]
N_PCA                  = 10            # PCA components before ridge regression
ZIP_FOLDS, ZIP_REPEATS = 5, 10
MIN_CELLS_PER_ZIP      = 3             # ZIP codes with fewer cells are excluded

RESULT_DIR = Path("results")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# Data loading
# ==========================================================
def rename_hex_cols(df):
    for c in ["h3_id", "h3index", "h3"]:
        if c in df.columns and "hex_id" not in df.columns:
            df = df.rename({c: "hex_id"})
    return df


def rename_hex(df):
    return rename_hex_cols(df).unique(subset=["hex_id"], keep="first")


def list_to_median(df):
    """Numeric list columns -> median; other list columns -> first element."""
    exprs = []
    for c, t in df.schema.items():
        if isinstance(t, pl.List):
            if t.inner.is_numeric():
                exprs.append(pl.col(c).list.eval(pl.element().cast(pl.Float64)).list.median().alias(c))
            else:
                exprs.append(pl.col(c).list.first().alias(c))
        else:
            exprs.append(pl.col(c))
    return df.select(exprs)


def numeric_cols(df, exclude=()):
    return [c for c, t in df.schema.items()
            if c != "hex_id" and c not in exclude and (t.is_numeric() or t == pl.Boolean)
            and not any(k in c.lower() for k in LABEL_KEYWORDS)]


def load_zip_lookup():
    """Return one ZIP code (5-digit string) per cell."""
    lk = rename_hex_cols(pl.read_parquet(ZIP_LOOKUP_PATH))
    if ZIP_LOOKUP_COUNTY and "County" in lk.columns:
        names = [ZIP_LOOKUP_COUNTY.lower(), f"{ZIP_LOOKUP_COUNTY.lower()} county"]
        lk = lk.filter(pl.col("County").cast(pl.Utf8).str.strip_chars().str.to_lowercase().is_in(names))
    z = pl.col(ZIP_LOOKUP_ZIP_COL)
    if isinstance(lk.schema[ZIP_LOOKUP_ZIP_COL], pl.List):
        z = z.list.first()
    # Same normalization as the outage table: string, strip ".0", zero-pad to 5 digits
    lk = lk.select(["hex_id", z.cast(pl.Utf8).str.split(".").list.first().str.zfill(5).alias("zip")])
    lk = lk.filter(pl.col("zip").is_not_null() & (pl.col("zip") != "00000"))
    n_multi = lk.group_by("hex_id").agg(pl.col("zip").n_unique()).filter(pl.col("zip") > 1).height
    lk = lk.sort(["hex_id", "zip"]).unique(subset=["hex_id"], keep="first")
    print(f"ZIP lookup: {lk.height:,} cells ({n_multi:,} cells map to more than one ZIP; first one kept)")
    return lk


def build_dataset():
    master = rename_hex(list_to_median(pl.read_parquet(MASTER_PATH)))
    wind = rename_hex(pl.read_parquet(WIND_PATH))
    outage = rename_hex(pl.read_parquet(OUTAGE_PATH))
    flood = rename_hex(pl.read_parquet(FLOOD_PATH))
    zips = load_zip_lookup()

    wind_cols = numeric_cols(wind, exclude=TARGETS)
    flood_cols = numeric_cols(flood, exclude=TARGETS)
    raw_cols = numeric_cols(master, exclude=set(TARGETS) | set(wind_cols) | set(flood_cols))
    print(f"Features: raw {len(raw_cols)}, wind {len(wind_cols)}, flood {len(flood_cols)}")

    # Base table: targets + ZIP + wind + flood (cells outside the floodplain get 0)
    base = outage.select(["hex_id"] + [c for c in TARGETS if c in outage.columns])
    base = base.join(zips, on="hex_id", how="inner")
    base = base.join(wind.select(["hex_id"] + wind_cols), on="hex_id", how="inner")
    base = base.join(flood.select(["hex_id"] + flood_cols), on="hex_id", how="left")
    base = base.with_columns([pl.col(c).fill_null(0) for c in flood_cols] +
                             [pl.col("hex_id").is_in(flood["hex_id"]).alias("has_floodplain")])

    # Feature sets; keep only cells present in every set so all sets share identical splits
    sets = {}
    for name, src in FEATURE_SETS.items():
        if src == "MASTER":
            df = master.select(["hex_id"] + raw_cols).rename({c: f"{name}|{c}" for c in raw_cols})
            sets[name] = (df, [f"{name}|{c}" for c in raw_cols])
        else:
            e = rename_hex(pl.read_parquet(src))
            cols = [c for c in e.columns if c.startswith("emb_")]
            df = e.select(["hex_id"] + cols).rename({c: f"{name}|{c}" for c in cols})
            sets[name] = (df, [f"{name}|{c}" for c in cols])

    common = set(base["hex_id"].to_list())
    for df, _ in sets.values():
        common &= set(df["hex_id"].to_list())
    data = base.filter(pl.col("hex_id").is_in(list(common)))
    for df, _ in sets.values():
        data = data.join(df, on="hex_id", how="left")
    data = data.sort("hex_id")

    n_zip = data["zip"].n_unique()
    print(f"Aligned cells: {data.height:,} in {n_zip} ZIP codes ({data.height / n_zip:.1f} cells per ZIP)")
    return data, sets, wind_cols, flood_cols


# ==========================================================
# Cell-level evaluation (two-layer MLP)
# ==========================================================
class S2VecTwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one(X_train, y_train, X_val, y_val, X_test, config, seed):
    torch.manual_seed(seed)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    X_train_t, y_train_t, X_val_t, y_val_t, X_test_t = t(X_train), t(y_train), t(X_val), t(y_val), t(X_test)

    batch_size = min(2048, max(64, len(X_train) // 10))
    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model = S2VecTwoLayerMLP(X_train.shape[1], config["hidden_dim"], config["dropout_rate"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best_loss, best_state, bad_epochs = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return best_loss, model(X_test_t).cpu().numpy()


def make_splits(n, mode, groups):
    """Return a list of (train_idx, val_idx, test_idx)."""
    idx = np.arange(n)
    splits = []
    if mode == "random":
        for seed in range(NUM_RUNS):
            train_val, test = train_test_split(idx, test_size=0.2, random_state=seed)
            train, val = train_test_split(train_val, test_size=0.25, random_state=seed)
            splits.append((train, val, test))
    elif mode == "zip_split":
        for seed in range(NUM_RUNS):
            train_val, test = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(idx, groups=groups))
            a, b = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed)
                        .split(train_val, groups=groups[train_val]))
            splits.append((train_val[a], train_val[b], test))
    elif mode == "zip_kfold":
        unique_groups = np.unique(groups)
        for r in range(N_REPEATS):
            perm = np.random.default_rng(r).permutation(unique_groups)
            fold_of = {g: i % N_FOLDS for i, g in enumerate(perm)}   # every ZIP is tested exactly once per repeat
            fold = np.array([fold_of[g] for g in groups])
            for k in range(N_FOLDS):
                test, train_val = idx[fold == k], idx[fold != k]
                seed = r * N_FOLDS + k
                a, b = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed)
                            .split(train_val, groups=groups[train_val]))
                splits.append((train_val[a], train_val[b], test))
    else:
        raise ValueError(f"Unknown split mode: {mode}")
    return splits


def zip_level_r2(y_true, y_pred, groups):
    """R^2 after averaging cell predictions within each ZIP (one sample per ZIP)."""
    d = pd.DataFrame({"g": groups, "y": y_true, "p": y_pred}).groupby("g")[["y", "p"]].mean()
    return (r2_score(d["y"], d["p"]) if len(d) >= 3 else np.nan), len(d)


def evaluate_cells(X, y, groups, mode):
    r2s, maes, zip_r2s, n_zips = [], [], [], []
    for run, (tr, va, te) in enumerate(make_splits(len(y), mode, groups)):
        imputer, scaler_x = SimpleImputer(strategy="mean"), StandardScaler()
        X_train = scaler_x.fit_transform(imputer.fit_transform(X[tr]))
        X_val = scaler_x.transform(imputer.transform(X[va]))
        X_test = scaler_x.transform(imputer.transform(X[te]))
        scaler_y = MinMaxScaler().fit(y[tr].reshape(-1, 1))
        scale = lambda a: scaler_y.transform(a.reshape(-1, 1)).ravel()

        best_val, best_pred = float("inf"), None
        for config in PARAM_GRID:
            val_loss, pred = train_one(X_train, scale(y[tr]), X_val, scale(y[va]), X_test, config, run)
            if val_loss < best_val:
                best_val, best_pred = val_loss, pred

        y_test = scale(y[te])
        r2s.append(r2_score(y_test, best_pred))
        maes.append(mean_absolute_error(y_test, best_pred))
        z, nz = zip_level_r2(y_test, best_pred, groups[te])
        zip_r2s.append(z)
        n_zips.append(nz)

    return {"R2_mean": np.mean(r2s), "R2_std": np.std(r2s),
            "MAE_mean": np.mean(maes), "MAE_std": np.std(maes),
            "ZIP_R2_mean": np.nanmean(zip_r2s), "ZIP_R2_std": np.nanstd(zip_r2s),
            "test_ZIPs": np.mean(n_zips), "runs": len(r2s)}


def run_cell_level(data, sets, hazard_cols):
    region_mask = {
        "All cells": np.ones(data.height, bool),
        "Floodplain only": data["has_floodplain"].to_numpy(),
        "No floodplain": ~data["has_floodplain"].to_numpy(),
    }
    groups_all = data["zip"].to_numpy()
    rows = []
    for region in REGIONS:
        for target in TARGETS:
            y_all = data[target].cast(pl.Float64).to_numpy()
            ok = region_mask[region] & np.isfinite(y_all)
            if ok.sum() < MIN_SAMPLES:
                print(f"{region} / {target}: only {ok.sum()} samples, skipped")
                continue
            for mode in SPLIT_MODES:
                for name, (_, cols) in sets.items():
                    X = data.select(cols + hazard_cols).cast(pl.Float64).to_numpy()[ok]
                    r = evaluate_cells(X, y_all[ok], groups_all[ok], mode)
                    rows.append({"Region": region, "Target": target, "Split": mode, "Features": name,
                                 "n_cells": int(ok.sum()), **r})
                    print(f"[cell] {region} | {target} | {mode:9s} | {name:14s} "
                          f"R2 {r['R2_mean']:.4f} +/- {r['R2_std']:.4f} | "
                          f"MAE {r['MAE_mean']:.4f} +/- {r['MAE_std']:.4f} | "
                          f"ZIP-level R2 {r['ZIP_R2_mean']:.4f} +/- {r['ZIP_R2_std']:.4f}")
    return pd.DataFrame(rows)


# ==========================================================
# ZIP-level evaluation
# ==========================================================
def zip_aggregate(df, cols, with_max=False):
    aggs = [pl.col(c).cast(pl.Float64).mean().alias(f"{c}__mean") for c in cols]
    if with_max:
        aggs += [pl.col(c).cast(pl.Float64).max().alias(f"{c}__max") for c in cols]
    return df.group_by("zip").agg(aggs)


def make_zip_model(name, n_features):
    if name == "ridge":
        return make_pipeline(SimpleImputer(), StandardScaler(),
                             PCA(n_components=min(N_PCA, n_features), random_state=SEED),
                             RidgeCV(alphas=np.logspace(-3, 4, 30)))
    if name == "gbm":
        return make_pipeline(SimpleImputer(), HistGradientBoostingRegressor(
            max_depth=2, learning_rate=0.05, max_iter=200, min_samples_leaf=8, random_state=SEED))
    raise ValueError(f"Unknown model: {name}")


def evaluate_zips(X, y, model_name):
    splits = list(RepeatedKFold(n_splits=ZIP_FOLDS, n_repeats=ZIP_REPEATS, random_state=SEED)
                  .split(np.zeros(len(y))))
    r2s, maes, rhos = [], [], []
    for rep in range(ZIP_REPEATS):        # each repeat predicts every ZIP once
        pred = np.zeros_like(y)
        for tr, te in splits[rep * ZIP_FOLDS:(rep + 1) * ZIP_FOLDS]:
            if X is None:
                pred[te] = y[tr].mean()
            else:
                pred[te] = make_zip_model(model_name, X.shape[1]).fit(X[tr], y[tr]).predict(X[te])
        r2s.append(r2_score(y, pred))
        maes.append(mean_absolute_error(y, pred))
        rhos.append(spearmanr(y, pred).correlation if np.std(pred) > 0 else np.nan)
    return {"R2_mean": np.mean(r2s), "R2_std": np.std(r2s), "MAE": np.mean(maes), "Spearman": np.nanmean(rhos)}


def run_zip_level(data, sets, wind_cols, flood_cols):
    labels = data.group_by("zip").agg([pl.col(t).mean() for t in TARGETS] + [pl.len().alias("n_cells")])
    flood_df = data.with_columns(pl.col("has_floodplain").cast(pl.Float64).alias("flood_frac"))
    hazard = zip_aggregate(data, wind_cols, with_max=True).join(
        zip_aggregate(flood_df, flood_cols + ["flood_frac"]), on="zip")
    hazard_cols = [c for c in hazard.columns if c != "zip"]

    zdf = labels.join(hazard, on="zip").filter(pl.col("n_cells") >= MIN_CELLS_PER_ZIP)
    for name, (_, cols) in sets.items():
        zdf = zdf.join(zip_aggregate(data, cols), on="zip", how="left")
    zdf = zdf.sort("zip")
    print(f"\nZIP-level table: {zdf.height} ZIP codes (at least {MIN_CELLS_PER_ZIP} cells each)")

    feature_sets = {"Wind only": []}
    feature_sets.update({name: [f"{c}__mean" for c in cols] for name, (_, cols) in sets.items()})

    rows = []
    for target in TARGETS:
        y_all = zdf[target].cast(pl.Float64).to_numpy()
        ok = np.isfinite(y_all)
        y = y_all[ok]
        r = evaluate_zips(None, y, "mean")
        rows.append({"Target": target, "Features": "Mean (baseline)", "Model": "-", "n_zip": int(ok.sum()), **r})
        for name, cols in feature_sets.items():
            X = zdf.select(hazard_cols + cols).cast(pl.Float64).to_numpy()[ok]
            for model in ZIP_MODELS:
                r = evaluate_zips(X, y, model)
                rows.append({"Target": target, "Features": name, "Model": model, "n_zip": int(ok.sum()), **r})
                print(f"[zip]  {target} | {name:14s} | {model:5s} R2 {r['R2_mean']:.4f} +/- {r['R2_std']:.4f} | "
                      f"MAE {r['MAE']:.4f} | Spearman {r['Spearman']:.3f}")
    return pd.DataFrame(rows)


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":
    print(f"Device: {device}")
    data, sets, wind_cols, flood_cols = build_dataset()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    cell_res = run_cell_level(data, sets, wind_cols + flood_cols)
    cell_res.to_csv(RESULT_DIR / "outage_cell_level.csv", index=False)

    # Summary: one row per target x feature set, with each split mode side by side
    fmt = lambda m, s: f"{m:.4f} +/- {s:.4f}"
    cell_res["cell R2"] = [fmt(a, b) for a, b in zip(cell_res.R2_mean, cell_res.R2_std)]
    cell_res["ZIP-level R2"] = [fmt(a, b) for a, b in zip(cell_res.ZIP_R2_mean, cell_res.ZIP_R2_std)]
    for region in cell_res.Region.unique():
        wide = cell_res[cell_res.Region == region].pivot_table(
            index=["Target", "Features"], columns="Split", values=["cell R2", "ZIP-level R2"], aggfunc="first")
        wide.columns = [f"{split} | {metric}" for metric, split in wide.columns]
        print(f"\nCell-level results ({region}):")
        print(wide.reindex(sorted(wide.columns), axis=1).to_string())

    if ZIP_LEVEL_EVAL:
        zip_res = run_zip_level(data, sets, wind_cols, flood_cols)
        zip_res.to_csv(RESULT_DIR / "outage_zip_level.csv", index=False)
        print("\nZIP-level results:")
        print(zip_res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nSaved results to {RESULT_DIR}/")