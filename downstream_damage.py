"""
Downstream evaluation: building damage from Hurricane Harvey (Harris County).

Compares two feature sets, each combined with the same hazard features
(MRMS rainfall features and flood-depth features):
  - Dataset only: raw cell features
  - Embedding:    cell embeddings

Evaluation follows the S2Vec protocol:
  - Two-layer MLP regressor trained with MSE and AdamW.
  - Target (mean PDE per cell) is min-max scaled to [0, 1].
  - 60/20/20 random train/validation/test split, repeated over 20 seeds.
  - In each split, the hyperparameter configuration with the lowest validation loss is selected;
    early stopping is based on the validation loss.
  - Reports test MAE (scaled target) and R^2 as mean +/- standard deviation.

Results are reported for three subsets: cells inside the floodplain, cells outside it, and all cells.
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
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"] = "load_as_storage"

# ==========================================================
# Configuration
# ==========================================================
MASTER_PATH = Path("predict/harris_final2.parquet")              # raw cell features
EMB_PATH    = Path("Harris_Foundation_Model.parquet")            # cell embeddings
RAIN_PATH   = Path("harris_county_h3_rain_features.parquet")     # MRMS rainfall features
FLOOD_PATH  = Path("Harvey_MaxDepth_results.parquet")            # flood-depth features (floodplain cells only)
PDE_PATH    = Path("harris_county_h3_pde_avg.parquet")           # target: mean PDE per cell
RESULT_CSV  = Path("results/damage_downstream.csv")

NUM_RUNS     = 20
MIN_SAMPLES  = 50
MAX_EPOCHS   = 100
PATIENCE     = 15
WEIGHT_DECAY = 1e-3
PARAM_GRID = [
    {"hidden_dim": 64,  "lr": 1e-3, "dropout_rate": 0.1},
    {"hidden_dim": 128, "lr": 1e-3, "dropout_rate": 0.1},
    {"hidden_dim": 128, "lr": 5e-4, "dropout_rate": 0.2},
    {"hidden_dim": 256, "lr": 1e-3, "dropout_rate": 0.1},
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# Model and early stopping
# ==========================================================
class S2VecTwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PyTorchEarlyStopping:
    def __init__(self, patience=15, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ==========================================================
# Training and evaluation
# ==========================================================
def train_and_eval_single_config(X_train, y_train, X_val, y_val, X_test, config):
    t_x_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    t_y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    t_x_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    t_y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    t_x_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    batch_size = min(2048, max(64, len(X_train) // 10))
    train_loader = DataLoader(TensorDataset(t_x_train, t_y_train), batch_size=batch_size, shuffle=True)

    model = S2VecTwoLayerMLP(X_train.shape[1], config["hidden_dim"], config["dropout_rate"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    early_stopping = PyTorchEarlyStopping(patience=PATIENCE)

    for _ in range(MAX_EPOCHS):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(t_x_val), t_y_val).item()

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break

    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)

    model.eval()
    with torch.no_grad():
        test_pred = model(t_x_test).cpu().numpy()
    return early_stopping.best_loss, test_pred


def run_downstream_eval(exp_name, region_name, target_name, df, feature_cols, num_runs=20):
    df_valid = df.filter(pl.col(target_name).is_not_null() & ~pl.col(target_name).is_nan())
    if len(df_valid) < MIN_SAMPLES:
        print(f"[{region_name} | {exp_name}] too few samples ({len(df_valid)}), skipping.")
        return None

    X_raw = df_valid.select(feature_cols).to_numpy()
    y_raw = df_valid.select(target_name).to_numpy().ravel()

    mae_list, r2_list = [], []
    for seed in tqdm(range(num_runs), desc=f"[{region_name}] {exp_name}", leave=False):
        X_temp, X_test, y_temp, y_test = train_test_split(X_raw, y_raw, test_size=0.20, random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=seed)

        imp = SimpleImputer(strategy="mean")
        scaler_x = StandardScaler()
        X_train_s = scaler_x.fit_transform(imp.fit_transform(X_train))
        X_val_s = scaler_x.transform(imp.transform(X_val))
        X_test_s = scaler_x.transform(imp.transform(X_test))

        scaler_y = MinMaxScaler(feature_range=(0, 1))
        y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
        y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

        best_val_loss, best_pred = float("inf"), None
        for config in PARAM_GRID:
            val_loss, pred = train_and_eval_single_config(X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, config)
            if val_loss < best_val_loss:
                best_val_loss, best_pred = val_loss, pred

        mae_list.append(mean_absolute_error(y_test_s, best_pred))
        y_pred_orig = scaler_y.inverse_transform(best_pred.reshape(-1, 1)).ravel()
        r2_list.append(r2_score(y_test, y_pred_orig))

    return {
        "Region": region_name,
        "Target": target_name,
        "Setting": exp_name,
        "MAE": f"{np.mean(mae_list):.4f} +/- {np.std(mae_list):.4f}",
        "R2": f"{np.mean(r2_list):.4f} +/- {np.std(r2_list):.4f}",
    }


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":
    print(f"Device: {device}")

    # 1. Raw features (list columns -> first element)
    df_master = pl.read_parquet(MASTER_PATH)
    exprs_raw = [pl.col(c).list.get(0).alias(c) if isinstance(t, pl.List) else pl.col(c)
                 for c, t in df_master.schema.items() if c != "hex_id"]
    df_master_clean = df_master.select(["hex_id"] + exprs_raw)

    # 2. Embeddings (a single list column is expanded into separate columns)
    df_emb = pl.read_parquet(EMB_PATH)
    list_cols = [c for c, t in df_emb.schema.items() if isinstance(t, pl.List) and c != "hex_id"]
    if list_cols:
        dim = df_emb[list_cols[0]].list.len().max()
        df_emb = df_emb.with_columns(
            pl.col(list_cols[0]).list.to_struct(fields=[f"emb_{i}" for i in range(dim)])
        ).unnest(list_cols[0])

    # 3. Flood-depth features; every cell in this table lies in the floodplain
    df_flood = pl.read_parquet(FLOOD_PATH)
    for c in ["h3_id", "h3index"]:
        if c in df_flood.columns:
            df_flood = df_flood.rename({c: "hex_id"})
    df_flood = df_flood.with_columns(pl.lit(True).alias("has_floodplain"))

    # 4. Rainfall features and PDE target
    df_rain = pl.read_parquet(RAIN_PATH)
    df_pde = pl.read_parquet(PDE_PATH)
    pde_col = [c for c in df_pde.columns if c != "hex_id"][0]
    df_pde = df_pde.rename({pde_col: "target_pde"})

    # 5. Cells present in the raw features, embeddings, rainfall and PDE tables
    df_base = (
        df_master_clean
        .join(df_emb, on="hex_id", how="inner")
        .join(df_rain, on="hex_id", how="inner")
        .join(df_pde, on="hex_id", how="inner")
    )

    # 6. Left join flood depth; cells outside the floodplain get 0 and has_floodplain = False
    df_merged = df_base.join(df_flood, on="hex_id", how="left")
    flood_cols = [c for c in df_flood.columns if c not in ["hex_id", "has_floodplain"]]
    df_merged = df_merged.with_columns(
        [pl.col(c).fill_null(0.0) for c in flood_cols] + [pl.col("has_floodplain").fill_null(False)]
    )
    print(f"Aligned cells: {df_merged.height:,}")

    # Feature sets
    raw_cols = [c for c in df_master_clean.columns if c != "hex_id"]
    emb_cols = [c for c in df_emb.columns if c != "hex_id"]
    rain_cols = [c for c in df_rain.columns if c != "hex_id"]
    features_raw = raw_cols + rain_cols + flood_cols
    features_emb = emb_cols + rain_cols + flood_cols

    regions = {
        "Floodplain only": df_merged.filter(pl.col("has_floodplain")),
        "No floodplain": df_merged.filter(~pl.col("has_floodplain")),
        "All cells": df_merged,
    }

    target = "target_pde"
    results = []
    for region_name, df_region in regions.items():
        print(f"\nRegion: {region_name} ({df_region.height:,} cells)")
        res = run_downstream_eval("Dataset only", region_name, target, df_region, features_raw, NUM_RUNS)
        if res:
            results.append(res)
        res = run_downstream_eval("Embedding", region_name, target, df_region, features_emb, NUM_RUNS)
        if res:
            results.append(res)

    df_res = pd.DataFrame(results)[["Region", "Target", "Setting", "MAE", "R2"]]
    print("\nDamage prediction results:")
    print(df_res.to_string(index=False))
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(RESULT_CSV, index=False)
    print(f"Saved {RESULT_CSV}")