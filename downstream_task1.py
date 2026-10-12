"""
Downstream evaluation of cell embeddings (following the S2Vec protocol).

  - Two-layer MLP regressor trained with MSE and AdamW.
  - Targets are min-max scaled to [0, 1]; cells with multiple labels use the median.
  - 60/20/20 random train/validation/test split, repeated over 20 seeds.
  - In each split, the hyperparameter configuration with the lowest validation loss is selected;
    early stopping is based on the validation loss.
  - Reports test R^2 and MAE (on scaled targets) as mean +/- standard deviation.
  - Rows are sorted by hex_id, so different embeddings are evaluated on identical splits.
"""
import os
import copy
import warnings

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==========================================================
# Configuration
# ==========================================================
EMB_PATH     = "./embedding/Harris_Foundation_Model.parquet"
DATASET_PATH = "./data/houston_harris_LIST_dataset.parquet"
TARGETS      = ["sovi", "nri_cri_value", "nri_cri_score"]
NUM_RUNS     = 20
ENABLE_HPO   = True
MAX_EPOCHS   = 200
PATIENCE     = 15
WEIGHT_DECAY = 1e-4

PARAM_GRID = [
    {"hidden_dim": 64,  "lr": 0.001,  "dropout_rate": 0.1},
    {"hidden_dim": 128, "lr": 0.001,  "dropout_rate": 0.1},
    {"hidden_dim": 256, "lr": 0.001,  "dropout_rate": 0.1},
    {"hidden_dim": 128, "lr": 0.0005, "dropout_rate": 0.2},
]
DEFAULT_CONFIG = {"hidden_dim": 128, "lr": 0.001, "dropout_rate": 0.1}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class S2VecTwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


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


def train_and_validate(X_train, y_train, X_val, y_val, X_test, input_dim, config):
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    batch_size = min(2048, max(64, len(X_train) // 10))
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    model = S2VecTwoLayerMLP(input_dim, config["hidden_dim"], config["dropout_rate"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=WEIGHT_DECAY)
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
            val_loss = criterion(model(X_val_t), y_val_t).item()

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break

    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)

    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t).cpu().numpy()
    return early_stopping.best_loss, test_pred


def evaluate_target(target_name, df, emb_cols, num_runs=20, enable_hpo=True):
    print(f"\nEvaluating target: {target_name} (runs: {num_runs}, HPO: {enable_hpo})")
    if target_name not in df.columns:
        return None

    df_valid = df.with_columns(pl.col(target_name).cast(pl.Float32, strict=False))
    df_valid = df_valid.filter(pl.col(target_name).is_not_null() & ~pl.col(target_name).is_nan())
    print(f"Valid samples: {len(df_valid)}")
    if len(df_valid) < 100:
        print("Insufficient samples, skipping.")
        return None

    X_raw = df_valid.select(emb_cols).cast(pl.Float32).to_numpy()
    y_raw = df_valid.select(target_name).to_numpy().ravel()
    if np.max(y_raw) == np.min(y_raw):
        return None

    param_grid = PARAM_GRID if enable_hpo else [DEFAULT_CONFIG]
    r2_list, mae_list = [], []

    for seed in tqdm(range(num_runs), desc=target_name, unit="run"):
        X_train_val, X_test, y_train_val, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=seed)

        scaler_x = StandardScaler()
        X_train_s = scaler_x.fit_transform(X_train)
        X_val_s = scaler_x.transform(X_val)
        X_test_s = scaler_x.transform(X_test)

        scaler_y = MinMaxScaler(feature_range=(0, 1))
        y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
        y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

        best_val_loss, best_preds = float("inf"), None
        for config in param_grid:
            val_loss, preds = train_and_validate(X_train_s, y_train_s, X_val_s, y_val_s, X_test_s,
                                                 X_raw.shape[1], config)
            if val_loss < best_val_loss:
                best_val_loss, best_preds = val_loss, preds

        r2_list.append(r2_score(y_test_s, best_preds))
        mae_list.append(mean_absolute_error(y_test_s, best_preds))

    r2_mean, r2_std = np.mean(r2_list), np.std(r2_list)
    mae_mean, mae_std = np.mean(mae_list), np.std(mae_list)
    print(f"{target_name}: R2 {r2_mean:.4f} +/- {r2_std:.4f} | MAE {mae_mean:.4f} +/- {mae_std:.4f}")

    return {
        "Target": target_name,
        "Test R2 (mean +/- std)": f"{r2_mean:.4f} +/- {r2_std:.4f}",
        "Test MAE (mean +/- std)": f"{mae_mean:.4f} +/- {mae_std:.4f}",
    }


if __name__ == "__main__":
    print(f"Device: {device}")
    print(f"Embedding: {EMB_PATH}")
    if not (os.path.exists(EMB_PATH) and os.path.exists(DATASET_PATH)):
        raise SystemExit("Embedding or label file not found; check the paths.")

    df_emb = pl.read_parquet(EMB_PATH)
    emb_cols = [c for c in df_emb.columns if c.startswith("emb_")]

    df_full = pl.read_parquet(DATASET_PATH)
    label_exprs = [pl.col("hex_id")]
    for target in TARGETS:
        if target not in df_full.columns:
            continue
        if isinstance(df_full.schema[target], pl.List):
            # A cell overlapping several tracts receives the median label
            label_exprs.append(pl.col(target).list.eval(pl.element().cast(pl.Float64)).list.median().alias(target))
        else:
            label_exprs.append(pl.col(target).cast(pl.Float64))
    df_labels = df_full.select(label_exprs).unique(subset=["hex_id"], keep="first")

    # Sort by hex_id so that the same seed yields identical splits for every embedding
    df_merged = df_labels.join(df_emb, on="hex_id", how="inner").sort("hex_id")
    print(f"Aligned cells: {len(df_merged)}")

    results = []
    for target in TARGETS:
        res = evaluate_target(target, df_merged, emb_cols, num_runs=NUM_RUNS, enable_hpo=ENABLE_HPO)
        if res:
            results.append(res)

    print("\nDownstream evaluation results:")
    if results:
        print(pd.DataFrame(results).to_string(index=False))