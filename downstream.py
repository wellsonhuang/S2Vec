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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluation device: {device}")


class S2VecTwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.mlp(x).squeeze(-1)


class PyTorchEarlyStopping:
    def __init__(self, patience=15, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
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


def train_and_validate(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, input_dim, config, max_epochs=200):
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    val_x_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    val_y_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    test_x_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    
    train_size = len(X_train_scaled)
    batch_size = min(2048, max(64, train_size // 10)) 
    
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), 
        batch_size=batch_size, 
        shuffle=True
    )
    
    model = S2VecTwoLayerMLP(input_dim, config['hidden_dim'], config['dropout_rate']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    criterion = nn.MSELoss()
    early_stopping = PyTorchEarlyStopping(patience=15)
    
    for epoch in range(max_epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x_t)
            val_loss = criterion(val_pred, val_y_t).item()
            
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break
            
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)
        
    model.eval()
    with torch.no_grad():
        test_pred = model(test_x_t).cpu().numpy()
        best_val_loss = early_stopping.best_loss
        
    return best_val_loss, test_pred


def evaluate_target_task_paper_standard(target_name, df, emb_cols, num_runs=20, enable_hpo=True):
    print(f"\nEvaluating target: {target_name} (Runs: {num_runs}, HPO: {enable_hpo})")
    
    if target_name not in df.columns:
        return None
        
    df_valid = df.with_columns(pl.col(target_name).cast(pl.Float32, strict=False))
    df_valid = df_valid.filter(pl.col(target_name).is_not_null() & ~pl.col(target_name).is_nan())
    
    valid_count = len(df_valid)
    print(f"Valid samples: {valid_count}")
    if valid_count < 100: 
        print("Insufficient samples, skipping evaluation.")
        return None

    X_raw = df_valid.select(emb_cols).cast(pl.Float32).to_numpy()
    y_raw = df_valid.select(target_name).to_numpy().ravel()

    if np.max(y_raw) == np.min(y_raw):
        return None

    param_grid = [
        {'hidden_dim': 64,  'lr': 0.001, 'dropout_rate': 0.1},
        {'hidden_dim': 128, 'lr': 0.001, 'dropout_rate': 0.1},
        {'hidden_dim': 256, 'lr': 0.001, 'dropout_rate': 0.1},
        {'hidden_dim': 128, 'lr': 0.0005, 'dropout_rate': 0.2},
    ] if enable_hpo else [{'hidden_dim': 128, 'lr': 0.001, 'dropout_rate': 0.1}]

    r2_list = []
    mae_list = []
    
    for seed in tqdm(range(num_runs), desc=f"Runs for {target_name}", unit="run"):
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_raw, y_raw, test_size=0.2, random_state=seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=seed
        )

        scaler_x = StandardScaler()
        X_train_scaled = scaler_x.fit_transform(X_train)
        X_val_scaled = scaler_x.transform(X_val)
        X_test_scaled = scaler_x.transform(X_test)

        scaler_y = MinMaxScaler(feature_range=(0, 1))
        y_train_norm = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_norm = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
        y_test_norm = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

        best_val_loss = float('inf')
        best_test_preds = None

        for config in param_grid:
            val_loss, test_preds = train_and_validate(
                X_train_scaled, y_train_norm, 
                X_val_scaled, y_val_norm, 
                X_test_scaled, input_dim=X_raw.shape[1], 
                config=config
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_preds = test_preds

        r2_list.append(r2_score(y_test_norm, best_test_preds))
        mae_list.append(mean_absolute_error(y_test_norm, best_test_preds))

    r2_mean, r2_std = np.mean(r2_list), np.std(r2_list)
    mae_mean, mae_std = np.mean(mae_list), np.std(mae_list)
    
    print(f"Completed {target_name} | Test R2: {r2_mean:.4f} +/- {r2_std:.4f} | MAE: {mae_mean:.4f} +/- {mae_std:.4f}")
    
    return {
        "Target Dataset": target_name,
        "Test R2 (Mean +/- Std)": f"{r2_mean:.4f} +/- {r2_std:.4f}",
        "Test MAE (Mean +/- Std)": f"{mae_mean:.4f} +/- {mae_std:.4f}"
    }


if __name__ == "__main__":
    emb_path = './embedding/Harris_Foundation_Model.parquet'
    dataset_path = './data/houston_harris_LIST_dataset.parquet'
    
    if os.path.exists(emb_path) and os.path.exists(dataset_path):
        df_emb = pl.read_parquet(emb_path)
        emb_cols = [col for col in df_emb.columns if col.startswith("emb_")]

        df_full = pl.read_parquet(dataset_path)
        target_features = ["sovi", "nri_cri_value", "nri_cri_score"]

        label_exprs = [pl.col("hex_id")]
        for target in target_features:
            if target in df_full.columns:
                dtype = df_full.schema[target]
                if isinstance(dtype, pl.List):
                    label_exprs.append(pl.col(target).list.first().alias(target))
                else:
                    label_exprs.append(pl.col(target))
                    
        df_labels = df_full.select(label_exprs)
        df_merged = df_emb.join(df_labels, on="hex_id", how="inner")
        print(f"Successfully aligned {len(df_merged)} cells.")

        final_results = []
        for target in target_features:
            res = evaluate_target_task_paper_standard(
                target, df_merged, emb_cols, 
                num_runs=20, 
                enable_hpo=True
            )
            if res:
                final_results.append(res)

        print("\nDownstream Evaluation Results:")
        if final_results:
            print(pd.DataFrame(final_results).to_string(index=False))