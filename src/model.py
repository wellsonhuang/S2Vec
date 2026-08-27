import os
import polars as pl
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl_lightning
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import h3
from scipy.spatial import cKDTree
from timm.models.vision_transformer import Block
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# ==========================================
# 1. Core MAE Architecture (Encoder & Decoder)
# ==========================================
class MAEEncoder(nn.Module):
    def __init__(self, embed_dim: int, depth: int, num_heads: int, dropout_prob: float):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads=num_heads, qkv_bias=True, proj_drop=dropout_prob, attn_drop=dropout_prob)
            for _ in range(depth)
        ])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks: 
            x = block(x)
        return self.norm(x)

class MAEDecoder(nn.Module):
    def __init__(self, decoder_dim: int, patch_dim: int, depth: int, num_heads: int, dropout_prob: float):
        super().__init__()
        self.norm = nn.LayerNorm(decoder_dim)
        self.blocks = nn.ModuleList([
            Block(decoder_dim, num_heads=num_heads, qkv_bias=True, proj_drop=dropout_prob, attn_drop=dropout_prob)
            for _ in range(depth)
        ])
        self.output = nn.Linear(decoder_dim, patch_dim)
        nn.init.xavier_uniform_(self.output.weight)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks: 
            x = block(x)
        return self.output(self.norm(x))

# ==========================================
# 2. Real-world Spatial Dataset (Dynamic Relative Coordinates)
# ==========================================
class H3SpatialDataset(Dataset):
    def __init__(self, parquet_path, seq_len=64):
        self.seq_len = seq_len
        print("Loading Parquet data and building spatial KD-Tree...")
        
        df = pl.read_parquet(parquet_path)
        self.hex_ids = df["hex_id"].to_list()
        
        # 🚀 變更點：資料在 Preprocessing 已經標準化過了，直接取用即可！
        self.X_normalized = df.drop("hex_id").cast(pl.Float32).to_numpy()     
        self.in_ch = self.X_normalized.shape[1]
        
        self.coords = np.array([h3.cell_to_latlng(h) for h in self.hex_ids])
        tree = cKDTree(self.coords)
        
        self.window_indices = np.zeros((len(self.hex_ids), self.seq_len), dtype=np.int32)
        self.relative_coords = np.zeros((len(self.hex_ids), self.seq_len, 2), dtype=np.float32)
        
        print("Computing geographic relative coordinates...")
        _, neighbors_all = tree.query(self.coords, k=self.seq_len)
        
        for i in range(len(self.hex_ids)):
            neighbor_indices = neighbors_all[i]
            self.window_indices[i] = neighbor_indices
            
            center_coord = self.coords[i]
            neighbor_coords = self.coords[neighbor_indices]
            
            self.relative_coords[i] = (neighbor_coords - center_coord) * 100.0 

        print(f"Spatial database constructed. Total grid centers: {len(self.hex_ids)}.")

    def __len__(self): 
        return len(self.hex_ids)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.X_normalized[self.window_indices[idx]], dtype=torch.float32)
        rel_coords = torch.tensor(self.relative_coords[idx], dtype=torch.float32)
        return x, rel_coords

# ==========================================
# 3. S2Vec Dynamic Coordinate Model (No-CLS Version)
# ==========================================
class S2VecModelSpatialH3(pl_lightning.LightningModule):
    def __init__(self, in_ch: int, num_heads=4, encoder_layers=4, decoder_layers=1, 
                 embed_dim=128, decoder_dim=256, mask_ratio=0.75, lr=0.0005, weight_decay=0.001, dropout_rate=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.in_ch = in_ch
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.mask_ratio = mask_ratio
        self.lr = lr
        self.weight_decay = weight_decay

        self.patch_embed = nn.Linear(in_ch, embed_dim)
        
        self.coord_to_pos_embed = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim)
        )
        self.coord_to_dec_pos_embed = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, decoder_dim)
        )

        self.encoder = MAEEncoder(embed_dim, encoder_layers, num_heads, dropout_prob=dropout_rate)
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder = MAEDecoder(decoder_dim, in_ch, decoder_layers, num_heads, dropout_prob=dropout_rate)

        torch.nn.init.normal_(self.mask_token, std=0.02)

    def random_masking(self, x, mask_ratio):
        B, N, D = x.shape
        if mask_ratio == 0.0:
            return x, torch.zeros([B, N], device=x.device), torch.arange(N, device=x.device).unsqueeze(0).repeat(B, 1)

        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def encode(self, x, rel_coords, mask_ratio=None):
        if mask_ratio is None: mask_ratio = self.mask_ratio
        
        pos_embed = self.coord_to_pos_embed(rel_coords) 
        x = self.patch_embed(x) + pos_embed 

        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        return self.encoder(x), mask, ids_restore

    def decode(self, x, ids_restore, rel_coords):
        x = self.decoder_embed(x)
        
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        dec_pos_embed = self.coord_to_dec_pos_embed(rel_coords)
        
        x_ = x_ + dec_pos_embed
        x_ = self.decoder(x_)
        return x_ 

    def forward(self, batch):
        inputs, rel_coords = batch
        latent, mask, ids_restore = self.encode(inputs, rel_coords, mask_ratio=self.mask_ratio)
        pred = self.decode(latent, ids_restore, rel_coords)
        return pred, inputs, mask

    def training_step(self, batch, batch_idx):
        rec, target, mask = self(batch)
        loss = (rec - target).pow(2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.95))

# ==========================================
# 4. Main Routine: Pre-training & Extraction
# ==========================================
if __name__ == "__main__":
    data_path = './data/houston_harris_FLATTENED_FILLED_DIM_REDUCED.parquet'
    dataset = H3SpatialDataset(data_path, seq_len=64)
    
    num_cores = min(4, os.cpu_count() or 1)
    
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=num_cores)
    
    print("\nStarting S2Vec Spatial Pre-training (Dynamic Coord, No-CLS)...")
    
    model = S2VecModelSpatialH3(in_ch=dataset.in_ch)
    
    early_stop_callback = EarlyStopping(
        monitor="train_loss", 
        min_delta=0.0001,     
        patience=5,           
        verbose=True,
        mode="min"            
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath="./checkpoints",
        filename="s2vec-best",
        save_top_k=1,
        mode="min"
    )
    
    trainer = pl_lightning.Trainer(
        max_epochs=300, 
        accelerator="auto", 
        gradient_clip_val=2.0,
        enable_checkpointing=True,
        logger=False,
        callbacks=[early_stop_callback, checkpoint_callback]
    )
    trainer.fit(model, train_dataloaders=train_loader)
    
    print("\nTraining complete. Loading best weights and extracting embeddings...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Loaded best checkpoint: {best_model_path}")
        model = S2VecModelSpatialH3.load_from_checkpoint(best_model_path)
    else:
        print("Warning: Checkpoint not found. Using current weights.")
        
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    extract_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=num_cores)
    all_embeddings = []
    
    with torch.no_grad():
        for batch_x, rel_coords in tqdm(extract_loader, desc="Extracting Embeddings"):
            batch_x = batch_x.to(device)
            rel_coords = rel_coords.to(device)
            
            latent, _, _ = model.encode(batch_x, rel_coords, mask_ratio=0.0)
            
            center_embedding = latent[:, 0, :] 
            all_embeddings.append(center_embedding.cpu().numpy())
            
    all_embeddings = np.vstack(all_embeddings)
    
    print("\nWriting embeddings to Parquet file...")
    df_embeddings = pl.DataFrame({"hex_id": dataset.hex_ids})
    
    embed_dim = all_embeddings.shape[1]
    emb_cols = [pl.Series(f"emb_{i}", all_embeddings[:, i]) for i in range(embed_dim)]
    df_embeddings = df_embeddings.with_columns(emb_cols)
    
    output_filename = "./embedding/final_dynamic_no_cls.parquet"
    
    os.makedirs(os.path.dirname(output_filename), exist_ok=True) 
    df_embeddings.write_parquet(output_filename)
    
    print(f"File saved to: {output_filename}")
    print(f"DataFrame shape: {df_embeddings.shape} (1 hex_id column and {embed_dim} embedding columns)")