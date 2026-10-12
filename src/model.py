"""
H3-S2Vec: S2Vec-style hierarchical masked autoencoder on the H3 grid.

Each resolution-6 parent cell is expanded into all of its resolution-8 children (49 cells),
which form one fixed-length token sequence.

  - Children without data are represented by a learnable missing token.
  - Masking and reconstruction are restricted to observed children; masked children are
    removed from attention with a key mask.
  - A macro token (mean of the visible children) is prepended as a parent-level summary.
  - Positions are encoded from each child's offset (km) to the parent center with Fourier
    features ("fourier"), learned H3 slot embeddings ("slot"), or both.
  - The validation set is split by parent; the checkpoint with the lowest validation loss is kept.
  - Embeddings are the final encoder outputs of each child, computed without masking.
"""
import os
import math
import json
import warnings
import logging

import numpy as np
import polars as pl
import h3
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl_lightning
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
torch.set_float32_matmul_precision("medium")

# ==========================================================
# Configuration
# ==========================================================
DATA_PATH   = "./data/texas_pre.parquet"
OUTPUT_PATH = "./embedding/texas_final.parquet"
CKPT_DIR    = "./checkpoints/texas_final"
HEX_COL     = "hex_id"
LABEL_KEYWORDS = ["sovi", "nri"]     # columns containing these keywords are never used as features

PARENT_RES  = 6
CHILD_RES   = None                   # None = inferred from the data (e.g., 8)

# Positional encoding
POS_MODE    = "fourier"              # "fourier" | "slot" | "both"
NUM_FREQ    = 8
FREQ_MIN    = 1 / 16.0               # cycles per km (wavelength 16 km)
FREQ_MAX    = 2.0                    # cycles per km (wavelength 0.5 km)

# Masking
MASKING        = "random"            # "random" | "block"
MASK_RATIO     = 0.5
BLOCK_NOISE_KM = 1.0

# Model (Table 1)
EMBED_DIM      = 128
DECODER_DIM    = 256
ENCODER_LAYERS = 2
DECODER_LAYERS = 3
NUM_HEADS      = 8
DROPOUT        = 0.3

# Training (Table 1)
BATCH_SIZE   = 64
MAX_EPOCHS   = 100
LR           = 5e-4
WEIGHT_DECAY = 0.01
COSINE_ALPHA = 0.1                   # final learning rate = LR * COSINE_ALPHA
GRAD_CLIP    = 2.0
VAL_FRAC     = 0.15                  # fraction of parents held out for validation
PATIENCE     = 30
MIN_VALID_TRAIN = 2                  # minimum observed children for a parent to be used
NUM_WORKERS  = 0
SEED         = 42


# ==========================================================
# Utilities
# ==========================================================
def h3_digits_slot(child: str, parent_res: int, child_res: int) -> int:
    """Fixed position of a child within its parent, derived from the H3 index digits."""
    v = h3.str_to_int(child)
    slot = 0
    for r in range(parent_res + 1, child_res + 1):
        d = (v >> ((15 - r) * 3)) & 7
        slot = slot * 7 + d
    return slot


def latlng_to_km(lat, lng, lat0, lng0):
    dy = (lat - lat0) * 110.574
    dx = (lng - lng0) * 111.320 * np.cos(np.radians(lat0))
    return dx, dy


# ==========================================================
# Dataset: one sample per parent, expanded into all children
# ==========================================================
class H3HierarchicalDataset(Dataset):
    """
    Each sample is one parent cell.
    state: 0 = position does not exist (pentagon parent),
           1 = cell exists but has no data (missing token),
           2 = cell has data.
    """
    def __init__(self, parquet_path, hex_col="hex_id", parent_res=6, child_res=None):
        print(f"Reading {parquet_path}")
        df = pl.read_parquet(parquet_path)
        df = df.filter(pl.col(hex_col).is_not_null()).unique(subset=[hex_col], keep="first")
        df = df.filter(pl.Series([h3.is_valid_cell(h) for h in df[hex_col].to_list()]))
        self.hex_ids = df[hex_col].to_list()

        if child_res is None:
            child_res = h3.get_resolution(self.hex_ids[0])
        self.parent_res, self.child_res = parent_res, child_res
        assert child_res > parent_res, "CHILD_RES must be larger than PARENT_RES"

        # Feature columns (exclude identifiers and labels)
        self.feature_cols = [
            c for c, t in df.schema.items()
            if t.is_numeric() and "hex" not in c.lower()
            and not any(k in c.lower() for k in LABEL_KEYWORDS)
        ]
        dropped = [c for c in df.columns if any(k in c.lower() for k in LABEL_KEYWORDS)]
        print(f"Features: {len(self.feature_cols)}; excluded label columns: {dropped}")

        X = df.select(self.feature_cols).cast(pl.Float32).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        mean, std = X.mean(0), X.std(0)
        std[std < 1e-8] = 1.0
        self.X = ((X - mean) / std).astype(np.float32)
        self.in_ch = self.X.shape[1]
        self.norm_stats = {"mean": mean.tolist(), "std": std.tolist()}

        # Group cells by parent and expand every parent into all children
        hex_to_row = {h: i for i, h in enumerate(self.hex_ids)}
        parents = sorted({h3.cell_to_parent(h, parent_res) for h in self.hex_ids})
        S = 7 ** (child_res - parent_res)
        P = len(parents)
        self.num_slots = S

        self.row_idx = np.full((P, S), -1, dtype=np.int64)    # data row of each child (-1 = no data)
        self.state = np.zeros((P, S), dtype=np.int64)
        self.coords = np.zeros((P, S, 2), dtype=np.float32)   # km offset from the parent center

        for p_i, p in enumerate(tqdm(parents, desc="Building parent-child index")):
            lat0, lng0 = h3.cell_to_latlng(p)
            for c in h3.cell_to_children(p, child_res):
                s = h3_digits_slot(c, parent_res, child_res)
                lat, lng = h3.cell_to_latlng(c)
                self.coords[p_i, s] = latlng_to_km(lat, lng, lat0, lng0)
                r = hex_to_row.get(c, -1)
                self.row_idx[p_i, s] = r
                self.state[p_i, s] = 2 if r >= 0 else 1

        self.parents = parents
        self.n_valid = (self.state == 2).sum(1)
        print(f"{len(self.hex_ids)} cells -> {P} parents ({S} positions each, "
              f"{self.n_valid.mean():.1f} observed on average)")

    def __len__(self):
        return len(self.parents)

    def __getitem__(self, idx):
        return idx

    def collate(self, idxs):
        idxs = np.asarray(idxs)
        rows = self.row_idx[idxs]                             # (B, S)
        x = self.X[np.clip(rows, 0, None)]                    # (B, S, F)
        x[rows < 0] = 0.0
        return (
            torch.from_numpy(x),
            torch.from_numpy(self.coords[idxs]),
            torch.from_numpy(self.state[idxs]),
            torch.from_numpy(rows),
        )


# ==========================================================
# Model
# ==========================================================
class TransformerBlock(nn.Module):
    """Pre-LN Transformer block with a key mask (tokens with False are never attended to)."""
    def __init__(self, dim, heads, drop):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(4 * dim, dim), nn.Dropout(drop),
        )

    def forward(self, x, key_valid):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=~key_valid, need_weights=False)
        x = x + self.drop(a)
        return x + self.mlp(self.n2(x))


class FourierPositionEncoding(nn.Module):
    """km offsets -> sin/cos(2*pi*f*x) with log-spaced frequencies in [FREQ_MIN, FREQ_MAX]."""
    def __init__(self, num_freq=8, fmin=1 / 16, fmax=2.0):
        super().__init__()
        f = torch.logspace(math.log10(fmin), math.log10(fmax), num_freq)
        self.register_buffer("freqs", 2 * math.pi * f)
        self.out_dim = 2 * 2 * num_freq

    def forward(self, coords):                      # (B, S, 2)
        z = coords.unsqueeze(-1) * self.freqs       # (B, S, 2, F)
        return torch.cat([torch.sin(z), torch.cos(z)], -1).flatten(-2)


class PosEmbed(nn.Module):
    def __init__(self, dim, num_slots, mode):
        super().__init__()
        self.mode = mode
        if mode in ("fourier", "both"):
            self.fourier = FourierPositionEncoding(NUM_FREQ, FREQ_MIN, FREQ_MAX)
            self.mlp = nn.Sequential(nn.Linear(self.fourier.out_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        if mode in ("slot", "both"):
            self.slot = nn.Parameter(torch.zeros(1, num_slots, dim))
            nn.init.normal_(self.slot, std=0.02)

    def forward(self, coords):
        out = 0
        if self.mode in ("fourier", "both"):
            out = out + self.mlp(self.fourier(coords))
        if self.mode in ("slot", "both"):
            out = out + self.slot
        return out


class S2VecModelSpatialH3(pl_lightning.LightningModule):
    def __init__(self, in_ch, num_slots, embed_dim=128, decoder_dim=256, encoder_layers=2,
                 decoder_layers=3, num_heads=8, dropout=0.3, mask_ratio=0.5, masking="random",
                 pos_mode="fourier", lr=5e-4, weight_decay=0.01, max_epochs=100, cosine_alpha=0.1,
                 block_noise_km=1.0):
        super().__init__()
        self.save_hyperparameters()

        self.patch_embed = nn.Linear(in_ch, embed_dim)
        self.macro_embed = nn.Linear(in_ch, embed_dim)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.macro_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.enc_pos = PosEmbed(embed_dim, num_slots, pos_mode)
        self.encoder = nn.ModuleList([TransformerBlock(embed_dim, num_heads, dropout) for _ in range(encoder_layers)])
        self.enc_norm = nn.LayerNorm(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.dec_pos = PosEmbed(decoder_dim, num_slots, pos_mode)
        self.decoder = nn.ModuleList([TransformerBlock(decoder_dim, num_heads, dropout) for _ in range(decoder_layers)])
        self.dec_norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, in_ch)

        for p in (self.missing_token, self.macro_pos, self.mask_token):
            nn.init.normal_(p, std=0.02)

    # ---------- Masking (observed children only) ----------
    def make_mask(self, coords, state, ratio):
        valid = state == 2
        B, S = valid.shape
        if ratio <= 0:
            return torch.zeros_like(valid)
        n_valid = valid.sum(1)
        n_mask = torch.floor(n_valid.float() * ratio).long()
        n_mask = torch.minimum(n_mask, (n_valid - 1).clamp(min=0))   # keep at least one visible child
        n_mask = torch.where(n_valid >= 2, n_mask.clamp(min=1), torch.zeros_like(n_mask))

        if self.hparams.masking == "block":
            # Pick a random observed child as the center and mask the closest children first
            r = torch.rand(B, S, device=coords.device).masked_fill(~valid, -1.0)
            center = coords[torch.arange(B), r.argmax(1)].unsqueeze(1)
            dist = (coords - center).norm(dim=-1)
            score = -(dist + torch.randn_like(dist) * self.hparams.block_noise_km)
        else:
            score = torch.rand(B, S, device=coords.device)
        score = score.masked_fill(~valid, float("-inf"))
        rank = score.argsort(dim=1, descending=True).argsort(dim=1)
        return rank < n_mask.unsqueeze(1)

    # ---------- Encoder ----------
    def encode(self, x, coords, state, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.hparams.mask_ratio
        masked = self.make_mask(coords, state, mask_ratio)
        visible = (state == 2) & ~masked

        tok = self.patch_embed(x)
        tok = torch.where((state == 1).unsqueeze(-1), self.missing_token.expand_as(tok), tok)
        tok = tok + self.enc_pos(coords)
        tok = tok * (~masked).unsqueeze(-1)            # zero out masked children (also excluded by the key mask)

        # Macro token: mean of visible children only (no information from masked children)
        vis_f = visible.unsqueeze(-1).float()
        macro_in = (x * vis_f).sum(1) / vis_f.sum(1).clamp(min=1.0)
        macro = self.macro_embed(macro_in).unsqueeze(1) + self.macro_pos

        seq = torch.cat([macro, tok], 1)
        key_valid = torch.cat([
            torch.ones(x.shape[0], 1, dtype=torch.bool, device=x.device),
            (state >= 1) & ~masked,    # missing tokens are visible; masked and non-existent positions are not
        ], 1)
        for blk in self.encoder:
            seq = blk(seq, key_valid)
        return self.enc_norm(seq), masked

    # ---------- Decoder ----------
    def decode(self, latent, masked, coords, state):
        d = self.decoder_embed(latent)
        child = d[:, 1:]
        child = torch.where(masked.unsqueeze(-1), self.mask_token.expand_as(child), child)
        child = child + self.dec_pos(coords)
        seq = torch.cat([d[:, :1], child], 1)
        key_valid = torch.cat([
            torch.ones(d.shape[0], 1, dtype=torch.bool, device=d.device),
            state >= 1,
        ], 1)
        for blk in self.decoder:
            seq = blk(seq, key_valid)
        return self.head(self.dec_norm(seq))[:, 1:]

    def _step(self, batch):
        x, coords, state, _ = batch
        latent, masked = self.encode(x, coords, state)
        pred = self.decode(latent, masked, coords, state)
        per_tok = (pred.float() - x.float()).pow(2).mean(-1)
        m = masked.float()
        return (per_tok * m).sum() / m.sum().clamp(min=1.0)    # MSE on masked observed children

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch[0].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch[0].shape[0])

    def configure_optimizers(self):
        hp = self.hparams
        opt = torch.optim.AdamW(self.parameters(), lr=hp.lr, weight_decay=hp.weight_decay, betas=(0.9, 0.95))
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp.max_epochs, eta_min=hp.lr * hp.cosine_alpha)
        return {"optimizer": opt, "lr_scheduler": sch}


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":
    pl_lightning.seed_everything(SEED, workers=True)
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    ds = H3HierarchicalDataset(DATA_PATH, HEX_COL, PARENT_RES, CHILD_RES)

    # Validation split by parent
    rng = np.random.default_rng(SEED)
    trainable = np.where(ds.n_valid >= MIN_VALID_TRAIN)[0]
    rng.shuffle(trainable)
    n_val = max(1, int(len(trainable) * VAL_FRAC))
    val_ids, train_ids = trainable[:n_val], trainable[n_val:]
    print(f"Training parents: {len(train_ids)}, validation parents: {len(val_ids)}")

    bs = min(BATCH_SIZE, len(train_ids))
    drop_last = len(train_ids) >= 2 * bs
    train_loader = DataLoader(torch.utils.data.Subset(ds, train_ids), batch_size=bs, shuffle=True,
                              drop_last=drop_last, collate_fn=ds.collate, num_workers=NUM_WORKERS)
    val_loader = DataLoader(torch.utils.data.Subset(ds, val_ids), batch_size=bs, shuffle=False,
                            collate_fn=ds.collate, num_workers=NUM_WORKERS)
    assert len(train_loader) > 0, "No training batches; reduce BATCH_SIZE"
    print(f"Batches per epoch: {len(train_loader)} (batch size {bs})")

    model = S2VecModelSpatialH3(
        in_ch=ds.in_ch, num_slots=ds.num_slots, embed_dim=EMBED_DIM, decoder_dim=DECODER_DIM,
        encoder_layers=ENCODER_LAYERS, decoder_layers=DECODER_LAYERS, num_heads=NUM_HEADS,
        dropout=DROPOUT, mask_ratio=MASK_RATIO, masking=MASKING, pos_mode=POS_MODE, lr=LR,
        weight_decay=WEIGHT_DECAY, max_epochs=MAX_EPOCHS, cosine_alpha=COSINE_ALPHA,
        block_noise_km=BLOCK_NOISE_KM,
    )

    ckpt_cb = ModelCheckpoint(dirpath=CKPT_DIR, filename="best", monitor="val_loss", mode="min",
                              save_top_k=1, enable_version_counter=False)
    es_cb = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE, min_delta=1e-4)
    trainer = pl_lightning.Trainer(
        max_epochs=MAX_EPOCHS, accelerator="auto",
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        gradient_clip_val=GRAD_CLIP, callbacks=[ckpt_cb, es_cb], logger=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_loader, val_loader)

    best = ckpt_cb.best_model_path
    assert best and os.path.exists(best), "Best checkpoint not found"
    print(f"Best checkpoint: {best} (val_loss = {float(ckpt_cb.best_model_score):.4f})")
    model = S2VecModelSpatialH3.load_from_checkpoint(best)

    # Extract embeddings: one unmasked forward pass per parent, encoder output of each observed child
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    all_emb = np.zeros((len(ds.hex_ids), EMBED_DIM), dtype=np.float32)
    done = np.zeros(len(ds.hex_ids), dtype=bool)
    ext_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ds.collate, num_workers=NUM_WORKERS)
    with torch.no_grad():
        for x, coords, state, rows in tqdm(ext_loader, desc="Extracting embeddings"):
            latent, _ = model.encode(x.to(device), coords.to(device), state.to(device), mask_ratio=0.0)
            lat = latent[:, 1:].float().cpu().numpy()
            rows = rows.numpy()
            sel = rows >= 0
            all_emb[rows[sel]] = lat[sel]
            done[rows[sel]] = True

    print(f"Extracted {done.sum()} / {len(done)} cells")
    out = pl.DataFrame({"hex_id": ds.hex_ids}).with_columns(
        [pl.Series(f"emb_{i}", all_emb[:, i]) for i in range(EMBED_DIM)]
    ).filter(pl.Series(done))
    out.write_parquet(OUTPUT_PATH)

    cfg = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, list, type(None)))}
    cfg.update({"child_res": ds.child_res, "num_features": ds.in_ch, "feature_cols": ds.feature_cols,
                "best_ckpt": best, "best_val_loss": float(ckpt_cb.best_model_score)})
    with open(OUTPUT_PATH.replace(".parquet", "_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved embeddings to {OUTPUT_PATH}")