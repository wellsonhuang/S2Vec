"""
H3-S2Vec-C: cell-centered variant of H3-S2Vec (Section 3.6).

Each resolution-8 cell forms its own sample:
  - Fine-scale window: H3 disk of radius K_RING around the cell (K=4 -> 61 cells). The center cell is
    placed first; positions are Fourier features of km offsets from the center cell.
  - A learnable CLS token is prepended. It carries no features and only aggregates the window through
    attention; its final encoder output is the cell embedding.
  - Coarse-scale tokens (Scale-MAE style): the resolution-6 parent of the center cell and the
    resolution-6 cells within COARSE_K rings (K=3 -> 37 tokens). Each token is the mean feature vector
    of its resolution-6 cell, computed after removing all cells inside the fine-scale window so that
    masked cells cannot leak through the averages.
  - Children without data use a learnable missing token. Masking and reconstruction are restricted to
    observed cells; the MSE is computed on masked cells only (same objective as H3-S2Vec).
  - Validation is split by the resolution-6 parent of the center cell, with a fixed validation mask;
    the checkpoint with the lowest validation loss is kept.
"""
import os
import math
import json
import warnings
import logging
from functools import lru_cache

import numpy as np
import polars as pl
import h3
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
import pytorch_lightning as pl_lightning
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
torch.set_float32_matmul_precision("medium")

# ==========================================================
# Configuration
# ==========================================================
DATA_PATH   = "./data/houston_harris_pre.parquet"
OUTPUT_PATH = "./embedding/Harris_Foundation_Model_C.parquet"
CKPT_DIR    = "./checkpoints/houston_harris_h3_s2vec_c"
HEX_COL     = "hex_id"
LABEL_KEYWORDS = ["sovi", "nri"]     # columns containing these keywords are never used as features

PARENT_RES  = 6
CHILD_RES   = None                   # None = inferred from the data (e.g., 8)

# Sample construction (Section 3.6)
K_RING      = 4                      # fine-scale window radius: 4 -> 61 cells
COARSE_K    = 3                      # coarse resolution-6 rings: 3 -> 37 tokens; 0 = disabled

# Positional encoding (Section 3.3)
NUM_FREQ    = 8
FREQ_MIN    = 1 / 16.0               # cycles per km (wavelength 16 km)
FREQ_MAX    = 2.0                    # cycles per km (wavelength 0.5 km)
COARSE_FREQ_MIN = 1 / 64.0
COARSE_FREQ_MAX = 1 / 4.0

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
VAL_FRAC     = 0.15                  # fraction of resolution-6 parents held out for validation
PATIENCE     = 30
MIN_VALID_TRAIN   = 2                # minimum observed cells in a window for it to be used
SAMPLES_PER_EPOCH = None             # None = all training windows (e.g., 50000 for Texas)
MAX_VAL_SAMPLES   = 3000
VAL_MASK_SEED     = 12345
NUM_WORKERS  = 0
SEED         = 42


# ==========================================================
# Utilities
# ==========================================================
latlng = lru_cache(maxsize=None)(h3.cell_to_latlng)


def latlng_to_km(lat, lng, lat0, lng0):
    dy = (lat - lat0) * 110.574
    dx = (lng - lng0) * 111.320 * np.cos(np.radians(lat0))
    return dx, dy


def disk_size(k):
    return 3 * k * (k + 1) + 1


# ==========================================================
# Dataset: one cell-centered window per cell
# ==========================================================
class H3CenteredDataset(Dataset):
    """
    rows/state/coords have shape (N, S); position 0 is the center cell.
    state: 0 = position does not exist (pentagon), 1 = cell has no data (missing token), 2 = cell has data.
    """
    def __init__(self, parquet_path):
        print(f"Reading {parquet_path}")
        df = pl.read_parquet(parquet_path)
        df = df.filter(pl.col(HEX_COL).is_not_null()).unique(subset=[HEX_COL], keep="first")
        df = df.filter(pl.Series([h3.is_valid_cell(h) for h in df[HEX_COL].to_list()]))
        self.hex_ids = df[HEX_COL].to_list()
        self.child_res = CHILD_RES or h3.get_resolution(self.hex_ids[0])
        assert self.child_res > PARENT_RES, "CHILD_RES must be larger than PARENT_RES"

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
        self.hex_to_row = {h: i for i, h in enumerate(self.hex_ids)}

        # Resolution-6 statistics (coarse tokens and validation split)
        par = [h3.cell_to_parent(h, PARENT_RES) for h in self.hex_ids]
        self.parents = sorted(set(par))
        self.pid = {p: i for i, p in enumerate(self.parents)}
        self.row_gid = np.array([self.pid[p] for p in par], dtype=np.int64)
        self.g_sum = np.zeros((len(self.parents), self.in_ch), np.float32)
        np.add.at(self.g_sum, self.row_gid, self.X)
        self.g_cnt = np.bincount(self.row_gid, minlength=len(self.parents)).astype(np.float32)
        self.cells_per_parent = 7 ** (self.child_res - PARENT_RES)
        self.C = disk_size(COARSE_K) if COARSE_K > 0 else 0

        N, S = len(self.hex_ids), disk_size(K_RING)
        self.S = S
        self.rows = np.full((N, S), -1, np.int64)
        self.state = np.zeros((N, S), np.int64)
        self.coords = np.zeros((N, S, 2), np.float32)
        self.cg = np.full((N, self.C), -1, np.int64)
        self.cxy = np.zeros((N, self.C, 2), np.float32)

        for i, h in enumerate(tqdm(self.hex_ids, desc=f"Building centered windows ({S} cells)")):
            lat0, lng0 = latlng(h)
            cells = sorted(h3.grid_disk(h, K_RING), key=h3.str_to_int)
            cells.remove(h)
            for j, c in enumerate([h] + cells[: S - 1]):
                self.coords[i, j] = latlng_to_km(*latlng(c), lat0, lng0)
                r = self.hex_to_row.get(c, -1)
                self.rows[i, j] = r
                self.state[i, j] = 2 if r >= 0 else 1
            if self.C:
                anchor = h3.cell_to_parent(h, PARENT_RES)
                ring = sorted((c for c in h3.grid_disk(anchor, COARSE_K) if c != anchor), key=h3.str_to_int)
                for j, c in enumerate([anchor] + ring[: self.C - 1]):
                    self.cg[i, j] = self.pid.get(c, -1)
                    self.cxy[i, j] = latlng_to_km(*latlng(c), lat0, lng0)

        self.sample_gid = self.row_gid            # sample i is centered on cell i
        self.n_valid = (self.state == 2).sum(1)
        print(f"{N} cells -> {N} windows ({S} positions each, {self.n_valid.mean():.1f} observed on average), "
              f"{self.C} coarse tokens")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return idx

    def collate(self, idxs):
        i = np.asarray(idxs)
        rows = self.rows[i]
        x = self.X[np.clip(rows, 0, None)]
        x[rows < 0] = 0.0
        B = len(i)
        if self.C:
            cg = self.cg[i]
            ok = cg >= 0
            csum = self.g_sum[np.clip(cg, 0, None)].copy()
            ccnt = self.g_cnt[np.clip(cg, 0, None)].copy()
            csum[~ok], ccnt[~ok] = 0.0, 0.0
            # Remove every cell of the fine-scale window from the coarse averages
            wg = np.where(rows >= 0, self.row_gid[np.clip(rows, 0, None)], -1)
            M = (wg[:, :, None] == cg[:, None, :]) & (wg[:, :, None] >= 0)
            csum -= np.einsum("bwc,bwf->bcf", M.astype(np.float32), x)
            ccnt -= M.sum(1)
            has = ccnt > 0.5
            cmean = np.where(has[..., None], csum / np.maximum(ccnt, 1.0)[..., None], 0.0)
            cx = np.concatenate([cmean, (ccnt / self.cells_per_parent)[..., None]], -1).astype(np.float32)
            cstate = np.where(has, 2, 1).astype(np.int64)
            cxy = self.cxy[i]
        else:
            cx = np.zeros((B, 0, self.in_ch + 1), np.float32)
            cstate = np.zeros((B, 0), np.int64)
            cxy = np.zeros((B, 0, 2), np.float32)
        arrs = (x, self.coords[i], self.state[i], rows, cx, cxy, cstate)
        return tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in arrs)


class IndexSubset(Dataset):
    def __init__(self, idx):
        self.idx = np.asarray(idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        return int(self.idx[k])


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
    """km offsets -> sin/cos(2*pi*f*x) with log-spaced frequencies in [fmin, fmax]."""
    def __init__(self, num_freq, fmin, fmax):
        super().__init__()
        f = torch.logspace(math.log10(fmin), math.log10(fmax), num_freq)
        self.register_buffer("freqs", 2 * math.pi * f)
        self.out_dim = 2 * 2 * num_freq

    def forward(self, coords):
        z = coords.unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(z), torch.cos(z)], -1).flatten(-2)


class PosEmbed(nn.Module):
    def __init__(self, dim, fmin, fmax):
        super().__init__()
        self.fourier = FourierPositionEncoding(NUM_FREQ, fmin, fmax)
        self.mlp = nn.Sequential(nn.Linear(self.fourier.out_dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, coords):
        return self.mlp(self.fourier(coords))


class H3S2VecCentered(pl_lightning.LightningModule):
    def __init__(self, in_ch, num_coarse=0, embed_dim=128, decoder_dim=256, encoder_layers=2,
                 decoder_layers=3, num_heads=8, dropout=0.3, mask_ratio=0.5, masking="random",
                 lr=5e-4, weight_decay=0.01, max_epochs=100, cosine_alpha=0.1, block_noise_km=1.0,
                 val_mask_seed=12345):
        super().__init__()
        self.save_hyperparameters()

        self.patch_embed = nn.Linear(in_ch, embed_dim)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.enc_pos = PosEmbed(embed_dim, FREQ_MIN, FREQ_MAX)
        self.encoder = nn.ModuleList([TransformerBlock(embed_dim, num_heads, dropout) for _ in range(encoder_layers)])
        self.enc_norm = nn.LayerNorm(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.dec_pos = PosEmbed(decoder_dim, FREQ_MIN, FREQ_MAX)
        self.decoder = nn.ModuleList([TransformerBlock(decoder_dim, num_heads, dropout) for _ in range(decoder_layers)])
        self.dec_norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, in_ch)

        if num_coarse > 0:
            self.coarse_embed = nn.Linear(in_ch + 1, embed_dim)     # mean features + fraction of cells observed
            self.coarse_missing = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.coarse_type = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.coarse_pos = PosEmbed(embed_dim, COARSE_FREQ_MIN, COARSE_FREQ_MAX)
            self.coarse_dec_pos = PosEmbed(decoder_dim, COARSE_FREQ_MIN, COARSE_FREQ_MAX)
            for p in (self.coarse_missing, self.coarse_type):
                nn.init.normal_(p, std=0.02)
        for p in (self.missing_token, self.mask_token, self.cls_token):
            nn.init.normal_(p, std=0.02)

    # ---------- Masking (observed cells only) ----------
    def make_mask(self, coords, state, ratio):
        valid = state == 2
        B, S = valid.shape
        if ratio <= 0:
            return torch.zeros_like(valid)
        n_valid = valid.sum(1)
        n_mask = torch.floor(n_valid.float() * ratio).long()
        n_mask = torch.minimum(n_mask, (n_valid - 1).clamp(min=0))   # keep at least one visible cell
        n_mask = torch.where(n_valid >= 2, n_mask.clamp(min=1), torch.zeros_like(n_mask))

        if self.hparams.masking == "block":
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
    def encode(self, x, coords, state, cx, cxy, cstate, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.hparams.mask_ratio
        masked = self.make_mask(coords, state, mask_ratio)
        B = x.shape[0]

        tok = self.patch_embed(x)
        tok = torch.where((state == 1).unsqueeze(-1), self.missing_token.expand_as(tok), tok)
        tok = (tok + self.enc_pos(coords)) * (~masked).unsqueeze(-1)

        cls = self.cls_token.expand(B, 1, -1) + self.enc_pos(torch.zeros(B, 1, 2, device=x.device))
        seq = torch.cat([cls, tok], 1)
        key_valid = torch.cat([
            torch.ones(B, 1, dtype=torch.bool, device=x.device),
            (state >= 1) & ~masked,    # missing tokens are visible; masked and non-existent positions are not
        ], 1)
        if self.hparams.num_coarse > 0:
            ct = self.coarse_embed(cx)
            ct = torch.where((cstate == 1).unsqueeze(-1), self.coarse_missing.expand_as(ct), ct)
            seq = torch.cat([seq, ct + self.coarse_type + self.coarse_pos(cxy)], 1)
            key_valid = torch.cat([key_valid, cstate >= 1], 1)
        for blk in self.encoder:
            seq = blk(seq, key_valid)
        return self.enc_norm(seq), masked

    # ---------- Decoder ----------
    def decode(self, latent, masked, coords, state, cxy, cstate):
        d = self.decoder_embed(latent)
        S = coords.shape[1]
        fine = d[:, 1:1 + S]
        fine = torch.where(masked.unsqueeze(-1), self.mask_token.expand_as(fine), fine)
        seq = torch.cat([d[:, :1], fine + self.dec_pos(coords)], 1)
        key_valid = torch.cat([torch.ones(d.shape[0], 1, dtype=torch.bool, device=d.device), state >= 1], 1)
        if self.hparams.num_coarse > 0:
            seq = torch.cat([seq, d[:, 1 + S:] + self.coarse_dec_pos(cxy)], 1)
            key_valid = torch.cat([key_valid, cstate >= 1], 1)
        for blk in self.decoder:
            seq = blk(seq, key_valid)
        return self.head(self.dec_norm(seq))[:, 1:1 + S]

    def _step(self, batch):
        x, coords, state, _, cx, cxy, cstate = batch
        latent, masked = self.encode(x, coords, state, cx, cxy, cstate)
        pred = self.decode(latent, masked, coords, state, cxy, cstate)
        per_tok = (pred.float() - x.float()).pow(2).mean(-1)
        m = masked.float()
        return (per_tok * m).sum() / m.sum().clamp(min=1.0)    # MSE on masked observed cells

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch[0].shape[0])
        return loss

    # Fixed validation mask so that val_loss is comparable across epochs
    def on_validation_epoch_start(self):
        self._rng = torch.get_rng_state()
        self._cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(self.hparams.val_mask_seed)

    def validation_step(self, batch, batch_idx):
        self.log("val_loss", self._step(batch), on_epoch=True, prog_bar=True, batch_size=batch[0].shape[0])

    def on_validation_epoch_end(self):
        torch.set_rng_state(self._rng)
        if self._cuda_rng is not None:
            torch.cuda.set_rng_state_all(self._cuda_rng)

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

    ds = H3CenteredDataset(DATA_PATH)

    # Validation split by the resolution-6 parent of the center cell
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(ds.parents))
    val_par = set(perm[: max(1, int(len(perm) * VAL_FRAC))].tolist())
    is_val = np.array([g in val_par for g in ds.sample_gid])
    usable = ds.n_valid >= MIN_VALID_TRAIN
    train_ids, val_ids = np.where(usable & ~is_val)[0], np.where(usable & is_val)[0]
    if len(val_ids) > MAX_VAL_SAMPLES:
        val_ids = rng.choice(val_ids, MAX_VAL_SAMPLES, replace=False)

    n_epoch = min(SAMPLES_PER_EPOCH or len(train_ids), len(train_ids))
    bs = min(BATCH_SIZE, n_epoch)
    train_loader = DataLoader(IndexSubset(train_ids), batch_size=bs, drop_last=n_epoch >= 2 * bs,
                              sampler=RandomSampler(IndexSubset(train_ids), num_samples=n_epoch),
                              collate_fn=ds.collate, num_workers=NUM_WORKERS)
    val_loader = DataLoader(IndexSubset(val_ids), batch_size=bs, shuffle=False,
                            collate_fn=ds.collate, num_workers=NUM_WORKERS)
    print(f"Training windows: {len(train_ids)} ({n_epoch} per epoch), validation windows: {len(val_ids)}")

    model = H3S2VecCentered(
        in_ch=ds.in_ch, num_coarse=ds.C, embed_dim=EMBED_DIM, decoder_dim=DECODER_DIM,
        encoder_layers=ENCODER_LAYERS, decoder_layers=DECODER_LAYERS, num_heads=NUM_HEADS,
        dropout=DROPOUT, mask_ratio=MASK_RATIO, masking=MASKING, lr=LR, weight_decay=WEIGHT_DECAY,
        max_epochs=MAX_EPOCHS, cosine_alpha=COSINE_ALPHA, block_noise_km=BLOCK_NOISE_KM,
        val_mask_seed=VAL_MASK_SEED,
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
    model = H3S2VecCentered.load_from_checkpoint(best)

    # Extract embeddings: one unmasked forward pass per cell, CLS output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    all_emb = np.zeros((len(ds), EMBED_DIM), dtype=np.float32)
    ext_loader = DataLoader(IndexSubset(np.arange(len(ds))), batch_size=256, shuffle=False,
                            collate_fn=ds.collate, num_workers=NUM_WORKERS)
    k = 0
    with torch.no_grad():
        for batch in tqdm(ext_loader, desc="Extracting embeddings"):
            x, coords, state, _, cx, cxy, cstate = (t.to(device) for t in batch)
            latent, _ = model.encode(x, coords, state, cx, cxy, cstate, mask_ratio=0.0)
            all_emb[k:k + x.shape[0]] = latent[:, 0].float().cpu().numpy()
            k += x.shape[0]

    out = pl.DataFrame({"hex_id": ds.hex_ids}).with_columns(
        [pl.Series(f"emb_{i}", all_emb[:, i]) for i in range(EMBED_DIM)]
    )
    out.write_parquet(OUTPUT_PATH)
    print(f"Saved {out.height} embeddings to {OUTPUT_PATH}")

    cfg = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, list, type(None)))}
    cfg.update({"child_res": ds.child_res, "num_features": ds.in_ch, "feature_cols": ds.feature_cols,
                "best_ckpt": best, "best_val_loss": float(ckpt_cb.best_model_score)})
    with open(OUTPUT_PATH.replace(".parquet", "_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
