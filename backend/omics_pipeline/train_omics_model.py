"""
OmicsCRISPR Phase 3 (v2) -- Three-Branch Late-Fusion Model

Architecture:
  Branch 1 - Sequence CNN  : raw 20-mer one-hot (4x20)
                             -> Conv1d(64) -> Conv1d(128) -> MaxPool
                             -> FC(128) -- learns positional motifs
  Branch 2 - Feature MLP   : 450-dim precomputed biochemical features
                             (Tm windows, GC clamp, hairpin, etc.)
                             -> FC(256) -> FC(128) -- captures rule-based signal
  Branch 3 - Omics MLP     : (rna_tpm, atac_sig, atac_n, splice_dist, ct_idx)
                             -> FC(64) -> FC(32) -- cell-type chromatin context
  Late fusion              : concat(128+128+32=288) -> FC(128) -> FC(64) -> FC(1)

Improvements vs v1:
  - Three branches instead of two (adds 450-dim precomputed features)
  - z-score normalization of all continuous inputs (train stats only, no leakage)
  - AdamW with weight decay for L2 regularisation
  - GELU activations throughout
  - LR linear warmup (5 epochs) + cosine decay
  - Patience extended to 15 epochs

Outputs in data/omics/model/:
  omics_model.pt         checkpoint (best val Pearson r) + normalisation stats
  training_log.json      epoch metrics
  model_config.json      architecture + hyperparameters

Run from backend/:
    python -m omics_pipeline.train_omics_model
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ── Paths ─────────────────────────────────────────────────────────────────────
_BACKEND = Path(__file__).parent.parent
sys.path.insert(0, str(_BACKEND))

from .config import FEATURES_DIR

MODEL_DIR = FEATURES_DIR.parent / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PT      = MODEL_DIR / "omics_model.pt"
TRAINING_LOG  = MODEL_DIR / "training_log.json"
MODEL_CONFIG  = MODEL_DIR / "model_config.json"

GUIDE_META_CSV = FEATURES_DIR / "guide_metadata.csv"
CELL_FEAT_CSV  = FEATURES_DIR / "cell_features.csv"
SEQ_NPZ        = FEATURES_DIR / "seq_features.npz"

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE    = 512
MAX_EPOCHS    = 150
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 15
WARMUP_EPOCHS = 5
TRAIN_FRAC    = 0.80
PEARSON_WGHT  = 0.3

SEQ_LEN       = 20
N_OMICS       = 6        # rna_tpm, atac_sig, atac_n, splice_dist, gene_effect, ct_idx
SEQ_FEAT_DIM  = 450
CNN_EMB_DIM   = 128
FEAT_EMB_DIM  = 128
OMICS_EMB_DIM = 32
FUSION_DIM    = 288      # CNN_EMB + FEAT_EMB + OMICS_EMB

BASES = "ACGT"
BASE_IDX = {b: i for i, b in enumerate(BASES)}


# ── Sequence one-hot ──────────────────────────────────────────────────────────

def seq_to_onehot(seq: str) -> np.ndarray:
    """Convert 20-mer string to (4, 20) float32 one-hot array."""
    x = np.zeros((4, SEQ_LEN), dtype=np.float32)
    for j, base in enumerate(seq[:SEQ_LEN]):
        i = BASE_IDX.get(base, -1)
        if i >= 0:
            x[i, j] = 1.0
    return x


# ── Dataset ───────────────────────────────────────────────────────────────────

class OmicsDataset(Dataset):
    def __init__(self, row_indices: np.ndarray, sequences: list[str],
                 guide_idxs: np.ndarray, seq_feats: np.ndarray,
                 omics: np.ndarray, labels: np.ndarray):
        self.row_indices = row_indices
        self.sequences   = sequences
        self.guide_idxs  = guide_idxs   # maps cell-feature row -> guide idx
        self.seq_feats   = seq_feats    # (n_guides, 450) normalised
        self.omics       = omics        # (n_rows, 5) normalised
        self.labels      = labels       # (n_rows,)

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, idx: int):
        row_i   = self.row_indices[idx]
        guide_i = self.guide_idxs[row_i]
        return (
            torch.from_numpy(seq_to_onehot(self.sequences[guide_i])),
            torch.from_numpy(self.seq_feats[guide_i]),
            torch.from_numpy(self.omics[row_i]),
            torch.tensor(self.labels[row_i], dtype=torch.float32),
        )


# ── Model ─────────────────────────────────────────────────────────────────────

class OmicsCRISPRModel(nn.Module):
    """
    Three-branch late-fusion model for multi-omics CRISPR efficacy prediction.
    Branches: (1) CNN on raw 20-mer, (2) MLP on 450-dim features, (3) MLP on omics.
    """
    def __init__(self, seq_feat_dim: int = SEQ_FEAT_DIM):
        super().__init__()

        # Branch 1: Sequence CNN (4 x 20) -> CNN_EMB_DIM
        self.seq_cnn = nn.Sequential(
            nn.Conv1d(4,   64,  kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64,  128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(4),    # -> (B, 128, 4)
            nn.Flatten(),               # -> (B, 512)
            nn.Linear(512, CNN_EMB_DIM),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        # Branch 2: Feature MLP (450-dim) -> FEAT_EMB_DIM
        self.feat_mlp = nn.Sequential(
            nn.Linear(seq_feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, FEAT_EMB_DIM),
            nn.BatchNorm1d(FEAT_EMB_DIM),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Branch 3: Omics MLP (5-dim) -> OMICS_EMB_DIM
        self.omics_mlp = nn.Sequential(
            nn.Linear(N_OMICS, 64),
            nn.GELU(),
            nn.Linear(64, OMICS_EMB_DIM),
            nn.GELU(),
        )

        # Late fusion: concat(288) -> scalar
        self.fusion = nn.Sequential(
            nn.Linear(FUSION_DIM, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, seq_oh: torch.Tensor, seq_feats: torch.Tensor,
                omics: torch.Tensor) -> torch.Tensor:
        cnn_emb   = self.seq_cnn(seq_oh)       # (B, 128)
        feat_emb  = self.feat_mlp(seq_feats)   # (B, 128)
        omics_emb = self.omics_mlp(omics)      # (B, 32)
        fused     = torch.cat([cnn_emb, feat_emb, omics_emb], dim=1)  # (B, 288)
        return self.fusion(fused).squeeze(-1)  # (B,)


# ── Loss ──────────────────────────────────────────────────────────────────────

def pearson_r(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_m   = pred   - pred.mean()
    target_m = target - target.mean()
    num      = (pred_m * target_m).sum()
    denom    = pred_m.norm() * target_m.norm() + 1e-8
    return num / denom


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse  = nn.functional.mse_loss(pred, target)
    pr   = 1.0 - pearson_r(pred, target)
    return (1.0 - PEARSON_WGHT) * mse + PEARSON_WGHT * pr


# ── LR schedule: linear warmup then cosine decay ──────────────────────────────

def make_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple:
    print("  Loading guide metadata...")
    guide_meta: dict[int, dict] = {}
    with open(GUIDE_META_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            guide_meta[int(row["guide_idx"])] = row

    n_guides   = len(guide_meta)
    sequences  = [""] * n_guides
    genes      = [""] * n_guides
    for gidx, row in guide_meta.items():
        sequences[gidx] = row.get("sequence") or row["guide_id"]
        genes[gidx]     = row["gene"]
    print(f"    {n_guides:,} guides")

    print("  Loading precomputed 450-dim sequence features...")
    seq_feats = np.load(SEQ_NPZ)["seq_features"].astype(np.float32)
    print(f"    shape: {seq_feats.shape}")

    print("  Loading cell-type omics features...")
    omics_list: list[list[float]] = []
    label_list: list[float]       = []
    guide_idx_list: list[int]     = []
    with open(CELL_FEAT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gidx = int(row["guide_idx"])
            omics_list.append([
                float(row["rna_tpm_log1p"]),
                float(row["atac_signal"]),
                float(row["atac_n_peaks"]),
                float(row["splice_dist_log"]),
                float(row["gene_effect"]),       # 6th feature — CERES essentiality
                float(row["cell_type_idx"]),
            ])
            # Use combined label (z-scored guide_eff + cell-type gene_effect)
            label_list.append(float(row["label"]))
            guide_idx_list.append(gidx)
    print(f"    {len(label_list):,} (guide, cell_type) rows")

    omics      = np.array(omics_list,    dtype=np.float32)
    labels     = np.array(label_list,    dtype=np.float32)
    guide_idxs = np.array(guide_idx_list, dtype=np.int32)

    omics[:, 2] = np.log1p(omics[:, 2])  # log1p(atac_n_peaks)

    return sequences, seq_feats, omics, labels, guide_idxs, genes


def gene_stratified_split(guide_idxs: np.ndarray, genes: list[str],
                          train_frac: float = TRAIN_FRAC, seed: int = 42
                          ) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique_genes = list({genes[i] for i in guide_idxs})
    rng.shuffle(unique_genes)
    train_genes = set(unique_genes[:int(len(unique_genes) * train_frac)])
    train_rows, val_rows = [], []
    for row_i, gidx in enumerate(guide_idxs):
        (train_rows if genes[gidx] in train_genes else val_rows).append(row_i)
    return np.array(train_rows), np.array(val_rows)


def normalise(arr: np.ndarray, mean=None, std=None):
    """z-score normalisation; compute stats if not given."""
    if mean is None:
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0) + 1e-8
    return (arr - mean) / std, mean, std


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, loader: DataLoader, device) -> tuple[float, float]:
    model.eval()
    all_pred, all_true = [], []
    total_loss = 0.0
    with torch.no_grad():
        for seq_oh, seq_f, omics, labels in loader:
            seq_oh = seq_oh.to(device)
            seq_f  = seq_f.to(device)
            omics  = omics.to(device)
            labels = labels.to(device)
            pred   = model(seq_oh, seq_f, omics)
            total_loss += combined_loss(pred, labels).item() * len(labels)
            all_pred.extend(pred.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    pred_a   = np.array(all_pred)
    true_a   = np.array(all_true)
    r = 0.0
    if pred_a.std() > 1e-8 and true_a.std() > 1e-8:
        r = float(np.corrcoef(pred_a, true_a)[0, 1])
    return avg_loss, r


# ── Main ──────────────────────────────────────────────────────────────────────

def train_omics_model() -> None:
    print(f"\n{'='*60}")
    print("  OmicsCRISPR Phase 3 (v2) -- Three-Branch Late-Fusion")
    print(f"{'='*60}")

    device = torch.device("cpu")
    print(f"  Device: {device}")

    # ── Load ──
    print("\nLoading data...")
    sequences, seq_feats, omics, labels, guide_idxs, genes = load_data()

    # ── Split ──
    print("\nBuilding gene-stratified split...")
    train_rows, val_rows = gene_stratified_split(guide_idxs, genes)
    train_genes = {genes[guide_idxs[i]] for i in train_rows}
    val_genes   = {genes[guide_idxs[i]] for i in val_rows}
    print(f"  Train rows: {len(train_rows):,}  |  Val rows: {len(val_rows):,}")
    print(f"  Train genes: {len(train_genes):,}  |  Val genes: {len(val_genes):,}  |  Overlap: 0")

    # ── Normalise features using training stats only (no leakage) ──
    print("\nNormalising features (training stats only)...")
    train_guide_idxs = np.unique(guide_idxs[train_rows])
    seq_feats_norm, sf_mean, sf_std = normalise(seq_feats,
                                                 seq_feats[train_guide_idxs].mean(0),
                                                 seq_feats[train_guide_idxs].std(0) + 1e-8)
    omics_norm, om_mean, om_std = normalise(omics,
                                             omics[train_rows].mean(0),
                                             omics[train_rows].std(0) + 1e-8)
    # Undo normalisation of discrete cell_type_idx (last column)
    omics_norm[:, 4] = omics[:, 4]

    # ── Datasets / loaders ──
    train_ds = OmicsDataset(train_rows, sequences, guide_idxs, seq_feats_norm, omics_norm, labels)
    val_ds   = OmicsDataset(val_rows,   sequences, guide_idxs, seq_feats_norm, omics_norm, labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model ──
    model     = OmicsCRISPRModel(seq_feat_dim=SEQ_FEAT_DIM).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = make_scheduler(optimizer, WARMUP_EPOCHS, MAX_EPOCHS)

    # ── Training loop ──
    print(f"\nTraining (up to {MAX_EPOCHS} epochs, patience={PATIENCE})...")
    print(f"  {'Epoch':>5}  {'TrainLoss':>10}  {'ValLoss':>10}  "
          f"{'ValR':>7}  {'LR':>8}  {'Time':>5}")
    print("  " + "-" * 54)

    log: list[dict] = []
    best_val_r   = -999.0
    best_state   = None
    patience_ctr = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for seq_oh, seq_f, omics_b, labels_b in train_loader:
            seq_oh   = seq_oh.to(device)
            seq_f    = seq_f.to(device)
            omics_b  = omics_b.to(device)
            labels_b = labels_b.to(device)

            optimizer.zero_grad()
            pred = model(seq_oh, seq_f, omics_b)
            loss = combined_loss(pred, labels_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(labels_b)

        train_loss /= len(train_loader.dataset)
        scheduler.step()

        val_loss, val_r = evaluate(model, val_loader, device)
        current_lr      = optimizer.param_groups[0]["lr"]
        elapsed         = time.time() - t0

        print(f"  {epoch:>5}  {train_loss:>10.4f}  {val_loss:>10.4f}  "
              f"{val_r:>7.4f}  {current_lr:>8.2e}  {elapsed:>4.1f}s")

        log.append({
            "epoch": epoch, "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5), "val_r": round(val_r, 5),
            "lr": current_lr,
        })

        if val_r > best_val_r:
            best_val_r   = val_r
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    # ── Save ──
    print(f"\n  Best val Pearson r: {best_val_r:.4f}")

    torch.save({
        "model_state_dict": best_state,
        "norm_stats": {
            "sf_mean": sf_mean.tolist(), "sf_std": sf_std.tolist(),
            "om_mean": om_mean.tolist(), "om_std": om_std.tolist(),
        },
        "model_config": {
            "seq_feat_dim": SEQ_FEAT_DIM, "cnn_emb_dim": CNN_EMB_DIM,
            "feat_emb_dim": FEAT_EMB_DIM, "omics_emb_dim": OMICS_EMB_DIM,
            "fusion_dim": FUSION_DIM, "n_omics": N_OMICS, "seq_len": SEQ_LEN,
        },
        "best_val_r": best_val_r,
    }, MODEL_PT)
    print(f"  Checkpoint -> {MODEL_PT.name}")

    with open(TRAINING_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    config = {
        "version": "v2 (three-branch)",
        "architecture": {
            "branch1": "CNN: Conv1d(4->64->128->128) + MaxPool + FC(512->128)",
            "branch2": "MLP: FC(450->256->128) on precomputed features",
            "branch3": "MLP: FC(5->64->32) on omics",
            "fusion":  "FC(288->128->64->1)",
        },
        "n_params": n_params,
        "hyperparams": {
            "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
            "lr": LR, "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE, "warmup_epochs": WARMUP_EPOCHS,
            "pearson_weight": PEARSON_WGHT,
        },
        "training": {
            "n_train_rows": len(train_rows), "n_val_rows": len(val_rows),
            "n_train_genes": len(train_genes), "n_val_genes": len(val_genes),
        },
        "best_val_r": best_val_r,
    }
    with open(MODEL_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"  Config    -> {MODEL_CONFIG.name}")
    print(f"\nPhase 3 (v2) complete.  Best val Pearson r = {best_val_r:.4f}")


if __name__ == "__main__":
    train_omics_model()
