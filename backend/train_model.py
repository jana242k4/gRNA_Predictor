"""
Train a GradientBoostingRegressor on real CRISPR gRNA efficiency data.

Data priority:
  1. Real experimental data  → data/combined_training_data.csv
     (download first with: python download_datasets.py)
     Sources:
       - Doench et al. 2016 Nat Biotechnol 34:184  (Azimuth / Rule Set 2)
       - Doench et al. 2014 Nat Biotechnol 32:1262 (Rule Set 1)
  2. Synthetic fallback       → generated from Doench 2016 scoring rules
     (used automatically when data/ CSV not found)

Run from the backend/ directory:
    python download_datasets.py   # once — downloads ~7,400 real guides
    python train_model.py         # train and save model

Output: app/models/xgb_model.pkl
"""
import sys
import csv
import math
import random
import numpy as np
import pickle
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).parent))

from app.services.feature_engineering import extract_features_batch, BASES
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error
import json

# XGBoost preferred; fall back to sklearn GBM if not installed
try:
    from xgboost import XGBRegressor
    _USE_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    _USE_XGB = False

RANDOM_SEED       = 42
N_SYNTHETIC       = 15_000
DATA_CSV          = Path(__file__).parent / "data" / "combined_training_data.csv"
MODEL_OUT         = Path(__file__).parent / "app" / "models" / "xgb_model.pkl"

# ---------------------------------------------------------------------------
# SantaLucia 1998 nearest-neighbor parameters (inline — no app import needed)
# ---------------------------------------------------------------------------
_NN = {
    "AA": (-7.9, -22.2), "AT": (-7.2, -20.4), "TA": (-7.2, -21.3),
    "CA": (-8.5, -22.7), "GT": (-8.4, -22.4), "CT": (-7.8, -21.0),
    "GA": (-8.2, -22.2), "CG": (-10.6, -27.2), "GC": (-9.8, -24.4),
    "GG": (-8.0, -19.9), "AC": (-7.8, -21.0), "TC": (-7.9, -22.2),
    "AG": (-8.2, -22.2), "TG": (-8.5, -22.7), "TT": (-7.9, -22.2),
    "CC": (-8.0, -19.9),
}
_R  = 1.987
_CT = 250e-9

_POS = {
    1:  {"G":  0.03, "A": -0.01, "C":  0.01, "T":  0.00},
    2:  {"G":  0.02, "A": -0.01, "C":  0.01, "T":  0.00},
    3:  {"C":  0.05, "A": -0.03, "G":  0.01, "T": -0.01},
    4:  {"C":  0.06, "T": -0.04, "G":  0.02, "A": -0.03},
    5:  {"G":  0.03, "A": -0.02, "C":  0.01, "T": -0.01},
    6:  {"G":  0.04, "A": -0.02, "C":  0.01, "T": -0.02},
    7:  {"A": -0.02, "G":  0.02, "C":  0.01, "T": -0.01},
    8:  {"G":  0.03, "A": -0.03, "C":  0.02, "T": -0.02},
    9:  {"G":  0.04, "C":  0.03, "A": -0.04, "T": -0.03},
    10: {"G":  0.08, "C":  0.04, "A": -0.06, "T": -0.04},
    11: {"G":  0.05, "C":  0.03, "A": -0.04, "T": -0.03},
    12: {"G":  0.06, "C":  0.04, "A": -0.04, "T": -0.04},
    13: {"G":  0.05, "C":  0.03, "A": -0.03, "T": -0.03},
    14: {"C":  0.05, "G":  0.04, "A": -0.03, "T": -0.04},
    15: {"G":  0.05, "C":  0.04, "A": -0.04, "T": -0.04},
    16: {"C":  0.04, "G":  0.04, "T": -0.05, "A": -0.03},
    17: {"G":  0.05, "C":  0.03, "T": -0.07, "A": -0.04},
    18: {"G":  0.06, "C":  0.03, "A": -0.06, "T": -0.05},
    19: {"G":  0.07, "C":  0.03, "A": -0.04, "T": -0.06},
    20: {"G":  0.12, "C":  0.04, "A": -0.10, "T": -0.08},
}


def _nn_tm(seq: str) -> float:
    n, dH, dS = len(seq), 0.0, 0.0
    for i in range(n - 1):
        di = seq[i:i + 2]
        if di in _NN:
            h, s = _NN[di]
            dH += h; dS += s
    for end in (seq[0], seq[-1]):
        if end in ("A", "T"):
            dH += 2.3; dS += 4.1
        elif end in ("G", "C"):
            dH += 0.1; dS -= 2.8
    denom = dS + _R * math.log(_CT / 4.0)
    return (dH * 1000.0 / denom - 273.15) if abs(denom) > 1e-10 else 0.0


def _synthetic_label(seq: str) -> float:
    """Synthetic label from Doench 2016 scoring rules (fallback only)."""
    gc = (seq.count("G") + seq.count("C")) / 20.0
    if 0.40 <= gc <= 0.70:
        gc_s = 1.0 - abs(gc - 0.55) * 1.5
    elif gc < 0.25 or gc > 0.85:
        gc_s = 0.10
    else:
        gc_s = max(0.15, 1.0 - abs(gc - 0.55) * 3.0)

    pos_s = sum(_POS[p].get(seq[p - 1], 0.0) for p in _POS)

    tm = _nn_tm(seq)
    if 55.0 <= tm <= 65.0:
        tm_pen = 0.0
    elif tm < 45.0 or tm > 75.0:
        tm_pen = 0.15
    else:
        gap = max(0.0, 55.0 - tm) + max(0.0, tm - 65.0)
        tm_pen = min(0.12, gap * 0.012)

    penalty = tm_pen
    if "T" * 4 in seq:   penalty += 0.28
    for b in "ACGT":
        if b * 5 in seq: penalty += 0.10
    if "G" * 4 in seq:   penalty += 0.08

    seed   = seq[-12:]
    seed_at = (seed.count("A") + seed.count("T")) / 12.0
    penalty += min(0.18, seed_at * 12 * 0.012)
    seed_gc  = (seed.count("G") + seed.count("C")) / 12.0
    if seed_gc < 0.30 or seed_gc > 0.80:
        penalty += 0.08

    u6  = 0.03 if seq[0] == "G" else -0.03
    raw = 0.50 * gc_s + 0.25 * (pos_s + 0.5) + 0.10 + u6
    return float(np.clip(raw - penalty, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_real_data(
    path: Path,
    sources: set | None = None,
) -> tuple[list[str], list[float], list[str]]:
    """Load combined_training_data.csv — returns (sequences, scores, thirty_mers).

    Args:
        sources: if given, only load rows whose 'source' column is in this set.
                 Defaults to human-cell sources (Doench2016, Doench2014, DeepHF2019).
    """
    if sources is None:
        sources = {"Doench2016", "Doench2014"}
    sequences, scores, thirty_mers = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("source", "Doench2016") not in sources:
                continue
            seq   = row["sequence"].strip().upper()
            score = float(row["score"])
            if len(seq) == 20 and set(seq) <= set("ACGT") and 0.0 <= score <= 1.0:
                sequences.append(seq)
                scores.append(score)
                thirty_mers.append(row.get("thirty_mer", "").strip())
    return sequences, scores, thirty_mers


def generate_synthetic(n: int = N_SYNTHETIC, seed: int = RANDOM_SEED):
    """Generate synthetic (sequence, score) pairs as fallback."""
    random.seed(seed); np.random.seed(seed)
    seqs, scores = [], []
    for _ in range(n):
        if random.random() < 0.30:
            gc_t = random.uniform(0.35, 0.65)
            seq  = "".join(
                random.choice(["G", "C"]) if random.random() < gc_t
                else random.choice(["A", "T"])
                for _ in range(20)
            )
        else:
            seq = "".join(random.choices(BASES, k=20))
        seqs.append(seq)
        scores.append(_synthetic_label(seq))
    noise  = np.random.normal(0, 0.04, n).astype(np.float32)
    scores = list(np.clip(np.array(scores, np.float32) + noise, 0.0, 1.0))
    return seqs, scores


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    # --- Choose data source ---
    if DATA_CSV.exists():
        print(f"Loading REAL experimental data from {DATA_CSV} ...")
        sequences, scores, thirty_mers = load_real_data(DATA_CSV)
        data_source = "real"
        n_ctx = sum(1 for t in thirty_mers if t)
        print(f"  Loaded {len(sequences)} validated guides ({n_ctx} with 30-mer context).")
    else:
        print(f"data/combined_training_data.csv not found.")
        print(f"Run: python download_datasets.py  to use real data.")
        print(f"Falling back to {N_SYNTHETIC} synthetic samples...\n")
        sequences, scores = generate_synthetic()
        thirty_mers = [""] * len(sequences)
        data_source = "synthetic"

    print(f"\nExtracting 450-dim feature vectors (guide + 30-mer context + segmented Tm + GC-clamp + hairpin + microhomology)...")
    X = extract_features_batch(sequences, thirty_mers)
    y = np.array(scores, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED
    )

    if _USE_XGB:
        print(f"Training XGBRegressor on {len(X_train)} samples  [xgboost]...")
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.7,
            gamma=0.1,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
    else:
        print(f"Training GradientBoostingRegressor on {len(X_train)} samples  [sklearn]...")
        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=3,
            subsample=0.8,
            random_state=RANDOM_SEED,
        )
        model.fit(X_train, y_train)

    # --- Validation metrics ---
    y_pred = model.predict(X_test)

    r2    = r2_score(y_test, y_pred)
    mae   = mean_absolute_error(y_test, y_pred)
    sp, sp_p  = spearmanr(y_test, y_pred)
    pe, pe_p  = pearsonr(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  Data source:     {data_source} ({len(sequences)} guides)")
    print(f"  Test set size:   {len(X_test)}")
    print(f"  R2:              {r2:.4f}")
    print(f"  MAE:             {mae:.4f}")
    print(f"  Spearman r:      {sp:.4f}  (p={sp_p:.2e})")
    print(f"  Pearson r:       {pe:.4f}  (p={pe_p:.2e})")
    if data_source == "real":
        print(f"\n  Benchmark (Doench 2016 paper):")
        print(f"    Azimuth Spearman r ~0.58 on held-out data")
        print(f"    CRISPOR  Spearman r ~0.47")
        print(f"    CRISPRscan Spearman r ~0.43")
    print(f"{'='*50}\n")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved: {MODEL_OUT}")
    return model


def cross_validate_model(n_splits: int = 5) -> dict:
    """
    5-fold cross-validation on the full Doench 2016+2014 training dataset.

    Uses KFold(shuffle=True, random_state=RANDOM_SEED) so folds are
    deterministic and reproducible.  Features are extracted once on the full
    dataset and then sliced per fold — no redundant re-extraction.

    Returns a dict with per-fold and aggregate Spearman r / Pearson r / MAE.
    Results are also saved to benchmark_results/cv_results.json.
    """
    if not DATA_CSV.exists():
        print("data/combined_training_data.csv not found — CV requires real data.")
        print("Run: python download_datasets.py  first.")
        return {}

    print(f"Loading data for {n_splits}-fold cross-validation...")
    sequences, scores, thirty_mers = load_real_data(DATA_CSV)
    print(f"  {len(sequences)} guides loaded.")

    print("Extracting features (once for all folds)...")
    X = extract_features_batch(sequences, thirty_mers)
    y = np.array(scores, dtype=np.float32)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if _USE_XGB:
            fold_model = XGBRegressor(
                n_estimators=500, learning_rate=0.03, max_depth=5,
                min_child_weight=3, subsample=0.8, colsample_bytree=0.7,
                gamma=0.1, reg_alpha=0.05, reg_lambda=1.0,
                random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
            )
            fold_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        else:
            from sklearn.ensemble import GradientBoostingRegressor as _GBR
            fold_model = _GBR(
                n_estimators=300, learning_rate=0.05, max_depth=5,
                min_samples_leaf=3, subsample=0.8, random_state=RANDOM_SEED,
            )
            fold_model.fit(X_tr, y_tr)

        y_pred = fold_model.predict(X_te)
        sp  = float(spearmanr(y_te, y_pred).statistic)
        pe  = float(pearsonr(y_te, y_pred)[0])
        mae = float(mean_absolute_error(y_te, y_pred))
        fold_results.append({"fold": fold_i, "n_test": len(y_te),
                              "spearman_r": sp, "pearson_r": pe, "mae": mae})
        print(f"  Fold {fold_i}/{n_splits}:  Spearman r = {sp:.4f}  "
              f"Pearson r = {pe:.4f}  MAE = {mae:.4f}  (n={len(y_te)})")

    sps  = [r["spearman_r"] for r in fold_results]
    pes  = [r["pearson_r"]  for r in fold_results]
    maes = [r["mae"]        for r in fold_results]

    summary = {
        "n_folds":          n_splits,
        "n_total":          len(sequences),
        "spearman_r_mean":  float(np.mean(sps)),
        "spearman_r_std":   float(np.std(sps, ddof=1)),
        "pearson_r_mean":   float(np.mean(pes)),
        "pearson_r_std":    float(np.std(pes, ddof=1)),
        "mae_mean":         float(np.mean(maes)),
        "mae_std":          float(np.std(maes, ddof=1)),
        "folds":            fold_results,
    }

    print(f"\n{'='*50}")
    print(f"  {n_splits}-Fold Cross-Validation Summary")
    print(f"  n = {len(sequences)} guides")
    print(f"  Spearman r:  {summary['spearman_r_mean']:.4f} ± {summary['spearman_r_std']:.4f}")
    print(f"  Pearson r:   {summary['pearson_r_mean']:.4f} ± {summary['pearson_r_std']:.4f}")
    print(f"  MAE:         {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}")
    print(f"{'='*50}\n")

    out_dir = Path(__file__).parent / "benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv_path = out_dir / "cv_results.json"
    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"CV results saved: {cv_path}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", action="store_true",
                        help="Run 5-fold cross-validation instead of training")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    args = parser.parse_args()

    if args.cv:
        cross_validate_model(n_splits=args.cv_folds)
    else:
        train()
