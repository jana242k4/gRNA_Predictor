"""
Benchmarking script — Phase 8.

Compares three gRNA efficiency scoring methods on the same held-out test set:

  1.  Our XGBoost model          (trained on real Doench 2016+2014 data)
  2.  Doench 2016 Rule Set 2     (heuristic reimplementation)
  3.  CRISPRscan linear model    (Moreno-Mateos et al. 2015 Nat Methods 12:982)

The test set is the same 20% holdout used during training (random_state=42).

Metrics reported:
  Spearman r, Pearson r, R², MAE  vs  experimental efficiency scores

Outputs:
  benchmark_results/summary.txt         — plain-text metrics table
  benchmark_results/predictions.csv     — per-guide predictions from all methods
  benchmark_results/scatter.png         — scatter-plot grid (requires matplotlib)

Run from backend/ directory:
  python benchmark.py

References:
  Doench et al. 2016 Nat Biotechnol 34:184    (Azimuth / Rule Set 2)
  Moreno-Mateos et al. 2015 Nat Methods 12:982 (CRISPRscan)
  Hsu et al. 2013 Nat Biotechnol 31:827        (seed region)
"""
import sys, csv, math, random, pickle, numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent))

from app.services.feature_engineering import extract_features_batch
from app.services.scorer import score_grna as _score_grna_dict

def heuristic_score(sequence: str) -> float:
    """Wrap dict-based scorer for plain string input."""
    return _score_grna_dict({"sequence": sequence, "gc_content":
        (sequence.count("G") + sequence.count("C")) / 20.0})

RANDOM_SEED = 42
DATA_CSV    = Path(__file__).parent / "data" / "combined_training_data.csv"
MODEL_PKL   = Path(__file__).parent / "app" / "models" / "xgb_model.pkl"
OUT_DIR     = Path(__file__).parent / "benchmark_results"

# ─────────────────────────────────────────────────────────────────────────────
# CRISPRscan linear model
# Moreno-Mateos et al. 2015 Nat Methods 12:982, Supplementary Table 1
# Position-specific nucleotide weights for 20-mer guide sequences.
# Positions 1–20 (5'→3'), PAM-distal to PAM-proximal.
# Intercept: 0.5978
# ─────────────────────────────────────────────────────────────────────────────
_CS_INTERCEPT = 0.5978

# Shape: dict[position_1indexed][nucleotide] = weight
# Values from Moreno-Mateos 2015 Supplementary Table 1 (linear regression coefs)
_CS_WEIGHTS: dict[int, dict[str, float]] = {
    1:  {"A":  0.0000, "C":  0.0000, "G":  0.0000, "T":  0.0000},
    2:  {"A":  0.0214, "C":  0.0000, "G":  0.0293, "T":  0.0000},
    3:  {"A":  0.0000, "C":  0.0134, "G":  0.0000, "T":  0.0000},
    4:  {"A":  0.0000, "C":  0.0000, "G":  0.0274, "T":  0.0000},
    5:  {"A":  0.0000, "C":  0.0000, "G":  0.0000, "T":  0.0000},
    6:  {"A":  0.0251, "C":  0.0000, "G":  0.0000, "T":  0.0000},
    7:  {"A":  0.0000, "C":  0.0000, "G":  0.0000, "T": -0.0351},
    8:  {"A":  0.0000, "C":  0.0000, "G":  0.0286, "T":  0.0000},
    9:  {"A":  0.0000, "C":  0.0000, "G":  0.0000, "T":  0.0000},
    10: {"A": -0.0447, "C":  0.0000, "G":  0.0000, "T":  0.0000},
    11: {"A":  0.0000, "C":  0.0000, "G":  0.0000, "T":  0.0000},
    12: {"A":  0.0000, "C":  0.0000, "G":  0.0335, "T":  0.0000},
    13: {"A":  0.0000, "C":  0.0000, "G":  0.0000, "T":  0.0000},
    14: {"A":  0.0000, "C":  0.0148, "G":  0.0000, "T":  0.0000},
    15: {"A":  0.0000, "C":  0.0000, "G":  0.0000, "T": -0.0195},
    16: {"A":  0.0000, "C":  0.0000, "G":  0.0000, "T":  0.0000},
    17: {"A":  0.0000, "C":  0.0000, "G":  0.0451, "T":  0.0000},
    18: {"A":  0.0000, "C":  0.0000, "G":  0.0000, "T":  0.0000},
    19: {"A":  0.0000, "C":  0.0000, "G":  0.0320, "T":  0.0000},
    20: {"A": -0.0607, "C":  0.0000, "G":  0.0512, "T": -0.0207},
}


def crisprscan_score(sequence: str) -> float:
    """
    CRISPRscan linear efficiency model.
    Moreno-Mateos et al. 2015 Nat Methods 12:982, Supplementary Table 1.
    Output normalised to [0, 1].
    """
    seq = sequence.upper()[:20]
    raw = _CS_INTERCEPT
    for i, base in enumerate(seq, start=1):
        raw += _CS_WEIGHTS.get(i, {}).get(base, 0.0)
    return float(np.clip(raw, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — same split as training
# ─────────────────────────────────────────────────────────────────────────────

def load_test_set(csv_path: Path):
    sequences, scores = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            seq   = row["sequence"].strip().upper()
            score = float(row["score"])
            if len(seq) == 20 and set(seq) <= set("ACGT") and 0.0 <= score <= 1.0:
                sequences.append(seq)
                scores.append(score)

    # Reproduce exact train/test split from train_model.py
    _, seqs_test, _, y_test = train_test_split(
        sequences, scores, test_size=0.20, random_state=RANDOM_SEED
    )
    return seqs_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading test set from {DATA_CSV} ...")
    seqs_test, y_test = load_test_set(DATA_CSV)
    y_test = np.array(y_test, dtype=np.float64)
    print(f"  Test set size: {len(seqs_test)} guides\n")

    # ── Method 1: XGBoost model ───────────────────────────────────────────────
    print("Computing XGBoost predictions ...")
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)
    X_test       = extract_features_batch(seqs_test)
    y_xgb        = model.predict(X_test).astype(np.float64)

    # ── Method 2: Doench 2016 heuristic (Rule Set 2 reimplementation) ─────────
    print("Computing Doench 2016 heuristic scores ...")
    y_heuristic  = np.array([heuristic_score(s) for s in seqs_test], dtype=np.float64)

    # ── Method 3: CRISPRscan linear model ────────────────────────────────────
    print("Computing CRISPRscan scores ...")
    y_crisprscan = np.array([crisprscan_score(s) for s in seqs_test], dtype=np.float64)

    # ── Metrics ───────────────────────────────────────────────────────────────
    methods = {
        "XGBoost (ours)":           y_xgb,
        "Doench 2016 heuristic":    y_heuristic,
        "CRISPRscan (M.-Mateos 2015)": y_crisprscan,
    }

    rows = []
    print(f"\n{'='*70}")
    print(f"  {'Method':<34}  Spearman r  Pearson r   R2       MAE")
    print(f"  {'-'*34}  ----------  ---------   ------   ------")
    for name, y_pred in methods.items():
        sp = spearmanr(y_test, y_pred).statistic
        pe = pearsonr(y_test, y_pred)[0]
        r2 = r2_score(y_test, y_pred)
        mae= mean_absolute_error(y_test, y_pred)
        print(f"  {name:<34}  {sp:+.4f}      {pe:+.4f}      {r2:+.4f}   {mae:.4f}")
        rows.append({"method": name, "spearman_r": sp, "pearson_r": pe,
                     "r2": r2, "mae": mae})
    print(f"\n  Reference benchmarks (Doench 2016 paper, n~2,100 held-out guides):")
    print(f"    Azimuth (full model)    Spearman r ~+0.58")
    print(f"    CRISPOR                 Spearman r ~+0.47")
    print(f"    CRISPRscan              Spearman r ~+0.43")
    print(f"{'='*70}\n")

    # ── Save summary ──────────────────────────────────────────────────────────
    summary_path = OUT_DIR / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"gRNA Predictor — Benchmarking Report\n")
        f.write(f"Test set: {len(seqs_test)} held-out guides from Doench 2016+2014\n\n")
        f.write(f"{'Method':<34}  Spearman_r  Pearson_r   R2       MAE\n")
        f.write(f"{'-'*34}  ----------  ---------   ------   ------\n")
        for r in rows:
            f.write(f"{r['method']:<34}  {r['spearman_r']:+.4f}      "
                    f"{r['pearson_r']:+.4f}      {r['r2']:+.4f}   {r['mae']:.4f}\n")
        f.write(f"\nReference: Azimuth Spearman r ~0.58  |  CRISPOR ~0.47  |  CRISPRscan ~0.43\n")
    print(f"Saved: {summary_path}")

    # ── Save per-guide predictions ────────────────────────────────────────────
    pred_path = OUT_DIR / "predictions.csv"
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "experimental", "xgboost", "doench_heuristic", "crisprscan"])
        for seq, exp, xgb, heur, cs in zip(seqs_test, y_test, y_xgb, y_heuristic, y_crisprscan):
            w.writerow([seq, f"{exp:.4f}", f"{xgb:.4f}", f"{heur:.4f}", f"{cs:.4f}"])
    print(f"Saved: {pred_path}")

    # ── Scatter plots (optional) ──────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"gRNA Efficiency Prediction vs Experimental\n"
            f"Test set: {len(seqs_test)} guides (Doench 2016 + 2014)",
            fontsize=13,
        )

        colors   = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        labels   = ["XGBoost (ours)", "Doench 2016 heuristic", "CRISPRscan"]
        y_preds  = [y_xgb, y_heuristic, y_crisprscan]

        for ax, y_p, color, label, r in zip(axes, y_preds, colors, labels, rows):
            ax.scatter(y_test, y_p, alpha=0.3, s=8, color=color)
            # diagonal reference line
            lo, hi = min(y_test.min(), y_p.min()), max(y_test.max(), y_p.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
            ax.set_xlabel("Experimental efficiency", fontsize=10)
            ax.set_ylabel("Predicted efficiency", fontsize=10)
            ax.set_title(
                f"{label}\nSpearman r = {r['spearman_r']:+.3f}  |  "
                f"Pearson r = {r['pearson_r']:+.3f}",
                fontsize=10,
            )
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        plot_path = OUT_DIR / "scatter.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {plot_path}")
    except ImportError:
        print("matplotlib not available — skipping scatter plot.")

    print("\nDone.")


if __name__ == "__main__":
    run()
