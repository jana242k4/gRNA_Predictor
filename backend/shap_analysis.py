"""
SHAP feature importance analysis for publication.

Generates:
  1. shap_summary_bar.png       — mean |SHAP| per feature group (publication Fig)
  2. shap_beeswarm.png          — beeswarm plot (top 20 individual features)
  3. shap_ablation.txt          — Spearman r when each feature group is zeroed
  4. shap_feature_importances.csv — ranked importances

Run from backend/ directory:
    python shap_analysis.py
"""
import sys, csv, pickle
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
    _USE_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor  # type: ignore
    _USE_XGB = False

sys.path.insert(0, str(Path(__file__).parent))

from app.services.feature_engineering import (
    extract_features_batch, GUIDE_LEN, BASES, DINUCS,
)

import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_SEED = 42
MODEL_PKL   = Path(__file__).parent / "app" / "models" / "xgb_model.pkl"
DATA_CSV    = Path(__file__).parent / "data" / "combined_training_data.csv"
OUT_DIR     = Path(__file__).parent / "benchmark_results"

# ── Feature group definitions (indices in 450-dim vector) ─────────────────
_GROUPS = [
    ("Pos. one-hot (guide)",       0,    80),
    ("GC content",                 80,   81),
    ("Melting Tm",                 81,   82),
    ("Dinucleotide freq.",         82,   98),
    ("Seed GC",                    98,   99),
    ("Poly-T flag",                99,  100),
    ("Pos.-specific dinucs",      100,  404),
    ("Upstream context (4bp)",    404,  420),
    ("Downstream context (6bp)",  420,  444),
    ("GC clamp (3' end)",         444,  445),
    ("RNA hairpin proxy",         445,  446),
    ("Microhomology",             446,  447),
    ("Segmented Tm (3 windows)",  447,  450),
]


_TRAIN_SOURCES = {"Doench2016", "Doench2014"}

def load_data():
    sequences, scores, thirty_mers = [], [], []
    with open(DATA_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("source", "Doench2016") not in _TRAIN_SOURCES:
                continue
            seq   = row["sequence"].strip().upper()
            score = float(row["score"])
            if len(seq) == 20 and set(seq) <= set("ACGT") and 0 <= score <= 1:
                sequences.append(seq)
                scores.append(score)
                thirty_mers.append(row.get("thirty_mer", "").strip())
    return sequences, np.array(scores, dtype=np.float32), thirty_mers


def _make_xgb():
    """Return a fresh XGBRegressor with the same hyperparameters as train_model.py."""
    if _USE_XGB:
        return XGBRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=5,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.7,
            gamma=0.1, reg_alpha=0.05, reg_lambda=1.0,
            random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
        )
    # sklearn GBM fallback (XGBRegressor was aliased above)
    return XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        min_samples_leaf=3, subsample=0.8, random_state=RANDOM_SEED,
    )


def true_ablation(X_train, X_test, y_train, y_test, sp_base: float) -> list[tuple]:
    """
    True ablation study: for each feature group, zero those dims in BOTH
    X_train and X_test, retrain a fresh model from scratch, and evaluate
    on X_test with the same dims zeroed.

    This is the correct ablation design — the model cannot learn to compensate
    for absent features because they are absent during training too.

    Contrast with the zeroing-only approach in `run()` which merely tests
    sensitivity of the *already-trained* model to missing features.

    Reference: Lipton & Steinhardt (2018) "Troubling Trends in ML Scholarship".
    """
    ablation_rows = []
    n_groups = len(_GROUPS)
    for gi, (label, lo, hi) in enumerate(_GROUPS, 1):
        print(f"  True ablation [{gi}/{n_groups}]: removing '{label}' (dims {lo}:{hi})...")
        X_tr_abl          = X_train.copy()
        X_te_abl          = X_test.copy()
        X_tr_abl[:, lo:hi] = 0.0
        X_te_abl[:, lo:hi] = 0.0

        m = _make_xgb()
        if _USE_XGB:
            m.fit(X_tr_abl, y_train,
                  eval_set=[(X_te_abl, y_test)], verbose=False)
        else:
            m.fit(X_tr_abl, y_train)

        sp_abl = spearmanr(y_test, m.predict(X_te_abl).astype(np.float64)).statistic
        delta  = sp_base - sp_abl
        ablation_rows.append((label, float(sp_abl), float(delta)))
        print(f"    r = {sp_abl:.4f}  (drop = {delta:+.4f})")

    ablation_rows.sort(key=lambda x: x[2], reverse=True)
    return ablation_rows


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model and data...")
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)

    sequences, y, thirty_mers = load_data()
    X = extract_features_batch(sequences, thirty_mers)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED
    )
    print(f"  {len(X_test)} held-out test samples (same split as train_model.py)")

    # ── 1. SHAP values (TreeExplainer, subset for speed) ──────────────────
    print("Computing SHAP values (TreeExplainer)...")
    n_shap = min(500, len(X_test))
    background = shap.maskers.Independent(X_test[:200], max_samples=200)
    explainer  = shap.TreeExplainer(model, background)
    shap_exp   = explainer(X_test[:n_shap], check_additivity=False)
    sv         = shap_exp.values                   # (n_shap, 447)
    mean_abs   = np.abs(sv).mean(axis=0)           # (447,)

    # ── 2. Feature-group bar chart ─────────────────────────────────────────
    group_imp = [(label, float(mean_abs[lo:hi].sum())) for label, lo, hi in _GROUPS]
    group_imp.sort(key=lambda x: x[1], reverse=True)
    labels, vals = zip(*group_imp)

    colors = plt.cm.Blues(np.linspace(0.40, 0.85, len(labels)))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(list(labels)[::-1], list(vals)[::-1],
                   color=list(colors)[::-1], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean |SHAP value| (summed over feature group)", fontsize=11)
    ax.set_title(
        "Feature Group Importance (SHAP)\n"
        "450-dim XGBoost model, Doench 2016 held-out test set",
        fontsize=12
    )
    ax.spines[["top", "right"]].set_visible(False)
    vmax = max(vals)
    for bar, v in zip(bars, list(vals)[::-1]):
        ax.text(bar.get_width() + vmax * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=8.5)
    plt.tight_layout()
    p1 = OUT_DIR / "shap_summary_bar.png"
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p1}")

    # ── 3. Beeswarm top-20 features ────────────────────────────────────────
    print("Generating beeswarm plot (top 20 individual features)...")
    feature_names = _make_feature_names()
    top20_idx = np.argsort(mean_abs)[-20:][::-1]

    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(
        shap.Explanation(
            values=sv[:, top20_idx],
            base_values=shap_exp.base_values,
            data=X_test[:n_shap, top20_idx],
            feature_names=[feature_names[i] for i in top20_idx],
        ),
        max_display=20,
        show=False,
        plot_size=None,
    )
    plt.title("Top 20 Individual Feature SHAP Values\n(colour = feature magnitude)", fontsize=12)
    plt.tight_layout()
    p2 = OUT_DIR / "shap_beeswarm.png"
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p2}")

    # ── 4a. Sensitivity ablation (zeroing on pre-trained model) ──────────
    print("Running sensitivity ablation (zeroing on pre-trained model)...")
    y_baseline = model.predict(X_test).astype(np.float64)
    sp_base    = spearmanr(y_test, y_baseline).statistic

    sensitivity_rows = []
    for label, lo, hi in _GROUPS:
        X_abl           = X_test.copy()
        X_abl[:, lo:hi]  = 0.0
        sp_abl           = spearmanr(y_test, model.predict(X_abl).astype(np.float64)).statistic
        delta            = sp_base - sp_abl
        sensitivity_rows.append((label, sp_abl, delta))
        print(f"  {label:<35} r={sp_abl:.4f}  (drop={delta:+.4f})")

    sensitivity_rows.sort(key=lambda x: x[2], reverse=True)

    p3 = OUT_DIR / "shap_ablation.txt"
    with open(p3, "w", encoding="utf-8") as f:
        f.write("Sensitivity Ablation: Effect of Zeroing Each Feature Group on Pre-Trained Model\n")
        f.write("NOTE: This measures model sensitivity, not true feature contribution.\n")
        f.write("See true_ablation.txt for the correct retrain-based ablation.\n\n")
        f.write(f"Baseline Spearman r = {sp_base:.4f}  (n={len(y_test)} held-out guides)\n\n")
        f.write(f"{'Feature group':<35}  Spearman_ablated  Delta\n")
        f.write(f"{'-'*35}  ----------------  -----\n")
        for label, sp_abl, delta in sensitivity_rows:
            f.write(f"{label:<35}  {sp_abl:+.4f}            {delta:+.4f}\n")
    print(f"Saved: {p3}")

    # ── 4b. True ablation (retrain model with each feature group removed) ─
    print("\nRunning TRUE ablation (retrains model for each feature group)...")
    print("This takes ~10 minutes. Each of 13 feature groups = 1 full model retrain.\n")
    true_rows = true_ablation(X_train, X_test, y_train, y_test, sp_base)

    p3b = OUT_DIR / "true_ablation.txt"
    with open(p3b, "w", encoding="utf-8") as f:
        f.write("True Ablation Study: Retrain Model with Each Feature Group Removed\n")
        f.write("Method: zero feature dims in both X_train and X_test, retrain from scratch.\n")
        f.write("This is the correct ablation — model cannot compensate for absent features.\n\n")
        f.write(f"Baseline Spearman r = {sp_base:.4f}  (n={len(y_test)} held-out guides)\n\n")
        f.write(f"{'Feature group':<35}  Spearman_retrained  Delta\n")
        f.write(f"{'-'*35}  ------------------  -----\n")
        for label, sp_abl, delta in true_rows:
            f.write(f"{label:<35}  {sp_abl:+.4f}              {delta:+.4f}\n")
    print(f"Saved: {p3b}")

    # ── 5. Ranked feature importances CSV ─────────────────────────────────
    ranked = sorted(enumerate(mean_abs), key=lambda x: x[1], reverse=True)
    p4 = OUT_DIR / "shap_feature_importances.csv"
    with open(p4, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "feature_index", "feature_name", "mean_abs_shap"])
        for rank, (idx, imp) in enumerate(ranked[:100], 1):
            w.writerow([rank, idx, feature_names[idx], f"{imp:.6f}"])
    print(f"Saved: {p4}")

    print(f"\nBaseline Spearman r = {sp_base:.4f}")
    print("Top 5 drops from ablation:")
    for label, sp_abl, delta in ablation_rows[:5]:
        print(f"  Remove '{label}': r -> {sp_abl:.4f}  (delta={delta:+.4f})")


def _make_feature_names():
    """Create human-readable names for all 450 features."""
    names = []
    for pos in range(GUIDE_LEN):
        for b in BASES:
            names.append(f"p{pos+1}_{b}")
    names.append("gc_content")
    names.append("melting_Tm")
    for d in DINUCS:
        names.append(f"dinuc_{d}")
    names.append("seed_GC")
    names.append("poly_T")
    for pos in range(19):
        for d in DINUCS:
            names.append(f"posdinuc_p{pos+1}_{d}")
    for pos in range(4):
        for b in BASES:
            names.append(f"up_p{pos-4}_{b}")
    for pos in range(6):
        for b in BASES:
            names.append(f"dn_p{pos+1}_{b}")
    names.append("gc_clamp_3prime")
    names.append("hairpin_proxy")
    names.append("microhomology")
    names.append("Tm_PAM_distal")
    names.append("Tm_PAM_proximal_8bp")
    names.append("Tm_30mer_ctx")
    return names


if __name__ == "__main__":
    run()
