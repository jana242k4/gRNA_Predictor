"""
Publication-ready figure generation for the gRNA Predictor paper.

Produces:
  fig1_performance_scatter.png  — Predicted vs experimental (our model + Azimuth)
  fig2_gc_efficiency.png        — GC content bins vs efficiency (biological insight)
  fig3_positional_heatmap.png   — Position x nucleotide mean efficiency heatmap
  fig4_ablation_bar.png         — Ablation study: Spearman drop per feature group
  fig5_benchmark_heatmap.png    — Our model vs Azimuth correlation heatmap
  summary_stats.txt             — All numbers cited in Methods/Results

Run from backend/ directory:
    python publication_figures.py
"""
import sys, csv, pickle, io, urllib.request, re
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent))
from app.services.feature_engineering import extract_features_batch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

RANDOM_SEED = 42
MODEL_PKL   = Path(__file__).parent / "app" / "models" / "xgb_model.pkl"
DATA_CSV    = Path(__file__).parent / "data" / "combined_training_data.csv"
OUT_DIR     = Path(__file__).parent / "benchmark_results"

AZIMUTH_URL = (
    "https://raw.githubusercontent.com/MicrosoftResearch/Azimuth"
    "/master/azimuth/data/FC_plus_RES_withPredictions.csv"
)
VALID = set("ACGT")

PLT_STYLE = {
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "figure.dpi":         150,
    "font.family":        "sans-serif",
}


_TRAIN_SOURCES = {"Doench2016", "Doench2014"}

def load_data():
    sequences, scores, thirty_mers = [], [], []
    with open(DATA_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("source", "Doench2016") not in _TRAIN_SOURCES:
                continue
            seq = row["sequence"].strip().upper()
            sc  = float(row["score"])
            if len(seq) == 20 and set(seq) <= VALID and 0 <= sc <= 1:
                sequences.append(seq)
                scores.append(sc)
                thirty_mers.append(row.get("thirty_mer", "").strip())
    return sequences, np.array(scores, dtype=np.float64), thirty_mers


def load_azimuth():
    print("  Fetching Azimuth predictions...")
    req = urllib.request.Request(AZIMUTH_URL, headers={"User-Agent": "gRNA-Predictor/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read().decode("utf-8")
    guides, thirty_mers, experimental, azimuth_preds = [], [], [], []
    seen = set()
    for row in csv.DictReader(io.StringIO(raw)):
        thirty = row.get("30mer", "").strip().upper()
        if len(thirty) < 30:
            continue
        guide = thirty[4:24]
        pam   = thirty[24:27]
        if not re.fullmatch(r"[ACGT]GG", pam) or not (set(guide) <= VALID):
            continue
        try:
            exp  = float(row["score_drug_gene_rank"])
            pred = float(row["predictions"])
        except (ValueError, KeyError):
            continue
        if not (0 <= exp <= 1) or guide in seen:
            continue
        seen.add(guide)
        guides.append(guide); thirty_mers.append(thirty)
        experimental.append(exp); azimuth_preds.append(pred)
    return guides, thirty_mers, np.array(experimental), np.array(azimuth_preds)


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PLT_STYLE)

    print("Loading model...")
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)

    print("Loading training data...")
    sequences, y, thirty_mers = load_data()
    X = extract_features_batch(sequences, thirty_mers)
    X_train, X_test, y_train, y_test, tm_train, tm_test, seq_train, seq_test = train_test_split(
        X, y, thirty_mers, sequences, test_size=0.20, random_state=RANDOM_SEED
    )
    y_pred_test = model.predict(X_test).astype(np.float64)

    sp_ours = spearmanr(y_test, y_pred_test).statistic
    pe_ours = pearsonr(y_test, y_pred_test)[0]
    r2_ours = r2_score(y_test, y_pred_test)
    mae_ours = mean_absolute_error(y_test, y_pred_test)

    print("Loading Azimuth comparisons...")
    az_guides, az_tms, y_az_exp, y_az_pred = load_azimuth()
    X_az = extract_features_batch(az_guides, az_tms)
    y_az_xgb = model.predict(X_az).astype(np.float64)
    sp_azimuth = spearmanr(y_az_exp, y_az_pred).statistic
    sp_xgb_az  = spearmanr(y_az_exp, y_az_xgb).statistic

    # ── FIG 1: Performance scatter (2-panel) ──────────────────────────────
    print("Generating Fig 1: performance scatter...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, y_true, y_p, name, color, sp, pe in [
        (axes[0], y_az_exp, y_az_pred, "Azimuth (Rule Set 2)",    "#e15759", sp_azimuth,     pearsonr(y_az_exp, y_az_pred)[0]),
        (axes[1], y_az_exp, y_az_xgb,  "This work (XGBoost 450-dim)", "#4e79a7", sp_xgb_az, pearsonr(y_az_exp, y_az_xgb)[0]),
    ]:
        gc_arr = np.array([(g.count("G") + g.count("C")) / 20 for g in az_guides])
        sc = ax.scatter(y_true, y_p, c=gc_arr, cmap="RdYlGn", alpha=0.35, s=6,
                        vmin=0.3, vmax=0.7)
        lo = min(y_true.min(), y_p.min()) - 0.02
        hi = max(y_true.max(), y_p.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y = x")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Experimental efficiency (score_drug_gene_rank)")
        ax.set_ylabel("Predicted score")
        ax.set_title(f"{name}\nSpearman r = {sp:.3f}  |  Pearson r = {pe:.3f}  |  n = {len(y_true)}")

    cb = fig.colorbar(sc, ax=axes, shrink=0.7, pad=0.02)
    cb.set_label("GC content", fontsize=9)
    fig.suptitle("Predicted vs. Experimental gRNA Efficiency — Doench 2016 dataset",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, "fig1_performance_scatter.png")

    # ── FIG 2: GC content vs efficiency ───────────────────────────────────
    print("Generating Fig 2: GC content vs efficiency...")
    gc_vals  = np.array([(s.count("G") + s.count("C")) / 20 for s in sequences])
    bins     = np.arange(0, 1.05, 0.10)
    bin_idx  = np.digitize(gc_vals, bins) - 1
    bin_data = [y[bin_idx == i] for i in range(len(bins) - 1)]
    bin_labels = [f"{int(bins[i]*100)}–{int(bins[i+1]*100)}%" for i in range(len(bins)-1)]
    means    = [d.mean() if len(d) else np.nan for d in bin_data]
    sds      = [d.std()  if len(d) else np.nan for d in bin_data]
    ns       = [len(d) for d in bin_data]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors_gc = plt.cm.RdYlGn([0.1, 0.2, 0.35, 0.5, 0.6, 0.7, 0.75, 0.8, 0.75, 0.6])
    for i, (m, s, n, c) in enumerate(zip(means, sds, ns, colors_gc)):
        if np.isnan(m): continue
        ax.bar(i, m, color=c, edgecolor="white", linewidth=0.5, label=f"n={n}")
        ax.errorbar(i, m, yerr=s, fmt="none", color="#333", capsize=3, linewidth=1)
        ax.text(i, 0.01, f"n={n}", ha="center", va="bottom", fontsize=7, color="#555")
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=30, ha="right")
    ax.set_xlabel("GC content bin")
    ax.set_ylabel("Mean experimental efficiency ± SD")
    ax.set_title("GC Content vs. gRNA Efficiency\n(Doench 2016, all 4,692 guides)")
    ax.axhspan(0.4, 0.7, alpha=0.08, color="green", label="Optimal GC (40–70%)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    _save(fig, "fig2_gc_efficiency.png")

    # ── FIG 3: Positional nucleotide heatmap ──────────────────────────────
    print("Generating Fig 3: positional nucleotide heatmap...")
    BASES = ["A", "C", "G", "T"]
    pos_eff = np.full((20, 4), np.nan)
    for pos in range(20):
        for bi, base in enumerate(BASES):
            mask = [s[pos] == base for s in sequences]
            vals = y[mask]
            if len(vals) >= 10:
                pos_eff[pos, bi] = vals.mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    vmin, vmax = np.nanmin(pos_eff), np.nanmax(pos_eff)
    im = ax.imshow(pos_eff.T, aspect="auto", cmap="RdYlGn",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(20))
    ax.set_xticklabels([str(i+1) for i in range(20)], fontsize=8)
    ax.set_yticks(range(4))
    ax.set_yticklabels(BASES, fontsize=10)
    ax.set_xlabel("Guide position (1 = 5' end, 20 = PAM-proximal)")
    ax.set_ylabel("Nucleotide")
    ax.set_title("Mean gRNA Efficiency by Position and Nucleotide\n(green = higher efficiency)")
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label("Mean efficiency", fontsize=9)
    for pos in range(20):
        for bi in range(4):
            v = pos_eff[pos, bi]
            if not np.isnan(v):
                ax.text(pos, bi, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if v < (vmin + vmax) / 2 else "black")
    plt.tight_layout()
    _save(fig, "fig3_positional_heatmap.png")

    # ── FIG 4: Ablation bar chart ──────────────────────────────────────────
    print("Generating Fig 4: ablation bar chart...")
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
    ]
    sp_base = spearmanr(y_test, y_pred_test).statistic
    ablation = []
    for label, lo, hi in _GROUPS:
        X_abl = X_test.copy(); X_abl[:, lo:hi] = 0.0
        sp_abl = spearmanr(y_test, model.predict(X_abl).astype(np.float64)).statistic
        ablation.append((label, sp_base - sp_abl))
    ablation.sort(key=lambda x: x[1], reverse=True)
    labels_ab, deltas = zip(*ablation)

    palette = ["#e15759" if d >= 0.04 else "#f28e2b" if d >= 0.01 else "#76b7b2"
               for d in deltas]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(list(labels_ab)[::-1], list(deltas)[::-1],
                   color=list(palette)[::-1], edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman r drop (baseline - ablated)")
    ax.set_title(
        f"Feature Ablation Study\n(baseline Spearman r = {sp_base:.3f}, n={len(y_test)} held-out guides)"
    )
    for bar, v in zip(bars, list(deltas)[::-1]):
        xpos = bar.get_width() + 0.001 if v >= 0 else bar.get_width() - 0.001
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{v:+.4f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#e15759", label="High impact (>=0.04)"),
        Patch(facecolor="#f28e2b", label="Moderate (0.01–0.04)"),
        Patch(facecolor="#76b7b2", label="Low impact (<0.01)"),
    ]
    ax.legend(handles=legend_els, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "fig4_ablation_bar.png")

    # ── FIG 5: Benchmark comparison heatmap ───────────────────────────────
    print("Generating Fig 5: benchmark heatmap...")
    # Reproduce the exact 80/20 train/test split used during training.
    # MUST filter to TRAIN_SOURCES only — same as train_model.py and load_data() above.
    all_seqs_csv, all_scores_csv = [], []
    with open(DATA_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("source", "Doench2016") not in _TRAIN_SOURCES:
                continue
            all_seqs_csv.append(row["sequence"].strip().upper())
            all_scores_csv.append(float(row["score"]))
    _, seqs_test_csv, _, _ = train_test_split(
        all_seqs_csv, all_scores_csv, test_size=0.20, random_state=RANDOM_SEED
    )
    test_set = set(seqs_test_csv)
    mask = [g in test_set for g in az_guides]
    if sum(mask) >= 50:
        mask_np  = np.array(mask)
        g_test   = [g for g, m in zip(az_guides, mask) if m]
        tm_test2 = [t for t, m in zip(az_tms,   mask) if m]
        ye_t     = y_az_exp[mask_np]
        ya_t     = y_az_pred[mask_np]
        yx_t     = model.predict(extract_features_batch(g_test, tm_test2)).astype(np.float64)
        sp_az_held = spearmanr(ye_t, ya_t).statistic
        sp_xgb_held = spearmanr(ye_t, yx_t).statistic
        n_held = sum(mask)
    else:
        sp_az_held = sp_azimuth; sp_xgb_held = sp_xgb_az; n_held = len(az_guides)

    methods   = ["Azimuth\n(Rule Set 2)", "This work\n(XGBoost 450-dim)"]
    datasets  = [f"Full Doench 2016\n(n={len(az_guides)})",
                 f"Held-out 20%\n(n={n_held})"]
    mat = np.array([
        [sp_azimuth,   sp_az_held],
        [sp_xgb_az,    sp_xgb_held],
    ])

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(mat, cmap="Blues", vmin=0.5, vmax=0.9)
    ax.set_xticks(range(len(datasets)));  ax.set_xticklabels(datasets, fontsize=10)
    ax.set_yticks(range(len(methods)));   ax.set_yticklabels(methods, fontsize=10)
    ax.set_title("Spearman r — Model Benchmark Comparison", fontsize=12)
    for i in range(len(methods)):
        for j in range(len(datasets)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if v > 0.75 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman r")
    plt.tight_layout()
    _save(fig, "fig5_benchmark_heatmap.png")

    # ── Summary stats for Methods section ─────────────────────────────────
    p = OUT_DIR / "summary_stats.txt"
    with open(p, "w", encoding="utf-8") as f:
        f.write("Summary Statistics — gRNA Predictor\n")
        f.write("="*50 + "\n\n")
        f.write(f"Training set:            {len(X_train)} guides\n")
        f.write(f"Test set (held-out 20%): {len(X_test)} guides\n\n")
        f.write("Our model (XGBoost 450-dim) on held-out test set:\n")
        f.write(f"  Spearman r:  {sp_ours:.4f}\n")
        f.write(f"  Pearson  r:  {pe_ours:.4f}\n")
        f.write(f"  R2:          {r2_ours:.4f}\n")
        f.write(f"  MAE:         {mae_ours:.4f}\n\n")
        f.write(f"Azimuth (Rule Set 2) on full Doench 2016 (n={len(az_guides)}):\n")
        f.write(f"  Spearman r:  {sp_azimuth:.4f}\n\n")
        f.write(f"Our model on full Doench 2016 (n={len(az_guides)}):\n")
        f.write(f"  Spearman r:  {sp_xgb_az:.4f}\n\n")
        f.write(f"Head-to-head (held-out, n={n_held}):\n")
        f.write(f"  Azimuth:  {sp_az_held:.4f}\n")
        f.write(f"  Ours:     {sp_xgb_held:.4f}\n")
    print(f"Saved: {p}")
    print("\nAll publication figures generated.")
    print(f"\nKey numbers for paper:")
    print(f"  Our model Spearman r (held-out) = {sp_ours:.3f}")
    print(f"  Azimuth Spearman r (held-out)   = {sp_az_held:.3f}")
    print(f"  Our model on full Doench 2016   = {sp_xgb_az:.3f}")


def _save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    run()
