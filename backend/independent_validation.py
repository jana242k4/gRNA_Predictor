"""
Independent cross-dataset validation — critical for publication.

Training set: Doench 2016 + Doench 2014 ONLY (Kim 2019 excluded — score scale
mismatch; kept as independent test set via 20% holdout in kim2019_holdout.csv).
Evaluates on datasets never seen during training:
  - Kim 2019 holdout (20% held-out from Kim 2019 — same assay, held-out split)
  - Chari et al. 2015 (293T/K562/A549/HepG2/SKNAS/U2OS — different lab, Nature Methods)
  - Xu et al. 2015 (human cell lines, Genome Research)
  - Moreno-Mateos et al. 2015 / CRISPRscan (zebrafish, in vivo — cross-organism)

Run from backend/ directory:
    python independent_validation.py
"""
import sys, csv, io, re, pickle, urllib.request
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from app.services.feature_engineering import extract_features_batch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_SEED = 42
MODEL_PKL   = Path(__file__).parent / "app" / "models" / "xgb_model.pkl"
DATA_CSV    = Path(__file__).parent / "data" / "combined_training_data.csv"
OUT_DIR     = Path(__file__).parent / "benchmark_results"

BASE_URL          = "https://raw.githubusercontent.com/maximilianh/crisporPaper/master/effData/"
KIM2019_HOLDOUT   = Path(__file__).parent / "data" / "kim2019_holdout.csv"
VALID             = set("ACGT")

DATASETS = [
    # (name, context_tab_url, score_column, score_scale, min_n)
    # --- Human cell lines (same organism, different labs / cell types) ---
    ("Chari 2015 (293T)",
     "chari2015Valid_293T.context.tab",   "modFreq", 100, 5),
    ("Chari 2015 (K562)",
     "chari2015Valid_K562.context.tab",   "modFreq", 100, 5),
    ("Chari 2015 (A549)",
     "chari2015Valid_A549.context.tab",   "modFreq", 100, 5),
    ("Chari 2015 (HepG2)",
     "chari2015Valid_HepG2.context.tab",  "modFreq", 100, 5),
    ("Chari 2015 (SKNAS)",
     "chari2015Valid_SKNAS.context.tab",  "modFreq", 100, 5),
    ("Chari 2015 (U2OS)",
     "chari2015Valid_U2OS.context.tab",   "modFreq", 100, 5),
    ("Xu 2015 (human cells)",
     "xu2015.context.tab",                "modFreq", 100, 10),
    # --- Cross-organism (expected lower — model trained on human) ---
    ("Moreno-Mateos 2015 (zebrafish)",
     "morenoMateos2015.context.tab",      "modFreq", 1,   50),
]


def fetch_dataset(name, url_suffix, score_col, score_scale, min_n):
    """Download a CRISPOR-formatted context.tab file and return (guides, thirty_mers, scores)."""
    url = BASE_URL + url_suffix
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "gRNA-Predictor/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8")
    except Exception as e:
        print(f"  SKIP {name}: download failed ({e})")
        return None

    guides, thirty_mers, scores = [], [], []
    seen = set()
    for row in csv.DictReader(io.StringIO(raw), delimiter="\t"):
        seq      = row.get("seq", "").strip().upper()
        long_seq = row.get("longSeq", "").strip().upper()
        try:
            sc = float(row[score_col]) / score_scale   # normalise to 0-1
        except (ValueError, KeyError):
            continue

        guide = seq[:20]
        pam   = seq[20:23] if len(seq) >= 23 else ""

        if len(guide) != 20 or not (set(guide) <= VALID):
            continue
        if not (0.0 <= sc <= 1.0):
            sc = min(1.0, max(0.0, sc))   # clamp extreme outliers
        if guide in seen:
            continue
        seen.add(guide)

        # Build 30-mer: 4bp upstream (from longSeq[26:30]) + guide (longSeq[30:50]) + PAM + NNN
        if len(long_seq) >= 50 and long_seq[30:50] == guide:
            upstream4 = long_seq[26:30]
            thirty    = upstream4 + guide + pam + "NNN"
        elif len(long_seq) >= 50:
            # fallback: guide appears elsewhere — try to locate it
            idx = long_seq.find(guide)
            if idx >= 4:
                upstream4 = long_seq[idx - 4: idx]
                thirty    = upstream4 + guide + pam + "NNN"
            else:
                thirty = ""
        else:
            thirty = ""

        guides.append(guide)
        thirty_mers.append(thirty)
        scores.append(sc)

    print(f"  {name}: {len(guides)} guides loaded (from {url_suffix})")
    if len(guides) < min_n:
        print(f"  SKIP {name}: too few guides ({len(guides)} < {min_n})")
        return None
    return guides, thirty_mers, np.array(scores, dtype=np.float64)


def load_kim2019_holdout():
    """Load the 20% Kim 2019 holdout set saved by download_datasets.py.

    These guides were withheld from training (deterministic 80/20 split, seed=42).
    Returns (guides, thirty_mers, scores) or None if file not found.
    """
    if not KIM2019_HOLDOUT.exists():
        print(f"  SKIP Kim 2019 holdout: {KIM2019_HOLDOUT} not found.")
        print(f"  Run: python download_datasets.py  to create it.")
        return None

    guides, thirty_mers, scores = [], [], []
    with open(KIM2019_HOLDOUT, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g  = row["sequence"].strip().upper()
            tm = row.get("thirty_mer", "").strip().upper()
            try:
                sc = float(row["score"])
            except (ValueError, KeyError):
                continue
            guides.append(g)
            thirty_mers.append(tm)
            scores.append(sc)

    print(f"  Kim 2019 holdout: {len(guides)} guides loaded from {KIM2019_HOLDOUT.name}")
    return guides, thirty_mers, np.array(scores, dtype=np.float64)


TRAINING_SOURCES = {"Doench2016", "Doench2014"}   # must match train_model.py sources

def load_training_guides():
    """Return guides actually used for training (filters by TRAINING_SOURCES)."""
    seqs = set()
    with open(DATA_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("source", "Doench2016") in TRAINING_SOURCES:
                seqs.add(row["sequence"].strip().upper())
    return seqs


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 n_boot: int = 10_000, ci: float = 0.95,
                 seed: int = 42) -> tuple[float, float]:
    """
    Bootstrap confidence interval for Spearman r.

    Resamples (y_true, y_pred) pairs with replacement n_boot times and
    computes the Spearman r for each resample.  Returns the (lower, upper)
    percentile bounds for the given CI level.

    Reference: Efron & Tibshirani (1993) An Introduction to the Bootstrap.
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    rs  = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx    = rng.integers(0, n, size=n)
        rs[i]  = spearmanr(y_true[idx], y_pred[idx]).statistic
    alpha = 1.0 - ci
    return float(np.percentile(rs, 100 * alpha / 2)), float(np.percentile(rs, 100 * (1 - alpha / 2)))


def metrics(y_true, y_pred, name):
    sp  = spearmanr(y_true, y_pred).statistic
    pe  = pearsonr(y_true, y_pred)[0]
    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    ci_lo, ci_hi = bootstrap_ci(y_true, y_pred)
    return {"name": name, "spearman": sp, "pearson": pe, "r2": r2, "mae": mae,
            "n": len(y_true), "ci_lo": ci_lo, "ci_hi": ci_hi}


def precision_at_k(y_true, y_pred, k, threshold_pct=80):
    """
    Fraction of top-k predicted guides that exceed the threshold_pct-th percentile
    of experimental efficiency.  Returns NaN when n < k.
    """
    if len(y_true) < k:
        return float("nan")
    threshold = np.percentile(y_true, threshold_pct)
    top_k_idx = np.argsort(y_pred)[::-1][:k]
    hits = sum(1 for i in top_k_idx if y_true[i] >= threshold)
    return hits / k


def _print_precision_at_k(y_true, y_pred, ks=(1, 3, 5, 10)):
    """Print precision@k line (top-20% hit threshold)."""
    parts = []
    for k in ks:
        p = precision_at_k(y_true, y_pred, k)
        parts.append(f"P@{k}={p:.2f}" if p == p else f"P@{k}=n/a")
    print(f"  Precision@k (top-20% threshold): {' | '.join(parts)}")


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)

    train_guides = load_training_guides()
    print(f"Training set: {len(train_guides)} unique guides\n")

    all_results = []

    # ── Kim 2019 holdout (20% withheld from training — same assay, held-out) ─
    print("Loading: Kim 2019 holdout (20% split withheld from training)...")
    kim_result = load_kim2019_holdout()
    if kim_result is not None:
        g_kim, tm_kim, y_kim = kim_result
        n_ov = sum(1 for g in g_kim if g in train_guides)
        print(f"  Overlap with training set: {n_ov}/{len(g_kim)} "
              f"({100*n_ov/len(g_kim):.1f}%) — should be ~0%")
        X_kim  = extract_features_batch(g_kim, tm_kim)
        yp_kim = model.predict(X_kim).astype(np.float64)
        m_kim  = metrics(y_kim, yp_kim, f"Kim 2019 holdout (n={len(g_kim)})")
        print(f"  Spearman r (all):   {m_kim['spearman']:+.4f}  "
              f"95% CI [{m_kim['ci_lo']:+.4f}, {m_kim['ci_hi']:+.4f}]  "
              f"Pearson r: {m_kim['pearson']:+.4f}")
        _print_precision_at_k(y_kim, yp_kim)

        # Novel-only: exclude guides whose sequence appeared in Doench training
        novel_mask = [g not in train_guides for g in g_kim]
        n_nov = sum(novel_mask)
        if n_nov >= 10:
            g_nov  = [g for g, m in zip(g_kim,  novel_mask) if m]
            tm_nov = [t for t, m in zip(tm_kim, novel_mask) if m]
            y_nov  = y_kim[np.array(novel_mask)]
            X_nov  = extract_features_batch(g_nov, tm_nov)
            yp_nov = model.predict(X_nov).astype(np.float64)
            m_nov  = metrics(y_nov, yp_nov, f"Kim 2019 novel-only (n={n_nov})")
            print(f"  Spearman r (novel): {m_nov['spearman']:+.4f}  "
                  f"95% CI [{m_nov['ci_lo']:+.4f}, {m_nov['ci_hi']:+.4f}]  "
                  f"n={n_nov} ({100*n_nov/len(g_kim):.0f}% of holdout)")
            all_results.append(("Kim 2019 (novel only)", m_nov, m_nov))
        else:
            all_results.append(("Kim 2019 (holdout, 20%)", m_kim, m_kim))
        print()

    # ── Evaluate on each CRISPOR-format dataset ───────────────────────────
    for name, url_suffix, score_col, scale, min_n in DATASETS:
        print(f"Fetching: {name}...")
        result = fetch_dataset(name, url_suffix, score_col, scale, min_n)
        if result is None:
            continue
        guides, thirty_mers, y_true = result

        # Check overlap with training set
        n_overlap = sum(1 for g in guides if g in train_guides)
        n_novel   = len(guides) - n_overlap
        print(f"  Overlap with training set: {n_overlap}/{len(guides)} ({100*n_overlap/len(guides):.1f}%)")
        print(f"  Novel (unseen) guides: {n_novel}")

        # Use only novel guides for the strictest test
        if n_novel >= 5:
            novel_mask   = [g not in train_guides for g in guides]
            guides_novel = [g for g, m in zip(guides, novel_mask) if m]
            tms_novel    = [t for t, m in zip(thirty_mers, novel_mask) if m]
            y_novel      = y_true[np.array(novel_mask)]
        else:
            guides_novel = guides; tms_novel = thirty_mers; y_novel = y_true

        X_all    = extract_features_batch(guides, thirty_mers)
        X_novel  = extract_features_batch(guides_novel, tms_novel)
        y_pred_all   = model.predict(X_all).astype(np.float64)
        y_pred_novel = model.predict(X_novel).astype(np.float64)

        m_all   = metrics(y_true,  y_pred_all,   f"{name} (all, n={len(guides)})")
        m_novel = metrics(y_novel, y_pred_novel, f"{name} (novel only, n={len(guides_novel)})")

        n_all = m_all['n']
        ci_note = ""
        if n_all < 30:
            # SE of Spearman r ≈ 1/sqrt(n-3); 95% CI ≈ ±1.96/sqrt(n-3)
            import math as _math
            se = 1.0 / _math.sqrt(max(n_all - 3, 1))
            ci_note = f"  *** n={n_all} is small; 95% CI ≈ ±{1.96*se:.2f} — treat with caution ***"
        print(f"  Spearman r (all guides):   {m_all['spearman']:+.4f}  "
              f"95% CI [{m_all['ci_lo']:+.4f}, {m_all['ci_hi']:+.4f}]"
              f"{('  (n='+str(n_all)+')') if n_all < 30 else ''}")
        print(f"  Spearman r (novel guides): {m_novel['spearman']:+.4f}  "
              f"95% CI [{m_novel['ci_lo']:+.4f}, {m_novel['ci_hi']:+.4f}]")
        if ci_note:
            print(f"  {ci_note}")
        _print_precision_at_k(y_true, y_pred_all)
        all_results.append((name, m_all, m_novel))
        print()

    if not all_results:
        print("No independent datasets could be loaded.")
        return

    # ── Summary table ─────────────────────────────────────────────────────
    print("=" * 70)
    print("  INDEPENDENT VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  {'Dataset':<35}  n_all  n_novel  Spearman(all)  Spearman(novel)  CI_note")
    print(f"  {'-'*35}  -----  -------  -------------  ---------------  -------")
    import math as _math
    for name, m_all, m_novel in all_results:
        n = m_all['n']
        if n < 30:
            se = 1.0 / _math.sqrt(max(n - 3, 1))
            ci_tag = f"±{1.96*se:.2f} (n={n})"
        else:
            ci_tag = ""
        print(f"  {name:<35}  {n:<5}  {m_novel['n']:<7}  "
              f"{m_all['spearman']:+.4f}         {m_novel['spearman']:+.4f}          {ci_tag}")
    if any(m_all['n'] < 30 for _, m_all, _ in all_results):
        print(f"\n  NOTE: Datasets with n<30 have wide 95% CIs — do not headline these correlations.")

    # ── Save results ──────────────────────────────────────────────────────
    p = OUT_DIR / "independent_validation.txt"
    with open(p, "w", encoding="utf-8") as f:
        f.write("Independent Cross-Dataset Validation\n")
        f.write("Model trained on: Doench 2016 + Doench 2014 (combined_training_data.csv)\n\n")
        f.write(f"{'Dataset':<35}  n_all  n_novel  Spearman_all  CI_95_low  CI_95_high  Spearman_novel  Pearson_all  MAE_all\n")
        f.write(f"{'-'*35}  -----  -------  ------------  ---------  ----------  --------------  -----------  -------\n")
        for name, m_all, m_novel in all_results:
            f.write(f"{name:<35}  {m_all['n']:<5}  {m_novel['n']:<7}  "
                    f"{m_all['spearman']:+.4f}        {m_all['ci_lo']:+.4f}     {m_all['ci_hi']:+.4f}      "
                    f"{m_novel['spearman']:+.4f}          "
                    f"{m_all['pearson']:+.4f}       {m_all['mae']:.4f}\n")
    print(f"\nSaved: {p}")

    # ── Multi-dataset comparison bar plot ─────────────────────────────────
    if len(all_results) >= 2:
        names_plot = [r[0] for r in all_results]
        sp_vals    = [r[1]["spearman"] for r in all_results]
        ns         = [r[1]["n"] for r in all_results]
        ci_lo      = [r[1]["ci_lo"]    for r in all_results]
        ci_hi      = [r[1]["ci_hi"]    for r in all_results]
        yerr_lo    = [max(0.0, sp - lo) for sp, lo in zip(sp_vals, ci_lo)]
        yerr_hi    = [max(0.0, hi - sp) for sp, hi in zip(sp_vals, ci_hi)]
        # Colour: blue for human, orange for cross-organism
        colors = ["#e05c5c" if "zebrafish" in n.lower() else "#4e79a7" for n in names_plot]

        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(names_plot))
        bars = ax.bar(x, sp_vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.errorbar(x, sp_vals, yerr=[yerr_lo, yerr_hi],
                    fmt="none", color="black", capsize=4, linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(names_plot, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Spearman r")
        ax.set_ylim(-0.15, 1.0)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(
            "Cross-Dataset Generalizability (model trained on Doench 2016+2014 only)\n"
            "Blue = human cell lines | Red = cross-organism (zebrafish)"
        )
        ax.spines[["top", "right"]].set_visible(False)
        for xi, (v, n) in enumerate(zip(sp_vals, ns)):
            yoff = 0.015 if v >= 0 else -0.05
            ax.text(xi, v + yoff, f"{v:.2f}\n(n={n})", ha="center", va="bottom", fontsize=7.5)
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="#4e79a7", label="Human cell lines"),
            Patch(color="#e05c5c", label="Cross-organism (zebrafish)"),
        ], fontsize=9)
        plt.tight_layout()
        pp = OUT_DIR / "fig6_independent_validation.png"
        fig.savefig(pp, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {pp}")


if __name__ == "__main__":
    run()
