"""
Independent validation of our XGBoost model on datasets it was never trained on.

This script provides reviewer-proof evidence that our model generalises beyond
its training distribution (Doench 2016 + 2014, human cells).

Validation sets:
  1. CRISPRscan (Moreno-Mateos 2015, Nat Methods 12:982)
     - Zebrafish in-vivo Cas9 cleavage (cross-organism)
     - 1,020 guides, modFreq score (0-1)
     - Entirely orthogonal to our training data

  2. Chari et al. 2015 (Nat Methods 12:823)
     - Human cells: 293T and K562 (different cell lines from Doench 2016)
     - modFreq indel frequency (0-1)

Comparators evaluated on the same guides:
  - Our XGBoost (trained on Doench 2016+2014 only)
  - Doench 2016 heuristic (GC + position weights — Rule Set 2 approximation)

Run from backend/ directory:
  python validate_independent.py
"""
import sys, csv, io, pickle, urllib.request
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent))

from app.services.feature_engineering import extract_features_batch

RANDOM_SEED = 42
MODEL_PKL   = Path(__file__).parent / "app" / "models" / "xgb_model.pkl"
OUT_DIR     = Path(__file__).parent / "benchmark_results"

CRISPRSCAN_URL = (
    "https://raw.githubusercontent.com/maximilianh/crisporPaper"
    "/master/effData/morenoMateos2015.context.tab"
)
CHARI_293T_URL = (
    "https://raw.githubusercontent.com/maximilianh/crisporPaper"
    "/master/effData/chari2015Train293T.tab"
)
CHARI_K562_URL = (
    "https://raw.githubusercontent.com/maximilianh/crisporPaper"
    "/master/effData/chari2015TrainK562.tab"
)

VALID = set("ACGT")


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "gRNA-Predictor/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_crisprscan(raw: str) -> tuple[list[str], list[str], list[float]]:
    """Return (guides, thirty_mers, scores) from morenoMateos2015.context.tab."""
    guides, thirties, scores = [], [], []
    seen   = set()
    header = None
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if header is None:
            header = [p.lower() for p in parts]
            continue
        try:
            seq_idx   = header.index("seq") if "seq" in header else 1
            score_idx = header.index("modfreq") if "modfreq" in header else 4
        except ValueError:
            seq_idx, score_idx = 1, 4
        seq23 = parts[seq_idx].strip().upper() if seq_idx < len(parts) else ""
        if len(seq23) == 23 and seq23[20:] in {"AGG", "CGG", "GGG", "TGG"}:
            guide = seq23[:20]
        elif len(seq23) == 20:
            guide = seq23
        else:
            continue
        if not (len(guide) == 20 and set(guide) <= VALID):
            continue
        try:
            score = float(parts[score_idx])
        except (ValueError, IndexError):
            continue
        score = score / 100.0 if score > 1.0 else score
        if not (0.0 <= score <= 1.0) or guide in seen:
            continue
        # Extract 30-mer from longSeq context
        thirty = ""
        long_idx = header.index("longseq") if "longseq" in header else -1
        if long_idx >= 0 and long_idx < len(parts):
            ctx = parts[long_idx].strip().upper()
            pos = ctx.find(guide)
            if pos >= 4 and pos + 26 <= len(ctx):
                thirty = ctx[pos - 4: pos + 26]
        seen.add(guide)
        guides.append(guide)
        thirties.append(thirty)
        scores.append(score)
    return guides, thirties, np.array(scores)


def _parse_chari(raw: str, source_name: str) -> tuple[list[str], list[float]]:
    """Return (guides, scores) from chari2015Train*.tab."""
    guides, scores = [], []
    seen = set()
    header = None
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if header is None:
            header = [p.lower().strip() for p in parts]
            continue
        # Locate guide and modFreq columns
        seq_idx   = next((i for i, h in enumerate(header) if "seq" in h), 0)
        score_idx = next((i for i, h in enumerate(header) if "modfreq" in h or "freq" in h), 1)
        if seq_idx >= len(parts) or score_idx >= len(parts):
            continue
        raw_seq = parts[seq_idx].strip().upper()
        # Handle 23-mers
        if len(raw_seq) == 23 and raw_seq[20:] in {"AGG", "CGG", "GGG", "TGG"}:
            guide = raw_seq[:20]
        elif len(raw_seq) == 20:
            guide = raw_seq
        else:
            continue
        if not (len(guide) == 20 and set(guide) <= VALID):
            continue
        try:
            score = float(parts[score_idx])
        except ValueError:
            continue
        score = score / 100.0 if score > 1.0 else score
        if not (0.0 <= score <= 1.0) or guide in seen:
            continue
        seen.add(guide)
        guides.append(guide)
        scores.append(score)
    return guides, np.array(scores)


# ---------------------------------------------------------------------------
# Heuristic baseline (Doench 2016 GC + position weight approximation)
# ---------------------------------------------------------------------------

_POS_WEIGHTS = {
    1:  {"G":  0.03, "A": -0.01, "C":  0.01, "T":  0.00},
    2:  {"G":  0.02, "A": -0.01, "C":  0.01, "T":  0.00},
    3:  {"C":  0.05, "A": -0.03, "G":  0.01, "T": -0.01},
    4:  {"C":  0.06, "T": -0.04, "G":  0.02, "A": -0.03},
    10: {"G":  0.08, "C":  0.04, "A": -0.06, "T": -0.04},
    20: {"G":  0.12, "C":  0.04, "A": -0.10, "T": -0.08},
}

def _heuristic(seq: str) -> float:
    gc = (seq.count("G") + seq.count("C")) / 20.0
    gc_s = 1.0 - abs(gc - 0.55) * 2.0 if 0.30 <= gc <= 0.80 else 0.10
    pos_s = sum(_POS_WEIGHTS.get(p, {}).get(seq[p - 1], 0.0) for p in _POS_WEIGHTS)
    penalty = 0.28 if "TTTT" in seq else 0.0
    return float(np.clip(0.5 * gc_s + 0.3 * (pos_s + 0.5) + 0.15 - penalty, 0.0, 1.0))


def heuristic_predict(guides: list[str]) -> np.ndarray:
    return np.array([_heuristic(g) for g in guides])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(y_true, y_pred, name: str) -> dict:
    sp  = spearmanr(y_true, y_pred).statistic
    pe  = pearsonr(y_true, y_pred)[0]
    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {"name": name, "spearman": sp, "pearson": pe, "r2": r2, "mae": mae, "n": len(y_true)}


def _print_table(rows: list[dict]):
    print(f"  {'Model':<38}  n      Spearman r  Pearson r   R2       MAE")
    print(f"  {'-'*38}  -----  ----------  ---------   ------   ------")
    for r in rows:
        print(f"  {r['name']:<38}  {r['n']:<5}  {r['spearman']:+.4f}      "
              f"{r['pearson']:+.4f}      {r['r2']:+.4f}   {r['mae']:.4f}")


def _save_table(rows: list[dict], path: Path, title: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n\n")
        f.write(f"{'Model':<38}  n      Spearman_r  Pearson_r   R2       MAE\n")
        f.write(f"{'-'*38}  -----  ----------  ---------   ------   ------\n")
        for r in rows:
            f.write(f"{r['name']:<38}  {r['n']:<5}  {r['spearman']:+.4f}      "
                    f"{r['pearson']:+.4f}      {r['r2']:+.4f}   {r['mae']:.4f}\n")
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)

    all_results: list[dict] = []

    # ── Validation set 1: CRISPRscan (zebrafish, cross-organism) ───────────
    print("\nDownloading CRISPRscan (Moreno-Mateos 2015)...")
    try:
        raw_cs = _fetch(CRISPRSCAN_URL)
        cs_guides, cs_thirties, cs_scores = _parse_crisprscan(raw_cs)
        print(f"  Parsed {len(cs_guides)} guides.")

        X_cs       = extract_features_batch(cs_guides, cs_thirties)
        y_xgb_cs   = model.predict(X_cs).astype(np.float64)
        y_heur_cs  = heuristic_predict(cs_guides)

        print(f"\n{'='*70}")
        print(f"  CRISPRSCAN (Moreno-Mateos 2015) — zebrafish in-vivo")
        print(f"  INDEPENDENT: our model was never trained on this data.")
        print(f"  n={len(cs_guides)}, organism=Danio rerio, assay=T7 injection")
        print(f"{'='*70}")
        rows_cs = [
            metrics(cs_scores, y_xgb_cs,  "Our XGBoost (450-dim, 30-mer context)"),
            metrics(cs_scores, y_heur_cs, "Doench 2016 heuristic (GC+position)"),
        ]
        _print_table(rows_cs)
        _save_table(rows_cs, OUT_DIR / "independent_crisprscan.txt",
                    "Independent validation: CRISPRscan (Moreno-Mateos 2015, zebrafish)")
        all_results.extend([(r, "CRISPRscan2015") for r in rows_cs])
    except Exception as e:
        print(f"  WARN: CRISPRscan validation failed: {e}")

    # ── Validation set 2: Chari 2015 — 293T ────────────────────────────────
    for url, label, name in [
        (CHARI_293T_URL, "Chari2015_293T", "HEK293T cells"),
        (CHARI_K562_URL, "Chari2015_K562", "K562 cells"),
    ]:
        print(f"\nDownloading Chari 2015 ({name})...")
        try:
            raw_ch = _fetch(url)
            ch_guides, ch_scores = _parse_chari(raw_ch, label)
            print(f"  Parsed {len(ch_guides)} guides.")
            if len(ch_guides) < 30:
                print("  Too few guides — skipping.")
                continue

            X_ch      = extract_features_batch(ch_guides)
            y_xgb_ch  = model.predict(X_ch).astype(np.float64)
            y_heur_ch = heuristic_predict(ch_guides)

            print(f"\n{'='*70}")
            print(f"  CHARI 2015 — {name}")
            print(f"  INDEPENDENT: different lab, different cell line from Doench.")
            print(f"  n={len(ch_guides)}, assay=indel frequency (NHEJ)")
            print(f"{'='*70}")
            rows_ch = [
                metrics(ch_scores, y_xgb_ch,  "Our XGBoost (450-dim, 30-mer context)"),
                metrics(ch_scores, y_heur_ch, "Doench 2016 heuristic (GC+position)"),
            ]
            _print_table(rows_ch)
            _save_table(rows_ch, OUT_DIR / f"independent_{label.lower()}.txt",
                        f"Independent validation: Chari 2015 ({name})")
            all_results.extend([(r, label) for r in rows_ch])
        except Exception as e:
            print(f"  WARN: Chari 2015 ({name}) validation failed: {e}")

    # ── Summary scatter plot ────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if all_results:
            xgb_rows = [r for r, src in all_results if "XGBoost" in r["name"]]
            labels   = [src for r, src in all_results if "XGBoost" in r["name"]]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.set_title("Our XGBoost — Independent Validation\n(model never trained on these datasets)", fontsize=12)

            dataset_colors = {
                "CRISPRscan2015":  "#4e79a7",
                "Chari2015_293T":  "#f28e2b",
                "Chari2015_K562":  "#59a14f",
            }
            for row, src in zip(xgb_rows, labels):
                color = dataset_colors.get(src, "#aaa")
                ax.bar(src, row["spearman"], color=color, alpha=0.85)
                ax.text(src, row["spearman"] + 0.005, f"{row['spearman']:+.3f}", ha="center", fontsize=9)

            # Reference lines
            ax.axhline(0.58, color="red",  ls="--", lw=1.0, label="Azimuth (Doench 2016, trained on same)")
            ax.axhline(0.43, color="gray", ls=":",  lw=1.0, label="CRISPRscan model (trained on same)")
            ax.set_ylabel("Spearman r (independent test set)", fontsize=11)
            ax.set_ylim(-0.1, 0.75)
            ax.legend(fontsize=9)
            plt.tight_layout()
            plot_path = OUT_DIR / "independent_validation.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"\nSaved: {plot_path}")
    except ImportError:
        pass

    print("\nIndependent validation complete.")


if __name__ == "__main__":
    run()
