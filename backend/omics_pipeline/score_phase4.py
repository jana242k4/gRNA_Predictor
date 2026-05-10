"""
OmicsCRISPR Phase 4 -- Splice Disruption Risk + Cell-Type Suitability

Module 1 – Splice Disruption Risk
  For each guide, computes a risk score (0–1) that the Cas9 cut will
  disrupt a splice donor or acceptor signal.

  Distance thresholds (Ke et al. 2018 Cell; Vaz-Drago et al. 2017 HMG):
    ≤3 bp   : critical — cut overlaps the splice signal consensus
    ≤10 bp  : high     — disrupts branch point or splice signal flanks
    ≤20 bp  : moderate — exonic splicing enhancer (ESE) disruption possible
    ≤50 bp  : low      — distant ESE / silencer effect
    >50 bp  : minimal

  Donor sites carry +10% extra risk vs acceptors (donor GT consensus is
  shorter and more sensitive to single-base disruption; Roca et al. 2013).

Module 2 – Cell-Type Suitability Scorer
  For every (guide, cell_type) pair produces a composite suitability score
  that combines five signals, each rank-normalised to [0, 1] within the
  set of guides for that cell type:

    guide_efficacy  (0.30) : base ML efficiency score (Doench-trained)
    rna_expression  (0.25) : log1p(TPM) of target gene in cell type
    atac_access     (0.20) : ATAC-seq signal at cut site in cell type
    splice_safety   (0.15) : 1 – splice_risk
    gene_essent     (0.10) : CERES essentiality (–gene_effect, clipped ≥0)

  Weights reflect the relative importance of each signal for predicting
  whether a guide will produce a clean, measurable phenotype in a given
  cell-type context.

Outputs  (data/omics/features/):
  splice_risk.csv         one row per guide (68k rows)
  cell_suitability.csv    one row per (guide, cell_type) (340k rows)
  phase4_summary.json     coverage and score distribution stats

Run from backend/:
    python -m omics_pipeline.score_phase4
"""
import bisect
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).parent.parent
sys.path.insert(0, str(_BACKEND))

from .config import FEATURES_DIR, SPLICE_BED_OUT

GUIDE_META_CSV      = FEATURES_DIR / "guide_metadata.csv"
CELL_FEAT_CSV       = FEATURES_DIR / "cell_features.csv"
SPLICE_RISK_CSV     = FEATURES_DIR / "splice_risk.csv"
CELL_SUIT_CSV       = FEATURES_DIR / "cell_suitability.csv"
PHASE4_SUMMARY_JSON = FEATURES_DIR / "phase4_summary.json"

# Suitability component weights (must sum to 1.0)
W_EFFICACY  = 0.30
W_RNA       = 0.25
W_ATAC      = 0.20
W_SPLICE    = 0.15
W_ESSENT    = 0.10


# ── Splice risk scoring ───────────────────────────────────────────────────────

def _splice_risk(dist_bp: float, site_type: str) -> float:
    """Piecewise-linear risk score, with donor sites 10% riskier."""
    if dist_bp <= 3:
        base = 0.95
    elif dist_bp <= 10:
        base = 0.95 - (dist_bp - 3) * (0.95 - 0.75) / 7.0
    elif dist_bp <= 20:
        base = 0.75 - (dist_bp - 10) * (0.75 - 0.40) / 10.0
    elif dist_bp <= 50:
        base = 0.40 - (dist_bp - 20) * (0.40 - 0.10) / 30.0
    else:
        base = max(0.0, 0.10 * math.exp(-(dist_bp - 50) / 80.0))

    if site_type == "donor":
        base = min(1.0, base * 1.10)

    return round(base, 4)


def _load_typed_splice_db() -> dict:
    """
    Load splice BED -> {(chr, strand): sorted list of (pos, type)}.
    pos is the single-base splice site position (end column, 0-based).
    """
    if not SPLICE_BED_OUT.exists():
        print(f"  WARNING: {SPLICE_BED_OUT.name} not found — splice risk set to 0")
        return {}

    db: dict[tuple, list] = {}
    with open(SPLICE_BED_OUT, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = (row["chr"], row["strand"])
            if key not in db:
                db[key] = []
            db[key].append((int(row["end"]), row["type"]))

    for key in db:
        db[key].sort(key=lambda x: x[0])

    total = sum(len(v) for v in db.values())
    print(f"  Loaded {total:,} typed splice sites across {len(db)} chr/strand buckets")
    return db


def _nearest_splice(db: dict, chr_: str, pos: int, strand: str) -> tuple[int, str]:
    """
    Binary search for nearest splice site to pos on (chr, strand).
    Returns (distance_bp, site_type). Distance is 0 if pos == site.
    """
    sites = db.get((chr_, strand))
    if not sites:
        return (999_999, "unknown")

    positions = [s[0] for s in sites]
    i = bisect.bisect_left(positions, pos)

    candidates = []
    if i < len(sites):
        candidates.append(sites[i])
    if i > 0:
        candidates.append(sites[i - 1])

    best_pos, best_type = min(candidates, key=lambda s: abs(s[0] - pos))
    return (abs(best_pos - pos), best_type)


# ── Module 1: build splice_risk.csv ──────────────────────────────────────────

def build_splice_risk() -> dict:
    """
    One row per guide: cut site position, nearest splice site distance,
    type (donor/acceptor), and risk score.
    Returns a {guide_id: splice_risk_float} dict for downstream use.
    """
    print(f"\n{'='*60}")
    print("  Phase 4 / Module 1 — Splice Disruption Risk")
    print(f"{'='*60}")
    t0 = time.time()

    db = _load_typed_splice_db()

    rows = []
    risk_map: dict[str, float] = {}

    with open(GUIDE_META_CSV, encoding="utf-8") as f:
        for guide in csv.DictReader(f):
            gid    = guide["guide_id"]
            gene   = guide["gene"]
            chr_   = guide["chr"]
            strand = guide["strand"]
            try:
                start = int(guide["start"])
            except (ValueError, KeyError):
                risk_map[gid] = 0.0
                continue

            # SpCas9 cut: 3 bp upstream of PAM
            cut = start + 17 if strand == "+" else start + 3

            if db:
                dist_bp, site_type = _nearest_splice(db, chr_, cut, strand)
            else:
                dist_bp, site_type = 999_999, "unknown"

            risk = _splice_risk(dist_bp, site_type) if db else 0.0
            risk_map[gid] = risk
            rows.append({
                "guide_id":             gid,
                "gene":                 gene,
                "chr":                  chr_,
                "cut_site":             cut,
                "nearest_splice_dist_bp": dist_bp,
                "nearest_splice_type":  site_type,
                "splice_risk":          risk,
            })

    with open(SPLICE_RISK_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "guide_id", "gene", "chr", "cut_site",
            "nearest_splice_dist_bp", "nearest_splice_type", "splice_risk",
        ])
        w.writeheader()
        w.writerows(rows)

    risks = [r["splice_risk"] for r in rows]
    high_risk = sum(1 for r in risks if r >= 0.75)
    mod_risk  = sum(1 for r in risks if 0.40 <= r < 0.75)
    low_risk  = sum(1 for r in risks if r < 0.40)

    print(f"  {len(rows):,} guides scored in {time.time()-t0:.1f}s")
    print(f"  Risk distribution: high(>=0.75)={high_risk:,}  "
          f"moderate(0.40-0.75)={mod_risk:,}  low(<0.40)={low_risk:,}")
    print(f"  Mean splice risk: {np.mean(risks):.3f}  "
          f"Median: {np.median(risks):.3f}")
    print(f"  -> {SPLICE_RISK_CSV.name}")

    return {
        "n_guides":        len(rows),
        "mean_risk":       round(float(np.mean(risks)), 4),
        "median_risk":     round(float(np.median(risks)), 4),
        "high_risk_n":     high_risk,
        "moderate_risk_n": mod_risk,
        "low_risk_n":      low_risk,
    }, risk_map


# ── Rank normalisation helper ─────────────────────────────────────────────────

def _rank_norm(values: np.ndarray) -> np.ndarray:
    """Rank-normalise to [0, 1]. Ties get average rank."""
    n = len(values)
    if n == 0:
        return values
    order = np.argsort(values)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    return (ranks - 1) / max(n - 1, 1)


# ── Module 2: build cell_suitability.csv ─────────────────────────────────────

def build_cell_suitability(risk_map: dict) -> dict:
    """
    One row per (guide, cell_type): composite suitability score.
    """
    print(f"\n{'='*60}")
    print("  Phase 4 / Module 2 — Cell-Type Suitability Scorer")
    print(f"{'='*60}")
    t0 = time.time()

    # Load guide efficacy and gene name from metadata
    efficacy: dict[str, float] = {}
    gene_map: dict[str, str] = {}
    with open(GUIDE_META_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gid = row["guide_id"]
            gene_map[gid] = row.get("gene", "")
            try:
                efficacy[gid] = float(row["efficacy"])
            except (ValueError, KeyError):
                pass

    # Load cell features
    print("  Loading cell_features.csv...")
    cell_rows: list[dict] = []
    with open(CELL_FEAT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cell_rows.append(row)
    print(f"  {len(cell_rows):,} (guide, cell_type) pairs loaded")

    # Group by cell_type for rank-normalisation within each cell type
    from collections import defaultdict
    by_ct: dict[str, list] = defaultdict(list)
    for row in cell_rows:
        by_ct[row["cell_type"]].append(row)

    output_rows = []

    for ct, rows in by_ct.items():
        n = len(rows)
        gids = [r["guide_id"] for r in rows]

        rna_vals   = np.array([float(r["rna_tpm_log1p"])  for r in rows])
        atac_vals  = np.array([float(r["atac_signal"])    for r in rows])
        ge_vals    = np.array([float(r["gene_effect"])    for r in rows])
        eff_vals   = np.array([efficacy.get(g, 0.0)       for g in gids])
        risk_vals  = np.array([risk_map.get(g, 0.0)       for g in gids])

        rna_r  = _rank_norm(rna_vals)
        atac_r = _rank_norm(atac_vals)
        # Gene essentiality: more negative gene_effect = more essential = higher score
        # Clip at 0 so non-essential (positive) genes don't invert the score
        essent_r = _rank_norm(np.clip(-ge_vals, 0, None))
        eff_r    = _rank_norm(eff_vals)
        safety_r = _rank_norm(1.0 - risk_vals)   # higher safety = higher rank

        suitability = (
            W_EFFICACY * eff_r +
            W_RNA      * rna_r +
            W_ATAC     * atac_r +
            W_SPLICE   * safety_r +
            W_ESSENT   * essent_r
        )

        for i, row in enumerate(rows):
            gid = row["guide_id"]
            output_rows.append({
                "guide_id":        gid,
                "gene":            gene_map.get(gid, ""),
                "cell_type":       ct,
                "efficacy":        round(eff_vals[i], 4),
                "rna_tpm_log1p":   round(rna_vals[i], 4),
                "atac_signal":     round(atac_vals[i], 4),
                "gene_effect":     round(ge_vals[i], 4),
                "splice_risk":     round(risk_vals[i], 4),
                "suitability_score": round(float(suitability[i]), 4),
            })

        p90 = float(np.percentile(suitability, 90))
        print(f"  {ct}: {n:,} guides  mean_suitability={float(np.mean(suitability)):.3f}  "
              f"top10%_threshold={p90:.3f}")

    # Sort by (cell_type, suitability desc) for readability
    output_rows.sort(key=lambda r: (r["cell_type"], -r["suitability_score"]))

    with open(CELL_SUIT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "guide_id", "gene", "cell_type", "efficacy",
            "rna_tpm_log1p", "atac_signal", "gene_effect",
            "splice_risk", "suitability_score",
        ])
        w.writeheader()
        w.writerows(output_rows)

    suit_scores = [r["suitability_score"] for r in output_rows]
    print(f"\n  {len(output_rows):,} (guide, cell_type) suitability scores written")
    print(f"  Mean={np.mean(suit_scores):.3f}  "
          f"p10={np.percentile(suit_scores,10):.3f}  "
          f"p90={np.percentile(suit_scores,90):.3f}")
    print(f"  -> {CELL_SUIT_CSV.name}")
    print(f"  Done in {time.time()-t0:.1f}s")

    return {
        "n_rows":        len(output_rows),
        "mean_suit":     round(float(np.mean(suit_scores)), 4),
        "p10_suit":      round(float(np.percentile(suit_scores, 10)), 4),
        "p90_suit":      round(float(np.percentile(suit_scores, 90)), 4),
        "weights": {
            "efficacy":       W_EFFICACY,
            "rna_expression": W_RNA,
            "atac_access":    W_ATAC,
            "splice_safety":  W_SPLICE,
            "gene_essent":    W_ESSENT,
        },
    }


# ── Master runner ─────────────────────────────────────────────────────────────

def run_phase4() -> None:
    print(f"\n{'='*60}")
    print("  OmicsCRISPR Phase 4")
    print(f"{'='*60}")
    t_total = time.time()

    splice_stats, risk_map = build_splice_risk()
    suit_stats = build_cell_suitability(risk_map)

    summary = {
        "splice_risk":        splice_stats,
        "cell_suitability":   suit_stats,
        "elapsed_s":          round(time.time() - t_total, 1),
    }
    with open(PHASE4_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Phase 4 complete in {time.time()-t_total:.1f}s")
    print(f"  splice_risk.csv      : {SPLICE_RISK_CSV.stat().st_size >> 10} KB")
    print(f"  cell_suitability.csv : {CELL_SUIT_CSV.stat().st_size >> 10} KB")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_phase4()
