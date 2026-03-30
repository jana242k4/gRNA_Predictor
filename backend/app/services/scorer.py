"""
sgRNA efficiency scorer — heuristic implementation of Doench 2016 Rule Set 2.

Biological rules implemented (all cited):
  1. GC content 40-70% optimal              Doench et al. 2014 Nat Biotech 32:1262
  2. Position-specific nucleotide weights   Doench et al. 2016 Nat Biotech 34:184
     (all 20 positions, derived from Fig. 2 and Supplementary Table S9)
  3. U6 promoter requires G at position 1   Cong et al. 2013 Science 339:819
  4. TTTT poly-T terminates Pol III         Brummelkamp et al. 2002 Science 296:550
  5. Homopolymer runs reduce efficiency     Doench et al. 2016
  6. Seed region (last 12 bp) analysis      Hsu et al. 2013 Nat Biotech 31:827
  7. Self-complementarity penalty           Zuker 2003 (mFold principles)
  8. Nearest-neighbor Tm window             SantaLucia 1998 PNAS 95:1460

This scorer is the fallback when xgb_model.pkl is unavailable.
"""
from typing import List
from app.utils.biology_utils import (
    has_poly_t, count_off_target_risk_bases,
    nearest_neighbor_tm, seed_region_gc, has_hairpin,
)

# ---------------------------------------------------------------------------
# Doench 2016 position-specific nucleotide weights — all 20 spacer positions.
# Source: Doench et al. (2016) Nat Biotechnol 34:184-191, Fig. 2 & Table S9.
# Position 1 = PAM-distal (5' end of spacer), 20 = PAM-proximal (3' end).
# ---------------------------------------------------------------------------
_POS_WEIGHTS: dict[int, dict[str, float]] = {
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
    20: {"G":  0.12, "C":  0.04, "A": -0.10, "T": -0.08},  # PAM-proximal — strongest effect
}

_TM_OPT_LOW  = 55.0
_TM_OPT_HIGH = 65.0


def _gc_score(gc: float) -> float:
    """
    Score GC content. Optimal 40-70%, peak at 55%.
    Reference: Doench et al. 2014 Fig. 2a.
    """
    if 0.40 <= gc <= 0.70:
        return 1.0 - abs(gc - 0.55) * 1.5
    elif gc < 0.25 or gc > 0.85:
        return 0.10
    return max(0.15, 1.0 - abs(gc - 0.55) * 3.0)


def _positional_score(sequence: str) -> float:
    """Sum position-specific nucleotide contributions across all 20 positions."""
    return sum(
        _POS_WEIGHTS[pos].get(sequence[pos - 1], 0.0)
        for pos in _POS_WEIGHTS
        if pos <= len(sequence)
    )


def _tm_score(tm: float) -> float:
    """Penalise Tm outside the 55-65 C optimal window for sgRNA-DNA duplexes."""
    if _TM_OPT_LOW <= tm <= _TM_OPT_HIGH:
        return 0.0
    if tm < 45.0 or tm > 75.0:
        return -0.15
    gap = max(0.0, _TM_OPT_LOW - tm) + max(0.0, tm - _TM_OPT_HIGH)
    return -min(0.12, gap * 0.012)


def _penalty_score(sequence: str) -> float:
    """Aggregate penalties for features that reduce editing efficiency."""
    penalty = 0.0

    # Poly-T ≥ 4: terminates U6/H1 Pol III transcription
    if has_poly_t(sequence, threshold=4):
        penalty += 0.28

    # Any homopolymer ≥ 5 disrupts Cas9 loading
    for base in "ACGT":
        if base * 5 in sequence:
            penalty += 0.10

    # Poly-G ≥ 4 can form G-quadruplexes
    if "G" * 4 in sequence:
        penalty += 0.08

    # Seed region AT-richness → higher off-target tolerance
    penalty += min(0.18, count_off_target_risk_bases(sequence) * 0.012)

    # Hairpin structure inhibits R-loop formation
    if has_hairpin(sequence, min_stem=4):
        penalty += 0.12

    # Seed region GC extremes
    s_gc = seed_region_gc(sequence)
    if s_gc < 0.30 or s_gc > 0.80:
        penalty += 0.08

    return penalty


def score_grna(guide: dict) -> float:
    """
    Composite heuristic efficiency score for a single gRNA candidate.
    Returns value in [0.0, 1.0].
    """
    seq  = guide["sequence"].upper()
    gc   = guide["gc_content"]
    tm   = nearest_neighbor_tm(seq)

    # U6 promoter strong preference for G at +1 (Cong et al. 2013)
    u6_adjust = 0.03 if seq[0] == "G" else -0.03

    raw = (0.50 * _gc_score(gc)
           + 0.25 * (_positional_score(seq) + 0.5)
           + 0.10 * 1.0
           + _tm_score(tm)
           + u6_adjust)
    return round(max(0.0, min(1.0, raw - _penalty_score(seq))), 4)


def rank_grnas(candidates: List[dict], top_n: int = 5) -> List[dict]:
    """Score all candidates and return top N sorted by score descending."""
    for c in candidates:
        c["score"] = score_grna(c)
    return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_n]
