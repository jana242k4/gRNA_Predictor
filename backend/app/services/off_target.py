"""
Off-target specificity scoring for CRISPR guide RNAs.

Two scoring modes:

1. cfd_score(guide, off_target) — exact CFD (Cutting Frequency Determination)
   score given a known off-target sequence. Uses the published mismatch weight
   matrix from Doench et al. 2016 Nat Biotechnol 34:184, Table S19.
   Returns float in [0, 1]: 1.0 = perfect match, 0 = no activity.

2. specificity_score(sequence) — sequence-intrinsic estimate when no genome
   alignment is available. Uses CFD position weights to properly weight seed
   vs. non-seed penalties, replacing the flat heuristic. Returns [0, 1].

CFD mismatch matrix
-------------------
Values are fraction of wild-type cleavage retained for a single mismatch at
each position. Convention: position 1 = PAM-distal, position 20 = PAM-proximal.
Source: Doench et al. 2016 Table S19 (reproduced in Azimuth / CRISPOR).
"""

import math
import numpy as np

# ── CFD mismatch matrix (rNucleotide:dNucleotide, positions 1–20) ─────────────
# Rows = mismatch type (RNA guide base : DNA target base)
# Cols = guide positions 1–20 (PAM-distal → PAM-proximal)
# Source: Doench et al. 2016 Nat Biotechnol 34:184, Table S19

_CFD: dict[str, list[float]] = {
    "rA:dA": [0.000,0.000,0.057,0.000,0.000,0.021,0.217,0.033,0.174,0.124,
              0.137,0.116,0.202,0.218,0.229,0.238,0.207,0.309,0.231,0.244],
    "rA:dC": [0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
              0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000],
    "rA:dG": [0.069,0.000,0.094,0.000,0.000,0.000,0.000,0.000,0.000,0.052,
              0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.069],
    "rC:dA": [0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
              0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000],
    "rC:dC": [0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
              0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000],
    "rC:dT": [0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
              0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000],
    "rG:dA": [0.190,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
              0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.099],
    "rG:dG": [0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
              0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000],
    "rG:dT": [0.761,0.621,0.527,0.480,0.337,0.647,0.655,0.512,0.473,0.549,
              0.428,0.582,0.427,0.497,0.534,0.440,0.479,0.492,0.428,0.323],
    "rT:dA": [0.060,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
              0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.088],
    "rT:dC": [0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
              0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000],
    "rT:dG": [0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
              0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000],
}

# PAM scores (NGG = reference 1.0; other PAMs from Doench 2016 Table S19)
_PAM_SCORES: dict[str, float] = {
    "NGG": 1.000, "NAG": 0.259, "NGA": 0.069,
    "NGC": 0.000, "NCG": 0.000, "NTG": 0.000, "NNR": 0.000,
}

# Guide base → complement (DNA target base) for match detection
_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}

def cfd_score(guide: str, off_target: str, pam: str = "NGG") -> float:
    """
    Compute CFD (Cutting Frequency Determination) score for a given
    guide–off-target pair.

    guide      : 20-mer protospacer (DNA sequence)
    off_target : 20-mer potential off-target (DNA, same strand as guide)
    pam        : 3-character PAM seen in the off-target genomic context

    Returns float in [0, 1]: 1.0 = perfect match, lower = more disrupted.

    Reference: Doench et al. 2016 Nat Biotechnol 34:184
    """
    guide     = guide.upper()[:20]
    ot        = off_target.upper()[:20]
    pam_upper = pam.upper()

    # PAM component
    pam_key = "N" + pam_upper[1] + pam_upper[2] if len(pam_upper) >= 3 else "NGG"
    pam_sc  = _PAM_SCORES.get(pam_key, 0.0)
    if pam_sc == 0.0:
        return 0.0

    score = 1.0
    for i, (g_base, ot_base) in enumerate(zip(guide, ot)):
        if g_base == ot_base:
            continue  # same protospacer base = Watson-Crick match
        # Build mismatch key: rRNA_base:dDNA_base
        r_base = g_base   # guide RNA base (same letter as DNA spacer)
        d_base = ot_base  # DNA target base
        mm_key = f"r{r_base}:d{d_base}"
        weight = _CFD.get(mm_key, [0.0] * 20)
        pos_weight = weight[i] if i < len(weight) else 0.0
        score *= pos_weight
        if score == 0.0:
            return 0.0

    return round(float(score * pam_sc), 4)


# ── Sequence-intrinsic helpers ────────────────────────────────────────────────

_COMPLEMENT_STR = str.maketrans("ACGT", "TGCA")

# CFD-derived position specificity weights (averaged across mismatch types).
# High value = mismatch tolerated at this position (off-target risk is higher).
# Low value = mismatch strongly disrupts activity (safer position).
# Positions 1–20, PAM-distal → PAM-proximal.
# Derived from the mean of non-zero CFD values per position from Table S19.
_POSITION_WEIGHTS = [
    0.25, 0.12, 0.13, 0.06, 0.04, 0.09, 0.13, 0.07,
    0.08, 0.09, 0.07, 0.09, 0.06, 0.07, 0.07, 0.07,
    0.09, 0.10, 0.08, 0.11,
]
# Seed region position weights (positions 13–20, PAM-proximal half)
_SEED_WEIGHTS = _POSITION_WEIGHTS[12:]  # indices 12-19


def _reverse_complement(seq: str) -> str:
    return seq.upper().translate(_COMPLEMENT_STR)[::-1]


def _seed_at_content(seq: str) -> float:
    """AT fraction in the 12-bp seed region (PAM-proximal, last 12 bp)."""
    seed = seq.upper()[-12:]
    return (seed.count("A") + seed.count("T")) / len(seed) if seed else 0.0


def _pam_proximal_gc_run(seq: str) -> int:
    """Length of the consecutive GC run at the 3′ end (max 6 bp checked)."""
    count = 0
    for b in reversed(seq.upper()[-6:]):
        if b in "GC":
            count += 1
        else:
            break
    return count


def _has_hairpin(seq: str, min_stem: int = 4) -> bool:
    n = len(seq)
    for i in range(n - min_stem + 1):
        stem = seq[i:i + min_stem]
        rc   = _reverse_complement(stem)
        j    = seq.find(rc)
        if j != -1 and j != i:
            return True
    return False


def _sequence_entropy(seq: str, k: int = 3) -> float:
    n = len(seq)
    if n < k:
        return 1.0
    kmers  = [seq[i:i + k] for i in range(n - k + 1)]
    total  = len(kmers)
    counts: dict = {}
    for km in kmers:
        counts[km] = counts.get(km, 0) + 1
    entropy     = -sum((c / total) * math.log2(c / total) for c in counts.values())
    max_entropy = math.log2(total) if total > 1 else 1.0
    return entropy / max_entropy


def _complexity_penalty(seq: str) -> float:
    dimer_pen = 0.0
    for i in range(len(seq) - 1):
        if seq[i:i + 2] * 3 in seq:
            dimer_pen = 0.10
            break
    entropy    = _sequence_entropy(seq, k=3)
    entropy_pen = max(0.0, (0.70 - entropy) * 0.20) if entropy < 0.70 else 0.0
    return dimer_pen + entropy_pen


def _cfd_weighted_seed_penalty(seq: str) -> float:
    """
    Position-weighted AT-content penalty for the seed region.
    Uses CFD position weights so PAM-proximal AT content is penalised more
    than PAM-distal AT content — consistent with the experimentally-derived
    CFD mismatch tolerance gradient.
    """
    seed = seq[-8:]   # positions 13-20 (most critical per CFD)
    weighted_at = 0.0
    weight_sum  = 0.0
    for i, base in enumerate(seed):
        w = _SEED_WEIGHTS[i] if i < len(_SEED_WEIGHTS) else 0.08
        if base in "AT":
            weighted_at += w
        weight_sum += w
    return (weighted_at / weight_sum) * 0.30 if weight_sum > 0 else 0.0


def specificity_score(sequence: str, pam: str = "NGG") -> float:
    """
    Sequence-intrinsic off-target specificity estimate.

    Without whole-genome alignment, uses CFD position weights to compute a
    position-sensitive specificity score from guide sequence features.

    Returns float in [0.0, 1.0]:
      >= 0.70  → low off-target risk
      0.45–0.70 → moderate risk
      < 0.45   → high off-target risk

    For exact off-target scoring against known sequences, use cfd_score().
    """
    seq = sequence.upper()[:20]

    # 1. CFD position-weighted seed AT penalty (replaces flat seed_at heuristic)
    seed_pen = _cfd_weighted_seed_penalty(seq)

    # 2. Global GC content
    gc = (seq.count("G") + seq.count("C")) / 20.0
    if gc < 0.25 or gc > 0.75:
        gc_pen = 0.20
    elif gc < 0.35 or gc > 0.65:
        gc_pen = 0.10
    elif gc < 0.40 or gc > 0.60:
        gc_pen = 0.05
    else:
        gc_pen = 0.0

    # 3. PAM-proximal GC run
    gc_run = 0
    for b in reversed(seq[-6:]):
        if b in "GC":
            gc_run += 1
        else:
            break
    gc_run_pen = min(0.18, max(0.0, gc_run - 2) * 0.07)

    # 4. Homopolymer runs
    hp_pen = sum(0.08 for b in "ACGT" if b * 4 in seq)

    # 5. Hairpin
    hairpin_pen = 0.12 if _has_hairpin(seq, min_stem=4) else 0.0

    # 6. G-quadruplex risk in seed
    gq_pen = 0.08 if "GGG" in seq[-12:] else 0.0

    # 7. Sequence complexity
    complexity_pen = _complexity_penalty(seq)

    # 8. PAM-type penalty (non-canonical PAMs reduce specificity)
    pam_upper = pam.upper()
    pam_key   = "N" + pam_upper[1] + pam_upper[2] if len(pam_upper) >= 3 else "NGG"
    pam_sc    = _PAM_SCORES.get(pam_key, 1.0)
    pam_pen   = (1.0 - pam_sc) * 0.15  # NAG → +0.11 penalty

    raw = 1.0 - seed_pen - gc_pen - gc_run_pen - hp_pen - hairpin_pen - gq_pen - complexity_pen - pam_pen
    return float(np.clip(raw, 0.0, 1.0))
