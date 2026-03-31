"""
Heuristic off-target specificity scoring for CRISPR guide RNAs.

Returns a specificity score in [0.0, 1.0]:
  1.0 = highly specific (low off-target risk)
  0.0 = poor specificity (high off-target risk)

Without a whole-genome alignment (no BLAST in this pipeline), we compute a
sequence-intrinsic specificity score from features that are strongly predictive
of Cas9 off-target activity, as characterised in the literature.

Biological basis
----------------
Seed region (positions 9–20, PAM-proximal)
  Mismatches in the last ~12 bp proximal to the PAM are poorly tolerated by
  Cas9, but a high AT content in this region increases tolerance to mismatches
  at other positions, raising off-target cleavage probability.
  Reference: Hsu et al. 2013 Nat Biotechnol 31:827

GC content
  Guides with <30% or >70% GC show elevated off-target rates.  The optimal
  40–60% window minimises non-specific binding while maintaining on-target
  efficacy.
  Reference: Doench et al. 2016 Nat Biotechnol 34:184; Kuscu et al. 2014

3′-end GC runs
  Three or more consecutive G/C bases immediately adjacent to the PAM (last
  4 bp of guide) stabilise the DNA:RNA hybrid and increase off-target cleavage
  at partially complementary sites.
  Reference: Wu et al. 2014 Nat Biotechnol 32:479; Pattanayak et al. 2013

Self-complementarity / hairpin
  Guides that form secondary structures can partially unfold and anneal to
  off-target loci, particularly at elevated temperatures.
  Reference: Thyme et al. 2016 Nucleic Acids Res

Seed-region G-runs
  G-rich stretches in the seed form G-quadruplexes and have been associated
  with increased off-target binding in cell-free assays.
  Reference: Tycko et al. 2016 Mol Cell

Sequence complexity / entropy
  Low-complexity guides (repetitive dinucleotides, low k-mer entropy) are
  significantly more likely to match multiple genomic loci with 1–2 mismatches
  because repeat motifs occur at elevated frequency across the genome.
  Dinucleotide-repeat guides (e.g. ATATAT...) show 3–5× more off-target hits
  in ChIP-seq–based off-target assays (Kuscu et al. 2014 Nat Biotechnol 32:677;
  Tsai et al. 2015 Nat Biotechnol 33:187).
"""
import math
import numpy as np

COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _reverse_complement(seq: str) -> str:
    return seq.upper().translate(COMPLEMENT)[::-1]


def _has_hairpin(seq: str, min_stem: int = 4) -> bool:
    """Detect palindromic sub-sequence that can form a hairpin."""
    n = len(seq)
    for i in range(n - min_stem + 1):
        stem = seq[i:i + min_stem]
        rc   = _reverse_complement(stem)
        j    = seq.find(rc)
        if j != -1 and j != i:
            return True
    return False


def _seed_at_content(seq: str) -> float:
    """AT fraction of PAM-proximal seed region (last 12 bp)."""
    seed = seq[-12:]
    return (seed.count("A") + seed.count("T")) / 12.0


def _pam_proximal_gc_run(seq: str) -> int:
    """Length of uninterrupted G/C run at the 3′ end (PAM-adjacent)."""
    run = 0
    for b in reversed(seq[-6:]):
        if b in "GC":
            run += 1
        else:
            break
    return run


def _sequence_entropy(seq: str, k: int = 3) -> float:
    """
    Normalised Shannon entropy of k-mer distribution over the guide.
    Range [0, 1]: 0 = maximally repetitive, 1 = maximally diverse.
    Low entropy indicates a repetitive sequence that is likely to match
    multiple genomic loci with 1–2 mismatches.
    """
    n = len(seq)
    if n < k:
        return 1.0
    kmers = [seq[i:i + k] for i in range(n - k + 1)]
    total = len(kmers)
    counts: dict = {}
    for km in kmers:
        counts[km] = counts.get(km, 0) + 1
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    max_entropy = math.log2(total) if total > 1 else 1.0
    return entropy / max_entropy


def _complexity_penalty(seq: str) -> float:
    """
    Penalty for low-complexity / repetitive sequences.
    Two components:
      - Dinucleotide repeat ≥ 6 bp (e.g. ATATAT, GCGCGC): +0.10
      - Normalised 3-mer entropy below 0.70 threshold: up to +0.12
    """
    dimer_pen = 0.0
    for i in range(len(seq) - 1):
        dimer = seq[i:i + 2]
        if dimer * 3 in seq:  # dimer repeated 3+ times consecutively
            dimer_pen = 0.10
            break

    entropy    = _sequence_entropy(seq, k=3)
    entropy_pen = max(0.0, (0.70 - entropy) * 0.20) if entropy < 0.70 else 0.0

    return dimer_pen + entropy_pen


def specificity_score(sequence: str) -> float:
    """
    Compute heuristic off-target specificity score for a 20-bp guide.

    Returns float in [0.0, 1.0]:
      ≥ 0.70  → low off-target risk  (green)
      0.45–0.70 → moderate risk       (yellow)
      < 0.45  → high off-target risk  (red)
    """
    seq = sequence.upper()[:20]

    # ── 1. Seed region AT content ─────────────────────────────────────────────
    seed_at     = _seed_at_content(seq)
    seed_pen    = seed_at * 0.28          # 0 (all GC) → 0.28 (all AT)

    # ── 2. Global GC content ──────────────────────────────────────────────────
    gc          = (seq.count("G") + seq.count("C")) / 20.0
    if gc < 0.25 or gc > 0.75:
        gc_pen  = 0.20
    elif gc < 0.35 or gc > 0.65:
        gc_pen  = 0.10
    elif gc < 0.40 or gc > 0.60:
        gc_pen  = 0.05
    else:
        gc_pen  = 0.0

    # ── 3. PAM-proximal GC run ────────────────────────────────────────────────
    gc_run      = _pam_proximal_gc_run(seq)
    gc_run_pen  = min(0.20, max(0.0, gc_run - 2) * 0.08)

    # ── 4. Homopolymer runs ───────────────────────────────────────────────────
    hp_pen = 0.0
    for b in "ACGT":
        if b * 4 in seq:
            hp_pen += 0.08

    # ── 5. Hairpin self-complementarity ───────────────────────────────────────
    hairpin_pen = 0.12 if _has_hairpin(seq, min_stem=4) else 0.0

    # ── 6. G-quadruplex risk (seed G-run ≥ 3) ────────────────────────────────
    seed       = seq[-12:]
    gq_pen     = 0.08 if "GGG" in seed else 0.0

    # ── 7. Sequence complexity (low-entropy / repetitive guides) ─────────────
    complexity_pen = _complexity_penalty(seq)

    raw = 1.0 - seed_pen - gc_pen - gc_run_pen - hp_pen - hairpin_pen - gq_pen - complexity_pen
    return float(np.clip(raw, 0.0, 1.0))
