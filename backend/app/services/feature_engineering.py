"""
Feature engineering for ML-based gRNA efficiency prediction.

Feature vector (450 dimensions total):
  [0:80]    Positional one-hot — 20 positions × 4 bases (guide sequence)
  [80]      GC content fraction (guide)
  [81]      Nearest-neighbor Tm (SantaLucia 1998, full guide), normalised 0-1
  [82:98]   Dinucleotide frequencies — all 16 pairs (guide)
  [98]      Seed region GC (last 12 bp, PAM-proximal)
  [99]      Poly-T flag — 1 if TTTT present (RNA Pol III termination)
  [100:404] Position-specific dinucleotide one-hot — 19 positions × 16 (guide)
  [404:420] 4-bp upstream context one-hot  — positions -4,-3,-2,-1
  [420:444] 6-bp downstream context one-hot — PAM+3 bp after (positions +1…+6)
  [444]     GC clamp — GC fraction in last 4 bp (3' end of guide, PAM-proximal)
  [445]     RNA hairpin proxy — longest internal complement stem / 10 (0-1)
  [446]     Microhomology length — max homology at SpCas9 cut site (0-1, from 30-mer)
  [447]     Tm of PAM-distal half (guide positions 0:8), normalised 0-1
  [448]     Tm of seed region (guide positions 12:20), normalised 0-1
  [449]     Tm of full 30-mer context, normalised 0-1 (0 if 30-mer unavailable)

When a 30-mer is not available (Doench 2014 guides), dims [404:420], [420:444],
[446], and [449] are zero-padded.

Biological references:
  - One-hot positional encoding:    Doench et al. 2016 Nat Biotechnol 34:184
  - Nearest-neighbor Tm:            SantaLucia 1998 PNAS 95:1460
  - Dinucleotide contributions:     Doench 2016 Supplementary Table S9
  - Seed region GC:                 Hsu et al. 2013 Nat Biotech 31:827
  - Poly-T flag:                    Brummelkamp et al. 2002 Science 296:550
  - Position-specific dinucs:       Doench 2016 Azimuth feature set
  - Flanking context:               Doench 2016 (30-mer); Kim et al. 2019
  - GC clamp (3' end):              Doench et al. 2016 (3'-end GC composition)
  - RNA hairpin proxy:              Zuker 2003; Lorenz et al. 2011 (ViennaRNA)
  - Microhomology:                  Bae et al. 2014 Genome Research 24:132
  - Segmented Tm windows:           Doench 2016 Azimuth (4 Tm sub-windows)
"""
import numpy as np
from typing import List, Optional
from app.utils.biology_utils import nearest_neighbor_tm, seed_region_gc, has_poly_t

BASES     = ["A", "C", "G", "T"]
DINUCS    = [a + b for a in BASES for b in BASES]   # 16 dinucleotides
GUIDE_LEN = 20
N_FEATURES = 450   # 80 + 1 + 1 + 16 + 1 + 1 + 304 + 16 + 24 + 1 + 1 + 1 + 1 + 1 + 1

_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}

_TM_MIN = 40.0
_TM_MAX = 80.0
_BASE_IDX = {b: i for i, b in enumerate(BASES)}
_DINUC_IDX = {d: i for i, d in enumerate(DINUCS)}


def _positional_onehot(sequence: str) -> np.ndarray:
    """80-dim one-hot: position i × base b → index i*4 + BASES.index(b)."""
    vec = np.zeros(GUIDE_LEN * 4, dtype=np.float32)
    for i, base in enumerate(sequence.upper()):
        idx = _BASE_IDX.get(base)
        if idx is not None:
            vec[i * 4 + idx] = 1.0
    return vec


def _gc_content(sequence: str) -> float:
    seq = sequence.upper()
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else 0.0


def _tm_normalised(sequence: str) -> float:
    tm = nearest_neighbor_tm(sequence)
    return float(np.clip((tm - _TM_MIN) / (_TM_MAX - _TM_MIN), 0.0, 1.0))


def _dinucleotide_freq(sequence: str) -> np.ndarray:
    """16-dim normalised dinucleotide frequency vector."""
    seq = sequence.upper()
    vec = np.zeros(16, dtype=np.float32)
    n   = len(seq) - 1
    if n <= 0:
        return vec
    for i in range(n):
        idx = _DINUC_IDX.get(seq[i:i + 2])
        if idx is not None:
            vec[idx] += 1
    vec /= n
    return vec


def _positional_dinucleotide_onehot(sequence: str) -> np.ndarray:
    """304-dim position-specific dinucleotide one-hot (Doench 2016 / Azimuth)."""
    seq = sequence.upper()
    vec = np.zeros(19 * 16, dtype=np.float32)
    for i in range(min(19, len(seq) - 1)):
        idx = _DINUC_IDX.get(seq[i:i + 2])
        if idx is not None:
            vec[i * 16 + idx] = 1.0
    return vec


def _context_onehot(context: str, length: int) -> np.ndarray:
    """One-hot encode an upstream or downstream context window (4×length dims)."""
    vec = np.zeros(length * 4, dtype=np.float32)
    ctx = context.upper()
    for i, base in enumerate(ctx[:length]):
        idx = _BASE_IDX.get(base)
        if idx is not None:
            vec[i * 4 + idx] = 1.0
    return vec


def _tm_segment(segment: str) -> float:
    """Normalised Tm for an arbitrary sub-sequence (SantaLucia 1998).

    Used for the three segmented Tm windows inspired by Azimuth's feature set:
      - PAM-distal  (guide positions  0:8)
      - Seed region (guide positions 12:20)
      - Full 30-mer context (when available)
    Returns 0.0 for segments shorter than 4 bases.
    """
    if len(segment) < 4:
        return 0.0
    tm = nearest_neighbor_tm(segment)
    return float(np.clip((tm - _TM_MIN) / (_TM_MAX - _TM_MIN), 0.0, 1.0))


def _gc_clamp(sequence: str) -> float:
    """GC fraction in the last 4 positions (3' end, PAM-proximal).

    Guides with 2-3 GC in the last 4 bp tend to have higher cleavage efficiency
    (Doench et al. 2016 positional analysis).  Returns 0-1.
    """
    tail = sequence.upper()[-4:]
    return (tail.count("G") + tail.count("C")) / 4.0


def _hairpin_proxy(sequence: str) -> float:
    """RNA secondary structure proxy: longest internal complementary stem / 10.

    Approximates the tendency for the guide RNA to form internal hairpins that
    compete with target binding.  Uses reverse-complement matching with a minimum
    loop of 4 unpaired bases.  Normalised to [0, 1] (capped at stem ≥ 10 bp).
    """
    seq = sequence.upper()
    n   = len(seq)
    max_stem = 0
    for i in range(n - 8):          # stem start (5' arm)
        for stem_len in range(2, (n - i) // 2 + 1):
            arm5 = seq[i: i + stem_len]
            # loop must be ≥ 4 nt; 3' arm starts at i + stem_len + 4
            arm3_start = i + stem_len + 4
            arm3_end   = arm3_start + stem_len
            if arm3_end > n:
                break
            arm3 = seq[arm3_start: arm3_end]
            # arm3 must be the reverse complement of arm5
            rc = "".join(_COMPLEMENT.get(b, "N") for b in reversed(arm3))
            if arm5 == rc:
                max_stem = max(max_stem, stem_len)
    return min(1.0, max_stem / 10.0)


def _microhomology(thirty_mer: str) -> float:
    """Microhomology length at the SpCas9 cut site, normalised to [0, 1].

    SpCas9 cuts between positions 17 and 18 of the 20-bp guide (3 bp upstream
    of the NGG PAM).  In the 30-mer context (4-bp upstream + 20-bp guide +
    6-bp downstream), the cut is at index 4+17 = 21.

    Microhomology: identical bases directly flanking the cut (Bae et al. 2014).
    Up to 6 bp checked on each side.  Score = longest matching prefix / 6.
    """
    if not thirty_mer or len(thirty_mer) < 27:
        return 0.0
    tm = thirty_mer.upper()
    cut = 21  # position after cut in 30-mer
    left  = tm[max(0, cut - 6): cut]       # up to 6 bp left of cut
    right = tm[cut: min(len(tm), cut + 6)] # up to 6 bp right of cut
    # Count matching bases from the cut outward
    max_mh = 0
    for length in range(1, min(len(left), len(right)) + 1):
        if left[-length:] == right[:length]:
            max_mh = length
    return min(1.0, max_mh / 6.0)


def extract_features(sequence: str,
                     thirty_mer: Optional[str] = None) -> np.ndarray:
    """
    Extract 450-dim feature vector from a 20-bp guide and optional 30-mer context.

    Args:
        sequence:   20-bp guide sequence (5'→3')
        thirty_mer: Full 30-mer (4bp upstream + 20bp guide + 6bp downstream).
                    When provided, upstream/downstream features are populated.
                    When None or empty, those 40 dims are zero-padded.

    Returns:
        np.ndarray of shape (444,)
    """
    seq = sequence.upper()[:GUIDE_LEN].ljust(GUIDE_LEN, "N")

    onehot    = _positional_onehot(seq)                              # 80
    gc        = np.array([_gc_content(seq)],  dtype=np.float32)     # 1
    tm        = np.array([_tm_normalised(seq)], dtype=np.float32)   # 1
    dinuc     = _dinucleotide_freq(seq)                              # 16
    seed_gc   = np.array([seed_region_gc(seq)], dtype=np.float32)   # 1
    poly_t    = np.array([1.0 if has_poly_t(seq, 4) else 0.0],
                         dtype=np.float32)                           # 1
    pos_dinuc = _positional_dinucleotide_onehot(seq)                 # 304

    # Flanking context — zero if not available
    if thirty_mer and len(thirty_mer) >= 30:
        tm30 = thirty_mer.upper()
        upstream   = _context_onehot(tm30[:4],  length=4)           # 16
        downstream = _context_onehot(tm30[24:], length=6)           # 24
    else:
        upstream   = np.zeros(16, dtype=np.float32)                 # 16
        downstream = np.zeros(24, dtype=np.float32)                 # 24

    # Advanced biological features
    gc_clamp  = np.array([_gc_clamp(seq)],                  dtype=np.float32)  # 1
    hairpin   = np.array([_hairpin_proxy(seq)],              dtype=np.float32)  # 1
    mh_score  = np.array([_microhomology(thirty_mer or "")], dtype=np.float32)  # 1

    # Segmented Tm windows (extending Doench 2016 Table S1 segmented-Tm approach):
    #   PAM-distal window : guide positions  0- 7 (8 bp, 5' end)
    #   PAM-proximal 8 bp : guide positions 12-19 (8 bp, 3' end / partial seed)
    #     NOTE: this is a windowed feature covering the distal half of the
    #     canonical 12-bp seed region (positions 9-20).  It is deliberately
    #     narrower than the full seed to capture Tm variation near the PAM.
    #     Off-target specificity (off_target.py) uses the full 12-bp seed.
    #   Full 30-mer context: upstream(4) + guide(20) + PAM+downstream(6)
    tm_pam_distal        = np.array([_tm_segment(seq[:8])],         dtype=np.float32)  # 1
    tm_pam_proximal_8bp  = np.array([_tm_segment(seq[12:])],        dtype=np.float32)  # 1
    if thirty_mer and len(thirty_mer) >= 30:
        tm_ctx = np.array([_tm_segment(thirty_mer.upper())], dtype=np.float32)  # 1
    else:
        tm_ctx = np.zeros(1, dtype=np.float32)                                  # 1

    return np.concatenate(
        [onehot, gc, tm, dinuc, seed_gc, poly_t, pos_dinuc,
         upstream, downstream, gc_clamp, hairpin, mh_score,
         tm_pam_distal, tm_pam_proximal_8bp, tm_ctx]
    )   # 450


def extract_features_batch(sequences: List[str],
                            thirty_mers: Optional[List[str]] = None
                            ) -> np.ndarray:
    """
    Extract features for a list of sequences.

    Args:
        sequences:   List of 20-bp guide sequences
        thirty_mers: Optional parallel list of 30-mer context strings.
                     Pass None to use guide-only features for all.

    Returns:
        np.ndarray of shape (N, 450)
    """
    if thirty_mers is None:
        thirty_mers = [""] * len(sequences)
    return np.vstack([
        extract_features(s, t) for s, t in zip(sequences, thirty_mers)
    ])
