"""
Utility functions for DNA/RNA biological operations.

Biological references implemented here:
  - SantaLucia 1998 PNAS 95:1460  — nearest-neighbor Tm parameters
  - Brummelkamp et al. 2002       — TTTT terminates U6/H1 Pol III transcription
  - Hsu et al. 2013 Nature Biotech — seed region (12 bp proximal to PAM) off-target tolerance
  - Jinek et al. 2012 Science      — seed region critical for Cas9 binding specificity
  - Doench et al. 2014/2016        — GC content, positional preferences
"""
import math
import re
from typing import Tuple

COMPLEMENT = str.maketrans("ACGTN", "TGCAN")

# ---------------------------------------------------------------------------
# SantaLucia 1998 DNA-DNA nearest-neighbor parameters
# (ΔH in kcal/mol, ΔS in cal/mol·K)
# Reference: SantaLucia J (1998) PNAS 95:1460-1465, Table 2
# ---------------------------------------------------------------------------
_NN: dict[str, Tuple[float, float]] = {
    "AA": (-7.9,  -22.2), "AT": (-7.2,  -20.4), "TA": (-7.2,  -21.3),
    "CA": (-8.5,  -22.7), "GT": (-8.4,  -22.4), "CT": (-7.8,  -21.0),
    "GA": (-8.2,  -22.2), "CG": (-10.6, -27.2), "GC": (-9.8,  -24.4),
    "GG": (-8.0,  -19.9), "AC": (-7.8,  -21.0), "TC": (-7.9,  -22.2),
    "AG": (-8.2,  -22.2), "TG": (-8.5,  -22.7), "TT": (-7.9,  -22.2),
    "CC": (-8.0,  -19.9),
}
_R  = 1.987   # cal / mol·K
_CT = 250e-9  # strand concentration (250 nM, standard for gRNA design tools)


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return sequence.upper().translate(COMPLEMENT)[::-1]


def gc_content(sequence: str) -> float:
    """GC fraction (0.0–1.0)."""
    seq = sequence.upper()
    gc = seq.count("G") + seq.count("C")
    return gc / len(seq) if len(seq) > 0 else 0.0


def is_valid_dna(sequence: str) -> bool:
    """Return True if the string contains only ACGTN bases."""
    return bool(re.fullmatch(r"[ACGTNacgtn]+", sequence))


def dna_to_rna(sequence: str) -> str:
    """Convert a DNA sequence to RNA (T → U)."""
    return sequence.upper().replace("T", "U")


def has_poly_t(sequence: str, threshold: int = 4) -> bool:
    """
    Detect poly-T runs that terminate RNA Pol III (U6/H1 promoter) transcription.
    Reference: Brummelkamp et al. 2002 Science 296:550.
    """
    return "T" * threshold in sequence.upper()


def nearest_neighbor_tm(sequence: str, oligo_conc_nM: float = 250.0) -> float:
    """
    Melting temperature via the SantaLucia 1998 nearest-neighbor model (DNA-DNA).
    This approximation is used by Primer3, Benchling, and most gRNA design tools.

    Formula:  Tm = ΔH / (ΔS + R·ln(CT/4)) − 273.15
    where ΔH (kcal/mol) and ΔS (cal/mol·K) are summed nearest-neighbor contributions
    plus initiation corrections.

    Args:
        sequence:      DNA sequence string (A/C/G/T)
        oligo_conc_nM: Oligonucleotide concentration in nM (default 250)

    Returns:
        Tm in °C, or 0.0 for sequences shorter than 2 nt.
    """
    seq = sequence.upper()
    n   = len(seq)
    if n < 2:
        return 0.0

    dH = 0.0   # kcal/mol
    dS = 0.0   # cal/mol·K

    # Stacking contributions
    for i in range(n - 1):
        di = seq[i:i + 2]
        if di in _NN:
            h, s = _NN[di]
            dH  += h
            dS  += s

    # Initiation corrections (SantaLucia 1998, Table 2)
    # Terminal A·T: ΔH = +2.3 kcal/mol, ΔS = +4.1 cal/mol·K
    # Terminal G·C: ΔH = +0.1 kcal/mol, ΔS = −2.8 cal/mol·K
    for end in (seq[0], seq[-1]):
        if end in ("A", "T"):
            dH += 2.3;  dS += 4.1
        elif end in ("G", "C"):
            dH += 0.1;  dS -= 2.8

    CT         = oligo_conc_nM * 1e-9
    dH_cal     = dH * 1000.0          # kcal → cal
    denominator = dS + _R * math.log(CT / 4.0)

    if abs(denominator) < 1e-10:
        return 0.0
    return round(dH_cal / denominator - 273.15, 2)


def seed_region_gc(sequence: str) -> float:
    """
    GC content of the PAM-proximal seed region (last 12 bp of guide).
    The seed region is critical for Cas9 binding and on-target specificity.
    Reference: Jinek et al. 2012 Science 337:816; Wu et al. 2014 Nat Biotech 32:670.
    """
    return gc_content(sequence[-12:])


def has_hairpin(sequence: str, min_stem: int = 4) -> bool:
    """
    Detect self-complementarity that can form hairpin structures.
    Hairpins in the spacer reduce CRISPR efficiency by preventing R-loop formation.

    Checks for any 'min_stem'-bp palindromic subsequence anywhere in the guide.
    """
    seq = sequence.upper()
    n   = len(seq)
    for i in range(n - min_stem + 1):
        stem   = seq[i:i + min_stem]
        rc_stem = reverse_complement(stem)
        # rc must appear at a different, non-overlapping position
        j = seq.find(rc_stem)
        if j != -1 and j != i:
            return True
    return False


def count_off_target_risk_bases(sequence: str) -> int:
    """
    Estimate off-target risk from seed region (last 12 bp) AT composition.
    A/T-rich seed regions tolerate mismatches better → more off-target risk.
    Reference: Hsu et al. 2013 Nat Biotech 31:827.
    """
    seed = sequence[-12:].upper()
    return seed.count("A") + seed.count("T")
