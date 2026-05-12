"""
PAM site detection and guide RNA extraction.
Supports:
  - SpCas9  (NGG)  — PAM 3' of guide
  - SpCas9  (NAG)  — PAM 3' of guide (low efficiency)
  - SaCas9  (NNGRRT) — PAM 3' of guide
  - Cas12a  (TTTV)   — PAM 5' of guide (upstream)
"""
import re
from typing import List, Tuple
from fastapi import HTTPException
from app.utils.biology_utils import reverse_complement, gc_content

MIN_SEQUENCE_LENGTH = 23  # shortest valid guide (20) + minimal PAM (3)
_VALID_DNA = re.compile(r'^[ACGT]+$')
_RNA_CHARS  = re.compile(r'[U]', re.IGNORECASE)
_AMBIGUOUS  = re.compile(r'[RYSWKMBDHVN]', re.IGNORECASE)


def clean_sequence(raw: str) -> str:
    """
    Strip FASTA headers and whitespace, return uppercase DNA string.
    Raises HTTPException with a human-readable message on validation failure.
    """
    lines = raw.strip().splitlines()
    seq_lines = [ln.strip() for ln in lines if not ln.startswith('>')]
    seq = ''.join(seq_lines).upper().replace(' ', '').replace('\t', '')

    if not seq:
        raise HTTPException(status_code=400, detail="Sequence is empty after stripping FASTA headers.")

    if _RNA_CHARS.search(seq):
        raise HTTPException(
            status_code=400,
            detail="RNA sequence detected (contains 'U'). Please convert U→T and resubmit a DNA sequence.",
        )

    bad = set(re.findall(r'[^ACGTU]', seq, re.IGNORECASE))
    if bad:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid characters in sequence: {', '.join(sorted(bad))}. Only A, C, G, T are accepted.",
        )

    if len(seq) < MIN_SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Sequence too short ({len(seq)} bp). Minimum length is {MIN_SEQUENCE_LENGTH} bp (guide + PAM).",
        )

    return seq

GUIDE_LENGTH = 20

# PAMs that are located 3' (downstream) of the guide — standard Cas9 style
CAS9_PAMS = {"NGG", "NAG", "NNGRRT"}
# PAMs located 5' (upstream) of the guide — Cas12a style
CAS12A_PAMS = {"TTTV"}


def _pam_to_regex(pam: str) -> str:
    """Convert IUPAC PAM notation to a regex character class string."""
    iupac = {
        "N": "[ACGT]", "R": "[AG]", "Y": "[CT]", "S": "[GC]",
        "W": "[AT]",   "K": "[GT]", "M": "[AC]", "B": "[CGT]",
        "D": "[AGT]",  "H": "[ACT]","V": "[ACG]",
        "A": "A", "C": "C", "G": "G", "T": "T",
    }
    return "".join(iupac.get(c, c) for c in pam.upper())


def _find_cas9_sites(sequence: str, pam: str, strand: str) -> List[Tuple[str, str, int, str]]:
    """PAM is 3' of guide: [20bp guide][PAM]"""
    results = []
    seq = sequence.upper()
    pam_re = _pam_to_regex(pam)
    full_pattern = rf"(?=([ACGTN]{{{GUIDE_LENGTH}}}({pam_re})))"

    for match in re.finditer(full_pattern, seq):
        full = match.group(1)
        guide = full[:GUIDE_LENGTH]
        pam_found = full[GUIDE_LENGTH:]
        pos = match.start()
        if guide.count("N") > 2:
            continue
        results.append((guide, pam_found, pos, strand))
    return results


def _find_cas12a_sites(sequence: str, pam: str, strand: str) -> List[Tuple[str, str, int, str]]:
    """PAM is 5' of guide: [PAM][20bp guide]"""
    results = []
    seq = sequence.upper()
    pam_re = _pam_to_regex(pam)
    full_pattern = rf"(?=(({pam_re})[ACGTN]{{{GUIDE_LENGTH}}}))"

    for match in re.finditer(full_pattern, seq):
        full = match.group(1)
        pam_found = match.group(2)
        guide = full[len(pam_found):]
        pos = match.start() + len(pam_found)  # guide starts after PAM
        if guide.count("N") > 2:
            continue
        results.append((guide, pam_found, pos, strand))
    return results


def find_all_grnas(sequence: str, pam: str = "NGG") -> List[dict]:
    """
    Find all valid gRNAs in both strands for the given PAM.
    Automatically handles 3'-PAM (Cas9) vs 5'-PAM (Cas12a) logic.
    Returns list of candidate dicts.
    """
    pam_upper = pam.upper()
    is_cas12a = pam_upper in CAS12A_PAMS

    finder = _find_cas12a_sites if is_cas12a else _find_cas9_sites

    candidates = []
    seq_len = len(sequence)

    # Forward strand
    for guide, pam_seq, pos, strand in finder(sequence, pam, "+"):
        candidates.append({
            "sequence": guide,
            "pam_sequence": pam_seq,
            "position": pos,
            "strand": strand,
            "gc_content": gc_content(guide),
        })

    # Reverse strand
    rc_seq = reverse_complement(sequence)
    for guide, pam_seq, pos, strand in finder(rc_seq, pam, "-"):
        # pos is the guide start in RC space for both Cas9 and Cas12a:
        #   Cas9:   pos = match.start()            (guide precedes PAM in RC)
        #   Cas12a: pos = match.start() + pam_len  (PAM already skipped)
        # Forward guide start = seq_len - pos - GUIDE_LENGTH (no pam_len)
        fwd_pos = seq_len - pos - GUIDE_LENGTH
        candidates.append({
            "sequence": guide,
            "pam_sequence": pam_seq,
            "position": max(0, fwd_pos),
            "strand": strand,
            "gc_content": gc_content(guide),
        })

    return candidates
