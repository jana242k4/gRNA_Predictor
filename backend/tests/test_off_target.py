"""
Unit tests for off_target.py specificity scoring.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.off_target import (
    specificity_score,
    _seed_at_content, _pam_proximal_gc_run,
    _has_hairpin, _sequence_entropy, _complexity_penalty,
)


# ── Score bounds ───────────────────────────────────────────────────────────

def test_score_in_range_balanced():
    assert 0.0 <= specificity_score("ATCGATCGATCGATCGATCG") <= 1.0


def test_score_in_range_all_gc():
    assert 0.0 <= specificity_score("GCGCGCGCGCGCGCGCGCGC") <= 1.0


def test_score_in_range_at_rich():
    assert 0.0 <= specificity_score("ATATATATATATATATATATAT"[:20]) <= 1.0


# ── AT-rich seed → lower score ─────────────────────────────────────────────

def test_at_rich_seed_lower_than_gc_rich():
    # Last 12 bp all AT → high seed AT penalty
    at_seed = "GCGCGCGCATATATATATATAT"[:20]
    gc_seed = "ATATATATGCGCGCGCGCGCGC"[:20]
    assert specificity_score(at_seed) < specificity_score(gc_seed)


# ── GC run at 3′ end → penalty ────────────────────────────────────────────

def test_gc_run_penalty():
    # 4 consecutive GCs at 3′ end → gc_run_pen triggered
    gc_run = "ATATATATATATATATGCGC"   # 4 GC at end → run ≥ 3 → penalty
    no_run = "ATATATATATATATATGCAT"   # run = 1 → no penalty
    assert specificity_score(gc_run) < specificity_score(no_run)


def test_gc_run_length():
    assert _pam_proximal_gc_run("ATATATATATATATATATGCGC"[:20]) >= 2
    assert _pam_proximal_gc_run("GCGCGCGCGCGCGCGCGCGC") == 6   # last 6 all GC


# ── Homopolymer run → penalty ─────────────────────────────────────────────

def test_homopolymer_penalty():
    poly_a = "ATCGATCGATCGATCGAAAA"   # poly-A(4) → penalty
    clean  = "ATCGATCGATCGATCGATCG"
    assert specificity_score(poly_a) < specificity_score(clean)


# ── Hairpin → penalty ─────────────────────────────────────────────────────

def test_hairpin_penalty():
    hairpin = "GCGCATATGCGCATATGCGC"   # palindromic → _has_hairpin returns True
    assert _has_hairpin(hairpin, min_stem=4)
    # A plain poly-A has no hairpin
    assert not _has_hairpin("AAAAAAAAAAAAAAAAAAAA", min_stem=4)


# ── G-quadruplex → penalty ────────────────────────────────────────────────

def test_gquad_penalty():
    ggg_seed = "ATCGATCGATCGGGGCGATCG"[:20]   # GGG in seed
    no_ggg   = "ATCGATCGATCGATCGATCG"
    assert specificity_score(ggg_seed) < specificity_score(no_ggg)


# ── Sequence complexity / entropy ─────────────────────────────────────────

def test_entropy_low_for_repeat():
    # ATATATATAT... is highly repetitive → low entropy
    repeat_seq = "ATATATATATATATATATATAT"[:20]
    entropy = _sequence_entropy(repeat_seq, k=3)
    assert entropy < 0.80, f"Expected low entropy for repeat, got {entropy:.3f}"


def test_entropy_high_for_diverse():
    # Diverse sequence with many unique k-mers should have higher entropy than a repeat
    repeat  = "ATATATATATATATATATATAT"[:20]
    diverse = "GCTAGCTAGCTAGCTAGCTA"[:20]
    assert _sequence_entropy(diverse, k=2) > _sequence_entropy(repeat, k=2), \
        "Diverse sequence should have higher k-mer entropy than a dimer repeat"


def test_dimer_repeat_penalty():
    # ATATAT repeated → triggers dimer penalty
    dimer_seq = "ATATATATATATATATATATAT"[:20]
    pen = _complexity_penalty(dimer_seq)
    assert pen > 0.0, f"Expected non-zero complexity penalty for dimer repeat, got {pen}"


def test_no_complexity_penalty_for_diverse():
    # A balanced sequence should have near-zero or small entropy penalty
    pen = _complexity_penalty("ATCGATCGATCGATCGATCG")
    # dimer: "AT" repeats 5 times but not "ATATAT" as substring — check
    assert pen <= 0.10, f"Expected low complexity penalty for diverse seq, got {pen}"


def test_complexity_lowers_score():
    repeat_seq = "ATATATATATATATATATATAT"[:20]
    diverse    = "ATCGATCGATCGATCGATCG"
    assert specificity_score(repeat_seq) < specificity_score(diverse)


# ── Seed AT content ────────────────────────────────────────────────────────

def test_seed_at_content_all_at():
    seq = "GCGCGCGCATATATATATATAT"[:20]  # last 12: ATATATATAT + 2 → high AT
    val = _seed_at_content(seq)
    assert val > 0.5


def test_seed_at_content_all_gc():
    seq = "ATATATATGCGCGCGCGCGCGC"[:20]  # last 12: all GC
    val = _seed_at_content(seq)
    assert val == 0.0
