"""
Unit tests for biology_utils.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.biology_utils import (
    reverse_complement, gc_content, is_valid_dna,
    nearest_neighbor_tm, has_poly_t, seed_region_gc,
    has_hairpin, count_off_target_risk_bases,
)


# ── reverse_complement ─────────────────────────────────────────────────────

def test_rc_simple():
    assert reverse_complement("ATCG") == "CGAT"


def test_rc_all_a():
    assert reverse_complement("AAAA") == "TTTT"


def test_rc_palindrome():
    # AATT is its own reverse complement
    assert reverse_complement("AATT") == "AATT"


def test_rc_lowercase():
    assert reverse_complement("atcg") == "CGAT"


def test_rc_with_n():
    assert reverse_complement("ATCGN") == "NCGAT"


def test_rc_double_rc_identity():
    seq = "ATCGATCGATCG"
    assert reverse_complement(reverse_complement(seq)) == seq


# ── gc_content ─────────────────────────────────────────────────────────────

def test_gc_content_50pct():
    assert abs(gc_content("ATCG") - 0.5) < 1e-6


def test_gc_content_0pct():
    assert gc_content("AAAA") == 0.0


def test_gc_content_100pct():
    assert gc_content("GCGC") == 1.0


def test_gc_content_empty():
    assert gc_content("") == 0.0


def test_gc_content_lowercase():
    assert abs(gc_content("atcg") - 0.5) < 1e-6


# ── is_valid_dna ───────────────────────────────────────────────────────────

def test_valid_dna_acgt():
    assert is_valid_dna("ATCGATCG")


def test_valid_dna_with_n():
    assert is_valid_dna("ATCGN")


def test_valid_dna_lowercase():
    assert is_valid_dna("atcg")


def test_invalid_dna_with_u():
    assert not is_valid_dna("AUCG")   # RNA base


def test_invalid_dna_with_space():
    assert not is_valid_dna("ATCG ATCG")


# ── nearest_neighbor_tm ────────────────────────────────────────────────────

def test_tm_reasonable_range():
    seq = "ATCGATCGATCGATCGATCG"
    tm  = nearest_neighbor_tm(seq)
    assert 30.0 <= tm <= 80.0, f"Tm={tm} outside expected range for 20-bp guide"


def test_tm_gc_rich_higher_than_at_rich():
    gc_rich = "GCGCGCGCGCGCGCGCGCGC"
    at_rich = "ATATATATATATATATATATAT"[:20]
    assert nearest_neighbor_tm(gc_rich) > nearest_neighbor_tm(at_rich)


def test_tm_short_sequence():
    # Short seq should still return a finite float, not crash
    tm = nearest_neighbor_tm("ATCG")
    assert isinstance(tm, float)


# ── has_poly_t ─────────────────────────────────────────────────────────────

def test_poly_t_found():
    assert has_poly_t("ATCGATTTTTCG", 4)


def test_poly_t_not_found():
    assert not has_poly_t("ATCGATCGATCG", 4)


def test_poly_t_exact_threshold():
    assert has_poly_t("TTTT", 4)
    assert not has_poly_t("TTT", 4)


# ── seed_region_gc ─────────────────────────────────────────────────────────

def test_seed_gc_all_gc():
    # last 12 bases are all GC
    seq = "ATATATGCGCGCGCGCGCGCGC"[:20]
    val = seed_region_gc(seq)
    assert val >= 0.8


def test_seed_gc_all_at():
    seq = "GCGCGCGCATATATATATATAT"[:20]
    val = seed_region_gc(seq)
    assert val <= 0.2


def test_seed_gc_range():
    for seq in ["ATCGATCGATCGATCGATCG", "GCGCGCGCGCGCGCGCGCGC"]:
        val = seed_region_gc(seq)
        assert 0.0 <= val <= 1.0


# ── has_hairpin ────────────────────────────────────────────────────────────

def test_hairpin_detected():
    # GCGC + loop + GCGC reverse complement
    seq = "GCGCATATGCGCATATGCGC"
    assert has_hairpin(seq, min_stem=4)


def test_no_hairpin_poly_a():
    assert not has_hairpin("AAAAAAAAAAAAAAAAAAAA", min_stem=4)


# ── count_off_target_risk_bases ────────────────────────────────────────────

def test_off_target_risk_all_at():
    seq = "ATATATATATATATATATATAT"[:20]
    val = count_off_target_risk_bases(seq)
    assert val > 0


def test_off_target_risk_all_gc():
    seq = "GCGCGCGCGCGCGCGCGCGC"
    val = count_off_target_risk_bases(seq)
    assert val == 0
