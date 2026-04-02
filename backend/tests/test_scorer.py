"""
Unit tests for scorer.py heuristic efficiency scoring.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.scorer import score_grna, rank_grnas


def _make_candidate(seq: str) -> dict:
    gc = (seq.count("G") + seq.count("C")) / len(seq)
    return {"sequence": seq, "gc_content": gc}


# ── Score bounds ───────────────────────────────────────────────────────────

def test_score_in_range():
    c = _make_candidate("ATCGATCGATCGATCGATCG")
    assert 0.0 <= score_grna(c) <= 1.0


def test_score_high_gc_in_range():
    c = _make_candidate("GCGCGCGCGCGCGCGCGCGC")
    assert 0.0 <= score_grna(c) <= 1.0


def test_score_at_rich_in_range():
    c = _make_candidate("ATATATATATATATATATATAT"[:20])
    assert 0.0 <= score_grna(c) <= 1.0


# ── Poly-T penalty ────────────────────────────────────────────────────────

def test_poly_t_lowers_score():
    poly_t = _make_candidate("ATCGATCGATCGATTTTTCG")   # TTTT present
    clean  = _make_candidate("ATCGATCGATCGATCGATCG")
    assert score_grna(poly_t) < score_grna(clean)


# ── Optimal GC scores higher than extreme GC ─────────────────────────────

def test_optimal_gc_beats_low_gc():
    optimal = _make_candidate("ATCGATCGATCGATCGATCG")   # 50% GC
    low_gc  = _make_candidate("ATATATATATATATATATATAT"[:20])  # 0% GC
    assert score_grna(optimal) > score_grna(low_gc)


def test_optimal_gc_beats_high_gc():
    optimal  = _make_candidate("ATCGATCGATCGATCGATCG")   # 50% GC
    full_gc  = _make_candidate("GCGCGCGCGCGCGCGCGCGC")  # 100% GC
    # 100% GC is outside optimal range → should score lower
    assert score_grna(optimal) > score_grna(full_gc)


# ── U6 G-start preference ─────────────────────────────────────────────────

def test_g_start_preferred():
    g_start = _make_candidate("GATCGATCGATCGATCGATC")
    a_start = _make_candidate("AATCGATCGATCGATCGATC")
    # Same sequence except first base — G start should score slightly higher
    assert score_grna(g_start) > score_grna(a_start)


# ── Ranking ───────────────────────────────────────────────────────────────

def test_rank_grnas_returns_top_n():
    candidates = [_make_candidate(s) for s in [
        "ATCGATCGATCGATCGATCG",
        "GCGCGCGCGCGCGCGCGCGC",
        "ATATATATATATATATATATAT"[:20],
        "GATCGATCGATCGATCGATC",
        "TATCGATCGATCGATCGATC",
    ]]
    ranked = rank_grnas(candidates, top_n=3)
    assert len(ranked) == 3


def test_rank_grnas_descending():
    candidates = [_make_candidate(s) for s in [
        "ATCGATCGATCGATCGATCG",
        "ATATATATATATATATATATAT"[:20],
        "GCGCGCGCGCGCGCGCGCGC",
    ]]
    ranked = rank_grnas(candidates, top_n=3)
    scores = [c["score"] for c in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_grnas_scores_all_candidates():
    seqs = ["ATCGATCGATCGATCGATCG", "GCGCGCGCGCGCGCGCGCGC"]
    candidates = [_make_candidate(s) for s in seqs]
    ranked = rank_grnas(candidates, top_n=2)
    for c in ranked:
        assert "score" in c
        assert 0.0 <= c["score"] <= 1.0
