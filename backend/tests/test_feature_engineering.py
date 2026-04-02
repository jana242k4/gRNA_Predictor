"""
Unit tests for feature_engineering.py — covers individual feature functions
and the full extract_features pipeline.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.feature_engineering import (
    extract_features, extract_features_batch,
    N_FEATURES, GUIDE_LEN,
    seed_region_gc, has_poly_t,
    _positional_onehot, _dinucleotide_freq,
)

SEQ_20  = "ATCGATCGATCGATCGATCG"   # balanced GC (50%)
SEQ_GC  = "GCGCGCGCGCGCGCGCGCGC"  # 100% GC
SEQ_AT  = "ATATATATATATATATATAN"   # ~0% GC (except N)
SEQ_AT2 = "ATATATATATATATATATATAT"[:20]  # 0% GC


# ── Full vector shape ──────────────────────────────────────────────────────

def test_extract_features_shape_no_context():
    feat = extract_features(SEQ_20)
    assert feat.shape == (N_FEATURES,), f"Expected ({N_FEATURES},) got {feat.shape}"


def test_extract_features_shape_with_thirty_mer():
    thirty = "AAAA" + SEQ_20 + "GCGNNN"
    feat   = extract_features(SEQ_20, thirty_mer=thirty)
    assert feat.shape == (N_FEATURES,)


def test_extract_features_batch_shape():
    seqs = [SEQ_20, SEQ_GC]
    X    = extract_features_batch(seqs)
    assert X.shape == (2, N_FEATURES)


# ── Context dims zero when no thirty_mer ──────────────────────────────────

def test_context_dims_zero_without_thirty_mer():
    feat = extract_features(SEQ_20)
    # upstream [404:420], downstream [420:444], Tm_30mer_ctx [449]
    assert np.all(feat[404:420] == 0.0), "Upstream context should be 0 without 30-mer"
    assert np.all(feat[420:444] == 0.0), "Downstream context should be 0 without 30-mer"
    assert feat[449] == 0.0, "Tm 30-mer ctx should be 0 without 30-mer"


def test_context_dims_nonzero_with_thirty_mer():
    thirty = "GCTA" + SEQ_20 + "GGGAAA"
    feat   = extract_features(SEQ_20, thirty_mer=thirty)
    # upstream context should have at least some non-zero entries
    assert np.any(feat[404:420] != 0.0), "Upstream context should be non-zero when 30-mer provided"


# ── GC content feature ─────────────────────────────────────────────────────

def test_gc_feature_50pct():
    feat = extract_features(SEQ_20)
    assert abs(feat[80] - 0.5) < 1e-4, f"Expected GC~0.5, got {feat[80]}"


def test_gc_feature_100pct():
    feat = extract_features(SEQ_GC)
    assert abs(feat[80] - 1.0) < 1e-4, f"Expected GC=1.0, got {feat[80]}"


def test_gc_feature_0pct():
    seq  = "ATATATATATATATATATATAT"[:20]
    feat = extract_features(seq)
    assert feat[80] < 0.05, f"Expected GC~0, got {feat[80]}"


# ── Positional one-hot ─────────────────────────────────────────────────────

def test_onehot_shape():
    oh = _positional_onehot(SEQ_20)
    assert oh.shape == (80,)


def test_onehot_valid_binary():
    oh = _positional_onehot(SEQ_20)
    assert set(np.unique(oh)) <= {0.0, 1.0}, "One-hot must be binary"


def test_onehot_one_hot_per_position():
    oh = _positional_onehot(SEQ_20)
    for pos in range(GUIDE_LEN):
        pos_slice = oh[pos * 4: pos * 4 + 4]
        assert pos_slice.sum() == 1.0, f"Position {pos} must have exactly one 1"


# ── GC clamp ───────────────────────────────────────────────────────────────

def test_gc_clamp_all_gc():
    # last 4 bases all GC → clamp = 1.0
    seq  = "ATATATATATATATATGCGC"
    feat = extract_features(seq)
    assert abs(feat[444] - 1.0) < 1e-4, f"GC clamp should be 1.0, got {feat[444]}"


def test_gc_clamp_all_at():
    # last 4 bases all AT → clamp = 0.0
    seq  = "GCGCGCGCGCGCGCGCATAT"
    feat = extract_features(seq)
    assert abs(feat[444] - 0.0) < 1e-4, f"GC clamp should be 0.0, got {feat[444]}"


def test_gc_clamp_half():
    seq  = "GCGCGCGCGCGCGCGCGCAT"  # last 4: GCAT → 2/4 = 0.5
    feat = extract_features(seq)
    assert abs(feat[444] - 0.5) < 1e-4, f"GC clamp should be 0.5, got {feat[444]}"


# ── Hairpin proxy ──────────────────────────────────────────────────────────

def test_hairpin_proxy_known_stem():
    # GCGCGCGC forms a perfect palindrome → should give non-zero hairpin proxy
    seq  = "GCGCGCGCATATATATGCGC"
    feat = extract_features(seq)
    assert feat[445] > 0.0, f"Hairpin proxy should be >0 for palindromic seq, got {feat[445]}"


def test_hairpin_proxy_no_stem():
    # poly-A has no self-complementarity
    seq  = "AAAAAAAAAAAAAAAAAAAA"
    feat = extract_features(seq)
    assert feat[445] == 0.0, f"Hairpin proxy should be 0 for poly-A, got {feat[445]}"


def test_hairpin_proxy_range():
    for seq in [SEQ_20, SEQ_GC, "GCGCATATGCGCATATGCGC"]:
        feat = extract_features(seq[:20])
        assert 0.0 <= feat[445] <= 1.0, f"Hairpin proxy out of [0,1]: {feat[445]}"


# ── Microhomology ──────────────────────────────────────────────────────────

def test_microhomology_range():
    thirty = "AAAA" + SEQ_20 + "GCGAAA"
    feat   = extract_features(SEQ_20, thirty_mer=thirty)
    assert 0.0 <= feat[446] <= 1.0, f"Microhomology out of [0,1]: {feat[446]}"


def test_microhomology_zero_without_thirty_mer():
    feat = extract_features(SEQ_20)
    assert feat[446] == 0.0, "Microhomology should be 0 without 30-mer context"


# ── Seed region GC ─────────────────────────────────────────────────────────

def test_seed_gc_all_gc():
    # "ATATATATATGCGCGCGCGC" — last 12 bases: "ATGCGCGCGCGC" → 10/12 GC
    seq = "ATATATATATGCGCGCGCGC"[:20]
    assert abs(seed_region_gc(seq) - (10/12)) < 0.01


def test_seed_gc_range():
    for seq in [SEQ_20, SEQ_GC, SEQ_AT2]:
        val = seed_region_gc(seq)
        assert 0.0 <= val <= 1.0


# ── Poly-T flag ────────────────────────────────────────────────────────────

def test_poly_t_detected():
    seq = "ATCGATCGATCGATTTTTCG"
    assert has_poly_t(seq, 4) is True


def test_poly_t_not_detected():
    assert has_poly_t(SEQ_20, 4) is False


# ── Dinucleotide frequencies ───────────────────────────────────────────────

def test_dinuc_shape():
    dv = _dinucleotide_freq(SEQ_20)
    assert dv.shape == (16,)


def test_dinuc_sums_to_one():
    dv = _dinucleotide_freq(SEQ_20)
    assert abs(dv.sum() - 1.0) < 1e-4, f"Dinuc freqs should sum to 1, got {dv.sum()}"


def test_dinuc_nonnegative():
    dv = _dinucleotide_freq(SEQ_20)
    assert np.all(dv >= 0.0)
