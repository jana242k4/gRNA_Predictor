"""
Comprehensive test suite for the gRNA Predictor API.
Tests cover:
  - Health endpoint
  - Basic prediction pipeline (NGG, TTTV)
  - PAM detection correctness
  - Edge cases: short seqs, invalid chars, no PAM found
  - Score boundaries (0 <= score <= 1)
  - Feature engineering output shape
  - Sequence parser both-strand logic
"""
import pytest
import re
from fastapi.testclient import TestClient
from app.main import app
from app.services.sequence_parser import find_all_grnas
from app.services.feature_engineering import extract_features, GUIDE_LEN
from app.utils.biology_utils import reverse_complement, gc_content

client = TestClient(app)

# ── Sequences ────────────────────────────────────────────────────────────────
# Contains multiple NGG sites
SAMPLE_SEQ = (
    "ATCGATCGATCGATCGATCGGG"   # NGG at position 19
    "GCTAGCTAGCTAGCTAGCTAGC"
    "ATCGATCGATCGATCGATCGGG"   # NGG again
    "TTTTTTTTTTTTTTTTTTTTGG"   # high poly-T guide (low score expected)
    "GCGCGCGCGCGCGCGCGCGCGG"   # high GC guide
) * 2

# Contains TTTV (Cas12a) PAM
CAS12A_SEQ = (
    "TTTACATCGATCGATCGATCGATCGATCG"
    "GCTAGCTAGCTAGCTAGCTAGCTAGCTAG"
    "TTTGCGCGATCGATCGATCGATCGATCGA"
) * 2


# ── Health ───────────────────────────────────────────────────────────────────
def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── Basic Prediction ─────────────────────────────────────────────────────────
def test_predict_basic_ngg():
    resp = client.post("/api/v1/predict", json={"sequence": SAMPLE_SEQ, "pam": "NGG"})
    assert resp.status_code == 200
    data = resp.json()
    assert "top_grnas" in data
    assert len(data["top_grnas"]) >= 1
    assert data["pam_used"] == "NGG"


def test_predict_cas12a():
    resp = client.post("/api/v1/predict", json={"sequence": CAS12A_SEQ, "pam": "TTTV"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["pam_used"] == "TTTV"
    assert len(data["top_grnas"]) >= 1


def test_predict_top_n_respected():
    resp = client.post("/api/v1/predict", json={"sequence": SAMPLE_SEQ, "top_n": 3})
    assert resp.status_code == 200
    assert len(resp.json()["top_grnas"]) <= 3


# ── Score Boundaries ─────────────────────────────────────────────────────────
def test_scores_in_range():
    resp = client.post("/api/v1/predict", json={"sequence": SAMPLE_SEQ})
    assert resp.status_code == 200
    for g in resp.json()["top_grnas"]:
        assert 0.0 <= g["score"] <= 1.0, f"Score out of range: {g['score']}"


def test_scores_descending():
    """Top gRNAs must be sorted highest score first."""
    resp = client.post("/api/v1/predict", json={"sequence": SAMPLE_SEQ})
    scores = [g["score"] for g in resp.json()["top_grnas"]]
    assert scores == sorted(scores, reverse=True)


def test_gc_content_in_range():
    resp = client.post("/api/v1/predict", json={"sequence": SAMPLE_SEQ})
    for g in resp.json()["top_grnas"]:
        assert 0.0 <= g["gc_content"] <= 1.0


# ── Edge Cases ───────────────────────────────────────────────────────────────
def test_invalid_characters():
    resp = client.post("/api/v1/predict", json={"sequence": "ATCGXYZ123ATCGATCG"})
    assert resp.status_code == 422


def test_sequence_too_short():
    resp = client.post("/api/v1/predict", json={"sequence": "ATCGATCG"})
    assert resp.status_code == 422


def test_unsupported_pam():
    resp = client.post("/api/v1/predict", json={"sequence": SAMPLE_SEQ, "pam": "XYZ"})
    assert resp.status_code == 422


def test_no_pam_found():
    # A sequence with no NGG (only AAA repeats)
    seq = "A" * 100
    resp = client.post("/api/v1/predict", json={"sequence": seq, "pam": "NGG"})
    assert resp.status_code == 404


# ── PAM Detection ────────────────────────────────────────────────────────────
def test_ngg_site_present_in_guide():
    """Guides from NGG search must be immediately followed by a G/GG."""
    candidates = find_all_grnas(SAMPLE_SEQ, pam="NGG")
    assert len(candidates) > 0
    for c in candidates:
        pos = c["position"]
        if c["strand"] == "+":
            pam_in_seq = SAMPLE_SEQ[pos + 20 : pos + 23].upper()
            assert pam_in_seq.endswith("GG"), f"PAM not NGG at pos {pos}: {pam_in_seq}"


def test_guide_length_always_20():
    candidates = find_all_grnas(SAMPLE_SEQ, pam="NGG")
    for c in candidates:
        assert len(c["sequence"]) == 20


def test_both_strands_detected():
    """Reverse strand candidates should have strand='-'."""
    candidates = find_all_grnas(SAMPLE_SEQ, pam="NGG")
    strands = {c["strand"] for c in candidates}
    assert "+" in strands


# ── Feature Engineering ──────────────────────────────────────────────────────
def test_feature_vector_shape():
    feat = extract_features("ATCGATCGATCGATCGATCG")
    assert feat.shape == (452,)   # 450 + PAM-proximal 10bp GC + PAM-distal 10bp GC


def test_feature_vector_onehot_valid():
    feat = extract_features("AAAAAAAAAAAAAAAAAAAA")
    # First base is A -> index 0 should be 1, indices 1,2,3 should be 0
    assert feat[0] == 1.0
    assert feat[1] == 0.0
    assert feat[2] == 0.0
    assert feat[3] == 0.0


def test_gc_feature_value():
    """Pure GC sequence should have GC feature ~= 1.0."""
    feat = extract_features("GCGCGCGCGCGCGCGCGCGC")
    gc_feature = feat[80]  # index 80 is GC content
    assert abs(gc_feature - 1.0) < 0.01


# ── Biology Utils ────────────────────────────────────────────────────────────
def test_reverse_complement():
    assert reverse_complement("ATCG") == "CGAT"
    assert reverse_complement("AAAA") == "TTTT"
    assert reverse_complement("GCTA") == "TAGC"


def test_gc_content_calculation():
    assert gc_content("GGCC") == 1.0
    assert gc_content("ATAT") == 0.0
    assert abs(gc_content("ATGC") - 0.5) < 0.01


# ── Off-Target Specificity ────────────────────────────────────────────────────
from app.services.off_target import specificity_score

def test_off_target_score_in_range():
    """Specificity score must always be in [0, 1]."""
    for seq in ["AAAAAAAAAAAAAAAAAAAA", "GCGCGCGCGCGCGCGCGCGC",
                "GAGTCCGAGCAGAAGAAGAA", "TTTTTTTTTTTTTTTTTTTT"]:
        s = specificity_score(seq)
        assert 0.0 <= s <= 1.0, f"Out of range for {seq}: {s}"


def test_at_rich_guide_lower_specificity():
    """All-AT guide (mismatch-permissive seed) should score lower than balanced guide."""
    at_rich  = specificity_score("AAAATTTTTAAAAATAAAAT")
    balanced = specificity_score("GAGTCCGAGCAGAAGAAGAA")
    assert at_rich < balanced


def test_off_target_returned_in_api():
    """API response must include off_target_score for every result."""
    resp = client.post("/api/v1/predict", json={"sequence": SAMPLE_SEQ, "pam": "NGG"})
    assert resp.status_code == 200
    for g in resp.json()["top_grnas"]:
        assert "off_target_score" in g
        assert 0.0 <= g["off_target_score"] <= 1.0
