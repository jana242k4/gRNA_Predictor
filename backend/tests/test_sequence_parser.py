import pytest
from app.services.sequence_parser import find_all_grnas

def test_spcas9_forward_strand():
    # Sequence: 20bp + NGG (CGG)
    seq = "ATGCGTACGTTAGCTAGCTA" + "CGG"
    results = find_all_grnas(seq, pam="NGG")
    assert len(results) >= 1
    found = [r for r in results if r["strand"] == "+"]
    assert len(found) == 1
    assert found[0]["sequence"] == "ATGCGTACGTTAGCTAGCTA"
    assert found[0]["pam_sequence"] == "CGG"

def test_spcas9_reverse_strand():
    # Target in reverse: CCN + 20bp guide
    # Forward: CCG + 20bp => Reverse complement will be 20bp + CGG (which is NGG)
    # Let's do 5' - CCG - ATGCGTACGTTAGCTAGCTA - 3'
    # RC: 5' - TAGCTAGCTAACGTACGCAT - CGG - 3'
    seq = "CCG" + "ATGCGTACGTTAGCTAGCTA"
    results = find_all_grnas(seq, pam="NGG")
    found = [r for r in results if r["strand"] == "-"]
    assert len(found) == 1
    assert found[0]["sequence"] == "TAGCTAGCTAACGTACGCAT"
    assert found[0]["pam_sequence"] == "CGG"

def test_cas12a_forward_strand():
    # Sequence: TTTV + 20bp 
    seq = "TTTA" + "ATGCGTACGTTAGCTAGCTA"
    results = find_all_grnas(seq, pam="TTTV")
    assert len(results) >= 1
    found = [r for r in results if r["strand"] == "+"]
    assert len(found) == 1
    assert found[0]["sequence"] == "ATGCGTACGTTAGCTAGCTA"
    assert found[0]["pam_sequence"] == "TTTA"

def test_no_pams_found():
    # Sequence with no NGG
    seq = "ATGCGTACGTTAGCTAGCTAAAAA" 
    results = find_all_grnas(seq, pam="NGG")
    assert len(results) == 0

def test_short_sequence():
    seq = "ATGC"
    results = find_all_grnas(seq, pam="NGG")
    assert len(results) == 0
