"""
AI model wrappers for gRNA efficiency prediction.

Loads a pre-trained XGBoost Regressor from xgb_model.pkl.
Falls back to the heuristic scorer if the model file is not found.
"""
import pickle
import logging
from pathlib import Path
from typing import List, Optional
import numpy as np

from app.services.feature_engineering import extract_features

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "xgb_model.pkl"

MODEL_INFO_ML = (
    "XGBoost Regressor (500 trees, lr=0.03, max_depth=5) trained on 4,692 real "
    "experimental guides from Doench et al. 2016 (Nat Biotechnol 34:184) and "
    "Doench et al. 2014 (Nat Biotechnol 32:1262). "
    "Features: 450-dim (positional one-hot, GC, Tm, dinucleotides, 30-mer context, "
    "GC clamp, hairpin, microhomology, segmented Tm)."
)

MODEL_INFO_HEURISTIC = (
    "Heuristic scoring based on Doench 2016 Rule Set 2 principles: "
    "GC content (40-70% optimal), position-specific nucleotide preferences, "
    "and poly-T / homopolymer penalties. "
    "Run `python train_model.py` in the backend/ folder to enable ML scoring."
)

_model: Optional[object] = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                _model = pickle.load(f)
            logger.info("XGBoost model loaded from %s", MODEL_PATH)
        except Exception as e:
            logger.warning("Could not load XGBoost model: %s. Falling back to heuristic.", e)
            _model = None
    else:
        logger.info("xgb_model.pkl not found. Using heuristic scorer.")
    return _model


def predict_efficiency(candidates: List[dict], full_sequence: str = "") -> List[dict]:
    """
    Predict efficiency scores for a list of gRNA candidates.
    Uses XGBoost model if available, otherwise falls back to heuristic.

    full_sequence: the original input sequence, used to build 30-mer context
    (4 bp upstream + 20 bp guide + 6 bp downstream) for context-dependent features.
    """
    from app.services.scorer import score_grna  # lazy import to avoid circular

    model = _load_model()

    if model is not None:
        seq_upper = full_sequence.upper()

        def _thirty_mer(c: dict) -> str:
            pos = c.get("position", 0)
            s, e = pos - 4, pos + 26
            if seq_upper and s >= 0 and e <= len(seq_upper):
                return seq_upper[s:e]
            return ""

        X = np.vstack([
            extract_features(c["sequence"], _thirty_mer(c))
            for c in candidates
        ])
        scores = model.predict(X)
        for c, score in zip(candidates, scores):
            c["score"] = round(float(np.clip(score, 0.0, 1.0)), 4)
            c["model_used"] = "XGBoost (Doench-trained)"
    else:
        for c in candidates:
            c["score"] = score_grna(c)
            c["model_used"] = "Heuristic (Doench-inspired)"

    return candidates


def get_model_info() -> str:
    model = _load_model()
    return MODEL_INFO_ML if model is not None else MODEL_INFO_HEURISTIC
