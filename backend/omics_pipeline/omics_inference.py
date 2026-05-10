"""
OmicsCRISPR Phase 5 -- Inference Module

Loads the Phase 3 three-branch model + Phase 4 lookup tables and exposes
a clean prediction API used by the FastAPI omics endpoints.

Lazy-loads everything on first use so importing this module at app startup
is cheap.  Handles the case where torch is not installed (Render free tier
may not have it) by returning None from get_predictor().
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_BACKEND = Path(__file__).parent.parent
sys.path.insert(0, str(_BACKEND))

from .config import FEATURES_DIR

# Paths
MODEL_DIR        = FEATURES_DIR.parent / "model"
MODEL_PT         = MODEL_DIR  / "omics_model.pt"
MODEL_CONFIG_JSON= MODEL_DIR  / "model_config.json"

GUIDE_META_CSV   = FEATURES_DIR / "guide_metadata.csv"
CELL_FEAT_CSV    = FEATURES_DIR / "cell_features.csv"
SPLICE_RISK_CSV  = FEATURES_DIR / "splice_risk.csv"
CELL_SUIT_CSV    = FEATURES_DIR / "cell_suitability.csv"
SEQ_NPZ          = FEATURES_DIR / "seq_features.npz"

ALL_CELL_TYPES = ["T_cell_CD4", "T_cell_CD8", "NK_cell", "B_cell", "K562"]

BASES    = "ACGT"
BASE_IDX = {b: i for i, b in enumerate(BASES)}

# 450-dim feature group labels (for attribution display)
FEAT_GROUPS = [
    ("Positional one-hot",       0,   80),
    ("GC content",               80,  81),
    ("Nearest-neighbor Tm",      81,  82),
    ("Dinucleotide frequencies", 82,  98),
    ("Seed GC",                  98,  99),
    ("Poly-T flag",              99,  100),
    ("Pos-specific dinucs",      100, 404),
    ("Upstream context (4 bp)",  404, 420),
    ("Downstream context (6 bp)",420, 444),
    ("GC clamp",                 444, 445),
    ("Hairpin proxy",            445, 446),
    ("Microhomology",            446, 447),
    ("Tm PAM-distal",            447, 448),
    ("Tm seed region",           448, 449),
    ("Tm full 30-mer",           449, 450),
]


# ── One-hot encoding ─────────────────────────────────────────────────────────

def seq_to_onehot(seq: str, seq_len: int = 20) -> np.ndarray:
    x = np.zeros((4, seq_len), dtype=np.float32)
    for j, base in enumerate(seq[:seq_len].upper()):
        i = BASE_IDX.get(base, -1)
        if i >= 0:
            x[i, j] = 1.0
    return x


# ── OmicsPredictor ────────────────────────────────────────────────────────────

class OmicsPredictor:
    """
    Wraps the three-branch PyTorch model + lookup tables.
    All heavy data is loaded lazily on first call to predict().
    """

    def __init__(self):
        self._model      = None
        self._norm_stats = None
        self._cfg        = None
        self._cell_feats : dict[tuple, dict]  = {}   # (guide_id, ct) -> feature dict
        self._splice_risk: dict[str, dict]    = {}   # guide_id -> {risk, dist, type}
        self._suitability: dict[tuple, float] = {}   # (guide_id, ct) -> score
        self._guide_meta : dict[str, dict]    = {}   # guide_id -> metadata row
        self._seq_feats  : Optional[np.ndarray] = None  # (N, 450)
        self._guide_order: list[str] = []             # guide_id order in seq_feats
        self._ready      = False
        self._torch_ok   = False

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._ready:
            return
        try:
            import torch
            self._torch_ok = True
        except ImportError:
            self._torch_ok = False

        self._load_lookup_tables()
        if self._torch_ok:
            self._load_model()
        self._ready = True

    def _load_model(self) -> None:
        import torch

        if not MODEL_PT.exists():
            self._torch_ok = False
            return

        try:
            ckpt = torch.load(MODEL_PT, map_location="cpu", weights_only=False)
            cfg  = ckpt.get("model_config", {})
            self._cfg        = cfg
            self._norm_stats = ckpt.get("norm_stats", {})

            from omics_pipeline.train_omics_model import OmicsCRISPRModel
            model = OmicsCRISPRModel(seq_feat_dim=cfg.get("seq_feat_dim", 450))
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self._model = model
        except Exception as e:
            print(f"[OmicsPredictor] Model load failed: {e}")
            self._torch_ok = False

    def _load_lookup_tables(self) -> None:
        # Guide metadata
        if GUIDE_META_CSV.exists():
            with open(GUIDE_META_CSV, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self._guide_meta[row["guide_id"]] = row
                    self._guide_order.append(row["guide_id"])

        # Sequence features
        if SEQ_NPZ.exists():
            data = np.load(SEQ_NPZ)
            self._seq_feats = data["seq_features"].astype(np.float32)

        # Cell-level omics features
        if CELL_FEAT_CSV.exists():
            with open(CELL_FEAT_CSV, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    key = (row["guide_id"], row["cell_type"])
                    self._cell_feats[key] = {
                        "rna_tpm_log1p":  float(row["rna_tpm_log1p"]),
                        "atac_signal":    float(row["atac_signal"]),
                        "atac_n_peaks":   int(row["atac_n_peaks"]),
                        "splice_dist_log":float(row["splice_dist_log"]),
                        "gene_effect":    float(row["gene_effect"]),
                        "cell_type_idx":  int(row["cell_type_idx"]),
                    }

        # Splice risk
        if SPLICE_RISK_CSV.exists():
            with open(SPLICE_RISK_CSV, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self._splice_risk[row["guide_id"]] = {
                        "splice_risk":            float(row["splice_risk"]),
                        "nearest_splice_dist_bp": int(row["nearest_splice_dist_bp"]),
                        "nearest_splice_type":    row["nearest_splice_type"],
                        "cut_site":               int(row["cut_site"]),
                    }

        # Cell suitability
        if CELL_SUIT_CSV.exists():
            with open(CELL_SUIT_CSV, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    key = (row["guide_id"], row["cell_type"])
                    self._suitability[key] = float(row["suitability_score"])

    # ── Normalisation ─────────────────────────────────────────────────────────

    def _normalise_seq_feats(self, raw: np.ndarray) -> np.ndarray:
        ns = self._norm_stats
        if not ns:
            return raw
        sf_mean = np.array(ns["sf_mean"], dtype=np.float32)
        sf_std  = np.array(ns["sf_std"],  dtype=np.float32)
        return (raw - sf_mean) / (sf_std + 1e-8)

    def _normalise_omics(self, omics: np.ndarray, ct_idx: int) -> np.ndarray:
        ns = self._norm_stats
        if not ns:
            return omics
        om_mean = np.array(ns["om_mean"], dtype=np.float32)
        om_std  = np.array(ns["om_std"],  dtype=np.float32)
        normed  = (omics - om_mean) / (om_std + 1e-8)
        normed[-1] = ct_idx  # restore raw cell_type_idx
        return normed

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        sequence: str,
        cell_types: Optional[list[str]] = None,
    ) -> dict:
        """
        Return per-cell-type predictions for a 20-mer guide sequence.
        Uses precomputed Phase 2/4 tables if the guide is in DepMap,
        otherwise returns only lookup-based suitability (omics_score = None).
        """
        self._load()
        seq    = sequence.upper().strip()[:20]
        cts    = cell_types or ALL_CELL_TYPES
        meta   = self._guide_meta.get(seq, {})
        in_db  = bool(meta)

        predictions = []
        for ct_idx, ct in enumerate(ALL_CELL_TYPES):
            if ct not in cts:
                continue

            cf  = self._cell_feats.get((seq, ct), {})
            sr  = self._splice_risk.get(seq, {})
            suit= self._suitability.get((seq, ct))

            omics_score = None
            if in_db and self._torch_ok and self._model and cf:
                omics_score = self._run_model(seq, cf, ct_idx)

            predictions.append({
                "cell_type":              ct,
                "omics_score":            omics_score,
                "suitability_score":      suit,
                "splice_risk":            sr.get("splice_risk"),
                "nearest_splice_dist_bp": sr.get("nearest_splice_dist_bp"),
                "nearest_splice_type":    sr.get("nearest_splice_type"),
                "features":               cf or None,
            })

        return {
            "guide_id":   seq,
            "in_depmap":  in_db,
            "gene":       meta.get("gene", ""),
            "chr":        meta.get("chr", ""),
            "strand":     meta.get("strand", ""),
            "efficacy":   float(meta["efficacy"]) if meta.get("efficacy") else None,
            "predictions": predictions,
        }

    def _run_model(self, seq: str, cf: dict, ct_idx: int) -> float:
        import torch

        guide_idx = self._guide_order.index(seq) if seq in self._guide_order else -1
        if guide_idx < 0 or self._seq_feats is None:
            return None

        raw_sf = self._seq_feats[guide_idx]
        sf_n   = self._normalise_seq_feats(raw_sf)

        raw_om = np.array([
            cf["rna_tpm_log1p"], cf["atac_signal"], cf["atac_n_peaks"],
            cf["splice_dist_log"], cf["gene_effect"], float(ct_idx),
        ], dtype=np.float32)
        om_n   = self._normalise_omics(raw_om, ct_idx)

        oh   = torch.from_numpy(seq_to_onehot(seq)).unsqueeze(0)   # (1,4,20)
        sf_t = torch.from_numpy(sf_n).unsqueeze(0)                 # (1,450)
        om_t = torch.from_numpy(om_n).unsqueeze(0)                 # (1,6)

        with torch.no_grad():
            score = float(self._model(oh, sf_t, om_t).item())
        return round(score, 4)

    # ── Attribution (Integrated Gradients) ────────────────────────────────────

    def explain(
        self,
        sequence: str,
        cell_type: str,
        n_steps: int = 50,
    ) -> Optional[dict]:
        """
        Integrated Gradients attribution for the 450-dim feature branch.
        Returns grouped attribution scores + raw branch contribution estimates.
        Returns None if the guide is not in precomputed tables or torch unavailable.
        """
        self._load()
        if not self._torch_ok or not self._model:
            return None

        seq     = sequence.upper().strip()[:20]
        ct_idx  = ALL_CELL_TYPES.index(cell_type) if cell_type in ALL_CELL_TYPES else 0
        cf      = self._cell_feats.get((seq, cell_type))
        guide_i = self._guide_order.index(seq) if seq in self._guide_order else -1

        if cf is None or guide_i < 0 or self._seq_feats is None:
            return None

        import torch

        raw_sf = self._seq_feats[guide_i]
        sf_n   = self._normalise_seq_feats(raw_sf)

        raw_om = np.array([
            cf["rna_tpm_log1p"], cf["atac_signal"], cf["atac_n_peaks"],
            cf["splice_dist_log"], cf["gene_effect"], float(ct_idx),
        ], dtype=np.float32)
        om_n   = self._normalise_omics(raw_om, ct_idx)

        oh     = torch.from_numpy(seq_to_onehot(seq)).unsqueeze(0)
        sf_inp = torch.tensor(sf_n, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
        om_t   = torch.tensor(om_n, dtype=torch.float32).unsqueeze(0)

        # Integrated Gradients for feature MLP branch
        sf_base = torch.zeros_like(sf_inp)
        alphas  = torch.linspace(0, 1, n_steps + 1, dtype=torch.float32)
        grads   = []

        for alpha in alphas:
            sf_interp = (sf_base + alpha * (sf_inp - sf_base)).detach().requires_grad_(True)
            out = self._model(oh, sf_interp, om_t)
            out.backward()
            grads.append(sf_interp.grad.detach().squeeze(0).numpy())

        avg_grad   = np.mean(grads, axis=0)
        ig_attrs   = avg_grad * (sf_n - np.zeros_like(sf_n))  # IG = avg_grad * (x - baseline)

        # Group attributions
        group_attrs = []
        for name, lo, hi in FEAT_GROUPS:
            group_attrs.append({
                "group":       name,
                "attribution": round(float(ig_attrs[lo:hi].sum()), 5),
                "n_dims":      hi - lo,
            })
        group_attrs.sort(key=lambda g: abs(g["attribution"]), reverse=True)

        # Branch contribution via ablation (zero each branch, measure output drop)
        full_out = float(self._model(oh, sf_inp.detach(), om_t).item())

        with torch.no_grad():
            no_cnn  = float(self._model(torch.zeros_like(oh), sf_inp.detach(), om_t).item())
            no_feat = float(self._model(oh, torch.zeros_like(sf_inp), om_t).item())
            no_om   = float(self._model(oh, sf_inp.detach(), torch.zeros_like(om_t)).item())

        def _contrib(ablated): return round(abs(full_out - ablated), 4)

        total_c = _contrib(no_cnn) + _contrib(no_feat) + _contrib(no_om) + 1e-9
        branch_contributions = {
            "sequence_cnn":  round(_contrib(no_cnn)  / total_c, 3),
            "feature_mlp":   round(_contrib(no_feat) / total_c, 3),
            "omics_mlp":     round(_contrib(no_om)   / total_c, 3),
        }

        return {
            "feature_groups":      group_attrs[:10],  # top-10 by |attribution|
            "branch_contributions": branch_contributions,
            "full_score":          round(full_out, 4),
        }

    # ── Gene-level query ──────────────────────────────────────────────────────

    def top_guides_for_gene(
        self,
        gene: str,
        cell_type: str,
        top_n: int = 10,
    ) -> list[dict]:
        """Return top N guides for a gene in a given cell type, sorted by suitability."""
        self._load()
        gene_u = gene.upper()
        rows = []
        seen = set()
        for gid, meta in self._guide_meta.items():
            if meta.get("gene", "").upper() != gene_u:
                continue
            key = (gid, cell_type)
            suit = self._suitability.get(key)
            if suit is None or key in seen:
                continue
            seen.add(key)
            sr = self._splice_risk.get(gid, {})
            rows.append({
                "guide_id":        gid,
                "efficacy":        float(meta.get("efficacy", 0)),
                "suitability_score": suit,
                "splice_risk":     sr.get("splice_risk", 0.0),
                "chr":             meta.get("chr", ""),
                "strand":          meta.get("strand", ""),
            })
        rows.sort(key=lambda r: r["suitability_score"], reverse=True)
        return rows[:top_n]

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def has_model(self) -> bool:
        return self._torch_ok and self._model is not None

    @property
    def n_guides(self) -> int:
        return len(self._guide_meta)

    @property
    def n_cell_feats(self) -> int:
        return len(self._cell_feats)


# ── Singleton ─────────────────────────────────────────────────────────────────

_predictor: Optional[OmicsPredictor] = None


def get_predictor() -> OmicsPredictor:
    global _predictor
    if _predictor is None:
        _predictor = OmicsPredictor()
    return _predictor
