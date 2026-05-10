"""
OmicsCRISPR FastAPI endpoints (Phase 5).

Routes:
  POST /omics/predict              — predict omics score + suitability for a guide
  POST /omics/explain              — integrated gradients feature attributions
  GET  /omics/gene/{gene}          — top guides for a gene by cell type
  GET  /omics/status               — pipeline data availability

All endpoints gracefully return 503 when the omics data/model is not available
(e.g. on Render free tier without torch), so the main /predict endpoint is
never affected.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional

router = APIRouter(prefix="/omics", tags=["omics"])

ALL_CELL_TYPES = ["T_cell_CD4", "T_cell_CD8", "NK_cell", "B_cell", "K562"]


# ── Request / response schemas ────────────────────────────────────────────────

class OmicsPredictRequest(BaseModel):
    sequence:   str             = Field(..., min_length=20, max_length=20,
                                        description="20-mer guide sequence (no PAM)")
    cell_types: Optional[list[str]] = Field(None,
                                        description="Cell types to score (default: all 5)")

class OmicsExplainRequest(BaseModel):
    sequence:  str = Field(..., min_length=20, max_length=20)
    cell_type: str = Field("K562", description="Cell type for attribution")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_predictor():
    try:
        from omics_pipeline.omics_inference import get_predictor
        return get_predictor()
    except Exception as e:
        raise HTTPException(status_code=503,
            detail=f"OmicsCRISPR module not available: {e}")


def _check_cell_types(cell_types: Optional[list[str]]) -> list[str]:
    if not cell_types:
        return ALL_CELL_TYPES
    invalid = [ct for ct in cell_types if ct not in ALL_CELL_TYPES]
    if invalid:
        raise HTTPException(status_code=422,
            detail=f"Unknown cell types: {invalid}. Valid: {ALL_CELL_TYPES}")
    return cell_types


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/predict")
def omics_predict(req: OmicsPredictRequest):
    """
    Multi-omics prediction for a 20-mer guide sequence.

    Returns per-cell-type scores combining the three-branch deep model (when
    the guide is in the DepMap precomputed set) with Phase 4 suitability and
    splice disruption risk.

    - **omics_score**: three-branch model output (combined label z-score)
    - **suitability_score**: Phase 4 composite (efficacy + RNA + ATAC + splice safety + essentiality)
    - **splice_risk**: probability the cut disrupts a splice site (0 = safe, 1 = critical)
    - **features**: raw omics values at this locus for the cell type
    """
    cts  = _check_cell_types(req.cell_types)
    pred = _get_predictor()
    result = pred.predict(req.sequence, cell_types=cts)

    if not result["in_depmap"]:
        result["note"] = (
            "Guide not found in DepMap Avana precomputed tables. "
            "omics_score and suitability_score are unavailable for novel sequences."
        )
    return result


@router.post("/explain")
def omics_explain(req: OmicsExplainRequest):
    """
    Integrated Gradients attribution for the 450-dim biochemical feature branch.

    Returns:
    - **feature_groups**: top-10 feature groups ranked by |attribution| (IG × input)
    - **branch_contributions**: relative contribution of each model branch via ablation
    - **full_score**: three-branch model output for this (guide, cell_type)

    Only available for guides in the precomputed DepMap set and requires torch.
    """
    if req.cell_type not in ALL_CELL_TYPES:
        raise HTTPException(status_code=422,
            detail=f"cell_type must be one of {ALL_CELL_TYPES}")

    pred   = _get_predictor()
    result = pred.explain(req.sequence, req.cell_type)

    if result is None:
        raise HTTPException(status_code=404,
            detail="Guide not found in precomputed tables or model unavailable. "
                   "Explanations require a guide present in the DepMap Avana dataset.")
    return result


@router.get("/gene/{gene}")
def omics_gene(
    gene:      str,
    cell_type: str  = Query("K562", description="Cell type to rank guides in"),
    top_n:     int  = Query(10, ge=1, le=50, description="Number of top guides to return"),
):
    """
    Return the top-N guides targeting a gene, ranked by cell-type suitability.

    Useful for choosing the best guide for a specific cell type when multiple
    DepMap Avana guides target the same gene.
    """
    if cell_type not in ALL_CELL_TYPES:
        raise HTTPException(status_code=422,
            detail=f"cell_type must be one of {ALL_CELL_TYPES}")

    pred  = _get_predictor()
    rows  = pred.top_guides_for_gene(gene, cell_type, top_n=top_n)

    if not rows:
        raise HTTPException(status_code=404,
            detail=f"No guides found for gene '{gene}' in precomputed tables.")

    return {
        "gene":      gene.upper(),
        "cell_type": cell_type,
        "top_guides": rows,
        "n_returned": len(rows),
    }


@router.get("/status")
def omics_status():
    """
    Report availability of Phase 2–4 data and the Phase 3 model.
    """
    pred = _get_predictor()
    pred._load()  # trigger lazy load

    from omics_pipeline.config import FEATURES_DIR, OMICS_DIR
    from pathlib import Path

    def _size(p: Path) -> str:
        if not p.exists():
            return "MISSING"
        s = p.stat().st_size
        return f"{s >> 20} MB" if s > 1 << 20 else f"{s >> 10} KB"

    return {
        "model_available":       pred.has_model,
        "n_guides_in_depmap":    pred.n_guides,
        "n_cell_feature_rows":   pred.n_cell_feats,
        "files": {
            "omics_model.pt":       _size(OMICS_DIR / "model" / "omics_model.pt"),
            "cell_features.csv":    _size(FEATURES_DIR / "cell_features.csv"),
            "splice_risk.csv":      _size(FEATURES_DIR / "splice_risk.csv"),
            "cell_suitability.csv": _size(FEATURES_DIR / "cell_suitability.csv"),
        },
        "cell_types": ALL_CELL_TYPES,
    }
