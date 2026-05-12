"""
Prediction endpoint.

Cut-site logic (SpCas9 / SaCas9 — 3' PAM):
  + strand: cut between guide positions 17 and 18  → cut_site = guide_pos + 17
  − strand: cut 3 bp from PAM end on genomic coords → cut_site = guide_pos + 3

Cas12a (TTTV — 5' PAM):
  Staggered cut ~18 bp into the guide → cut_site = guide_pos + 18

Proximity score uses Gaussian decay with sigma = 50 bp:
  proximity = exp(−distance² / (2 × 50²))
  → 0 bp = 1.0,  50 bp = 0.61,  100 bp = 0.14

Biological justification for σ = 50 bp:
  - SpCas9 can edit efficiently within ±50 bp of its cut site for most HDR
    applications (Paquet et al. 2016 Nature 533:125; Richardson et al. 2016
    Nat Biotechnol 34:339).
  - Guides >200 bp from a desired edit site carry non-trivial HDR failure risk
    due to reduced template proximity.
  - σ = 50 gives: guides within 25 bp → >78% proximity weight;
    guides at 100 bp → 14% proximity weight (effectively deprioritised).
  - Sensitivity analysis (compare_azimuth.py) confirms ranking order is robust
    to σ ∈ {25, 50, 100} bp for typical use cases.

Scoring formula (multi-objective):
  eff_adj  = efficiency × specificity          (off-target penalty applied)
  Without target: combined_score = eff_adj
  With target:    combined_score = (1−w) × eff_adj + w × proximity
    where w = request.proximity_weight (default 0.4)

Ranking is always by combined_score (incorporates off-target risk in all modes).
"""
import math
from fastapi import APIRouter, HTTPException, Request
from app.models.schemas import PredictRequest, PredictResponse, GRNAResult
from app.services.sequence_parser import find_all_grnas, clean_sequence
from app.models.ai_models import predict_efficiency, get_model_info
from app.services.off_target import specificity_score
from app.core.config import settings
from app.core.exceptions import SequenceTooLongError
from app.core.limiter import limiter

router = APIRouter()

_CAS12A_PAMS    = {"TTTV"}
_PROXIMITY_SIGMA = 50.0   # bp — controls how fast proximity decays with distance


def _cut_site(candidate: dict, pam: str) -> int:
    """
    Predicted genomic cut position (1-indexed).

    SpCas9 / SaCas9 (3'-PAM):
      + strand → cut between guide nt 17-18 (3 bp upstream of PAM)
      − strand → cut 3 bp from PAM end in genomic coordinates

    Cas12a (5'-PAM): staggered cut ~18 bp into guide
    """
    pos    = candidate["position"]   # 0-indexed guide start on + strand
    strand = candidate["strand"]

    if pam in _CAS12A_PAMS:
        offset = 18
    else:
        offset = 17 if strand == "+" else 3

    return pos + offset + 1   # convert to 1-indexed


def _proximity_score(distance: int, sigma: float = _PROXIMITY_SIGMA) -> float:
    """Gaussian proximity score: 1.0 at 0 bp, ~0.61 at sigma bp."""
    return math.exp(-(distance ** 2) / (2.0 * sigma ** 2))


@router.post("/predict", response_model=PredictResponse)
@limiter.limit("30/minute")
def predict_grnas(request: Request, req: PredictRequest):
    """
    Return the top N predicted guide RNAs, optionally ranked by combined
    efficiency + proximity score when a target_position is provided.
    """
    sequence = clean_sequence(req.sequence)

    if len(sequence) > settings.MAX_SEQUENCE_LENGTH:
        raise SequenceTooLongError(settings.MAX_SEQUENCE_LENGTH)

    candidates = find_all_grnas(sequence, pam=req.pam)

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=f"No valid PAM ({req.pam}) sites found in the provided sequence.",
        )

    # Pre-filter by GC content before ML inference to keep response fast
    MAX_CANDIDATES = 300
    if len(candidates) > MAX_CANDIDATES:
        gc_filtered = [c for c in candidates if 0.35 <= c["gc_content"] <= 0.75]
        candidates = gc_filtered[:MAX_CANDIDATES] if gc_filtered else candidates[:MAX_CANDIDATES]

    # --- Efficiency scoring (ML or heuristic) ---
    scored = predict_efficiency(candidates, full_sequence=sequence)

    # --- Cut site, proximity, and off-target specificity ---
    target = req.target_position   # 1-indexed or None
    w      = req.proximity_weight

    for c in scored:
        cs            = _cut_site(c, req.pam)
        c["cut_site"] = cs
        spec = specificity_score(c["sequence"], pam=req.pam)
        c["off_target_score"] = round(spec, 3)

        # Efficiency adjusted by off-target specificity (multiplicative penalty)
        eff_adj = c["score"] * spec

        if target is not None:
            dist                    = abs(cs - target)
            c["distance_to_target"] = dist
            c["combined_score"]     = round(
                (1.0 - w) * eff_adj + w * _proximity_score(dist), 4
            )
        else:
            c["distance_to_target"] = None
            # No target: rank by specificity-adjusted efficiency
            c["combined_score"]     = round(eff_adj, 4)

    # Always rank by combined_score (incorporates off-target risk in all modes)
    ranked = sorted(scored, key=lambda x: x["combined_score"], reverse=True)[: req.top_n]

    results = [
        GRNAResult(
            rank=i + 1,
            sequence=g["sequence"],
            pam_sequence=g["pam_sequence"],
            position=g["position"],
            strand=g["strand"],
            score=g["score"],
            gc_content=round(g["gc_content"], 3),
            model_used=g["model_used"],
            cut_site=g["cut_site"],
            distance_to_target=g["distance_to_target"],
            combined_score=g["combined_score"],
            off_target_score=g["off_target_score"],
        )
        for i, g in enumerate(ranked)
    ]

    return PredictResponse(
        total_candidates=len(candidates),
        top_grnas=results,
        sequence_length=len(sequence),
        pam_used=req.pam,
        model_info=get_model_info(),
        target_position=target,
        proximity_weight=w if target is not None else None,
    )
