from __future__ import annotations
from pydantic import BaseModel, field_validator, model_validator
from typing import List, Optional


class PredictRequest(BaseModel):
    sequence: str
    pam: str = "NGG"
    top_n: int = 5
    # Optional: rank results by proximity to this genomic position (1-indexed).
    # combined_score = (1 - proximity_weight) * efficiency + proximity_weight * proximity
    target_position: Optional[int] = None
    proximity_weight: float = 0.4   # 0.0 = efficiency only, 1.0 = proximity only

    @field_validator("sequence", mode="before")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        v = v.upper().strip()
        invalid = set(v) - set("ACGTN")
        if invalid:
            raise ValueError(f"Invalid characters in sequence: {invalid}")
        if len(v) < 23:
            raise ValueError("Sequence must be at least 23 bp long.")
        return v

    @field_validator("pam", mode="before")
    @classmethod
    def validate_pam(cls, v: str) -> str:
        supported = {"NGG", "NAG", "NNGRRT", "TTTV"}
        v = v.upper().strip()
        if v not in supported:
            raise ValueError(f"PAM must be one of: {supported}")
        return v

    @field_validator("proximity_weight", mode="before")
    @classmethod
    def validate_proximity_weight(cls, v: float) -> float:
        if not (0.0 <= float(v) <= 1.0):
            raise ValueError("proximity_weight must be between 0.0 and 1.0.")
        return float(v)

    @model_validator(mode="after")
    def validate_target_position(self) -> "PredictRequest":
        if self.target_position is not None:
            seq_len = len(self.sequence)
            if not (1 <= self.target_position <= seq_len):
                raise ValueError(
                    f"target_position must be between 1 and {seq_len}."
                )
        return self


class GRNAResult(BaseModel):
    rank: int
    sequence: str
    pam_sequence: str
    position: int
    strand: str
    score: float                              # ML / heuristic efficiency score
    gc_content: float
    model_used: str
    cut_site: int                              # predicted cut position (1-indexed)
    distance_to_target: Optional[int] = None   # |cut_site - target_position|, bp
    combined_score: Optional[float] = None    # weighted efficiency + proximity
    off_target_score: float = 1.0             # specificity score (1=high, 0=low)


class PredictResponse(BaseModel):
    total_candidates: int
    top_grnas: List[GRNAResult]
    sequence_length: int
    pam_used: str
    model_info: str
    target_position: Optional[int] = None
    proximity_weight: Optional[float] = None
