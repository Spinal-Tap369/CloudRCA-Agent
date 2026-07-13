from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas import EvidenceItem, RootCauseEntity


class CandidateAssessment(BaseModel):
    candidate_id: str = ""
    rank: int | None = None
    role: str = Field(default="rival")
    verdict: str = ""
    support: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)


class CandidateAnalysis(BaseModel):
    selected_candidate_id: str = ""
    selected_candidate_rank: int | None = None
    root_cause: RootCauseEntity
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    assessments: list[CandidateAssessment] = Field(default_factory=list)
    rival_summary: str = ""
    evidence_summaries: list[EvidenceItem] = Field(default_factory=list)
    recommended_remediation: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class CandidateInspection(BaseModel):
    candidate_id: str = ""
    verdict: str = ""
    causal_story: str = ""
    support: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    symptom_fit: str = ""


class RivalInspectionSet(BaseModel):
    rivals: list[CandidateAssessment] = Field(default_factory=list)
    rival_summary: str = ""


class TournamentChoice(BaseModel):
    selected_candidate_id: str = ""
    reasoning: str = ""
    rejected_candidates: list[str] = Field(default_factory=list)
