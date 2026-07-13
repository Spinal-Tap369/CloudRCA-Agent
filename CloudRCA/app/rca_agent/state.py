from __future__ import annotations

from typing import Any, TypedDict


class RCAAgentState(TypedDict, total=False):
    scenario_dir: str
    scenario_id: str
    graph_pack: dict[str, Any]
    graph_report: str
    top_candidates: list[dict[str, Any]]
    candidate_table: list[dict[str, Any]]
    incident_pattern: dict[str, Any]
    primary_case: dict[str, Any]
    rival_cases: dict[str, Any]
    candidate_analysis: dict[str, Any]
    tournament_result: dict[str, Any]
    causal_audit: dict[str, Any]
    propagation_audit: dict[str, Any]
    contradiction_audit: dict[str, Any]
    confidence_audit: dict[str, Any]
    proposed_result: dict[str, Any]
    final_result: dict[str, Any]
    validation_errors: list[str]
    attempts: int
    max_attempts: int
