from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.rca_agent.audits import (
    audit_causal_paths,
    audit_contradictions,
    audit_propagation,
    calibrate_confidence,
)
from app.rca_agent.candidate_table import build_candidate_table, default_winner
from app.rca_agent.case import build_case
from app.rca_agent.llm import invoke_json
from app.rca_agent.models import CandidateInspection, RivalInspectionSet, TournamentChoice
from app.rca_agent.pattern import classify_incident_pattern
from app.rca_agent.prompts import (
    diagnosis_prompt,
    primary_candidate_prompt,
    repair_prompt,
    rival_candidates_prompt,
    tournament_prompt,
)
from app.rca_agent.state import RCAAgentState
from app.rca_agent.tournament import adjudicate_tournament
from app.rca_agent.validators import fallback_result, postprocess_result, validate_result
from app.schemas import DiagnosisResult


def build_case_node(state: RCAAgentState) -> dict[str, Any]:
    case = build_case(state["scenario_dir"])
    return {
        **case,
        "attempts": int(state.get("attempts") or 0),
        "max_attempts": int(state.get("max_attempts") or 2),
        "validation_errors": [],
    }


def normalize_candidates_node(state: RCAAgentState) -> dict[str, Any]:
    table = build_candidate_table(state.get("graph_pack") or {})
    return {
        "candidate_table": table,
        "top_candidates": table[:10],
    }


def classify_pattern_node(state: RCAAgentState) -> dict[str, Any]:
    return {
        "incident_pattern": classify_incident_pattern(state.get("candidate_table") or [])
    }


def inspect_primary_node(state: RCAAgentState) -> dict[str, Any]:
    primary = default_winner(state.get("candidate_table") or []) or {}

    try:
        raw = invoke_json(primary_candidate_prompt(state))
        inspection = CandidateInspection.model_validate(raw)
        return {"primary_case": inspection.model_dump()}
    except (ValueError, ValidationError, RuntimeError) as exc:
        return {
            "primary_case": {
                "candidate_id": primary.get("id", ""),
                "verdict": "Primary candidate retained from graph ranking.",
                "causal_story": primary.get("selection_reason") or "Graph-ranked selectable candidate.",
                "support": list(primary.get("supporting_evidence") or [])[:4],
                "concerns": [f"LLM primary inspection failed: {exc}"],
                "symptom_fit": (
                    f"causal_paths={primary.get('causal_path_count', 0)}, "
                    f"affected_symptoms={primary.get('affected_symptom_count', 0)}"
                ),
            }
        }


def inspect_rivals_node(state: RCAAgentState) -> dict[str, Any]:
    try:
        raw = invoke_json(rival_candidates_prompt(state))
        rivals = RivalInspectionSet.model_validate(raw)
        return {"rival_cases": rivals.model_dump()}
    except (ValueError, ValidationError, RuntimeError) as exc:
        candidate_table = state.get("candidate_table") or []
        rivals = []

        for candidate in candidate_table[1:6]:
            rivals.append(
                {
                    "candidate_id": candidate.get("id"),
                    "rank": candidate.get("rank"),
                    "role": "rival" if candidate.get("root_selectable") else "context",
                    "verdict": candidate.get("selection_reason") or "Graph rival candidate.",
                    "support": list(candidate.get("supporting_evidence") or [])[:3],
                    "concerns": list(candidate.get("blockers") or [])[:3],
                }
            )

        return {
            "rival_cases": {
                "rivals": rivals,
                "rival_summary": f"LLM rival inspection failed: {exc}",
            }
        }


def tournament_node(state: RCAAgentState) -> dict[str, Any]:
    llm_choice = TournamentChoice()

    try:
        raw = invoke_json(tournament_prompt(state))
        llm_choice = TournamentChoice.model_validate(raw)
    except (ValueError, ValidationError, RuntimeError) as exc:
        llm_choice = TournamentChoice(reasoning=f"LLM tournament failed: {exc}")

    result = adjudicate_tournament(
        candidate_table=state.get("candidate_table") or [],
        llm_selected_candidate_id=llm_choice.selected_candidate_id,
        llm_reasoning=llm_choice.reasoning,
    )
    result["rejected_candidates"] = llm_choice.rejected_candidates
    return {"tournament_result": result}


def causal_path_audit_node(state: RCAAgentState) -> dict[str, Any]:
    winner = (state.get("tournament_result") or {}).get("winner") or {}
    return {
        "causal_audit": audit_causal_paths(state.get("candidate_table") or [], winner)
    }


def propagation_audit_node(state: RCAAgentState) -> dict[str, Any]:
    winner = (state.get("tournament_result") or {}).get("winner") or {}
    return {
        "propagation_audit": audit_propagation(state.get("candidate_table") or [], winner)
    }


def contradiction_audit_node(state: RCAAgentState) -> dict[str, Any]:
    winner = (state.get("tournament_result") or {}).get("winner") or {}
    return {
        "contradiction_audit": audit_contradictions(state.get("candidate_table") or [], winner)
    }


def confidence_audit_node(state: RCAAgentState) -> dict[str, Any]:
    winner = (state.get("tournament_result") or {}).get("winner") or {}
    return {
        "confidence_audit": calibrate_confidence(
            winner=winner,
            causal_audit=state.get("causal_audit") or {},
            contradiction_audit=state.get("contradiction_audit") or {},
            pattern=state.get("incident_pattern") or {},
        )
    }


def draft_diagnosis_node(state: RCAAgentState) -> dict[str, Any]:
    try:
        raw = invoke_json(diagnosis_prompt(state))
        result = postprocess_result(DiagnosisResult.model_validate(raw))
        return {"proposed_result": result.model_dump()}
    except (ValueError, ValidationError, RuntimeError) as exc:
        result = fallback_result(state)
        return {
            "proposed_result": result.model_dump(),
            "validation_errors": [f"Diagnosis drafting failed: {exc}"],
        }


def validate_diagnosis_node(state: RCAAgentState) -> dict[str, Any]:
    proposed = state.get("proposed_result") or {}

    try:
        result = postprocess_result(DiagnosisResult.model_validate(proposed))
    except ValidationError as exc:
        return {"validation_errors": [f"Diagnosis schema validation failed: {exc}"]}

    errors = validate_result(
        result=result,
        graph_pack=state.get("graph_pack") or {},
        workflow_state=state,
    )

    if errors:
        return {"validation_errors": errors}

    return {
        "validation_errors": [],
        "final_result": result.model_dump(),
    }


def repair_diagnosis_node(state: RCAAgentState) -> dict[str, Any]:
    attempts = int(state.get("attempts") or 0) + 1

    try:
        raw = invoke_json(repair_prompt(state))
        result = postprocess_result(DiagnosisResult.model_validate(raw))
        return {
            "attempts": attempts,
            "proposed_result": result.model_dump(),
        }
    except (ValueError, ValidationError, RuntimeError) as exc:
        return {
            "attempts": attempts,
            "validation_errors": [f"Diagnosis repair failed: {exc}"],
        }


def fallback_node(state: RCAAgentState) -> dict[str, Any]:
    result = postprocess_result(fallback_result(state))
    return {
        "validation_errors": [],
        "final_result": result.model_dump(),
        "proposed_result": result.model_dump(),
    }


def validation_route(state: RCAAgentState) -> str:
    if not state.get("validation_errors"):
        return "done"

    if int(state.get("attempts") or 0) < int(state.get("max_attempts") or 2):
        return "repair"

    return "fallback"
