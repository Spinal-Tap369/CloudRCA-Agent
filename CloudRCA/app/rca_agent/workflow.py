from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from app.rca_agent.nodes import (
    build_case_node,
    causal_path_audit_node,
    classify_pattern_node,
    confidence_audit_node,
    contradiction_audit_node,
    draft_diagnosis_node,
    fallback_node,
    inspect_primary_node,
    inspect_rivals_node,
    normalize_candidates_node,
    propagation_audit_node,
    repair_diagnosis_node,
    tournament_node,
    validate_diagnosis_node,
    validation_route,
)
from app.rca_agent.state import RCAAgentState
from app.schemas import DiagnosisResult


@lru_cache(maxsize=1)
def build_rca_workflow():
    workflow = StateGraph(RCAAgentState)

    workflow.add_node("build_case", build_case_node)
    workflow.add_node("normalize_candidates", normalize_candidates_node)
    workflow.add_node("classify_pattern", classify_pattern_node)
    workflow.add_node("inspect_primary", inspect_primary_node)
    workflow.add_node("inspect_rivals", inspect_rivals_node)
    workflow.add_node("tournament", tournament_node)
    workflow.add_node("causal_path_audit", causal_path_audit_node)
    workflow.add_node("propagation_audit", propagation_audit_node)
    workflow.add_node("contradiction_audit", contradiction_audit_node)
    workflow.add_node("confidence_audit", confidence_audit_node)
    workflow.add_node("draft_diagnosis", draft_diagnosis_node)
    workflow.add_node("validate_diagnosis", validate_diagnosis_node)
    workflow.add_node("repair_diagnosis", repair_diagnosis_node)
    workflow.add_node("fallback", fallback_node)

    workflow.add_edge(START, "build_case")
    workflow.add_edge("build_case", "normalize_candidates")
    workflow.add_edge("normalize_candidates", "classify_pattern")
    workflow.add_edge("classify_pattern", "inspect_primary")
    workflow.add_edge("inspect_primary", "inspect_rivals")
    workflow.add_edge("inspect_rivals", "tournament")
    workflow.add_edge("tournament", "causal_path_audit")
    workflow.add_edge("causal_path_audit", "propagation_audit")
    workflow.add_edge("propagation_audit", "contradiction_audit")
    workflow.add_edge("contradiction_audit", "confidence_audit")
    workflow.add_edge("confidence_audit", "draft_diagnosis")
    workflow.add_edge("draft_diagnosis", "validate_diagnosis")
    workflow.add_conditional_edges(
        "validate_diagnosis",
        validation_route,
        {
            "done": END,
            "repair": "repair_diagnosis",
            "fallback": "fallback",
        },
    )
    workflow.add_edge("repair_diagnosis", "validate_diagnosis")
    workflow.add_edge("fallback", END)

    return workflow.compile()


def diagnose_scenario_with_workflow(scenario_dir: str | Path) -> DiagnosisResult:
    result_state = build_rca_workflow().invoke(
        {
            "scenario_dir": str(scenario_dir),
            "attempts": 0,
            "max_attempts": 2,
        }
    )

    final_result = result_state.get("final_result")

    if not final_result:
        raise RuntimeError(
            "LangGraph RCA workflow finished without a final_result. "
            f"Validation errors: {result_state.get('validation_errors')}"
        )

    return DiagnosisResult.model_validate(final_result)
