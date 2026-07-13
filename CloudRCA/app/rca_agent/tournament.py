from __future__ import annotations

from typing import Any

from app.rca_agent.candidate_table import (
    candidate_by_id,
    default_winner,
    same_failure_type,
    same_family,
)


def _override_allowed(default: dict[str, Any], selected: dict[str, Any]) -> tuple[bool, str]:
    if selected.get("root_selectable") is not True:
        return False, "selected candidate is not root-selectable"

    if selected.get("selection_class") == "context_only":
        return False, "selected candidate is context-only"

    if same_family(default, selected):
        return True, "selected candidate is in the default winner's root family"

    if same_failure_type(default, selected):
        return False, "same failure type; graph rank decides between similar candidates"

    default_strong = default.get("selection_class") == "strong_root"
    default_paths = int(default.get("causal_path_count") or 0)
    selected_strength = int(selected.get("evidence_strength") or 0)
    default_strength = int(default.get("evidence_strength") or 0)

    if default_strong and default_paths > 0:
        return False, "default winner is a strong root with causal paths"

    if selected_strength >= default_strength + 20 and int(selected.get("causal_path_count") or 0) >= default_paths:
        return True, "selected candidate has substantially stronger direct evidence"

    return False, "lower-ranked candidate did not justify overriding graph rank"


def adjudicate_tournament(
    candidate_table: list[dict[str, Any]],
    llm_selected_candidate_id: str | None,
    llm_reasoning: str = "",
) -> dict[str, Any]:
    default = default_winner(candidate_table)

    if not default:
        return {
            "winner_candidate_id": "",
            "winner": {},
            "default_candidate_id": "",
            "llm_selected_candidate_id": llm_selected_candidate_id or "",
            "override_accepted": False,
            "override_rejected_reason": "no candidates available",
            "decision_rule": "no candidate",
            "llm_reasoning": llm_reasoning,
        }

    selected = candidate_by_id(candidate_table, llm_selected_candidate_id)

    if selected is None or selected.get("id") == default.get("id"):
        return {
            "winner_candidate_id": default.get("id"),
            "winner": default,
            "default_candidate_id": default.get("id"),
            "llm_selected_candidate_id": llm_selected_candidate_id or "",
            "override_accepted": False,
            "override_rejected_reason": "",
            "decision_rule": "graph-ranked default winner",
            "llm_reasoning": llm_reasoning,
        }

    allowed, reason = _override_allowed(default, selected)
    winner = selected if allowed else default

    return {
        "winner_candidate_id": winner.get("id"),
        "winner": winner,
        "default_candidate_id": default.get("id"),
        "llm_selected_candidate_id": selected.get("id"),
        "override_accepted": allowed,
        "override_rejected_reason": "" if allowed else reason,
        "decision_rule": "accepted LLM override" if allowed else "rejected LLM override",
        "llm_reasoning": llm_reasoning,
    }
