from __future__ import annotations

from typing import Any


CHAOS_KINDS = {
    "NetworkChaos",
    "PodChaos",
    "StressChaos",
    "JVMChaos",
    "Schedule",
}

EVIDENCE_STRENGTH = {
    "active_mutation": 96,
    "active_configuration": 92,
    "network_policy": 90,
    "namespace_policy": 88,
    "scheduling_constraint": 86,
    "endpoint_misconfiguration": 84,
    "rollout_failure": 82,
    "autoscaling_policy": 80,
    "stateful_backend": 76,
    "traffic_configuration": 74,
    "metric_anomaly": 45,
    "inactive_configuration": 35,
    "symptom": 25,
    "context": 10,
}


def _signals(row: dict[str, Any]) -> set[str]:
    return {str(signal) for signal in row.get("signals") or []}


def _evidence_type(row: dict[str, Any]) -> str:
    kind = str(row.get("kind") or "")
    signals = _signals(row)
    hypothesis = str(row.get("best_hypothesis") or "")

    if "active mutation event" in signals or kind in CHAOS_KINDS:
        return "active_mutation"

    if "active configuration content signal" in signals:
        return "active_configuration"

    if kind == "NetworkPolicy" or "network policy" in signals:
        return "network_policy"

    if kind in {"Namespace", "ResourceQuota", "LimitRange"} and (
        "namespace resource policy enforcement" in signals
        or "resource saturation or quota" in signals
        or hypothesis == "namespace_resource_policy"
    ):
        return "namespace_policy"

    if "scheduling constraint failure" in signals:
        return "scheduling_constraint"

    if "workload endpoint misconfiguration" in signals:
        return "endpoint_misconfiguration"

    if "workload rollout failure" in signals:
        return "rollout_failure"

    if kind == "HorizontalPodAutoscaler" or "autoscaling policy" in signals:
        return "autoscaling_policy"

    if "stateful backend failure" in signals:
        return "stateful_backend"

    if "high traffic volume configuration" in signals:
        return "traffic_configuration"

    if "inactive configuration context" in signals:
        return "inactive_configuration"

    if "metric anomaly or error signal" in signals:
        return "metric_anomaly"

    if row.get("root_selectable") is True:
        return "symptom"

    return "context"


def _blockers(row: dict[str, Any], evidence_type: str) -> list[str]:
    blockers = []

    if row.get("root_selectable") is not True:
        blockers.append("not root-selectable")

    if row.get("selection_class") == "context_only":
        blockers.append("context-only")

    if evidence_type == "inactive_configuration":
        blockers.append("inactive configuration")

    if row.get("why_not_root"):
        blockers.append(str(row["why_not_root"]))

    return blockers


def _candidate_family(row: dict[str, Any]) -> dict[str, Any]:
    family = row.get("candidate_family")
    return family if isinstance(family, dict) else {}


def build_candidate_table(graph_pack: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []

    for row in (graph_pack.get("candidate_dossiers") or [])[:limit]:
        if not isinstance(row, dict):
            continue

        evidence_type = _evidence_type(row)
        family = _candidate_family(row)
        signals = list(row.get("signals") or [])

        table.append(
            {
                "rank": int(row.get("rank") or len(table) + 1),
                "id": row.get("id"),
                "kind": row.get("kind"),
                "name": row.get("name"),
                "namespace": row.get("namespace"),
                "score": int(row.get("score") or 0),
                "selection_class": row.get("selection_class"),
                "selection_reason": row.get("selection_reason"),
                "root_selectable": row.get("root_selectable") is True,
                "evidence_type": evidence_type,
                "evidence_strength": EVIDENCE_STRENGTH.get(evidence_type, 0),
                "affected_symptom_count": int(row.get("affected_symptom_count") or 0),
                "causal_path_count": int(row.get("causal_path_count") or 0),
                "best_hypothesis": row.get("best_hypothesis"),
                "signals": signals[:10],
                "reasons": list(row.get("reasons") or [])[:8],
                "supporting_evidence": list(row.get("supporting_evidence") or [])[:6],
                "context_details": list(row.get("context_details") or [])[:6],
                "evidence_paths": list(row.get("evidence_paths") or [])[:8],
                "causal_paths": list(row.get("causal_paths") or [])[:5],
                "caution": row.get("caution"),
                "why_not_root": row.get("why_not_root"),
                "family_id": family.get("id"),
                "family_kind": family.get("kind"),
                "family_members": list(family.get("members") or [])[:12],
                "blockers": _blockers(row, evidence_type),
            }
        )

    return table


def selectable_candidates(candidate_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in candidate_table
        if row.get("root_selectable") is True and row.get("selection_class") != "context_only"
    ]


def default_winner(candidate_table: list[dict[str, Any]]) -> dict[str, Any] | None:
    selectable = selectable_candidates(candidate_table)
    return selectable[0] if selectable else (candidate_table[0] if candidate_table else None)


def candidate_by_id(candidate_table: list[dict[str, Any]], candidate_id: str | None) -> dict[str, Any] | None:
    if not candidate_id:
        return None

    for candidate in candidate_table:
        if candidate.get("id") == candidate_id:
            return candidate

    return None


def same_family(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if not left or not right:
        return False

    left_family = str(left.get("family_id") or "")
    right_family = str(right.get("family_id") or "")

    return bool(left_family and right_family and left_family == right_family)


def same_failure_type(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if not left or not right:
        return False

    return str(left.get("evidence_type") or "") == str(right.get("evidence_type") or "")
