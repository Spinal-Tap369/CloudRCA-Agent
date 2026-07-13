from __future__ import annotations

from typing import Any

from app.rca_agent.candidate_table import default_winner, same_failure_type, same_family


def audit_causal_paths(candidate_table: list[dict[str, Any]], winner: dict[str, Any]) -> dict[str, Any]:
    paths = list(winner.get("causal_paths") or [])
    path_count = int(winner.get("causal_path_count") or 0)
    affected_count = int(winner.get("affected_symptom_count") or 0)

    return {
        "candidate_id": winner.get("id"),
        "passes": path_count > 0 or affected_count > 0,
        "causal_path_count": path_count,
        "affected_symptom_count": affected_count,
        "sample_paths": paths[:4],
        "notes": (
            "Winner reaches symptoms through graph paths."
            if path_count > 0
            else "Winner has limited explicit causal paths; rely on direct evidence and nearby symptoms."
        ),
    }


def audit_propagation(candidate_table: list[dict[str, Any]], winner: dict[str, Any]) -> dict[str, Any]:
    paths = [str(path) for path in winner.get("causal_paths") or [] if path]
    services: list[str] = []
    workloads: list[str] = []

    for path in paths:
        for part in path.split(" -> "):
            if part.startswith("Service:"):
                value = part.split(":")[-1]
                if value not in services:
                    services.append(value)
            elif part.startswith(("Deployment:", "Pod:", "StatefulSet:", "DaemonSet:")):
                value = part.split(":")[-1]
                if value not in workloads:
                    workloads.append(value)

    chain = " -> ".join(services[:6]) if services else ""

    return {
        "candidate_id": winner.get("id"),
        "has_paths": bool(paths),
        "affected_services": services[:12],
        "affected_workloads": workloads[:12],
        "representative_chain": chain,
        "sample_paths": paths[:5],
        "guidance": (
            "Describe blast radius using affected services and representative causal paths. "
            "Avoid broad service claims that are not supported by paths."
        ),
    }


def audit_contradictions(candidate_table: list[dict[str, Any]], winner: dict[str, Any]) -> dict[str, Any]:
    default = default_winner(candidate_table)
    warnings: list[str] = []
    blockers: list[str] = []
    rivals: list[dict[str, Any]] = []

    if not winner:
        blockers.append("no tournament winner")
        return {"passes": False, "warnings": warnings, "blockers": blockers, "rivals": rivals}

    if winner.get("selection_class") == "context_only":
        blockers.append("winner is context-only")

    if winner.get("root_selectable") is not True:
        blockers.append("winner is not root-selectable")

    if winner.get("evidence_type") == "inactive_configuration":
        warnings.append("winner is based on inactive configuration context")

    if default and winner.get("id") != default.get("id") and not same_family(default, winner):
        blockers.append("winner differs from graph default outside the same root family")

    for rival in candidate_table[:5]:
        if rival.get("id") == winner.get("id"):
            continue

        if same_family(winner, rival):
            rivals.append(
                {
                    "candidate_id": rival.get("id"),
                    "relationship": "same_root_family",
                    "rank": rival.get("rank"),
                }
            )
            continue

        if same_failure_type(winner, rival) and int(rival.get("rank") or 999) < int(winner.get("rank") or 999):
            blockers.append(
                f"higher-ranked similar-failure rival exists: {rival.get('id')}"
            )

        if int(rival.get("evidence_strength") or 0) > int(winner.get("evidence_strength") or 0) + 15:
            warnings.append(
                f"rival has stronger evidence type: {rival.get('id')}"
            )

    return {
        "passes": not blockers,
        "warnings": warnings,
        "blockers": blockers,
        "rivals": rivals[:6],
    }


def calibrate_confidence(
    winner: dict[str, Any],
    causal_audit: dict[str, Any],
    contradiction_audit: dict[str, Any],
    pattern: dict[str, Any],
) -> dict[str, Any]:
    value = 0.62
    reasons: list[str] = []

    if winner.get("selection_class") == "strong_root":
        value += 0.12
        reasons.append("strong root candidate")
    elif winner.get("selection_class") == "weak_root":
        value += 0.04
        reasons.append("weak root candidate")

    if int(winner.get("rank") or 99) == 1:
        value += 0.08
        reasons.append("ranked first by graph")

    if int(winner.get("evidence_strength") or 0) >= 80:
        value += 0.08
        reasons.append("direct causal evidence type")

    if int(causal_audit.get("causal_path_count") or 0) > 0:
        value += 0.05
        reasons.append("causal paths reach symptoms")

    if contradiction_audit.get("warnings"):
        value -= min(0.08, 0.03 * len(contradiction_audit["warnings"]))
        reasons.append("contradiction warnings")

    if contradiction_audit.get("blockers"):
        value -= 0.25
        reasons.append("contradiction blockers")

    pattern_confidence = float(pattern.get("confidence") or 0.0)
    if pattern_confidence >= 0.85:
        value += 0.03
        reasons.append("incident pattern agrees with top evidence")

    value = max(0.35, min(0.94, value))

    return {
        "candidate_id": winner.get("id"),
        "confidence": round(value, 2),
        "reasons": reasons,
    }
