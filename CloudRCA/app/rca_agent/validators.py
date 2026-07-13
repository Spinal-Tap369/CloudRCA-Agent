from __future__ import annotations

import re
from typing import Any

from app.rca_agent.quality import presentation_violations
from app.schemas import DiagnosisResult, EvidenceItem, RootCauseEntity


ALLOWED_ROOT_KINDS = {
    "Deployment",
    "Pod",
    "Service",
    "ConfigMap",
    "Secret",
    "Certificate",
    "Node",
    "NetworkPolicy",
    "Namespace",
    "ResourceQuota",
    "LimitRange",
    "NetworkChaos",
    "PodChaos",
    "StressChaos",
    "JVMChaos",
    "Schedule",
    "HorizontalPodAutoscaler",
    "Unknown",
}


def coerce_root_kind(kind: str | None) -> str:
    if not kind:
        return "Unknown"

    raw = str(kind).strip()
    aliases = {
        "HPA": "HorizontalPodAutoscaler",
        "Network Policy": "NetworkPolicy",
        "Resource Quota": "ResourceQuota",
        "Limit Range": "LimitRange",
    }

    if raw in aliases:
        return aliases[raw]

    for allowed in ALLOWED_ROOT_KINDS:
        if raw.lower() == allowed.lower():
            return allowed

    return raw


def normalize_token(value: str) -> str:
    value = str(value).lower().strip()
    value = value.replace("_", "-")
    value = re.sub(r"[^a-z0-9-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-")


def workload_base(name: str) -> str:
    name = normalize_token(name)
    name = re.sub(r"-[a-f0-9]{8,10}-[a-z0-9]{4,6}$", "", name)
    name = re.sub(r"-[a-z0-9]{5,10}-[a-z0-9]{4,6}$", "", name)
    return name


def candidate_rows(graph_pack: dict[str, Any]) -> list[dict[str, Any]]:
    contract = graph_pack.get("candidate_contract")

    if isinstance(contract, dict) and isinstance(contract.get("selectable_root_entities"), list):
        return [
            row
            for row in contract["selectable_root_entities"]
            if isinstance(row, dict)
        ]

    return [
        row
        for row in graph_pack.get("candidate_dossiers", []) or []
        if isinstance(row, dict) and row.get("root_selectable") is True
    ]


def namespace_matches(entity_namespace: str | None, candidate_namespace: Any) -> bool:
    entity_ns = str(entity_namespace or "").strip()
    candidate_ns = str(candidate_namespace or "").strip()
    return not entity_ns or not candidate_ns or entity_ns == candidate_ns


def name_matches_candidate(entity_name: str, candidate_name: Any) -> bool:
    entity = normalize_token(entity_name)
    candidate = normalize_token(str(candidate_name or ""))

    if not entity or not candidate:
        return False

    if entity == candidate:
        return True

    if len(entity) >= 4 and candidate.startswith(f"{entity}-"):
        return True

    if len(candidate) >= 4 and entity.startswith(f"{candidate}-"):
        return True

    entity_base = workload_base(entity)
    candidate_base = workload_base(candidate)
    return bool(entity_base and candidate_base and entity_base == candidate_base)


def entity_matches_candidate(entity: RootCauseEntity, candidate: dict[str, Any]) -> bool:
    entity_kind = coerce_root_kind(entity.kind)
    candidate_kind = coerce_root_kind(str(candidate.get("kind") or ""))

    if entity_kind != candidate_kind:
        return False

    if not namespace_matches(entity.namespace, candidate.get("namespace")):
        return False

    return name_matches_candidate(entity.name, candidate.get("name"))


def matching_candidate(entity: RootCauseEntity, graph_pack: dict[str, Any]) -> dict[str, Any] | None:
    for candidate in candidate_rows(graph_pack):
        if entity_matches_candidate(entity, candidate):
            return candidate

    return None


def _family_id(candidate: dict[str, Any] | None) -> str:
    if not candidate:
        return ""

    direct = str(candidate.get("family_id") or "")

    if direct:
        return direct

    family = candidate.get("candidate_family")

    if isinstance(family, dict):
        return str(family.get("id") or "")

    return ""


def _same_root_family(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    left_family = _family_id(left)
    right_family = _family_id(right)
    return bool(left_family and right_family and left_family == right_family)


def _candidate_id(candidate: dict[str, Any] | None) -> str:
    return str((candidate or {}).get("id") or "")


def _winner_from_state(workflow_state: dict[str, Any] | None) -> dict[str, Any]:
    if not workflow_state:
        return {}

    tournament = workflow_state.get("tournament_result")

    if isinstance(tournament, dict) and isinstance(tournament.get("winner"), dict):
        return tournament["winner"]

    return {}


def postprocess_result(result: DiagnosisResult) -> DiagnosisResult:
    result.should_auto_remediate = False

    for entity in result.root_cause_entities:
        entity.kind = coerce_root_kind(entity.kind)

        if entity.kind not in ALLOWED_ROOT_KINDS:
            entity.kind = "Unknown"

    return result


def validate_result(
    result: DiagnosisResult,
    graph_pack: dict[str, Any],
    workflow_state: dict[str, Any] | None = None,
) -> list[str]:
    candidates = candidate_rows(graph_pack)
    candidates_with_paths = [
        row
        for row in candidates
        if int(row.get("causal_path_count") or 0) > 0
    ]
    violations: list[str] = []

    if not result.root_cause_entities:
        violations.append("No root_cause_entities were returned.")

    if result.should_auto_remediate:
        violations.append("should_auto_remediate must be false.")

    if not result.evidence:
        violations.append("At least one evidence item is required.")

    if not result.recommended_remediation:
        violations.append("At least one recommended remediation is required.")

    violations.extend(presentation_violations(result))

    for evidence in result.evidence:
        if evidence.source_type == "ground_truth_hidden":
            violations.append("Ground truth evidence must not be cited.")

    winner = _winner_from_state(workflow_state)

    for entity in result.root_cause_entities:
        entity.kind = coerce_root_kind(entity.kind)

        if entity.kind == "Unknown":
            if candidates_with_paths:
                violations.append(
                    "Unknown root returned even though graph candidates have causal paths."
                )
            continue

        if entity.kind not in ALLOWED_ROOT_KINDS:
            violations.append(f"Unsupported root kind: {entity.kind}.")
            continue

        candidate = matching_candidate(entity, graph_pack)

        if candidate is None:
            namespace = entity.namespace or ""
            violations.append(
                f"Unsupported root entity {entity.kind}:{namespace}:{entity.name}; "
                "it does not match selectable graph candidates."
            )
            continue

        if candidate.get("root_selectable") is not True:
            violations.append(
                f"Selected root {candidate.get('id')} is not root-selectable."
            )

        if candidate.get("selection_class") == "context_only":
            violations.append(
                f"Selected root {candidate.get('id')} is context-only."
            )

        if winner and _candidate_id(candidate) != _candidate_id(winner):
            if not _same_root_family(candidate, winner):
                violations.append(
                    f"Selected root {_candidate_id(candidate)} does not match audited "
                    f"winner {_candidate_id(winner)} or its root family."
                )

    return violations


def source_type_from_path(path: str | None) -> str:
    lower = str(path or "").lower()

    if "alert" in lower:
        return "alerts"
    if "event" in lower:
        return "events"
    if "log" in lower:
        return "logs"
    if "trace" in lower or "span" in lower:
        return "traces"
    if "metric" in lower:
        return "metrics"

    return "topology"


def fallback_result(state: dict[str, Any]) -> DiagnosisResult:
    graph_pack = state.get("graph_pack") or {}
    tournament = state.get("tournament_result")
    winner = tournament.get("winner") if isinstance(tournament, dict) else None
    candidates = candidate_rows(graph_pack)
    candidate = winner if isinstance(winner, dict) and winner else (candidates[0] if candidates else {})
    symptoms = graph_pack.get("symptoms") or []
    scenario_id = str(state.get("scenario_id") or "unknown")

    if not candidate:
        return DiagnosisResult(
            scenario_id=scenario_id,
            incident_summary="The graph found symptoms but no selectable root candidate.",
            root_cause_entities=[
                RootCauseEntity(kind="Unknown", name="Unknown", namespace=None, confidence=0.3)
            ],
            evidence=[
                EvidenceItem(
                    source_type="topology",
                    source_path=None,
                    summary="No selectable graph root candidate was available.",
                    supports_root_cause=False,
                )
            ],
            reasoning_summary="The deterministic fallback could not select a graph-grounded root cause.",
            recommended_remediation=["Review the graph pack and source evidence manually."],
            should_auto_remediate=False,
            limitations=["No selectable root candidate was available."],
        )

    source_paths = candidate.get("evidence_paths") or []
    support = candidate.get("supporting_evidence") or []
    evidence_items = []

    for index, summary in enumerate(support[:4]):
        source_path = source_paths[index] if index < len(source_paths) else None
        evidence_items.append(
            EvidenceItem(
                source_type=source_type_from_path(source_path),
                source_path=source_path,
                summary=str(summary)[:600],
                supports_root_cause=True,
            )
        )

    if not evidence_items:
        evidence_items.append(
            EvidenceItem(
                source_type="topology",
                source_path=source_paths[0] if source_paths else None,
                summary=(
                    f"Graph candidate {candidate.get('id')} is selectable because "
                    f"{candidate.get('selection_reason') or 'it has causal evidence'}."
                ),
                supports_root_cause=True,
            )
        )

    symptom_names = [
        str(row.get("name") or row.get("id"))
        for row in symptoms[:5]
        if isinstance(row, dict)
    ]
    confidence = 0.86 if candidate.get("selection_class") == "strong_root" else 0.74

    return DiagnosisResult(
        scenario_id=scenario_id,
        incident_summary=(
            "The scenario shows service degradation and Kubernetes/telemetry symptoms"
            + (f" involving {', '.join(symptom_names)}." if symptom_names else ".")
        ),
        root_cause_entities=[
            RootCauseEntity(
                kind=str(candidate.get("kind") or "Unknown"),
                name=str(candidate.get("name") or "Unknown"),
                namespace=str(candidate.get("namespace") or "") or None,
                confidence=confidence,
            )
        ],
        evidence=evidence_items,
        reasoning_summary=(
            f"Deterministic fallback selected graph candidate {candidate.get('id')} "
            f"at rank {candidate.get('rank')} with selection class "
            f"{candidate.get('selection_class')}. "
            f"Reason: {candidate.get('selection_reason') or 'graph candidate evidence'}."
        ),
        recommended_remediation=[
            f"Inspect {candidate.get('kind')}/{candidate.get('name')} and the cited evidence paths before making changes."
        ],
        should_auto_remediate=False,
        limitations=[
            "The LLM workflow could not produce a fully valid graph-grounded diagnosis, so the deterministic graph fallback was used."
        ],
    )
