from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.graph.ingest import has_any, iter_tsv, parse_json, relative_path
from app.graph.k8s_inventory import mark_from_text
from app.graph.records import IncidentGraph
from app.graph.relationships import add_edge, add_hypothesis, ensure_node
from app.graph.parsers import normalize_kind


CHAOS_OR_MUTATION_KINDS = {
    "NetworkChaos",
    "PodChaos",
    "StressChaos",
    "DNSChaos",
    "HTTPChaos",
    "IOChaos",
    "TimeChaos",
    "JVMChaos",
    "Schedule",
}

NAMESPACE_POLICY_TERMS = {
    "exceeded quota",
    "resource quota",
    "resourcequota",
    "limitrange",
    "limit range",
    "forbidden",
    "insufficient memory",
    "insufficient cpu",
    "unschedulable",
}

POLICY_EVENT_REASONS = {
    "failedcreate",
    "failedscheduling",
    "failed",
}

SCHEDULING_CONSTRAINT_TERMS = {
    "node selector",
    "node(s) didn't match",
    "node affinity",
    "pod affinity",
    "taint",
    "toleration",
    "unschedulable",
}


def _event_object(body: dict[str, Any]) -> dict[str, Any] | None:
    obj = body.get("object") if isinstance(body.get("object"), dict) else body
    return obj if isinstance(obj, dict) else None


def _event_involved_object(event: dict[str, Any]) -> dict[str, Any]:
    involved = event.get("regarding") or event.get("involvedObject") or {}
    return involved if isinstance(involved, dict) else {}


def _resource_quota_names(text: str) -> set[str]:
    names = set()

    for match in re.findall(r"\b(?:exceeded quota|quota):\s*([a-z0-9][a-z0-9.-]+)", text.lower()):
        names.add(match.rstrip(".,;"))

    return names


def _mark_namespace_policy_event(
    graph: IncidentGraph,
    target_node_id: str,
    namespace: str,
    reason: str,
    note: str,
    text: str,
    rel_path: str,
    timestamp: str | None,
) -> None:
    reason_key = reason.lower()
    lower_text = text.lower()

    if not namespace:
        return

    if reason_key not in POLICY_EVENT_REASONS and not has_any(lower_text, NAMESPACE_POLICY_TERMS):
        return

    if not has_any(lower_text, NAMESPACE_POLICY_TERMS):
        return

    namespace_node = ensure_node(
        graph,
        kind="Namespace",
        name=namespace,
        namespace="",
        evidence_path=rel_path,
        category="events",
        summary=f"Namespace policy event {reason}: {note[:220]}",
        timestamp=timestamp,
    )
    namespace_node.signals.add("namespace resource policy enforcement")
    namespace_node.signals.add("resource saturation or quota")
    add_hypothesis(
        namespace_node,
        "namespace_resource_policy",
        1500,
        "namespace resource policy enforcement event",
    )
    add_edge(
        graph,
        source=namespace_node.key.id,
        target=target_node_id,
        relation="namespace-policy-affects-workload",
        evidence_path=rel_path,
        confidence=0.95,
        summary=f"Namespace policy event affects workload: {reason}",
    )

    for quota_name in _resource_quota_names(text):
        quota_node = ensure_node(
            graph,
            kind="ResourceQuota",
            name=quota_name,
            namespace=namespace,
            evidence_path=rel_path,
            category="events",
            summary=f"ResourceQuota enforcement event {reason}: {note[:220]}",
            timestamp=timestamp,
        )
        quota_node.signals.add("namespace-level resource policy")
        quota_node.signals.add("resource saturation or quota")
        add_hypothesis(
            quota_node,
            "namespace_resource_policy",
            1300,
            "quota enforcement event",
        )
        add_edge(
            graph,
            source=namespace_node.key.id,
            target=quota_node.key.id,
            relation="namespace-defines-resource-policy",
            evidence_path=rel_path,
            confidence=0.9,
            summary=f"Namespace policy includes ResourceQuota {quota_name}",
        )
        add_edge(
            graph,
            source=quota_node.key.id,
            target=target_node_id,
            relation="quota-blocks-workload",
            evidence_path=rel_path,
            confidence=0.98,
            summary=f"ResourceQuota {quota_name} blocked workload creation",
        )


def ingest_k8s_events(graph: IncidentGraph, scenario_path: Path) -> None:
    path = scenario_path / "k8s_events_raw.tsv"
    rel_path = relative_path(scenario_path, path)

    for row in iter_tsv(path) or []:
        body = parse_json(row.get("Body", ""))

        if not isinstance(body, dict):
            continue

        event = _event_object(body)

        if not event:
            continue

        involved = _event_involved_object(event)
        kind = normalize_kind(str(involved.get("kind") or "Event"))
        name = str(involved.get("name") or event.get("metadata", {}).get("name") or "")
        namespace = str(involved.get("namespace") or event.get("metadata", {}).get("namespace") or "")
        reason = str(event.get("reason") or "")
        note = str(event.get("note") or event.get("message") or "")
        timestamp = row.get("TimestampTime") or row.get("Timestamp")

        if not name:
            continue

        node = ensure_node(
            graph,
            kind=kind,
            name=name,
            namespace=namespace,
            evidence_path=rel_path,
            category="events",
            summary=f"Kubernetes event {reason}: {note[:220]}",
            timestamp=timestamp,
        )
        text = f"{reason} {note} {json.dumps(event)[:3000]}"
        node.reasons.add(f"kubernetes event reason: {reason}" if reason else "kubernetes event")

        if kind in CHAOS_OR_MUTATION_KINDS:
            node.signals.add("chaos or mutation object")
            add_hypothesis(node, kind.lower(), 1200, "Chaos Mesh or scheduled mutation event")

        if reason.lower() in {"failed", "backoff", "failedcreate", "failedscheduling"}:
            node.affected_score += 40

        if reason.lower() in {"started", "applied", "spawned", "updated", "finalizerinited"}:
            if kind in CHAOS_OR_MUTATION_KINDS:
                node.signals.add("active mutation event")

        mark_from_text(node, text)

        if reason.lower() == "failedscheduling" and has_any(note, SCHEDULING_CONSTRAINT_TERMS):
            node.signals.add("scheduling constraint failure")
            add_hypothesis(
                node,
                "scheduling_constraint",
                1200,
                "failed scheduling constraint event",
            )

        _mark_namespace_policy_event(
            graph=graph,
            target_node_id=node.key.id,
            namespace=namespace,
            reason=reason,
            note=note,
            text=text,
            rel_path=rel_path,
            timestamp=timestamp,
        )

        if kind == "ReplicaSet":
            match = re.search(r"\b(?:Created|Deleted) pod:\s+([a-z0-9][a-z0-9-]+)", note)

            if match:
                pod_node = ensure_node(
                    graph,
                    kind="Pod",
                    name=match.group(1),
                    namespace=namespace,
                    evidence_path=rel_path,
                    category="events",
                    summary=f"ReplicaSet event references pod {match.group(1)}",
                    timestamp=timestamp,
                )
                add_edge(
                    graph,
                    source=node.key.id,
                    target=pod_node.key.id,
                    relation="owns",
                    evidence_path=rel_path,
                    confidence=0.8,
                )
