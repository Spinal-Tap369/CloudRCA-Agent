from __future__ import annotations

import re
from typing import Any

from app.graph.candidates import build_candidate_dossiers, get_paths_to_symptoms
from app.graph.records import GraphNode, IncidentGraph
from app.graph.scoring import is_system_infra_node
from app.graph.parsers import CONTROL_KINDS


ROOT_CAPABLE_KINDS = {
    "ConfigMap",
    "Secret",
    "Deployment",
    "StatefulSet",
    "DaemonSet",
    "Pod",
    "Service",
    "NetworkPolicy",
    "Namespace",
    "ResourceQuota",
    "LimitRange",
    "HorizontalPodAutoscaler",
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


def normalize_name(value: str) -> str:
    value = str(value).lower().strip().replace("_", "-")
    value = re.sub(r"[^a-z0-9-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-")


def node_to_dict(node: GraphNode) -> dict[str, Any]:
    attributes: dict[str, Any] = {}

    if "configuration_findings" in node.attributes:
        attributes["configuration_findings"] = node.attributes["configuration_findings"]

    if "active_configuration_findings" in node.attributes:
        attributes["active_configuration_findings"] = node.attributes["active_configuration_findings"]

    if "inactive_configuration_context" in node.attributes:
        attributes["inactive_configuration_context"] = node.attributes["inactive_configuration_context"]

    if "configuration_context" in node.attributes:
        attributes["configuration_context"] = node.attributes["configuration_context"]

    if "traffic_user_count" in node.attributes:
        attributes["traffic_user_count"] = node.attributes["traffic_user_count"]

    return {
        "id": node.key.id,
        "kind": node.kind,
        "name": node.name,
        "namespace": node.namespace,
        "affected_score": node.affected_score,
        "candidate_score": node.candidate_score,
        "signals": sorted(node.signals),
        "reasons": sorted(node.reasons),
        "hypothesis_tags": sorted(node.hypothesis_scores.keys()),
        "attributes": attributes,
        "evidence_paths": node.evidence_paths[:12],
        "evidence": [
            {
                "path": item.path,
                "category": item.category,
                "summary": item.summary,
                "timestamp": item.timestamp,
            }
            for item in node.evidence[:8]
        ],
    }


def get_symptoms(graph: IncidentGraph, limit: int = 25) -> list[dict[str, Any]]:
    nodes = [
        node
        for node in graph.nodes.values()
        if node.affected_score > 0 and not is_system_infra_node(node)
    ]
    nodes.sort(
        key=lambda node: (
            node.affected_score,
            len(node.evidence_paths),
            node.candidate_score,
        ),
        reverse=True,
    )
    return [node_to_dict(node) for node in nodes[:limit]]


def get_control_objects(graph: IncidentGraph, limit: int = 60) -> list[dict[str, Any]]:
    nodes = [
        node
        for node in graph.nodes.values()
        if node.kind in CONTROL_KINDS or node.kind in ROOT_CAPABLE_KINDS
    ]
    nodes = [node for node in nodes if not is_system_infra_node(node)]
    nodes.sort(
        key=lambda node: (
            node.candidate_score,
            len(node.hypothesis_scores),
            len(node.evidence_paths),
        ),
        reverse=True,
    )
    return [node_to_dict(node) for node in nodes[:limit]]


def get_hypothesis_seeds(graph: IncidentGraph, limit: int = 30) -> list[dict[str, Any]]:
    candidates = graph.candidates[:limit] if graph.candidates else []
    candidate_ids = {candidate.node_id for candidate in candidates}
    rows = []

    for candidate in candidates:
        node = graph.nodes.get(candidate.node_id)

        if node is None:
            continue

        row = node_to_dict(node)
        row["attention_score"] = candidate.score
        row["candidate_rank"] = len(rows) + 1
        row["best_hypothesis"] = candidate.best_hypothesis
        row["note"] = "candidate for investigation; not a final diagnosis"
        rows.append(row)

    if len(rows) >= limit:
        return rows

    fallback = [
        node
        for node in graph.nodes.values()
        if node.key.id not in candidate_ids
        and node.candidate_score > 0
        and not is_system_infra_node(node)
    ]
    fallback.sort(key=lambda node: node.candidate_score, reverse=True)

    for node in fallback[: limit - len(rows)]:
        row = node_to_dict(node)
        row["attention_score"] = node.candidate_score
        row["candidate_rank"] = len(rows) + 1
        row["best_hypothesis"] = node.best_hypothesis
        row["note"] = "candidate for investigation; not a final diagnosis"
        rows.append(row)

    return rows


def get_edges(graph: IncidentGraph, limit: int = 120) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    sorted_edges = sorted(
        graph.edges,
        key=lambda edge: (
            edge.confidence,
            edge.relation not in {"same-log-context"},
            edge.evidence_path or "",
        ),
        reverse=True,
    )

    for edge in sorted_edges:
        key = (edge.source, edge.target, edge.relation)

        if key in seen:
            continue

        seen.add(key)
        rows.append(
            {
                "source": edge.source,
                "target": edge.target,
                "relation": edge.relation,
                "confidence": edge.confidence,
                "evidence_path": edge.evidence_path,
                "summary": edge.summary,
            }
        )

        if len(rows) >= limit:
            break

    return rows


def _generated_name_match(candidate: str, target: str) -> bool:
    if candidate == target:
        return True

    if len(target) >= 4 and candidate.startswith(f"{target}-"):
        return True

    if len(candidate) >= 4 and target.startswith(f"{candidate}-"):
        return True

    return False


def find_nodes_by_name(graph: IncidentGraph, name: str) -> list[GraphNode]:
    target = normalize_name(name)

    if not target:
        return []

    matches = []

    for node in graph.nodes.values():
        candidate = normalize_name(node.name)

        if _generated_name_match(candidate, target):
            matches.append(node)

    matches.sort(key=lambda node: (node.candidate_score, node.affected_score), reverse=True)
    return matches


def build_agent_graph_pack(graph: IncidentGraph) -> dict[str, Any]:
    symptoms = get_symptoms(graph)
    controls = get_control_objects(graph)
    seeds = get_hypothesis_seeds(graph)
    edges = get_edges(graph)
    dossiers = build_candidate_dossiers(graph, limit=60)
    seed_paths = {}

    for seed in seeds[:15]:
        seed_paths[seed["id"]] = get_paths_to_symptoms(graph, seed["id"], max_depth=5, max_paths=6)

    return {
        "scenario_path": str(graph.scenario_path),
        "files_seen": graph.files_seen,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "symptoms": symptoms,
        "control_objects": controls,
        "hypothesis_seeds": seeds,
        "root_candidates": dossiers,
        "candidate_dossiers": dossiers,
        "important_edges": edges,
        "paths_from_seeds_to_symptoms": seed_paths,
        "instruction": (
            "Use candidate dossiers as the primary RCA guide. "
            "Validate each candidate against typed causal paths, supporting evidence, "
            "and symptom-only cautions before selecting a root cause."
        ),
    }
