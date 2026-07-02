from __future__ import annotations

from collections import deque
import re
from typing import Any

from app.graph.models import GraphNode, IncidentGraph
from app.graph.parsers import CHAOS_KINDS, CONTROL_KINDS, infer_workload_base, is_observability_name


SYSTEM_INFRA_TERMS = {
    "coredns",
    "kube-",
    "clickhouse",
    "operator",
    "autoscaler",
    "persistentvolume",
    "persistentvolumes",
    "storageclass",
    "volumeattachment",
    "metrics-server",
    "cert-manager",
    "otel-collector",
    "opentelemetry-collector",
    "prometheus",
    "grafana",
}


def normalize_name(value: str) -> str:
    value = str(value).lower().strip()
    value = value.replace("_", "-")
    value = re.sub(r"[^a-z0-9-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-")


def node_to_dict(node: GraphNode) -> dict[str, Any]:
    return {
        "id": node.key.id,
        "kind": node.kind,
        "name": node.name,
        "namespace": node.namespace,
        "affected_score": node.affected_score,
        "signals": sorted(node.signals),
        "reasons": sorted(node.reasons),
        "hypothesis_tags": sorted(node.hypothesis_scores.keys()),
        "evidence_paths": node.evidence_paths[:10],
    }


def is_system_infra_node(node: GraphNode) -> bool:
    lower = node.name.lower()

    return is_observability_name(node.name) or any(term in lower for term in SYSTEM_INFRA_TERMS)


def has_config_fault_signal(node: GraphNode) -> bool:
    text = " ".join(
        [
            node.name,
            " ".join(node.signals),
            " ".join(node.reasons),
            " ".join(node.hypothesis_scores.keys()),
            " ".join(node.evidence_paths),
        ]
    ).lower()

    return any(
        term in text
        for term in [
            "feature",
            "flag",
            "defaultvariant",
            "variant",
            "failure",
            "toggle",
            "rollout",
            "configuration",
            "config",
            "secret",
            "certificate",
            "tls",
            "timeout",
            "endpoint",
        ]
    )


def has_fault_signal(node: GraphNode) -> bool:
    if node.affected_score > 0:
        return True

    if node.hypothesis_scores:
        return True

    text = " ".join(node.signals).lower() + " " + " ".join(node.reasons).lower()

    return any(
        term in text
        for term in [
            "alert",
            "error",
            "failure",
            "timeout",
            "latency",
            "crash",
            "restart",
            "resource",
            "network",
            "configuration",
            "chaos",
        ]
    )


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
        ),
        reverse=True,
    )

    return [node_to_dict(node) for node in nodes[:limit]]


def get_control_objects(graph: IncidentGraph, limit: int = 50) -> list[dict[str, Any]]:
    nodes = [
        node
        for node in graph.nodes.values()
        if node.kind in CONTROL_KINDS and not is_system_infra_node(node)
    ]

    nodes.sort(
        key=lambda node: (
            node.kind in set(CHAOS_KINDS.values()),
            has_config_fault_signal(node),
            has_fault_signal(node),
            len(node.evidence_paths),
        ),
        reverse=True,
    )

    return [node_to_dict(node) for node in nodes[:limit]]


def _attention_score(graph: IncidentGraph, node: GraphNode) -> int:
    """
    This is not a root-cause score.

    It is only used to order nodes that the agent should inspect.
    """
    score = 0

    if node.kind in set(CHAOS_KINDS.values()):
        score += 1000

    if node.kind in {"ConfigMap", "Secret"} and has_config_fault_signal(node):
        score += 900

    if node.kind in {"Deployment", "StatefulSet", "DaemonSet", "Pod"} and has_fault_signal(node):
        score += 500

    if node.kind == "NetworkPolicy" and has_fault_signal(node):
        score += 500

    if node.kind == "Namespace":
        text = " ".join(node.signals).lower() + " " + " ".join(node.reasons).lower()
        if "namespace-level resource policy" in text or "quota" in text or "limit" in text:
            score += 1200
        elif has_fault_signal(node):
            score += 300

    if node.affected_score > 0:
        score += min(node.affected_score, 300)

    score += min(len(node.evidence_paths) * 10, 150)

    # Symptom-heavy Services should be inspected, but not over-promoted.
    if node.kind == "Service":
        score -= 250

    if is_system_infra_node(node):
        score -= 1000

    return score


def get_hypothesis_seeds(graph: IncidentGraph, limit: int = 30) -> list[dict[str, Any]]:
    """
    Return nodes worth investigating.

    These are not final root-cause predictions.
    The agent must compare evidence and decide.
    """
    scored: list[tuple[int, GraphNode]] = []

    for node in graph.nodes.values():
        score = _attention_score(graph, node)

        if score <= 0:
            continue

        scored.append((score, node))

    scored.sort(key=lambda item: item[0], reverse=True)

    result: list[dict[str, Any]] = []

    for score, node in scored[:limit]:
        row = node_to_dict(node)
        row["attention_score"] = score
        row["note"] = "hypothesis seed only; not final root cause"
        result.append(row)

    return result


def get_edges(graph: IncidentGraph, limit: int = 100) -> list[dict[str, Any]]:
    rows = []

    seen = set()

    for edge in graph.edges:
        key = (edge.source, edge.target, edge.relation)

        if key in seen:
            continue

        seen.add(key)

        rows.append(
            {
                "source": edge.source,
                "target": edge.target,
                "relation": edge.relation,
                "evidence_path": edge.evidence_path,
            }
        )

        if len(rows) >= limit:
            break

    return rows


def find_nodes_by_name(graph: IncidentGraph, name: str) -> list[GraphNode]:
    target = normalize_name(name)

    if not target:
        return []

    matches = []

    for node in graph.nodes.values():
        candidate = normalize_name(node.name)

        if candidate == target or candidate in target or target in candidate:
            matches.append(node)

    return matches


def get_paths_to_symptoms(
    graph: IncidentGraph,
    source_node_id: str,
    max_depth: int = 4,
    max_paths: int = 10,
) -> list[list[dict[str, Any]]]:
    symptom_ids = {
        node.key.id
        for node in graph.nodes.values()
        if node.affected_score > 0 and not is_system_infra_node(node)
    }

    adjacency: dict[str, list[tuple[str, str, str | None]]] = {}

    for edge in graph.edges:
        adjacency.setdefault(edge.source, []).append(
            (edge.target, edge.relation, edge.evidence_path)
        )

    queue = deque([(source_node_id, [])])
    visited = {(source_node_id, 0)}
    paths: list[list[dict[str, Any]]] = []

    while queue and len(paths) < max_paths:
        node_id, path = queue.popleft()

        if len(path) >= max_depth:
            continue

        for next_id, relation, evidence_path in adjacency.get(node_id, []):
            next_path = path + [
                {
                    "source": node_id,
                    "target": next_id,
                    "relation": relation,
                    "evidence_path": evidence_path,
                }
            ]

            if next_id in symptom_ids:
                paths.append(next_path)

                if len(paths) >= max_paths:
                    break

            state = (next_id, len(next_path))

            if state not in visited:
                visited.add(state)
                queue.append((next_id, next_path))

    return paths


def build_agent_graph_pack(graph: IncidentGraph) -> dict[str, Any]:
    symptoms = get_symptoms(graph)
    controls = get_control_objects(graph)
    seeds = get_hypothesis_seeds(graph)
    edges = get_edges(graph)

    seed_paths = {}

    for seed in seeds[:15]:
        seed_paths[seed["id"]] = get_paths_to_symptoms(graph, seed["id"])

    return {
        "scenario_path": str(graph.scenario_path),
        "files_seen": graph.files_seen,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "symptoms": symptoms,
        "control_objects": controls,
        "hypothesis_seeds": seeds,
        "important_edges": edges,
        "paths_from_seeds_to_symptoms": seed_paths,
        "instruction": (
            "This graph pack is evidence for an RCA agent. "
            "Hypothesis seeds are not final answers. "
            "The agent must compare supporting and refuting evidence before selecting root cause."
        ),
    }
