from __future__ import annotations

import re
from pathlib import Path

from app.graph.io import read_scenario_files
from app.graph.models import EvidenceFile, EvidenceRef, GraphEdge, GraphNode, IncidentGraph, NodeKey
from app.graph.parsers import (
    CHAOS_KINDS,
    CONTROL_KINDS,
    EntityRef,
    extract_entities_from_line,
    extract_entities_from_path,
    infer_workload_base,
    is_observability_name,
)

from app.graph.render import render_graph_context
from app.graph.namespace_policy import apply_namespace_policy_signals


ALERT_TERMS = {
    "requesterrorrate",
    "request error rate",
    "requestlatency",
    "request latency",
    "error rate",
    "latency",
    "cputhrottlinghigh",
    "cpu throttling",
    "alertname",
}

ERROR_TERMS = {
    "error",
    "failed",
    "failure",
    "exception",
    "timeout",
    "deadline_exceeded",
    "unavailable",
    "5xx",
    "500",
    "502",
    "503",
    "504",
}

RESOURCE_TERMS = {
    "cpu throttling",
    "cputhrottlinghigh",
    "oomkilled",
    "out of memory",
    "memory pressure",
    "evicted",
    "resource limit",
    "resourcequota",
    "resource quota",
    "quota",
    "exceeded quota",
    "exceededquotas",
    "forbidden",
    "failedcreate",
    "minimumreplicasunavailable",
    "insufficient memory",
    "memory-stress",
    "cpu-stress",
}

CRASH_TERMS = {
    "crashloopbackoff",
    "crash",
    "restart",
    "restarted",
    "back-off",
    "readiness probe",
    "liveness probe",
    "pod-kill",
    "pod-failure",
}

NETWORK_TERMS = {
    "connection refused",
    "connection reset",
    "networkpolicy",
    "network policy",
    "network-delay",
    "network delay",
    "network-partition",
    "network partition",
    "dns",
    "ingress",
    "egress",
    "unreachable",
}

CONFIG_TERMS = {
    "configmap",
    "config map",
    "secret",
    "feature flag",
    "defaultvariant",
    "configuration",
    "config",
}

TRAFFIC_TERMS = {
    "request volume",
    "high number of requests",
    "high traffic",
    "traffic spike",
    "request spike",
    "requests per second",
    "rps",
    "concurrent users",
    "overload",
    "overloaded",
    "flood",
    "overload",
    "overloaded",
}

CERT_SECRET_TERMS = {
    "certificate",
    "cert",
    "tls",
    "secret",
    "token expired",
}


def _has_any(text: str, terms: set[str]) -> bool:
    lower = text.lower()
    return any(term in lower for term in terms)


def _node_id(kind: str, name: str, namespace: str = "") -> str:
    return NodeKey(kind=kind, name=name, namespace=namespace).id


def _add_node(graph: IncidentGraph, entity: EntityRef, evidence_file: EvidenceFile, reason: str = "") -> GraphNode:
    key = NodeKey(kind=entity.kind or "Unknown", name=entity.name, namespace=entity.namespace or "")
    node = graph.nodes.get(key.id)

    if node is None:
        node = GraphNode(key=key)
        graph.nodes[key.id] = node

    node.evidence.append(
        EvidenceRef(
            path=evidence_file.relative_path,
            category=evidence_file.category,
            summary=reason,
        )
    )

    if reason:
        node.reasons.add(reason)

    return node


def _add_hypothesis(node: GraphNode, name: str, points: int) -> None:
    # Cap repeated evidence from snapshot rows. One object can appear many times
    # in k8s_objects_raw.tsv; repeated inventory lines should not create fake certainty.
    current = node.hypothesis_scores.get(name, 0)
    node.hypothesis_scores[name] = min(current + points, 8000)


def _bump_global(graph: IncidentGraph, name: str, points: int) -> None:
    graph.global_scores[name] = graph.global_scores.get(name, 0) + points


def _is_important_line(line: str, category: str) -> bool:
    lower = line.lower()

    if category in {"alerts", "events", "logs", "traces", "kubernetes_objects"}:
        return True

    return (
        _has_any(lower, ALERT_TERMS)
        or _has_any(lower, ERROR_TERMS)
        or _has_any(lower, RESOURCE_TERMS)
        or _has_any(lower, CRASH_TERMS)
        or _has_any(lower, NETWORK_TERMS)
        or _has_any(lower, CONFIG_TERMS)
        or _has_any(lower, TRAFFIC_TERMS)
        or any(kind.lower() in lower for kind in CHAOS_KINDS.values())
    )


def evidence_category_is_k8s_object(category: str) -> bool:
    return category in {"kubernetes_objects", "structured_text"}


def _has_config_fault_signal(lower: str) -> bool:
    """
    Generic config-content signal.

    This does not hardcode flagd-config, cartFailure, adFailure, or scenario IDs.
    It detects config objects whose values look like feature flags, failure flags,
    rollout switches, timeouts, cert/secret material, or endpoint overrides.
    """
    if any(
        term in lower
        for term in [
            "defaultvariant",
            "feature flag",
            "featureflag",
            "variants",
            "rollout",
            "toggle",
            "enabled",
            "disabled",
            "timeout",
            "endpoint",
            "certificate",
            "tls",
            "token",
            "password",
        ]
    ):
        return True

    # Captures generic flag-style names like cartFailure, adFailure, paymentFailure
    # without hardcoding the service names.
    if re.search(r"[a-z0-9_-]+failure", lower):
        return True

    return False


def _has_namespace_policy_signal(lower: str) -> bool:
    return any(
        term in lower
        for term in [
            "resourcequota",
            "resource quota",
            "limitrange",
            "limit range",
            "quota",
            "exceeded quota",
            "exceededquotas",
            "forbidden",
            "failedcreate",
            "minimumreplicasunavailable",
            "insufficient memory",
            "memory quota",
            "requests.memory",
            "limits.memory",
        ]
    )


def _score_node_from_line(graph: IncidentGraph, node: GraphNode, line: str, category: str) -> None:
    lower = line.lower()

    if category == "alerts" or _has_any(lower, ALERT_TERMS):
        node.affected_score += 20
        node.signals.add("alert/latency/error-rate")
        node.reasons.add("mentioned in alert-like evidence")
        _bump_global(graph, "alerts", 1)

    if _has_any(lower, ERROR_TERMS):
        node.affected_score += 8
        node.signals.add("error/failure/timeout")
        _bump_global(graph, "errors", 1)

    if _has_any(lower, RESOURCE_TERMS):
        node.signals.add("resource saturation")
        _add_hypothesis(node, "resource_saturation", 80 if node.kind in {"Pod", "Deployment", "StressChaos"} else 20)
        _bump_global(graph, "resource_saturation", 2)

    if node.kind in {"ResourceQuota", "LimitRange", "Namespace"} and _has_namespace_policy_signal(lower):
        node.signals.add("namespace-level resource policy")
        node.reasons.add("namespace resource quota/limit evidence")
        _add_hypothesis(node, "namespace_resource_policy", 1200)
        _bump_global(graph, "namespace_resource_policy", 5)

    if _has_any(lower, CRASH_TERMS):
        node.signals.add("crash/restart/pod disruption")
        _add_hypothesis(node, "crash_or_pod_disruption", 90 if node.kind in {"Pod", "Deployment", "PodChaos"} else 20)
        _bump_global(graph, "crash_or_pod_disruption", 2)

    if _has_any(lower, NETWORK_TERMS):
        node.signals.add("network/dependency disruption")
        _add_hypothesis(node, "network_or_dependency", 90 if node.kind in {"NetworkChaos", "NetworkPolicy", "Service"} else 25)
        _bump_global(graph, "network_or_dependency", 2)

    if _has_any(lower, CONFIG_TERMS):
        node.signals.add("configuration/control-plane signal")

        # Static inventory is not fault evidence. ConfigMaps/Secrets are only
        # strong candidates when their contents carry fault-like config signals.
        if node.kind in {"ConfigMap", "Secret"} and _has_config_fault_signal(lower):
            _add_hypothesis(node, "configuration_or_secret", 900)
            node.reasons.add("configuration object contains fault-like config content")
        elif node.kind not in {"ConfigMap", "Secret"}:
            _add_hypothesis(node, "configuration_or_secret", 10)

        _bump_global(graph, "configuration_or_secret", 2)

    if _has_any(lower, TRAFFIC_TERMS):
        node.signals.add("traffic source / overload")
        _add_hypothesis(node, "traffic_overload", 100 if node.kind in {"Pod", "Deployment"} else 25)
        _bump_global(graph, "traffic_overload", 2)

    if node.kind in CHAOS_KINDS.values():
        node.signals.add("chaos object")
        _add_hypothesis(node, node.kind.lower(), 400)
        _bump_global(graph, node.kind.lower(), 10)

    if node.kind in {"ConfigMap", "Secret"} and evidence_category_is_k8s_object(category):
        node.signals.add("kubernetes configuration object")
        node.reasons.add("configuration object parsed from Kubernetes object snapshot")

        if _has_config_fault_signal(lower):
            _add_hypothesis(node, "configuration_or_secret", 900)
            node.reasons.add("configuration snapshot contains fault-like values")

    # Do not add per-line control-object score. Control kind is handled once
    # in the ranker. Repeated inventory rows should not inflate candidates.

    if is_observability_name(node.name):
        node.reasons.add("observability node; suppress unless directly causal")


def _add_family_edges(graph: IncidentGraph) -> None:
    nodes = list(graph.nodes.values())

    for node in nodes:
        if node.kind != "Pod":
            continue

        base = infer_workload_base(node.name)

        if not base or base == node.name:
            continue

        for other in nodes:
            if other.name != base:
                continue

            if other.key.id == node.key.id:
                continue

            graph.edges.append(
                GraphEdge(
                    source=other.key.id,
                    target=node.key.id,
                    relation="owns-or-selects-pod-family",
                )
            )


def _add_co_mention_edges(graph: IncidentGraph, entities: list[EntityRef], evidence_file: EvidenceFile, line: str) -> None:
    ids = []

    for entity in entities:
        key = NodeKey(kind=entity.kind or "Unknown", name=entity.name, namespace=entity.namespace or "")
        if key.id in graph.nodes:
            ids.append(key.id)

    ids = list(dict.fromkeys(ids))

    if len(ids) < 2:
        return

    relation = "co-mentioned"

    lower = line.lower()
    if _has_any(lower, ERROR_TERMS):
        relation = "co-mentioned-in-error"
    if _has_any(lower, NETWORK_TERMS):
        relation = "network-related-co-mention"
    if _has_any(lower, CONFIG_TERMS):
        relation = "config-related-co-mention"

    for source in ids[:10]:
        for target in ids[:10]:
            if source == target:
                continue

            graph.edges.append(
                GraphEdge(
                    source=source,
                    target=target,
                    relation=relation,
                    evidence_path=evidence_file.relative_path,
                )
            )


def _add_selector_like_edges(graph: IncidentGraph) -> None:
    """
    Generic approximation:
    if a control object name contains another resource name, link it to that resource.

    Example:
    NetworkChaos/foo-payment-delay -> Service/payment
    StressChaos/bar-valkey-memory-stress -> Service/valkey
    """
    nodes = list(graph.nodes.values())
    control_nodes = [node for node in nodes if node.kind in CONTROL_KINDS]

    for control in control_nodes:
        control_name = control.name.lower()

        for target in nodes:
            if target.key.id == control.key.id:
                continue

            target_name = target.name.lower()

            if len(target_name) < 3:
                continue

            if target_name in control_name or control_name in target_name:
                graph.edges.append(
                    GraphEdge(
                        source=control.key.id,
                        target=target.key.id,
                        relation="selector-or-name-implied-target",
                    )
                )


def _add_namespace_edges(graph: IncidentGraph) -> None:
    nodes = list(graph.nodes.values())
    namespaces = [node for node in nodes if node.kind == "Namespace"]

    for namespace_node in namespaces:
        ns_name = namespace_node.name

        for target in nodes:
            if target.key.id == namespace_node.key.id:
                continue

            # Direct namespace field match.
            if target.namespace == ns_name:
                graph.edges.append(
                    GraphEdge(
                        source=namespace_node.key.id,
                        target=target.key.id,
                        relation="namespace-contains-resource",
                    )
                )

            # Fallback for snapshots where nodes lack namespace but evidence mentions the namespace.
            elif any(ns_name in path for path in target.evidence_paths):
                graph.edges.append(
                    GraphEdge(
                        source=namespace_node.key.id,
                        target=target.key.id,
                        relation="namespace-context-from-evidence",
                    )
                )


def build_graph(scenario_dir: str | Path) -> IncidentGraph:
    files = read_scenario_files(scenario_dir)
    graph = IncidentGraph(scenario_path=Path(scenario_dir), files_seen=len(files))

    graph.global_scores = {
        "alerts": 0,
        "errors": 0,
        "resource_saturation": 0,
        "crash_or_pod_disruption": 0,
        "network_or_dependency": 0,
        "configuration_or_secret": 0,
        "traffic_overload": 0,
    }

    # Seed nodes from file paths first. Metric files are useful for this.
    for evidence_file in files:
        for entity in extract_entities_from_path(evidence_file.relative_path):
            _add_node(graph, entity, evidence_file, reason="entity inferred from file path")

    # Scan evidence lines. Raw metric rows are too noisy, so only path-seeded nodes
    # are used for metrics. Other evidence types are scanned.
    for evidence_file in files:
        if evidence_file.category == "metrics":
            continue

        lines = evidence_file.text.splitlines()

        for line in lines[:8000]:
            if not _is_important_line(line, evidence_file.category):
                continue

            entities = extract_entities_from_line(line, evidence_file.relative_path)

            if not entities:
                continue

            for entity in entities:
                node = _add_node(graph, entity, evidence_file, reason="entity inferred from evidence content")
                _score_node_from_line(graph, node, line, evidence_file.category)

            _add_co_mention_edges(graph, entities, evidence_file, line)

    _add_family_edges(graph)
    _add_selector_like_edges(graph)
    _add_namespace_edges(graph)

    graph.signals = {
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "files_seen": graph.files_seen,
        "control_nodes": len([n for n in graph.nodes.values() if n.kind in CONTROL_KINDS]),
        "chaos_nodes": len([n for n in graph.nodes.values() if n.kind in CHAOS_KINDS.values()]),
        "affected_nodes": len([n for n in graph.nodes.values() if n.affected_score > 0]),
    }

    # rank_candidates(graph)
    apply_namespace_policy_signals(graph)
    return graph


def build_sre_graph_context(scenario_dir: str | Path) -> str:
    return render_graph_context(build_graph(scenario_dir))
