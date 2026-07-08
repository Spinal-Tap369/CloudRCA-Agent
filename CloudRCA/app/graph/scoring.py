from __future__ import annotations

from collections import defaultdict, deque

from app.graph.records import GraphNode, IncidentGraph, RootCandidate
from app.graph.parsers import is_observability_name
from app.graph.relationships import add_hypothesis


SYSTEM_INFRA_TERMS = {
    "cilium",
    "clustermesh",
    "coredns",
    "kube-",
    "clickhouse",
    "opensearch",
    "operator",
    "persistentvolume",
    "storageclass",
    "volumeattachment",
    "metrics-server",
    "cert-manager",
    "otel-collector",
    "opentelemetry-collector",
    "prometheus",
    "grafana",
    "ingress-nginx",
    "jaeger",
}

SYSTEM_NAMESPACES = {
    "cert-manager",
    "data-recorders",
    "kube-node-lease",
    "kube-public",
    "kube-system",
    "ingress-nginx",
    "monitoring",
    "opensearch",
    "opentelemetry-collectors",
}

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

WORKLOAD_CONTROLLER_KINDS = {"Deployment", "StatefulSet", "DaemonSet", "ReplicaSet"}

ROLLOUT_POD_FAILURE_SIGNALS = {
    "image pull failure",
    "container startup failure",
}


def _mark_workload_rollout_failure(node: GraphNode) -> None:
    node.signals.add("workload rollout failure")
    node.signals.add("configuration/control-plane signal")
    add_hypothesis(
        node,
        "workload_configuration",
        1700,
        "selected pod has rollout failure evidence",
    )


def is_system_infra_node(node: GraphNode) -> bool:
    lower = node.name.lower()

    if node.namespace.lower() in SYSTEM_NAMESPACES:
        return True

    return is_observability_name(node.name) or any(term in lower for term in SYSTEM_INFRA_TERMS)


def symptom_ids(graph: IncidentGraph) -> set[str]:
    return {
        node.key.id
        for node in graph.nodes.values()
        if node.affected_score > 0 and not is_system_infra_node(node)
    }


def reachable_symptoms(
    graph: IncidentGraph,
    source_id: str,
    max_depth: int = 5,
) -> tuple[int, int]:
    symptoms = symptom_ids(graph)
    adjacency: dict[str, list[str]] = defaultdict(list)

    for edge in graph.edges:
        if edge.confidence < 0.3:
            continue

        adjacency[edge.source].append(edge.target)

    queue = deque([(source_id, 0)])
    visited = {source_id}
    reached: set[str] = set()
    path_count = 0

    while queue:
        node_id, depth = queue.popleft()

        if depth >= max_depth:
            continue

        for next_id in adjacency.get(node_id, []):
            if next_id in symptoms:
                reached.add(next_id)
                path_count += 1

            if next_id not in visited:
                visited.add(next_id)
                queue.append((next_id, depth + 1))

    return len(reached), path_count


def candidate_kind_prior(kind: str) -> int:
    if kind in CHAOS_OR_MUTATION_KINDS:
        return 1100

    if kind in {"ConfigMap", "Secret", "NetworkPolicy", "ResourceQuota", "LimitRange"}:
        return 900

    if kind == "HorizontalPodAutoscaler":
        return 800

    if kind == "Namespace":
        return 550

    if kind in {"Deployment", "StatefulSet", "DaemonSet"}:
        return 450

    if kind == "Pod":
        return 220

    if kind == "Service":
        return 100

    return 0


def apply_relationship_features(graph: IncidentGraph) -> None:
    for edge in graph.edges:
        if edge.relation not in {"owns", "selects-pod", "owns-or-selects-pod-family"}:
            continue

        source = graph.nodes.get(edge.source)
        target = graph.nodes.get(edge.target)

        if source is None or target is None:
            continue

        if source.kind not in WORKLOAD_CONTROLLER_KINDS or target.kind != "Pod":
            continue

        if target.signals.intersection(ROLLOUT_POD_FAILURE_SIGNALS):
            _mark_workload_rollout_failure(source)

    for edge in graph.edges:
        if edge.relation != "owns":
            continue

        source = graph.nodes.get(edge.source)
        target = graph.nodes.get(edge.target)

        if source is None or target is None:
            continue

        if source.kind not in WORKLOAD_CONTROLLER_KINDS or target.kind not in WORKLOAD_CONTROLLER_KINDS:
            continue

        if "workload rollout failure" in target.signals:
            _mark_workload_rollout_failure(source)


def score_candidates(graph: IncidentGraph) -> None:
    candidates: list[RootCandidate] = []
    apply_relationship_features(graph)

    for node in graph.nodes.values():
        score = candidate_kind_prior(node.kind)

        if score <= 0 and not node.hypothesis_scores:
            continue

        score += min(sum(node.hypothesis_scores.values()), 1400)
        reached_symptoms, path_count = reachable_symptoms(graph, node.key.id)
        score += min(reached_symptoms * 90, 900)
        score += min(path_count * 15, 300)
        score += min(len(node.evidence_paths) * 8, 160)

        if node.kind == "Service" and node.affected_score > 0:
            score -= 250

        if node.kind == "Pod" and node.affected_score > 0 and not node.hypothesis_scores:
            score -= 120

        if "backend dependency target" in node.signals:
            score += 350

        if "namespace resource policy enforcement" in node.signals:
            score += 1800

        if node.kind in CHAOS_OR_MUTATION_KINDS:
            if "active mutation event" in node.signals:
                score += 2500
            elif "chaos or mutation object" in node.signals:
                score += 1200

        if node.kind == "NetworkPolicy" and "network policy" in node.signals:
            score += 1800

        if node.kind == "HorizontalPodAutoscaler":
            if "autoscaling policy" in node.signals:
                score += 1500
            if "autoscaling policy group" in node.signals:
                score += 1800

        if node.kind == "ResourceQuota" and "resource saturation or quota" in node.signals:
            score += 1400

        if "workload endpoint misconfiguration" in node.signals:
            score += 2400

        if "workload rollout failure" in node.signals:
            score += 1800

        if "stateful backend failure" in node.signals:
            score += 2200

        if node.kind == "Pod":
            if "crash/restart/pod disruption" in node.signals:
                score += 850

            if "high traffic volume configuration" in node.signals:
                score += 1000

            if "scheduling constraint failure" in node.signals:
                score += 1600

            if "metric anomaly or error signal" in node.signals:
                score += 400

            if "resource saturation or quota" in node.signals:
                score += 250

        if node.kind in {"ConfigMap", "Secret"}:
            if "active configuration content signal" in node.signals and "referenced by workload" in node.signals:
                score += 900
            elif "active configuration content signal" in node.signals:
                score += 500

            if "inactive configuration context" in node.signals and "active configuration content signal" not in node.signals:
                score -= 1800

        if is_system_infra_node(node):
            score -= 1200

        if "autoscaling policy group" in node.signals:
            score += 1200

        node.candidate_score = max(score, 0)

        if node.candidate_score <= 0:
            continue

        candidates.append(
            RootCandidate(
                node_id=node.key.id,
                name=node.name,
                kind=node.kind,
                namespace=node.namespace,
                score=node.candidate_score,
                best_hypothesis=node.best_hypothesis,
                reasons=sorted(node.reasons)[:12],
                evidence_paths=node.evidence_paths[:12],
                supporting_evidence=[
                    item.summary
                    for item in node.evidence
                    if item.summary
                ][:8],
                causal_path_count=path_count,
                affected_symptom_count=reached_symptoms,
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    graph.candidates = candidates
