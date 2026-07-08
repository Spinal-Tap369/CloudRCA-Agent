from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.graph.ingest import name_mentioned, normalize_token, relative_path
from app.graph.records import EvidenceRef, GraphEdge, GraphNode, IncidentGraph, NodeKey
from app.graph.parsers import infer_workload_base, is_observability_name, normalize_kind


WORKLOAD_KINDS = {
    "Deployment",
    "ReplicaSet",
    "StatefulSet",
    "DaemonSet",
    "Job",
    "CronJob",
    "Pod",
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

NOISY_NAME_PARTS = {
    "chaos",
    "demo",
    "deployment",
    "delay",
    "http",
    "jvm",
    "memory",
    "network",
    "otel",
    "partition",
    "pod",
    "return",
    "service",
    "stress",
}


def add_evidence(
    node: GraphNode,
    path: str,
    category: str,
    summary: str,
    timestamp: str | None = None,
) -> None:
    if not path:
        return

    key = (path, summary[:200])

    for item in node.evidence:
        if (item.path, item.summary[:200]) == key:
            return

    if len(node.evidence) >= 80:
        return

    node.evidence.append(
        EvidenceRef(
            path=path,
            category=category,
            summary=summary[:500],
            timestamp=timestamp,
        )
    )


def ensure_node(
    graph: IncidentGraph,
    kind: str,
    name: str,
    namespace: str = "",
    evidence_path: str = "",
    category: str = "",
    summary: str = "",
    timestamp: str | None = None,
) -> GraphNode:
    kind = normalize_kind(kind or "Unknown")
    name = str(name or "").strip() or "Unknown"
    namespace = str(namespace or "").strip()
    key = NodeKey(kind=kind, name=name, namespace=namespace)
    node = graph.nodes.get(key.id)

    if node is None:
        node = GraphNode(key=key)
        graph.nodes[key.id] = node

    if evidence_path:
        add_evidence(
            node=node,
            path=evidence_path,
            category=category,
            summary=summary,
            timestamp=timestamp,
        )

    return node


def _edge_keys(graph: IncidentGraph) -> set[tuple[str, str, str, str | None]]:
    keys = graph.signals.setdefault("_edge_keys", set())
    return keys  # type: ignore[return-value]


def add_edge(
    graph: IncidentGraph,
    source: str,
    target: str,
    relation: str,
    evidence_path: str | None = None,
    confidence: float = 0.5,
    summary: str = "",
) -> None:
    if not source or not target or source == target:
        return

    key = (source, target, relation, evidence_path)
    keys = _edge_keys(graph)

    if key in keys:
        return

    keys.add(key)
    graph.edges.append(
        GraphEdge(
            source=source,
            target=target,
            relation=relation,
            evidence_path=evidence_path,
            confidence=confidence,
            summary=summary[:500],
        )
    )


def add_hypothesis(node: GraphNode, tag: str, points: int, reason: str) -> None:
    current = node.hypothesis_scores.get(tag, 0)
    node.hypothesis_scores[tag] = min(current + points, 5000)
    node.reasons.add(reason)


def collect_refs(obj: dict[str, Any], ref_key: str) -> set[str]:
    refs: set[str] = set()
    direct_keys = {
        "configMap": ["configMap", "configMapRef", "configMapKeyRef"],
        "secret": ["secret", "secretRef", "secretKeyRef"],
    }.get(ref_key, [ref_key])

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key in direct_keys:
                direct = value.get(key)

                if isinstance(direct, dict) and direct.get("name"):
                    refs.add(str(direct["name"]))

            for child in value.values():
                walk(child)

        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(obj.get("spec", {}))
    return refs


def selector_matches(selector: dict[str, Any], labels: dict[str, str]) -> bool:
    if not selector:
        return False

    match_labels = selector.get("matchLabels") if isinstance(selector.get("matchLabels"), dict) else selector

    if not isinstance(match_labels, dict) or not match_labels:
        return False

    for key, value in match_labels.items():
        if str(labels.get(str(key), "")) != str(value):
            return False

    return True


def extract_selector(obj: dict[str, Any]) -> dict[str, Any]:
    spec = obj.get("spec") if isinstance(obj.get("spec"), dict) else {}
    selector = spec.get("selector")
    return selector if isinstance(selector, dict) else {}


def _target_ref(spec: dict[str, Any]) -> tuple[str, str] | None:
    ref = spec.get("scaleTargetRef")

    if isinstance(ref, dict) and ref.get("kind") and ref.get("name"):
        return normalize_kind(str(ref["kind"])), str(ref["name"])

    return None


def add_object_relationships(graph: IncidentGraph, objects: list[dict[str, Any]]) -> None:
    by_id = {item["node"].key.id: item for item in objects}
    pods = [item for item in objects if item["kind"] == "Pod"]
    namespaced = defaultdict(list)

    for item in objects:
        namespaced[item["namespace"]].append(item)

    for item in objects:
        node = item["node"]
        obj = item["obj"]
        namespace = item["namespace"]
        metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
        owner_refs = metadata.get("ownerReferences")

        if isinstance(owner_refs, list):
            for owner in owner_refs:
                if not isinstance(owner, dict) or not owner.get("kind") or not owner.get("name"):
                    continue

                owner_node = ensure_node(
                    graph,
                    kind=str(owner["kind"]),
                    name=str(owner["name"]),
                    namespace=namespace,
                    evidence_path=item["path"],
                    category="kubernetes_objects",
                    summary=f"{node.kind} owner reference points to {owner['kind']} {owner['name']}",
                )
                add_edge(
                    graph,
                    source=owner_node.key.id,
                    target=node.key.id,
                    relation="owns",
                    evidence_path=item["path"],
                    confidence=0.95,
                )

        if item["kind"] in WORKLOAD_KINDS:
            _add_config_relationships(graph, item)

        if item["kind"] == "Service":
            _add_service_selector_edges(graph, item, pods)

        if item["kind"] in {"Deployment", "ReplicaSet", "StatefulSet", "DaemonSet"}:
            _add_workload_selector_edges(graph, item, pods)

        if item["kind"] == "HorizontalPodAutoscaler":
            _add_hpa_edges(graph, item)

        if item["kind"] == "NetworkPolicy":
            _add_network_policy_edges(graph, item, pods)

    _add_namespace_edges(graph, namespaced)
    _add_pod_family_edges(graph, pods, by_id)
    _add_hpa_aggregate_edges(graph)


def add_name_implied_edges(graph: IncidentGraph) -> None:
    control_kinds = CHAOS_OR_MUTATION_KINDS | {"NetworkPolicy", "HorizontalPodAutoscaler"}
    target_kinds = {"Service", "Deployment", "Pod", "StatefulSet", "DaemonSet"}
    controls = [node for node in graph.nodes.values() if node.kind in control_kinds]

    for control in controls:
        for target in graph.nodes.values():
            if target.kind not in target_kinds:
                continue

            if is_observability_name(target.name):
                continue

            if name_mentioned(target.name, control.name):
                add_edge(
                    graph,
                    source=control.key.id,
                    target=target.key.id,
                    relation="name-implied-target",
                    confidence=0.55,
                    summary=f"{control.kind} name mentions {target.name}",
                )
            elif control.kind in CHAOS_OR_MUTATION_KINDS and _shares_target_name_part(control.name, target.name):
                add_edge(
                    graph,
                    source=control.key.id,
                    target=target.key.id,
                    relation="name-implied-target",
                    confidence=0.45,
                    summary=f"{control.kind} name shares a target token with {target.name}",
                )


def _shares_target_name_part(control_name: str, target_name: str) -> bool:
    control_token = normalize_token(control_name)

    if not control_token:
        return False

    for part in normalize_token(target_name).split("-"):
        if len(part) < 5 or part in NOISY_NAME_PARTS:
            continue

        if part in control_token:
            return True

    return False


def _add_config_relationships(graph: IncidentGraph, item: dict[str, Any]) -> None:
    node = item["node"]
    obj = item["obj"]
    namespace = item["namespace"]

    for config_name in collect_refs(obj, "configMap"):
        config_node = ensure_node(
            graph,
            kind="ConfigMap",
            name=config_name,
            namespace=namespace,
            evidence_path=item["path"],
            category="kubernetes_objects",
            summary=f"{node.kind} {node.name} references ConfigMap {config_name}",
        )
        config_node.signals.add("referenced by workload")
        add_hypothesis(config_node, "configuration_or_secret", 300, "referenced by workload spec")
        add_edge(
            graph,
            source=config_node.key.id,
            target=node.key.id,
            relation="configures-workload",
            evidence_path=item["path"],
            confidence=0.95,
            summary=f"{node.kind} references ConfigMap {config_name}",
        )

    for secret_name in collect_refs(obj, "secret"):
        secret_node = ensure_node(
            graph,
            kind="Secret",
            name=secret_name,
            namespace=namespace,
            evidence_path=item["path"],
            category="kubernetes_objects",
            summary=f"{node.kind} {node.name} references Secret {secret_name}",
        )
        secret_node.signals.add("referenced by workload")
        add_hypothesis(secret_node, "configuration_or_secret", 250, "referenced by workload spec")
        add_edge(
            graph,
            source=secret_node.key.id,
            target=node.key.id,
            relation="configures-workload",
            evidence_path=item["path"],
            confidence=0.9,
            summary=f"{node.kind} references Secret {secret_name}",
        )


def _add_service_selector_edges(
    graph: IncidentGraph,
    item: dict[str, Any],
    pods: list[dict[str, Any]],
) -> None:
    node = item["node"]
    selector = extract_selector(item["obj"])

    for pod in pods:
        if pod["namespace"] != item["namespace"]:
            continue

        if selector_matches(selector, pod["node"].labels):
            add_edge(
                graph,
                source=node.key.id,
                target=pod["node"].key.id,
                relation="service-selects-pod",
                evidence_path=item["path"],
                confidence=0.9,
            )
            add_edge(
                graph,
                source=pod["node"].key.id,
                target=node.key.id,
                relation="pod-serves-service",
                evidence_path=item["path"],
                confidence=0.85,
            )


def _add_workload_selector_edges(
    graph: IncidentGraph,
    item: dict[str, Any],
    pods: list[dict[str, Any]],
) -> None:
    selector = extract_selector(item["obj"])

    for pod in pods:
        if pod["namespace"] != item["namespace"]:
            continue

        if selector_matches(selector, pod["node"].labels):
            add_edge(
                graph,
                source=item["node"].key.id,
                target=pod["node"].key.id,
                relation="selects-pod",
                evidence_path=item["path"],
                confidence=0.85,
            )


def _add_hpa_edges(graph: IncidentGraph, item: dict[str, Any]) -> None:
    spec = item["obj"].get("spec") if isinstance(item["obj"].get("spec"), dict) else {}
    target = _target_ref(spec)

    if not target:
        return

    target_kind, target_name = target
    target_node = ensure_node(
        graph,
        kind=target_kind,
        name=target_name,
        namespace=item["namespace"],
        evidence_path=item["path"],
        category="kubernetes_objects",
        summary=f"HPA {item['node'].name} targets {target_kind} {target_name}",
    )
    add_edge(
        graph,
        source=item["node"].key.id,
        target=target_node.key.id,
        relation="scales-workload",
        evidence_path=item["path"],
        confidence=0.9,
    )


def _add_network_policy_edges(
    graph: IncidentGraph,
    item: dict[str, Any],
    pods: list[dict[str, Any]],
) -> None:
    spec = item["obj"].get("spec") if isinstance(item["obj"].get("spec"), dict) else {}
    selector = spec.get("podSelector") if isinstance(spec.get("podSelector"), dict) else {}

    for pod in pods:
        if pod["namespace"] != item["namespace"]:
            continue

        if selector_matches(selector, pod["node"].labels) or not selector:
            add_edge(
                graph,
                source=item["node"].key.id,
                target=pod["node"].key.id,
                relation="network-policy-selects-pod",
                evidence_path=item["path"],
                confidence=0.8,
            )


def _add_namespace_edges(
    graph: IncidentGraph,
    namespaced: dict[str, list[dict[str, Any]]],
) -> None:
    for namespace, items in namespaced.items():
        if not namespace:
            continue

        namespace_node = ensure_node(
            graph,
            kind="Namespace",
            name=namespace,
            namespace="",
            evidence_path="k8s_objects_raw.tsv",
            category="kubernetes_objects",
            summary=f"Namespace {namespace} contains resources",
        )

        for item in items:
            if item["node"].key.id == namespace_node.key.id:
                continue

            add_edge(
                graph,
                source=namespace_node.key.id,
                target=item["node"].key.id,
                relation="namespace-contains-resource",
                evidence_path=item["path"],
                confidence=0.75,
            )


def _add_pod_family_edges(
    graph: IncidentGraph,
    pods: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
) -> None:
    for pod in pods:
        base = infer_workload_base(pod["name"])

        if not base or base == pod["name"]:
            continue

        for kind in ["Deployment", "ReplicaSet", "StatefulSet", "DaemonSet"]:
            key = NodeKey(kind=kind, name=base, namespace=pod["namespace"]).id

            if key in by_id or key in graph.nodes:
                add_edge(
                    graph,
                    source=key,
                    target=pod["node"].key.id,
                    relation="owns-or-selects-pod-family",
                    confidence=0.55,
                )


def _add_hpa_aggregate_edges(graph: IncidentGraph) -> None:
    hpas_by_namespace: dict[str, list[GraphNode]] = defaultdict(list)

    for node in graph.nodes.values():
        if node.kind == "HorizontalPodAutoscaler" and node.namespace:
            hpas_by_namespace[node.namespace].append(node)

    for namespace, hpas in hpas_by_namespace.items():
        if len(hpas) < 2:
            continue

        aggregate = ensure_node(
            graph,
            kind="HorizontalPodAutoscaler",
            name=f"{namespace}-horizontal-pod-autoscalers",
            namespace=namespace,
            evidence_path="k8s_objects_raw.tsv",
            category="kubernetes_objects",
            summary=f"Aggregate HPA policy group for namespace {namespace}",
        )
        aggregate.signals.add("autoscaling policy group")
        add_hypothesis(
            aggregate,
            "autoscaling_policy",
            700,
            "namespace contains multiple horizontal pod autoscalers",
        )

        for hpa in hpas:
            add_edge(
                graph,
                source=aggregate.key.id,
                target=hpa.key.id,
                relation="contains-hpa",
                evidence_path="k8s_objects_raw.tsv",
                confidence=0.85,
            )


def graph_relative_path(graph: IncidentGraph, path: str) -> str:
    return relative_path(graph.scenario_path, graph.scenario_path / path)
