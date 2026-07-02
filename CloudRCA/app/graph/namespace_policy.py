from __future__ import annotations

from collections import Counter
import re
from pathlib import Path

from app.graph.models import EvidenceRef, GraphEdge, GraphNode, IncidentGraph, NodeKey


POLICY_SIGNAL_TERMS = {
    "resourcequota",
    "resource quota",
    "limitrange",
    "limit range",
    "quota",
    "exceeded quota",
    "exceededquotas",
    "forbidden",
    "failedcreate",
    "failed create",
    "minimumreplicasunavailable",
    "minimum replicas unavailable",
    "insufficient memory",
    "memory quota",
    "requests.memory",
    "limits.memory",
    "cannot create",
    "cannot schedule",
    "failed scheduling",
    "failedscheduling",
    "pod scheduling",
    "0/1 replicas",
    "replicas created",
    "exceeded",
}


BAD_NAMESPACE_WORDS = {
    "namespace",
    "namespaces",
    "status",
    "statuses",
    "selector",
    "selectors",
    "field",
    "fields",
    "label",
    "labels",
    "name",
    "names",
    "is",
    "isn",
    "are",
    "was",
    "were",
    "of",
    "and",
    "or",
    "to",
    "from",
    "for",
    "if",
    "as",
    "by",
    "in",
    "on",
    "that",
    "this",
    "will",
    "only",
    "where",
    "pod",
    "pods",
    "service",
    "services",
    "deployment",
    "deployments",
    "replicaset",
    "replicasets",
    "quota",
    "memory",
    "cpu",
    "default",
    "true",
    "false",
    "null",
    "none",
    "rejected",
    "fails",
    "failed",
    "enforcement",
}


RESOURCE_KINDS_CONSTRAINED_BY_NAMESPACE = {
    "Pod",
    "Deployment",
    "ReplicaSet",
    "StatefulSet",
    "DaemonSet",
    "Service",
    "ConfigMap",
    "Secret",
}


TEXT_SUFFIXES = {
    ".txt",
    ".log",
    ".json",
    ".yaml",
    ".yml",
    ".tsv",
    ".csv",
    ".out",
}


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.suffix == ""


def _is_ground_truth(path: Path) -> bool:
    return "ground" in path.name.lower() and "truth" in path.name.lower()


def _has_policy_signal(text: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in POLICY_SIGNAL_TERMS)


def _clean_namespace(value: str) -> str:
    value = str(value).strip().strip("'\"`.,:;()[]{}")
    lower = value.lower()

    if not value or lower in BAD_NAMESPACE_WORDS:
        return ""

    if not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", value):
        return ""

    if "-" not in value and len(value) < 4:
        return ""

    return value


def _extract_strict_namespaces(line: str) -> set[str]:
    namespaces: set[str] = set()

    patterns = [
        # JSON/YAML-ish fields.
        r"""["']namespace["']\s*[:=]\s*["']([a-z0-9][a-z0-9-]*)["']""",
        r"""["']metadata\.namespace["']\s*[:=]\s*["']([a-z0-9][a-z0-9-]*)["']""",
        r"""\bnamespace\s*[:=]\s*["']([a-z0-9][a-z0-9-]*)["']""",

        # Kubernetes resource path references.
        r"""\bnamespaces/([a-z0-9][a-z0-9-]*)\b""",

        # Human-readable messages.
        r"""\bin\s+namespace\s+["']?([a-z0-9][a-z0-9-]*)["']?\b""",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, line, flags=re.IGNORECASE):
            namespace = _clean_namespace(match.group(1))
            if namespace:
                namespaces.add(namespace)

    return namespaces


def _relative_path(graph: IncidentGraph, path: Path) -> str:
    try:
        return str(path.relative_to(graph.scenario_path))
    except ValueError:
        return str(path)


def _ensure_node(
    graph: IncidentGraph,
    kind: str,
    name: str,
    namespace: str = "",
) -> GraphNode:
    key = NodeKey(kind=kind, name=name, namespace=namespace)
    existing = graph.nodes.get(key.id)

    if existing is not None:
        return existing

    node = GraphNode(key=key)
    graph.nodes[key.id] = node
    return node


def _add_evidence_once(
    node: GraphNode,
    relative_path: str,
    category: str,
    summary: str,
) -> None:
    if relative_path not in node.evidence_paths:
        node.evidence.append(
            EvidenceRef(
                path=relative_path,
                category=category,
                summary=summary[:500],
            )
        )


def _add_edge_once(
    graph: IncidentGraph,
    source: str,
    target: str,
    relation: str,
    evidence_path: str | None = None,
) -> None:
    for edge in graph.edges:
        if edge.source == source and edge.target == target and edge.relation == relation:
            return

    graph.edges.append(
        GraphEdge(
            source=source,
            target=target,
            relation=relation,
            evidence_path=evidence_path,
        )
    )


def _connect_namespace_to_resources(
    graph: IncidentGraph,
    namespace_node: GraphNode,
    evidence_path: str,
) -> None:
    for target in list(graph.nodes.values()):
        if target.key.id == namespace_node.key.id:
            continue

        if target.kind not in RESOURCE_KINDS_CONSTRAINED_BY_NAMESPACE:
            continue

        if target.namespace == namespace_node.name:
            _add_edge_once(
                graph,
                namespace_node.key.id,
                target.key.id,
                "namespace-resource-policy-constrains-resource",
                evidence_path,
            )
            continue

        if target.namespace in {"", "_"}:
            _add_edge_once(
                graph,
                namespace_node.key.id,
                target.key.id,
                "namespace-resource-policy-may-constrain-resource",
                evidence_path,
            )


def _choose_dominant_namespace(namespace_counts: Counter[str]) -> str:
    if not namespace_counts:
        return ""

    ranked = namespace_counts.most_common(2)
    top_name, top_count = ranked[0]

    if top_count < 3:
        return ""

    if len(ranked) == 1:
        return top_name

    second_count = ranked[1][1]

    # Require clear dominance. This prevents random multi-namespace evidence from
    # creating a fake namespace root.
    if top_count >= second_count * 3:
        return top_name

    return ""


def apply_namespace_policy_signals(graph: IncidentGraph) -> None:
    """
    Infer namespace-level resource-policy nodes from Kubernetes evidence.

    Rule:
    - policy/quota/scheduling evidence is required
    - namespace comes from same line if available
    - otherwise namespace may come from dominant explicit namespace metadata in the scenario
    """

    scenario_path = Path(graph.scenario_path)

    namespace_counts: Counter[str] = Counter()
    namespace_hits: dict[str, list[tuple[str, str]]] = {}
    policy_hits_without_namespace: list[tuple[str, str]] = []

    for path in scenario_path.rglob("*"):
        if not path.is_file():
            continue

        if _is_ground_truth(path):
            continue

        if not _is_text_file(path):
            continue

        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue

        relative_path = _relative_path(graph, path)

        for line in lines:
            namespaces = _extract_strict_namespaces(line)

            for namespace in namespaces:
                namespace_counts[namespace] += 1

            if not _has_policy_signal(line):
                continue

            if namespaces:
                for namespace in namespaces:
                    namespace_hits.setdefault(namespace, []).append((relative_path, line.strip()))
            else:
                policy_hits_without_namespace.append((relative_path, line.strip()))

    dominant_namespace = _choose_dominant_namespace(namespace_counts)

    if dominant_namespace and policy_hits_without_namespace:
        namespace_hits.setdefault(dominant_namespace, []).extend(policy_hits_without_namespace[:20])

    for namespace, hits in namespace_hits.items():
        if not hits:
            continue

        node = _ensure_node(graph, kind="Namespace", name=namespace)

        node.signals.add("namespace-level resource policy")
        node.signals.add("quota/scheduling failure")
        node.reasons.add("namespace inferred from Kubernetes quota/scheduling evidence")
        node.hypothesis_scores["namespace_resource_policy"] = max(
            node.hypothesis_scores.get("namespace_resource_policy", 0),
            1600,
        )

        for relative_path, line in hits[:10]:
            _add_evidence_once(
                node,
                relative_path=relative_path,
                category="namespace_policy",
                summary=line,
            )

        _connect_namespace_to_resources(graph, node, hits[0][0])
