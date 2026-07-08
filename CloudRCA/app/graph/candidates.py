from __future__ import annotations

import re
from collections import defaultdict, deque

from app.graph.records import GraphNode, IncidentGraph, RootCandidate
from app.graph.scoring import is_system_infra_node, symptom_ids


SYMPTOM_ONLY_KINDS = {"Service", "Pod"}
NAMESPACE_ROOT_SIGNALS = {
    "namespace resource policy enforcement",
}
POD_ROOT_SIGNALS = {
    "crash/restart/pod disruption",
    "container startup failure",
    "high traffic volume configuration",
    "image pull failure",
    "resource saturation or quota",
    "scheduling constraint failure",
    "stateful backend failure",
}
POD_ROOT_HYPOTHESES = {
    "crash_or_pod_disruption",
    "resource_or_quota",
    "scheduling_constraint",
    "stateful_backend_dependency",
    "workload_configuration",
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

DIVERSITY_QUOTAS = [
    (CHAOS_OR_MUTATION_KINDS, 8),
    ({"ConfigMap", "Secret"}, 8),
    ({"Deployment", "StatefulSet", "DaemonSet"}, 8),
    ({"ResourceQuota", "LimitRange"}, 6),
    ({"Namespace"}, 3),
    ({"Pod"}, 10),
    ({"HorizontalPodAutoscaler"}, 6),
    ({"NetworkPolicy"}, 4),
]

WORKLOAD_CONTROLLER_KINDS = {"Deployment", "StatefulSet", "DaemonSet"}
WORKLOAD_FAMILY_KINDS = {"Deployment", "ReplicaSet", "StatefulSet", "DaemonSet", "Pod"}


def get_paths_to_symptoms(
    graph: IncidentGraph,
    source_node_id: str,
    max_depth: int = 5,
    max_paths: int = 10,
) -> list[list[dict[str, object]]]:
    symptoms = symptom_ids(graph)
    adjacency: dict[str, list[tuple[str, str, str | None, float, str]]] = defaultdict(list)

    for edge in graph.edges:
        if edge.confidence < 0.3:
            continue

        adjacency[edge.source].append(
            (
                edge.target,
                edge.relation,
                edge.evidence_path,
                edge.confidence,
                edge.summary,
            )
        )

    for edges in adjacency.values():
        edges.sort(key=lambda item: item[3], reverse=True)

    queue = deque([(source_node_id, [])])
    visited = {(source_node_id, 0)}
    paths: list[list[dict[str, object]]] = []

    while queue and len(paths) < max_paths:
        node_id, path = queue.popleft()

        if len(path) >= max_depth:
            continue

        for next_id, relation, evidence_path, confidence, summary in adjacency.get(node_id, []):
            next_path = path + [
                {
                    "source": node_id,
                    "target": next_id,
                    "relation": relation,
                    "confidence": confidence,
                    "evidence_path": evidence_path,
                    "summary": summary,
                }
            ]

            if next_id in symptoms:
                paths.append(next_path)

                if len(paths) >= max_paths:
                    break

            state = (next_id, len(next_path))

            if state not in visited:
                visited.add(state)
                queue.append((next_id, next_path))

    paths.sort(
        key=lambda path: (
            min(float(step["confidence"]) for step in path) if path else 0,
            -len(path),
        ),
        reverse=True,
    )
    return paths[:max_paths]


def path_text(path: list[dict[str, object]]) -> str:
    if not path:
        return ""

    return " -> ".join([str(path[0]["source"])] + [str(step["target"]) for step in path])


def _selection_policy(
    node: GraphNode,
    symptom_only_warning: bool,
) -> tuple[bool, str, str]:
    if node.kind == "Namespace":
        if node.signals.intersection(NAMESPACE_ROOT_SIGNALS) or "namespace_resource_policy" in node.hypothesis_scores:
            return True, "strong_root", "namespace has direct policy enforcement evidence"

        return False, "context_only", "namespace is broad context unless direct namespace policy evidence is present"

    if "traffic source workload" in node.signals and "high traffic volume configuration" not in node.signals:
        if "backend dependency target" not in node.signals:
            return False, "context_only", "traffic source role is normal context without abnormal traffic evidence"

    if node.kind == "Pod":
        if "high traffic volume configuration" in node.signals:
            return True, "strong_root", "pod has explicit high-traffic configuration evidence"

        if node.signals.intersection(POD_ROOT_SIGNALS) or node.hypothesis_scores.keys() & POD_ROOT_HYPOTHESES:
            return True, "strong_root", "pod has explicit pod-level causal evidence"

        if "backend dependency target" in node.signals:
            return True, "weak_root", "pod is a traced backend dependency target"

        return False, "symptom", "pod appears affected; require explicit pod-level causal evidence before selecting it"

    if node.kind in CHAOS_OR_MUTATION_KINDS:
        return True, "strong_root", "chaos or mutation object with causal evidence"

    if node.kind in {"Deployment", "StatefulSet", "DaemonSet"}:
        if node.signals.intersection({"workload endpoint misconfiguration", "workload rollout failure"}):
            return True, "strong_root", "workload controller has rollout or endpoint configuration evidence"

    if node.kind in {"NetworkPolicy", "HorizontalPodAutoscaler"}:
        return True, "strong_root", f"{node.kind} is a control-policy candidate"

    if node.kind in {"ResourceQuota", "LimitRange"}:
        policy_signals = {"namespace resource policy enforcement", "resource saturation or quota"}
        if node.signals.intersection(policy_signals):
            return True, "strong_root", f"{node.kind} has direct resource policy evidence"

        return False, "context_only", f"{node.kind} is namespace policy context without enforcement evidence"

    if node.kind in {"ConfigMap", "Secret"}:
        if "active configuration content signal" in node.signals and "referenced by workload" in node.signals:
            return True, "strong_root", "configuration object has active content and workload references"

        if "active configuration content signal" in node.signals:
            return True, "weak_root", "configuration object has active content"

        if "inactive configuration context" in node.signals and "referenced by workload" in node.signals:
            return True, "weak_root", "configuration has inactive flag context and workload references"

        if "inactive configuration context" in node.signals:
            return False, "context_only", "configuration only contains inactive fault-related settings"

        return False, "context_only", "configuration object lacks active fault evidence"

    if "backend dependency target" in node.signals:
        return True, "weak_root", "candidate is a traced backend dependency target"

    if symptom_only_warning and not node.hypothesis_scores:
        return False, "symptom", "node is alert-heavy symptom evidence without causal support"

    return True, "weak_root", "candidate has root-cause evidence"


def _evidence_priority(row: dict[str, object]) -> int:
    kind = str(row.get("kind") or "")
    signals = set(row.get("signals") or [])

    if kind in CHAOS_OR_MUTATION_KINDS and "active mutation event" in signals:
        return 0

    if kind in {"NetworkPolicy", "HorizontalPodAutoscaler"}:
        return 1

    if kind in {"ResourceQuota", "LimitRange"}:
        if signals.intersection({"namespace resource policy enforcement", "resource saturation or quota"}):
            return 1

        return 11

    if kind == "Namespace" and "namespace resource policy enforcement" in signals:
        return 1

    if kind in {"ConfigMap", "Secret"} and "active configuration content signal" in signals:
        return 2

    if "workload endpoint misconfiguration" in signals:
        return 3

    if "workload rollout failure" in signals:
        return 4

    if "scheduling constraint failure" in signals:
        return 5

    if "high traffic volume configuration" in signals:
        return 4

    if "stateful backend failure" in signals:
        return 6

    if kind in {"ConfigMap", "Secret"} and "inactive configuration context" in signals:
        return 7

    if signals.intersection(
        {
            "container startup failure",
            "image pull failure",
            "resource saturation or quota",
            "crash/restart/pod disruption",
        }
    ):
        return 8

    if "backend dependency target" in signals:
        return 12

    return 10


def _selection_class_priority(row: dict[str, object]) -> int:
    return {
        "strong_root": 0,
        "weak_root": 1,
        "symptom": 2,
        "context_only": 3,
    }.get(str(row.get("selection_class") or ""), 4)


def _kind_role_priority(row: dict[str, object], evidence_priority: int) -> int:
    kind = str(row.get("kind") or "")

    if evidence_priority in {2, 3}:
        if kind in WORKLOAD_CONTROLLER_KINDS:
            return 0

        if kind == "Pod":
            return 1

    return 0


def _scope_sort_value(row: dict[str, object]) -> int:
    signals = set(row.get("signals") or [])

    if "scheduling constraint failure" in signals:
        return -int(row.get("affected_symptom_count") or 0)

    return 0


def _candidate_evidence_details(node: GraphNode) -> list[str]:
    details = []
    findings = node.attributes.get("active_configuration_findings") or node.attributes.get("configuration_findings")

    if isinstance(findings, list):
        for item in findings[:8]:
            if not isinstance(item, dict):
                continue

            path = str(item.get("path") or "")
            value = str(item.get("value") or "")
            reason = str(item.get("reason") or "")
            detail = f"config {path}={value}"

            if reason:
                detail = f"{detail} ({reason})"

            details.append(detail)

    traffic_user_count = node.attributes.get("traffic_user_count")

    if traffic_user_count:
        details.append(f"configured traffic users={traffic_user_count}")

    return details[:10]


def _candidate_context_details(node: GraphNode) -> list[str]:
    details = []

    for attribute_name in ["inactive_configuration_context", "configuration_context"]:
        findings = node.attributes.get(attribute_name)

        if not isinstance(findings, list):
            continue

        for item in findings[:6]:
            if not isinstance(item, dict):
                continue

            path = str(item.get("path") or "")
            value = str(item.get("value") or "")
            reason = str(item.get("reason") or "")
            detail = f"config {path}={value}"

            if reason:
                detail = f"{detail} ({reason})"

            details.append(detail)

    return details[:10]


def _workload_family_name(name: str) -> str:
    value = str(name or "").lower()
    value = re.sub(r"-[a-f0-9]{8,10}-[a-z0-9]{4,6}$", "", value)
    value = re.sub(r"-[a-f0-9]{8,10}$", "", value)
    return value


def _chaos_schedule_for_node(graph: IncidentGraph, node: GraphNode) -> GraphNode | None:
    if node.kind == "Schedule":
        return node

    for candidate in graph.nodes.values():
        if candidate.kind != "Schedule" or candidate.namespace != node.namespace:
            continue

        if node.name == candidate.name or node.name.startswith(f"{candidate.name}-"):
            return candidate

    return None


def _chaos_family_name(name: str) -> str:
    return re.sub(r"-[a-z0-9]{5}$", "", str(name or "").lower())


def _candidate_family(graph: IncidentGraph, node: GraphNode) -> dict[str, object]:
    if node.kind in WORKLOAD_FAMILY_KINDS:
        family_name = _workload_family_name(node.name)
        family_id = f"workload:{node.namespace}:{family_name}"
        return {
            "id": family_id,
            "name": family_name,
            "kind": "Workload",
            "role": "controller" if node.kind in WORKLOAD_CONTROLLER_KINDS else node.kind.lower(),
            "members": _family_members(graph, family_id),
        }

    if node.kind in CHAOS_OR_MUTATION_KINDS:
        schedule = _chaos_schedule_for_node(graph, node)

        if schedule:
            family_name = schedule.name
            family_id = f"chaos:{schedule.namespace}:{schedule.name}"
            role = "controller" if node.key.id == schedule.key.id else "member"
        else:
            family_name = _chaos_family_name(node.name)
            family_id = f"chaos:{node.namespace}:{family_name}"
            role = "member"

        return {
            "id": family_id,
            "name": family_name,
            "kind": "Chaos",
            "role": role,
            "members": _family_members(graph, family_id),
        }

    return {
        "id": node.key.id,
        "name": node.name,
        "kind": node.kind,
        "role": "self",
        "members": [node.key.id],
    }


def _family_id_for_node(graph: IncidentGraph, node: GraphNode) -> str:
    if node.kind in WORKLOAD_FAMILY_KINDS:
        return f"workload:{node.namespace}:{_workload_family_name(node.name)}"

    if node.kind in CHAOS_OR_MUTATION_KINDS:
        schedule = _chaos_schedule_for_node(graph, node)

        if schedule:
            return f"chaos:{schedule.namespace}:{schedule.name}"

        return f"chaos:{node.namespace}:{_chaos_family_name(node.name)}"

    return node.key.id


def _family_members(graph: IncidentGraph, family_id: str) -> list[str]:
    members = [
        node.key.id
        for node in graph.nodes.values()
        if _family_id_for_node(graph, node) == family_id
    ]
    members.sort()
    return members[:12]


def _dossier_sort_key(row: dict[str, object]) -> tuple[object, ...]:
    evidence_priority = _evidence_priority(row)
    return (
        _selection_class_priority(row),
        evidence_priority,
        _kind_role_priority(row, evidence_priority),
        _scope_sort_value(row),
        -int(row.get("score") or 0),
        -int(row.get("causal_path_count") or 0),
        str(row.get("kind") or ""),
        str(row.get("id") or ""),
    )


def build_candidate_dossiers(graph: IncidentGraph, limit: int = 35) -> list[dict[str, object]]:
    dossiers: list[dict[str, object]] = []
    selected = _select_candidate_pool(graph, limit=limit)

    for candidate, node in selected:
        paths = get_paths_to_symptoms(graph, candidate.node_id, max_depth=5, max_paths=8)
        symptom_targets = sorted({str(path[-1]["target"]) for path in paths if path})
        symptom_only_warning = (
            node.kind in SYMPTOM_ONLY_KINDS
            and node.affected_score > 0
            and candidate.affected_symptom_count <= 1
        )
        caution = ""

        if symptom_only_warning:
            caution = "This node is alert-heavy and may be a symptom unless other evidence supports causality."
        elif node.kind == "Namespace":
            caution = "Namespace candidates are broad; require namespace-scoped evidence before selecting them."

        root_selectable, selection_class, selection_reason = _selection_policy(node, symptom_only_warning)
        context_details = _candidate_context_details(node)
        why_not_root = ""

        if selection_class == "context_only":
            why_not_root = selection_reason

        if node.kind in {"ConfigMap", "Secret"} and "inactive configuration context" in node.signals and "active configuration content signal" not in node.signals:
            caution = "Configuration findings are inactive context; select only when no stronger direct evidence explains the symptoms."

            if not root_selectable:
                why_not_root = "no active fault-like configuration setting was found"

        dossiers.append(
            {
                "rank": 0,
                "id": candidate.node_id,
                "kind": candidate.kind,
                "name": candidate.name,
                "namespace": candidate.namespace,
                "score": candidate.score,
                "best_hypothesis": candidate.best_hypothesis,
                "signals": sorted(node.signals),
                "reasons": candidate.reasons,
                "supporting_evidence": candidate.supporting_evidence,
                "evidence_details": _candidate_evidence_details(node),
                "context_details": context_details,
                "evidence_paths": candidate.evidence_paths,
                "affected_symptom_count": len(symptom_targets),
                "causal_path_count": len(paths),
                "root_selectable": root_selectable,
                "selection_class": selection_class,
                "selection_reason": selection_reason,
                "why_not_root": why_not_root,
                "candidate_family": _candidate_family(graph, node),
                "causal_paths": [
                    {
                        "path": path_text(path),
                        "relations": [step["relation"] for step in path],
                        "min_confidence": min(float(step["confidence"]) for step in path) if path else 0,
                    }
                    for path in paths[:5]
                ],
                "caution": caution,
            }
        )

    dossiers.sort(key=_dossier_sort_key)

    for rank, dossier in enumerate(dossiers, start=1):
        dossier["rank"] = rank

    return dossiers


def _select_candidate_pool(
    graph: IncidentGraph,
    limit: int,
) -> list[tuple[RootCandidate, GraphNode]]:
    eligible = []

    for candidate in graph.candidates:
        node = graph.nodes.get(candidate.node_id)

        if node is None or is_system_infra_node(node):
            continue

        eligible.append((candidate, node))

    selected: list[tuple[RootCandidate, GraphNode]] = []
    seen: set[str] = set()

    def add(candidate: RootCandidate, node: GraphNode) -> bool:
        if candidate.node_id in seen:
            return False

        seen.add(candidate.node_id)
        selected.append((candidate, node))
        return True

    for candidate, node in eligible[:10]:
        add(candidate, node)

    for kinds, quota in DIVERSITY_QUOTAS:
        added = 0

        for candidate, node in eligible:
            if node.kind not in kinds:
                continue

            if add(candidate, node):
                added += 1

            if added >= quota:
                break

    for candidate, node in eligible:
        if len(selected) >= limit:
            break

        add(candidate, node)

    return selected
