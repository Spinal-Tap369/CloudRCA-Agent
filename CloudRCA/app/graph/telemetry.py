from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from app.graph.ingest import has_any, iter_tsv, normalize_token, parse_mapping, relative_path
from app.graph.k8s_inventory import (
    ALERT_TERMS,
    CONFIG_TERMS,
    CRASH_TERMS,
    ERROR_TERMS,
    NETWORK_TERMS,
    RESOURCE_TERMS,
    mark_from_text,
)
from app.graph.records import GraphNode, IncidentGraph
from app.graph.relationships import add_edge, add_hypothesis, ensure_node
from app.graph.parsers import is_observability_name


METRIC_ROW_LIMIT = 8_000
LOG_ROW_LIMIT = 50_000
TRACE_ROW_LIMIT = 50_000

STATEFUL_BACKEND_TERMS = {
    "database",
    "db",
    "kafka",
    "mongo",
    "mysql",
    "postgres",
    "postgresql",
    "redis",
    "valkey",
}

STATEFUL_BACKEND_ERROR_TERMS = {
    "authentication",
    "can't access",
    "connect",
    "connection",
    "database",
    "password",
    "queue",
    "redis",
    "storage",
    "wrongpass",
}

INACTIVE_FEATURE_FLAG_VARIANTS = {"", "0", "false", "none", "null", "off"}


def ingest_metrics(graph: IncidentGraph, scenario_path: Path) -> None:
    metrics_dir = scenario_path / "metrics"

    if not metrics_dir.exists():
        return

    for path in sorted(metrics_dir.glob("*.tsv")):
        entity = _find_node_for_metric_file(path)

        if not entity:
            continue

        kind, name = entity
        rel_path = relative_path(scenario_path, path)
        metric_counts: Counter[str] = Counter()
        status_error_rows = 0
        suspicious_rows = 0
        namespace = ""

        for idx, row in enumerate(iter_tsv(path) or []):
            if idx >= METRIC_ROW_LIMIT:
                break

            metric_name = str(row.get("metric_name") or "")
            metric_counts[metric_name] += 1
            namespace = namespace or str(row.get("namespace") or "")
            status_code = str(row.get("status_code") or "")
            metric_text = metric_name.lower()

            if status_code.startswith("5"):
                status_error_rows += 1

            if (
                has_any(metric_text, ALERT_TERMS | ERROR_TERMS | RESOURCE_TERMS | CRASH_TERMS)
                or status_code.startswith("5")
                or has_any(metric_text, {"restart", "throttl", "oom", "latency", "error"})
            ):
                suspicious_rows += 1

        node = ensure_node(
            graph,
            kind=kind,
            name=name,
            namespace=namespace,
            evidence_path=rel_path,
            category="metrics",
            summary=f"Metric file with {sum(metric_counts.values())} sampled rows",
        )
        node.attributes["metric_names"] = list(metric_counts.keys())[:20]

        if suspicious_rows or status_error_rows:
            node.affected_score += min(10 + suspicious_rows // 20 + status_error_rows // 10, 180)
            node.signals.add("metric anomaly or error signal")

            if status_error_rows:
                node.signals.add("5xx metric signal")

            if kind == "Pod":
                add_hypothesis(node, "pod_metric_anomaly", 120, "pod metric anomaly")


def _find_node_for_metric_file(path: Path) -> tuple[str, str] | None:
    name = path.name

    if name.startswith("pod_") and name.endswith("_raw.tsv"):
        return "Pod", name[len("pod_") : -len("_raw.tsv")]

    if name.startswith("service_") and name.endswith("_raw.tsv"):
        return "Service", name[len("service_") : -len("_raw.tsv")]

    return None


def ingest_alerts(graph: IncidentGraph, scenario_path: Path) -> None:
    alerts_dir = scenario_path / "alerts"

    if not alerts_dir.exists():
        return

    for path in sorted(alerts_dir.glob("*.json")):
        rel_path = relative_path(scenario_path, path)

        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            continue

        for alert in _iter_alert_dicts(data):
            labels = alert.get("labels") if isinstance(alert.get("labels"), dict) else {}
            alert_name = str(labels.get("alertname") or "alert")
            namespace = str(labels.get("namespace") or labels.get("k8s_namespace_name") or "")
            summary = f"Alert {alert_name}"
            candidates = [
                ("Pod", labels.get("pod")),
                ("Service", labels.get("service_name") or labels.get("service")),
                ("Deployment", labels.get("deployment")),
                ("HorizontalPodAutoscaler", labels.get("horizontalpodautoscaler")),
                ("Namespace", namespace if namespace else None),
            ]

            for kind, name in candidates:
                if not name:
                    continue

                node = ensure_node(
                    graph,
                    kind=kind,
                    name=str(name),
                    namespace=namespace if kind != "Namespace" else "",
                    evidence_path=rel_path,
                    category="alerts",
                    summary=summary,
                )
                node.affected_score += 20 if kind in {"Pod", "Service"} else 8
                node.signals.add("alert/latency/error-rate")

                if kind == "HorizontalPodAutoscaler":
                    node.signals.add("autoscaling alert")
                    add_hypothesis(node, "autoscaling_policy", 350, "alert references HPA")


def _iter_alert_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("labels"), dict):
            yield value

        for child in value.values():
            yield from _iter_alert_dicts(child)

    elif isinstance(value, list):
        for child in value:
            yield from _iter_alert_dicts(child)


def ingest_logs(graph: IncidentGraph, scenario_path: Path) -> None:
    path = scenario_path / "otel_logs_raw.tsv"
    rel_path = relative_path(scenario_path, path)

    for idx, row in enumerate(iter_tsv(path) or []):
        if idx >= LOG_ROW_LIMIT:
            break

        body = str(row.get("Body") or "")
        text = body.lower()

        if not has_any(text, ERROR_TERMS | CONFIG_TERMS | RESOURCE_TERMS | NETWORK_TERMS | CRASH_TERMS):
            continue

        attrs = parse_mapping(row.get("ResourceAttributes", ""))
        refs = _node_refs_from_resource_attributes(attrs)
        service_name = str(row.get("ServiceName") or refs["service"] or "")
        namespace = refs["namespace"]
        timestamp = row.get("TimestampTime") or row.get("Timestamp")
        nodes: list[GraphNode] = []

        if service_name:
            nodes.append(
                ensure_node(
                    graph,
                    kind="Service",
                    name=service_name,
                    namespace=namespace,
                    evidence_path=rel_path,
                    category="logs",
                    summary=f"Log signal: {body[:220]}",
                    timestamp=timestamp,
                )
            )

        if refs["pod"]:
            nodes.append(
                ensure_node(
                    graph,
                    kind="Pod",
                    name=refs["pod"],
                    namespace=namespace,
                    evidence_path=rel_path,
                    category="logs",
                    summary=f"Log signal: {body[:220]}",
                    timestamp=timestamp,
                )
            )

        if refs["deployment"]:
            nodes.append(
                ensure_node(
                    graph,
                    kind="Deployment",
                    name=refs["deployment"],
                    namespace=namespace,
                    evidence_path=rel_path,
                    category="logs",
                    summary=f"Log signal: {body[:220]}",
                    timestamp=timestamp,
                )
            )

        for node in nodes:
            node.affected_score += 12
            mark_from_text(node, body)

        for source in nodes:
            for target in nodes:
                add_edge(
                    graph,
                    source=source.key.id,
                    target=target.key.id,
                    relation="same-log-context",
                    evidence_path=rel_path,
                    confidence=0.35,
                )


def _node_refs_from_resource_attributes(attrs: dict[str, Any]) -> dict[str, str]:
    return {
        "namespace": str(attrs.get("k8s.namespace.name") or ""),
        "pod": str(attrs.get("k8s.pod.name") or ""),
        "deployment": str(attrs.get("k8s.deployment.name") or ""),
        "service": str(attrs.get("service.name") or ""),
    }


def _feature_flag_events(raw_value: str) -> list[dict[str, str]]:
    text = str(raw_value or "").strip()

    if not text or "feature_flag.key" not in text:
        return []

    try:
        loaded = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []

    items = loaded if isinstance(loaded, list) else [loaded]
    events = []

    for item in items:
        if not isinstance(item, dict):
            continue

        key = str(item.get("feature_flag.key") or "")

        if not key:
            continue

        events.append(
            {
                "key": key,
                "provider": str(item.get("feature_flag.provider.name") or ""),
                "variant": str(item.get("feature_flag.result.variant") or ""),
            }
        )

    return events


def _record_feature_flag_evidence(
    graph: IncidentGraph,
    source_node: GraphNode,
    namespace: str,
    raw_events_attrs: str,
    rel_path: str,
) -> None:
    def append_finding(attribute_name: str, finding: dict[str, str]) -> None:
        values = config_node.attributes.setdefault(attribute_name, [])

        if not isinstance(values, list):
            return

        key = (finding["path"], finding["value"], finding["status"])

        if any((item.get("path"), item.get("value"), item.get("status")) == key for item in values if isinstance(item, dict)):
            return

        if len(values) < 40:
            values.append(finding)

    for event in _feature_flag_events(raw_events_attrs):
        if event["provider"].lower() != "flagd":
            continue

        config_node = ensure_node(
            graph,
            kind="ConfigMap",
            name="flagd-config",
            namespace=namespace,
            evidence_path=rel_path,
            category="traces",
            summary=f"Feature flag {event['key']} evaluated with variant {event['variant']}",
        )
        finding = {
            "path": f"feature_flag.{event['key']}.result.variant",
            "value": event["variant"],
            "reason": "runtime feature flag evaluation",
            "status": "active",
        }
        variant_key = event["variant"].strip().lower()

        if variant_key in INACTIVE_FEATURE_FLAG_VARIANTS:
            finding["status"] = "inactive"
            finding["reason"] = "inactive runtime feature flag evaluation"
            config_node.signals.add("inactive configuration context")
            append_finding("inactive_configuration_context", finding)
        else:
            config_node.signals.add("active configuration content signal")
            config_node.signals.add("configuration content signal")
            config_node.signals.add("feature flag runtime signal")
            append_finding("active_configuration_findings", finding)
            append_finding("configuration_findings", finding)
            add_hypothesis(
                config_node,
                "configuration_or_secret",
                1500,
                "runtime feature flag evaluation",
            )

        add_edge(
            graph,
            source=config_node.key.id,
            target=source_node.key.id,
            relation="feature-flag-affects-service",
            evidence_path=rel_path,
            confidence=0.8 if finding["status"] == "active" else 0.45,
            summary=f"{source_node.name} evaluated feature flag {event['key']} as {event['variant']}",
        )


def ingest_traces(graph: IncidentGraph, scenario_path: Path) -> None:
    path = scenario_path / "otel_traces_raw.tsv"
    rel_path = relative_path(scenario_path, path)
    name_index = _build_name_index(graph, {"Service", "Deployment", "Pod"})
    dependency_index = _build_dependency_name_index(graph, {"Service", "Deployment", "Pod"})

    for idx, row in enumerate(iter_tsv(path) or []):
        if idx >= TRACE_ROW_LIMIT:
            break

        source_service = str(row.get("ServiceName") or "")
        span_name = str(row.get("SpanName") or "")
        status_code = str(row.get("StatusCode") or "")
        status_message = str(row.get("StatusMessage") or "")
        raw_span_attrs = str(row.get("SpanAttributes") or "")
        raw_events_attrs = str(row.get("Events.Attributes") or "")
        quick_text = " ".join(
            [
                span_name,
                status_code,
                status_message,
                raw_span_attrs[:1500],
                raw_events_attrs[:1500],
            ]
        )
        quick_is_error = status_code.lower() not in {"", "unset", "ok"} or has_any(
            quick_text.lower(),
            ERROR_TERMS,
        )
        has_dependency_fields = any(
            key in raw_span_attrs
            for key in ["server.address", "net.peer.name", "peer.service", "db.system"]
        )
        has_feature_flag_fields = "feature_flag.key" in raw_events_attrs

        if not quick_is_error and not has_dependency_fields and not has_feature_flag_fields:
            continue

        span_attrs = parse_mapping(raw_span_attrs)
        resource_attrs = parse_mapping(row.get("ResourceAttributes", ""))
        refs = _node_refs_from_resource_attributes(resource_attrs)
        namespace = refs["namespace"]
        dependency_names = _dependency_names(span_attrs)
        text = " ".join(
            [
                span_name,
                status_code,
                status_message,
                str(span_attrs),
                str(row.get("Events.Attributes") or ""),
            ]
        )
        is_error = quick_is_error

        if not source_service and refs["service"]:
            source_service = refs["service"]

        if not source_service:
            continue

        source = ensure_node(
            graph,
            kind="Service",
            name=source_service,
            namespace=namespace,
            evidence_path=rel_path,
            category="traces",
            summary=f"Trace span {span_name[:160]}",
        )

        if is_error:
            source.affected_score += 8
            source.signals.add("trace error or timeout")

        _record_feature_flag_evidence(
            graph=graph,
            source_node=source,
            namespace=namespace,
            raw_events_attrs=raw_events_attrs,
            rel_path=rel_path,
        )

        targets = _matching_nodes_by_name(
            text=text,
            name_index=name_index,
        ) if is_error else []
        dependency_targets = _dependency_nodes_by_name(dependency_index, dependency_names, namespace)
        dependency_target_ids = {target.key.id for target in dependency_targets}
        targets.extend(dependency_targets)
        targets = list({target.key.id: target for target in targets}.values())
        is_dependency_span = bool(dependency_names) or bool(span_attrs.get("db.system"))

        for target in targets:
            if target.key.id == source.key.id:
                continue

            add_edge(
                graph,
                source=source.key.id,
                target=target.key.id,
                relation="trace-dependency",
                evidence_path=rel_path,
                confidence=0.55 if is_error else 0.45,
                summary=f"Trace span references {target.name}",
            )

            if is_error:
                target.affected_score += 4
                target.signals.add("trace error or timeout")

            if is_dependency_span and target.key.id in dependency_target_ids:
                target.signals.add("backend dependency target")
                add_hypothesis(
                    target,
                    "backend_dependency",
                    800 if source.affected_score > 0 or is_error else 450,
                    "target of traced backend dependency",
                )
                add_edge(
                    graph,
                    source=target.key.id,
                    target=source.key.id,
                    relation="backend-dependency-for-service",
                    evidence_path=rel_path,
                    confidence=0.8 if is_error else 0.7,
                    summary=f"{source.name} depends on {target.name}",
                )

                if is_error and _is_stateful_backend(target) and has_any(
                    text.lower(),
                    STATEFUL_BACKEND_ERROR_TERMS,
                ):
                    target.signals.add("stateful backend failure")
                    add_hypothesis(
                        target,
                        "stateful_backend_dependency",
                        1200,
                        "traced stateful backend failure",
                    )


def _build_name_index(
    graph: IncidentGraph,
    kinds: set[str],
) -> dict[str, list[tuple[str, GraphNode]]]:
    index: dict[str, list[tuple[str, GraphNode]]] = {}

    for node in graph.nodes.values():
        if node.kind not in kinds:
            continue

        if is_observability_name(node.name):
            continue

        token = normalize_token(node.name)

        if len(token) <= 3:
            continue

        parts = [part for part in token.split("-") if len(part) > 3]
        anchor = max(parts, key=len) if parts else token
        index.setdefault(anchor, []).append((token, node))

    return index


def _build_dependency_name_index(
    graph: IncidentGraph,
    kinds: set[str],
) -> dict[str, list[GraphNode]]:
    index: dict[str, list[GraphNode]] = {}

    for node in graph.nodes.values():
        if node.kind not in kinds or is_observability_name(node.name):
            continue

        keys = {
            normalize_token(node.name),
            _workload_base(node.name),
        }

        for key in keys:
            if len(key) <= 2:
                continue

            index.setdefault(key, []).append(node)

    return index


def _dependency_names(span_attrs: dict[str, Any]) -> list[str]:
    names = []

    for key in [
        "server.address",
        "net.peer.name",
        "peer.service",
        "db.connection_string",
    ]:
        value = span_attrs.get(key)

        if value:
            names.append(str(value))

    return list(dict.fromkeys(names))


def _is_stateful_backend(node: GraphNode) -> bool:
    token = normalize_token(f"{node.name} {node.kind}")
    return any(term in token for term in STATEFUL_BACKEND_TERMS)


def _workload_base(name: str) -> str:
    token = normalize_token(name)
    token = re.sub(r"-[a-z0-9]{8,10}-[a-z0-9]{4,6}$", "", token)
    return token


def _dependency_nodes_by_name(
    dependency_index: dict[str, list[GraphNode]],
    names: list[str],
    namespace: str,
) -> list[GraphNode]:
    normalized_names = {_workload_base(name) for name in names if name}

    if not normalized_names:
        return []

    matches = []

    for name in normalized_names:
        for node in dependency_index.get(name, []):
            if namespace and node.namespace and node.namespace != namespace:
                continue

            matches.append(node)

    unique = list({node.key.id: node for node in matches}.values())
    kind_order = {"Pod": 0, "Service": 1, "Deployment": 2}
    unique.sort(key=lambda node: (kind_order.get(node.kind, 9), -node.candidate_score))
    return unique[:8]


def _matching_nodes_by_name(
    text: str,
    name_index: dict[str, list[tuple[str, GraphNode]]],
) -> list[GraphNode]:
    normalized_text = normalize_token(text)

    if not normalized_text:
        return []

    matches: dict[str, GraphNode] = {}
    parts = {part for part in normalized_text.split("-") if len(part) > 3}

    for part in parts:
        for token, node in name_index.get(part, []):
            if token in normalized_text:
                matches[node.key.id] = node

    ordered = list(matches.values())
    ordered.sort(key=lambda item: (len(item.name), item.kind), reverse=True)
    return ordered[:8]
