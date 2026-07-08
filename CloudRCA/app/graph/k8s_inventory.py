from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.graph.ingest import has_any, iter_tsv, parse_json, relative_path
from app.graph.records import GraphNode, IncidentGraph
from app.graph.relationships import add_hypothesis, ensure_node, extract_selector
from app.graph.parsers import CONTROL_KINDS, normalize_kind


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

ALERT_TERMS = {
    "alert",
    "latency",
    "requestlatency",
    "request latency",
    "requesterrorrate",
    "request error rate",
    "error rate",
    "5xx",
}

ERROR_TERMS = {
    "error",
    "failed",
    "failure",
    "exception",
    "timeout",
    "deadline_exceeded",
    "unavailable",
    "refused",
    "reset",
    "5xx",
    "500",
    "502",
    "503",
    "504",
}

CONFIG_TERMS = {
    "configmap",
    "config map",
    "secret",
    "feature flag",
    "featureflag",
    "defaultvariant",
    "configuration",
    "rollout",
    "toggle",
    "endpoint",
    "certificate",
    "tls",
    "token",
}

RESOURCE_TERMS = {
    "resourcequota",
    "resource quota",
    "limitrange",
    "limit range",
    "quota",
    "exceeded quota",
    "failedcreate",
    "unschedulable",
    "insufficient memory",
    "memory pressure",
    "oom",
    "oomkilled",
    "cpu throttling",
}

CRASH_TERMS = {
    "crashloopbackoff",
    "crash",
    "restart",
    "back-off",
    "readiness probe",
    "liveness probe",
    "imagepullbackoff",
    "errimagepull",
}

IMAGE_PULL_TERMS = {
    "imagepullbackoff",
    "errimagepull",
    "back-off pulling image",
    "couldn't parse image name",
    "failed to pull image",
    "failed to apply default image tag",
    "inspectfailed",
    "invalid image",
    "invalid reference format",
    "invalidimagename",
}

CONTAINER_STARTUP_TERMS = {
    "crashloopbackoff",
    "back-off restarting failed container",
    "exec format error",
    "invalid command",
    "executable file not found",
}

NETWORK_TERMS = {
    "networkpolicy",
    "network policy",
    "network-delay",
    "network delay",
    "network-partition",
    "network partition",
    "connection refused",
    "unreachable",
    "dns",
    "ingress",
    "egress",
}

TRAFFIC_SOURCE_TERMS = {
    "locust",
    "load-generator",
    "load generator",
    "traffic generator",
    "k6",
    "jmeter",
}

TRAFFIC_DRIVER_ENV_NAMES = {
    "LOCUST_AUTOSTART",
    "LOCUST_SPAWN_RATE",
    "LOCUST_USERS",
    "K6_ITERATIONS",
    "K6_VUS",
    "JMETER_THREADS",
}

BENIGN_STATUS_REASONS = {
    "completed",
    "minimumreplicasavailable",
    "newreplicasetavailable",
    "podcompleted",
    "replicasetupdated",
}

WORKLOAD_SPEC_KINDS = {
    "Deployment",
    "StatefulSet",
    "DaemonSet",
    "ReplicaSet",
    "Pod",
}

ENDPOINT_ENV_MARKERS = {
    "ADDR",
    "ADDRESS",
    "ENDPOINT",
    "HOST",
    "PORT",
    "URL",
    "URI",
    "DSN",
    "CONNECTION",
}

PLACEHOLDER_PORTS = {0, 9999}

CONFIG_FINDING_TERMS = {
    "auth",
    "delay",
    "error",
    "fail",
    "failure",
    "fault",
    "gc",
    "invalid",
    "latency",
    "memory",
    "password",
    "queue",
    "stress",
    "timeout",
    "wrong",
}

CONFIG_VALUE_KEYS = {
    "defaultVariant",
    "state",
    "enabled",
    "value",
}

ACTIVE_CONFIG_VALUES = {"enabled", "on", "true", "yes", "1"}
INACTIVE_CONFIG_VALUES = {"disabled", "off", "false", "no", "0"}


def object_text(obj: dict[str, Any], max_chars: int = 40_000) -> str:
    try:
        return json.dumps(obj, sort_keys=True)[:max_chars]
    except TypeError:
        return str(obj)[:max_chars]


def _short_value(value: Any, redact: bool = False, max_chars: int = 120) -> str:
    if redact:
        return "<redacted>"

    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True)
    else:
        text = str(value)

    text = re.sub(r"\s+", " ", text).strip()
    return text[: max_chars - 3] + "..." if len(text) > max_chars else text


def _parse_config_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    text = value.strip()

    if not text or text[:1] not in {"{", "["}:
        return value

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _walk_config_leaves(value: Any, path: list[str]) -> list[tuple[list[str], Any]]:
    value = _parse_config_value(value)

    if isinstance(value, dict):
        leaves = []

        for key, child in value.items():
            leaves.extend(_walk_config_leaves(child, path + [str(key)]))

        return leaves

    if isinstance(value, list):
        leaves = []

        for index, child in enumerate(value[:20]):
            leaves.extend(_walk_config_leaves(child, path + [str(index)]))

        return leaves

    return [(path, value)]


def _normalized_config_value(value: Any) -> str:
    return str(value).strip().lower()


def _config_finding(path: list[str], value: Any, redact: bool = False) -> dict[str, str] | None:
    path_text = ".".join(path)
    lower_path = path_text.lower()
    lower_value = str(value).lower()
    value_key = _normalized_config_value(value)
    leaf = path[-1] if path else ""
    fault_related_path = any(term in lower_path for term in CONFIG_FINDING_TERMS)
    fault_related_value = any(term in lower_value for term in CONFIG_FINDING_TERMS)
    in_feature_flag = ".flags." in f".{lower_path}."
    status = "context"
    reason = ""

    if leaf == "defaultVariant":
        if fault_related_path and value_key in ACTIVE_CONFIG_VALUES:
            status = "active"
            reason = "active fault flag default"
        elif fault_related_path and value_key in INACTIVE_CONFIG_VALUES:
            status = "inactive"
            reason = "inactive fault flag default"
        else:
            reason = "feature flag default"

    elif in_feature_flag and fault_related_path:
        reason = "feature flag metadata"

    elif leaf in CONFIG_VALUE_KEYS and fault_related_path:
        if value_key in ACTIVE_CONFIG_VALUES:
            status = "active"
            reason = "active fault-related config value"
        elif value_key in INACTIVE_CONFIG_VALUES:
            status = "inactive"
            reason = "inactive fault-related config value"
        else:
            status = "active"
            reason = "fault-related config value"

    elif leaf.lower() == "description" and fault_related_path:
        reason = "fault-related config description"

    elif fault_related_path:
        status = "active"
        reason = "fault-related config key"

    elif fault_related_value:
        status = "active"
        reason = "fault-related config value"

    if not reason:
        return None

    return {
        "path": path_text,
        "value": _short_value(value, redact=redact),
        "reason": reason,
        "status": status,
    }


def _config_finding_priority(finding: dict[str, str]) -> tuple[int, str]:
    path = finding.get("path", "").lower()
    value = finding.get("value", "").lower()
    leaf = path.rsplit(".", 1)[-1]

    if finding.get("status") == "active" and leaf == "defaultvariant":
        return 0, path

    if finding.get("status") == "active":
        return 1, path

    if finding.get("status") == "inactive" and leaf == "defaultvariant":
        return 2, path

    if leaf == "description":
        return 3, path

    return 4, path


def extract_configuration_findings(kind: str, data: Any, limit: int = 24) -> list[dict[str, str]]:
    if not isinstance(data, dict):
        return []

    findings = []
    redact = kind == "Secret"

    for key, value in data.items():
        for path, leaf_value in _walk_config_leaves(value, ["data", str(key)]):
            finding = _config_finding(path, leaf_value, redact=redact)

            if not finding:
                continue

            findings.append(finding)

    findings.sort(key=_config_finding_priority)
    return findings[:limit]


def mark_from_text(node: GraphNode, text: str) -> None:
    lower = text.lower()

    if has_any(lower, ALERT_TERMS):
        node.affected_score += 20
        node.signals.add("alert/latency/error-rate")

    if has_any(lower, ERROR_TERMS):
        node.affected_score += 8
        node.signals.add("error/failure/timeout")

    if has_any(lower, CRASH_TERMS):
        node.signals.add("crash/restart/pod disruption")
        add_hypothesis(node, "crash_or_pod_disruption", 160, "crash or restart signal")

    if has_any(lower, IMAGE_PULL_TERMS):
        node.signals.add("image pull failure")
        add_hypothesis(node, "workload_configuration", 600, "image pull failure")

    if has_any(lower, CONTAINER_STARTUP_TERMS):
        node.signals.add("container startup failure")
        add_hypothesis(node, "workload_configuration", 600, "container startup failure")

    if has_any(lower, RESOURCE_TERMS):
        node.signals.add("resource saturation or quota")
        add_hypothesis(node, "resource_or_quota", 180, "resource or quota signal")

    if has_any(lower, NETWORK_TERMS):
        node.signals.add("network/dependency disruption")
        add_hypothesis(node, "network_or_dependency", 160, "network disruption signal")

    if has_any(lower, CONFIG_TERMS):
        node.signals.add("configuration/control-plane signal")

        if node.kind in {"ConfigMap", "Secret"}:
            add_hypothesis(
                node,
                "configuration_or_secret",
                550,
                "configuration object contains relevant keys or values",
            )


def _container_specs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    containers = []

    def add_from(pod_spec: Any) -> None:
        if not isinstance(pod_spec, dict):
            return

        for key in ["containers", "initContainers"]:
            values = pod_spec.get(key)

            if isinstance(values, list):
                containers.extend(item for item in values if isinstance(item, dict))

    add_from(spec)

    template = spec.get("template") if isinstance(spec.get("template"), dict) else {}
    add_from(template.get("spec"))
    return containers


def _env_pairs(spec: dict[str, Any]) -> list[tuple[str, str]]:
    pairs = []

    for container in _container_specs(spec):
        env = container.get("env")

        if not isinstance(env, list):
            continue

        for item in env:
            if not isinstance(item, dict):
                continue

            name = str(item.get("name") or "")
            value = str(item.get("value") or "")

            if name:
                pairs.append((name, value))

    return pairs


def _mark_traffic_source(node: GraphNode, obj: dict[str, Any]) -> None:
    spec = obj.get("spec") if isinstance(obj.get("spec"), dict) else {}
    metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    labels = metadata.get("labels") if isinstance(metadata.get("labels"), dict) else {}
    env_pairs = _env_pairs(spec)
    role_text = object_text(
        {
            "name": node.name,
            "labels": labels,
            "containers": [
                {
                    "name": container.get("name"),
                    "image": container.get("image"),
                }
                for container in _container_specs(spec)
            ],
        },
        max_chars=20_000,
    ).lower()
    has_driver_env = any(name.upper() in TRAFFIC_DRIVER_ENV_NAMES for name, _ in env_pairs)
    has_traffic_role = has_any(role_text, TRAFFIC_SOURCE_TERMS)

    if not has_driver_env and not has_traffic_role:
        return

    node.signals.add("traffic source workload")
    node.reasons.add("traffic source workload")

    for name, value in env_pairs:
        env_name = name.upper()

        if "USERS" not in env_name and "VUS" not in env_name:
            continue

        match = re.search(r"\d+", value)

        if not match:
            continue

        user_count = int(match.group(0))
        node.attributes["traffic_user_count"] = max(
            int(node.attributes.get("traffic_user_count") or 0),
            user_count,
        )

        if user_count >= 100:
            node.signals.add("high traffic volume configuration")
            add_hypothesis(node, "traffic_volume", 1200, "high configured user count")


def _is_endpoint_env(name: str) -> bool:
    env_name = name.upper()
    return any(marker in env_name for marker in ENDPOINT_ENV_MARKERS)


def _port_values(value: str) -> list[int]:
    ports = []

    for match in re.finditer(r"(?::|^)(\d{1,5})(?:\D|$)", value):
        port = int(match.group(1))

        if 0 <= port <= 65535:
            ports.append(port)

    return ports


def _container_command_text(spec: dict[str, Any]) -> str:
    parts = []

    for container in _container_specs(spec):
        for key in ["image", "command", "args"]:
            value = container.get(key)

            if isinstance(value, list):
                parts.extend(str(item) for item in value)
            elif value:
                parts.append(str(value))

    return " ".join(parts)


def _mark_workload_spec_risk(node: GraphNode, obj: dict[str, Any]) -> None:
    if node.kind not in WORKLOAD_SPEC_KINDS:
        return

    spec = obj.get("spec") if isinstance(obj.get("spec"), dict) else {}

    for name, value in _env_pairs(spec):
        if not _is_endpoint_env(name):
            continue

        node.signals.add("workload configuration surface")

        ports = _port_values(value)
        if any(port in PLACEHOLDER_PORTS for port in ports):
            node.signals.add("workload endpoint misconfiguration")
            add_hypothesis(
                node,
                "workload_configuration",
                1500,
                "placeholder endpoint port in workload environment",
            )

    command_text = _container_command_text(spec).lower()

    if re.search(r"\b(?:invalid|nonexistent|doesnotexist|not-found)\b", command_text):
        node.signals.add("workload rollout failure")
        add_hypothesis(
            node,
            "workload_configuration",
            1200,
            "fault-like image or command value in workload spec",
        )


def _mark_status_faults(node: GraphNode, obj: dict[str, Any]) -> None:
    status = obj.get("status") if isinstance(obj.get("status"), dict) else {}
    fault_texts = []

    for key in ["message", "reason", "phase"]:
        value = status.get(key)

        if value:
            fault_texts.append(str(value))

    for condition in status.get("conditions") or []:
        if not isinstance(condition, dict):
            continue

        reason = str(condition.get("reason") or "")
        message = str(condition.get("message") or "")
        condition_status = str(condition.get("status") or "").lower()
        reason_key = reason.lower()

        if condition_status == "true" and reason_key in BENIGN_STATUS_REASONS:
            continue

        if condition_status == "false" or has_any(f"{reason} {message}", ERROR_TERMS | CRASH_TERMS):
            fault_texts.extend([reason, message])

    for key in ["unavailableReplicas", "readyReplicas", "replicas"]:
        if key in status:
            fault_texts.append(f"{key}={status[key]}")

    for container_status in status.get("containerStatuses") or []:
        if not isinstance(container_status, dict):
            continue

        restart_count = int(container_status.get("restartCount") or 0)

        if restart_count > 0:
            fault_texts.append(f"restart count {restart_count}")

        for state_key in ["state", "lastState"]:
            state = container_status.get(state_key)

            if not isinstance(state, dict):
                continue

            for details in state.values():
                if not isinstance(details, dict):
                    continue

                reason = str(details.get("reason") or "")
                message = str(details.get("message") or "")

                if reason.lower() in BENIGN_STATUS_REASONS:
                    continue

                fault_texts.extend([reason, message])

    fault_text = " ".join(text for text in fault_texts if text)

    if fault_text:
        mark_from_text(node, fault_text)


def mark_k8s_object_node(node: GraphNode, obj: dict[str, Any]) -> None:
    kind = node.kind

    if kind in CONTROL_KINDS or kind in ROOT_CAPABLE_KINDS:
        node.signals.add("control or workload object")

    if kind in {"ConfigMap", "Secret"}:
        node.signals.add("kubernetes configuration object")
        data = obj.get("data", {})
        data_text = object_text(data if isinstance(data, dict) else {}, max_chars=30_000)
        findings = extract_configuration_findings(kind, data)
        active_findings = [item for item in findings if item.get("status") == "active"]
        inactive_findings = [item for item in findings if item.get("status") == "inactive"]
        context_findings = [item for item in findings if item.get("status") == "context"]

        if active_findings:
            node.attributes["configuration_findings"] = active_findings[:12]
            node.attributes["active_configuration_findings"] = active_findings[:12]
            node.signals.add("active configuration content signal")
            node.signals.add("configuration content signal")
            add_hypothesis(
                node,
                "configuration_or_secret",
                1700,
                "configuration data contains active fault indicators",
            )

        if inactive_findings:
            node.attributes["inactive_configuration_context"] = inactive_findings[:12]
            node.signals.add("inactive configuration context")
            node.reasons.add("configuration data contains inactive fault-related settings")

        if context_findings:
            node.attributes["configuration_context"] = context_findings[:12]

        if has_any(data_text, CONFIG_TERMS) or re.search(r"[a-z0-9_-]+failure", data_text.lower()):
            node.signals.add("configuration/control-plane signal")

        if active_findings:
            mark_from_text(node, data_text)

    if kind == "NetworkPolicy":
        node.signals.add("network policy")
        add_hypothesis(node, "network_policy", 900, "network policy object")

    if kind in {"ResourceQuota", "LimitRange"}:
        node.signals.add("namespace-level resource policy")
        add_hypothesis(node, "namespace_resource_policy", 900, "namespace resource policy object")

    if kind == "HorizontalPodAutoscaler":
        node.signals.add("autoscaling policy")
        add_hypothesis(node, "autoscaling_policy", 850, "horizontal pod autoscaler object")

    if kind == "Namespace":
        add_hypothesis(node, "namespace_scope", 200, "namespace object")

    _mark_workload_spec_risk(node, obj)
    _mark_status_faults(node, obj)
    _mark_traffic_source(node, obj)


def ingest_k8s_objects(graph: IncidentGraph, scenario_path: Path) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    path = scenario_path / "k8s_objects_raw.tsv"
    rel_path = relative_path(scenario_path, path)

    for row in iter_tsv(path) or []:
        obj = parse_json(row.get("Body", ""))

        if not isinstance(obj, dict):
            continue

        kind = normalize_kind(str(obj.get("kind") or "Unknown"))
        metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
        name = str(metadata.get("name") or "")
        namespace = str(metadata.get("namespace") or "")

        if not name:
            continue

        timestamp = row.get("TimestampTime") or row.get("Timestamp")
        node = ensure_node(
            graph,
            kind=kind,
            name=name,
            namespace=namespace,
            evidence_path=rel_path,
            category="kubernetes_objects",
            summary=f"{kind} object {name} parsed from Kubernetes snapshot",
            timestamp=timestamp,
        )
        labels = metadata.get("labels") if isinstance(metadata.get("labels"), dict) else {}
        node.labels.update({str(key): str(value) for key, value in labels.items()})
        node.attributes.setdefault("api_version", obj.get("apiVersion"))
        node.attributes.setdefault("resource_version", metadata.get("resourceVersion"))
        owner_refs = metadata.get("ownerReferences")

        if isinstance(owner_refs, list):
            node.attributes["owner_references"] = [
                {
                    "kind": item.get("kind"),
                    "name": item.get("name"),
                }
                for item in owner_refs
                if isinstance(item, dict)
            ][:10]

        selector = extract_selector(obj)

        if selector:
            node.attributes["selector"] = selector

        mark_k8s_object_node(node, obj)
        objects.append(
            {
                "node": node,
                "obj": obj,
                "kind": kind,
                "name": name,
                "namespace": namespace,
                "path": rel_path,
            }
        )

    return objects
