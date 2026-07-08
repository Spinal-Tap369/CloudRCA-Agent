
from __future__ import annotations

import re
from dataclasses import dataclass


K8S_KINDS = {
    "pod": "Pod",
    "service": "Service",
    "deployment": "Deployment",
    "replicaset": "ReplicaSet",
    "statefulset": "StatefulSet",
    "daemonset": "DaemonSet",
    "job": "Job",
    "cronjob": "CronJob",
    "configmap": "ConfigMap",
    "secret": "Secret",
    "namespace": "Namespace",
    "node": "Node",
    "ingress": "Ingress",
    "networkpolicy": "NetworkPolicy",
    "resourcequota": "ResourceQuota",
    "limitrange": "LimitRange",
    "horizontalpodautoscaler": "HorizontalPodAutoscaler",
}

CHAOS_KINDS = {
    "networkchaos": "NetworkChaos",
    "podchaos": "PodChaos",
    "stresschaos": "StressChaos",
    "dnschaos": "DNSChaos",
    "httpchaos": "HTTPChaos",
    "iochaos": "IOChaos",
    "timechaos": "TimeChaos",
    "jvmchaos": "JVMChaos",
    "schedule": "Schedule",
}

ALL_KINDS = {**K8S_KINDS, **CHAOS_KINDS}

CONTROL_KINDS = {
    "ConfigMap",
    "Secret",
    "Deployment",
    "StatefulSet",
    "DaemonSet",
    "NetworkPolicy",
    "ResourceQuota",
    "LimitRange",
    "Namespace",
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

OBSERVABILITY_TERMS = {
    "prometheus",
    "kube-prometheus",
    "cadvisor",
    "metrics/cadvisor",
    "otel-collector",
    "opentelemetry-collector",
    "grafana",
    "jaeger",
    "tempo",
    "loki",
}

BAD_VALUES = {
    "",
    "true",
    "false",
    "null",
    "none",
    "nan",
    "inf",
    "info",
    "debug",
    "warning",
    "error",
    "http",
    "https",
    "grpc",
    "tcp",
    "udp",
    "status",
    "duration",
    "timestamp",
}


@dataclass(frozen=True)
class EntityRef:
    kind: str
    name: str
    namespace: str = ""


def normalize_name(value: str) -> str:
    value = str(value).strip().strip('"').strip("'").strip()
    value = value.replace("\\", "/")

    if "/" in value and not value.startswith("otel-demo"):
        value = value.split("/")[-1]

    value = re.sub(r"^[./]+", "", value)
    value = re.sub(r"[,;{}\[\]()]$", "", value)
    return value


def normalize_kind(value: str) -> str:
    raw = str(value).strip()
    lower = raw.lower().replace("_", "").replace("-", "")

    if lower in ALL_KINDS:
        return ALL_KINDS[lower]

    return raw[:1].upper() + raw[1:] if raw else "Unknown"


def is_bad_name(value: str) -> bool:
    value = normalize_name(value)
    lower = value.lower()

    if lower in BAD_VALUES:
        return True

    if len(lower) < 2 or len(lower) > 160:
        return True

    if re.match(r"^\d{4}-\d{2}-\d{2}", lower):
        return True

    if re.match(r"^\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?$", lower):
        return True

    if re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", lower):
        return True

    if lower.startswith(("registry.", "docker.io/", "ghcr.io/", "quay.io/", "gcr.io/", "k8s.gcr.io/")):
        return True

    if lower.startswith(("http.", "rpc.", "process.", "container.", "go_")):
        return True

    return False


def looks_like_resource_name(value: str, allow_plain: bool = False) -> bool:
    value = normalize_name(value)
    lower = value.lower()

    if is_bad_name(lower):
        return False

    if allow_plain and re.match(r"^[a-z0-9][a-z0-9-]{1,}[a-z0-9]$", lower):
        return True

    if "-" in lower and re.match(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$", lower):
        return True

    if re.match(r"^[a-z0-9][a-z0-9-]*-[a-z0-9]{8,10}-[a-z0-9]{4,6}$", lower):
        return True

    return False


def infer_workload_base(name: str) -> str:
    lower = normalize_name(name).lower()
    lower = re.sub(r"-[a-z0-9]{8,10}-[a-z0-9]{4,6}$", "", lower)
    return lower


def infer_kind_from_name(name: str, context: str = "") -> str:
    text = f"{name} {context}".lower()

    for lower_kind, canonical in CHAOS_KINDS.items():
        if lower_kind in text:
            return canonical

    if "network-delay" in text or "network-partition" in text or "partition-" in text:
        return "NetworkChaos"

    if "memory-stress" in text or "cpu-stress" in text:
        return "StressChaos"

    if "pod-kill" in text or "pod-failure" in text:
        return "PodChaos"

    if "configmap" in text or "config map" in text:
        return "ConfigMap"

    if "secret" in text:
        return "Secret"

    if "namespace" in text:
        return "Namespace"

    if "deployment" in text:
        return "Deployment"

    if "service" in text or "service_" in text:
        return "Service"

    if "pod_" in text or re.match(r"^[a-z0-9][a-z0-9-]*-[a-z0-9]{8,10}-[a-z0-9]{4,6}$", name.lower()):
        return "Pod"

    return "Unknown"


def dedupe_entities(items: list[EntityRef]) -> list[EntityRef]:
    result: dict[tuple[str, str, str], EntityRef] = {}

    for item in items:
        if not item.name:
            continue

        key = (item.kind, item.name, item.namespace)
        result[key] = item

    return list(result.values())


def find_explicit_kind(text: str) -> str | None:
    lower = text.lower()

    # Supports:
    #   kind: ConfigMap
    #   "kind": "ConfigMap"
    #   resource_kind=ConfigMap
    match = re.search(
        r"""["']?(?:kind|resource_kind|object_kind)["']?\s*[:=]\s*["']?([A-Za-z][A-Za-z0-9]+)""",
        text,
    )

    if match:
        return normalize_kind(match.group(1))

    # Do not infer kind from any random occurrence of words like "secret",
    # "service", or "persistentvolumes". Kubernetes object identity needs an
    # explicit kind field or a path/filename signal.
    return None


def extract_entities_from_path(relative_path: str) -> list[EntityRef]:
    lower = relative_path.replace("\\", "/").lower()
    found: list[EntityRef] = []

    patterns = [
        (r"pod_([^/]+?)_raw\.tsv", "Pod"),
        (r"service_([^/]+?)_raw\.tsv", "Service"),
        (r"deployment_([^/]+?)_raw\.tsv", "Deployment"),
        (r"replicaset_([^/]+?)_raw\.tsv", "ReplicaSet"),
        (r"configmap_([^/]+?)_raw\.tsv", "ConfigMap"),
        (r"secret_([^/]+?)_raw\.tsv", "Secret"),
        (r"namespace_([^/]+?)_raw\.tsv", "Namespace"),
    ]

    for pattern, kind in patterns:
        for match in re.findall(pattern, lower):
            name = normalize_name(match)

            if looks_like_resource_name(name, allow_plain=True):
                found.append(EntityRef(kind=kind, name=name))

                if kind == "Pod":
                    base = infer_workload_base(name)
                    if base != name and looks_like_resource_name(base, allow_plain=True):
                        found.append(EntityRef(kind="Deployment", name=base))

    for match in re.findall(
        r"\b[a-z0-9][a-z0-9-]*(?:network-delay|network-partition|memory-stress|cpu-stress|pod-kill|pod-failure)[a-z0-9-]*\b",
        lower,
    ):
        found.append(EntityRef(kind=infer_kind_from_name(match, lower), name=match))

    for match in re.findall(r"\bpartition-[a-z0-9-]+\b", lower):
        found.append(EntityRef(kind="NetworkChaos", name=match))

    return dedupe_entities(found)


def _extract_jsonish_k8s_identity(line: str) -> list[EntityRef]:
    explicit_kind = find_explicit_kind(line)

    if not explicit_kind:
        return []

    found: list[EntityRef] = []

    # Handles JSON-ish Kubernetes object rows:
    #   "kind": "ConfigMap", "metadata": {"name": "flagd-config", "namespace": "otel-demo"}
    metadata_match = re.search(
        r"""["']?metadata["']?\s*[:=]\s*\{(?P<meta>.{0,1500}?)\}""",
        line,
    )

    search_area = metadata_match.group("meta") if metadata_match else line

    name_match = re.search(
        r"""["']?name["']?\s*[:=]\s*["']([^"']+)["']""",
        search_area,
    )

    namespace_match = re.search(
        r"""["']?namespace["']?\s*[:=]\s*["']([^"']+)["']""",
        search_area,
    )

    if not name_match:
        return []

    name = normalize_name(name_match.group(1))
    namespace = normalize_name(namespace_match.group(1)) if namespace_match else ""

    if looks_like_resource_name(name, allow_plain=True):
        found.append(EntityRef(kind=explicit_kind, name=name, namespace=namespace))

    return found


def extract_entities_from_line(line: str, relative_path: str = "") -> list[EntityRef]:
    context = f"{relative_path}\n{line}"
    explicit_kind = find_explicit_kind(context)
    found: list[EntityRef] = []

    found.extend(_extract_jsonish_k8s_identity(line))

    key_patterns = [
        (r"""["']?(?:service\.name|service_name|service|svc)["']?\s*[:=]\s*["']?([a-zA-Z0-9_.:/-]+)""", "Service"),
        (r"""["']?(?:k8s\.pod\.name|pod_name|pod)["']?\s*[:=]\s*["']?([a-zA-Z0-9_.:/-]+)""", "Pod"),
        (r"""["']?(?:deployment|deployment_name|workload|workload_name)["']?\s*[:=]\s*["']?([a-zA-Z0-9_.:/-]+)""", "Deployment"),
        (r"""["']?(?:configmap|config_map|configmap_name)["']?\s*[:=]\s*["']?([a-zA-Z0-9_.:/-]+)""", "ConfigMap"),
        # Do not extract generic namespace fields here.
        # metadata.namespace is an attribute of most Kubernetes objects, not a root-cause object.
        (r"""["']?(?:metadata\.name|metadata_name)["']?\s*[:=]\s*["']?([a-zA-Z0-9_.:/-]+)""", "Unknown"),
    ]

    for pattern, default_kind in key_patterns:
        for match in re.findall(pattern, line):
            name = normalize_name(match)

            if not looks_like_resource_name(name, allow_plain=True):
                continue

            kind = explicit_kind if default_kind == "Unknown" and explicit_kind else default_kind

            if kind == "Unknown":
                kind = infer_kind_from_name(name, context)

            found.append(EntityRef(kind=kind, name=name))

    for match in re.findall(
        r"\b[a-z0-9][a-z0-9-]*(?:network-delay|network-partition|memory-stress|cpu-stress|pod-kill|pod-failure)[a-z0-9-]*\b",
        line.lower(),
    ):
        found.append(EntityRef(kind=infer_kind_from_name(match, context), name=match))

    for match in re.findall(r"\bpartition-[a-z0-9-]+\b", line.lower()):
        found.append(EntityRef(kind="NetworkChaos", name=match))

    return dedupe_entities(found)


def is_observability_name(name: str) -> bool:
    lower = name.lower()
    return any(term in lower for term in OBSERVABILITY_TERMS)
