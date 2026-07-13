from __future__ import annotations

from collections import Counter
from typing import Any


PATTERN_BY_EVIDENCE = {
    "active_mutation": "chaos_or_fault_injection",
    "active_configuration": "configuration_or_feature_flag",
    "inactive_configuration": "configuration_context",
    "network_policy": "network_policy_block",
    "namespace_policy": "namespace_resource_policy",
    "scheduling_constraint": "pod_scheduling_failure",
    "endpoint_misconfiguration": "workload_endpoint_misconfiguration",
    "rollout_failure": "workload_rollout_failure",
    "autoscaling_policy": "autoscaling_policy",
    "stateful_backend": "stateful_backend_failure",
    "traffic_configuration": "traffic_source_overload",
}

GUIDANCE_BY_PATTERN = {
    "chaos_or_fault_injection": "Prefer active Chaos Mesh or scheduled mutation objects with paths to symptoms.",
    "configuration_or_feature_flag": "Prefer active referenced ConfigMaps or Secrets; inactive flags are context.",
    "network_policy_block": "Prefer NetworkPolicy when traffic blocking evidence reaches affected services.",
    "namespace_resource_policy": "Prefer Namespace or quota objects only with enforcement evidence.",
    "pod_scheduling_failure": "Prefer the highest-ranked pod with direct FailedScheduling evidence.",
    "workload_endpoint_misconfiguration": "Prefer the workload controller or pod with explicit endpoint misconfiguration.",
    "workload_rollout_failure": "Prefer the workload controller when pods inherit rollout or startup failure evidence.",
    "autoscaling_policy": "Prefer HPA objects with policy evidence and symptom reach.",
    "stateful_backend_failure": "Prefer the stateful backend only when it explains downstream symptoms.",
    "traffic_source_overload": "Prefer load-generator or traffic-source workload evidence over downstream symptoms.",
}


def classify_incident_pattern(candidate_table: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidate_table:
        return {
            "pattern": "unknown",
            "confidence": 0.2,
            "primary_evidence_type": "",
            "guidance": "No candidate table was available.",
            "evidence_type_counts": {},
        }

    top_selectable = [
        row
        for row in candidate_table[:6]
        if row.get("root_selectable") is True
    ]
    scope = top_selectable or candidate_table[:6]
    counts = Counter(str(row.get("evidence_type") or "") for row in scope)
    primary_type = str(scope[0].get("evidence_type") or "")
    pattern = PATTERN_BY_EVIDENCE.get(primary_type, "general_kubernetes_incident")

    confidence = 0.72

    if counts[primary_type] >= 2:
        confidence += 0.08

    if scope[0].get("selection_class") == "strong_root":
        confidence += 0.08

    if int(scope[0].get("causal_path_count") or 0) > 0:
        confidence += 0.05

    return {
        "pattern": pattern,
        "confidence": round(min(confidence, 0.95), 2),
        "primary_evidence_type": primary_type,
        "guidance": GUIDANCE_BY_PATTERN.get(pattern, "Prefer graph-ranked selectable roots with direct causal evidence."),
        "evidence_type_counts": dict(counts),
        "top_candidate_id": scope[0].get("id"),
        "top_candidate_kind": scope[0].get("kind"),
    }
