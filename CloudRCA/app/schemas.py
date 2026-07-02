from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


AllowedSourceType = Literal[
    "alerts",
    "events",
    "logs",
    "metrics",
    "traces",
    "topology",
    "ground_truth_hidden",
    "other",
]


class RootCauseEntity(BaseModel):
    kind: str = Field(
        description="Type of suspected root-cause entity, e.g. Deployment, Pod, Service, ConfigMap, Node, NetworkPolicy."
    )
    name: str = Field(description="Name of the suspected root-cause entity.")
    namespace: str | None = Field(default=None, description="Kubernetes namespace if available.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence from 0.0 to 1.0.")


class EvidenceItem(BaseModel):
    source_type: AllowedSourceType
    source_path: str | None = Field(default=None, description="Relative file path, if known.")
    summary: str = Field(description="Short evidence summary.")
    supports_root_cause: bool = Field(description="Whether this evidence supports the suspected root cause.")

    @field_validator("source_type", mode="before")
    @classmethod
    def normalize_source_type(cls, value: object) -> str:
        """
        Make the schema tolerant to LLM variants like:
        k8s_events, kubernetes_events, prometheus_alerts, otel_traces, etc.
        """
        if value is None:
            return "other"

        normalized = str(value).strip().lower()
        normalized = normalized.replace("-", "_").replace(" ", "_")

        mapping = {
            "alert": "alerts",
            "prometheus_alert": "alerts",
            "prometheus_alerts": "alerts",

            "event": "events",
            "k8s_event": "events",
            "k8s_events": "events",
            "kubernetes_event": "events",
            "kubernetes_events": "events",

            "log": "logs",
            "application_log": "logs",
            "application_logs": "logs",
            "pod_log": "logs",
            "pod_logs": "logs",

            "metric": "metrics",
            "prometheus_metric": "metrics",
            "prometheus_metrics": "metrics",

            "trace": "traces",
            "otel_trace": "traces",
            "otel_traces": "traces",
            "opentelemetry_trace": "traces",
            "opentelemetry_traces": "traces",

            "service_topology": "topology",
            "dependency_graph": "topology",
            "graph": "topology",

            "ground_truth": "ground_truth_hidden",
            "groundtruth": "ground_truth_hidden",
        }

        allowed = {
            "alerts",
            "events",
            "logs",
            "metrics",
            "traces",
            "topology",
            "ground_truth_hidden",
            "other",
        }

        mapped = mapping.get(normalized, normalized)

        if mapped in allowed:
            return mapped

        return "other"


class DiagnosisResult(BaseModel):
    scenario_id: str
    incident_summary: str
    root_cause_entities: list[RootCauseEntity]
    evidence: list[EvidenceItem]
    reasoning_summary: str
    recommended_remediation: list[str]
    should_auto_remediate: bool = False
    limitations: list[str] = []


class EvidenceBundle(BaseModel):
    scenario_id: str
    scenario_path: str
    files_seen: int
    evidence_text: str