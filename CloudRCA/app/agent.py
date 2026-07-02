from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.llm_client import GeminiClient
from app.scenario_loader import inspect_scenario, is_text_file, normalize_path
from app.schemas import DiagnosisResult, EvidenceBundle
from app.topology_graph import build_sre_graph_context


SUSPICIOUS_KEYWORDS = [
    "error",
    "failed",
    "failure",
    "exception",
    "timeout",
    "latency",
    "unavailable",
    "crash",
    "crashloopbackoff",
    "back-off",
    "oom",
    "oomkilled",
    "restart",
    "restarted",
    "unhealthy",
    "readiness",
    "liveness",
    "probe",
    "evicted",
    "throttle",
    "throttling",
    "denied",
    "refused",
    "connection refused",
    "5xx",
    "500",
    "502",
    "503",
    "504",
]


CAUSAL_KEYWORDS = [
    "load-generator",
    "load generator",
    "loadgenerator",
    "traffic",
    "request volume",
    "high number of requests",
    "number of users",
    "users",
    "user count",
    "locust",
    "capacity",
    "overload",
    "overloaded",
    "autoscaler",
    "autoscaling",
    "replica",
    "replicas",
    "frontend-proxy",
    "requestlatency",
    "request latency",
    "requesterrorrate",
    "request error rate",
    "error rate",
    "deployment",
    "rollout",
    "configured",
    "configuration",
    "config",
    "changed",
    "change",
    "env",
    "environment",
    "limit",
    "limits",
    "resource",
    "cpu",
    "memory",
    "queue",
    "connection",
    "certificate",
    "secret",
    "networkpolicy",
    "network policy",
]


CATEGORY_FILE_HINTS = {
    "alerts": ["alert", "alerts", "prometheus"],
    "events": ["event", "events", "k8s_events", "kubernetes_event"],
    "logs": ["log", "logs", "stdout", "stderr"],
    "metrics": ["metric", "metrics", "timeseries", "promql"],
    "traces": ["trace", "traces", "span", "jaeger", "otel_trace"],
    "topology": ["topology", "graph", "dependency", "service_map"],
    "kubernetes_specs": [
        "deployment",
        "pod",
        "service",
        "configmap",
        "secret",
        "manifest",
        "spec",
        "yaml",
        "yml",
    ],
}


CATEGORY_ORDER = [
    "alerts",
    "topology",
    "traces",
    "metrics",
    "logs",
    "events",
    "kubernetes_specs",
    "other",
]


def _read_text(path: Path, max_chars: int = 80_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def _category_for_path(relative_path: str) -> str:
    lower_path = relative_path.lower()

    for category, hints in CATEGORY_FILE_HINTS.items():
        if any(hint in lower_path for hint in hints):
            return category

    return "other"


def _score_text(relative_path: str, text: str) -> int:
    lower_path = relative_path.lower()
    lower_text = text.lower()

    score = 0

    category = _category_for_path(relative_path)

    category_bonus = {
        "alerts": 25,
        "topology": 22,
        "traces": 20,
        "metrics": 18,
        "logs": 16,
        "events": 14,
        "kubernetes_specs": 12,
        "other": 0,
    }
    score += category_bonus.get(category, 0)

    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in lower_text:
            score += 3

    for keyword in CAUSAL_KEYWORDS:
        if keyword in lower_text:
            score += 5

    # File-name hints matter too.
    for keyword in CAUSAL_KEYWORDS:
        if keyword in lower_path:
            score += 6

    # Prefer compact files; huge files can drown the model.
    if len(text) < 20_000:
        score += 4

    return score


def _extract_relevant_lines(text: str, keywords: list[str], max_lines: int = 80) -> str:
    lines = text.splitlines()
    selected: list[str] = []

    lower_keywords = [keyword.lower() for keyword in keywords]

    for idx, line in enumerate(lines):
        lower_line = line.lower()

        if any(keyword in lower_line for keyword in lower_keywords):
            start = max(0, idx - 1)
            end = min(len(lines), idx + 2)

            for selected_line in lines[start:end]:
                if selected_line not in selected:
                    selected.append(selected_line)

            if len(selected) >= max_lines:
                break

    return "\n".join(selected[:max_lines])


def _make_file_index(files: list[dict[str, Any]], max_files: int = 250) -> str:
    rows = []

    for file in files[:max_files]:
        rel_path = file["relative_path"]
        category = _category_for_path(rel_path)
        rows.append(f"- [{category}] {rel_path}")

    return "\n".join(rows)

def _build_traffic_overload_appendix(root: Path, files: list[dict[str, Any]]) -> str:
    """
    Add focused evidence for common ITBench SRE overload scenarios.

    This does not use ground_truth.yaml. It searches scenario evidence for
    traffic generators, frontend-proxy overload, user counts, and request volume.
    """
    focus_terms = [
        "load-generator",
        "load generator",
        "loadgenerator",
        "frontend-proxy",
        "requestlatency",
        "request latency",
        "requesterrorrate",
        "request error rate",
        "error rate",
        "traffic",
        "request volume",
        "number of users",
        "users",
        "locust",
        "overload",
        "overloaded",
        "autoscaler",
        "autoscaling",
    ]

    chunks: list[str] = []

    for file_info in files:
        relative_path = file_info["relative_path"]

        if "ground_truth" in relative_path.lower():
            continue

        path = Path(file_info["absolute_path"])

        if not is_text_file(path):
            continue

        text = _read_text(path, max_chars=120_000)

        if not text.strip():
            continue

        lower_text = text.lower()
        lower_path = relative_path.lower()

        if not any(term in lower_text or term in lower_path for term in focus_terms):
            continue

        matched_lines = _extract_relevant_lines(
            text,
            keywords=focus_terms,
            max_lines=120,
        )

        if matched_lines.strip():
            chunks.append(
                f"\n--- TRAFFIC/OVERLOAD FOCUS SOURCE: {relative_path} ---\n"
                f"{matched_lines[:8_000]}\n"
            )

        if len("\n".join(chunks)) > 20_000:
            break

    if not chunks:
        return ""

    return (
        "\n\nTRAFFIC/OVERLOAD FOCUSED EVIDENCE:\n"
        "The following evidence was collected because request latency/error-rate incidents "
        "are often caused by upstream traffic sources, load generators, user-count changes, "
        "or capacity limits.\n"
        + "\n".join(chunks)
    )


def collect_evidence(scenario_dir: str | Path, max_total_chars: int = 70_000) -> EvidenceBundle:
    """
    Collect a compact evidence bundle from an ITBench scenario.

    Ground truth is intentionally excluded from the prompt.
    The agent must diagnose from snapshot evidence only.
    """
    root = normalize_path(scenario_dir)
    inspection = inspect_scenario(root)
    scenario_id = root.name

    grouped_candidates: dict[str, list[tuple[int, str]]] = {
        category: [] for category in CATEGORY_ORDER
    }

    file_index = _make_file_index(inspection["files"])

    for file_info in inspection["files"]:
        relative_path = file_info["relative_path"]

        # Never leak benchmark answers into the diagnosis prompt.
        if "ground_truth" in relative_path.lower():
            continue

        path = Path(file_info["absolute_path"])

        if not is_text_file(path):
            continue

        text = _read_text(path)

        if not text.strip():
            continue

        category = _category_for_path(relative_path)
        score = _score_text(relative_path, text)

        if score <= 0:
            continue

        matched_lines = _extract_relevant_lines(
            text,
            keywords=SUSPICIOUS_KEYWORDS + CAUSAL_KEYWORDS,
            max_lines=100,
        )

        chunk = (
            f"\n\n--- CATEGORY: {category} | SCORE: {score} | SOURCE: {relative_path} ---\n"
        )

        if matched_lines.strip():
            chunk += f"\nMATCHED SIGNAL LINES:\n{matched_lines[:6_000]}\n"

        chunk += f"\nFILE PREVIEW:\n{text[:6_000]}\n"

        grouped_candidates.setdefault(category, []).append((score, chunk))

    selected_text = (
        "SCENARIO FILE INDEX:\n"
        f"{file_index}\n\n"
        "EVIDENCE CHUNKS:\n"
    )

    # Balanced selection: do not let one noisy alert file dominate everything.
    for category in CATEGORY_ORDER:
        candidates = grouped_candidates.get(category, [])
        candidates.sort(key=lambda item: item[0], reverse=True)

        max_per_category = {
            "alerts": 4,
            "topology": 4,
            "traces": 4,
            "metrics": 4,
            "logs": 4,
            "events": 4,
            "kubernetes_specs": 6,
            "other": 2,
        }.get(category, 3)

        for _, chunk in candidates[:max_per_category]:
            if len(selected_text) + len(chunk) > max_total_chars:
                break
            selected_text += chunk

    if not selected_text.strip():
        selected_text = json.dumps(
            {
                "scenario_id": scenario_id,
                "file_count": inspection["total_files"],
                "files": [
                    file["relative_path"]
                    for file in inspection["files"][:200]
                    if "ground_truth" not in file["relative_path"].lower()
                ],
            },
            indent=2,
        )
    
    traffic_appendix = _build_traffic_overload_appendix(root, inspection["files"])

    if traffic_appendix:
        remaining_chars = max_total_chars - len(selected_text)
        if remaining_chars > 5_000:
            selected_text += traffic_appendix[:remaining_chars]
    
        graph_context = build_sre_graph_context(root)

        if graph_context.strip():
            graph_block = (
                "\n\nDETERMINISTIC SRE GRAPH ANALYSIS:\n"
                f"{graph_context[:25_000]}\n"
            )

            # Put graph context before raw evidence so the model sees the structure first.
            selected_text = graph_block + "\n\n" + selected_text

    return EvidenceBundle(
        scenario_id=scenario_id,
        scenario_path=str(root),
        files_seen=inspection["total_files"],
        evidence_text=selected_text,
    )


def build_diagnosis_prompt(bundle: EvidenceBundle) -> str:
    return f"""
You are CloudRCA, a read-only SRE root-cause-analysis agent.

Task:
Analyze one ITBench SRE Kubernetes incident snapshot and identify the minimal independent root-cause entity.

Important RCA rules:
- Do not simply name the component that is alerting.
- Do not treat downstream failures as root causes.
- Do not choose the scariest alert solely because it is critical.
- Alerts such as KubeSchedulerDown, KubeControllerManagerDown, KubeClientCertificateExpiration, RequestLatency, RequestErrorRate, or CPUThrottlingHigh may be symptoms.
- Prefer the entity or configuration change that best explains the earliest abnormal signal and the propagation pattern.
- If an application service is overloaded, prioritize upstream traffic sources, load-generator pods, callers, user-count changes, recent configuration changes, replica changes, and capacity limits.
- If frontend-proxy, frontend, checkout, or another service is alerting, ask what upstream component could be driving traffic or errors into it.
- If a load-generator, traffic source, config, deployment, pod, node, secret, certificate, or network policy appears to cause multiple downstream alerts, prefer that causal entity.
- The root cause should be the causal Kubernetes entity, not the alert rule name.
- Do not claim certainty if evidence is weak or if the evidence only shows symptoms.
- Do not invent files, metrics, services, pods, namespaces, or alerts.
- Do not recommend automatic remediation.
- Return ONLY valid JSON.
- Do not wrap the JSON in markdown.
- The field should_auto_remediate must always be false.

Entity guidance:
- If evidence points to a pod, use kind "Pod".
- If evidence points to a deployment/workload, use kind "Deployment".
- If evidence points to a service, use kind "Service".
- If evidence points to a config object, use kind "ConfigMap" or "Secret".
- If evidence points to an expiring certificate, use kind "Certificate" or "Secret" where appropriate.
- If evidence only shows symptoms and no causal entity, use name "Unknown" with low confidence.

Required JSON shape:
{{
  "scenario_id": "{bundle.scenario_id}",
  "incident_summary": "short summary of the incident",
  "root_cause_entities": [
    {{
      "kind": "Deployment | Pod | Service | ConfigMap | Secret | Certificate | Node | NetworkPolicy | Unknown",
      "name": "entity name or Unknown",
      "namespace": "namespace or null",
      "confidence": 0.0
    }}
  ],
  "evidence": [
    {{
      "source_type": "alerts | events | logs | metrics | traces | topology | other",
      "source_path": "relative path if available, else null",
      "summary": "specific evidence summary",
      "supports_root_cause": true
    }}
  ],
  "reasoning_summary": "explain why the suspected entity is the likely root cause and distinguish it from symptoms",
  "recommended_remediation": [
    "safe read-only or human-approved remediation recommendation"
  ],
  "should_auto_remediate": false,
  "limitations": [
    "what evidence was missing or uncertain"
  ]
}}

Scenario:
{bundle.scenario_id}

Evidence:
{bundle.evidence_text}
""".strip()


def _extract_json(text: str) -> dict[str, Any]:
    """
    Gemini should return pure JSON, but this makes the parser more tolerant.
    """
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model response:\n{cleaned[:1000]}")

    return json.loads(cleaned[start : end + 1])


def diagnose_scenario(scenario_dir: str | Path) -> DiagnosisResult:
    bundle = collect_evidence(scenario_dir)
    prompt = build_diagnosis_prompt(bundle)

    llm = GeminiClient()
    raw_response = llm.generate_json(prompt)

    try:
        data = _extract_json(raw_response)
        result = DiagnosisResult.model_validate(data)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise RuntimeError(
            f"Failed to parse/validate Gemini diagnosis.\n"
            f"Error: {exc}\n\n"
            f"Raw response:\n{raw_response[:3000]}"
        ) from exc

    # Safety override for MVP.
    result.should_auto_remediate = False

    return result