from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.llm_client import GeminiClient
from app.scenario_loader import inspect_scenario, is_text_file, normalize_path
from app.schemas import DiagnosisResult, EvidenceBundle


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


CATEGORY_HINTS = {
    "alerts": ["alert", "alerts", "prometheus"],
    "events": ["event", "events", "k8s_events"],
    "logs": ["log", "logs"],
    "metrics": ["metric", "metrics", "promql", "timeseries"],
    "traces": ["trace", "traces", "span"],
    "topology": ["topology", "graph", "dependency", "service_map"],
}


def _read_text(path: Path, max_chars: int = 40_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def _score_text(relative_path: str, text: str) -> int:
    lower_path = relative_path.lower()
    lower_text = text.lower()

    score = 0

    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in lower_text:
            score += 3

    for category, hints in CATEGORY_HINTS.items():
        if any(hint in lower_path for hint in hints):
            score += 5

    # Prefer compact files over huge noisy files.
    if len(text) < 20_000:
        score += 2

    return score


def collect_evidence(scenario_dir: str | Path, max_total_chars: int = 60_000) -> EvidenceBundle:
    """
    Collect a compact evidence bundle from an ITBench scenario.

    This is intentionally read-only. It does not use ground_truth.yaml as evidence,
    because the agent should diagnose without seeing the answer.
    """
    root = normalize_path(scenario_dir)
    inspection = inspect_scenario(root)
    scenario_id = root.name

    candidate_chunks: list[tuple[int, str]] = []

    for file_info in inspection["files"]:
        relative_path = file_info["relative_path"]

        # Never leak ground truth into the diagnosis prompt.
        if "ground_truth" in relative_path.lower():
            continue

        path = Path(file_info["absolute_path"])

        if not is_text_file(path):
            continue

        text = _read_text(path)

        if not text.strip():
            continue

        score = _score_text(relative_path, text)

        if score <= 0:
            continue

        chunk = (
            f"\n\n--- SOURCE: {relative_path} ---\n"
            f"{text[:8_000]}"
        )

        candidate_chunks.append((score, chunk))

    candidate_chunks.sort(key=lambda item: item[0], reverse=True)

    selected_text = ""
    for _, chunk in candidate_chunks:
        if len(selected_text) + len(chunk) > max_total_chars:
            break
        selected_text += chunk

    if not selected_text.strip():
        # Fallback: include file listing if no useful text was found.
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
Analyze one ITBench SRE Kubernetes incident snapshot and identify the likely root cause.

Important rules:
- Use only the evidence provided below.
- Do not claim certainty if evidence is weak.
- Do not invent files, metrics, services, pods, namespaces, or alerts.
- Do not recommend automatic remediation.
- Return ONLY valid JSON.
- Do not wrap the JSON in markdown.
- The field should_auto_remediate must always be false.

Required JSON shape:
{{
  "scenario_id": "{bundle.scenario_id}",
  "incident_summary": "short summary of the incident",
  "root_cause_entities": [
    {{
      "kind": "Deployment | Pod | Service | ConfigMap | Node | NetworkPolicy | Unknown",
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
  "reasoning_summary": "explain why the suspected entity is likely the root cause",
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