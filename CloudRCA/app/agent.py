from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.graph.builder import build_graph
from app.graph.render import render_graph_context
from app.graph.tools import build_agent_graph_pack
from app.llm_client import GeminiClient
from app.scenario_loader import inspect_scenario, is_text_file, normalize_path
from app.schemas import DiagnosisResult, EvidenceBundle


SUSPICIOUS_KEYWORDS = [
    "error", "failed", "failure", "exception", "timeout", "latency", "unavailable",
    "crash", "crashloopbackoff", "back-off", "imagepullbackoff", "errimagepull",
    "oom", "oomkilled", "restart", "restarted", "unhealthy", "readiness", "liveness",
    "probe", "evicted", "throttle", "throttling", "denied", "refused",
    "connection refused", "forbidden", "quota", "failedcreate",
    "minimumreplicasunavailable", "pending", "unschedulable", "networkpolicy",
    "network policy", "blocked", "partition", "delay", "chaos", "stress",
    "invalid image", "invalid command", "wrong architecture", "5xx", "500", "502",
    "503", "504",
]


CAUSAL_KEYWORDS = [
    "deployment", "rollout", "image", "container image", "command", "args",
    "configured", "configuration", "config", "configmap", "secret", "changed",
    "change", "env", "environment", "limit", "limits", "resource", "resourcequota",
    "resource quota", "limitrange", "cpu", "memory", "requests.memory",
    "limits.memory", "queue", "connection", "certificate", "tls", "networkpolicy",
    "network policy", "chaos", "networkchaos", "podchaos", "stresschaos",
    "jvmchaos", "schedule", "horizontalpodautoscaler", "hpa", "autoscaler",
    "autoscaling", "replica", "replicas", "load-generator", "load generator",
    "traffic", "request volume", "number of users", "users", "overload",
]


CATEGORY_FILE_HINTS = {
    "alerts": ["alert", "alerts", "prometheus"],
    "events": ["event", "events", "k8s_events", "kubernetes_event"],
    "logs": ["log", "logs", "stdout", "stderr"],
    "metrics": ["metric", "metrics", "timeseries", "promql"],
    "traces": ["trace", "traces", "span", "jaeger", "otel_trace"],
    "topology": ["topology", "graph", "dependency", "service_map"],
    "kubernetes_specs": [
        "k8s_objects", "deployment", "pod", "service", "configmap", "secret",
        "manifest", "spec", "yaml", "yml",
    ],
}


CATEGORY_ORDER = [
    "alerts", "events", "kubernetes_specs", "logs", "traces", "metrics", "topology", "other",
]


ALLOWED_ROOT_KINDS = {
    "Deployment", "Pod", "Service", "ConfigMap", "Secret", "Certificate", "Node",
    "NetworkPolicy", "Namespace", "ResourceQuota", "LimitRange", "NetworkChaos",
    "PodChaos", "StressChaos", "JVMChaos", "Schedule", "HorizontalPodAutoscaler",
    "Unknown",
}

GRAPH_CANDIDATE_LIMIT = 60


def _read_text(path: Path, max_chars: int = 120_000) -> str:
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


def _normalize_token(value: str) -> str:
    value = str(value).lower().strip()
    value = value.replace("_", "-")
    value = re.sub(r"[^a-z0-9-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-")


def _workload_base(name: str) -> str:
    name = _normalize_token(name)
    name = re.sub(r"-[a-f0-9]{8,10}-[a-z0-9]{4,6}$", "", name)
    name = re.sub(r"-[a-z0-9]{5,10}-[a-z0-9]{4,6}$", "", name)
    return name


def _terms_from_graph_pack(pack: dict[str, Any]) -> list[str]:
    terms: set[str] = set()

    def add(value: Any) -> None:
        if value is None:
            return

        text = str(value).strip()
        if not text:
            return

        terms.add(text)

        normalized = _normalize_token(text)
        if normalized:
            terms.add(normalized)

        base = _workload_base(text)
        if base:
            terms.add(base)

    for section in ["symptoms", "control_objects", "hypothesis_seeds"]:
        for row in pack.get(section, []) or []:
            add(row.get("id"))
            add(row.get("kind"))
            add(row.get("name"))
            add(row.get("namespace"))

            for signal in row.get("signals", []) or []:
                add(signal)

            for tag in row.get("hypothesis_tags", []) or []:
                add(tag)

    for edge in pack.get("important_edges", []) or []:
        add(edge.get("source"))
        add(edge.get("target"))
        add(edge.get("relation"))

    return sorted(terms, key=lambda term: (len(term), term), reverse=True)[:180]


def _compact_node(row: dict[str, Any], evidence_limit: int = 4) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "kind": row.get("kind"),
        "name": row.get("name"),
        "namespace": row.get("namespace"),
        "affected_score": row.get("affected_score"),
        "candidate_score": row.get("candidate_score"),
        "signals": (row.get("signals") or [])[:8],
        "reasons": (row.get("reasons") or [])[:8],
        "hypothesis_tags": (row.get("hypothesis_tags") or [])[:8],
        "evidence_paths": (row.get("evidence_paths") or [])[:evidence_limit],
    }


def _compact_candidate_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": row.get("rank"),
        "id": row.get("id"),
        "kind": row.get("kind"),
        "name": row.get("name"),
        "namespace": row.get("namespace"),
        "score": row.get("score"),
        "best_hypothesis": row.get("best_hypothesis"),
        "affected_symptom_count": row.get("affected_symptom_count"),
        "causal_path_count": row.get("causal_path_count"),
        "root_selectable": row.get("root_selectable"),
        "selection_reason": row.get("selection_reason"),
        "caution": row.get("caution"),
    }


def _compact_candidate_dossier(row: dict[str, Any]) -> dict[str, Any]:
    compact = _compact_candidate_summary(row)
    compact.update(
        {
            "signals": (row.get("signals") or [])[:8],
            "reasons": (row.get("reasons") or [])[:8],
            "supporting_evidence": (row.get("supporting_evidence") or [])[:4],
            "evidence_paths": (row.get("evidence_paths") or [])[:6],
            "causal_paths": (row.get("causal_paths") or [])[:3],
        }
    )
    return compact


def _candidate_contract(candidate_dossiers: list[dict[str, Any]]) -> dict[str, Any]:
    selectable = [
        _compact_candidate_summary(row)
        for row in candidate_dossiers
        if row.get("root_selectable") is True
    ]
    context_only = [
        _compact_candidate_summary(row)
        for row in candidate_dossiers
        if row.get("root_selectable") is not True
    ][:25]

    return {
        "root_selection_rule": (
            "Every root_cause_entities item must match one selectable_root_entities entry "
            "by kind, name, and namespace. Copy candidate kind/name/namespace exactly."
        ),
        "outside_candidate_policy": (
            "Alert names, metric names, log phrases, and monitoring targets are evidence only. "
            "Do not return them as root entities unless they are listed as selectable roots."
        ),
        "context_only_policy": (
            "Context-only entities can explain scope or ownership but must not be returned as root causes."
        ),
        "unknown_policy": (
            "Use kind Unknown only when no selectable root candidate has causal paths to symptoms."
        ),
        "selectable_root_entities": selectable,
        "context_only_entities": context_only,
        "valid_root_entities": selectable,
    }


def _compact_graph_pack(pack: dict[str, Any]) -> dict[str, Any]:
    candidate_dossiers = [
        _compact_candidate_dossier(row)
        for row in (pack.get("candidate_dossiers") or [])[:GRAPH_CANDIDATE_LIMIT]
    ]

    return {
        "scenario_path": pack.get("scenario_path"),
        "files_seen": pack.get("files_seen"),
        "node_count": pack.get("node_count"),
        "edge_count": pack.get("edge_count"),
        "symptoms": [
            _compact_node(row)
            for row in (pack.get("symptoms") or [])[:25]
        ],
        "control_objects": [
            _compact_node(row)
            for row in (pack.get("control_objects") or [])[:45]
        ],
        "hypothesis_seeds": [
            _compact_node(row)
            for row in (pack.get("hypothesis_seeds") or [])[:45]
        ],
        "root_candidates": [
            _compact_candidate_summary(row)
            for row in (pack.get("root_candidates") or [])[:GRAPH_CANDIDATE_LIMIT]
        ],
        "candidate_contract": _candidate_contract(candidate_dossiers),
        "candidate_dossiers": candidate_dossiers,
        "paths_from_seeds_to_symptoms": {
            key: value[:3]
            for key, value in list((pack.get("paths_from_seeds_to_symptoms") or {}).items())[:30]
        },
        "important_edges": pack.get("important_edges", [])[:100],
        "instruction": pack.get("instruction"),
    }


def _score_text(relative_path: str, text: str, focus_terms: list[str]) -> int:
    lower_path = relative_path.lower()
    lower_text = text.lower()

    category = _category_for_path(relative_path)
    score = {
        "alerts": 35,
        "events": 32,
        "kubernetes_specs": 30,
        "logs": 24,
        "traces": 22,
        "metrics": 20,
        "topology": 14,
        "other": 0,
    }.get(category, 0)

    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in lower_text:
            score += 4

    for keyword in CAUSAL_KEYWORDS:
        if keyword in lower_text:
            score += 5

    for term in focus_terms:
        lower_term = term.lower()
        if not lower_term:
            continue
        if lower_term in lower_path:
            score += 18
        if lower_term in lower_text:
            score += 12

    if len(text) < 30_000:
        score += 4

    return score


def _extract_relevant_lines(
    text: str,
    keywords: list[str],
    max_lines: int = 120,
    context: int = 1,
) -> str:
    lines = text.splitlines()
    selected: list[str] = []
    seen: set[str] = set()
    lower_keywords = [keyword.lower() for keyword in keywords if keyword.strip()]

    for idx, line in enumerate(lines):
        lower_line = line.lower()

        if not any(keyword in lower_line for keyword in lower_keywords):
            continue

        start = max(0, idx - context)
        end = min(len(lines), idx + context + 1)

        for selected_line in lines[start:end]:
            if selected_line in seen:
                continue

            seen.add(selected_line)
            selected.append(selected_line)

            if len(selected) >= max_lines:
                break

        if len(selected) >= max_lines:
            break

    return "\n".join(selected[:max_lines])


def _make_file_index(files: list[dict[str, Any]], max_files: int = 300) -> str:
    rows = []

    for file in files[:max_files]:
        rel_path = file["relative_path"]

        if "ground_truth" in rel_path.lower():
            continue

        category = _category_for_path(rel_path)
        rows.append(f"- [{category}] {rel_path}")

    return "\n".join(rows)


def _collect_raw_evidence(
    files: list[dict[str, Any]],
    focus_terms: list[str],
    max_total_chars: int,
) -> str:
    grouped_candidates: dict[str, list[tuple[int, str]]] = {
        category: [] for category in CATEGORY_ORDER
    }

    keywords = list(dict.fromkeys(SUSPICIOUS_KEYWORDS + CAUSAL_KEYWORDS + focus_terms))

    for file_info in files:
        relative_path = file_info["relative_path"]

        if "ground_truth" in relative_path.lower():
            continue

        path = Path(file_info["absolute_path"])

        if not is_text_file(path):
            continue

        text = _read_text(path)

        if not text.strip():
            continue

        category = _category_for_path(relative_path)
        score = _score_text(relative_path, text, focus_terms)

        if score <= 0:
            continue

        matched_lines = _extract_relevant_lines(
            text,
            keywords=keywords,
            max_lines=120,
            context=1,
        )

        chunk = f"\n\n--- CATEGORY: {category} | SCORE: {score} | SOURCE: {relative_path} ---\n"

        if matched_lines.strip():
            chunk += f"\nMATCHED SIGNAL LINES:\n{matched_lines[:8_000]}\n"
        else:
            chunk += f"\nFILE PREVIEW:\n{text[:4_000]}\n"

        grouped_candidates.setdefault(category, []).append((score, chunk))

    selected_text = ""

    for category in CATEGORY_ORDER:
        candidates = grouped_candidates.get(category, [])
        candidates.sort(key=lambda item: item[0], reverse=True)

        max_per_category = {
            "alerts": 5,
            "events": 6,
            "kubernetes_specs": 8,
            "logs": 5,
            "traces": 4,
            "metrics": 5,
            "topology": 2,
            "other": 2,
        }.get(category, 3)

        for _, chunk in candidates[:max_per_category]:
            if len(selected_text) + len(chunk) > max_total_chars:
                break
            selected_text += chunk

    return selected_text[:max_total_chars]


def _build_evidence_bundle(
    scenario_dir: str | Path,
    max_total_chars: int = 120_000,
) -> tuple[EvidenceBundle, dict[str, Any]]:
    """
    Build an agent-ready evidence bundle.

    The graph pack is the primary evidence substrate.
    Raw evidence is selected after graph construction using graph-discovered entities.
    """
    root = normalize_path(scenario_dir)
    inspection = inspect_scenario(root)
    scenario_id = root.name

    graph = build_graph(root)
    graph_pack = build_agent_graph_pack(graph)
    compact_pack = _compact_graph_pack(graph_pack)

    graph_json = json.dumps(compact_pack, indent=2, ensure_ascii=False)
    rendered_graph = render_graph_context(graph)
    file_index = _make_file_index(inspection["files"])
    focus_terms = _terms_from_graph_pack(compact_pack)

    selected_text = (
        "AGENT GRAPH PACK JSON:\n"
        "This is the primary structured evidence. Candidate dossiers are leads, not final answers.\n"
        f"{graph_json[:75_000]}\n\n"
        "GRAPH CONTEXT SUMMARY:\n"
        f"{rendered_graph[:18_000]}\n\n"
        "SCENARIO FILE INDEX:\n"
        f"{file_index}\n\n"
    )

    remaining = max_total_chars - len(selected_text)

    if remaining > 10_000:
        raw_evidence = _collect_raw_evidence(
            files=inspection["files"],
            focus_terms=focus_terms,
            max_total_chars=remaining,
        )

        selected_text += (
            "\n\nFOCUSED RAW EVIDENCE:\n"
            "These lines are selected using symptoms, graph seeds, graph paths, and suspicious Kubernetes/SRE terms.\n"
            "Use raw evidence to validate graph candidates, not to replace the candidate-dossier comparison.\n"
            f"{raw_evidence}\n"
        )

    bundle = EvidenceBundle(
        scenario_id=scenario_id,
        scenario_path=str(root),
        files_seen=inspection["total_files"],
        evidence_text=selected_text[:max_total_chars],
    )
    return bundle, compact_pack


def collect_evidence(scenario_dir: str | Path, max_total_chars: int = 120_000) -> EvidenceBundle:
    """
    Build an agent-ready evidence bundle.

    The graph pack is the primary evidence substrate.
    Raw evidence is selected after graph construction using graph-discovered entities.
    """
    bundle, _ = _build_evidence_bundle(scenario_dir, max_total_chars=max_total_chars)
    return bundle


def build_diagnosis_prompt(bundle: EvidenceBundle) -> str:
    return f"""
You are CloudRCA, a read-only Kubernetes/SRE root-cause-analysis agent.

You are given:
1. A neutral incident graph pack.
2. A graph context summary.
3. Focused raw evidence from alerts, events, logs, metrics, traces, and Kubernetes objects.

Critical interpretation rules:
- The graph is not the answer. It is an evidence map.
- Candidate dossiers are the primary investigation list. Compare them before selecting a root cause.
- Root cause entities must be selected from candidate_contract.selectable_root_entities.
- Entities in candidate_contract.context_only_entities are scope/context evidence, not valid root-cause outputs.
- Copy the selected candidate kind, name, and namespace exactly into root_cause_entities.
- Hypothesis seeds are candidates to investigate, not final root causes.
- Start from observed symptoms.
- Prefer a root cause that explains multiple symptoms through graph paths.
- Prefer control/mutation objects when supported: Deployment, ConfigMap, Secret, NetworkPolicy, Namespace, ResourceQuota, LimitRange, Chaos objects, Schedule, HPA.
- Treat heavily affected Services as symptoms unless there is direct evidence they are the causal object.
- Treat CPU throttling, latency, and 5xx-heavy Pods/Deployments as symptoms unless their candidate dossier provides causal support.
- Treat downstream pods/services as symptoms when an upstream/control object explains them.
- Use paths_from_seeds_to_symptoms to test causal reachability.
- Compare at least three plausible hypotheses when available.
- Reject hypotheses that only explain a local symptom or are observability noise.
- Do not let raw alert volume override a stronger candidate dossier with causal paths.
- Alert names, metric names, Prometheus targets, and monitoring target-discovery failures are evidence, not root entities, unless they appear in candidate_contract.selectable_root_entities.
- Unknown is invalid when candidate_contract.selectable_root_entities contains candidates with causal paths to symptoms.
- Do not invent files, metrics, services, pods, namespaces, alerts, or object names.
- Do not use ground_truth.yaml. It is not included.
- Do not recommend automatic remediation.
- Return ONLY valid JSON.
- Do not wrap the JSON in markdown.
- should_auto_remediate must always be false.

Specific RCA guidance:
- ImagePullBackOff / ErrImagePull usually points to a Deployment image/config root, not the pod itself, if a deployment node exists.
- CrashLoopBackOff caused by bad command/args usually points to the Deployment, not just the crashed pod.
- Feature flag / defaultVariant / flag configuration failures usually point to ConfigMap.
- NetworkPolicy blocking ingress/egress points to NetworkPolicy.
- Quota, FailedCreate, unschedulable, or memory quota enforcement can point to Namespace/ResourceQuota/LimitRange.
- Chaos Mesh objects such as NetworkChaos, PodChaos, StressChaos, JVMChaos, Schedule are root-capable when they connect to symptoms.
- Load-generator or traffic-source Pods are root-capable when their candidate dossier connects them to latency/error symptoms through graph paths and traffic evidence.

Allowed root cause kinds:
{sorted(ALLOWED_ROOT_KINDS)}

Required JSON shape:
{{
  "scenario_id": "{bundle.scenario_id}",
  "incident_summary": "short summary of the incident",
  "root_cause_entities": [
    {{
      "kind": "Deployment | Pod | Service | ConfigMap | Secret | Certificate | Node | NetworkPolicy | Namespace | ResourceQuota | LimitRange | NetworkChaos | PodChaos | StressChaos | JVMChaos | Schedule | HorizontalPodAutoscaler | Unknown",
      "name": "entity name or Unknown",
      "namespace": "namespace or null",
      "confidence": 0.0
    }}
  ],
  "evidence": [
    {{
      "source_type": "alerts | events | logs | metrics | traces | topology | graph | kubernetes | other",
      "source_path": "relative path if available, else null",
      "summary": "specific evidence summary",
      "supports_root_cause": true
    }}
  ],
  "reasoning_summary": "Compare plausible hypotheses. Explain why the selected root cause best explains the symptoms and why alternatives are symptoms or weaker.",
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
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model response:\n{cleaned[:1000]}")

    return json.loads(cleaned[start : end + 1])


def _coerce_root_kind(kind: str | None) -> str:
    if not kind:
        return "Unknown"

    raw = str(kind).strip()

    aliases = {
        "HPA": "HorizontalPodAutoscaler",
        "Network Policy": "NetworkPolicy",
        "Resource Quota": "ResourceQuota",
        "Limit Range": "LimitRange",
    }

    if raw in aliases:
        return aliases[raw]

    for allowed in ALLOWED_ROOT_KINDS:
        if raw.lower() == allowed.lower():
            return allowed

    return raw


def _postprocess_result(result: DiagnosisResult) -> DiagnosisResult:
    result.should_auto_remediate = False

    for entity in result.root_cause_entities:
        entity.kind = _coerce_root_kind(entity.kind)

        if entity.kind not in ALLOWED_ROOT_KINDS:
            entity.kind = "Unknown"

    return result


def _parse_model_result(raw_response: str) -> DiagnosisResult:
    data = _extract_json(raw_response)
    return DiagnosisResult.model_validate(data)


def _candidate_rows(compact_pack: dict[str, Any]) -> list[dict[str, Any]]:
    contract = compact_pack.get("candidate_contract")

    if isinstance(contract, dict) and isinstance(contract.get("selectable_root_entities"), list):
        return [
            row
            for row in contract["selectable_root_entities"]
            if isinstance(row, dict)
        ]

    if isinstance(contract, dict) and isinstance(contract.get("valid_root_entities"), list):
        return [
            row
            for row in contract["valid_root_entities"]
            if isinstance(row, dict)
        ]

    return [
        row
        for row in compact_pack.get("candidate_dossiers", []) or []
        if isinstance(row, dict)
    ]


def _namespace_matches(entity_namespace: str | None, candidate_namespace: Any) -> bool:
    entity_ns = str(entity_namespace or "").strip()
    candidate_ns = str(candidate_namespace or "").strip()
    return not entity_ns or not candidate_ns or entity_ns == candidate_ns


def _name_matches_candidate(entity_name: str, candidate_name: Any) -> bool:
    entity = _normalize_token(entity_name)
    candidate = _normalize_token(str(candidate_name or ""))

    if not entity or not candidate:
        return False

    if entity == candidate:
        return True

    if len(entity) >= 4 and candidate.startswith(f"{entity}-"):
        return True

    if len(candidate) >= 4 and entity.startswith(f"{candidate}-"):
        return True

    entity_base = _workload_base(entity)
    candidate_base = _workload_base(candidate)
    return bool(entity_base and candidate_base and entity_base == candidate_base)


def _entity_matches_candidate(entity: Any, candidate: dict[str, Any]) -> bool:
    entity_kind = _coerce_root_kind(getattr(entity, "kind", None))
    candidate_kind = _coerce_root_kind(str(candidate.get("kind") or ""))

    if entity_kind != candidate_kind:
        return False

    if not _namespace_matches(getattr(entity, "namespace", None), candidate.get("namespace")):
        return False

    return _name_matches_candidate(str(getattr(entity, "name", "")), candidate.get("name"))


def _candidate_contract_violations(
    result: DiagnosisResult,
    compact_pack: dict[str, Any],
) -> list[str]:
    candidates = _candidate_rows(compact_pack)
    candidates_with_paths = [
        row
        for row in candidates
        if int(row.get("causal_path_count") or 0) > 0
    ]
    violations = []

    if not result.root_cause_entities:
        return ["No root_cause_entities were returned."]

    for entity in result.root_cause_entities:
        entity.kind = _coerce_root_kind(entity.kind)

        if entity.kind == "Unknown":
            if candidates_with_paths:
                violations.append(
                    "Unknown root returned even though graph candidates have causal paths."
                )
            continue

        if not candidates:
            continue

        if not any(_entity_matches_candidate(entity, candidate) for candidate in candidates):
            namespace = entity.namespace or ""
            violations.append(
                f"Unsupported root entity {entity.kind}:{namespace}:{entity.name}; "
                "it does not match candidate_contract.selectable_root_entities."
            )

    return violations


def _contract_retry_prompt(
    bundle: EvidenceBundle,
    compact_pack: dict[str, Any],
    previous_result: DiagnosisResult,
    violations: list[str],
) -> str:
    correction_pack = {
        "candidate_contract": compact_pack.get("candidate_contract"),
        "candidate_dossiers": compact_pack.get("candidate_dossiers", [])[:GRAPH_CANDIDATE_LIMIT],
        "symptoms": compact_pack.get("symptoms", []),
        "important_edges": compact_pack.get("important_edges", [])[:80],
    }

    return f"""
Your previous diagnosis violated the graph candidate contract.

Violations:
{json.dumps(violations, indent=2)}

Correction rules:
- Return ONLY valid JSON in the required schema.
- Select root_cause_entities from candidate_contract.selectable_root_entities.
- Copy the selected candidate kind, name, and namespace exactly.
- Do not return alert names, metric names, Prometheus targets, monitoring target names, or free-form infrastructure component names unless they are listed as selectable root entities.
- Use Unknown only when no listed candidate has causal paths to symptoms.
- Use the candidate dossiers and paths below as the primary evidence.

Previous JSON:
{previous_result.model_dump_json(indent=2)}

Scenario:
{bundle.scenario_id}

Graph candidate evidence:
{json.dumps(correction_pack, indent=2, ensure_ascii=False)[:90_000]}
""".strip()


def diagnose_scenario_legacy(scenario_dir: str | Path) -> DiagnosisResult:
    bundle, compact_pack = _build_evidence_bundle(scenario_dir)
    prompt = build_diagnosis_prompt(bundle)

    llm = GeminiClient()
    raw_response = llm.generate_json(prompt)

    try:
        result = _parse_model_result(raw_response)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise RuntimeError(
            f"Failed to parse/validate LLM diagnosis.\n"
            f"Error: {exc}\n\n"
            f"Raw response:\n{raw_response[:3000]}"
        ) from exc

    result = _postprocess_result(result)
    violations = _candidate_contract_violations(result, compact_pack)

    if violations:
        retry_prompt = _contract_retry_prompt(
            bundle=bundle,
            compact_pack=compact_pack,
            previous_result=result,
            violations=violations,
        )
        retry_response = llm.generate_json(retry_prompt)

        try:
            result = _postprocess_result(_parse_model_result(retry_response))
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise RuntimeError(
                f"Failed to parse/validate retry diagnosis.\n"
                f"Original candidate-contract violations: {violations}\n"
                f"Error: {exc}\n\n"
                f"Retry response:\n{retry_response[:3000]}"
            ) from exc

        violations = _candidate_contract_violations(result, compact_pack)

    if violations:
        raise RuntimeError(
            "LLM diagnosis violated the graph candidate contract after retry.\n"
            f"Violations: {violations}\n"
            "Inspect candidate_dossiers or use a stronger model for this scenario."
        )

    return result


def diagnose_scenario(scenario_dir: str | Path) -> DiagnosisResult:
    from app.rca_agent.workflow import diagnose_scenario_with_workflow

    return diagnose_scenario_with_workflow(scenario_dir)
