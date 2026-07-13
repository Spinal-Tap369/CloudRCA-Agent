from __future__ import annotations

import json
from typing import Any


def _case_payload(state: dict[str, Any], candidate_limit: int = 8) -> dict[str, Any]:
    graph_pack = state.get("graph_pack") or {}

    return {
        "scenario_id": state.get("scenario_id"),
        "candidate_contract": graph_pack.get("candidate_contract"),
        "incident_pattern": state.get("incident_pattern"),
        "candidate_table": (state.get("candidate_table") or [])[:candidate_limit],
        "top_candidates": (graph_pack.get("candidate_dossiers") or [])[:candidate_limit],
        "symptoms": graph_pack.get("symptoms", [])[:20],
        "important_edges": graph_pack.get("important_edges", [])[:80],
        "graph_report": str(state.get("graph_report") or "")[:18_000],
    }


def primary_candidate_prompt(state: dict[str, Any]) -> str:
    payload = _case_payload(state, candidate_limit=6)
    candidate_table = state.get("candidate_table") or []
    primary = candidate_table[0] if candidate_table else {}

    return f"""
You are CloudRCA's primary-candidate investigator.

Use only the graph evidence below. Do not use ground_truth.yaml.

Inspect the graph-ranked primary candidate. Build both the supporting case and the skeptical case.

Return only JSON with this shape:
{{
  "candidate_id": "{primary.get('id', '')}",
  "verdict": "short verdict",
  "causal_story": "how this candidate could explain the symptoms",
  "support": ["specific graph evidence supporting this candidate"],
  "concerns": ["specific weakness, ambiguity, or missing evidence"],
  "symptom_fit": "how well it covers the symptoms"
}}

Graph evidence:
{json.dumps(payload, indent=2, ensure_ascii=False)[:80_000]}
""".strip()


def rival_candidates_prompt(state: dict[str, Any]) -> str:
    payload = _case_payload(state, candidate_limit=8)

    return f"""
You are CloudRCA's rival-hypothesis investigator.

Use only the graph evidence below. Do not use ground_truth.yaml.

Inspect candidates ranked 2 through 6 as rivals to the primary candidate.

Rules:
- Identify whether each rival is a true alternate root, same root family, downstream symptom, or context.
- If a rival has the same failure type as a higher-ranked candidate, say graph rank should decide unless there is a specific override reason.
- Do not promote inactive configuration context above direct causal evidence.

Return only JSON:
{{
  "rivals": [
    {{
      "candidate_id": "candidate id",
      "rank": 2,
      "role": "primary|rival|symptom|context",
      "verdict": "short verdict",
      "support": ["specific supporting graph evidence"],
      "concerns": ["specific weakness or alternative explanation"]
    }}
  ],
  "rival_summary": "short comparison of the main rivals"
}}

Graph evidence:
{json.dumps(payload, indent=2, ensure_ascii=False)[:85_000]}
""".strip()


def tournament_prompt(state: dict[str, Any]) -> str:
    payload = _case_payload(state, candidate_limit=8)

    return f"""
You are CloudRCA's candidate tournament judge.

Use the primary case, rival cases, and graph evidence below.

Rules:
- Candidate rank is meaningful. Candidate #1 is the default winner.
- A lower-ranked candidate can win only with a concrete graph-supported override.
- If two candidates have the same evidence_type, prefer the higher-ranked candidate.
- Same-root-family candidates can be considered equivalent, but select one concrete candidate id.
- Context-only and inactive-config candidates cannot beat direct active evidence.

Return only JSON:
{{
  "selected_candidate_id": "candidate id from candidate_table",
  "reasoning": "why this candidate wins",
  "rejected_candidates": ["candidate ids rejected as weaker"]
}}

Primary case:
{json.dumps(state.get("primary_case") or {}, indent=2, ensure_ascii=False)[:15_000]}

Rival cases:
{json.dumps(state.get("rival_cases") or {}, indent=2, ensure_ascii=False)[:20_000]}

Graph evidence:
{json.dumps(payload, indent=2, ensure_ascii=False)[:80_000]}
""".strip()


def diagnosis_prompt(state: dict[str, Any]) -> str:
    payload = _case_payload(state, candidate_limit=10)
    tournament = state.get("tournament_result") or {}
    winner = tournament.get("winner") or {}
    confidence = state.get("confidence_audit") or {}

    return f"""
You are CloudRCA's final RCA writer.

Write a graph-grounded diagnosis using the audited tournament winner.

Rules:
- Return only valid JSON.
- Copy the audited winner kind, name, and namespace exactly.
- root_cause_entities must use selectable graph candidates only.
- Do not cite ground truth.
- Do not recommend automatic remediation.
- should_auto_remediate must be false.
- Evidence must be concrete and tied to graph paths, alerts, events, traces, logs, metrics, or topology.
- Explain why major rivals are weaker.
- Write for SRE stakeholders, not graph developers.
- Do not mention internal terms: rank, score, strong_root, weak_root, context_only, best_hypothesis, candidate_contract, selection_class, evidence_type, root_selectable, candidate_dossiers, audited winner.
- Explain propagation using the affected services and sample paths only when supported.

Audited winner:
{json.dumps(winner, indent=2, ensure_ascii=False)[:20_000]}

Confidence audit:
{json.dumps(confidence, indent=2, ensure_ascii=False)}

Required JSON shape:
{{
  "scenario_id": "{state.get('scenario_id')}",
  "incident_summary": "short summary",
  "root_cause_entities": [
    {{
      "kind": "{winner.get('kind', 'Unknown')}",
      "name": "{winner.get('name', 'Unknown')}",
      "namespace": {json.dumps(winner.get("namespace"))},
      "confidence": {confidence.get('confidence', 0.7)}
    }}
  ],
  "evidence": [
    {{
      "source_type": "alerts|events|logs|metrics|traces|topology|other",
      "source_path": "relative path or null",
      "summary": "specific evidence summary",
      "supports_root_cause": true
    }}
  ],
  "reasoning_summary": "candidate comparison and causal explanation",
  "recommended_remediation": ["safe read-only or human-approved action"],
  "should_auto_remediate": false,
  "limitations": ["uncertainty or missing evidence"]
}}

Primary case:
{json.dumps(state.get("primary_case") or {}, indent=2, ensure_ascii=False)[:15_000]}

Rival cases:
{json.dumps(state.get("rival_cases") or {}, indent=2, ensure_ascii=False)[:18_000]}

Tournament and audits:
{json.dumps({
    "tournament_result": state.get("tournament_result"),
    "causal_audit": state.get("causal_audit"),
    "propagation_audit": state.get("propagation_audit"),
    "contradiction_audit": state.get("contradiction_audit"),
    "confidence_audit": state.get("confidence_audit"),
}, indent=2, ensure_ascii=False)[:30_000]}

Graph evidence:
{json.dumps(payload, indent=2, ensure_ascii=False)[:80_000]}
""".strip()


def repair_prompt(state: dict[str, Any]) -> str:
    payload = _case_payload(state, candidate_limit=10)
    tournament = state.get("tournament_result") or {}

    return f"""
Your previous RCA failed deterministic graph validation.

Fix the diagnosis and return only valid JSON in the DiagnosisResult schema.

Validation errors:
{json.dumps(state.get("validation_errors") or [], indent=2)}

Previous diagnosis:
{json.dumps(state.get("proposed_result") or {}, indent=2, ensure_ascii=False)[:25_000]}

Rules:
- Select root_cause_entities from candidate_contract.selectable_root_entities.
- The audited tournament winner is the required root unless a same-family candidate is used.
- Copy kind, name, and namespace exactly.
- Do not select context-only entities.
- Do not treat inactive configuration context as active root evidence.
- Do not return Unknown if selectable candidates have causal paths.
- Remove internal terms such as rank, score, strong_root, weak_root, context_only, best_hypothesis, candidate_contract, selection_class, evidence_type, root_selectable, candidate_dossiers, audited winner.
- should_auto_remediate must be false.

Audited tournament winner:
{json.dumps(tournament.get("winner") or {}, indent=2, ensure_ascii=False)[:20_000]}

Graph evidence:
{json.dumps(payload, indent=2, ensure_ascii=False)[:90_000]}
""".strip()
