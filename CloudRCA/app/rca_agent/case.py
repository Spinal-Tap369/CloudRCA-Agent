from __future__ import annotations

from pathlib import Path
from typing import Any

from app.graph.builder import build_graph
from app.graph.render import render_graph_context
from app.graph.tools import build_agent_graph_pack
from app.scenario_loader import inspect_scenario, normalize_path


GRAPH_CANDIDATE_LIMIT = 60


def compact_node(row: dict[str, Any], evidence_limit: int = 4) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "kind": row.get("kind"),
        "name": row.get("name"),
        "namespace": row.get("namespace"),
        "affected_score": row.get("affected_score"),
        "candidate_score": row.get("candidate_score"),
        "signals": (row.get("signals") or [])[:10],
        "reasons": (row.get("reasons") or [])[:10],
        "hypothesis_tags": (row.get("hypothesis_tags") or [])[:10],
        "evidence_paths": (row.get("evidence_paths") or [])[:evidence_limit],
    }


def compact_candidate_summary(row: dict[str, Any]) -> dict[str, Any]:
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
        "selection_class": row.get("selection_class"),
        "selection_reason": row.get("selection_reason"),
        "caution": row.get("caution"),
        "why_not_root": row.get("why_not_root"),
        "candidate_family": row.get("candidate_family"),
    }


def compact_candidate_dossier(row: dict[str, Any]) -> dict[str, Any]:
    compact = compact_candidate_summary(row)
    compact.update(
        {
            "signals": (row.get("signals") or [])[:10],
            "reasons": (row.get("reasons") or [])[:10],
            "supporting_evidence": (row.get("supporting_evidence") or [])[:6],
            "context_details": (row.get("context_details") or [])[:6],
            "evidence_paths": (row.get("evidence_paths") or [])[:8],
            "causal_paths": (row.get("causal_paths") or [])[:5],
        }
    )
    return compact


def candidate_contract(candidate_dossiers: list[dict[str, Any]]) -> dict[str, Any]:
    selectable = [
        compact_candidate_summary(row)
        for row in candidate_dossiers
        if row.get("root_selectable") is True
    ]
    context_only = [
        compact_candidate_summary(row)
        for row in candidate_dossiers
        if row.get("root_selectable") is not True
    ][:25]

    return {
        "root_selection_rule": (
            "Every root_cause_entities item must match one selectable_root_entities entry "
            "by kind, name, and namespace."
        ),
        "family_policy": (
            "A root-family candidate can represent the same causal object family as its members. "
            "Prefer the member with the clearest direct evidence."
        ),
        "context_only_policy": (
            "Context-only entities can explain scope or ownership but must not be returned as root causes."
        ),
        "inactive_config_policy": (
            "Inactive configuration findings are context. Select them only when no stronger direct evidence explains the symptoms."
        ),
        "selectable_root_entities": selectable,
        "context_only_entities": context_only,
    }


def compact_graph_pack(pack: dict[str, Any]) -> dict[str, Any]:
    candidate_dossiers = [
        compact_candidate_dossier(row)
        for row in (pack.get("candidate_dossiers") or [])[:GRAPH_CANDIDATE_LIMIT]
    ]

    return {
        "scenario_path": pack.get("scenario_path"),
        "files_seen": pack.get("files_seen"),
        "node_count": pack.get("node_count"),
        "edge_count": pack.get("edge_count"),
        "symptoms": [
            compact_node(row)
            for row in (pack.get("symptoms") or [])[:25]
        ],
        "control_objects": [
            compact_node(row)
            for row in (pack.get("control_objects") or [])[:45]
        ],
        "hypothesis_seeds": [
            compact_node(row)
            for row in (pack.get("hypothesis_seeds") or [])[:45]
        ],
        "root_candidates": [
            compact_candidate_summary(row)
            for row in (pack.get("root_candidates") or [])[:GRAPH_CANDIDATE_LIMIT]
        ],
        "candidate_contract": candidate_contract(candidate_dossiers),
        "candidate_dossiers": candidate_dossiers,
        "paths_from_seeds_to_symptoms": {
            key: value[:4]
            for key, value in list((pack.get("paths_from_seeds_to_symptoms") or {}).items())[:35]
        },
        "important_edges": pack.get("important_edges", [])[:120],
        "instruction": pack.get("instruction"),
    }


def build_case(scenario_dir: str | Path) -> dict[str, Any]:
    root = normalize_path(scenario_dir)
    inspection = inspect_scenario(root)
    graph = build_graph(root)
    pack = compact_graph_pack(build_agent_graph_pack(graph))

    return {
        "scenario_dir": str(root),
        "scenario_id": root.name,
        "files_seen": inspection["total_files"],
        "graph_pack": pack,
        "graph_report": render_graph_context(graph)[:24_000],
        "top_candidates": (pack.get("candidate_dossiers") or [])[:10],
    }
