from __future__ import annotations

from app.graph.models import IncidentGraph
from app.graph.tools import build_agent_graph_pack


def render_graph_context(graph: IncidentGraph) -> str:
    pack = build_agent_graph_pack(graph)
    lines: list[str] = []

    lines.append("INCIDENT GRAPH RCA PACK")
    lines.append("")
    lines.append("Interpretation rules:")
    lines.append("- Candidate dossiers are leads, not final answers.")
    lines.append("- Prefer candidates with typed paths to multiple symptoms.")
    lines.append("- Treat alert-heavy Services and Pods as symptoms unless causal evidence supports them.")
    lines.append("- ground_truth.yaml is excluded from graph construction.")
    lines.append("")

    lines.append("Graph size:")
    lines.append(f"- files_seen={pack['files_seen']}")
    lines.append(f"- node_count={pack['node_count']}")
    lines.append(f"- edge_count={pack['edge_count']}")
    lines.append("")

    lines.append("Observed symptoms:")
    for symptom in pack["symptoms"][:15]:
        lines.append(
            f"- {symptom['id']} | affected_score={symptom['affected_score']} | "
            f"signals={symptom['signals'][:5]} | evidence={symptom['evidence_paths'][:4]}"
        )

    if not pack["symptoms"]:
        lines.append("- No strong symptoms parsed.")

    lines.append("")
    lines.append("Top root-cause candidates:")
    for candidate in pack["candidate_dossiers"][:10]:
        lines.append(
            f"- rank={candidate['rank']} id={candidate['id']} score={candidate['score']} "
            f"hypothesis={candidate['best_hypothesis']} symptoms={candidate['affected_symptom_count']} "
            f"paths={candidate['causal_path_count']}"
        )

        if candidate["signals"]:
            lines.append(f"  signals={candidate['signals'][:6]}")

        family = candidate.get("candidate_family") or {}
        if family:
            lines.append(
                f"  family={family.get('kind')}:{family.get('name')} "
                f"role={family.get('role')} members={family.get('members', [])[:5]}"
            )

        if candidate["supporting_evidence"]:
            lines.append(f"  evidence={candidate['supporting_evidence'][:3]}")

        if candidate.get("evidence_details"):
            lines.append(f"  details={candidate['evidence_details'][:5]}")

        if candidate.get("context_details"):
            lines.append(f"  context={candidate['context_details'][:5]}")

        if candidate.get("why_not_root"):
            lines.append(f"  why_not_root={candidate['why_not_root']}")

        if candidate["caution"]:
            lines.append(f"  caution={candidate['caution']}")

        for path in candidate["causal_paths"][:3]:
            lines.append(
                f"  path={path['path']} | relations={path['relations']} "
                f"| min_confidence={path['min_confidence']}"
            )

    if not pack["candidate_dossiers"]:
        lines.append("- No candidate dossiers found.")

    lines.append("")
    lines.append("Control and mutation objects:")
    for control in pack["control_objects"][:20]:
        lines.append(
            f"- {control['id']} | candidate_score={control['candidate_score']} | "
            f"signals={control['signals'][:5]} | evidence={control['evidence_paths'][:4]}"
        )

    if not pack["control_objects"]:
        lines.append("- No control or mutation objects parsed.")

    lines.append("")
    lines.append("Important typed edges:")
    for edge in pack["important_edges"][:40]:
        lines.append(
            f"- {edge['source']} -> {edge['target']} | relation={edge['relation']} "
            f"| confidence={edge['confidence']} | evidence={edge['evidence_path']}"
        )

    lines.append("")
    lines.append("Agent task:")
    lines.append("- Compare the top candidates against symptoms and typed paths.")
    lines.append("- Explain why weaker candidates are symptoms, downstream effects, or unsupported.")
    lines.append("- Return a root cause only with specific graph evidence.")

    return "\n".join(lines)
