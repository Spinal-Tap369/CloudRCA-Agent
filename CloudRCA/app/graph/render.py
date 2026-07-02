from __future__ import annotations

from app.graph.models import IncidentGraph
from app.graph.tools import build_agent_graph_pack


def render_graph_context(graph: IncidentGraph) -> str:
    pack = build_agent_graph_pack(graph)

    lines: list[str] = []

    lines.append("NEUTRAL INCIDENT GRAPH FOR RCA AGENT")
    lines.append("")
    lines.append("Rules:")
    lines.append("- This graph is not the RCA answer.")
    lines.append("- Hypothesis seeds are nodes worth investigating, not final root causes.")
    lines.append("- The agent must use the graph to compare evidence, paths, symptoms, and alternatives.")
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
            f"signals={symptom['signals'][:6]} | evidence={symptom['evidence_paths'][:5]}"
        )

    if not pack["symptoms"]:
        lines.append("- No strong symptoms parsed.")
    lines.append("")

    lines.append("Control / mutation objects parsed:")
    for control in pack["control_objects"][:20]:
        lines.append(
            f"- {control['id']} | signals={control['signals'][:6]} | "
            f"tags={control['hypothesis_tags'][:6]} | evidence={control['evidence_paths'][:5]}"
        )

    if not pack["control_objects"]:
        lines.append("- No control objects parsed.")
    lines.append("")

    lines.append("Hypothesis seeds for agent investigation:")
    for seed in pack["hypothesis_seeds"][:20]:
        lines.append(
            f"- {seed['id']} | attention_score={seed['attention_score']} | "
            f"signals={seed['signals'][:6]} | tags={seed['hypothesis_tags'][:6]} | "
            f"evidence={seed['evidence_paths'][:5]}"
        )

    if not pack["hypothesis_seeds"]:
        lines.append("- No hypothesis seeds found.")
    lines.append("")

    lines.append("Paths from hypothesis seeds to symptoms:")
    shown = 0

    for seed_id, paths in pack["paths_from_seeds_to_symptoms"].items():
        if not paths:
            continue

        lines.append(f"- seed={seed_id}")

        for path in paths[:3]:
            path_text = " -> ".join([step["source"] for step in path] + [path[-1]["target"]])
            relations = [step["relation"] for step in path]
            lines.append(f"  path={path_text} | relations={relations}")

            shown += 1

            if shown >= 20:
                break

        if shown >= 20:
            break

    if shown == 0:
        lines.append("- No seed-to-symptom paths found.")
    lines.append("")

    lines.append("Important edges:")
    for edge in pack["important_edges"][:40]:
        lines.append(
            f"- {edge['source']} -> {edge['target']} | "
            f"relation={edge['relation']} | evidence={edge['evidence_path']}"
        )

    lines.append("")
    lines.append("Agent instruction:")
    lines.append("- Start from symptoms.")
    lines.append("- Inspect control objects and hypothesis seeds.")
    lines.append("- Prefer causes that explain multiple symptoms through graph paths.")
    lines.append("- Treat heavily affected Services as symptoms unless evidence supports direct causality.")
    lines.append("- Compare at least two alternatives before final diagnosis.")
    lines.append("- Return root cause only after citing supporting graph evidence.")

    return "\n".join(lines)
