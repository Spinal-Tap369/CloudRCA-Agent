from __future__ import annotations

from pathlib import Path

from app.graph.k8s_events import ingest_k8s_events
from app.graph.k8s_inventory import ingest_k8s_objects
from app.graph.parsers import CONTROL_KINDS
from app.graph.records import IncidentGraph
from app.graph.relationships import add_name_implied_edges, add_object_relationships
from app.graph.scoring import score_candidates, symptom_ids
from app.graph.telemetry import ingest_alerts, ingest_logs, ingest_metrics, ingest_traces


def build_graph(scenario_dir: str | Path) -> IncidentGraph:
    scenario_path = Path(scenario_dir).resolve()
    graph = IncidentGraph(scenario_path=scenario_path)
    graph.files_seen = _count_evidence_files(scenario_path)

    objects = ingest_k8s_objects(graph, scenario_path)
    add_object_relationships(graph, objects)
    ingest_k8s_events(graph, scenario_path)
    ingest_metrics(graph, scenario_path)
    ingest_alerts(graph, scenario_path)
    ingest_logs(graph, scenario_path)
    ingest_traces(graph, scenario_path)
    add_name_implied_edges(graph)
    score_candidates(graph)

    graph.signals.pop("_edge_keys", None)
    graph.signals.update(
        {
            "kubernetes_object_count": len(objects),
            "control_node_count": sum(
                1 for node in graph.nodes.values() if node.kind in CONTROL_KINDS
            ),
            "affected_node_count": len(symptom_ids(graph)),
            "candidate_node_count": len(graph.candidates),
        }
    )
    return graph


def build_sre_graph_context(scenario_dir: str | Path) -> str:
    from app.graph.render import render_graph_context

    return render_graph_context(build_graph(scenario_dir))


def _count_evidence_files(scenario_path: Path) -> int:
    count = 0

    for path in scenario_path.rglob("*"):
        if not path.is_file():
            continue

        lower = path.name.lower()

        if "ground" in lower and "truth" in lower:
            continue

        count += 1

    return count
