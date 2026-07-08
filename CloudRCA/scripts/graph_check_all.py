from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from app.graph.builder import build_graph
from app.graph.render import render_graph_context
from app.graph.tools import build_agent_graph_pack, get_paths_to_symptoms
from app.graph.validate import (
    extract_gt_fault_entries,
    extract_gt_propagations,
    extract_gt_root_groups,
    extract_gt_roots,
    load_ground_truth,
    node_matches_gt_root,
)


TOP_K_VALUES = [1, 3, 5, 10]


def scenario_sort_key(path: Path) -> int:
    match = re.search(r"(\d+)", path.name)
    return int(match.group(1)) if match else 999999


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def print_rows(rows: list[dict[str, Any]]) -> None:
    header = (
        "Scenario    Symptoms  Candidates  GT roots  Present  Candidate  Selectable  Connected  Best rank  Fam rank  Top3  Top10  GT kinds"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        best_rank = row["best_root_rank"] if row["best_root_rank"] is not None else "-"
        family_rank = row["best_family_root_rank"] if row.get("best_family_root_rank") is not None else "-"
        print(
            f"{row['scenario']:<11} "
            f"{row['symptom_count']:>8} "
            f"{row['candidate_count']:>11} "
            f"{row['gt_root_count']:>8} "
            f"{row['present_count']:>7}/{row['gt_root_count']:<2} "
            f"{row['candidate_root_count']:>9}/{row['gt_root_count']:<2} "
            f"{row['selectable_root_count']:>10}/{row['gt_root_count']:<2} "
            f"{row['connected_root_count']:>9}/{row['gt_root_count']:<2} "
            f"{str(best_rank):>9} "
            f"{str(family_rank):>8} "
            f"{row['top_k_counts']['top_3']:>4}/{row['gt_root_count']:<2} "
            f"{row['top_k_counts']['top_10']:>5}/{row['gt_root_count']:<2} "
            f"{','.join(row['gt_root_kinds'])}"
        )


def print_comparison_preview(rows: list[dict[str, Any]]) -> None:
    header = "Scenario    GT root cause                         Graph top candidates"
    print(header)
    print("-" * len(header))

    for row in rows:
        gt_text = "; ".join(_format_root_group(item) for item in row.get("ground_truth", {}).get("root_cause_groups", []))

        if not gt_text:
            gt_text = "; ".join(_format_gt_root(item) for item in row.get("root_results", []))

        top_candidates = []

        for item in row.get("candidate_comparison", [])[:3]:
            marker = "*" if item["match_type"] == "direct-root" else "~" if item["match_type"] == "root-family" else ""
            top_candidates.append(
                f"{item['rank']}:{item['kind']}/{item['name']}[{item['match_type']}]{marker}"
            )

        print(f"{row['scenario']:<11} {gt_text[:35]:<35} {'; '.join(top_candidates)}")


def _candidate_indexes(pack: dict[str, Any]) -> tuple[dict[str, int], set[str], dict[str, str], dict[str, int]]:
    ranks = {}
    selectable_ids = set()
    family_ids = {}
    family_ranks = {}

    for index, candidate in enumerate(pack.get("candidate_dossiers", []) or [], start=1):
        candidate_id = candidate.get("id")

        if not candidate_id:
            continue

        candidate_id = str(candidate_id)
        rank = int(candidate.get("rank") or index)
        ranks[candidate_id] = rank

        if candidate.get("root_selectable") is True:
            selectable_ids.add(candidate_id)

        family = candidate.get("candidate_family") if isinstance(candidate.get("candidate_family"), dict) else {}
        family_id = str(family.get("id") or candidate_id)
        family_ids[candidate_id] = family_id
        family_ranks[family_id] = min(rank, family_ranks.get(family_id, rank))

    return ranks, selectable_ids, family_ids, family_ranks


def _evaluate_root(
    graph: Any,
    candidate_ranks: dict[str, int],
    selectable_ids: set[str],
    candidate_family_ids: dict[str, str],
    family_ranks: dict[str, int],
    gt_root: dict[str, Any],
) -> dict[str, Any]:
    matched_nodes = [
        node
        for node in graph.nodes.values()
        if node_matches_gt_root(node, gt_root)
    ]
    matched_ids = {node.key.id for node in matched_nodes}
    candidate_matches = sorted(
        candidate_ranks[node_id]
        for node_id in matched_ids
        if node_id in candidate_ranks
    )
    best_rank = candidate_matches[0] if candidate_matches else None
    family_matches = sorted(
        family_ranks[family_id]
        for node_id in matched_ids
        for family_id in [candidate_family_ids.get(node_id)]
        if family_id and family_id in family_ranks
    )
    best_family_rank = family_matches[0] if family_matches else best_rank
    connected = False

    for node in matched_nodes:
        paths = get_paths_to_symptoms(graph, node.key.id, max_depth=5, max_paths=1)

        if paths:
            connected = True
            break

    top_k = {
        f"top_{value}": best_rank is not None and best_rank <= value
        for value in TOP_K_VALUES
    }
    family_top_k = {
        f"top_{value}": best_family_rank is not None and best_family_rank <= value
        for value in TOP_K_VALUES
    }

    return {
        "expected": gt_root,
        "present": bool(matched_nodes),
        "in_candidates": bool(candidate_matches),
        "selectable": any(node_id in selectable_ids for node_id in matched_ids),
        "connected_to_symptoms": connected,
        "best_rank": best_rank,
        "best_family_rank": best_family_rank,
        "top_k": top_k,
        "family_top_k": family_top_k,
        "matched_nodes": [
            {
                "id": node.key.id,
                "name": node.name,
                "kind": node.kind,
                "namespace": node.namespace,
                "candidate_score": node.candidate_score,
                "signals": sorted(node.signals),
                "evidence_paths": node.evidence_paths[:10],
            }
            for node in matched_nodes
        ],
    }


def _format_gt_root(root: dict[str, Any]) -> str:
    expected = root.get("expected", root)
    names = expected.get("root_names") or [expected.get("name") or expected.get("id") or ""]
    kind = str(expected.get("kind") or (expected.get("root_kinds") or [""])[0] or "")
    namespace = str(expected.get("namespace") or "")
    name_text = ",".join(str(item) for item in names if item)
    ns_text = f"{namespace}/" if namespace else ""
    return f"{kind}:{ns_text}{name_text}".strip(":")


def _format_root_group(group: dict[str, Any]) -> str:
    kind = str(group.get("kind") or "")
    namespace = str(group.get("namespace") or "")
    name = str(group.get("name") or group.get("id") or "")
    filters = ",".join(group.get("filters") or [])
    target = name or filters
    ns_text = f"{namespace}/" if namespace else ""
    return f"{kind}:{ns_text}{target}".strip(":")


def _candidate_root_neighbors(graph: Any, root_results: list[dict[str, Any]]) -> dict[str, set[str]]:
    root_ids: dict[str, str] = {}

    for result in root_results:
        expected = result.get("expected", {})
        root_id = str(expected.get("id") or _format_gt_root(expected))

        for node in result.get("matched_nodes", []) or []:
            node_id = str(node.get("id") or "")

            if node_id:
                root_ids[node_id] = root_id

    neighbors: dict[str, set[str]] = {}

    for edge in graph.edges:
        if edge.source in root_ids:
            neighbors.setdefault(edge.target, set()).add(root_ids[edge.source])

        if edge.target in root_ids:
            neighbors.setdefault(edge.source, set()).add(root_ids[edge.target])

    return neighbors


def _candidate_comparison(
    graph: Any,
    pack: dict[str, Any],
    gt_roots: list[dict[str, Any]],
    root_results: list[dict[str, Any]],
    fault_entries: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    root_neighbors = _candidate_root_neighbors(graph, root_results)
    _, _, candidate_family_ids, _ = _candidate_indexes(pack)
    root_family_ids = {
        candidate_family_ids[node_id]
        for result in root_results
        for node in result.get("matched_nodes", []) or []
        for node_id in [str(node.get("id") or "")]
        if node_id in candidate_family_ids
    }
    rows = []

    for index, candidate in enumerate(pack.get("candidate_dossiers", [])[:limit], start=1):
        node = graph.nodes.get(candidate.get("id"))
        root_matches = []
        fault_matches = []

        if node is not None:
            root_matches = [
                str(root.get("id") or _format_gt_root(root))
                for root in gt_roots
                if node_matches_gt_root(node, root)
            ]
            fault_matches = [
                str(fault.get("id") or fault.get("name") or _format_gt_root(fault))
                for fault in fault_entries
                if node_matches_gt_root(node, fault)
            ]

        neighbor_matches = sorted(root_neighbors.get(str(candidate.get("id") or ""), set()))
        candidate_family = candidate.get("candidate_family") if isinstance(candidate.get("candidate_family"), dict) else {}
        candidate_family_id = str(candidate_family.get("id") or "")
        family_matches = sorted(root_family_ids.intersection({candidate_family_id}))

        if root_matches:
            match_type = "direct-root"
        elif family_matches:
            match_type = "root-family"
        elif neighbor_matches:
            match_type = "root-neighbor"
        elif fault_matches:
            match_type = "fault-entry"
        else:
            match_type = "unmatched"

        rows.append(
            {
                "rank": int(candidate.get("rank") or index),
                "id": candidate.get("id"),
                "kind": candidate.get("kind"),
                "name": candidate.get("name"),
                "namespace": candidate.get("namespace"),
                "score": candidate.get("score"),
                "selection_class": candidate.get("selection_class"),
                "match_type": match_type,
                "matched_roots": root_matches,
                "matched_root_families": family_matches,
                "neighbor_roots": neighbor_matches,
                "matched_fault_entries": fault_matches,
                "candidate_family": candidate_family,
                "signals": candidate.get("signals", [])[:8],
                "evidence_details": candidate.get("evidence_details", [])[:8],
                "context_details": candidate.get("context_details", [])[:8],
                "why_not_root": candidate.get("why_not_root", ""),
                "causal_paths": candidate.get("causal_paths", [])[:3],
            }
        )

    return rows


def _markdown_cell(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = text.replace("|", "\\|")
    return text


def _fault_entry_text(fault: dict[str, Any]) -> str:
    entity = f"{fault.get('kind')}/{fault.get('name')}".strip("/")
    condition = str(fault.get("condition") or "")
    changed = str(fault.get("changed_element") or "")

    if changed:
        return f"{entity}: {condition}; changed {changed}".strip("; ")

    return f"{entity}: {condition}".strip(": ")


def write_comparison_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    lines = [
        "# Graph Ground Truth Comparison",
        "",
        "This report compares graph candidates against ground_truth.yaml. The graph is built without reading ground truth.",
        "",
    ]

    for row in rows:
        lines.append(f"## {row['scenario']}")

        if row.get("error"):
            lines.append(f"- error: {row['error']}")
            lines.append("")
            continue

        lines.append(
            f"- coverage: present={row['present_count']}/{row['gt_root_count']}, "
            f"candidate={row['candidate_root_count']}/{row['gt_root_count']}, "
            f"selectable={row['selectable_root_count']}/{row['gt_root_count']}, "
            f"best_rank={row['best_root_rank'] or '-'}, "
            f"best_family_rank={row['best_family_root_rank'] or '-'}"
        )

        root_groups = row.get("ground_truth", {}).get("root_cause_groups", [])
        fault_entries = row.get("ground_truth", {}).get("fault_entries", [])
        propagations = row.get("ground_truth", {}).get("propagations", [])

        lines.append("- root_cause_groups:")
        if root_groups:
            for group in root_groups:
                lines.append(f"  - {_format_root_group(group)}")
        else:
            lines.append("  - none")

        lines.append("- fault_entries:")
        if fault_entries:
            for fault in fault_entries:
                lines.append(f"  - {_fault_entry_text(fault)}")
        else:
            lines.append("  - none")

        if propagations:
            lines.append("- propagation_hints:")
            for propagation in propagations[:6]:
                lines.append(
                    f"  - {propagation['source']} -> {propagation['target']}: "
                    f"{propagation['condition'] or propagation['effect']}"
                )

        lines.append("")
        lines.append("| Rank | Graph candidate | Match | Family | Score | Signals | Evidence details | Context / why not root |")
        lines.append("| ---: | --- | --- | --- | ---: | --- | --- | --- |")

        for candidate in row.get("candidate_comparison", [])[:10]:
            entity = f"{candidate.get('kind')}/{candidate.get('namespace') or '_'}/{candidate.get('name')}"
            family = candidate.get("candidate_family") if isinstance(candidate.get("candidate_family"), dict) else {}
            family_text = f"{family.get('kind')}/{family.get('name')}:{family.get('role')}" if family else ""
            context_text = "; ".join(candidate.get("context_details") or [])

            if candidate.get("why_not_root"):
                context_text = f"{context_text}; why_not_root={candidate['why_not_root']}".strip("; ")

            lines.append(
                "| "
                f"{candidate['rank']} | "
                f"{_markdown_cell(entity)} | "
                f"{_markdown_cell(candidate['match_type'])} | "
                f"{_markdown_cell(family_text)} | "
                f"{candidate.get('score') or 0} | "
                f"{_markdown_cell(', '.join(candidate.get('signals') or []))} | "
                f"{_markdown_cell('; '.join(candidate.get('evidence_details') or []))} | "
                f"{_markdown_cell(context_text)} |"
            )

        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sre_root", help="Path to ITBench-Lite SRE scenario root")
    parser.add_argument(
        "--output-dir",
        default="CloudRCA/outputs/graphs",
        help="Where to save graph summaries",
    )
    args = parser.parse_args()

    sre_root = Path(args.sre_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_dirs = sorted(
        [
            path
            for path in sre_root.iterdir()
            if path.is_dir() and path.name.lower().startswith("scenario-")
        ],
        key=scenario_sort_key,
    )

    if not scenario_dirs:
        raise RuntimeError(f"No Scenario-* folders found under {sre_root}")

    total_roots = 0
    total_scenarios_with_gt = 0
    root_present_count = 0
    root_candidate_count = 0
    root_selectable_count = 0
    root_connected_count = 0
    top_k_counts = {f"top_{value}": 0 for value in TOP_K_VALUES}
    family_top_k_counts = {f"top_{value}": 0 for value in TOP_K_VALUES}
    rows: list[dict[str, Any]] = []

    for scenario_dir in scenario_dirs:
        print(f"Processing {scenario_dir.name}...", flush=True)

        try:
            graph = build_graph(scenario_dir)
            pack = build_agent_graph_pack(graph)
            graph_file = output_dir / f"{scenario_dir.name}-agent-graph.txt"
            graph_file.write_text(render_graph_context(graph), encoding="utf-8")
            gt = load_ground_truth(scenario_dir)
            gt_roots = extract_gt_roots(gt)
            gt_root_groups = extract_gt_root_groups(gt)
            gt_fault_entries = extract_gt_fault_entries(gt)
            gt_propagations = extract_gt_propagations(gt)
            candidate_ranks, selectable_ids, candidate_family_ids, family_ranks = _candidate_indexes(pack)
            root_results = [
                _evaluate_root(
                    graph=graph,
                    candidate_ranks=candidate_ranks,
                    selectable_ids=selectable_ids,
                    candidate_family_ids=candidate_family_ids,
                    family_ranks=family_ranks,
                    gt_root=gt_root,
                )
                for gt_root in gt_roots
            ]
            candidate_comparison = _candidate_comparison(
                graph=graph,
                pack=pack,
                gt_roots=gt_roots,
                root_results=root_results,
                fault_entries=gt_fault_entries,
            )
            has_gt = bool(root_results)
            present_count = sum(int(item["present"]) for item in root_results)
            candidate_root_count = sum(int(item["in_candidates"]) for item in root_results)
            selectable_root_count = sum(int(item["selectable"]) for item in root_results)
            connected_root_count = sum(int(item["connected_to_symptoms"]) for item in root_results)
            scenario_top_k_counts = {
                f"top_{value}": sum(int(item["top_k"][f"top_{value}"]) for item in root_results)
                for value in TOP_K_VALUES
            }
            scenario_family_top_k_counts = {
                f"top_{value}": sum(int(item["family_top_k"][f"top_{value}"]) for item in root_results)
                for value in TOP_K_VALUES
            }
            root_ranks = [
                item["best_rank"]
                for item in root_results
                if item["best_rank"] is not None
            ]
            best_root_rank = min(root_ranks) if root_ranks else None
            family_root_ranks = [
                item["best_family_rank"]
                for item in root_results
                if item["best_family_rank"] is not None
            ]
            best_family_root_rank = min(family_root_ranks) if family_root_ranks else None

            if has_gt:
                total_scenarios_with_gt += 1
                total_roots += len(root_results)
                root_present_count += present_count
                root_candidate_count += candidate_root_count
                root_selectable_count += selectable_root_count
                root_connected_count += connected_root_count

                for key, value in scenario_top_k_counts.items():
                    top_k_counts[key] += value

                for key, value in scenario_family_top_k_counts.items():
                    family_top_k_counts[key] += value

            scenario_root_count = len(root_results)
            all_present = scenario_root_count > 0 and present_count == scenario_root_count
            all_candidates = scenario_root_count > 0 and candidate_root_count == scenario_root_count
            all_selectable = scenario_root_count > 0 and selectable_root_count == scenario_root_count
            all_connected = scenario_root_count > 0 and connected_root_count == scenario_root_count
            gt_root_kinds = sorted(
                {
                    str(root.get("kind") or "")
                    for root in gt_roots
                    if root.get("kind")
                }
            )

            rows.append(
                {
                    "scenario": scenario_dir.name,
                    "has_ground_truth_root": has_gt,
                    "root_present": all_present,
                    "root_in_candidates": all_candidates,
                    "root_selectable": all_selectable,
                    "root_connected_to_symptoms": all_connected,
                    "gt_root_count": scenario_root_count,
                    "present_count": present_count,
                    "candidate_root_count": candidate_root_count,
                    "selectable_root_count": selectable_root_count,
                    "connected_root_count": connected_root_count,
                    "best_root_rank": best_root_rank,
                    "best_family_root_rank": best_family_root_rank,
                    "top_k_counts": scenario_top_k_counts,
                    "family_top_k_counts": scenario_family_top_k_counts,
                    "root_results": root_results,
                    "ground_truth": {
                        "root_cause_groups": gt_root_groups,
                        "fault_entries": gt_fault_entries,
                        "propagations": gt_propagations,
                    },
                    "candidate_comparison": candidate_comparison,
                    "matched_root_nodes": [
                        node
                        for item in root_results
                        for node in item["matched_nodes"]
                    ],
                    "top_candidates": pack.get("candidate_dossiers", [])[:8],
                    "symptom_count": len(pack["symptoms"]),
                    "candidate_count": len(pack.get("candidate_dossiers", [])),
                    "gt_root_names": sorted(
                        {
                            name
                            for root in gt_roots
                            for name in root.get("root_names", [])
                        }
                    ),
                    "gt_root_kinds": gt_root_kinds,
                    "gt_root_filters": sorted(
                        {
                            filter_value
                            for root in gt_roots
                            for filter_value in root.get("root_filters", [])
                        }
                    ),
                    "graph_file": str(graph_file),
                }
            )
            print(
                f"{scenario_dir.name}: roots={present_count}/{scenario_root_count} present "
                f"candidates={candidate_root_count}/{scenario_root_count} "
                f"selectable={selectable_root_count}/{scenario_root_count} "
                f"connected={connected_root_count}/{scenario_root_count} "
                f"top3={scenario_top_k_counts['top_3']}/{scenario_root_count} "
                f"candidates={len(pack.get('candidate_dossiers', []))}",
                flush=True,
            )
        except Exception as exc:
            rows.append(
                {
                    "scenario": scenario_dir.name,
                    "has_ground_truth_root": False,
                    "root_present": False,
                    "root_in_candidates": False,
                    "root_selectable": False,
                    "root_connected_to_symptoms": False,
                    "gt_root_count": 0,
                    "present_count": 0,
                    "candidate_root_count": 0,
                    "selectable_root_count": 0,
                    "connected_root_count": 0,
                    "best_root_rank": None,
                    "best_family_root_rank": None,
                    "top_k_counts": {f"top_{value}": 0 for value in TOP_K_VALUES},
                    "family_top_k_counts": {f"top_{value}": 0 for value in TOP_K_VALUES},
                    "root_results": [],
                    "ground_truth": {
                        "root_cause_groups": [],
                        "fault_entries": [],
                        "propagations": [],
                    },
                    "candidate_comparison": [],
                    "error": str(exc),
                    "matched_root_nodes": [],
                    "top_candidates": [],
                    "symptom_count": 0,
                    "candidate_count": 0,
                    "gt_root_names": [],
                    "gt_root_kinds": [],
                    "gt_root_filters": [],
                    "graph_file": "",
                }
            )
            print(f"{scenario_dir.name}: error={exc}", flush=True)

    summary = {
        "total_scenarios_with_gt": total_scenarios_with_gt,
        "total_with_gt": total_roots,
        "total_gt_roots": total_roots,
        "root_present_count": root_present_count,
        "root_in_candidates_count": root_candidate_count,
        "root_selectable_count": root_selectable_count,
        "root_connected_to_symptoms_count": root_connected_count,
        "top_k_counts": top_k_counts,
        "family_top_k_counts": family_top_k_counts,
        "root_present_rate": round(root_present_count / total_roots, 3) if total_roots else 0,
        "root_in_candidates_rate": round(root_candidate_count / total_roots, 3) if total_roots else 0,
        "root_selectable_rate": round(root_selectable_count / total_roots, 3) if total_roots else 0,
        "root_connected_to_symptoms_rate": round(root_connected_count / total_roots, 3) if total_roots else 0,
        "top_k_rates": {
            key: round(value / total_roots, 3) if total_roots else 0
            for key, value in top_k_counts.items()
        },
        "family_top_k_rates": {
            key: round(value / total_roots, 3) if total_roots else 0
            for key, value in family_top_k_counts.items()
        },
        "rows": rows,
    }
    summary_file = output_dir / "agent_graph_coverage_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    comparison_file = output_dir / "agent_graph_ground_truth_comparison.md"
    write_comparison_markdown(rows, comparison_file)

    print_rows(rows)
    print()
    print_comparison_preview(rows)
    print()
    print(f"Saved graph summaries to: {output_dir}")
    print(f"Saved coverage summary to: {summary_file}")
    print(f"Saved ground-truth comparison to: {comparison_file}")
    print(
        f"Root present: {root_present_count}/{total_roots} | "
        f"Root in candidates: {root_candidate_count}/{total_roots} | "
        f"Root selectable: {root_selectable_count}/{total_roots} | "
        f"Root connected: {root_connected_count}/{total_roots} | "
        f"Top3: {top_k_counts['top_3']}/{total_roots} | "
        f"Top10: {top_k_counts['top_10']}/{total_roots} | "
        f"FamilyTop1: {family_top_k_counts['top_1']}/{total_roots}"
    )


if __name__ == "__main__":
    main()
