from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

from app.graph.builder import build_graph
from app.graph.render import render_graph_context
from app.graph.tools import (
    build_agent_graph_pack,
    find_nodes_by_name,
    get_paths_to_symptoms,
    normalize_name,
)


console = Console()


def find_ground_truth_file(scenario_dir: Path) -> Path | None:
    candidates: list[Path] = []

    for pattern in ["*ground*truth*.yaml", "*ground*truth*.yml", "*ground_truth*.json"]:
        candidates.extend(scenario_dir.rglob(pattern))

    return candidates[0] if candidates else None


def load_ground_truth(scenario_dir: Path) -> dict[str, Any]:
    gt_file = find_ground_truth_file(scenario_dir)

    if gt_file is None:
        return {}

    text = gt_file.read_text(encoding="utf-8", errors="ignore")

    if gt_file.suffix.lower() == ".json":
        loaded = json.loads(text)
    else:
        loaded = yaml.safe_load(text)

    if not isinstance(loaded, dict):
        return {}

    loaded["_ground_truth_file"] = str(gt_file)
    return loaded


def regex_filter_to_hint(value: str) -> str:
    value = str(value).strip()
    value = value.replace("\\b", "")
    value = value.replace(".*", "")
    value = value.replace(".+", "")
    value = value.replace("^", "")
    value = value.replace("$", "")
    value = value.replace("\\", "")
    value = value.strip(" .*-_")
    return value


def extract_gt_root_info(gt: dict[str, Any]) -> dict[str, Any]:
    root_names: set[str] = set()
    root_kinds: set[str] = set()
    root_filters: set[str] = set()

    for fault in gt.get("fault", []) or []:
        if not isinstance(fault, dict):
            continue

        entity = fault.get("entity", {})

        if not isinstance(entity, dict):
            continue

        name = entity.get("name")
        kind = entity.get("kind")
        group_id = entity.get("group_id")

        if name:
            root_names.add(str(name))
        if group_id:
            root_names.add(str(group_id))
        if kind:
            root_kinds.add(str(kind))

    for group in gt.get("groups", []) or []:
        if not isinstance(group, dict):
            continue

        if not bool(group.get("root_cause")):
            continue

        for key in ["id", "name"]:
            value = group.get(key)
            if value:
                root_names.add(str(value))

        kind = group.get("kind")
        if kind:
            root_kinds.add(str(kind))

        for filter_value in group.get("filter", []) or []:
            filter_text = str(filter_value)
            root_filters.add(filter_text)

            hint = regex_filter_to_hint(filter_text)
            if hint:
                root_names.add(hint)

    return {
        "root_names": sorted(root_names),
        "root_kinds": sorted(root_kinds),
        "root_filters": sorted(root_filters),
        "ground_truth_file": gt.get("_ground_truth_file", ""),
    }


def node_matches_gt_name(node_name: str, gt_info: dict[str, Any]) -> bool:
    candidate = normalize_name(node_name)

    if not candidate:
        return False

    for raw_name in gt_info["root_names"]:
        gt_name = normalize_name(raw_name)

        if not gt_name:
            continue

        if candidate == gt_name:
            return True

        if candidate in gt_name or gt_name in candidate:
            return True

    for filter_text in gt_info["root_filters"]:
        try:
            if re.search(filter_text, node_name):
                return True
        except re.error:
            pass

        hint = normalize_name(regex_filter_to_hint(filter_text))

        if hint and (hint in candidate or candidate in hint):
            return True

    return False


def node_matches_gt_kind(node_kind: str, gt_info: dict[str, Any]) -> bool:
    candidate = str(node_kind).lower().strip()

    if not candidate:
        return False

    return candidate in {str(kind).lower().strip() for kind in gt_info["root_kinds"]}


def scenario_sort_key(path: Path) -> int:
    match = re.search(r"(\d+)", path.name)
    return int(match.group(1)) if match else 999999


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
        [p for p in sre_root.iterdir() if p.is_dir() and p.name.lower().startswith("scenario-")],
        key=scenario_sort_key,
    )

    if not scenario_dirs:
        raise RuntimeError(f"No Scenario-* folders found under {sre_root}")

    table = Table(title="Graph coverage check for RCA agent")
    table.add_column("Scenario")
    table.add_column("Symptoms", justify="right")
    table.add_column("Seeds", justify="right")
    table.add_column("Root present")
    table.add_column("Root in seeds")
    table.add_column("Root connected")
    table.add_column("GT roots")
    table.add_column("GT kinds")

    total = 0
    root_present_count = 0
    root_seed_count = 0
    root_connected_count = 0

    rows: list[dict[str, Any]] = []

    for scenario_dir in scenario_dirs:
        try:
            graph = build_graph(scenario_dir)
            pack = build_agent_graph_pack(graph)

            graph_text = render_graph_context(graph)
            graph_file = output_dir / f"{scenario_dir.name}-agent-graph.txt"
            graph_file.write_text(graph_text, encoding="utf-8")

            gt = load_ground_truth(scenario_dir)
            gt_info = extract_gt_root_info(gt)

            has_gt = bool(gt_info["root_names"]) or bool(gt_info["root_kinds"]) or bool(gt_info["root_filters"])

            matched_root_nodes = []

            for node in graph.nodes.values():
                if node_matches_gt_name(node.name, gt_info):
                    matched_root_nodes.append(node)

            root_present = bool(matched_root_nodes)

            seed_ids = {seed["id"] for seed in pack["hypothesis_seeds"]}
            root_in_seeds = any(node.key.id in seed_ids for node in matched_root_nodes)

            root_connected = False

            for node in matched_root_nodes:
                paths = get_paths_to_symptoms(graph, node.key.id, max_depth=4, max_paths=1)
                if paths:
                    root_connected = True
                    break

            if has_gt:
                total += 1
                root_present_count += int(root_present)
                root_seed_count += int(root_in_seeds)
                root_connected_count += int(root_connected)

            table.add_row(
                scenario_dir.name,
                str(len(pack["symptoms"])),
                str(len(pack["hypothesis_seeds"])),
                "[green]yes[/green]" if root_present else "[red]no[/red]",
                "[green]yes[/green]" if root_in_seeds else "[red]no[/red]",
                "[green]yes[/green]" if root_connected else "[red]no[/red]",
                ", ".join(gt_info["root_names"][:3]),
                ", ".join(gt_info["root_kinds"]),
            )

            rows.append(
                {
                    "scenario": scenario_dir.name,
                    "has_ground_truth_root": has_gt,
                    "root_present": root_present,
                    "root_in_hypothesis_seeds": root_in_seeds,
                    "root_connected_to_symptoms": root_connected,
                    "matched_root_nodes": [
                        {
                            "id": node.key.id,
                            "name": node.name,
                            "kind": node.kind,
                            "namespace": node.namespace,
                            "signals": sorted(node.signals),
                            "evidence_paths": node.evidence_paths[:10],
                        }
                        for node in matched_root_nodes
                    ],
                    "symptom_count": len(pack["symptoms"]),
                    "hypothesis_seed_count": len(pack["hypothesis_seeds"]),
                    "gt_root_names": gt_info["root_names"],
                    "gt_root_kinds": gt_info["root_kinds"],
                    "gt_root_filters": gt_info["root_filters"],
                    "graph_file": str(graph_file),
                }
            )

        except Exception as exc:
            table.add_row(
                scenario_dir.name,
                "0",
                "0",
                "[red]ERROR[/red]",
                "[red]ERROR[/red]",
                "[red]ERROR[/red]",
                str(exc)[:80],
                "",
            )

    summary = {
        "total_with_gt": total,
        "root_present_count": root_present_count,
        "root_in_hypothesis_seeds_count": root_seed_count,
        "root_connected_to_symptoms_count": root_connected_count,
        "root_present_rate": round(root_present_count / total, 3) if total else 0,
        "root_in_hypothesis_seeds_rate": round(root_seed_count / total, 3) if total else 0,
        "root_connected_to_symptoms_rate": round(root_connected_count / total, 3) if total else 0,
        "rows": rows,
    }

    summary_file = output_dir / "agent_graph_coverage_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    console.print(table)
    console.print()
    console.print(f"Saved graph summaries to: {output_dir}")
    console.print(f"Saved coverage summary to: {summary_file}")
    console.print(
        f"Root present: {root_present_count}/{total} | "
        f"Root in seeds: {root_seed_count}/{total} | "
        f"Root connected: {root_connected_count}/{total}"
    )


if __name__ == "__main__":
    main()
