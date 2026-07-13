from __future__ import annotations

import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

from app.agent import diagnose_scenario
from app.evaluator import evaluate_result


console = Console()


def _scenario_dirs(root: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith("Scenario-")
        ],
        key=lambda path: int(path.name.split("-")[-1]) if path.name.split("-")[-1].isdigit() else path.name,
    )


def _result_path(results_dir: Path, scenario_dir: Path) -> Path:
    direct = results_dir / f"{scenario_dir.name}-result.json"

    if direct.exists():
        return direct

    return results_dir / f"{scenario_dir.name}.json"


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        data.model_dump() if hasattr(data, "model_dump") else data,
        indent=2,
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CloudRCA RCA outputs across ITBench-Lite SRE scenarios."
    )
    parser.add_argument(
        "scenarios_root",
        type=str,
        help="Directory containing Scenario-* folders.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="CloudRCA/outputs",
        help="Directory containing saved Scenario-*-result.json files.",
    )
    parser.add_argument(
        "--run-agent",
        action="store_true",
        help="Run the agent before evaluation. This may spend LLM API quota.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="CloudRCA/outputs/agent_eval_summary.json",
        help="Path to write aggregate evaluation JSON.",
    )

    args = parser.parse_args()
    scenarios_root = Path(args.scenarios_root)
    results_dir = Path(args.results_dir)
    rows = []

    for scenario_dir in _scenario_dirs(scenarios_root):
        result_path = _result_path(results_dir, scenario_dir)

        if args.run_agent:
            console.print(f"[bold]Running agent:[/bold] {scenario_dir.name}")
            result = diagnose_scenario(scenario_dir)
            result_path = results_dir / f"{scenario_dir.name}-result.json"
            _write_json(result_path, result)

        if not result_path.exists():
            console.print(f"[yellow]Skipping {scenario_dir.name}: no result file at {result_path}[/yellow]")
            continue

        evaluation = evaluate_result(scenario_dir=scenario_dir, result_file=result_path)
        rows.append(evaluation)

    table = Table(title="CloudRCA RCA Evaluation")
    table.add_column("Scenario", style="bold")
    table.add_column("Strict", justify="right")
    table.add_column("Family", justify="right")
    table.add_column("Match", justify="right")
    table.add_column("Prop", justify="right")
    table.add_column("Safety", justify="right")
    table.add_column("Matched roots")

    for row in rows:
        table.add_row(
            row.scenario_id,
            "1" if row.passed_strict else "0",
            "1" if row.passed_family else "0",
            row.best_match_type,
            f"{row.propagation_coverage:.3f}",
            f"{row.safety_schema_score:.3f}",
            ", ".join(row.root_family_matches) or "-",
        )

    console.print(table)

    scenario_count = len(rows)
    strict_passes = sum(1 for row in rows if row.passed_strict)
    family_passes = sum(1 for row in rows if row.passed_family)
    exact_matches = sum(1 for row in rows if row.best_match_type == "exact")
    group_matches = sum(1 for row in rows if row.best_match_type == "group_filter")
    alias_matches = sum(1 for row in rows if row.best_match_type == "alias")
    family_matches = sum(1 for row in rows if row.best_match_type == "family")
    safe_passes = sum(1 for row in rows if row.safe_no_auto_remediation and row.valid_schema)
    propagation_full = sum(1 for row in rows if row.propagation_coverage >= 1.0)

    summary = {
        "scenario_count": len(rows),
        "strict_accuracy": round(strict_passes / scenario_count, 3) if scenario_count else 0.0,
        "family_accuracy": round(family_passes / scenario_count, 3) if scenario_count else 0.0,
        "safe_output_accuracy": round(safe_passes / scenario_count, 3) if scenario_count else 0.0,
        "propagation_full_coverage_accuracy": round(propagation_full / scenario_count, 3) if scenario_count else 0.0,
        "counts": {
            "strict_passes": strict_passes,
            "family_passes": family_passes,
            "exact_matches": exact_matches,
            "group_filter_matches": group_matches,
            "alias_matches": alias_matches,
            "family_only_matches": family_matches,
            "safe_outputs": safe_passes,
            "full_propagation_coverage": propagation_full,
        },
        "rows": [row.model_dump() for row in rows],
    }
    output_path = Path(args.output)
    _write_json(output_path, summary)
    console.print(f"\n[bold green]Saved eval summary:[/bold green] {output_path}")


if __name__ == "__main__":
    main()
