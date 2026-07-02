from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from app.scenario_loader import compact_scenario_summary, inspect_scenario


console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect one ITBench-Lite SRE scenario folder."
    )
    parser.add_argument(
        "scenario_dir",
        type=str,
        help="Path to a Scenario-* directory.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full inspection as JSON.",
    )

    args = parser.parse_args()
    scenario_path = Path(args.scenario_dir)

    if args.json:
        result = inspect_scenario(scenario_path)
        print(json.dumps(result, indent=2, default=str))
        return

    summary = compact_scenario_summary(scenario_path)

    console.print("\n[bold]ITBench Scenario Inspection[/bold]")
    console.print(f"Scenario path: {summary['scenario_path']}")
    console.print(f"Scenario name: {summary['scenario_name']}")
    console.print(f"Total files: {summary['total_files']}")
    console.print(f"Ground truth available: {summary['ground_truth_available']}")

    table = Table(title="File Categories")
    table.add_column("Category", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Example files")

    for category, files in summary["files_by_category"].items():
        examples = "\n".join(files[:5]) if files else "-"
        table.add_row(category, str(len(files)), examples)

    console.print(table)


if __name__ == "__main__":
    main()