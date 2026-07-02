from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from app.evaluator import evaluate_result


console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one CloudRCA diagnosis result against ITBench ground truth."
    )
    parser.add_argument(
        "scenario_dir",
        type=str,
        help="Path to one ITBench Scenario-* directory.",
    )
    parser.add_argument(
        "result_file",
        type=str,
        help="Path to one CloudRCA result JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save eval JSON.",
    )

    args = parser.parse_args()

    eval_result = evaluate_result(
        scenario_dir=args.scenario_dir,
        result_file=args.result_file,
    )

    table = Table(title=f"Evaluation: {eval_result.scenario_id}")
    table.add_column("Check", style="bold")
    table.add_column("Result")

    table.add_row("Score", str(eval_result.score))
    table.add_row("Valid schema", str(eval_result.valid_schema))
    table.add_row("Entity name match", str(eval_result.entity_name_match))
    table.add_row("Entity kind match", str(eval_result.entity_kind_match))
    table.add_row("Has evidence", str(eval_result.has_evidence))
    table.add_row("Has remediation", str(eval_result.has_remediation))
    table.add_row("Safe no auto-remediation", str(eval_result.safe_no_auto_remediation))
    table.add_row("Predicted entities", ", ".join(eval_result.predicted_entities) or "-")
    table.add_row("Predicted kinds", ", ".join(eval_result.predicted_kinds) or "-")
    table.add_row("GT entities", ", ".join(eval_result.ground_truth_entities) or "-")
    table.add_row("GT kinds", ", ".join(eval_result.ground_truth_kinds) or "-")
    table.add_row("GT filters", ", ".join(eval_result.ground_truth_filters) or "-")
    table.add_row("GT actions", ", ".join(eval_result.ground_truth_recommended_actions) or "-")
    table.add_row("Ground truth file", eval_result.ground_truth_file or "-")

    console.print(table)

    if eval_result.notes:
        console.print("\n[bold yellow]Notes:[/bold yellow]")
        for note in eval_result.notes:
            console.print(f"- {note}")

    output_path = Path(args.output) if args.output else None

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(eval_result.model_dump(), indent=2),
            encoding="utf-8",
        )
        console.print(f"\n[bold green]Saved eval:[/bold green] {output_path}")


if __name__ == "__main__":
    main()