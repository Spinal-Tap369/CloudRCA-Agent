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

    table.add_row("Strict RCA pass", str(eval_result.passed_strict))
    table.add_row("Family RCA pass", str(eval_result.passed_family))
    table.add_row("Best match type", eval_result.best_match_type)
    table.add_row("Valid schema", str(eval_result.valid_schema))
    table.add_row("Strict precision", str(eval_result.root_group_precision))
    table.add_row("Strict recall", str(eval_result.root_group_recall))
    table.add_row("Strict F1", str(eval_result.root_group_f1))
    table.add_row("Family precision", str(eval_result.root_family_precision))
    table.add_row("Family recall", str(eval_result.root_family_recall))
    table.add_row("Family F1", str(eval_result.root_family_f1))
    table.add_row("Kind match rate", str(eval_result.kind_match_rate))
    table.add_row("Propagation coverage", str(eval_result.propagation_coverage))
    table.add_row("Alert coverage", str(eval_result.alert_coverage))
    table.add_row("Evidence present/quality", str(eval_result.evidence_quality_score))
    table.add_row("Remediation match", str(eval_result.remediation_match_score))
    table.add_row("Safety/schema", str(eval_result.safety_schema_score))
    table.add_row("Has evidence", str(eval_result.has_evidence))
    table.add_row("Has remediation", str(eval_result.has_remediation))
    table.add_row("Safe no auto-remediation", str(eval_result.safe_no_auto_remediation))
    table.add_row("Strict matched roots", ", ".join(eval_result.root_group_matches) or "-")
    table.add_row("Family matched roots", ", ".join(eval_result.root_family_matches) or "-")
    table.add_row("Unmatched GT roots", ", ".join(eval_result.unmatched_ground_truth_roots) or "-")
    table.add_row("Unmatched predictions", ", ".join(eval_result.unmatched_predicted_entities) or "-")
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
