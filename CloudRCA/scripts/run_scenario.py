from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from app.agent import diagnose_scenario


console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CloudRCA diagnosis on one ITBench SRE scenario."
    )
    parser.add_argument(
        "scenario_dir",
        type=str,
        help="Path to one ITBench Scenario-* folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="CloudRCA/outputs",
        help="Directory where result JSON will be saved.",
    )

    args = parser.parse_args()

    scenario_path = Path(args.scenario_dir).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit(f"Running CloudRCA on:\n{scenario_path}", title="CloudRCA"))

    result = diagnose_scenario(scenario_path)

    output_path = output_dir / f"{scenario_path.name}-result.json"

    output_path.write_text(
        json.dumps(result.model_dump(), indent=2),
        encoding="utf-8",
    )

    console.print(f"\n[bold green]Saved result:[/bold green] {output_path}")
    console.print("\n[bold]Incident summary:[/bold]")
    console.print(result.incident_summary)

    console.print("\n[bold]Suspected root cause:[/bold]")
    for entity in result.root_cause_entities:
        console.print(
            f"- {entity.kind}/{entity.name} "
            f"namespace={entity.namespace} "
            f"confidence={entity.confidence}"
        )

    console.print("\n[bold]Recommended remediation:[/bold]")
    for action in result.recommended_remediation:
        console.print(f"- {action}")


if __name__ == "__main__":
    main()