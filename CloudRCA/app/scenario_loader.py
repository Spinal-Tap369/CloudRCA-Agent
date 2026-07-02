from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".csv",
    ".tsv",
    ".log",
}


CATEGORY_KEYWORDS = {
    "alerts": ["alert", "alerts", "prometheus"],
    "metrics": ["metric", "metrics", "timeseries", "promql"],
    "events": ["event", "events", "k8s_events", "kubernetes_event"],
    "logs": ["log", "logs", "stdout", "stderr"],
    "traces": ["trace", "traces", "span", "jaeger"],
    "topology": ["topology", "graph", "dependency", "service_map"],
    "ground_truth": ["ground_truth", "answer", "root_cause", "label"],
    "kubernetes_specs": ["deployment", "pod", "service", "configmap", "secret", "manifest", "spec"],
}


def normalize_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS


def classify_file(path: Path) -> list[str]:
    """
    Classify a file using its path/name only.
    This is intentionally simple for the MVP.
    """
    text = str(path).lower()
    categories: list[str] = []

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            categories.append(category)

    if not categories:
        categories.append("other")

    return categories


def list_scenario_files(scenario_dir: str | Path) -> list[dict[str, Any]]:
    """
    Return metadata for all files in a scenario folder.
    """
    root = normalize_path(scenario_dir)

    if not root.exists():
        raise FileNotFoundError(f"Scenario directory does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Scenario path is not a directory: {root}")

    files: list[dict[str, Any]] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        rel_path = path.relative_to(root)

        files.append(
            {
                "relative_path": str(rel_path),
                "absolute_path": str(path),
                "name": path.name,
                "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "is_text": is_text_file(path),
                "categories": classify_file(rel_path),
            }
        )

    return files


def read_text_preview(path: str | Path, max_chars: int = 2_000) -> str:
    """
    Safely read a small preview from a text-like file.
    """
    file_path = normalize_path(path)

    if not file_path.exists() or not file_path.is_file():
        return ""

    if not is_text_file(file_path):
        return ""

    try:
        return file_path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def find_files_by_category(
    scenario_dir: str | Path,
    category: str,
) -> list[dict[str, Any]]:
    """
    Find files that belong to a given category.
    Example categories: alerts, metrics, events, logs, topology, ground_truth.
    """
    files = list_scenario_files(scenario_dir)
    return [file for file in files if category in file["categories"]]


def load_ground_truth(scenario_dir: str | Path) -> dict[str, Any] | None:
    """
    Try to load ground_truth.yaml / ground_truth.yml if present.
    """
    root = normalize_path(scenario_dir)

    candidates = list(root.rglob("ground_truth.yaml")) + list(root.rglob("ground_truth.yml"))

    if not candidates:
        return None

    ground_truth_path = candidates[0]

    try:
        with ground_truth_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        return {
            "path": str(ground_truth_path.relative_to(root)),
            "data": data,
        }
    except Exception as exc:
        return {
            "path": str(ground_truth_path.relative_to(root)),
            "error": str(exc),
        }


def inspect_scenario(scenario_dir: str | Path) -> dict[str, Any]:
    """
    Main inspection function used by scripts and later by the agent.
    """
    root = normalize_path(scenario_dir)
    files = list_scenario_files(root)

    category_counts: dict[str, int] = {}

    for file in files:
        for category in file["categories"]:
            category_counts[category] = category_counts.get(category, 0) + 1

    ground_truth = load_ground_truth(root)

    return {
        "scenario_path": str(root),
        "scenario_name": root.name,
        "total_files": len(files),
        "category_counts": category_counts,
        "files": files,
        "ground_truth": ground_truth,
    }


def compact_scenario_summary(scenario_dir: str | Path) -> dict[str, Any]:
    """
    Smaller summary useful for printing and LLM context later.
    """
    inspection = inspect_scenario(scenario_dir)

    return {
        "scenario_path": inspection["scenario_path"],
        "scenario_name": inspection["scenario_name"],
        "total_files": inspection["total_files"],
        "category_counts": inspection["category_counts"],
        "ground_truth_available": inspection["ground_truth"] is not None,
        "files_by_category": {
            category: [
                file["relative_path"]
                for file in inspection["files"]
                if category in file["categories"]
            ]
            for category in CATEGORY_KEYWORDS.keys()
        },
    }