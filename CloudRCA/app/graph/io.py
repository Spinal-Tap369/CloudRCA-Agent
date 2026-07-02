from __future__ import annotations

from pathlib import Path

from app.graph.models import EvidenceFile
from app.scenario_loader import inspect_scenario, is_text_file, normalize_path


def classify_file(relative_path: str) -> str:
    lower = relative_path.lower().replace("\\", "/")

    if "ground_truth" in lower:
        return "ground_truth"
    if "alert" in lower:
        return "alerts"
    if "trace" in lower or "span" in lower:
        return "traces"
    if "log" in lower:
        return "logs"
    if "metric" in lower:
        return "metrics"
    if "event" in lower:
        return "events"
    if "k8s_object" in lower or "kubernetes" in lower or "object" in lower:
        return "kubernetes_objects"
    if lower.endswith((".yaml", ".yml", ".json", ".tsv", ".csv", ".txt")):
        return "structured_text"

    return "other"


def read_text(path: Path, max_chars: int = 300_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def read_scenario_files(scenario_dir: str | Path) -> list[EvidenceFile]:
    root = normalize_path(scenario_dir)
    inspection = inspect_scenario(root)

    files: list[EvidenceFile] = []

    for file_info in inspection["files"]:
        rel = file_info["relative_path"]
        category = classify_file(rel)

        if category == "ground_truth":
            continue

        path = Path(file_info["absolute_path"])

        if not is_text_file(path):
            continue

        text = read_text(path)

        if not text.strip():
            continue

        files.append(
            EvidenceFile(
                relative_path=rel,
                absolute_path=path,
                category=category,
                text=text,
            )
        )

    return files
