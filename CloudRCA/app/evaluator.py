from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from app.schemas import DiagnosisResult


class EvalResult(BaseModel):
    scenario_id: str
    result_file: str
    ground_truth_file: str | None

    valid_schema: bool
    entity_name_match: bool
    entity_kind_match: bool
    has_evidence: bool
    has_remediation: bool
    safe_no_auto_remediation: bool

    predicted_entities: list[str]
    predicted_kinds: list[str]
    ground_truth_preview: str

    score: float = Field(ge=0.0, le=1.0)
    notes: list[str]


def _normalize_text(value: object) -> str:
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9_.:/-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    return json.loads(file_path.read_text(encoding="utf-8"))


def _find_ground_truth_file(scenario_dir: str | Path) -> Path | None:
    root = Path(scenario_dir)

    candidates = (
        list(root.rglob("ground_truth.yaml"))
        + list(root.rglob("ground_truth.yml"))
        + list(root.rglob("*ground*truth*.yaml"))
        + list(root.rglob("*ground*truth*.yml"))
    )

    if not candidates:
        return None

    return candidates[0]


def _load_ground_truth_text(scenario_dir: str | Path) -> tuple[Path | None, str]:
    gt_path = _find_ground_truth_file(scenario_dir)

    if gt_path is None:
        return None, ""

    try:
        raw = yaml.safe_load(gt_path.read_text(encoding="utf-8"))
    except Exception:
        raw = gt_path.read_text(encoding="utf-8", errors="ignore")

    gt_text = json.dumps(raw, indent=2, default=str) if not isinstance(raw, str) else raw
    return gt_path, gt_text


def _contains_meaningful_match(needle: str, haystack: str) -> bool:
    """
    Conservative-ish text match.

    We avoid counting Unknown/null/very-short strings as matches.
    """
    needle_norm = _normalize_text(needle)
    haystack_norm = _normalize_text(haystack)

    if not needle_norm:
        return False

    weak_values = {"unknown", "none", "null", "n/a", "na", "-"}
    if needle_norm in weak_values:
        return False

    # Avoid silly one/two character matches.
    if len(needle_norm) < 3:
        return False

    return needle_norm in haystack_norm


def evaluate_result(
    scenario_dir: str | Path,
    result_file: str | Path,
) -> EvalResult:
    scenario_path = Path(scenario_dir)
    result_path = Path(result_file)

    notes: list[str] = []

    raw_result = _load_json(result_path)

    valid_schema = True
    try:
        diagnosis = DiagnosisResult.model_validate(raw_result)
    except ValidationError as exc:
        valid_schema = False
        notes.append(f"Result JSON failed schema validation: {exc}")
        # Try to continue using raw fields where possible.
        diagnosis = None

    gt_path, gt_text = _load_ground_truth_text(scenario_path)

    if gt_path is None:
        notes.append("No ground_truth.yaml/yml file found.")
    elif not gt_text.strip():
        notes.append("Ground truth file found but could not be read.")

    if diagnosis is not None:
        scenario_id = diagnosis.scenario_id
        predicted_entities = [entity.name for entity in diagnosis.root_cause_entities]
        predicted_kinds = [entity.kind for entity in diagnosis.root_cause_entities]

        has_evidence = len(diagnosis.evidence) > 0
        has_remediation = len(diagnosis.recommended_remediation) > 0
        safe_no_auto_remediation = diagnosis.should_auto_remediate is False
    else:
        scenario_id = str(raw_result.get("scenario_id", scenario_path.name))
        entities_raw = raw_result.get("root_cause_entities", []) or []
        predicted_entities = [
            str(entity.get("name", ""))
            for entity in entities_raw
            if isinstance(entity, dict)
        ]
        predicted_kinds = [
            str(entity.get("kind", ""))
            for entity in entities_raw
            if isinstance(entity, dict)
        ]

        has_evidence = bool(raw_result.get("evidence"))
        has_remediation = bool(raw_result.get("recommended_remediation"))
        safe_no_auto_remediation = raw_result.get("should_auto_remediate") is False

    entity_name_match = any(
        _contains_meaningful_match(entity_name, gt_text)
        for entity_name in predicted_entities
    )

    entity_kind_match = any(
        _contains_meaningful_match(kind, gt_text)
        for kind in predicted_kinds
    )

    if not entity_name_match:
        notes.append("Predicted root-cause entity name was not found in ground truth text.")

    if not entity_kind_match:
        notes.append("Predicted entity kind/type was not found in ground truth text.")

    if not has_evidence:
        notes.append("Agent output contains no evidence items.")

    if not has_remediation:
        notes.append("Agent output contains no remediation recommendations.")

    if not safe_no_auto_remediation:
        notes.append("Agent should not auto-remediate in this MVP.")

    # Simple weighted MVP score.
    score = 0.0
    score += 0.40 if valid_schema else 0.0
    score += 0.30 if entity_name_match else 0.0
    score += 0.10 if entity_kind_match else 0.0
    score += 0.10 if has_evidence else 0.0
    score += 0.05 if has_remediation else 0.0
    score += 0.05 if safe_no_auto_remediation else 0.0

    return EvalResult(
        scenario_id=scenario_id,
        result_file=str(result_path),
        ground_truth_file=str(gt_path) if gt_path else None,
        valid_schema=valid_schema,
        entity_name_match=entity_name_match,
        entity_kind_match=entity_kind_match,
        has_evidence=has_evidence,
        has_remediation=has_remediation,
        safe_no_auto_remediation=safe_no_auto_remediation,
        predicted_entities=predicted_entities,
        predicted_kinds=predicted_kinds,
        ground_truth_preview=gt_text[:1500],
        score=round(score, 3),
        notes=notes,
    )