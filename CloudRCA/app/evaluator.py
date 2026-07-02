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

    ground_truth_entities: list[str]
    ground_truth_kinds: list[str]
    ground_truth_aliases: list[str]
    ground_truth_filters: list[str]
    ground_truth_recommended_actions: list[str]

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


def _safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _filter_to_hint(pattern: str) -> str:
    """
    Convert simple regex-like filters into useful text hints.

    Examples:
    load-generator-.* -> load-generator
    frontend-proxy\\b -> frontend-proxy
    """
    text = str(pattern)
    text = text.replace("\\b", "")
    text = text.replace(".*", "")
    text = text.replace(".+", "")
    text = text.replace("^", "")
    text = text.replace("$", "")
    text = text.strip()

    # Remove trailing regex syntax / punctuation.
    text = re.sub(r"[^a-zA-Z0-9_.:/-]+$", "", text)

    return text


def _load_ground_truth_structured(scenario_dir: str | Path) -> tuple[Path | None, dict[str, Any], str]:
    gt_path = _find_ground_truth_file(scenario_dir)

    if gt_path is None:
        return None, {}, ""

    raw_text = gt_path.read_text(encoding="utf-8", errors="ignore")

    try:
        raw = yaml.safe_load(raw_text) or {}
    except Exception:
        raw = {}

    return gt_path, raw, raw_text


def _extract_ground_truth_summary(gt: dict[str, Any]) -> dict[str, list[str]]:
    root_names: set[str] = set()
    root_kinds: set[str] = set()
    root_group_ids: set[str] = set()
    aliases: set[str] = set()
    filters: set[str] = set()
    recommended_actions: set[str] = set()

    # fault.entity is usually the strongest label.
    for fault in _safe_list(gt.get("fault")):
        if not isinstance(fault, dict):
            continue

        entity = fault.get("entity", {})
        if isinstance(entity, dict):
            name = entity.get("name")
            group_id = entity.get("group_id")
            kind = entity.get("kind")

            if name:
                root_names.add(str(name))
            if group_id:
                root_names.add(str(group_id))
                root_group_ids.add(str(group_id))
            if kind:
                root_kinds.add(str(kind))

    # groups can identify root cause and regex filters.
    for group in _safe_list(gt.get("groups")):
        if not isinstance(group, dict):
            continue

        group_id = str(group.get("id", ""))
        kind = str(group.get("kind", ""))

        if group.get("root_cause") is True:
            if group_id:
                root_names.add(group_id)
                root_group_ids.add(group_id)
            if kind:
                root_kinds.add(kind)

            for item in _safe_list(group.get("filter")):
                if item:
                    filters.add(str(item))
                    hint = _filter_to_hint(str(item))
                    if hint:
                        root_names.add(hint)

    # aliases can connect equivalent/related benchmark group IDs.
    for alias_group in _safe_list(gt.get("aliases")):
        if not isinstance(alias_group, list):
            continue

        alias_values = [str(item) for item in alias_group]

        if any(item in root_group_ids or item in root_names for item in alias_values):
            for item in alias_values:
                aliases.add(item)

    # Recommended actions.
    for item in _safe_list(gt.get("recommended_actions")):
        if not isinstance(item, dict):
            continue

        solution = item.get("solution", {})
        if not isinstance(solution, dict):
            continue

        for action in _safe_list(solution.get("actions")):
            if action:
                recommended_actions.add(str(action))

    return {
        "root_names": sorted(root_names),
        "root_kinds": sorted(root_kinds),
        "aliases": sorted(aliases),
        "filters": sorted(filters),
        "recommended_actions": sorted(recommended_actions),
    }


def _is_weak_prediction(value: str) -> bool:
    norm = _normalize_text(value)
    weak_values = {"unknown", "none", "null", "n/a", "na", "-"}
    return not norm or norm in weak_values or len(norm) < 3


def _matches_any_name(predicted: str, gt_names: list[str], gt_aliases: list[str], gt_filters: list[str]) -> bool:
    if _is_weak_prediction(predicted):
        return False

    predicted_norm = _normalize_text(predicted)

    candidates = list(gt_names) + list(gt_aliases)

    for candidate in candidates:
        candidate_norm = _normalize_text(candidate)
        if not candidate_norm or len(candidate_norm) < 3:
            continue

        if predicted_norm == candidate_norm:
            return True

        if predicted_norm in candidate_norm:
            return True

        if candidate_norm in predicted_norm:
            return True

    # Match regex-like filters and simplified hints.
    for pattern in gt_filters:
        hint = _filter_to_hint(pattern)
        hint_norm = _normalize_text(hint)

        if hint_norm and (hint_norm in predicted_norm or predicted_norm in hint_norm):
            return True

        try:
            if re.search(pattern, predicted, flags=re.IGNORECASE):
                return True
        except re.error:
            pass

    return False


def _matches_any_kind(predicted: str, gt_kinds: list[str]) -> bool:
    if _is_weak_prediction(predicted):
        return False

    predicted_norm = _normalize_text(predicted)

    for kind in gt_kinds:
        kind_norm = _normalize_text(kind)
        if predicted_norm == kind_norm:
            return True

    return False


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
        diagnosis = None

    gt_path, gt_structured, gt_raw_text = _load_ground_truth_structured(scenario_path)

    if gt_path is None:
        notes.append("No ground_truth.yaml/yml file found.")

    gt_summary = _extract_ground_truth_summary(gt_structured)

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
        _matches_any_name(
            predicted=entity_name,
            gt_names=gt_summary["root_names"],
            gt_aliases=gt_summary["aliases"],
            gt_filters=gt_summary["filters"],
        )
        for entity_name in predicted_entities
    )

    entity_kind_match = any(
        _matches_any_kind(kind, gt_summary["root_kinds"])
        for kind in predicted_kinds
    )

    if gt_summary["root_names"]:
        notes.append(f"Ground truth root entities: {', '.join(gt_summary['root_names'])}")

    if not entity_name_match:
        notes.append("Predicted root-cause entity name did not match ground truth entity, aliases, or filters.")

    if not entity_kind_match:
        notes.append("Predicted entity kind/type did not match ground truth kind.")

    if not has_evidence:
        notes.append("Agent output contains no evidence items.")

    if not has_remediation:
        notes.append("Agent output contains no remediation recommendations.")

    if not safe_no_auto_remediation:
        notes.append("Agent should not auto-remediate in this MVP.")

    # Simple weighted MVP score.
    score = 0.0
    score += 0.35 if valid_schema else 0.0
    score += 0.35 if entity_name_match else 0.0
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
        ground_truth_entities=gt_summary["root_names"],
        ground_truth_kinds=gt_summary["root_kinds"],
        ground_truth_aliases=gt_summary["aliases"],
        ground_truth_filters=gt_summary["filters"],
        ground_truth_recommended_actions=gt_summary["recommended_actions"],
        ground_truth_preview=gt_raw_text[:1500],
        score=round(score, 3),
        notes=notes,
    )
