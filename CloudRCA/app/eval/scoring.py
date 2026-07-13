from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.eval.ground_truth import GroundTruthCase, load_ground_truth
from app.eval.matching import (
    MATCH_STRENGTH,
    classify_entity_group_match,
    kind_compatible,
    result_text,
    stronger_match,
    text_mentions_group,
    token_overlap,
)
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

    passed_strict: bool
    passed_family: bool
    best_match_type: str
    root_exact_match: bool
    root_group_match: bool
    root_alias_match: bool
    root_family_match: bool

    predicted_root_count: int
    ground_truth_root_count: int
    matched_root_count_strict: int
    matched_root_count_family: int

    root_group_precision: float = Field(ge=0.0, le=1.0)
    root_group_recall: float = Field(ge=0.0, le=1.0)
    root_group_f1: float = Field(ge=0.0, le=1.0)
    root_family_precision: float = Field(ge=0.0, le=1.0)
    root_family_recall: float = Field(ge=0.0, le=1.0)
    root_family_f1: float = Field(ge=0.0, le=1.0)
    root_group_matches: list[str]
    root_family_matches: list[str]
    root_match_details: list[dict[str, str]]
    unmatched_predicted_entities: list[str]
    unmatched_ground_truth_roots: list[str]

    kind_match_rate: float = Field(ge=0.0, le=1.0)
    propagation_coverage: float = Field(ge=0.0, le=1.0)
    alert_coverage: float = Field(ge=0.0, le=1.0)
    evidence_quality_score: float = Field(ge=0.0, le=1.0)
    remediation_match_score: float = Field(ge=0.0, le=1.0)
    safety_schema_score: float = Field(ge=0.0, le=1.0)

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


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _root_group_id(group_id: str, group_name: str) -> str:
    return group_id or group_name


def _entity_label(entity: Any) -> str:
    namespace = getattr(entity, "namespace", None) or ""
    return f"{getattr(entity, 'kind', '')}:{namespace}:{getattr(entity, 'name', '')}"


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _root_matches(diagnosis: DiagnosisResult | None, gt: GroundTruthCase) -> dict[str, Any]:
    if diagnosis is None:
        return {
            "predicted_count": 0,
            "ground_truth_count": len(gt.root_groups),
            "strict_precision": 0.0,
            "strict_recall": 0.0,
            "strict_f1": 0.0,
            "family_precision": 0.0,
            "family_recall": 0.0,
            "family_f1": 0.0,
            "strict_matches": [],
            "family_matches": [],
            "details": [],
            "unmatched_predictions": [],
            "unmatched_gt": [group.id for group in gt.root_groups],
            "best_match_type": "wrong",
        }

    predictions = [
        entity
        for entity in diagnosis.root_cause_entities
        if entity.kind != "Unknown" and entity.name != "Unknown"
    ]
    strict_groups: set[str] = set()
    family_groups: set[str] = set()
    strict_predictions: set[int] = set()
    family_predictions: set[int] = set()
    details: list[dict[str, str]] = []
    best_match_type = "wrong"

    for index, entity in enumerate(predictions):
        for group in gt.root_groups:
            match_type = classify_entity_group_match(entity, group, gt.aliases)

            if match_type == "wrong":
                continue

            group_id = _root_group_id(group.id, group.name)
            best_match_type = stronger_match(best_match_type, match_type)
            details.append(
                {
                    "predicted": _entity_label(entity),
                    "ground_truth": group_id,
                    "match_type": match_type,
                }
            )

            if match_type in {"exact", "group_filter", "alias"}:
                strict_groups.add(group_id)
                strict_predictions.add(index)

            if match_type in {"exact", "group_filter", "alias", "family"}:
                family_groups.add(group_id)
                family_predictions.add(index)

    strict_precision = len(strict_predictions) / len(predictions) if predictions else 0.0
    strict_recall = len(strict_groups) / len(gt.root_groups) if gt.root_groups else 0.0
    family_precision = len(family_predictions) / len(predictions) if predictions else 0.0
    family_recall = len(family_groups) / len(gt.root_groups) if gt.root_groups else 0.0

    unmatched_predictions = [
        _entity_label(entity)
        for index, entity in enumerate(predictions)
        if index not in family_predictions
    ]
    unmatched_gt = [
        _root_group_id(group.id, group.name)
        for group in gt.root_groups
        if _root_group_id(group.id, group.name) not in family_groups
    ]

    return {
        "predicted_count": len(predictions),
        "ground_truth_count": len(gt.root_groups),
        "strict_precision": strict_precision,
        "strict_recall": strict_recall,
        "strict_f1": _f1(strict_precision, strict_recall),
        "family_precision": family_precision,
        "family_recall": family_recall,
        "family_f1": _f1(family_precision, family_recall),
        "strict_matches": sorted(strict_groups),
        "family_matches": sorted(family_groups),
        "details": details,
        "unmatched_predictions": unmatched_predictions,
        "unmatched_gt": unmatched_gt,
        "best_match_type": best_match_type,
    }


def _kind_match_rate(diagnosis: DiagnosisResult | None, gt: GroundTruthCase) -> float:
    if diagnosis is None or not diagnosis.root_cause_entities or not gt.root_groups:
        return 0.0

    matches = 0

    for group in gt.root_groups:
        if any(kind_compatible(entity.kind, group.kind, allow_family=True) for entity in diagnosis.root_cause_entities):
            matches += 1

    return matches / len(gt.root_groups)


def _propagation_coverage(text: str, gt: GroundTruthCase) -> float:
    target_ids = {
        propagation.target
        for propagation in gt.propagations
        if propagation.target and propagation.source
    }
    target_groups = [gt.groups[item] for item in target_ids if item in gt.groups]

    if not target_groups:
        return 1.0

    covered = sum(1 for group in target_groups if text_mentions_group(text, group, gt.aliases))
    return covered / len(target_groups)


def _alert_coverage(text: str, gt: GroundTruthCase) -> float:
    group_ids = {alert.group_id for alert in gt.alerts if alert.group_id}
    groups = [gt.groups[item] for item in group_ids if item in gt.groups]

    if not groups:
        return 1.0

    covered = sum(1 for group in groups if text_mentions_group(text, group, gt.aliases))
    return covered / len(groups)


def _evidence_quality(diagnosis: DiagnosisResult | None) -> float:
    if diagnosis is None:
        return 0.0

    if not diagnosis.evidence:
        return 0.0

    source_types = {item.source_type for item in diagnosis.evidence}
    supporting = sum(1 for item in diagnosis.evidence if item.supports_root_cause)
    score = 0.45

    if supporting:
        score += 0.25

    if len(source_types) >= 2:
        score += 0.15

    if any(item.source_path for item in diagnosis.evidence):
        score += 0.15

    return min(score, 1.0)


def _remediation_score(diagnosis: DiagnosisResult | None, gt: GroundTruthCase) -> float:
    if diagnosis is None or not diagnosis.recommended_remediation:
        return 0.0

    if not gt.recommended_actions:
        return 1.0

    best = 0.0

    for predicted in diagnosis.recommended_remediation:
        for expected in gt.recommended_actions:
            best = max(best, token_overlap(predicted, expected))

    return min(best * 2.5, 1.0)


def _safety_schema_score(valid_schema: bool, diagnosis: DiagnosisResult | None) -> float:
    if not valid_schema or diagnosis is None:
        return 0.0

    score = 0.5

    if diagnosis.should_auto_remediate is False:
        score += 0.3

    if diagnosis.root_cause_entities:
        score += 0.1

    if diagnosis.limitations is not None:
        score += 0.1

    return min(score, 1.0)


def _ground_truth_lists(gt: GroundTruthCase) -> dict[str, list[str]]:
    aliases = sorted({item for values in gt.aliases.values() for item in values})
    filters = sorted({item for group in gt.root_groups for item in group.filters})
    return {
        "entities": sorted({group.id or group.name for group in gt.root_groups if group.id or group.name}),
        "kinds": sorted({group.kind for group in gt.root_groups if group.kind}),
        "aliases": aliases,
        "filters": filters,
        "recommended_actions": gt.recommended_actions,
    }


def evaluate_result(scenario_dir: str | Path, result_file: str | Path) -> EvalResult:
    scenario_path = Path(scenario_dir)
    result_path = Path(result_file)
    raw_result = _load_json(result_path)
    notes: list[str] = []

    valid_schema = True

    try:
        diagnosis = DiagnosisResult.model_validate(raw_result)
    except ValidationError as exc:
        valid_schema = False
        diagnosis = None
        notes.append(f"Result JSON failed schema validation: {exc}")

    gt = load_ground_truth(scenario_path)

    if gt.path is None:
        notes.append("No ground_truth.yaml/yml file found.")

    scenario_id = (
        diagnosis.scenario_id
        if diagnosis is not None
        else str(raw_result.get("scenario_id") or scenario_path.name)
    )
    predicted_entities = [
        entity.name for entity in diagnosis.root_cause_entities
    ] if diagnosis is not None else [
        str(entity.get("name", ""))
        for entity in raw_result.get("root_cause_entities", []) or []
        if isinstance(entity, dict)
    ]
    predicted_kinds = [
        entity.kind for entity in diagnosis.root_cause_entities
    ] if diagnosis is not None else [
        str(entity.get("kind", ""))
        for entity in raw_result.get("root_cause_entities", []) or []
        if isinstance(entity, dict)
    ]

    root_match = _root_matches(diagnosis, gt)
    text = result_text(raw_result)
    kind_rate = _kind_match_rate(diagnosis, gt)
    propagation = _propagation_coverage(text, gt)
    alert = _alert_coverage(text, gt)
    evidence_quality = _evidence_quality(diagnosis)
    remediation = _remediation_score(diagnosis, gt)
    safety = _safety_schema_score(valid_schema, diagnosis)

    has_evidence = bool(diagnosis and diagnosis.evidence)
    has_remediation = bool(diagnosis and diagnosis.recommended_remediation)
    safe_no_auto_remediation = bool(diagnosis and diagnosis.should_auto_remediate is False)
    strict_root_correct = bool(gt.root_groups) and len(root_match["strict_matches"]) == len(gt.root_groups)
    family_root_correct = bool(gt.root_groups) and len(root_match["family_matches"]) == len(gt.root_groups)
    entity_name_match = root_match["strict_f1"] > 0
    entity_kind_match = kind_rate > 0

    if root_match["strict_matches"]:
        notes.append(f"Strict matched root groups: {', '.join(root_match['strict_matches'])}")
    if root_match["family_matches"] and root_match["family_matches"] != root_match["strict_matches"]:
        notes.append(f"Family matched root groups: {', '.join(root_match['family_matches'])}")
    if root_match["unmatched_gt"]:
        notes.append(f"Unmatched ground-truth root groups: {', '.join(root_match['unmatched_gt'])}")
    if root_match["unmatched_predictions"]:
        notes.append(f"Unmatched predicted roots: {', '.join(root_match['unmatched_predictions'])}")
    if propagation < 1.0:
        notes.append(f"Propagation coverage is partial: {propagation:.2f}")
    if alert < 1.0:
        notes.append(f"Alert-group coverage is partial: {alert:.2f}")

    score = 1.0 if family_root_correct else 0.0
    gt_lists = _ground_truth_lists(gt)

    return EvalResult(
        scenario_id=scenario_id,
        result_file=str(result_path),
        ground_truth_file=str(gt.path) if gt.path else None,
        valid_schema=valid_schema,
        entity_name_match=entity_name_match,
        entity_kind_match=entity_kind_match,
        has_evidence=has_evidence,
        has_remediation=has_remediation,
        safe_no_auto_remediation=safe_no_auto_remediation,
        passed_strict=strict_root_correct,
        passed_family=family_root_correct,
        best_match_type=str(root_match["best_match_type"]),
        root_exact_match=root_match["best_match_type"] == "exact",
        root_group_match=root_match["best_match_type"] == "group_filter",
        root_alias_match=root_match["best_match_type"] == "alias",
        root_family_match=root_match["best_match_type"] == "family",
        predicted_root_count=int(root_match["predicted_count"]),
        ground_truth_root_count=int(root_match["ground_truth_count"]),
        matched_root_count_strict=len(root_match["strict_matches"]),
        matched_root_count_family=len(root_match["family_matches"]),
        root_group_precision=round(root_match["strict_precision"], 3),
        root_group_recall=round(root_match["strict_recall"], 3),
        root_group_f1=round(root_match["strict_f1"], 3),
        root_family_precision=round(root_match["family_precision"], 3),
        root_family_recall=round(root_match["family_recall"], 3),
        root_family_f1=round(root_match["family_f1"], 3),
        root_group_matches=root_match["strict_matches"],
        root_family_matches=root_match["family_matches"],
        root_match_details=root_match["details"],
        unmatched_predicted_entities=root_match["unmatched_predictions"],
        unmatched_ground_truth_roots=root_match["unmatched_gt"],
        kind_match_rate=round(kind_rate, 3),
        propagation_coverage=round(propagation, 3),
        alert_coverage=round(alert, 3),
        evidence_quality_score=round(evidence_quality, 3),
        remediation_match_score=round(remediation, 3),
        safety_schema_score=round(safety, 3),
        predicted_entities=predicted_entities,
        predicted_kinds=predicted_kinds,
        ground_truth_entities=gt_lists["entities"],
        ground_truth_kinds=gt_lists["kinds"],
        ground_truth_aliases=gt_lists["aliases"],
        ground_truth_filters=gt_lists["filters"],
        ground_truth_recommended_actions=gt_lists["recommended_actions"],
        ground_truth_preview=gt.raw_text[:1500],
        score=round(score, 3),
        notes=notes,
    )
