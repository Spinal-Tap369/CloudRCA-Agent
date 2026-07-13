from __future__ import annotations

import re
from typing import Any

from app.eval.ground_truth import GroundTruthGroup
from app.schemas import RootCauseEntity


WORKLOAD_KINDS = {"Deployment", "ReplicaSet", "StatefulSet", "DaemonSet", "Pod"}
CHAOS_KINDS = {"NetworkChaos", "PodChaos", "StressChaos", "JVMChaos", "Schedule"}
MATCH_STRENGTH = {
    "wrong": 0,
    "family": 1,
    "alias": 2,
    "group_filter": 3,
    "exact": 4,
}


def normalize_text(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9_.:/-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(value))


def filter_to_hint(pattern: str) -> str:
    text = str(pattern)
    text = text.replace("\\b", "")
    text = text.replace(".*", "")
    text = text.replace(".+", "")
    text = text.replace("^", "")
    text = text.replace("$", "")
    text = re.sub(r"\\[wdsWDDS][+*]?", "", text)
    text = re.sub(r"[^a-zA-Z0-9_.:/-]+$", "", text)
    return text.strip()


def kind_compatible(predicted_kind: str, gt_kind: str, allow_family: bool = True) -> bool:
    predicted = str(predicted_kind or "")
    expected = str(gt_kind or "")

    if not expected:
        return True

    if predicted.lower() == expected.lower():
        return True

    if allow_family and predicted in WORKLOAD_KINDS and expected in WORKLOAD_KINDS:
        return True

    if allow_family and predicted in CHAOS_KINDS and expected in CHAOS_KINDS:
        return True

    return False


def namespace_compatible(predicted_namespace: str | None, gt_namespace: str) -> bool:
    predicted = str(predicted_namespace or "").strip()
    expected = str(gt_namespace or "").strip()
    return not expected or not predicted or predicted == expected


def name_matches(value: str, names: list[str], filters: list[str]) -> bool:
    predicted = normalize_text(value)
    predicted_compact = compact_token(value)

    if not predicted or predicted in {"unknown", "none", "null", "n/a", "na"}:
        return False

    for name in names:
        candidate = normalize_text(name)
        candidate_compact = compact_token(name)

        if not candidate:
            continue

        if predicted == candidate:
            return True

        if len(candidate_compact) >= 4 and candidate_compact in predicted_compact:
            return True

        if len(predicted_compact) >= 4 and predicted_compact in candidate_compact:
            return True

    for pattern in filters:
        hint = filter_to_hint(pattern)
        hint_compact = compact_token(hint)

        if hint_compact and hint_compact in predicted_compact:
            return True

        try:
            if re.search(pattern, value, flags=re.IGNORECASE):
                return True
        except re.error:
            continue

    return False


def _matches_names(value: str, names: list[str]) -> bool:
    return name_matches(value, names, [])


def _matches_filters(value: str, filters: list[str]) -> bool:
    return name_matches(value, [], filters)


def group_names(group: GroundTruthGroup, aliases: dict[str, set[str]]) -> list[str]:
    names = [group.id, group.name]
    names.extend(aliases.get(group.id, set()))
    names.extend(filter_to_hint(item) for item in group.filters)
    return [name for name in names if name]


def entity_matches_group(
    entity: RootCauseEntity,
    group: GroundTruthGroup,
    aliases: dict[str, set[str]],
    allow_family: bool = True,
) -> bool:
    if not kind_compatible(entity.kind, group.kind, allow_family=allow_family):
        return False

    if not namespace_compatible(entity.namespace, group.namespace):
        return False

    return name_matches(entity.name, group_names(group, aliases), group.filters)


def classify_entity_group_match(
    entity: RootCauseEntity,
    group: GroundTruthGroup,
    aliases: dict[str, set[str]],
) -> str:
    if not namespace_compatible(entity.namespace, group.namespace):
        return "wrong"

    exact_kind = str(entity.kind or "").lower() == str(group.kind or "").lower()
    compatible_kind = kind_compatible(entity.kind, group.kind, allow_family=True)

    if not compatible_kind:
        return "wrong"

    canonical_names = [name for name in [group.id, group.name] if name]
    alias_names = sorted(aliases.get(group.id, set()))
    filter_match = _matches_filters(entity.name, group.filters)
    canonical_match = _matches_names(entity.name, canonical_names)
    alias_match = _matches_names(entity.name, alias_names)

    if exact_kind and canonical_match:
        return "exact"

    if exact_kind and filter_match:
        return "group_filter"

    if exact_kind and alias_match:
        return "alias"

    if compatible_kind and (canonical_match or filter_match or alias_match):
        return "family"

    return "wrong"


def stronger_match(left: str, right: str) -> str:
    return left if MATCH_STRENGTH.get(left, 0) >= MATCH_STRENGTH.get(right, 0) else right


def text_mentions_group(text: str, group: GroundTruthGroup, aliases: dict[str, set[str]]) -> bool:
    lowered = normalize_text(text)
    compact = compact_token(text)

    for name in group_names(group, aliases):
        name_norm = normalize_text(name)
        name_compact = compact_token(name)

        if name_norm and name_norm in lowered:
            return True

        if len(name_compact) >= 4 and name_compact in compact:
            return True

    return False


def token_overlap(left: str, right: str) -> float:
    left_tokens = {token for token in normalize_text(left).split() if len(token) >= 3}
    right_tokens = {token for token in normalize_text(right).split() if len(token) >= 3}

    if not left_tokens or not right_tokens:
        return 0.0

    return len(left_tokens.intersection(right_tokens)) / len(left_tokens.union(right_tokens))


def result_text(raw_result: dict[str, Any]) -> str:
    chunks = [
        str(raw_result.get("incident_summary") or ""),
        str(raw_result.get("reasoning_summary") or ""),
        " ".join(str(item) for item in raw_result.get("recommended_remediation") or []),
        " ".join(str(item) for item in raw_result.get("limitations") or []),
    ]

    for evidence in raw_result.get("evidence") or []:
        if isinstance(evidence, dict):
            chunks.append(str(evidence.get("summary") or ""))

    return "\n".join(chunks)
