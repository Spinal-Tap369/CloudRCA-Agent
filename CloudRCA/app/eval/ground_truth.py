from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GroundTruthGroup:
    id: str
    kind: str
    name: str = ""
    namespace: str = ""
    filters: list[str] = field(default_factory=list)
    root_cause: bool = False


@dataclass
class GroundTruthPropagation:
    source: str
    target: str
    condition: str = ""
    effect: str = ""


@dataclass
class GroundTruthAlert:
    id: str
    group_id: str
    description: str = ""


@dataclass
class GroundTruthCase:
    path: Path | None
    raw_text: str
    raw: dict[str, Any]
    root_groups: list[GroundTruthGroup]
    groups: dict[str, GroundTruthGroup]
    aliases: dict[str, set[str]]
    propagations: list[GroundTruthPropagation]
    alerts: list[GroundTruthAlert]
    recommended_actions: list[str]


def safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def find_ground_truth_file(scenario_dir: str | Path) -> Path | None:
    root = Path(scenario_dir)
    candidates = (
        list(root.rglob("ground_truth.yaml"))
        + list(root.rglob("ground_truth.yml"))
        + list(root.rglob("*ground*truth*.yaml"))
        + list(root.rglob("*ground*truth*.yml"))
    )
    return candidates[0] if candidates else None


def _group_from_raw(raw: dict[str, Any]) -> GroundTruthGroup:
    filters = [str(item) for item in safe_list(raw.get("filter")) if item]
    return GroundTruthGroup(
        id=str(raw.get("id") or ""),
        kind=str(raw.get("kind") or ""),
        name=str(raw.get("name") or ""),
        namespace=str(raw.get("namespace") or ""),
        filters=filters,
        root_cause=raw.get("root_cause") is True,
    )


def _alias_map(raw_aliases: Any) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = {}

    for group in safe_list(raw_aliases):
        if not isinstance(group, list):
            continue

        values = {str(item) for item in group if item}

        for value in values:
            aliases.setdefault(value, set()).update(values - {value})

    return aliases


def _recommended_actions(raw: dict[str, Any]) -> list[str]:
    actions: list[str] = []

    for item in safe_list(raw.get("recommended_actions")):
        if not isinstance(item, dict):
            continue

        solution = item.get("solution")

        if not isinstance(solution, dict):
            continue

        for action in safe_list(solution.get("actions")):
            if action:
                actions.append(str(action))

    return actions


def _fault_root_groups(raw: dict[str, Any], groups: dict[str, GroundTruthGroup]) -> list[GroundTruthGroup]:
    roots: list[GroundTruthGroup] = []

    for fault in safe_list(raw.get("fault")):
        if not isinstance(fault, dict):
            continue

        entity = fault.get("entity")

        if not isinstance(entity, dict):
            continue

        group_id = str(entity.get("group_id") or "")

        if group_id and group_id in groups:
            roots.append(groups[group_id])
            continue

        name = str(entity.get("name") or group_id or "")
        kind = str(entity.get("kind") or "")

        if name or kind:
            roots.append(
                GroundTruthGroup(
                    id=group_id or name,
                    kind=kind,
                    name=name,
                    namespace=str(entity.get("namespace") or ""),
                    root_cause=True,
                )
            )

    return roots


def load_ground_truth(scenario_dir: str | Path) -> GroundTruthCase:
    path = find_ground_truth_file(scenario_dir)

    if path is None:
        return GroundTruthCase(
            path=None,
            raw_text="",
            raw={},
            root_groups=[],
            groups={},
            aliases={},
            propagations=[],
            alerts=[],
            recommended_actions=[],
        )

    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    raw = yaml.safe_load(raw_text) or {}

    if not isinstance(raw, dict):
        raw = {}

    body = raw.get("spec") if isinstance(raw.get("spec"), dict) else raw

    groups = {
        group.id: group
        for group in (_group_from_raw(item) for item in safe_list(body.get("groups")) if isinstance(item, dict))
        if group.id
    }
    root_groups = [group for group in groups.values() if group.root_cause]

    if not root_groups:
        for group in _fault_root_groups(body, groups):
            if group.id and all(existing.id != group.id for existing in root_groups):
                root_groups.append(group)

    aliases = _alias_map(body.get("aliases"))
    propagations = [
        GroundTruthPropagation(
            source=str(item.get("source") or ""),
            target=str(item.get("target") or ""),
            condition=str(item.get("condition") or ""),
            effect=str(item.get("effect") or ""),
        )
        for item in safe_list(body.get("propagations"))
        if isinstance(item, dict)
    ]
    alerts = [
        GroundTruthAlert(
            id=str(item.get("id") or ""),
            group_id=str(item.get("group_id") or ""),
            description=str((item.get("metadata") or {}).get("description") or "")
            if isinstance(item.get("metadata"), dict)
            else "",
        )
        for item in safe_list(body.get("alerts"))
        if isinstance(item, dict)
    ]

    return GroundTruthCase(
        path=path,
        raw_text=raw_text,
        raw=raw,
        root_groups=root_groups,
        groups=groups,
        aliases=aliases,
        propagations=propagations,
        alerts=alerts,
        recommended_actions=_recommended_actions(body),
    )
