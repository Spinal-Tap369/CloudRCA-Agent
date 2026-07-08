from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from app.graph.pack import normalize_name


def find_ground_truth_file(scenario_dir: Path) -> Path | None:
    for pattern in ["*ground*truth*.yaml", "*ground*truth*.yml", "*ground_truth*.json"]:
        matches = list(scenario_dir.rglob(pattern))

        if matches:
            return matches[0]

    return None


def load_ground_truth(scenario_dir: Path) -> dict[str, Any]:
    gt_file = find_ground_truth_file(scenario_dir)

    if gt_file is None:
        return {}

    text = gt_file.read_text(encoding="utf-8", errors="ignore")
    loaded = json.loads(text) if gt_file.suffix.lower() == ".json" else yaml.safe_load(text)

    if not isinstance(loaded, dict):
        return {}

    loaded["_ground_truth_file"] = str(gt_file)
    return loaded


def regex_filter_to_hint(value: str) -> str:
    value = str(value).strip()
    value = value.replace("\\b", "")
    value = value.replace(".*", "")
    value = value.replace(".+", "")
    value = value.replace("^", "")
    value = value.replace("$", "")
    value = value.replace("\\", "")
    return value.strip(" .*-_")


def extract_gt_root_info(gt: dict[str, Any]) -> dict[str, Any]:
    root_names: set[str] = set()
    root_kinds: set[str] = set()
    root_filters: set[str] = set()

    for root in extract_gt_roots(gt):
        root_names.update(root["root_names"])
        root_kinds.update(root["root_kinds"])
        root_filters.update(root["root_filters"])

    return {
        "root_names": sorted(root_names),
        "root_kinds": sorted(root_kinds),
        "root_filters": sorted(root_filters),
        "ground_truth_file": gt.get("_ground_truth_file", ""),
    }


def _payload(gt: dict[str, Any]) -> dict[str, Any]:
    return gt.get("spec") if isinstance(gt.get("spec"), dict) else gt


def extract_gt_roots(gt: dict[str, Any]) -> list[dict[str, Any]]:
    payload = _payload(gt)
    roots: list[dict[str, Any]] = []
    root_groups = [
        group
        for group in payload.get("groups", []) or []
        if isinstance(group, dict) and bool(group.get("root_cause"))
    ]

    if root_groups:
        for index, group in enumerate(root_groups, start=1):
            roots.append(_root_from_group(index, group))
    else:
        faults = payload.get("fault", []) or []

        if isinstance(faults, dict):
            faults = [faults]

        for index, fault in enumerate(faults, start=1):
            if isinstance(fault, dict):
                root = _root_from_fault(index, fault)

                if root:
                    roots.append(root)

    return roots


def extract_gt_root_groups(gt: dict[str, Any]) -> list[dict[str, Any]]:
    groups = []

    for group in _payload(gt).get("groups", []) or []:
        if not isinstance(group, dict) or not bool(group.get("root_cause")):
            continue

        groups.append(
            {
                "id": str(group.get("id") or ""),
                "kind": str(group.get("kind") or ""),
                "name": str(group.get("name") or ""),
                "namespace": str(group.get("namespace") or ""),
                "filters": [str(item) for item in group.get("filter", []) or []],
            }
        )

    return groups


def _short_text(value: Any, max_chars: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[: max_chars - 3] + "..." if len(text) > max_chars else text


def extract_gt_fault_entries(gt: dict[str, Any]) -> list[dict[str, Any]]:
    faults = _payload(gt).get("fault", []) or []

    if isinstance(faults, dict):
        faults = [faults]

    entries = []

    for index, fault in enumerate(faults, start=1):
        if not isinstance(fault, dict):
            continue

        entity = fault.get("entity", {})

        if not isinstance(entity, dict):
            entity = {}

        changed = fault.get("changed", {})

        if not isinstance(changed, dict):
            changed = {}

        name = str(entity.get("name") or entity.get("group_id") or "")
        kind = str(entity.get("kind") or "")

        entries.append(
            {
                "id": str(entity.get("group_id") or name or f"fault-{index}"),
                "kind": kind,
                "name": name,
                "namespace": str(entity.get("namespace") or ""),
                "condition": str(fault.get("condition") or ""),
                "category": str(fault.get("category") or ""),
                "fault_mechanism": str(fault.get("fault_mechanism") or ""),
                "changed_element": str(changed.get("element") or ""),
                "changed_from": _short_text(changed.get("from")),
                "changed_to": _short_text(changed.get("to")),
                "root_names": [name] if name else [],
                "root_kinds": [kind] if kind else [],
                "root_filters": [],
            }
        )

    return entries


def extract_gt_propagations(gt: dict[str, Any]) -> list[dict[str, str]]:
    propagations = []

    for item in _payload(gt).get("propagations", []) or []:
        if not isinstance(item, dict):
            continue

        propagations.append(
            {
                "source": str(item.get("source") or ""),
                "target": str(item.get("target") or ""),
                "condition": _short_text(item.get("condition")),
                "effect": _short_text(item.get("effect")),
            }
        )

    return propagations


def _root_from_group(index: int, group: dict[str, Any]) -> dict[str, Any]:
    root_names: set[str] = set()
    root_filters: set[str] = set()

    for key in ["id", "name"]:
        value = group.get(key)

        if value:
            root_names.add(str(value))

    for filter_value in group.get("filter", []) or []:
        filter_text = str(filter_value)
        root_filters.add(filter_text)
        hint = regex_filter_to_hint(filter_text)

        if hint:
            root_names.add(hint)

    kind = str(group.get("kind") or "")
    namespace = str(group.get("namespace") or "")
    root_id = str(group.get("id") or group.get("name") or f"root-{index}")

    return {
        "id": root_id,
        "source": "group",
        "kind": kind,
        "namespace": namespace,
        "root_names": sorted(root_names),
        "root_kinds": [kind] if kind else [],
        "root_filters": sorted(root_filters),
    }


def _root_from_fault(index: int, fault: dict[str, Any]) -> dict[str, Any] | None:
    entity = fault.get("entity", {})

    if not isinstance(entity, dict):
        return None

    root_names = {
        str(value)
        for value in [entity.get("name"), entity.get("group_id")]
        if value
    }
    kind = str(entity.get("kind") or "")
    root_id = str(entity.get("group_id") or entity.get("name") or f"root-{index}")

    return {
        "id": root_id,
        "source": "fault",
        "kind": kind,
        "namespace": str(entity.get("namespace") or ""),
        "root_names": sorted(root_names),
        "root_kinds": [kind] if kind else [],
        "root_filters": [],
    }


def node_matches_gt_kind(node_kind: str, gt_info: dict[str, Any]) -> bool:
    if not gt_info["root_kinds"]:
        return True

    candidate = str(node_kind).lower().strip()
    return candidate in {str(kind).lower().strip() for kind in gt_info["root_kinds"]}


def generated_name_match(candidate: str, target: str) -> bool:
    if candidate == target:
        return True

    if len(target) >= 4 and candidate.startswith(f"{target}-"):
        return True

    if len(candidate) >= 4 and target.startswith(f"{candidate}-"):
        return True

    return False


def name_aliases(value: str) -> list[str]:
    name = normalize_name(value)
    aliases = {name}

    if name.endswith("service") and len(name) > len("service"):
        aliases.add(name[: -len("service")])

    if name.endswith("-service") and len(name) > len("-service"):
        aliases.add(name[: -len("-service")])

    return sorted(alias for alias in aliases if alias)


def token_segment_match(candidate: str, target: str) -> bool:
    if len(target) < 2:
        return False

    return bool(re.search(rf"(?:^|-){re.escape(target)}(?:-|$)", candidate))


def node_matches_gt_name(node_name: str, gt_info: dict[str, Any]) -> bool:
    candidate = normalize_name(node_name)

    if not candidate:
        return False

    for raw_name in gt_info["root_names"]:
        for gt_name in name_aliases(raw_name):
            if generated_name_match(candidate, gt_name) or token_segment_match(candidate, gt_name):
                return True

    for filter_text in gt_info["root_filters"]:
        try:
            if re.search(filter_text, node_name):
                return True
        except re.error:
            pass

        hints = name_aliases(regex_filter_to_hint(filter_text))

        for hint in hints:
            if generated_name_match(candidate, hint) or token_segment_match(candidate, hint):
                return True

    return False


def node_matches_gt(node: Any, gt_info: dict[str, Any]) -> bool:
    return node_matches_gt_kind(node.kind, gt_info) and node_matches_gt_name(node.name, gt_info)


def node_matches_gt_root(node: Any, gt_root: dict[str, Any]) -> bool:
    namespace = str(gt_root.get("namespace") or "").strip()

    if namespace and node.kind != "Namespace" and node.namespace:
        if normalize_name(node.namespace) != normalize_name(namespace):
            return False

    return node_matches_gt_kind(node.kind, gt_root) and node_matches_gt_name(node.name, gt_root)
