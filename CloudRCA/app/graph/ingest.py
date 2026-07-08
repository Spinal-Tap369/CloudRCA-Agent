from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


try:
    csv.field_size_limit(2_147_483_647)
except OverflowError:
    csv.field_size_limit(1_000_000_000)


def normalize_token(value: str) -> str:
    value = str(value).lower().strip().replace("_", "-")
    value = re.sub(r"[^a-z0-9-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-")


def has_any(text: str, terms: Iterable[str]) -> bool:
    lower = text.lower()
    return any(term in lower for term in terms)


def relative_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def iter_tsv(path: Path) -> Iterable[dict[str, str]]:
    if not path.exists():
        return

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as file:
        yield from csv.DictReader(file, delimiter="\t")


def parse_json(value: str) -> Any:
    text = str(value or "").strip()

    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_mapping(value: str) -> dict[str, Any]:
    text = str(value or "").strip()

    if not text:
        return {}

    try:
        loaded = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {}

    return loaded if isinstance(loaded, dict) else {}


def name_mentioned(name: str, text: str) -> bool:
    token = normalize_token(name)

    if not token:
        return False

    normalized_text = normalize_token(text)

    if len(token) <= 3:
        return bool(re.search(rf"(?:^|-){re.escape(token)}(?:-|$)", normalized_text))

    return token in normalized_text
