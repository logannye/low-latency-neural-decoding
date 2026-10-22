"""Day-level development roles; final recordings stay sealed until recipe freeze."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DEVELOPMENT_DAYS = tuple(f"S{i}" for i in range(5))
SEALED_DAYS = tuple(f"S{i}" for i in range(5, 13))
FORWARD_FOLDS = (
    {"name": "A", "train_days": ["S0", "S1"], "evaluation_days": ["S2"]},
    {"name": "B", "train_days": ["S0", "S1", "S2"], "evaluation_days": ["S3"]},
    {"name": "C", "train_days": ["S0", "S1", "S2", "S3"], "evaluation_days": ["S4"]},
)


def day_number(day: str) -> int:
    if not isinstance(day, str) or re.fullmatch(r"S(?:[0-9]|1[0-2])", day) is None:
        raise ValueError(f"Invalid H1 day: {day!r}")
    return int(day[1:])


def assert_allowed(
    records: list[dict[str, Any]], allowed_days: list[str] | tuple[str, ...]
) -> None:
    """Reject sealed, wrong-day, minival, duplicate, or mismatched stream inputs."""
    allowed = set(allowed_days)
    if not allowed or not allowed.issubset(DEVELOPMENT_DAYS):
        raise ValueError(
            "Development access may only allow S0-S4; sealed days need a frozen protocol"
        )
    seen_assets: set[str] = set()
    seen_files: set[str] = set()
    for record in records:
        day = record.get("day")
        day_number(day)
        if day not in allowed:
            raise ValueError(f"Split contamination: {day} is not in {sorted(allowed)}")
        if record.get("provider_split") != "held-in-calib":
            raise ValueError(
                "Only held-in-calib may enter development; minival overlaps calibration"
            )
        file_id = record.get("file_id", "")
        if not re.fullmatch(rf"{re.escape(day)}_set_[1-9][0-9]*", file_id):
            raise ValueError("File/day identity mismatch")
        asset_id = record.get("asset_id")
        if not asset_id or asset_id in seen_assets or file_id in seen_files:
            raise ValueError("Missing or duplicate stream/asset identity")
        seen_assets.add(asset_id)
        seen_files.add(file_id)


def split_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "development_folds": list(FORWARD_FOLDS),
        "sealed_early": {
            "train_days_after_recipe_freeze": list(DEVELOPMENT_DAYS),
            "evaluation_days": ["S5"],
            "status": "sealed",
        },
        "sealed_later": {
            "train_days_after_recipe_freeze": [f"S{i}" for i in range(6)],
            "evaluation_days": [f"S{i}" for i in range(6, 13)],
            "provider_split": "held-out-calib",
            "status": "sealed",
        },
        "minival_role": "overlapping interface checks only; excluded from training and scores",
        "stream_boundary": "Each provider file resets state; trials do not reset state",
        "later_evidence_label": "local evaluation on public FALCON calibration recordings",
    }


def write_split_manifest(path: str | Path) -> None:
    Path(path).write_text(json.dumps(split_manifest(), indent=2) + "\n")
