from __future__ import annotations

import pytest

from neural_decoder.splits import FORWARD_FOLDS, assert_allowed, day_number, split_manifest


def _row(day: str, asset: str = "asset-1") -> dict[str, str]:
    return {
        "day": day,
        "file_id": f"{day}_set_1",
        "provider_split": "held-in-calib",
        "asset_id": asset,
    }


def test_forward_folds_are_day_disjoint_and_chronological() -> None:
    for fold in FORWARD_FOLDS:
        assert not set(fold["train_days"]) & set(fold["evaluation_days"])
        assert max(map(day_number, fold["train_days"])) < min(
            map(day_number, fold["evaluation_days"])
        )
    assert split_manifest()["sealed_later"]["status"] == "sealed"


@pytest.mark.parametrize("day", ["S2", "S5", "S6", "S10", "S12"])
def test_training_rejects_unauthorized_day(day: str) -> None:
    with pytest.raises(ValueError, match="contamination"):
        assert_allowed([_row(day)], ["S0", "S1"])


def test_cannot_authorize_sealed_days_by_changing_allowed_list() -> None:
    with pytest.raises(ValueError, match="sealed"):
        assert_allowed([_row("S5")], ["S5"])


def test_minival_and_duplicate_crops_are_not_new_independent_training_files() -> None:
    row = _row("S0")
    row["provider_split"] = "held-in-minival"
    with pytest.raises(ValueError, match="overlaps"):
        assert_allowed([row], ["S0"])
    with pytest.raises(ValueError, match="duplicate"):
        assert_allowed([_row("S0"), _row("S0")], ["S0"])


def test_file_day_mismatch_is_rejected() -> None:
    row = _row("S0")
    row["file_id"] = "S5_set_1"
    with pytest.raises(ValueError, match="mismatch"):
        assert_allowed([row], ["S0"])


def test_numeric_replay_order_differs_from_official_lexical_group_order() -> None:
    assert sorted(["S10", "S6"], key=day_number) == ["S6", "S10"]
    assert sorted(["S10", "S6"]) == ["S10", "S6"]
