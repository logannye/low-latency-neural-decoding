"""Guard the distinction between unchanged-target recovery and easier subsets."""

import copy

import pytest

from neural_experiments.v8.metrics import summarize


def rows():
    return [
        {
            "participant": "T2",
            "session_id": f"T2_day_{day}",
            "post_implant_day": day,
            "method": method,
            "score_mask_sha256": f"mask{day}",
            "metrics": {"r2": value, "summed_mse": 1 - value, "angular_error_degrees": 90},
            "secondary": {"onset": {"r2": 0.99}},
        }
        for day in (1, 2, 3)
        for method, value in [("zero", -0.1), ("baseline", -0.2), ("selected", 0.1)]
    ]


def test_primary_gate_requires_actual_primary_recovery():
    values = rows()
    assert summarize(values, "selected")["gates"]["useful_T2_primary_recovery"]
    for row in values:
        if row["method"] == "selected":
            row["metrics"]["r2"] = -0.05
    assert not summarize(values, "selected")["gates"]["useful_T2_primary_recovery"]


def test_missing_selected_sessions_cannot_pass():
    values = [r for r in rows() if r["post_implant_day"] != 3]
    assert not summarize(values, "selected")["gates"]["useful_T2_primary_recovery"]


def test_methods_cannot_change_mask_or_drop_cases():
    values = rows()
    values[0]["score_mask_sha256"] = "different"
    with pytest.raises(ValueError, match="scoring mask"):
        summarize(values, "selected")
    with pytest.raises(ValueError, match="complete unique"):
        summarize(rows()[:-1], "selected")


def test_participant_weighting_does_not_reward_more_sessions():
    values = rows()
    for day in range(8):
        for row in rows()[:3]:
            other = copy.deepcopy(row)
            other.update(participant="T5", session_id=f"T5_day_{day}", post_implant_day=day)
            other["metrics"]["r2"] = 0.5
            values.append(other)
    result = summarize(values, "selected")
    assert result["methods"]["selected"]["mean_r2"] == pytest.approx(0.3)


def test_unscorable_kept_in_denominator_and_never_passes_gate():
    values = rows()
    for row in values:
        if row["post_implant_day"] == 3:
            row["metrics"]["r2"] = None
            row["metrics"]["summed_mse"] = None
    result = summarize(values, "selected")
    assert result["selected_sessions"] == 3
    assert result["methods"]["selected"]["scored_sessions"] == 2
    assert not result["gates"]["useful_T2_primary_recovery"]


def test_factorial_contrasts_reconstruct_known_interaction():
    from neural_experiments.v8.metrics import factorial_effects

    values = []
    for r in (0, 1):
        for p in (0, 1):
            for w in (0, 1):
                name = (
                    f"{('baseline', 'selected')[r]}_{('all', 'clean')[p]}_"
                    f"{('uniform', 'equal_trial')[w]}"
                )
                if (r, p, w) == (0, 0, 0):
                    name = "baseline"
                values.append(
                    {
                        "participant": "T2",
                        "session_id": "case",
                        "method": name,
                        "metrics": {"r2": 0.2 * r + 0.3 * p + 0.4 * w + 0.5 * r * p},
                    }
                )
    effects = factorial_effects(values)
    assert effects["representation"]["T2_mean_effect"] == pytest.approx(0.45)
    assert effects["clean_phase"]["T2_mean_effect"] == pytest.approx(0.55)
    assert effects["equal_trial"]["T2_mean_effect"] == pytest.approx(0.4)
    assert effects["representation_x_phase"]["T2_mean_effect"] == pytest.approx(0.5)
    assert effects["three_way"]["T2_mean_effect"] == pytest.approx(0.0)
