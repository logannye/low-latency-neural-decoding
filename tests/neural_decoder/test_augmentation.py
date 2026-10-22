import copy

import numpy as np
import pytest
import torch

from neural_decoder.train_augmented import compare_summaries
from neural_decoder.train_temporal import apply_channel_dropout, fit_temporal


def test_zero_dropout_is_identity_and_does_not_advance_rng():
    features = torch.ones(3, 17, 176)
    before = torch.get_rng_state().clone()
    assert apply_channel_dropout(features, 0.0) is features
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


def test_dropout_is_perchunk_channel_time_constant_seeded_and_without_rescaling():
    features = torch.full((8, 21, 176), 3.0)
    torch.manual_seed(71)
    actual = apply_channel_dropout(features, 0.1)
    torch.manual_seed(71)
    expected = apply_channel_dropout(features, 0.1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert (actual == 0).any() and (actual == 3).any()
    assert torch.all((actual == 0) | (actual == 3))
    assert torch.equal(actual, actual[:, :1, :].expand_as(actual))
    assert not torch.equal(actual[0], actual[1])
    assert torch.all(features == 3)  # original training tensor is not mutated


@pytest.mark.parametrize("probability", [-0.1, 1, float("nan"), float("inf")])
def test_invalid_dropout_fails(probability):
    with pytest.raises(ValueError):
        apply_channel_dropout(torch.ones(2, 3, 176), probability)


def test_default_training_path_is_bit_identical_to_explicit_zero_dropout():
    rng = np.random.default_rng(9)
    counts = rng.poisson(2, (70, 176))
    targets = rng.normal(size=(70, 7)).astype(np.float32)
    training = [(counts, targets, np.ones(70, dtype=bool))]
    config = {
        "burn_in_bins": 14,
        "chunk_scored_bins": 32,
        "batch_chunks": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
    }
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        first, transforms_a, losses_a = fit_temporal(training, config, 7, 1)
        second, transforms_b, losses_b = fit_temporal(
            training, {**config, "channel_dropout": 0}, 7, 1
        )
    finally:
        torch.set_num_threads(old_threads)
    assert losses_a == losses_b
    for name, weight in first.state_dict().items():
        torch.testing.assert_close(weight, second.state_dict()[name], rtol=0, atol=0)
    for name, value in transforms_a.items():
        np.testing.assert_array_equal(value, transforms_b[name])


def test_selection_uses_all_seeds_and_guards_each_seed_average_day():
    def summary(values):
        return {
            "sealed_days_scored": [],
            "seeds": [
                {
                    "seed": seed,
                    "scores": {
                        "r2_mean": float(np.mean(row)),
                        "per_day": {
                            f"S{index + 2}": {"r2_variance_weighted": value}
                            for index, value in enumerate(row)
                        },
                    },
                }
                for seed, row in enumerate(values)
            ],
        }

    rules = {
        "all_seeds_required": [0, 1, 2],
        "all_days_required": ["S2", "S3", "S4"],
        "minimum_seed_average_mean_r2_gain": 0.005,
        "maximum_seed_average_day_harm": 0.01,
    }
    baseline = summary([[0.1, 0.2, 0.3]] * 3)
    improved = summary([[0.11, 0.21, 0.31]] * 3)
    assert compare_summaries(improved, baseline, rules)["retain_channel_dropout"]
    harmed = summary([[0.08, 0.25, 0.35]] * 3)
    result = compare_summaries(harmed, baseline, rules)
    assert result["criteria"]["mean_gain"]
    assert not result["criteria"]["day_harm"]
    missing = copy.deepcopy(improved)
    missing["seeds"].pop()
    with pytest.raises(ValueError, match="every"):
        compare_summaries(missing, baseline, rules)
