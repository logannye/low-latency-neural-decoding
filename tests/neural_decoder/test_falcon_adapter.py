"""Synthetic pinned-H1 call semantics, without FALCON installation or submission."""

from pathlib import Path

import numpy as np
import pytest
import torch

from neural_decoder.falcon_adapter import H1DecoderAdapter
from neural_decoder.features import fir_weights
from neural_decoder.models import LinearDecoder, LinearModel
from neural_decoder.temporal import CausalTCN, TCNStream, freeze_tcn


@pytest.fixture(params=["ridge", "tcn"])
def decoder_factory(request):
    if request.param == "ridge":
        rng = np.random.default_rng(7)
        model = LinearModel(
            mean=np.linspace(0, 1, 176),
            scale=np.linspace(0.5, 2, 176),
            weights=fir_weights(),
            coefficient=rng.normal(0, 0.1, (352, 7)),
            intercept=np.linspace(-1, 1, 7),
            history=2,
        ).validate()
        return lambda: LinearDecoder(model)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        frozen = freeze_tcn(CausalTCN())
    return lambda: TCNStream(frozen)


def test_movement_like_replay_consumes_unscored_bins_and_preserves_trial_history(decoder_factory):
    counts = np.random.default_rng(14).poisson(3, (63, 176)).astype(np.uint32)
    score_mask = np.ones(63, dtype=bool)
    score_mask[:4] = False
    score_mask[21:35] = False
    trial_boundary = np.zeros(63, dtype=bool)
    trial_boundary[[18, 44]] = True
    adapter = H1DecoderAdapter(decoder_factory())
    adapter.reset([Path("nonexistent-file-for-synthetic-test.nwb")])
    # The pinned H1 movement path calls predict unconditionally and never passes mask/trials.
    observed = np.concatenate([adapter.predict(row[None]) for row in counts])
    expected = decoder_factory().predict_chunk(counts)
    np.testing.assert_array_equal(observed, expected)
    assert observed.shape == (63, 7)

    # Explicitly demonstrate that skipping the unscored observations would change scored values.
    skipped = decoder_factory().predict_chunk(counts[score_mask])
    assert not np.allclose(skipped, expected[score_mask])
    compatibility = H1DecoderAdapter(decoder_factory())
    compatibility.reset(["another-tag"])
    with_hooks = []
    for row, done in zip(counts, trial_boundary, strict=True):
        compatibility.on_done(np.asarray([done]))
        with_hooks.append(compatibility.predict(row[None]))
    np.testing.assert_array_equal(np.concatenate(with_hooks), expected)


def test_file_reset_and_opaque_tags_never_select_models(decoder_factory):
    counts = np.random.default_rng(9).poisson(4, (28, 176))
    adapter = H1DecoderAdapter(decoder_factory())
    adapter.reset(["S0_set_1"])
    first = np.concatenate([adapter.predict(row[None]) for row in counts])
    adapter.reset([Path("sealed-labels-are-not-opened/S12_set_2.nwb")])
    second = np.concatenate([adapter.predict(row[None]) for row in counts])
    np.testing.assert_array_equal(first, second)
    adapter.reset()
    np.testing.assert_array_equal(adapter.predict(counts[0][None]), first[:1])


def test_observe_advances_exactly_one_bin_and_returns_none(decoder_factory):
    counts = np.random.default_rng(17).poisson(3, (26, 176))
    adapter = H1DecoderAdapter(decoder_factory())
    adapter.reset(["file"])
    for row in counts[:13]:
        assert adapter.observe(row[None]) is None
    actual = np.concatenate([adapter.predict(row[None]) for row in counts[13:]])
    expected = decoder_factory().predict_chunk(counts)[13:]
    np.testing.assert_array_equal(actual, expected)


def test_batch_and_input_rejections_preserve_valid_stream_state(decoder_factory):
    adapter = H1DecoderAdapter(decoder_factory())
    reference = H1DecoderAdapter(decoder_factory())
    first = np.ones((1, 176))
    np.testing.assert_array_equal(adapter.predict(first), reference.predict(first))
    for batch in (0, 2, -1, True):
        with pytest.raises(ValueError, match="batch size one"):
            H1DecoderAdapter(decoder_factory(), batch_size=batch)
        with pytest.raises(ValueError, match="batch size one"):
            adapter.set_batch_size(batch)
    assert adapter.batch_size == 1
    for bad in (
        np.zeros((2, 176)),
        np.zeros(176),
        np.zeros((1, 175)),
        np.full((1, 176), np.nan),
        np.full((1, 176), -1),
    ):
        with pytest.raises(ValueError):
            adapter.predict(bad)
    for bad in ([], ["file-one", "file-two"], "not-a-tag-list"):
        with pytest.raises(ValueError):
            adapter.reset(bad)
    for bad in (np.zeros(2, dtype=bool), np.array([2]), np.array([np.nan])):
        with pytest.raises(ValueError):
            adapter.on_done(bad)
    np.testing.assert_array_equal(adapter.predict(first), reference.predict(first))


def test_wrong_model_dimensions_and_outputs_are_rejected():
    small = LinearModel(
        np.zeros(4), np.ones(4), fir_weights(), np.zeros((4, 7)), np.zeros(7), history=1
    )
    with pytest.raises(ValueError, match="176 input"):
        H1DecoderAdapter(LinearDecoder(small))
    wrong_outputs = LinearModel(
        np.zeros(176), np.ones(176), fir_weights(), np.zeros((176, 6)), np.zeros(6), history=1
    )
    with pytest.raises(ValueError, match="seven outputs"):
        H1DecoderAdapter(LinearDecoder(wrong_outputs))
    with pytest.raises(TypeError, match="LinearDecoder or TCNStream"):
        H1DecoderAdapter(object())


@pytest.mark.parametrize("bad", [np.zeros(6), np.full(7, np.nan)])
def test_invalid_decoder_output_fails_explicitly(decoder_factory, monkeypatch, bad):
    decoder = decoder_factory()
    adapter = H1DecoderAdapter(decoder)
    monkeypatch.setattr(decoder, "step", lambda _: bad)
    with pytest.raises((ValueError, FloatingPointError)):
        adapter.predict(np.zeros((1, 176)))
