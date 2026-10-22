"""Small synthetic checks for the standalone transfer, without public arrays."""

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import r2_score

from neural_decoder.models import fit_linear

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "transfer_m2", ROOT / "neural_experiments/transfer_m2.py"
)
TRANSFER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRANSFER)
CONFIG = ROOT / "neural_experiments/configs/m2-minival-transfer-v1.json"


def test_locked_inventory_and_roles(tmp_path):
    config = TRANSFER.checked_config(CONFIG)
    assert len(config["assets"]) == 7
    assert sum(a["content_size"] for a in config["assets"]) == 124175536
    config["assets"][2]["role"] = "training"
    altered = tmp_path / "config.json"
    altered.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="role"):
        TRANSFER.checked_config(altered)


def test_bin_endpoints_overflow_and_cadence():
    times = np.array([0.0, 0.02, 0.04])
    spikes = [np.array([-0.01, 0.0, 0.019, 0.02, 0.04, 0.06, 0.061])]
    np.testing.assert_array_equal(TRANSFER.independent_bins(spikes, times, 0.02)[:, 0], [2, 1, 2])
    with pytest.raises(ValueError, match="overflow"):
        TRANSFER.independent_bins([np.zeros(256)], times, 0.02)
    with pytest.raises(ValueError, match="regularly"):
        TRANSFER.independent_bins(spikes, np.array([0.0, 0.02, 0.08]), 0.02)


def test_stream_transfer_causality_and_metadata_boundary():
    rng = np.random.default_rng(3)
    counts = rng.poisson(0.7, (145, 96))
    targets = rng.normal(size=(145, 2))
    mask = rng.random(145) > 0.2
    model = fit_linear([(counts, targets, mask)], history=1, regularization=0.1)
    prediction = TRANSFER.predict_counts(model, counts)
    checks = TRANSFER.verify_stream(model, counts, prediction, 1e-10)
    assert checks["max_batch_stream_abs_error"] < 1e-10
    assert all(value for key, value in checks.items() if key != "max_batch_stream_abs_error")
    with pytest.raises(TypeError):
        TRANSFER.predict_counts(model, counts, eval_mask=~mask)


def test_calendar_grouping_unequal_files_and_signed_r2():
    rng = np.random.default_rng(6)
    records, predictions = [], {}
    for index, (day, length) in enumerate(zip(TRANSFER.EVALUATION_DAYS, [13, 11, 7], strict=True)):
        for run in range(2 if index < 2 else 1):
            size = length + run
            file_id = f"{day}_Run{run + 1}"
            targets = rng.normal(size=(size, 2))
            mask = np.ones(size, dtype=bool)
            mask[2] = False
            predictions[file_id] = targets + (index + 1)
            records.append(
                {
                    "file_id": file_id,
                    "day": day,
                    "targets": targets,
                    "counts": np.zeros((size, 96)),
                    "eval_mask": mask,
                }
            )

    def independent_helper(preds, targets, mask, lengths):
        # Equal-day aggregation from independently counted scored row boundaries.
        values, full_offset, scored_offset = [], 0, 0
        for day in sorted(lengths):
            full_length = sum(lengths[day])
            day_mask = mask[full_offset : full_offset + full_length]
            scored_length = int(day_mask.sum())
            values.append(
                r2_score(
                    targets[scored_offset : scored_offset + scored_length],
                    preds[full_offset : full_offset + full_length][day_mask],
                    multioutput="variance_weighted",
                )
            )
            full_offset += full_length
            scored_offset += scored_length
        return {"R2 Mean": np.mean(values), "R2 Std.": np.std(values)}

    scores = TRANSFER.score_calendar_days(records[::-1], predictions, independent_helper)
    assert scores["num_days"] == 3 and scores["num_files"] == 5
    assert scores["r2_mean"] < 0
    assert scores["per_day"]["2020-10-20"]["num_scored_bins"] == 25


def test_nwb_schema_and_complete_bin_export(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    from pynwb import NWBHDF5IO, NWBFile, TimeSeries
    from pynwb.behavior import BehavioralTimeSeries

    times = np.arange(50) * 0.02
    nwb = NWBFile("synthetic", "synthetic-m2", datetime(2020, 10, 19, tzinfo=UTC))
    for channel in range(96):
        nwb.add_unit(id=channel, spike_times=np.array([0.01, 0.03, 0.07]))
    container = BehavioralTimeSeries(name="finger_vel")
    target = np.column_stack([np.sin(times), np.cos(times)])
    for index, name in enumerate(["index", "mrs"]):
        container.create_timeseries(name=name, data=target[:, index], timestamps=times, unit="AU")
    nwb.add_acquisition(container)
    mask = np.ones(50, dtype=bool)
    mask[20:25] = False
    nwb.add_acquisition(TimeSeries(name="eval_mask", data=mask, timestamps=times, unit="bool"))
    nwb.add_trial(start_time=0.0, stop_time=0.4)
    nwb.add_trial(start_time=0.5, stop_time=0.98)
    path = tmp_path / "synthetic.nwb"
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    expected_counts = np.zeros((50, 96), dtype=np.uint8)
    expected_counts[[0, 1, 3]] = 1
    expected_trials = np.zeros(50, dtype=bool)
    expected_trials[[0, 25]] = True
    config = TRANSFER.checked_config(CONFIG)

    def reference_loader(path, task):
        assert task == "m2"
        return expected_counts, target, expected_trials, mask

    record = TRANSFER.prepare(path, config["assets"][0], config, reference_loader)
    assert record["counts"].shape == (50, 96)
    assert record["metadata"]["num_scored_bins"] == 45
    assert record["metadata"]["num_trials"] == 2
    np.testing.assert_array_equal(record["release_times"], times + 0.02)
