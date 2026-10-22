"""Runner/broker integration with synthetic labels; no real assessment inputs."""

import json
from pathlib import Path

import numpy as np
import pytest

from neural_experiments.v5 import model, study


def target_file(path, n=5650):
    angle = np.arange(n) * 0.04
    y = np.column_stack((np.cos(angle), np.sin(angle))).astype(np.float32)
    mask = np.ones(n, dtype=bool)
    mask[::17] = False
    np.savez(path, y=y, eval_mask=mask)
    return y, mask


def synthetic_item(tmp_path, n=5650):
    path = tmp_path / "targets.npz"
    y, mask = target_file(path, n)
    item = {
        "session_id": "synthetic_day_10",
        "participant_id": "synthetic_person",
        "post_implant_day": 10,
        "n_bins": n,
        "n_channels": 3,
        "target_path": path.name,
        "target_sha256": study.digest(path),
    }
    return item, y, mask


def replay_fixture(tmp_path, monkeypatch, n=5650):
    item, y, mask = synthetic_item(tmp_path, n)
    rng = np.random.default_rng(619)
    features = {
        "x_tx3": rng.poisson(1, (n, 3)).astype(float),
        "x_tx45": rng.poisson(0.2, (n, 3)).astype(float),
        "x_sbp": rng.lognormal(1, 0.4, (n, 3)),
    }
    segments = [(0, 2800), (2800, n)]
    item["segments"] = [{"start": a, "end": b} for a, b in segments]
    case = {"item": item, "features": features, "segments": segments}

    def load_case(_index, _item, labels=False):
        return {**case, **({"y": y.copy(), "eval_mask": mask.copy()} if labels else {})}

    monkeypatch.setattr(study, "load_case", load_case)
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "relative", lambda p: str(Path(p).resolve().relative_to(tmp_path)))
    output = study.init_output(tmp_path / "run")
    replay = study.CaseReplay(
        tmp_path / "index.json",
        item,
        [],
        {item["session_id"]: (0, 1, 2)},
        {"tx45": {}, "tx45_sbp": {}},
        output,
        "new_sessions",
    )
    return replay, y, mask


def test_broker_releases_only_prospective_prefix_and_requires_increasing_requests(tmp_path):
    item, y, mask = synthetic_item(tmp_path)
    broker = study.PrefixLabels(tmp_path / "index.json", item)
    observed, eligible = broker.request(30)
    assert observed.shape == (3000, 2) and eligible.shape == (3000,)
    assert not observed[:1500].any() and not eligible[:1500].any()
    np.testing.assert_array_equal(observed[1500:], y[1500:3000])
    np.testing.assert_array_equal(eligible[1500:], mask[1500:3000])
    with pytest.raises(ValueError, match="increase"):
        broker.request(30)
    observed, _ = broker.request(240)
    assert len(observed) == len(y)
    assert broker.requests == [30, 240]


def test_continuation_broker_never_opens_scorer_target_file(tmp_path):
    known = tmp_path / "known.npz"
    y, mask = target_file(known, 6000)
    item = {
        "target_path": "do-not-open-scorer-targets.npz",
        "target_sha256": "not-a-valid-scorer-hash",
        "n_bins": 20000,
        "known_prefix_target_path": str(known),
        "known_prefix_target_sha256": study.digest(known),
        "known_prefix_n_bins": 6000,
    }
    actual, use = study.PrefixLabels(tmp_path / "index.json", item).request(240)
    assert len(actual) == 6000
    np.testing.assert_array_equal(actual[1500:], y[1500:])
    np.testing.assert_array_equal(use[1500:], mask[1500:])


def test_stage_cache_saves_all_half_fits_but_full_refits_only_selected_candidate(
    tmp_path, monkeypatch
):
    replay, _, _ = replay_fixture(tmp_path, monkeypatch)
    stage = replay.get_stage(30)
    assert stage["train_stop"] == 2250 and stage["validation_start"] == 2350
    assert stage["stop"] == 3000
    assert replay.broker.requests == [30]
    assert len(replay.saved_fits) == 6
    assert all(r["phase"] == "validation" for r in replay.saved_fits)
    assert all(r["fit"]["label_stop_bin"] == 2250 for r in replay.saved_fits)
    assert all(r["fit"]["normalization_bins"] == 1500 for r in replay.saved_fits)
    assert replay.get_stage(30) is stage
    chosen, row = replay.full_fit("tx45_sbp_scratch1", 30)
    assert len(replay.saved_fits) == 7
    assert row["phase"] == "full" and row["fit"]["label_stop_bin"] == 3000
    assert replay.full_fit("tx45_sbp_scratch1", 30)[0] is chosen
    assert len(replay.saved_fits) == 7
    # Final refit uses a separate broker for the same permitted prefix only.
    assert replay.broker.requests == [30]
    loaded = model.Decoder.load(tmp_path / row["model_path"])
    np.testing.assert_array_equal(loaded.weights, chosen.weights)


def test_short_stage_abstention_records_actual_cost_and_no_validation_fit(tmp_path, monkeypatch):
    replay, _, _ = replay_fixture(tmp_path, monkeypatch)
    evidence = replay.get_stage(240)
    assert evidence["unavailable"] == "requested_checkpoint_not_observed"
    assert evidence["available_label_seconds"] == 83
    assert replay.saved_fits == []
    # Policy behavior is tested separately; the runner evidence preserves actual duration.
    row = study.row_metadata(replay.item, "new_sessions", "adaptive", 240, "abstain")
    assert row["label_seconds"] == 240 and row["available_label_seconds"] == 83


def test_daily_controls_have_later_normalizer_but_same_prospective_label_start(
    tmp_path, monkeypatch
):
    replay, _, _ = replay_fixture(tmp_path, monkeypatch)
    controls = replay.controls()
    assert set(controls) == {"zero", "history0", "history30", "daily120", "daily240"}
    adaptive = replay.designs["tx45_sbp"]
    raw = replay.raw["tx45_sbp"]
    for name in ("daily120", "daily240"):
        fitted = controls[name]
        np.testing.assert_array_equal(fitted.mean, raw.mean(0))
        assert not np.array_equal(fitted.mean, adaptive[1])
    assert all(r["fit"]["label_start_bin"] == 1500 for r in replay.saved_fits)
    for row in replay.saved_fits:
        if row["daily_control"]:
            assert row["fit"]["normalization_bins"] == 5650
            assert row["fit"]["available_label_seconds"] == 83
            assert row["fit"]["complete_requested_interval"] is False
    # Counterfactual control labels do not advance the policy's broker.
    assert replay.broker.requests == []
    metadata = study.row_metadata(replay.item, "new_sessions", "daily240", 240, "accept")
    row = study.save_score(
        replay.output,
        "short-daily240-score",
        controls["daily240"],
        study.load_case(replay.index_path, replay.item, True),
        "new_sessions",
        metadata,
    )
    assert row["metrics"]["r2"] is None and row["metrics"]["n"] == 0
    assert row["metrics"]["summed_mse"] is None
    assert row["label_seconds"] == 240 and row["available_label_seconds"] == 83
    assert Path(tmp_path / row["trace_path"]).is_file()


def test_history_probe_rows_keep_chronology_and_convert_summed_mse_once():
    def item(day, person="person", sid=None):
        return {"session_id": sid or f"s{day}", "participant_id": person, "post_implant_day": day}

    items = [item(d) for d in (1, 2, 3, 4, 5, 20)] + [item(9, "other")]
    query = item(10)
    layouts = {s["session_id"]: (0, 1) for s in [*items, query]}
    layouts["s5"] = (1, 2)  # Different layout is excluded before selecting latest three.
    probes = {}
    for old in items:
        for family in ("tx45", "tx45_sbp"):
            probes[old["session_id"], family] = {
                "metrics": {"n": 500, "summed_mse": 0.6},
                "history": [
                    {
                        "disposition": "used"
                        if old["post_implant_day"] > 2
                        else "insufficient_labels"
                    }
                ],
            }
    evidence = study.historical_evidence(query, items, layouts, probes)
    assert all([r["session_id"] for r in f["cases"]] == ["s3", "s4"] for f in evidence)

    class Replay:
        history = {f: [{"disposition": "used"}] for f in ("tx45", "tx45_sbp")}

    query["n_bins"] = 30000
    quality = study.quality_evidence(query, items, layouts, probes, Replay())
    assert all(
        p["mse"] == 0.3 and p["n"] == 500 for c in quality["prior_candidates"] for p in c["probes"]
    )


def test_source_policy_choice_excludes_whole_target_and_ties_keep_grid_order():
    configurations = [{"policy_id": "p0"}, {"policy_id": "p1"}]
    rows = []
    for person, scores in (("A", (0.9, -0.9)), ("B", (0.2, 0.4)), ("C", (0.2, 0.4))):
        for k, value in enumerate(scores):
            rows.append(
                {
                    "participant": person,
                    "policy_id": f"p{k}",
                    "metrics": {"r2": value},
                    "label_seconds": 30,
                    "daily120_r2": 0.4,
                }
            )
    first = study.choose_policy(rows, configurations, ["A", "B", "C"])[0]
    assert first["policy_id"] == "p1"
    changed = json.loads(json.dumps(rows))
    for r in changed:
        if r["participant"] == "A":
            r["metrics"]["r2"] *= -100
    assert study.choose_policy(changed, configurations, ["A", "B", "C"])[0] == first
