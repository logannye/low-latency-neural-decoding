"""Synthetic adversarial tests for the separate V5 arithmetic reference."""

import copy
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from neural_experiments.v5 import audit, policy


def prefix(budget=30):
    n = 1500 + budget * 50
    angle = np.arange(n) * 2 * np.pi / 100
    y = np.column_stack((np.cos(angle), np.sin(angle)))
    mask = np.ones(n, bool)
    mask[:1500] = False
    segments = [{"start": 0, "end": n}]
    predictions = {c["candidate_id"]: y * 0.75 for c in audit.grids()[0]}
    return y, mask, segments, predictions


def history():
    return {
        "current_session_id": "current",
        "current_post_implant_day": 9,
        "warmup_available": True,
        "prior_candidates": [
            {"candidate_id": f + "_prior10", "prior_available": False, "probes": []}
            for f in audit.FAMILIES
        ],
    }


def test_audit_does_not_import_v5_producer_modules():
    source = inspect.getsource(audit)
    assert "from neural_experiments.v5" not in source
    assert "import neural_experiments.v5" not in source


@pytest.mark.parametrize("budget", audit.BUDGETS)
def test_independent_group_evidence_matches_producer(budget):
    y, mask, segments, predictions = prefix(budget)
    # Make a boundary cut through a group, so that the group and burn must be omitted.
    boundary = 1500 + budget * 25 + 220
    segments = [{"start": 0, "end": boundary}, {"start": boundary, "end": len(y)}]
    reference = audit.stage_reference(y, mask, segments, budget, predictions)
    actual = policy.stage_evidence(
        y, mask, segments, 1500 + budget * 25, 1600 + budget * 25, len(y), predictions
    )
    audit.compare_structure(actual, reference)
    assert all(not (g["start"] < boundary < g["stop"]) for g in reference["groups"])


def test_future_label_or_prediction_arrays_rejected():
    y, mask, segments, predictions = prefix()
    with pytest.raises(ValueError, match="prefix"):
        audit.stage_reference(
            np.pad(y, ((0, 1), (0, 0))),
            np.pad(mask, (0, 1)),
            [{"start": 0, "end": len(y) + 1}],
            30,
            predictions,
        )
    predictions[next(iter(predictions))] = np.pad(y, ((0, 1), (0, 0)))
    with pytest.raises(ValueError, match="prediction"):
        audit.stage_reference(y, mask, segments, 30, predictions)


def test_warmup_labels_cannot_change_stage_validation():
    y, mask, segments, predictions = prefix()
    before = audit.stage_reference(y, mask, segments, 30, predictions)
    y[:1500] = 999
    after = audit.stage_reference(y, mask, segments, 30, predictions)
    audit.compare_structure(before, after)


def test_float32_negative_axis_uses_broker_float64_sector_arithmetic():
    y = np.array([[-1, -1e-8], [-1, 0], [-1, 1e-8]], dtype=np.float32)
    mask = np.ones(3, bool)
    assert audit.sectors(y, mask) == audit.sectors(y.astype(np.float64), mask)
    assert audit.sectors(y, mask) == [2, 0, 0, 0, 0, 0, 0, 1]


@pytest.mark.parametrize("config", audit.grids()[1])
def test_decision_separate_reference_has_same_early_stop(config):
    y, mask, segments, predictions = prefix()
    stage = audit.stage_reference(y, mask, segments, 30, predictions)
    expected, requested = audit.decision_reference(history(), {30: stage}, config)
    calls = []

    def get_stage(budget):
        calls.append(budget)
        assert budget == 30
        return stage

    actual = policy.decide(history(), get_stage, config)
    audit.compare_structure(actual, expected)
    assert calls == requested == [30]


def test_history_acceptance_never_needs_stage_evidence():
    h = history()
    h["prior_candidates"][0].update(
        prior_available=True,
        probes=[
            {"session_id": f"p{i}", "post_implant_day": i, "n": 100, "mse": 0.2} for i in (1, 2)
        ],
    )
    expected, requests = audit.decision_reference(h, {}, audit.grids()[1][0])
    assert requests == []
    assert expected["budget_seconds"] == 0
    assert expected["candidate_id"] == "tx45_prior10"


def test_short_checkpoint_records_requested_and_observed_cost():
    stage = {
        "budget_seconds": 30,
        "unavailable": "requested_checkpoint_not_observed",
        "available_label_seconds": 12.8,
    }
    result, requests = audit.decision_reference(history(), {30: stage}, audit.grids()[1][0])
    assert requests == [30]
    assert result["budget_seconds"] == 30
    assert result["available_label_seconds"] == 12.8
    assert result["decision_time_seconds"] == 42.8
    assert result["action"] == "abstain"


def source_rows():
    return [
        {
            "participant": p,
            "session_id": f"{p}-{i}",
            "policy_id": f"p{k}",
            "label_seconds": 240 if i else 30 * k,
            "metrics": {"r2": None if i else 0.3 - 0.005 * k},
            "daily120_r2": 0.3,
        }
        for p in ("a", "b", "c")
        for i in range(2)
        for k in range(4)
    ]


def test_policy_ranking_excludes_entire_target_and_keeps_unavailable_cost():
    rows = source_rows()
    before = audit.selection_reference(rows, ["a", "b", "c"])
    assert before[0]["ranking"][0]["mean_label_seconds"] == 120
    altered = copy.deepcopy(rows)
    for row in altered:
        if row["participant"] == "a":
            row["metrics"]["r2"] = -100
            row["daily120_r2"] = 20
            row["label_seconds"] = 999
    after = audit.selection_reference(altered, ["a", "b", "c"])
    assert before[0] == after[0]
    assert before[1] != after[1]


def test_duplicate_or_missing_artifact_rejected():
    with pytest.raises(ValueError, match="inventory"):
        audit.unique([{"id": 1}, {"id": 1}], lambda r: r["id"])
    with pytest.raises(ValueError, match="inventory"):
        audit.unique([{"id": 1}], lambda r: r["id"], [1, 2])


def test_fit_mask_excludes_warmup_and_reburns_every_block():
    mask = audit.label_mask(np.ones(4000, bool), [(0, 2000), (2000, 4000)], 3000)
    assert len(mask) == 3000
    assert not mask[:1500].any()
    assert not mask[2000:2062].any()
    assert mask[1500:2000].all() and mask[2062:].all()


def fitted_fixture(tmp_path, candidate_id="tx45_prior10"):
    from neural_experiments.v5 import model

    n = 4000
    candidate = next(c for c in model.candidate_grid() if c["candidate_id"] == candidate_id)
    family = candidate["config"]["family"]
    dimensions = 2 if family == "tx45" else 4
    rng = np.random.default_rng(81)
    raw = rng.normal(size=(n, dimensions))
    z, mean, scale = audit.reference_design(raw, 1500, 2)
    angle = np.arange(n) * 0.04
    labels = {
        "y": np.column_stack((np.cos(angle), np.sin(angle))).astype(np.float32),
        "eval_mask": np.ones(n, bool),
    }
    segments = [{"start": 0, "end": 2700}, {"start": 2700, "end": n}]
    prior = rng.normal(0, 0.01, size=(dimensions + 1, 2))
    weights, receipt = model.fit_stage(
        z,
        labels["y"],
        labels["eval_mask"],
        [(s["start"], s["end"]) for s in segments],
        3000,
        candidate,
        prior,
    )
    path = tmp_path / "model.npz"
    model.Decoder(candidate["config"], mean, scale, weights).save(path)
    item = {
        "participant_id": "p",
        "session_id": "p-9",
        "post_implant_day": 9,
        "n_channels": 2,
        "n_bins": n,
        "segments": segments,
    }
    row = {
        "participant": "p",
        "session_id": "p-9",
        "post_implant_day": 9,
        "family": family,
        "candidate_id": candidate_id,
        "budget_seconds": 30,
        "phase": "full",
        "daily_control": False,
        "fit": receipt,
        "history": [],
        "model_path": path.name,
        "model_sha256": audit.sha256(path),
    }
    verifier = object.__new__(audit.Verifier)
    verifier.root, verifier.residuals, verifier.models_checked, verifier.fits_checked = (
        tmp_path,
        [],
        0,
        0,
    )
    return SimpleNamespace(
        verifier=verifier,
        row=row,
        item=item,
        raw={family: raw},
        prior={family: prior},
        history={family: []},
        labels=labels,
        path=path,
    )


@pytest.mark.parametrize("candidate_id", [c["candidate_id"] for c in audit.grids()[0]])
def test_saved_fit_checked_independently_for_each_candidate(tmp_path, candidate_id):
    f = fitted_fixture(tmp_path, candidate_id)
    f.verifier.fit(f.row, f.item, f.raw, f.prior, f.history, f.labels)
    assert f.verifier.fits_checked == 1
    assert max(f.verifier.residuals) < 1e-10


def test_rehashed_wrong_fit_weights_fail_normal_equations(tmp_path):
    f = fitted_fixture(tmp_path)
    arrays = audit.numeric(f.path)
    arrays["weights"][0, 0] += 0.01
    np.savez(f.path, **arrays)
    f.row["model_sha256"] = audit.sha256(f.path)
    with pytest.raises(ValueError, match="optimum"):
        f.verifier.fit(f.row, f.item, f.raw, f.prior, f.history, f.labels)


def test_future_labels_do_not_change_fit_verification_but_prefix_does(tmp_path):
    f = fitted_fixture(tmp_path)
    f.labels["y"][3000:] *= -1
    f.verifier.fit(f.row, f.item, f.raw, f.prior, f.history, f.labels)
    f.labels["y"][1500:3000] *= -1
    with pytest.raises(ValueError, match="optimum"):
        f.verifier.fit(f.row, f.item, f.raw, f.prior, f.history, f.labels)


def test_tampered_calibration_receipt_rejected(tmp_path):
    f = fitted_fixture(tmp_path)
    f.row["fit"]["label_start_bin"] = 0
    with pytest.raises(ValueError, match="boundaries"):
        f.verifier.fit(f.row, f.item, f.raw, f.prior, f.history, f.labels)


def aggregate_rows():
    rows = []
    for person in ["T2", "T5", "T6", "T7", "T8", "T9", "T10", "T11"]:
        for session in range(3):
            for method in audit.METHODS:
                unavailable = person == "T6" and session == 0
                r2 = None if unavailable else (0.2 if method != "zero" else -0.02)
                rows.append(
                    {
                        "domain": "new_sessions",
                        "participant": person,
                        "session_id": f"{person}-{session}",
                        "method": method,
                        "action": "abstain" if method == "zero" else "accept",
                        "label_seconds": 240 if unavailable else 30,
                        "available_label_seconds": 220 if unavailable else 30,
                        "eval_mask_sha256": f"mask-{person}-{session}",
                        "metrics": {"r2": r2, "angular_error_degrees": None if unavailable else 60},
                    }
                )
    return rows


def test_independent_summary_keeps_unavailable_cost_and_matches_producer():
    from neural_experiments.v5 import metrics

    rows = aggregate_rows()
    expected = audit.summary_reference(rows)
    audit.compare_structure(metrics.summarize(rows), expected)
    values = expected["domains"]["new_sessions"]["methods"]["adaptive"]
    assert values["requested_sessions"] == 24 and values["scored_sessions"] == 23
    assert values["mean_requested_label_seconds"] == 38.75
    assert expected["gates"]["adaptive_useful"]


def test_mismatched_pairing_mask_rejected():
    rows = aggregate_rows()
    rows[1]["eval_mask_sha256"] = "different"
    with pytest.raises(ValueError, match="Paired"):
        audit.summary_reference(rows)


def test_missing_person_prevents_usefulness_even_with_good_remaining_scores():
    rows = [r for r in aggregate_rows() if r["participant"] != "T11"]
    assert not audit.summary_reference(rows)["gates"]["adaptive_useful"]


def test_synthetic_continuation_runner_is_audited_with_separate_prefix_labels(
    tmp_path, monkeypatch
):
    from neural_experiments.v5 import study

    n, known_n = 16000, 13500
    rng = np.random.default_rng(208)
    angle = np.arange(known_n) * 0.03
    labels = {
        "y": np.column_stack((np.cos(angle), np.sin(angle))).astype(np.float32),
        "eval_mask": np.ones(known_n, bool),
    }
    known_path = tmp_path / "known.npz"
    np.savez(known_path, **labels)
    item = {
        "participant_id": "T3",
        "session_id": "T3-synthetic",
        "post_implant_day": 9,
        "n_bins": n,
        "n_channels": 2,
        "segments": [{"start": 0, "end": n}],
        "known_prefix_target_path": str(known_path),
        "known_prefix_n_bins": known_n,
        "known_prefix_target_sha256": audit.sha256(known_path),
        "target_path": "scorer-not-used-by-broker.npz",
        "target_sha256": "unused",
    }
    features = {
        "x_tx3": rng.poisson(2, (n, 2)).astype(float),
        "x_tx45": rng.poisson(1, (n, 2)).astype(float),
        "x_sbp": rng.lognormal(1, 0.3, (n, 2)),
    }
    case = {"item": item, "features": features, "segments": [(0, n)]}
    scorer = {"y": np.zeros((n, 2), np.float32), "eval_mask": np.zeros(n, bool)}
    scorer["y"][14000:, 0] = 1
    scorer["eval_mask"][14000:] = True
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "relative", lambda p: str(p.relative_to(tmp_path)))
    monkeypatch.setattr(study, "load_case", lambda *args, **kwargs: case)
    output = study.init_output(tmp_path / "run")
    replay = study.CaseReplay(
        tmp_path / "index.json",
        item,
        [],
        {item["session_id"]: (0, 1)},
        {f: {} for f in audit.FAMILIES},
        output,
        "t3_blocks",
    )
    h = {**history(), "current_session_id": item["session_id"]}
    decision = policy.decide(h, replay.get_stage, policy.policy_grid()[0])
    fitted = replay.decoder_for(decision)
    metadata = {
        **study.row_metadata(
            item,
            "t3_blocks",
            "adaptive",
            decision["budget_seconds"],
            decision["action"],
            decision["candidate_id"],
        ),
        "policy_id": "p0",
        "decision": decision,
        "label_requests": replay.broker.requests,
    }
    row = study.save_score(output, "adaptive", fitted, {**case, **scorer}, "t3_blocks", metadata)
    controls = []
    for method, fitted in replay.controls().items():
        budget = {"zero": 0, "history0": 0, "history30": 30, "daily120": 120, "daily240": 240}[
            method
        ]
        controls.append(
            study.save_score(
                output,
                method,
                fitted,
                {**case, **scorer},
                "t3_blocks",
                study.row_metadata(
                    item, "t3_blocks", method, budget, "abstain" if method == "zero" else "accept"
                ),
            )
        )
    verifier = object.__new__(audit.Verifier)
    verifier.root, verifier.items, verifier.layouts = tmp_path, [], {item["session_id"]: (0, 1)}
    verifier.errors, verifier.residuals = [], []
    verifier.models_checked = verifier.traces_checked = verifier.fits_checked = 0
    raw = {
        c["config"]["family"]: audit.reference_representation(
            features, item["segments"], c["config"]
        )
        for c in audit.grids()[0][::3]
    }
    evidence = {"history_evidence": h, "stages": list(replay.stages.values()), "controls": controls}
    audit.check_case(
        verifier,
        item,
        raw,
        replay.priors,
        replay.history,
        scorer,
        evidence,
        replay.saved_fits,
        [row],
        {},
        source=False,
        calibration_labels=labels,
        domain="t3_blocks",
    )
    assert verifier.traces_checked == 6
    assert row["metrics"]["n"] == 2000
    assert max(verifier.errors) < 1e-10
