"""Independent references reject leakage, chronology changes and artifact tampering."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from neural_experiments.v4 import audit, model


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(19)
    features = {
        name: rng.uniform(0, 5, (400, 4)).astype(np.float32)
        for name in ("x_tx3", "x_tx45", "x_sbp")
    }
    labels = rng.normal(size=(400, 2))
    labels /= np.linalg.norm(labels, axis=1)[:, None]
    mask = np.ones(400, dtype=bool)
    mask[::7] = False
    return features, labels, mask, [(0, 240), (240, 400)]


def config(family="tx45", baseline=None, gamma=1.0):
    return {
        "family": family,
        "tau_seconds": 0.2 if family.endswith("sbp") else 0.4,
        "baseline_tau_seconds": baseline,
        "adapt_gamma": gamma,
    }


@pytest.mark.parametrize("family", ["tx3", "tx45", "tx3_sbp", "tx45_sbp"])
@pytest.mark.parametrize("baseline", [None, 5.0])
def test_direct_recurrence_matches_production_and_preserves_causality(synthetic, family, baseline):
    features, _, _, segments = synthetic
    cfg = config(family, baseline)
    expected = audit.reference_representation(features, segments, cfg)
    np.testing.assert_allclose(model.representation(features, segments, cfg), expected, atol=1e-12)
    altered = {k: v.copy() for k, v in features.items()}
    for v in altered.values():
        v[180:] *= 20
    changed = audit.reference_representation(altered, segments, cfg)
    np.testing.assert_array_equal(changed[:180], expected[:180])
    if baseline is not None:
        np.testing.assert_array_equal(expected[[0, 240]], 0)


def test_mask_uses_observed_global_stride_after_block_burn():
    mask = np.ones(180, dtype=bool)
    mask[70:82] = False
    actual = audit.reference_mask(mask, [(0, 73), (73, 180)], stride=5)
    assert np.flatnonzero(actual).tolist() == [65, 135, 140, 145, 150, 155, 160, 165, 170, 175]
    assert not audit.reference_mask(mask, [(0, 73), (73, 180)], 62).any()


def test_equal_session_history_does_not_pool_unequal_counts():
    a = {"gram": np.eye(3), "rhs": np.ones((3, 2)), "n": 50}
    b = {"gram": 3 * np.eye(3), "rhs": 5 * np.ones((3, 2)), "n": 5000}
    b["gram"][-1, -1] = 1
    got = audit.reference_history([a, b], 3)
    np.testing.assert_allclose(got, [[1, 1], [1, 1], [3, 3]])
    np.testing.assert_allclose(model.solve_history([a, b], 3), got)
    with pytest.raises(ValueError, match="Insufficient history"):
        audit.reference_history([{**a, "n": 49}], 3)


@pytest.mark.parametrize("gamma", [1.0, 10.0])
def test_adaptation_optimum_and_low_label_fallback(synthetic, gamma):
    features, y, mask, segments = synthetic
    raw = audit.reference_representation(features, segments, config())
    z, _, _ = audit.reference_design(raw, 150, 4)
    training = audit.reference_mask(mask, segments, 150)
    stats = audit.reference_statistics(z[:150], y[:150], training)
    prior = np.full((5, 2), 0.03)
    weights = model.adapt(stats, prior, gamma)
    assert (
        audit.validate_weights(weights, stats, prior, gamma)["normal_equation_relative_residual"]
        < 1e-12
    )
    wrong = weights.copy()
    wrong[-1, 0] += 0.02
    with pytest.raises(ValueError, match="optimum"):
        audit.validate_weights(wrong, stats, prior, gamma)
    for count in (0, 49):
        low = {**stats, "n": count}
        assert audit.validate_weights(prior.copy(), low, prior, gamma)["fallback"]
        with pytest.raises(ValueError, match="fallback"):
            audit.validate_weights(np.zeros_like(prior), low, prior, gamma)


def test_post_prefix_labels_and_mask_cannot_change_statistics(synthetic):
    features, y, mask, segments = synthetic
    raw = audit.reference_representation(features, segments, config())
    z, _, _ = audit.reference_design(raw, 150, 4)
    before = audit.reference_statistics(z[:150], y[:150], audit.reference_mask(mask, segments, 150))
    y[150:] = np.nan
    mask[150:] = False
    after = audit.reference_statistics(z[:150], y[:150], audit.reference_mask(mask, segments, 150))
    np.testing.assert_array_equal(before["rhs"], after["rhs"])


@pytest.mark.parametrize("bias", [0, 1])
def test_daily_fit_export_and_full_replay(synthetic, tmp_path, bias):
    features, y, mask, segments = synthetic
    cfg = config("tx3_sbp", 5.0)
    raw = audit.reference_representation(features, segments, cfg)
    z, mean, scale = audit.reference_design(raw, 150, 4)
    stats = audit.reference_statistics(z[:150], y[:150], audit.reference_mask(mask, segments, 150))
    weights = model.solve_daily(stats, 10.0, bias)
    audit.validate_weights(weights, stats, daily_penalty=10.0, free_bias=bool(bias))
    fitted = model.Decoder(cfg, mean, scale, weights)
    path = tmp_path / "model.npz"
    fitted.save(path)
    arrays = audit.numeric(path)
    audit.validate_normalization(arrays, raw, 150, 4, cfg)
    np.testing.assert_allclose(
        audit.replay(arrays, raw), fitted.predict(features, segments), atol=1e-12
    )
    with pytest.raises(ValueError, match="normalization"):
        audit.validate_normalization(arrays, raw, 151, 4, cfg)
    arrays["mean"][0] += 0.1
    with pytest.raises(ValueError, match="normalization"):
        audit.validate_normalization(arrays, raw, 150, 4, cfg)


def test_history_selection_cannot_replace_invalid_or_include_query():
    items = [
        {"participant_id": "T3", "session_id": str(i), "post_implant_day": i} for i in range(6)
    ]
    layouts = {str(i): (1, 2) for i in range(6)}
    layouts["4"] = (4, 5)
    stats = {str(i): {"gram": np.eye(2), "rhs": np.ones((2, 2)), "n": 100} for i in range(6)}
    stats["2"]["n"] = 0
    _, receipt = audit.checked_prior(items[-1], items, layouts, stats, 2)
    assert [r["session_id"] for r in receipt] == ["1", "2", "3"]
    assert receipt[1]["disposition"] == "insufficient_labels"
    assert all(r["post_implant_day"] < 5 for r in receipt)


def test_selection_excludes_whole_person_and_targeted_query_future():
    grid = [{"candidate_id": "a"}, {"candidate_id": "b"}]
    rows = [
        {
            "participant": person,
            "session_id": sid,
            "post_implant_day": day,
            "candidate_id": candidate,
            "kind": kind,
            "metrics": {"r2": value},
        }
        for kind in ("global", "targeted")
        for person, sid, day, values in [
            ("T2", "other", 0, [0.0, 1.0]),
            ("T3", "early", 1, [2.0, 0.0]),
            ("T3", "later", 2, [-100.0, 100.0]),
        ]
        for candidate, value in zip(["a", "b"], values, strict=True)
    ]
    items = [
        {"participant_id": "T3", "session_id": sid, "post_implant_day": day}
        for sid, day in [("early", 1), ("later", 2)]
    ]
    choices, targeted = audit.reconstruct_choices(rows, grid, grid, items, ["T2", "T3"])
    assert choices[1]["candidate_id"] == "b"
    assert targeted[0]["selection_scope"] == "other_people"
    assert targeted[0]["candidate_id"] == "b"
    assert targeted[1]["source_session_ids"] == ["early"]
    assert targeted[1]["candidate_id"] == "a"


def test_locked_grids_have_exact_order_and_count():
    plan = json.loads((audit.ROOT / "neural_experiments/v4/plan.json").read_text())
    global_grid, targeted = audit.candidate_grids(plan)
    assert len(global_grid) == len(targeted) == 16
    assert global_grid[0]["config"] == config("tx45", None, 1.0)
    assert global_grid[-1]["config"] == config("tx3_sbp", 5.0, 10.0)
    assert targeted[0]["bias"] == 0
    assert targeted[-1]["bias"] == 1


@pytest.mark.parametrize(
    "key,value",
    [("format_version", np.asarray(4.0)), ("config", np.asarray(json.dumps(config()).encode()))],
)
def test_export_types_match_deployable_model_contract(synthetic, key, value):
    features, _, _, segments = synthetic
    raw = audit.reference_representation(features, segments, config())
    _, mean, scale = audit.reference_design(raw, 150, 4)
    arrays = {
        "format_version": np.asarray(4, np.int64),
        "config": np.asarray(json.dumps(config())),
        "mean": mean,
        "scale": scale,
        "weights": np.zeros((5, 2)),
    }
    arrays[key] = value
    with pytest.raises(ValueError, match="format|Unicode"):
        audit.validate_normalization(arrays, raw, 150, 4, config())


@pytest.fixture
def summary_rows():
    plan = audit.read(audit.ROOT / "neural_experiments/v4/plan.json")
    rows = []
    for domain in ("new_sessions", "t3_blocks"):
        for i in range(8):
            person = plan["new_session_participants"][i] if domain == "new_sessions" else "T3"
            for method in plan["assessment"][domain]["methods"]:
                r2 = {
                    "zero": 0.0,
                    "prefix_mean30": 0.01,
                    "v3_daily120": 0.22,
                    "targeted_daily120": 0.3,
                }.get(method, 0.2)
                rows.append(
                    {
                        "domain": domain,
                        "session_id": f"{domain}-{i}",
                        "participant": person,
                        "method": method,
                        "eval_mask_sha256": "same-mask-per-session",
                        "metrics": {"r2": r2, "angular_error_degrees": 55.0},
                    }
                )
    return rows, plan


def test_summary_uses_eight_people_and_eight_t3_sessions_separately(summary_rows):
    rows, plan = summary_rows
    result = audit.reconstruct_summary(rows, plan)
    assert all(result["gates"].values())
    for domain in result["domains"].values():
        assert domain["methods"]["selected_prior30"]["n_units"] == 8
        assert domain["methods"]["selected_prior30"]["mean_r2"] == pytest.approx(0.2)
    pair = next(
        c
        for c in result["domains"]["new_sessions"]["contrasts"]
        if c["arm"] == "selected_prior30" and c["control"] == "v3_daily120"
    )
    np.testing.assert_allclose(pair["gain_bootstrap_95"], [-0.02, -0.02])


def test_missing_person_never_passes_reduced_label_gate(summary_rows):
    rows, plan = summary_rows
    rows = [r for r in rows if not (r["domain"] == "new_sessions" and r["participant"] == "T2")]
    result = audit.reconstruct_summary(rows, plan)
    assert not result["gates"]["reduced_label_useful"]
    assert not result["gates"]["calibration_noninferiority"]
    assert not result["gates"]["zero_fresh_label_useful"]
    assert result["gates"]["T3_recovered"]


def test_comparison_rejects_different_scoring_masks(summary_rows):
    rows, plan = summary_rows
    rows[0]["eval_mask_sha256"] = "different"
    with pytest.raises(ValueError, match="different scoring masks"):
        audit.reconstruct_summary(rows, plan)


def test_session_imbalance_does_not_change_participant_weight(summary_rows):
    rows, plan = summary_rows
    extra = [
        {**r, "session_id": r["session_id"] + "-extra"}
        for r in rows
        if r["domain"] == "new_sessions" and r["participant"] == "T2"
    ]
    before, after = (
        audit.reconstruct_summary(rows, plan),
        audit.reconstruct_summary(rows + extra, plan),
    )
    for method in plan["assessment"]["new_sessions"]["methods"]:
        a, b = [s["domains"]["new_sessions"]["methods"][method] for s in (before, after)]
        assert a["mean_r2"] == b["mean_r2"]
        assert a["r2_bootstrap_95"] == b["r2_bootstrap_95"]


@pytest.mark.parametrize("tamper", [None, "prediction", "mask", "hash"])
def test_full_artifact_verification_detects_rehashed_trace_tampering(synthetic, tmp_path, tamper):
    features, y, mask, segments = synthetic
    cfg = config()
    raw = audit.reference_representation(features, segments, cfg)
    z, mean, scale = audit.reference_design(raw, 150, 4)
    stats = audit.reference_statistics(z[:150], y[:150], audit.reference_mask(mask, segments, 150))
    weights = model.solve_daily(stats, 1.0, 1)
    fitted = model.Decoder(cfg, mean, scale, weights)
    model_path, trace_path = tmp_path / "model.npz", tmp_path / "trace.npz"
    fitted.save(model_path)
    prediction = fitted.predict(features, segments)
    np.savez_compressed(trace_path, prediction=prediction, eval_mask=mask)
    metrics, _ = audit.score_reference(y, prediction, mask)
    metrics["mse"] = float(np.square(y[mask] - prediction[mask]).mean())
    row = {
        "model_path": "model.npz",
        "model_sha256": audit.sha256(model_path),
        "trace_path": "trace.npz",
        "trace_sha256": audit.sha256(trace_path),
        "eval_mask_sha256": audit.mask_hash(mask),
        "metrics": metrics,
    }
    if tamper is not None:
        changed_mask = mask.copy()
        if tamper == "prediction":
            prediction[200, 0] += 0.1
        elif tamper == "mask":
            changed_mask[200] = not changed_mask[200]
        np.savez_compressed(trace_path, prediction=prediction, eval_mask=changed_mask)
        row["trace_sha256"] = "wrong" if tamper == "hash" else audit.sha256(trace_path)
        with pytest.raises(ValueError, match="replay|mask|hash"):
            audit.check_artifacts(
                row, raw, {"n_bins": 400, "n_channels": 4}, cfg, 150, {"y": y}, mask, tmp_path
            )
    else:
        _, _, _, error = audit.check_artifacts(
            row, raw, {"n_bins": 400, "n_channels": 4}, cfg, 150, {"y": y}, mask, tmp_path
        )
        assert error < 1e-12


def test_acquisition_enrichment_is_bound_to_original_archive(tmp_path):
    """The preparer adds archive identity fields absent from receipt.selected."""
    root = tmp_path
    (root / "acquisition").mkdir()
    (root / "raw/T2").mkdir(parents=True)
    archive = {"file_id": 123, "provider_sha256": "archive-digest"}
    members = []
    for rank in range(1, 13):
        sid = f"T2_day_{rank}_decoding"
        path = root / f"raw/T2/{sid}.mat"
        path.write_bytes(b"synthetic raw identity")
        members.append(
            {
                "participant_id": "T2",
                "session_id": sid,
                "post_implant_day": rank,
                "regular_session_rank": rank,
                "archive_ordinal": rank,
                "member_name": f"decoding/T2/{sid}.mat",
                "member_size": path.stat().st_size,
                "member_sha256": audit.sha256(path),
                "raw_path": str(path.relative_to(root)),
            }
        )
    enrich = {
        "archive_file_id": 123,
        "archive_sha256_published": "archive-digest",
        "full_archive_sha256_verified": False,
    }
    old = {
        m["session_id"]: {"participant_id": "T2", "source": {**m, **enrich}} for m in members[:9]
    }
    parents = {"v2_index_sha256": "v2", "v3_index_sha256": "v3"}
    receipt = {
        "status": "complete",
        "participant_id": "T2",
        "plan_sha256": "acquisition",
        **parents,
        "archive": archive,
        "selection": {
            "regular_session_ranks_one_based": [10, 11, 12],
            "scores_or_duration_used_for_selection": False,
        },
        "seen_members": [
            {
                "regular_file": True,
                "name": m["member_name"],
                "size": m["member_size"],
                "ordinal": m["archive_ordinal"],
            }
            for m in members
        ],
        "verified_prior_members": members[:9],
        "selected": members[9:],
        "tar_end_observed": False,
        "missing_requested_ranks": [],
    }
    receipt_path = root / "acquisition/T2.json"
    receipt_path.write_text(json.dumps(receipt))
    new = {
        m["session_id"]: {
            "source": {**m, **enrich},
            "post_implant_day": m["post_implant_day"],
            "raw_source_root": str(root),
            "n_bins": 30000,
            "full_prepared_n_bins": 40000,
        }
        for m in members[9:]
    }
    inputs = SimpleNamespace(
        path=root / "index.json",
        items=new,
        index={
            "role": "new_sessions",
            "participants": ["T2"],
            **parents,
            "acquisition_sources": [
                {"path": "acquisition/T2.json", "sha256": audit.sha256(receipt_path)}
            ],
        },
    )
    development = SimpleNamespace(items=old, index=parents)
    plan = {"new_session_participants": ["T2"], "acquisition_plan_sha256": "acquisition"}
    audit.validate_fresh_provenance(inputs, development, plan)
    next(iter(new.values()))["source"]["archive_file_id"] = 999
    with pytest.raises(ValueError, match="authenticated selection"):
        audit.validate_fresh_provenance(inputs, development, plan)
