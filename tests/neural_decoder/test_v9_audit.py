"""Independent V9 geometry, causal information boundaries and artifact mutation."""

import ast
import inspect

import numpy as np
import pytest

from neural_experiments.v9 import audit


def test_auditor_has_no_v9_producer_imports():
    for node in ast.walk(ast.parse(inspect.getsource(audit))):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("neural_experiments.v9")
        if isinstance(node, ast.Import):
            assert not any(n.name.startswith("neural_experiments.v9") for n in node.names)


def test_cardinal_sectors_do_not_split_cardinal_noise_into_fake_directions():
    for sector, angle in enumerate([0, np.pi / 2, np.pi, -np.pi / 2]):
        for jitter in (-0.1, 0.0, 0.1):
            assert (
                audit.sector_reference([np.cos(angle + jitter), np.sin(angle + jitter)]) == sector
            )


def test_common_mask_purges_maximum_purchase_even_when_method_stopped_early():
    n = 22000
    ids = np.arange(n) // 1900
    case = {
        "context": {
            "trial_id": ids,
            "trial_onset_seconds": ids * 1900 * 0.02,
            "trial_stop_bin": np.minimum((ids + 1) * 1900, n),
        },
        "eval_mask": np.ones(n, bool),
        "segments": [(0, n)],
    }
    mask, touched = audit.masks_reference(case)
    assert not mask[:19600].any()
    assert not np.isin(ids[mask], touched).any()
    # The trial crossing390s cannot score its remainder after392s.
    assert not mask[19600:20900].any()
    assert mask[20900:].all()
    short = {
        "context": {
            "trial_id": ids[:19000],
            "trial_onset_seconds": ids[:19000] * 1900 * 0.02,
            "trial_stop_bin": np.minimum((ids[:19000] + 1) * 1900, 19000),
        },
        "eval_mask": np.ones(19000, bool),
        "segments": [(0, 19000)],
    }
    assert not audit.masks_reference(short)[0].any()


def test_boxcar_future_invariance_and_block_reset():
    x = np.arange(100, dtype=float).reshape(50, 2)
    segments = [(0, 23), (23, 50)]
    expected = audit.boxcar_reference(x, segments)
    assert np.array_equal(expected[23], x[23])
    np.testing.assert_array_equal(expected[20], x[6:21].mean(0))
    changed = x.copy()
    changed[30:] = 1e6
    actual = audit.boxcar_reference(changed, segments)
    np.testing.assert_array_equal(expected[:30], actual[:30])


def test_weighted_mean_ridge_equations_invariant_to_weight_scaling():
    rng = np.random.default_rng(85)
    x = np.c_[rng.normal(size=(30, 3)), np.ones(30)]
    y = rng.normal(size=(30, 2))
    w = rng.uniform(0.1, 1, 30)
    a, b = audit.ridge_reference(x, y, w, 0.1)
    aa, bb = audit.ridge_reference(x, y, w * 100, 0.1)
    np.testing.assert_allclose(a, aa, atol=1e-14)
    np.testing.assert_allclose(b, bb, atol=1e-14)
    coef = np.linalg.solve(a, b)
    assert np.max(np.abs(a @ coef - b)) < 1e-14
    with pytest.raises(ValueError, match="negative"):
        audit.ridge_reference(x, y, -w, 0.1)


def test_source_selection_requires_complete_paired_inventory_and_ties_are_locked():
    configs = {"a": {}, "b": {}}
    rows = [
        {"session_id": s, "method": m, "score_mask_sha256": "same", "metrics": {"r2": 0.1}}
        for s in ("s1", "s2")
        for m in configs
    ]
    assert audit.choose_reference(rows, configs, ["s1", "s2"])["selected_method"] == "a"
    with pytest.raises(ValueError, match="inventory"):
        audit.choose_reference(rows[:-1], configs, ["s1", "s2"])
    rows[-1]["score_mask_sha256"] = "changed"
    with pytest.raises(ValueError, match="Unpaired"):
        audit.choose_reference(rows, configs, ["s1", "s2"])


def fixture(n=23000, trial_length=200):
    rng = np.random.default_rng(44)
    t = np.arange(n)
    trial = t // trial_length
    start = trial * trial_length
    direction = np.c_[np.cos((trial % 4) * np.pi / 2), np.sin((trial % 4) * np.pi / 2)]
    displacement = direction * (1 + 0.1 * np.sin((t - start) / 17))[:, None]
    target = np.tile([3.0, 2.0], (n, 1))
    return {
        "item": {
            "participant_id": "T2",
            "session_id": "synthetic",
            "post_implant_day": 1,
            "n_bins": n,
            "n_channels": 2,
            "far_distance": 0.2,
        },
        "features": {
            "x_tx3": rng.poisson(1, (n, 2)).astype(np.float32),
            "x_tx45": rng.poisson(0.3, (n, 2)).astype(np.float32),
            "x_sbp": rng.gamma(2, 1, (n, 2)).astype(np.float32),
        },
        "segments": [(0, n)],
        "y": direction.astype(np.float32),
        "eval_mask": np.ones(n, bool),
        "context": {
            "trial_id": trial,
            "trial_age_seconds": (t - start) * 0.02 + 0.01,
            "trial_start_bin": start,
            "trial_stop_bin": np.minimum(start + trial_length, n),
            "trial_onset_seconds": start * 0.02,
            "trial_right_censored": start + trial_length > n,
            "reference40_available": np.ones(n, bool),
            "reference40_distance": np.ones(n),
            "reference40_direction": direction,
            "target_position": target,
            "cursor_position": target - displacement,
        },
    }


def test_common_trial_windows_and_quota_are_independent_of_rt_or_target_head():
    case = fixture()
    records = audit.trial_records_reference(case)
    first = next(r for r in records if r["eligible"])
    assert first["windows"][0.8][1] - first["windows"][0.2][0] == 45
    assert first["windows"][0.2][0] == first["start_bin"] + 10
    assert first["completion_bin"] == first["stop_bin"] + 1
    quota = audit.quota_reference(records, len(case["y"]))
    assert quota["attained"] and len(quota["trial_ids"]) == 24
    assert quota["sector_counts"] == [6, 6, 6, 6]
    assert quota["stop_bin"] == 6401


def test_direction_diversity_shortage_times_out_without_dropping_case():
    case = fixture()
    case["context"]["reference40_direction"][:] = [1, 0]
    records = audit.trial_records_reference(case)
    quota = audit.quota_reference(records, len(case["y"]))
    assert not quota["attained"] and quota["stop_bin"] == 19500
    assert quota["sector_counts"][0] > 24 and quota["sector_counts"][1:] == [0, 0, 0]
    assert quota["observed_label_seconds"] == 360


def test_common_geometry_not_selected_rt_only_rejects_target_move_and_direction_cancel():
    case = fixture()
    ix = np.flatnonzero(case["context"]["trial_id"] == 10)
    # Motion between the two RT windows must fail the COMMON span, even though
    # either individually selected endpoint can be perfectly stationary.
    case["context"]["target_position"][ix[30:35], 0] += 0.1
    records = audit.trial_records_reference(case)
    assert not next(r for r in records if r["trial_id"] == 10)["eligible"]
    other = fixture()
    where = np.flatnonzero(other["context"]["trial_id"] == 11)[10:55]
    target = other["context"]["target_position"][where]
    other["context"]["cursor_position"][where] = (
        target - np.c_[np.cos(np.arange(45)), np.sin(np.arange(45))]
    )
    assert not next(r for r in audit.trial_records_reference(other) if r["trial_id"] == 11)[
        "eligible"
    ]


def test_completion_guard_and_future_labels_cannot_change_attained_quota():
    import copy

    case = fixture()
    records = audit.trial_records_reference(case)
    before = audit.quota_reference(records, len(case["y"]))
    changed = copy.deepcopy(case)
    stop = before["stop_bin"]
    changed["context"]["cursor_position"][stop:] = np.nan
    changed["y"][stop:] = np.nan
    after = audit.quota_reference(audit.trial_records_reference(changed), len(case["y"]))
    assert before == after
    # Last completed paired bin is not yet proof the raw go has ended.
    assert not audit.quota_reference(records, stop - 1)["attained"]
    assert audit.quota_reference(records, stop)["attained"]


def test_actual_summary_equal_person_pairing_and_missing_gate():
    import copy

    from neural_experiments.v9 import metrics

    rows = []
    for p, count in (("T2", 6), ("T7", 2)):
        for day in range(count):
            for m in audit.METHODS:
                r2 = (0.3 if m == audit.PRIMARY else 0.1) if p == "T2" else 0.7
                rows.append(
                    {
                        "session_id": f"{p}-{day}",
                        "participant": p,
                        "post_implant_day": day,
                        "method": m,
                        "score_mask_sha256": "common",
                        "metrics": {"r2": r2, "summed_mse": 1 - r2},
                    }
                )
    expected = audit.summary_reference(rows)
    audit.compare_structure(metrics.summarize(rows), expected)
    assert expected["methods"][audit.PRIMARY]["mean_r2"] == pytest.approx(0.5)
    assert all(expected["gates"].values())
    missing = copy.deepcopy(rows)
    for r in missing:
        if r["session_id"] == "T2-5":
            r["metrics"] = {"r2": None, "summed_mse": None}
    assert not any(audit.summary_reference(missing)["gates"].values())
    # No gate may secretly condition on successful calibration quotas.
    for r in rows:
        r["quota_attained"] = False
    audit.compare_structure(audit.summary_reference(rows), expected)
    rows[0]["score_mask_sha256"] = "changed"
    with pytest.raises(ValueError, match="Unpaired"):
        audit.summary_reference(rows)


def exported(tmp_path, monkeypatch, n=23000, rt=0.2, ridge=0.1):
    from neural_experiments.v9 import model

    monkeypatch.setattr(audit, "ROOT", tmp_path)
    case = fixture(n)
    configs = audit.configurations_reference(rt, ridge)
    result = model.evaluate_session(case, configs, artifact_dir=tmp_path / "artifacts")
    return case, configs, result


@pytest.mark.parametrize("n", [1000, 7499, 8000, 19599, 23000])
def test_all_twelve_actual_exports_independently_reconstructed(tmp_path, monkeypatch, n):
    case, configs, result = exported(tmp_path, monkeypatch, n=n)
    rows, errors, residuals = audit.check_session(case, configs, result["rows"], result)
    assert len(rows) == 12 and max(errors) < 1e-10 and max(residuals) < 1e-10
    if n < 19600:
        assert all(r["metrics"]["r2"] is None for r in rows)
    zero = next(r for r in rows if r["method"] == "zero")
    assert zero["training"]["training_rows"] == 0 and zero["available_label_seconds"] == 0


def test_final_configuration_grid_matches_locked_producer():
    from neural_experiments.v9 import development

    plan = {
        "source_grid": [
            {"rt_seconds": rt, "ridge": r}
            for rt, r in ((0.2, 0.1), (0.2, 1.0), (0.8, 0.1), (0.8, 1.0))
        ]
    }
    audit.compare_structure(development.source_grid(plan), audit.grid_reference())
    for config in audit.grid_reference().values():
        audit.compare_structure(
            development.assessment_configs(config),
            audit.configurations_reference(config["rt_seconds"], config["ridge"]),
        )


def test_nondefault_rt_and_penalty_actual_fit_parity(tmp_path, monkeypatch):
    case, configs, result = exported(tmp_path, monkeypatch, rt=0.8, ridge=1.0)
    audit.check_session(case, configs, result["rows"], result)


@pytest.mark.parametrize("field", ["mean", "scale", "weights", "version"])
def test_rehashed_model_tampering_fails_independent_optimum(tmp_path, monkeypatch, field):
    case, configs, result = exported(tmp_path, monkeypatch)
    row = next(r for r in result["rows"] if r["method"] == audit.PRIMARY)
    path = row["model"]["model_path"]
    arrays = audit.numeric(path)
    if field == "version":
        arrays[field] = arrays[field].astype(float)
    else:
        arrays[field].flat[0] += 0.01
    np.savez(path, **arrays)
    row["model"]["model_sha256"] = audit.sha256(path)
    with pytest.raises(ValueError, match="version|normalization|optimum"):
        audit.check_session(case, configs, result["rows"], result)


@pytest.mark.parametrize(
    "field",
    [
        "primary_mask",
        "quota_sector_mean__prediction",
        "quota_sector_mean__training_weights",
        "quota_sector_mean__secondary_endpoint",
    ],
)
def test_rehashed_trace_tampering_fails(tmp_path, monkeypatch, field):
    case, configs, result = exported(tmp_path, monkeypatch)
    path = result["rows"][0]["trace"]["trace_path"]
    arrays = audit.numeric(path)
    if arrays[field].dtype == bool:
        arrays[field].flat[-1] = not arrays[field].flat[-1]
    else:
        arrays[field].flat[-1] += 0.1
    np.savez(path, **arrays)
    for row in result["rows"]:
        row["trace"]["trace_sha256"] = audit.sha256(path)
    with pytest.raises(ValueError, match="mask|replay|training|secondary"):
        audit.check_session(case, configs, result["rows"], result)


def test_exact_raw_onset_purge_catches_first_pair_after_calibration():
    case = fixture()
    ids = case["context"]["trial_id"]
    # Synthetic go starts389.99s; no complete pair belongs to the paid interval.
    ids[19500:21000] = 999
    case["context"]["trial_onset_seconds"][19500:21000] = 389.99
    case["context"]["trial_stop_bin"][19500:21000] = 21000
    mask, purchased = audit.masks_reference(case)
    assert 999 in purchased and not mask[19600:21000].any()
    from neural_experiments.v9 import model

    actual, purged = model.score_mask(case)
    np.testing.assert_array_equal(actual, mask)
    np.testing.assert_array_equal(purged, purchased)


def test_target_definition_is_normalized_mean_displacement_not_mean_unit():
    case = fixture()
    record = next(r for r in audit.trial_records_reference(case) if r["trial_id"] == 10)
    a, b = record["windows"][0.2]
    # A small, high-magnitude directional deviation makes the two definitions differ.
    angles = np.pi + np.linspace(-0.1, 0.1, 15)
    delta = np.c_[np.cos(angles), np.sin(angles)] * np.linspace(0.3, 3, 15)[:, None]
    case["context"]["cursor_position"][a:b] = case["context"]["target_position"][a:b] - delta
    rebuilt = next(r for r in audit.trial_records_reference(case) if r["trial_id"] == 10)
    expected = delta.mean(0) / np.linalg.norm(delta.mean(0))
    np.testing.assert_allclose(rebuilt["window_targets"][0.2], expected, atol=1e-14)
    unit_mean = (delta / np.linalg.norm(delta, axis=1)[:, None]).mean(0)
    unit_mean /= np.linalg.norm(unit_mean)
    assert np.linalg.norm(unit_mean - expected) > 0.01


def test_passed_source_receipt_must_bind_exact_recipe_payload_and_phase(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(audit, "ROOT", tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    selected = audit.grid_reference()["grid_00"]
    payload = {
        "source_audit_status": "PENDING",
        "primary_method": audit.PRIMARY,
        "recipe_choice": {"method": "grid_00", "config": selected},
        "configurations": audit.configurations_reference(0.2, 0.1),
    }
    original = {
        "phase": "development",
        "status": "complete",
        "selection": payload,
        "code_sha256": {},
    }
    mpath = source / "manifest.json"
    mpath.write_text(json.dumps(original))
    receipt = {
        "status": "PASS",
        "phase": "development",
        "manifest_sha256": audit.sha256(mpath),
        "selection_sha256": audit.canonical_sha256(payload),
        "plan_sha256": audit.PLAN_SHA,
    }
    apath = tmp_path / "source-audit.json"
    apath.write_text(json.dumps(receipt))
    published = {
        **payload,
        "source_audit_status": "PASS",
        "source_manifest_sha256": audit.sha256(mpath),
        "source_audit_sha256": audit.sha256(apath),
        "source_audit_path": "source-audit.json",
    }
    ppath = tmp_path / "selection.json"
    ppath.write_text(json.dumps(published))
    manifest = {
        "source_selection": {"path": "selection.json", "sha256": audit.sha256(ppath)},
        "code_sha256": {},
        "configurations": published["configurations"],
        "primary_method": audit.PRIMARY,
    }
    assert audit.source_binding_reference(manifest, source, apath)[
        "selection_sha256"
    ] == audit.sha256(ppath)
    published["recipe_choice"] = {"method": "grid_01", "config": audit.grid_reference()["grid_01"]}
    ppath.write_text(json.dumps(published))
    manifest["source_selection"]["sha256"] = audit.sha256(ppath)
    with pytest.raises(ValueError, match="selection"):
        audit.source_binding_reference(manifest, source, apath)
    receipt["phase"] = "assessment"
    apath.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="completed source"):
        audit.source_binding_reference(manifest, source, apath)


def test_fit_future_target_and_feature_changes_cannot_affect_quota_artifact():
    import copy

    from neural_experiments.v9 import model

    case = fixture()
    config = audit.configurations_reference(0.8, 0.1)[audit.PRIMARY]
    model_before, receipt, _ = model.fit(case, config)
    stop = receipt["label_stop_bin"]
    other = copy.deepcopy(case)
    other["y"][stop:] = np.nan
    other["context"]["cursor_position"][stop:] = np.nan
    for key in other["features"]:
        other["features"][key][stop:] += 1000
    after, after_receipt, _ = model.fit(other, config)
    assert receipt == after_receipt
    for field in ("mean", "scale", "weights"):
        np.testing.assert_array_equal(getattr(model_before, field), getattr(after, field))
    raw = audit.representations_reference(other)[config["representation"]]
    ref = audit.fit_reference(other, config, raw)
    np.testing.assert_allclose(ref["gram"] @ after.weights, ref["rhs"], atol=1e-12)
