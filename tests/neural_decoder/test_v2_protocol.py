"""Independent attacks on V2 person, time, scorer, and artifact boundaries."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from neural_experiments.v2 import audit, transfer


def recording(n=80, channels=3):
    rng = np.random.default_rng(12)
    x = rng.poisson(2, (n, channels)).astype(np.float32)
    y = np.column_stack((np.arange(n), -np.arange(n))).astype(np.float32)
    return transfer.Recording(
        {"participant_id": "T2", "session_id": "T2_day_1_decoding"},
        x,
        y,
        np.arange(n) % 3 != 0,
        [(0, n // 2), (n // 2, n)],
    )


def network(channels=3):
    torch.manual_seed(23)
    return transfer.AdapterTCN(channels, width=4, dilations=(1, 2)).eval()


def test_future_features_and_all_labels_cannot_change_past_prediction():
    data = recording()
    stats = transfer.fit_transform([data.prefix(12)])
    model = network()
    reference = transfer.predict(model, data.x, data.segments, stats)
    altered = copy.deepcopy(data)
    altered.x[31:] += 1000
    changed = transfer.predict(model, altered.x, altered.segments, stats)
    np.testing.assert_array_equal(reference[:31], changed[:31])
    altered = copy.deepcopy(data)
    altered.y[:] = np.nan
    altered.mask[:] = ~altered.mask
    np.testing.assert_array_equal(
        reference, transfer.predict(model, altered.x, altered.segments, stats)
    )


def test_block_resets_isolate_history_and_incremental_matches_batch():
    data = recording()
    stats = transfer.fit_transform([data.prefix(12)])
    model = network()
    whole = transfer.predict(model, data.x, data.segments, stats)
    decoder = transfer.IncrementalDecoder(model, stats)
    streamed = []
    for start, stop in data.segments:
        decoder.reset()
        streamed.extend(decoder.step(x) for x in data.x[start:stop])
    np.testing.assert_allclose(streamed, whole, atol=2e-6, rtol=2e-6)
    changed = copy.deepcopy(data)
    changed.x[:40] = 1000
    np.testing.assert_array_equal(
        transfer.predict(model, changed.x, changed.segments, stats)[40:], whole[40:]
    )


def test_calibration_prefix_statistics_and_loss_rows_ignore_future():
    data = recording()
    prefix = data.prefix(60)
    stats = transfer.fit_transform([prefix])
    chunks = transfer.make_chunks([prefix], stats, burn=6, length=11)
    altered = copy.deepcopy(data)
    altered.x[60:] = 1e7
    altered.y[60:] = 1e7
    altered.mask[60:] = ~altered.mask[60:]
    altered_prefix = altered.prefix(60)
    later_stats = transfer.fit_transform([altered_prefix])
    later_chunks = transfer.make_chunks([altered_prefix], later_stats, burn=6, length=11)
    for actual, expected in zip(later_stats, stats, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(later_chunks, chunks, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Exact source rows, with no duplicates, padding, or block startup loss.
    eligible = prefix.mask.copy()
    eligible[:6] = False
    eligible[40:46] = False
    np.testing.assert_array_equal(np.sort(chunks[1][chunks[2]][:, 0]), np.flatnonzero(eligible))
    poisoned = copy.deepcopy(prefix)
    poisoned.y[:] = np.nan
    poisoned.mask[:] = False
    for actual, expected in zip(transfer.fit_transform([poisoned]), stats, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("damage", ["nan_mean", "infinite_scale", "wrong_mean_shape", "zero_scale"])
def test_model_loader_rejects_invalid_normalization(tmp_path, damage):
    model = network()
    path = tmp_path / "model.npz"
    transfer.export_model(model, (np.zeros(3, np.float32), np.ones(3, np.float32)), path)
    values = audit.numeric(path)
    if damage == "nan_mean":
        values["input_mean"][0] = np.nan
    elif damage == "infinite_scale":
        values["input_scale"][0] = np.inf
    elif damage == "wrong_mean_shape":
        values["input_mean"] = np.zeros(4, np.float32)
    else:
        values["input_scale"][0] = 0
    np.savez(path, **values)
    with pytest.raises(ValueError):
        transfer.load_model(path)


@pytest.fixture
def role_fixture():
    config = json.loads(
        (Path(__file__).parents[2] / "neural_experiments/v2/transfer_plan.json").read_text()
    )
    sessions = [
        {
            "participant_id": p,
            "session_id": f"{p}_day_{day}_decoding",
            "post_implant_day": day,
            "bin_seconds": 0.02,
            "n_bins": 30001,
        }
        for p in config["participants"]
        for day in [1, 12, 3]
    ]
    grouped = audit.sessions_by_person({"sessions": sessions}, config)
    train, validation = audit.source_roles(grouped, "T2")
    prefix = {"session_id": "T2_day_1_decoding", "start_bin": 0, "stop_bin": 6000}
    identity = {
        "target_participant": "T2",
        "method": "pretrained_frozen_core",
        "seed": 1,
        "budget_seconds": 120,
        "source_train_session_ids": train,
        "source_validation_session_ids": validation,
        "source_participants": sorted(set(config["participants"]) - {"T2"}),
        "calibration": prefix.copy(),
        "normalization": prefix.copy(),
    }
    return config, grouped, identity


def test_source_role_split_is_chronological_and_whole_person(role_fixture):
    config, grouped, identity = role_fixture
    audit.validate_model_roles(identity, grouped, config)
    assert identity["source_validation_session_ids"] == [
        f"{p}_day_12_decoding" for p in sorted(set(config["participants"]) - {"T2"})
    ]
    assert not any(s.startswith("T2_") for s in identity["source_train_session_ids"])


@pytest.mark.parametrize(
    "damage",
    [
        "target_in_source",
        "future_norm",
        "future_labels",
        "wrong_first_day",
        "train_validation_swap",
        "scratch_consumes_source",
        "duplicate_session",
    ],
)
def test_independent_role_audit_rejects_leakage(role_fixture, damage):
    config, grouped, original = role_fixture
    identity = copy.deepcopy(original)
    if damage == "target_in_source":
        identity["source_train_session_ids"].append("T2_day_1_decoding")
    elif damage == "future_norm":
        identity["normalization"]["stop_bin"] += 1
    elif damage == "future_labels":
        identity["calibration"]["stop_bin"] += 1
    elif damage == "wrong_first_day":
        identity["calibration"]["session_id"] = "T2_day_3_decoding"
    elif damage == "train_validation_swap":
        identity["source_train_session_ids"], identity["source_validation_session_ids"] = (
            identity["source_validation_session_ids"],
            identity["source_train_session_ids"],
        )
    elif damage == "scratch_consumes_source":
        identity["method"] = "scratch"
    else:
        identity["source_train_session_ids"].append(identity["source_train_session_ids"][0])
    with pytest.raises(ValueError):
        audit.validate_model_roles(identity, grouped, config)


def test_independent_metric_keeps_constant_axis_errors_and_zero_predictions():
    y = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    p = y.copy()
    p[:, 1] = 2
    actual = audit.score_arrays(y, p, np.ones(4, bool))
    assert actual["r2"] == -3.0
    assert actual["per_output_r2"] == [1.0, None]
    zero = audit.score_arrays(y, np.zeros_like(y), np.ones(4, bool))
    assert zero["angular_error_degrees"] == 90
    assert zero["zero_prediction_count"] == 4
    audit.compare_metrics(transfer.score(y, p, np.ones(4, bool)), actual)


def test_scoring_rows_and_undefined_values_cannot_be_silently_changed():
    y = np.array([[1.0, 0], [np.nan, np.nan], [-1.0, 0]])
    mask = np.array([True, False, True])
    metrics = audit.score_arrays(y, np.zeros((3, 2)), mask)
    assert metrics["n"] == 2
    assert audit.mask_hash(mask) != audit.mask_hash(~mask)
    with pytest.raises(ValueError, match="Nonfinite"):
        audit.score_arrays(y, np.zeros((3, 2)), np.ones(3, bool))
    with pytest.raises(ValueError, match="N by 2"):
        audit.score_arrays(y, np.zeros((2, 2)), mask)
    tampered = copy.deepcopy(metrics)
    tampered["per_output_r2"][1] = 0.0
    with pytest.raises(ValueError, match="Undefined"):
        audit.compare_metrics(tampered, metrics)


def test_hash_and_path_mismatches_fail_closed(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text("{}")
    original = audit.sha256(path)
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="hash mismatch"):
        audit.verify(path, original)
    with pytest.raises(ValueError, match="relative"):
        audit.contained(tmp_path, "../other.json")
    with pytest.raises(ValueError, match="relative"):
        audit.contained(tmp_path, "/tmp/unrelated.json")


def test_export_audit_rejects_future_statistics_and_changed_frozen_core(tmp_path, role_fixture):
    config, _, identity = role_fixture
    x = recording().x
    model = transfer.AdapterTCN(3, 32, config["model"]["dilations"])
    path = tmp_path / "model.npz"
    transfer.export_model(model, transfer.fit_transform([recording()]), path)
    arrays = audit.numeric(path)
    identity["parameter_count"] = sum(p.numel() for p in model.parameters())
    core = {k.removeprefix("core."): v.copy() for k, v in arrays.items() if k.startswith("core.")}
    audit.validate_model_export(identity, arrays, x, config, core)
    altered = {k: v.copy() for k, v in arrays.items()}
    altered["input_mean"][0] += 0.01
    with pytest.raises(ValueError, match="normalization"):
        audit.validate_model_export(identity, altered, x, config, core)
    altered = {k: v.copy() for k, v in arrays.items()}
    altered["core.0.weight"][0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="Frozen core"):
        audit.validate_model_export(identity, altered, x, config, core)


def test_aggregation_uses_equal_days_then_seeds_then_people(role_fixture):
    config, _, _ = role_fixture
    scored = []
    for person, gain in (("T2", 0.2), ("T3", -0.1)):
        for method in ("scratch", "pretrained_frozen_core"):
            for seed in range(3):
                for day, score in ((1, -0.2), (2, 0.4)):
                    scored.append(
                        {
                            "target_participant": person,
                            "method": method,
                            "seed": seed,
                            "budget_seconds": 120,
                            "session_id": f"{person}_day_{day}",
                            "partition": "future_day",
                            "metrics": {
                                "r2": score
                                + seed / 100
                                + (gain if method.startswith("pretrained") else 0)
                            },
                        }
                    )
    result = audit.reconstruct_summary(scored, config)
    effect = result["primary_contrast"]
    assert effect["n_people"] == 2
    assert effect["mean_difference"] == pytest.approx(0.05)
    assert effect["exact_two_sided_sign_flip_p"] == 1.0
    assert effect["positive_people"] == 1
    # Six seeds and twelve sessions remain two independent participant units.
    group = next(
        g
        for g in result["groups"]
        if g["partition"] == "future_day"
        and g["budget_seconds"] == 120
        and g["method"] == "scratch"
    )
    assert group["mean_person_r2"] == pytest.approx(0.11)
    assert [p["n_seeds"] for p in group["people"]] == [3, 3]


def synthetic_study(root, config):
    """Receipt-contract fixture only: synthetic data/weights, no model training."""
    cache, run = root / "cache", root / "run"
    cache.mkdir()
    run.mkdir()

    def save_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n")
        return str(path.relative_to(root)), audit.sha256(path)

    policy = {
        "selection": "first_n_complete_session_mat_members_in_provider_archive_order",
        "scores_used_for_selection": False,
        "ordering_after_selection": "numeric_post_implant_day_then_session_id",
        "neural_feature": "tx_4_5",
        "fallback": None,
        "count_conversion": "sum_two_consecutive_10ms_bins_within_each_block",
        "bin_seconds": 0.02,
        "normalization": "none",
        "upstream_feature_causality": "unverified",
    }
    index = {
        "schema_version": 1,
        "participants": list(audit.PARTICIPANTS),
        "sessions": [],
        "selection": {**policy, "max_sessions_per_participant": 6},
        "acquisition_sources": [],
    }
    archives = []
    for i, person in enumerate(audit.PARTICIPANTS):
        archive = {
            "participant_id": person,
            "file_id": 4952113 + i,
            "size": 1,
            "provider_sha256": audit.ARCHIVE_DIGESTS[person],
            "filename": f"decoding_{person}.tar.gz",
        }
        archives.append(archive)
        selected, seen = [], []
        for ordinal, (day, n) in enumerate(((1, 1500), (2, 8))):
            sid = f"{person}_day_{day}_decoding"
            raw = cache / "raw" / person / f"{sid}.mat"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(b"SYNTHETIC PROTOCOL TEST, NOT A PROVIDER MAT FILE")
            member = {
                "participant_id": person,
                "session_id": sid,
                "post_implant_day": day,
                "archive_ordinal": ordinal,
                "member_name": f"decoding/{person}/{sid}.mat",
                "member_size": raw.stat().st_size,
                "member_sha256": audit.sha256(raw),
                "raw_path": str(raw.relative_to(cache)),
            }
            selected.append(member)
            seen.append(
                {
                    "ordinal": ordinal,
                    "name": member["member_name"],
                    "size": member["member_size"],
                    "regular_file": True,
                }
            )
            x = np.ones((n, 96), np.float32)
            y = np.zeros((n, 2), np.float32)
            y[:, 0] = np.where(np.arange(n) % 2, 1, -1)
            fp, tp = cache / f"{sid}.features.npz", cache / f"{sid}.targets.npz"
            np.savez_compressed(
                fp,
                x=x,
                timestamps=(np.arange(n) + 1) * 0.02,
                sequence_id=np.zeros(n, np.int64),
                channel_ids=np.arange(96),
            )
            np.savez_compressed(tp, y=y, eval_mask=np.ones(n, bool))
            index["sessions"].append(
                {
                    "participant_id": person,
                    "session_id": sid,
                    "post_implant_day": day,
                    "n_bins": n,
                    "n_channels": 96,
                    "bin_seconds": 0.02,
                    "segments": [{"start": 0, "end": n, "sequence_id": 0}],
                    "feature_path": fp.name,
                    "target_path": tp.name,
                    "feature_sha256": audit.sha256(fp),
                    "target_sha256": audit.sha256(tp),
                    "source": {**member, "archive_sha256_published": archive["provider_sha256"]},
                }
            )
        receipt = {
            "status": "complete",
            "participant_id": person,
            "requested_sessions": 6,
            "archive": archive,
            "policy": policy,
            "selected": selected,
            "seen_members": seen,
            "tar_end_observed": True,
            "full_archive_sha256_verified": False,
        }
        path = cache / "acquisition" / f"{person}.json"
        _, digest = save_json(path, receipt)
        index["acquisition_sources"].append(
            {"path": str(path.relative_to(cache)), "sha256": digest}
        )
    provider = cache / "provider-manifest.json"
    _, digest = save_json(provider, {"metadata_sources": [], "archives": archives})
    index["provider"] = {
        "doi": "10.5061/dryad.x0k6djj1h",
        "version_id": 462702,
        "license": "CC0-1.0",
        "manifest_path": provider.name,
        "manifest_sha256": digest,
    }
    index_path = cache / "index.json"
    ip, ih = save_json(index_path, index)
    cp, ch = save_json(root / "plan.json", config)
    save_json(run / "plan.json", config)
    save_json(run / "input_index.json", index)
    manifest = {
        "status": "complete",
        "index_path": ip,
        "index_sha256": ih,
        "config_path": cp,
        "config_sha256": ch,
        "source": {},
        "source_snapshots": {},
        "models": [],
        "unavailable": [],
        "pretraining": [],
    }
    for name in ("transfer.py", "data.py"):
        snapshot = run / "source" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text("# Synthetic contract fixture; not executed training code.\n")
        original = f"neural_experiments/v2/{name}"
        manifest["source"][original] = audit.sha256(snapshot)
        manifest["source_snapshots"][original] = str(snapshot.relative_to(root))
    grouped = audit.sessions_by_person(index, config)
    core = {
        f"{i}.{k}": np.zeros((32, 32, 3) if k == "weight" else (32,), np.float32)
        for i in range(5)
        for k in ("weight", "bias")
    }
    stats = {"input_mean": np.ones(96, np.float32), "input_scale": np.full(96, 0.1, np.float32)}
    for person in config["participants"]:
        train, validation = audit.source_roles(grouped, person)
        people = sorted(set(grouped) - {person})
        for seed in config["seeds"]:
            directory = run / "pretrain" / person / str(seed)
            directory.mkdir(parents=True)
            core_path = directory / "core.npz"
            np.savez(core_path, **core)
            core_hash = audit.sha256(core_path)
            norm_path = directory / "source_normalization.npz"
            np.savez(norm_path, **{f"{p}__{k}": v for p in people for k, v in stats.items()})
            receipt = {
                "seed": seed,
                "source_people": people,
                "source_train_session_ids": train,
                "source_validation_session_ids": validation,
                "source_normalization": {
                    p: [{"session_id": f"{p}_day_1_decoding", "start_bin": 0, "stop_bin": 1500}]
                    for p in people
                },
                "source_normalization_path": str(norm_path.relative_to(root)),
                "source_normalization_sha256": audit.sha256(norm_path),
                "losses": [
                    {"step": i + 1, "participant": people[i % 8], "loss": 1.0, "n": 10}
                    for i in range(config["source_training"]["steps"])
                ],
                "checkpoints": [
                    {"step": step, "validation_equal_person_mse": loss, "validation_people": 8}
                    for loss, step in enumerate(config["source_training"]["checkpoints"], start=1)
                ],
                "selected_step": config["source_training"]["checkpoints"][0],
                "core_sha256": core_hash,
            }
            rp, rh = save_json(directory / "training.json", receipt)
            manifest["pretraining"].append(
                {
                    "target_participant": person,
                    "seed": seed,
                    "receipt_path": rp,
                    "receipt_sha256": rh,
                }
            )
            for method in config["methods"]:
                if method == "ridge" and seed != 0:
                    continue
                for budget in (120, 300):
                    manifest["unavailable"].append(
                        {
                            "target_participant": person,
                            "method": method,
                            "seed": seed,
                            "budget_seconds": budget,
                            "reason": "earliest session shorter than prefix",
                        }
                    )
                directory = run / "models" / f"{person}-{method}-{seed}"
                directory.mkdir(parents=True)
                arrays = (
                    {
                        **stats,
                        "coefficient": np.zeros((2, 96), np.float32),
                        "intercept": np.zeros(2, np.float32),
                    }
                    if method == "ridge"
                    else {
                        **stats,
                        "format_version": np.array(2),
                        "dilations": np.array([1, 2, 4, 8, 16]),
                        "adapter.weight": np.zeros((32, 96), np.float32),
                        "adapter.bias": np.zeros(32, np.float32),
                        "head.weight": np.zeros((2, 32), np.float32),
                        "head.bias": np.zeros(2, np.float32),
                        **{f"core.{k}": v for k, v in core.items()},
                    }
                )
                model_path = directory / "model.npz"
                np.savez(model_path, **arrays)
                effective = np.ones(1500, bool)
                effective[:62] = False
                prefix = {
                    "session_id": f"{person}_day_1_decoding",
                    "start_bin": 0,
                    "stop_bin": 1500,
                }
                identity = {
                    "target_participant": person,
                    "method": method,
                    "seed": seed,
                    "budget_seconds": 30,
                    "source_participants": people if method.startswith("pretrained") else [],
                    "source_train_session_ids": train if method.startswith("pretrained") else [],
                    "source_validation_session_ids": validation
                    if method.startswith("pretrained")
                    else [],
                    "calibration": {
                        **prefix,
                        "eligible_count": 1500,
                        "mask_sha256": audit.mask_hash(np.ones(1500, bool)),
                        "training_count": 1438,
                        "training_mask_sha256": audit.mask_hash(effective),
                    },
                    "normalization": prefix,
                    "model_path": str(model_path.relative_to(root)),
                    "model_sha256": audit.sha256(model_path),
                    "core_sha256": core_hash if method.startswith("pretrained") else None,
                    "parameter_count": sum(
                        v.size
                        for k, v in arrays.items()
                        if k not in {"input_mean", "input_scale", "format_version", "dilations"}
                    ),
                    "traces": [],
                }
                for row in grouped[person]:
                    n = row["n_bins"]
                    first = row["post_implant_day"] == 1
                    mask = np.full(n, not first, bool)
                    prediction = directory / f"{row['session_id']}.npy"
                    np.save(prediction, np.zeros((n, 2), np.float32), allow_pickle=False)
                    metrics = (
                        {
                            "r2": None,
                            "per_output_r2": [None, None],
                            "angular_error_degrees": None,
                            "zero_prediction_count": 0,
                            "n": 0,
                            "unavailable_reason": "fewer than minimum scored evaluation bins",
                        }
                        if first
                        else {
                            "r2": 0.0,
                            "per_output_r2": [0.0, None],
                            "angular_error_degrees": 90.0,
                            "zero_prediction_count": n,
                            "n": n,
                        }
                    )
                    partition = "same_session_suffix" if first else "future_day"
                    identity["traces"].append(
                        {
                            "session_id": row["session_id"],
                            "partition": partition,
                            "start_bin": n if first else 0,
                            "stop_bin": n,
                            "prediction_path": str(prediction.relative_to(root)),
                            "prediction_sha256": audit.sha256(prediction),
                            "score_mask_sha256": audit.mask_hash(mask),
                            "metrics": metrics,
                        }
                    )
                rp, rh = save_json(directory / "identity.json", identity)
                manifest["models"].append({"identity_path": rp, "identity_sha256": rh})
    _, manifest_hash = save_json(run / "manifest.json", manifest)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(transfer, "ROOT", root)
        summary = transfer.summarize(manifest, config)
    summary["manifest_sha256"] = manifest_hash
    save_json(run / "summary.json", summary)
    return run, index_path


def test_complete_synthetic_protocol_inventory_and_tampered_trace(tmp_path, role_fixture):
    config, _, _ = role_fixture
    run, index_path = synthetic_study(tmp_path, config)
    receipt = audit.audit(run, index_path, tmp_path)
    assert receipt["models_checked"] == 90
    assert receipt["traces_checked"] == 180
    assert receipt["unavailable_models"] == 180
    assert receipt["source_cores_checked"] == 27
    assert receipt["selected_members_checked"] == 18
    # Alter a prediction after every receipt was finalized; it must fail closed.
    manifest = audit.read_json(run / "manifest.json")
    identity_path = tmp_path / manifest["models"][0]["identity_path"]
    identity = audit.read_json(identity_path)
    path = tmp_path / identity["traces"][1]["prediction_path"]
    np.save(path, np.ones((8, 2), np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match="hash mismatch"):
        audit.audit(run, index_path, tmp_path)
    # Updating every digest cannot hide scores that disagree with the new trace.
    identity["traces"][1]["prediction_sha256"] = audit.sha256(path)
    identity_path.write_text(json.dumps(identity))
    manifest["models"][0]["identity_sha256"] = audit.sha256(identity_path)
    (run / "manifest.json").write_text(json.dumps(manifest))
    summary = audit.read_json(run / "summary.json")
    summary["manifest_sha256"] = audit.sha256(run / "manifest.json")
    (run / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Independent aggregate mismatch: trace.metrics.r2"):
        audit.audit(run, index_path, tmp_path)


def test_shared_source_optimizer_deduplicates_core_and_keeps_other_adapter_fixed():
    torch.manual_seed(71)
    shared = transfer.SharedTCN({"T2": 3, "T3": 4}, 4, (1, 2))
    assert shared.models["T2"].core is shared.models["T3"].core
    parameters = list(shared.parameters())
    assert len(parameters) == len({id(p) for p in parameters})
    expected = sum(p.numel() for p in shared.models["T2"].core.parameters())
    for model in shared.models.values():
        expected += sum(
            p.numel() for module in (model.adapter, model.head) for p in module.parameters()
        )
    assert sum(p.numel() for p in parameters) == expected
    optimizer = torch.optim.AdamW(parameters, lr=0.01)
    other_before = copy.deepcopy(shared.models["T3"].state_dict())
    core_before = copy.deepcopy(shared.models["T2"].core.state_dict())
    data = recording()
    stats = transfer.fit_transform([data])
    chunks = transfer.make_chunks([data], stats, burn=6, length=16)
    loss, n = transfer.train_batch(shared.models["T2"], chunks, [0, 1], optimizer, 0.1)
    assert np.isfinite(loss) and n > 0
    assert any(
        not torch.equal(value, shared.models["T2"].core.state_dict()[key])
        for key, value in core_before.items()
    )
    for key, value in other_before.items():
        if not key.startswith("core."):
            torch.testing.assert_close(value, shared.models["T3"].state_dict()[key], atol=0, rtol=0)


def test_ridge_preserves_empty_future_session_as_null_trace(tmp_path, monkeypatch, role_fixture):
    config, _, _ = role_fixture
    model, stats = transfer.fit_ridge(recording(), config["ridge"], burn=6)
    empty = recording(n=0)
    monkeypatch.setattr(transfer, "ROOT", tmp_path)
    traces = transfer.evaluate(
        model, stats, [empty], "different_calibration_session", config, tmp_path, "ridge"
    )
    assert len(traces) == 1
    assert traces[0]["partition"] == "future_day"
    assert traces[0]["metrics"]["r2"] is None
    assert traces[0]["metrics"]["n"] == 0
    assert np.load(tmp_path / traces[0]["prediction_path"], allow_pickle=False).shape == (0, 2)


def test_actual_training_run_export_and_scoring_path_on_tiny_synthetic_cohort(
    tmp_path, monkeypatch, role_fixture
):
    """Exercise production optimizer/run APIs; these are not locked study settings."""
    locked, _, _ = role_fixture
    plan = copy.deepcopy(locked)
    plan.update(
        participants=["T2", "T3", "T5"],
        seeds=[0],
        calibration_budgets_seconds=[1.4, 5.0],
        common_suffix_start_seconds=1.4,
        source_session_cap_seconds=4,
        evaluation_session_cap_seconds=4,
    )
    plan["model"].update(width=4, dilations=[1, 2])
    plan["source_training"].update(steps=2, checkpoints=[1, 2], batch_chunks=2, chunk_bins=16)
    plan["adaptation"].update(epochs=1, batch_chunks=2, chunk_bins=16)
    original_file = Path(transfer.__file__)
    copied_file = tmp_path / "neural_experiments/v2/transfer.py"
    copied_file.parent.mkdir(parents=True)
    copied_file.write_bytes(original_file.read_bytes())
    copied_file.with_name("data.py").write_bytes(original_file.with_name("data.py").read_bytes())
    monkeypatch.setattr(transfer, "__file__", str(copied_file))
    monkeypatch.setattr(transfer, "ROOT", tmp_path)
    rng = np.random.default_rng(43)
    records = []
    for person, channels in zip(plan["participants"], (3, 4, 5), strict=True):
        for day, n in ((1, 130), (2, 7)):
            x = rng.poisson(2, (n, channels)).astype(np.float32)
            angle = np.linspace(-2.5, 2.5, n)
            y = np.column_stack((np.cos(angle), np.sin(angle))).astype(np.float32)
            records.append(
                transfer.Recording(
                    {
                        "participant_id": person,
                        "session_id": f"{person}_day_{day}",
                        "post_implant_day": day,
                    },
                    x,
                    y,
                    np.ones(n, bool),
                    [(0, 0), (0, n)],
                )
            )
    index = {"synthetic_only": True, "participants": plan["participants"]}
    index_path, plan_path, output = (
        tmp_path / "index.json",
        tmp_path / "plan.json",
        tmp_path / "run",
    )
    index_path.write_text(json.dumps(index))
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(transfer, "load_recordings", lambda _: (index, records))
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        transfer.run(index_path, plan_path, output)
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)
    manifest = audit.read_json(output / "manifest.json")
    assert manifest["status"] == "complete"
    assert len(manifest["models"]) == 12
    assert len(manifest["unavailable"]) == 12
    assert len(manifest["pretraining"]) == 3
    records_by_id = {r.session_id: r for r in records}
    for reference in manifest["models"]:
        identity_path = tmp_path / reference["identity_path"]
        audit.verify(identity_path, reference["identity_sha256"])
        identity = audit.read_json(identity_path)
        model_path = tmp_path / identity["model_path"]
        audit.verify(model_path, identity["model_sha256"])
        if identity["method"].startswith("pretrained"):
            assert identity["target_participant"] not in identity["source_participants"]
        else:
            assert identity["source_participants"] == []
        assert identity["calibration"]["eligible_count"] == 70
        assert identity["calibration"]["training_count"] == 64
        assert len(identity["traces"]) == 2
        for trace in identity["traces"]:
            record = records_by_id[trace["session_id"]]
            prediction_path = tmp_path / trace["prediction_path"]
            audit.verify(prediction_path, trace["prediction_sha256"])
            prediction = np.load(prediction_path, allow_pickle=False)
            mask = record.mask.copy()
            mask[: trace["start_bin"]] = False
            audit.compare_metrics(trace["metrics"], audit.score_arrays(record.y, prediction, mask))
            if identity["method"] != "ridge":
                restored, stats = transfer.load_model(model_path)
                np.testing.assert_array_equal(
                    transfer.predict(restored, record.x, record.segments, stats), prediction
                )
            else:
                arrays = audit.numeric(model_path)
                stats = (arrays["input_mean"], arrays["input_scale"])
                x = transfer.filtered(record.x, record.segments, stats, plan["ridge"])
                restored_prediction = x @ arrays["coefficient"].T + arrays["intercept"]
                np.testing.assert_array_equal(restored_prediction, prediction)
        if identity["method"] == "pretrained_frozen_core":
            arrays = audit.numeric(model_path)
            core = audit.numeric(
                output / "pretrain" / identity["target_participant"] / "seed-0/core.npz"
            )
            for key, value in core.items():
                np.testing.assert_array_equal(arrays[f"core.{key}"], value)
    # A second call cannot overwrite the retained model/prediction evidence.
    with pytest.raises(FileExistsError):
        transfer.run(index_path, plan_path, output)
