"""Bounded final-model lifecycle checks, separate from clean numeric timing.

Numeric resume does not guarantee exactly-once delivery or atomic checkpoint writes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from neural_decoder.artifacts import environment, sha256, source_identity, write_json
from neural_decoder.data import load_feature_file
from neural_decoder.snapshot import CheckpointedONNXStream
from neural_decoder.temporal import TCNStream, load_tcn
from neural_decoder.train import checked_development_index

PERIOD_NS = 20_000_000
ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE_CHECKS = {
    "same_file_resume_exact",
    "state_resume_exact",
    "channel_count",
    "nonfinite",
    "negative",
    "channel_order",
    "duplicate",
    "missing_input",
    "timestamp_gap",
    "explicit_discontinuity_reset",
    "reject_model_identity",
    "reject_graph_sha256",
    "reject_precision",
    "reject_stream_id",
    "truncated_rejected_prior_complete_retained",
    "wrong_next_sequence",
    "arithmetic_failure_poisoned_until_reset",
}


def _report_digest(receipt):
    report = {key: receipt[key] for key in ("lifecycle", "overhead", "faults", "qualified_arms")}
    return hashlib.sha256(json.dumps(report, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _execution_sources(source):
    return {
        name: digest
        for name, digest in source["files"].items()
        if name.endswith(".py")
        or name
        in {
            "pyproject.toml",
            "neural_experiments/configs/h1-final-closure-v1.json",
            "neural_experiments/H1_FINAL_CLOSURE_PROTOCOL.md",
        }
    }


def channel_signature(channel_ids):
    ids = np.asarray(channel_ids)
    if ids.shape != (176,) or ids.dtype.kind not in "iu" or len(set(ids.tolist())) != 176:
        raise ValueError("expected 176 unique integer channel IDs")
    return hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()


class CadencedONNXStream:
    """Validate announced metadata before numeric mutation; poison on gaps/failures."""

    def __init__(self, decoder, *, stream_id, channel_ids, period_ns=PERIOD_NS):
        if period_ns != PERIOD_NS:
            raise ValueError("fixed 20 ms cadence required")
        self.channel_signature = channel_signature(channel_ids)
        self.period_ns = period_ns
        self.wrapper = CheckpointedONNXStream(decoder, stream_id=stream_id)
        self.unusable = False

    @property
    def decoder(self):
        return self.wrapper.decoder

    def reset(self, *, stream_id):
        self.wrapper.reset(stream_id=stream_id)
        self.unusable = False

    def discontinuity(self):
        self.unusable = True

    def step(self, counts, *, sequence_id, timestamp_ns, channel_ids):
        if self.unusable:
            raise RuntimeError("stream unusable until reset or validated restore")
        if channel_signature(channel_ids) != self.channel_signature:
            raise ValueError("announced channel order differs from input manifest")
        for value in (sequence_id, timestamp_ns):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError("sequence and timestamp must be integers")
            if value < 0 or value >= np.iinfo(np.int64).max:
                raise ValueError("sequence or timestamp outside supported range")
        if sequence_id > self.wrapper.next_sequence_id:
            self.unusable = True
            raise ValueError("missing input; explicit reset or restore required")
        if sequence_id != self.wrapper.next_sequence_id:
            raise ValueError("duplicate or out-of-order sequence")
        if timestamp_ns != sequence_id * self.period_ns:
            self.unusable = True
            raise ValueError("cadence discontinuity; explicit reset or restore required")
        values = np.asarray(counts)
        if (
            values.shape != (176,)
            or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all()
            or (values < 0).any()
        ):
            raise ValueError("expected finite nonnegative counts with 176 channels")
        try:
            return self.wrapper.step(values, sequence_id=sequence_id, timestamp_ns=timestamp_ns)
        except Exception:
            # An unexpected backend exception may have partially advanced internal state.
            self.unusable = True
            raise

    def snapshot(self, path):
        if self.unusable:
            raise RuntimeError("cannot checkpoint unusable stream")
        path = Path(path)
        sidecar = path.with_suffix(path.suffix + ".json")
        if path.exists() or sidecar.exists():
            raise FileExistsError(path)
        self.wrapper.snapshot(path)
        metadata = {
            "schema_version": 1,
            "snapshot_sha256": sha256(path),
            "channel_signature": self.channel_signature,
            "period_ns": self.period_ns,
            "timestamp_origin_ns": 0,
        }
        with sidecar.open("x") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")

    def restore(self, path, *, expected_stream_id, expected_next_sequence_id, channel_ids):
        path = Path(path)
        metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
        expected = {
            "schema_version": 1,
            "snapshot_sha256": sha256(path),
            "channel_signature": channel_signature(channel_ids),
            "period_ns": self.period_ns,
            "timestamp_origin_ns": 0,
        }
        if metadata != expected or expected["channel_signature"] != self.channel_signature:
            raise ValueError("checkpoint metadata or byte identity mismatch")
        if (
            isinstance(expected_next_sequence_id, (bool, np.bool_))
            or not isinstance(expected_next_sequence_id, (int, np.integer))
            or expected_next_sequence_id < 0
        ):
            raise ValueError("expected next sequence must be a nonnegative integer")
        # Validate using a fresh candidate; failed restore cannot mutate the active decoder.
        candidate = CheckpointedONNXStream(copy.copy(self.decoder), stream_id=expected_stream_id)
        candidate.restore(
            path,
            expected_stream_id=expected_stream_id,
            expected_next_sequence_id=expected_next_sequence_id,
        )
        expected_timestamp = (
            -1
            if expected_next_sequence_id == 0
            else (expected_next_sequence_id - 1) * self.period_ns
        )
        if candidate.last_timestamp_ns != expected_timestamp:
            raise ValueError("snapshot timestamp does not match fixed cadence")
        self.wrapper = candidate
        self.unusable = False


def _reject(action, stream):
    before = stream.decoder.state_snapshot()
    sequence = stream.wrapper.next_sequence_id
    try:
        action()
    except (ValueError, RuntimeError, OSError):
        pass
    else:
        raise AssertionError("invalid lifecycle operation accepted")
    for a, b in zip(before, stream.decoder.state_snapshot(), strict=True):
        np.testing.assert_array_equal(a, b)
    assert stream.wrapper.next_sequence_id == sequence


def check_onnx_lifecycle(factory, counts, channels, output):
    """Known-answer full resume plus explicit invalid metadata/state rejection."""
    output.mkdir()
    counts = counts[:128]
    stream_id = "lifecycle/cycle-0/file-0"

    def fresh():
        return CadencedONNXStream(factory(), stream_id=stream_id, channel_ids=channels)

    def step(stream, i, values=None, **kw):
        args = {"sequence_id": i, "timestamp_ns": i * PERIOD_NS, "channel_ids": channels}
        args.update(kw)
        return stream.step(counts[i] if values is None else values, **args)

    uninterrupted = fresh()
    expected = np.stack([step(uninterrupted, i) for i in range(len(counts))])
    partial = fresh()
    split = 47
    for i in range(split):
        step(partial, i)
    checkpoint = output / "complete-47.npz"
    partial.snapshot(checkpoint)
    complete_hash = sha256(checkpoint)
    restored = fresh()
    restored.restore(
        checkpoint,
        expected_stream_id=stream_id,
        expected_next_sequence_id=split,
        channel_ids=channels,
    )
    actual = np.stack([step(restored, i) for i in range(split, len(counts))])
    np.testing.assert_array_equal(actual, expected[split:])
    for a, b in zip(
        restored.decoder.state_snapshot(), uninterrupted.decoder.state_snapshot(), strict=True
    ):
        np.testing.assert_array_equal(a, b)
    checks = {"same_file_resume_exact": True, "state_resume_exact": True}
    for name, values, extra in [
        ("channel_count", np.ones(175), {}),
        ("nonfinite", np.full(176, np.nan), {}),
        ("negative", -np.ones(176), {}),
        ("channel_order", counts[split], {"channel_ids": channels[::-1]}),
        ("duplicate", counts[split], {"sequence_id": split - 1}),
    ]:
        _reject(lambda: step(partial, split, values, **extra), partial)
        checks[name] = True
    for name, extra in [
        ("missing_input", {"sequence_id": split + 1}),
        ("timestamp_gap", {"timestamp_ns": (split + 1) * PERIOD_NS}),
    ]:
        _reject(lambda: step(partial, split, **extra), partial)
        assert partial.unusable
        _reject(lambda: step(partial, split), partial)
        partial.restore(
            checkpoint,
            expected_stream_id=stream_id,
            expected_next_sequence_id=split,
            channel_ids=channels,
        )
        checks[name] = True
    partial.discontinuity()
    _reject(lambda: step(partial, split), partial)
    partial.reset(stream_id="lifecycle/cycle-1/file-0")
    np.testing.assert_array_equal(step(partial, 0), expected[0])
    checks["explicit_discontinuity_reset"] = True
    # Mutate numeric identity fields while preserving container readability and sidecar hash.
    with np.load(checkpoint, allow_pickle=False) as archive:
        original = {key: archive[key] for key in archive.files}
    for field in ("model_identity", "graph_sha256", "precision", "stream_id"):
        values = {key: value.copy() for key, value in original.items()}
        values[field] = np.frombuffer(b"wrong-identity", dtype=np.uint8)
        corrupt = output / f"wrong-{field}.npz"
        np.savez(corrupt, **values)
        sidecar = json.loads(checkpoint.with_suffix(".npz.json").read_text())
        sidecar["snapshot_sha256"] = sha256(corrupt)
        write_json(corrupt.with_suffix(".npz.json"), sidecar)
        _reject(
            lambda: partial.restore(
                corrupt,
                expected_stream_id=stream_id,
                expected_next_sequence_id=split,
                channel_ids=channels,
            ),
            partial,
        )
        checks[f"reject_{field}"] = True
    truncated = output / "interrupted-later.npz"
    truncated.write_bytes(checkpoint.read_bytes()[:200])
    sidecar = json.loads(checkpoint.with_suffix(".npz.json").read_text())
    sidecar["snapshot_sha256"] = sha256(truncated)
    write_json(truncated.with_suffix(".npz.json"), sidecar)
    _reject(
        lambda: partial.restore(
            truncated,
            expected_stream_id=stream_id,
            expected_next_sequence_id=split,
            channel_ids=channels,
        ),
        partial,
    )
    assert sha256(checkpoint) == complete_hash
    checks["truncated_rejected_prior_complete_retained"] = True
    _reject(
        lambda: partial.restore(
            checkpoint,
            expected_stream_id=stream_id,
            expected_next_sequence_id=split + 1,
            channel_ids=channels,
        ),
        partial,
    )
    checks["wrong_next_sequence"] = True
    # A synthetic backend failure can occur after partial arithmetic mutation.
    failing = fresh()
    original_step = failing.decoder.step

    def fail(_counts):
        raise RuntimeError("injected arithmetic failure")

    failing.decoder.step = fail
    _reject(lambda: step(failing, 0), failing)
    assert failing.unusable
    failing.decoder.step = original_step
    _reject(lambda: step(failing, 0), failing)
    failing.reset(stream_id=stream_id)
    np.testing.assert_array_equal(step(failing, 0), expected[0])
    checks["arithmetic_failure_poisoned_until_reset"] = True
    np.save(output / "uninterrupted.npy", expected, allow_pickle=False)
    np.save(output / "restored-suffix.npy", actual, allow_pickle=False)
    return {
        "status": "passed",
        "checks": checks,
        "bins": len(counts),
        "split": split,
        "numeric_state_bytes": uninterrupted.decoder.state_nbytes,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "guarantee": "deterministic numeric resume only; no exactly-once delivery or atomic write",
    }


def check_numpy(frozen, counts):
    decoder = TCNStream(frozen)
    values = counts[:512]
    full = decoder.predict_chunk(values)
    state = decoder.state_snapshot()
    decoder.reset()
    chunks = np.concatenate(
        [
            decoder.predict_chunk(values[:11]),
            decoder.predict_chunk(values[11:127]),
            decoder.predict_chunk(values[127:]),
        ]
    )
    np.testing.assert_array_equal(full, chunks)
    current = decoder.state_snapshot()
    assert state[:2] == current[:2]
    np.testing.assert_array_equal(state[2], current[2])
    for a, b in zip(state[3], current[3], strict=True):
        np.testing.assert_array_equal(a, b)
    decoder.reset()
    np.testing.assert_array_equal(decoder.step(values[0]), full[0])
    return {
        "status": "passed",
        "chunk_exact": True,
        "state_exact": True,
        "reset_startup_exact": True,
        "bins": len(values),
        "numeric_state_bytes": decoder.state_nbytes,
    }


def overhead(factory, counts, channels, output):
    """Complete-step wrapper validation included; separate diagnostic from campaign."""
    rows = []
    n = 3001
    for repeat, order in enumerate((("bare", "adapter"), ("adapter", "bare"))):
        for name in order:
            decoder = factory()
            if name == "adapter":
                decoder = CadencedONNXStream(
                    decoder, stream_id="overhead/warmup", channel_ids=channels
                )

            def run(i, row):
                if name == "bare":
                    return decoder.step(row)
                return decoder.step(
                    row, sequence_id=i, timestamp_ns=i * PERIOD_NS, channel_ids=channels
                )

            for i, row in enumerate(counts[:500]):
                run(i, row)
            if name == "bare":
                decoder.reset()
            else:
                decoder.reset(stream_id=f"overhead/{repeat}")
            stamps = np.empty((n, 2), dtype=np.int64)
            prediction = np.empty((n, 7), dtype=np.float32)
            for i, row in enumerate(counts[:n]):
                stamps[i, 0] = time.perf_counter_ns()
                result = run(i, row)
                stamps[i, 1] = time.perf_counter_ns()
                prediction[i] = result
            path = output / f"overhead-{repeat}-{name}.npz"
            np.savez(path, timestamps=stamps, predictions=prediction)
            durations = (stamps[:, 1] - stamps[:, 0]) / 1e6
            rows.append(
                {
                    "repeat": repeat,
                    "path": path.name,
                    "arm": name,
                    "bins": n,
                    "service_ms": dict(
                        zip(
                            ("p50", "p95", "p99", "max"),
                            [float(v) for v in np.percentile(durations, [50, 95, 99, 100])],
                            strict=True,
                        )
                    ),
                }
            )
    return rows


def stall_check(decoder, counts, output):
    decoder.reset()
    n = 31
    stamps = np.empty((n, 5), dtype=np.int64)
    predictions = np.empty((n, 7), dtype=np.float32)
    origin = time.perf_counter_ns()
    for i in range(n):
        release = origin + i * PERIOD_NS
        while (remaining := release - time.perf_counter_ns()) > 0:
            time.sleep(remaining / 1e9)
        if i == 10:
            time.sleep(0.06)
        start = time.perf_counter_ns()
        prediction = decoder.step(counts[i])
        complete = time.perf_counter_ns()
        stamps[i] = release, start, complete, complete, release + PERIOD_NS
        predictions[i] = prediction
    np.savez(output / "injected-stall.npz", timestamps=stamps, predictions=predictions)
    misses = int(np.count_nonzero(stamps[:, 3] > stamps[:, 4]))
    assert misses >= 2
    return {
        "bins": n,
        "injected_before_sequence": 10,
        "requested_stall_ns": 60_000_000,
        "deadline_misses": misses,
        "maximum_backlog_bins": int(np.max((stamps[:, 1] - stamps[:, 0]) // PERIOD_NS)),
        "policy": "retain every observation and drain backlog; no input discontinuity",
    }


def audit_saved_traces(folder, lifecycle, overhead_rows, fault, counts, graph_path):
    """Direct ORT replay and separate timestamp arithmetic, executed once off-clock."""
    from neural_experiments.audit_h1_final_closure import graph_session, graph_trace

    if set(lifecycle["checks"]) != LIFECYCLE_CHECKS or not all(lifecycle["checks"].values()):
        raise ValueError("lifecycle required check inventory differs")
    if lifecycle["bins"] != 128 or lifecycle["split"] != 47:
        raise ValueError("lifecycle replay extent changed")
    engine = graph_session(graph_path)
    expected = graph_trace(engine, counts[:3001])
    np.testing.assert_array_equal(np.load(folder / "uninterrupted.npy"), expected[:128])
    np.testing.assert_array_equal(np.load(folder / "restored-suffix.npy"), expected[47:128])
    with np.load(folder / "complete-47.npz", allow_pickle=False) as checkpoint:
        states = [checkpoint[f"history{i}"].copy() for i in range(3)]
        if checkpoint["next_sequence_id"].item() != 47:
            raise ValueError("retained checkpoint is not the declared split")
    resumed = []
    for row in counts[47:128]:
        feed = {"counts": row.astype(np.float32)}
        feed.update({f"history{i}": state for i, state in enumerate(states)})
        prediction, *states = engine.run(None, feed)
        resumed.append(prediction)
    np.testing.assert_array_equal(np.asarray(resumed), expected[47:128])
    expected_order = [(0, "bare"), (0, "adapter"), (1, "adapter"), (1, "bare")]
    if [(r["repeat"], r["arm"]) for r in overhead_rows] != expected_order:
        raise ValueError("wrapper overhead counterbalanced order changed")
    for record in overhead_rows:
        if record["path"] != f"overhead-{record['repeat']}-{record['arm']}.npz":
            raise ValueError("wrapper timing artifact name changed")
        with np.load(folder / record["path"], allow_pickle=False) as values:
            stamps, predictions = values["timestamps"], values["predictions"]
        if stamps.shape != (3001, 2) or stamps.dtype.kind not in "iu":
            raise ValueError("invalid wrapper timing extent or integer timestamps")
        if np.any(stamps[:, 1] < stamps[:, 0]) or np.any(stamps[1:, 0] < stamps[:-1, 1]):
            raise ValueError("wrapper timestamp ordering violated")
        np.testing.assert_array_equal(predictions, expected)
        actual = np.percentile((stamps[:, 1] - stamps[:, 0]) / 1e6, [50, 95, 99, 100])
        np.testing.assert_allclose(list(record["service_ms"].values()), actual, rtol=0, atol=0)
        if record["bins"] != 3001:
            raise ValueError("wrapper overhead bin count changed")
    with np.load(folder / "injected-stall.npz", allow_pickle=False) as values:
        stamps, predictions = values["timestamps"], values["predictions"]
    if stamps.shape != (31, 5) or stamps.dtype.kind not in "iu":
        raise ValueError("invalid stall timestamp extent")
    release, start, ready, complete, deadline = stamps.T
    if (
        np.any(start < release)
        or np.any(ready < start)
        or np.any(complete < ready)
        or np.any(start[1:] < complete[:-1])
        or not np.array_equal(np.diff(release), np.full(30, PERIOD_NS))
        or not np.array_equal(deadline, release + PERIOD_NS)
    ):
        raise ValueError("stall schedule or event ordering changed")
    np.testing.assert_array_equal(predictions, expected[:31])
    if (
        fault["bins"] != 31
        or fault["injected_before_sequence"] != 10
        or fault["requested_stall_ns"] != 60_000_000
        or fault["deadline_misses"] != int(np.count_nonzero(complete > deadline))
        or fault["maximum_backlog_bins"] != int(((start - release) // PERIOD_NS).max())
        or fault["deadline_misses"] < 2
    ):
        raise ValueError("retained fault counts or declared injection changed")
    return {
        "status": "passed",
        "direct_ort_bins": 3001,
        "restored_bins": 81,
        "overhead_traces": 4,
        "stall_bins": 31,
    }


def validate_lifecycle_receipt(lifecycle_path, quality_run, quality_audit, root=ROOT):
    """Hash-only admission of the already independently checked lifecycle evidence."""
    from neural_experiments.audit_h1_final_closure import validate_audit_receipt
    from neural_experiments.h1_final_closure import artifact

    quality_receipt = validate_audit_receipt(quality_run, quality_audit, root)
    root, path = Path(root).resolve(), Path(lifecycle_path).resolve()
    receipt = json.loads(path.read_text())
    if (
        receipt.get("status") != "complete"
        or receipt.get("kind") != "h1_final_lifecycle"
        or receipt["quality_run"] != str(Path(quality_run).resolve())
        or receipt["quality_audit_sha256"] != sha256(quality_audit)
        or receipt["quality_manifest_sha256"] != sha256(Path(quality_run) / "manifest.json")
        or receipt["config_sha256"] != quality_receipt["config_sha256"]
        or receipt["statuses"]["independent_verification"] != "passed"
    ):
        raise ValueError("lifecycle receipt status or quality binding changed")
    actual = {
        str(p.relative_to(path.parent))
        for p in path.parent.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    if actual != set(receipt["artifacts"]):
        raise ValueError("lifecycle artifact inventory differs")
    for name, digest in receipt["artifacts"].items():
        artifact(path.parent, {"path": name, "sha256": digest})
    for name, digest in receipt["source"]["files"].items():
        artifact(path.parent / "source", {"path": name, "sha256": digest})
        committed = subprocess.check_output(
            ["git", "show", f"{receipt['source']['head']}:{name}"], cwd=root
        )
        if hashlib.sha256(committed).hexdigest() != digest:
            raise ValueError("lifecycle source snapshot differs from committed bytes")
    for name, digest in _execution_sources(receipt["source"]).items():
        artifact(root, {"path": name, "sha256": digest})
    if set(receipt["lifecycle"]) != {"B", "C", "E"}:
        raise ValueError("lifecycle arm inventory differs")
    expected = {arm: row.get("status") == "passed" for arm, row in receipt["lifecycle"].items()}
    if receipt["qualified_arms"] != expected:
        raise ValueError("lifecycle qualification inconsistent with checks")
    for arm, row in receipt["lifecycle"].items():
        if row["status"] == "passed":
            if not quality_receipt["qualified_arms"][arm]:
                raise ValueError("quality-ineligible lifecycle path cannot qualify")
            if arm in ("C", "E") and (
                set(row["checks"]) != LIFECYCLE_CHECKS or not all(row["checks"].values())
            ):
                raise ValueError("lifecycle required checks missing or failed")
            if arm == "B" and not all(
                row[key] for key in ("chunk_exact", "state_exact", "reset_startup_exact")
            ):
                raise ValueError("NumPy lifecycle check failed")
        elif row["status"] not in {"failed", "ineligible"}:
            raise ValueError("lifecycle execution state is not terminal")
    validation = json.loads((path.parent / "validation.json").read_text())
    if (
        validation["status"] != "passed"
        or validation["report_sha256"] != _report_digest(receipt)
        or validation["execution_source"] != _execution_sources(receipt["source"])
    ):
        raise ValueError("independent lifecycle validation report binding differs")
    expected_verified = {arm for arm in ("C", "E") if expected[arm]}
    if set(validation["arms"]) != expected_verified:
        raise ValueError("independent lifecycle validation omitted a qualified arm")
    if set(validation["artifacts"]) != {
        name for name in receipt["artifacts"] if Path(name).suffix in (".npy", ".npz")
    }:
        raise ValueError("independent lifecycle validation raw inventory differs")
    for name, digest in validation["artifacts"].items():
        if receipt["artifacts"].get(name) != digest:
            raise ValueError("independent lifecycle evidence identity changed")
    return receipt


def fault_check(config_path, quality_run, quality_audit, output, root=ROOT):
    from neural_experiments.h1_final_closure import require_committed
    from neural_experiments.h1_final_closure_runtime import admit_quality, make_decoder

    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root / "results/neural-h1"):
        raise ValueError("private lifecycle evidence must remain under results/neural-h1")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "kind": "h1_final_lifecycle",
        "status": "running",
        "command": sys.argv,
        "started_utc": datetime.now(UTC).isoformat(),
        "quality_run": str(Path(quality_run).resolve()),
        "statuses": {
            "implementation_execution": "running",
            "artifact_completion": "incomplete",
            "independent_verification": "not_run",
            "float32_equivalence": "not_run",
            "compression_quality": "not_run",
            "runtime_goals": "not_run",
        },
    }
    write_json(output / "start.json", manifest)
    write_json(output / "manifest.json", manifest)
    try:
        manifest.update(
            source=source_identity(root, output),
            environment=environment(),
            config_sha256=sha256(config_path),
        )
        (output / "configuration.json").write_bytes(Path(config_path).read_bytes())
        config, receipt, model = admit_quality(config_path, quality_run, quality_audit, root)
        eligible = [arm for arm, passed in receipt["qualified_arms"].items() if passed]
        if "B" not in eligible:
            raise ValueError("mandatory NumPy lifecycle requires qualified B")
        manifest.update(
            quality_audit_sha256=sha256(quality_audit),
            quality_manifest_sha256=sha256(Path(quality_run) / "manifest.json"),
        )
        manifest["statuses"].update(
            float32_equivalence="passed"
            if all(receipt["gates"]["float32_equivalence"].values())
            else "failed",
            compression_quality="passed" if receipt["gates"]["compression_quality"] else "failed",
        )
        require_committed(root, manifest["source"]["files"])
        index_path = root / config["inputs"]["development_index"]["path"]
        if sha256(index_path) != config["inputs"]["development_index"]["sha256"]:
            raise ValueError("lifecycle development index identity changed")
        index = checked_development_index(index_path)
        item = next(row for row in index["files"] if row["file_id"] == "S0_set_1")
        bundle = load_feature_file(index_path.parent / item["features"], item["features_sha256"])
        counts, channels = bundle["counts"], bundle["channel_ids"]
        if counts.shape != (item["n_bins"], 176) or not np.array_equal(
            channels, item["channel_ids"]
        ):
            raise ValueError("lifecycle feature shape/channel manifest mismatch")
        frozen = load_tcn(root / model["model"]["path"])
        torch.set_num_threads(1)
        lifecycle, overhead_rows, faults, validations = {}, {}, {}, {}
        manifest.update(lifecycle=lifecycle, overhead=overhead_rows, faults=faults)
        with threadpool_limits(limits=1):
            lifecycle["B"] = check_numpy(frozen, counts)
            for arm in ("C", "E"):
                if arm not in eligible:
                    lifecycle[arm] = {"status": "ineligible", "reason": "quality gate"}
                    continue
                graph_record = model["arms"][arm]["graph"]
                graph = Path(quality_run) / graph_record["path"]
                if sha256(graph) != graph_record["sha256"]:
                    raise ValueError("lifecycle graph changed after quality admission")

                def factory(arm=arm, graph=graph):
                    return make_decoder(arm, root / model["model"]["path"], graph)

                folder = output / arm
                try:
                    lifecycle[arm] = check_onnx_lifecycle(factory, counts, channels, folder)
                    overhead_rows[arm] = overhead(factory, counts, channels, folder)
                    faults[arm] = stall_check(factory(), counts, folder)
                    validations[arm] = audit_saved_traces(
                        folder, lifecycle[arm], overhead_rows[arm], faults[arm], counts, graph
                    )
                except Exception as error:
                    lifecycle[arm] = {
                        "status": "failed",
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                        "partial_checks": lifecycle.get(arm),
                    }
        manifest["qualified_arms"] = {
            arm: row["status"] == "passed" for arm, row in lifecycle.items()
        }
        verification = {
            "status": "passed",
            "report_sha256": _report_digest(manifest),
            "execution_source": _execution_sources(manifest["source"]),
            "arms": validations,
            "artifacts": {
                str(path.relative_to(output)): sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.suffix in (".npy", ".npz")
            },
            "method": (
                "direct ORT whole and checkpoint feeds; separate integer timestamp arithmetic"
            ),
        }
        write_json(output / "validation.json", verification)
        manifest["status"] = "complete"
        manifest["statuses"].update(
            implementation_execution="partial_failure"
            if any(row["status"] == "failed" for row in lifecycle.values())
            else "complete",
            artifact_completion="complete",
            independent_verification="passed",
        )
    except BaseException as error:
        manifest.update(status="failed", error=str(error), traceback=traceback.format_exc())
        manifest["statuses"]["implementation_execution"] = "failed"
        raise
    finally:
        manifest["finished_utc"] = datetime.now(UTC).isoformat()
        manifest["artifacts"] = {
            str(path.relative_to(output)): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fault-check"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--quality-run", type=Path, required=True)
    parser.add_argument("--quality-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fault_check(args.config, args.quality_run, args.quality_audit, args.output)
    print(json.dumps({"status": result["status"], "qualified_arms": result["qualified_arms"]}))


if __name__ == "__main__":
    main()
