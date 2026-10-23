import copy
import json

import pytest

from neural_experiments.v7 import runtime_public as public

HASH = "a" * 64


def fixture():
    blocks = []
    for mode, repeats, n in (("fast", 3, 30000), ("paced", 2, 3000)):
        for repeat in range(repeats):
            block = {k: 0.1 for k in public.BLOCK_NUMBERS}
            block.update(
                mode=mode,
                repeat=repeat,
                n=n,
                deadline_misses=0,
                negative_queue_count=0,
                predictions=["SENSITIVE"],
                path="/Users/SENSITIVE/block.npz",
                timestamps_ns=[[1, 2, 3]],
            )
            blocks.append(block)
    summary = {key: 1 for key in public.REFIT_NUMBERS}
    manifest = {
        "status": "complete",
        "kind": "V7B_source_runtime",
        "plan_sha256": HASH,
        "source_manifest_sha256": HASH,
        "source_audit_sha256": HASH,
        "artifacts": {
            key: {"sha256": HASH, "path": "/Users/SENSITIVE/input"} for key in ("model", "features")
        },
        "source_sha256": {"neural_experiments/v7/runtime_continual.py": HASH},
        "representative": {
            "participant": "T2",
            "post_implant_day": 250,
            "method": "periodic240",
            "fit_stop_seconds": 540,
            "deployment_seconds": 542,
            "training_rows": 9493,
            "weights": ["SENSITIVE"],
        },
        "environment": {
            "python": "3.12",
            "platform": "macOS-arm64",
            "numpy": "2.5",
            "scipy": "1.18",
            "threadpools": [{"num_threads": 1, "path": "/Users/SENSITIVE/library"}],
            "secret": "SENSITIVE",
        },
        "blocks": blocks,
        "refit_summary": summary,
        "model_load_ms": 0.7,
        "numeric_model_array_bytes": 6160,
        "numpy_incremental_vs_batch_max_error": 1e-14,
        "post_deployment_vs_source_trace_max_error": 0.0,
        "protocol": {
            k: 1
            for k in (
                "fast_repeats",
                "paced_repeats",
                "paced_seconds_each",
                "refit_repeats",
                "period_ms",
                "warmup_bins",
                "threads",
                "parity_atol",
                "parity_rtol",
                "refit_allowance_seconds",
            )
        },
        "secret": "SENSITIVE",
    }
    audit = {key: HASH for key in public.AUDIT_HASHES}
    audit.update(
        status="PASS",
        blocks_checked=5,
        refits_checked=10,
        blocks=copy.deepcopy(blocks),
        refit_summary=copy.deepcopy(summary),
        refit_max_numeric_error=0,
        post_deployment_source_trace_max_error=1e-14,
        timed_prediction_max_error=1e-14,
        secret="SENSITIVE",
    )
    return manifest, audit


def call(manifest, audit):
    return public.aggregate(
        manifest, audit, manifest_sha256=HASH, audit_sha256=HASH, annotation_sha256=HASH
    )


def test_allowlist_drops_private_fields_at_every_level():
    result = call(*fixture())
    text = json.dumps(result)
    assert "SENSITIVE" not in text
    assert '"path"' not in text and '"timestamps_ns"' not in text
    assert '"predictions"' not in text and '"weights"' not in text
    assert result["audit"]["status"] == "PASS"
    assert len(result["blocks"]) == 5
    assert result["blocks"][3]["n"] == 3000
    assert "outside numerical audit" in result["annotation"]["evidence_role"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "FAIL"),
        ("manifest_sha256", "b" * 64),
        ("model_sha256", "b" * 64),
        ("blocks_checked", 4),
    ],
)
def test_rejects_failed_or_mismatched_audit(key, value):
    manifest, audit = fixture()
    audit[key] = value
    with pytest.raises(ValueError):
        call(manifest, audit)


def test_rejects_timing_mismatch_and_private_array_in_allowlisted_field():
    manifest, audit = fixture()
    audit["blocks"][0]["service_p99_ms"] = 99
    with pytest.raises(ValueError, match="timing"):
        call(manifest, audit)
    manifest, audit = fixture()
    manifest["numeric_model_array_bytes"] = [1, 2, 3]
    with pytest.raises(ValueError, match="aggregate"):
        call(manifest, audit)
    manifest, audit = fixture()
    manifest["environment"]["platform"] = "/Users/SENSITIVE"
    with pytest.raises(ValueError, match="environment"):
        call(manifest, audit)


def test_publish_binds_exact_inputs_and_never_overwrites(tmp_path):
    manifest, audit = fixture()
    source, receipt, note = [tmp_path / p for p in ("manifest.json", "audit.json", "note.md")]
    source.write_text(json.dumps(manifest))
    audit["manifest_sha256"] = public.digest(source)
    receipt.write_text(json.dumps(audit))
    note.write_text("Runtime operational annotation.")
    output = tmp_path / "public.json"
    result = public.publish(source, receipt, note, output)
    assert result["audit"]["receipt_sha256"] == public.digest(receipt)
    assert result["annotation"]["report_sha256"] == public.digest(note)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        public.publish(source, receipt, note, output)
    assert output.read_bytes() == before
