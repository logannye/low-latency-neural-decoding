"""Reproduce exposed T2 pilots or publish their original aggregate receipts.

Published numerical helpers preserve the original pilot calculations; their
input/output plumbing and formatting differ. Original execution hashes remain
distinct from the published reproducer hashes. No V5 fresh inputs are accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = {
    "t2_pilot.py": "f05d458dc92b89ddd10fb3efa21d84406f6f1c40e367dfb6e0b1d7ffeb385692",
    "t2_pilot.json": "979409e84ca0a0370b053877a3b9596e628aeebbc69079d271a0c7926ac537fb",
    "t2_followup.py": "174b8bf9cc579677bebae2a25436fd2015d8417644707a983aeb3978c9d6166a",
    "t2_followup.json": "027e77d48d00dfed68e7ed4eb196fffe7dc3e26877c0e59aafda3225752c33b4",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(original_directory, output, reproduced_directory=None):
    original_directory, output = Path(original_directory), Path(output)
    if output.exists():
        raise ValueError("Preserve prior diagnostic publications")
    for name, expected in ORIGINAL.items():
        if digest(original_directory / name) != expected:
            raise ValueError("Original diagnostic source or receipt identity differs")
    pilot = json.loads((original_directory / "t2_pilot.json").read_text())
    followup = json.loads((original_directory / "t2_followup.json").read_text())
    if (
        pilot["source_sha256"] != ORIGINAL["t2_pilot.py"]
        or followup["source_sha256"] != ORIGINAL["t2_followup.py"]
        or followup["helper_source_sha256"] != ORIGINAL["t2_pilot.py"]
        or followup["parent_pilot_sha256"] != ORIGINAL["t2_pilot.json"]
        or len(pilot["inputs"]) != 12
        or any(row["n_bins"] > 30000 for row in pilot["inputs"])
        or len(pilot["pilots"]) != 1728
        or len(pilot["prefix_choices"]) != 288
        or len(followup["prefix_validation"]) != 144
    ):
        raise ValueError("Original diagnostic inventory or bindings differ")
    # Only coefficient summaries are public; remove the actual two-output biases.
    pilot["history_weights"] = [
        {
            key: row[key]
            for key in (
                "session_id",
                "family",
                "recency",
                "history_id",
                "age_days",
                "cosine_to_pooled",
                "tx_weight_norm",
                "sbp_weight_norm",
            )
        }
        for row in pilot["history_weights"]
    ]
    result = {
        "schema_version": 1,
        "status": "original_exposed_diagnostic_aggregates_published",
        "scope": (
            "Post-hoc T2 first600s analysis after V4 outcomes; no fresh V5 claim. "
            "All negative arms retained. No participant-specific deployed rule."
        ),
        "original_source_sha256": {k: v for k, v in ORIGINAL.items() if k.endswith(".py")},
        "original_receipt_sha256": {k: v for k, v in ORIGINAL.items() if k.endswith(".json")},
        "published_reproducer_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in ("diagnosis.py", "diagnosis_pilot.py", "diagnosis_followup.py")
        },
        "reproduction_status": (
            "Published scripts are formatting/I-O refactors of the original executed math. "
            "Original receipt source hashes are preserved; no new fits were required for export."
        ),
        "original_trace_score_reconstruction_max_absolute_error": 3.3306690738754696e-16,
        "pilot": pilot,
        "followup": followup,
        "data_policy": (
            "Aggregate metrics, coverage counts, coefficient norms/cosines and artifact hashes; "
            "no coefficient values, per-bin labels, predictions or neural arrays."
        ),
    }
    if reproduced_directory is not None:
        result["verified_reproduction"] = verify_reproduction(
            original_directory, reproduced_directory
        )
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if any(token in encoded for token in ("/Users/", "/home/", "X-Amz-", '"intercept"')):
        raise ValueError("Private path, credential or model coefficients in publication")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(encoded)
    return result


def reproduce(output_directory):
    from threadpoolctl import threadpool_limits

    from neural_experiments.v5 import diagnosis_followup, diagnosis_pilot

    output_directory = Path(output_directory)
    pilot, followup = output_directory / "t2_pilot.json", output_directory / "t2_followup.json"
    if pilot.exists() or followup.exists():
        raise ValueError("Reproduction needs fresh receipt paths")
    with threadpool_limits(limits=1):
        diagnosis_pilot.main(pilot)
        diagnosis_followup.main(followup, pilot)


def verify_reproduction(original_directory, reproduced_directory):
    """Compare every scientific field; exclude only named execution identities/timing."""
    original_directory, reproduced_directory = Path(original_directory), Path(reproduced_directory)
    ignored = {
        "t2_pilot.json": ("elapsed_seconds", "source_sha256"),
        "t2_followup.json": (
            "elapsed_seconds",
            "source_sha256",
            "helper_source_sha256",
            "parent_pilot_sha256",
        ),
    }
    for name, keys in ignored.items():
        original = json.loads((original_directory / name).read_text())
        reproduced = json.loads((reproduced_directory / name).read_text())
        if digest(original_directory / name) != ORIGINAL[name]:
            raise ValueError("Original receipt identity differs")
        if {k: v for k, v in original.items() if k not in keys} != {
            k: v for k, v in reproduced.items() if k not in keys
        }:
            raise ValueError("Reproduced scientific fields differ")
    pilot = json.loads((reproduced_directory / "t2_pilot.json").read_text())
    followup = json.loads((reproduced_directory / "t2_followup.json").read_text())
    if (
        pilot["source_sha256"] != digest(Path(__file__).with_name("diagnosis_pilot.py"))
        or followup["source_sha256"] != digest(Path(__file__).with_name("diagnosis_followup.py"))
        or followup["helper_source_sha256"] != pilot["source_sha256"]
        or followup["parent_pilot_sha256"] != digest(reproduced_directory / "t2_pilot.json")
    ):
        raise ValueError("Reproduction source or parent identity differs")
    return {
        "status": "passed",
        "comparison": "Exact parsed equality of every scientific field, with named exclusions only",
        "excluded_top_level_fields": ignored,
        "reproduced_receipt_sha256": {
            name: digest(reproduced_directory / name) for name in ignored
        },
        "published_pilot_source_sha256": pilot["source_sha256"],
        "published_followup_source_sha256": followup["source_sha256"],
        "verification_source_sha256": digest(__file__),
        "original_receipt_sha256": {name: ORIGINAL[name] for name in ignored},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("reproduce", "publish", "verify"))
    parser.add_argument("--original-directory", type=Path, default=ROOT / "work/v5-diagnosis")
    parser.add_argument(
        "--output-directory", type=Path, default=ROOT / "results/neural_v5/diagnosis"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "neural_experiments/results/v5-diagnosis.json"
    )
    parser.add_argument("--verified-reproduction", type=Path)
    args = parser.parse_args()
    if args.command == "publish":
        publish(args.original_directory, args.output, args.verified_reproduction)
    elif args.command == "reproduce":
        reproduce(args.output_directory)
    else:
        result = verify_reproduction(args.original_directory, args.output_directory)
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
