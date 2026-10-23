"""Post-development scalar-gain oracle diagnostic, never a deployable result.

Uses held-out target dot products to choose a separate best gain per session.
This upper-bounds only scaling an existing prediction by a scalar in [0,1].
It is not a bound on information in the original neural recordings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_experiments.v3.study import digest, write_json


def best_scalar(error):
    power = error["prediction_squared_norm"]
    dot = error["target_prediction_dot"]
    variance = error["target_total_variance"]
    gain = min(1.0, max(0.0, dot / power)) if power > 0 else 0.0
    mse = error["target_squared_norm"] + gain * gain * power - 2 * gain * dot
    return {
        "oracle_gain": gain,
        "oracle_summed_mse": mse,
        "oracle_r2": 1 - mse / variance if variance > 1e-12 else None,
    }


def run(path, output):
    path = Path(path)
    report = json.loads(path.read_text())
    result = {
        "study": "V8",
        "phase": report["phase"],
        "error_report_sha256": digest(path),
        "code_sha256": digest(__file__),
        "evidence_role": (
            "Post-development diagnostic; per-session held-out-target oracle, "
            "not deployable accuracy"
        ),
        "scope": (
            "Upper bound only for scalar multiplication of each existing output in [0,1]; "
            "not a neural-information bound"
        ),
        "rows": [
            {
                **{k: r[k] for k in ("participant", "session_id", "method")},
                **best_scalar(r["primary"]),
            }
            for r in report["rows"]
            if r["primary"] is not None
        ],
    }
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.errors, args.output)
    print(json.dumps({"rows": len(result["rows"])}))
