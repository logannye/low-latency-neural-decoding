"""Known answers plus independently executed, pinned official scoring code.

The fixture is byte-identical upstream MIT source. AST selection executes only
its scorer, maps, and grouping function, avoiding unrelated NWB/torch/hydra
imports. Tests never deserialize externally supplied files: the official wrapper
receives only pickles written here from deterministic synthetic arrays.
"""

import ast
import hashlib
import json
import logging
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import r2_score

from neural_decoder.metrics import (
    FALCON_EVALUATOR_SHA256,
    FileEvaluation,
    assemble_official_regression,
    regression_metrics,
    score_files,
)


@pytest.fixture(scope="module")
def official_scorer():
    source = Path(__file__).with_name("fixtures") / "falcon_evaluator_770c9b9.py.txt"
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FALCON_EVALUATOR_SHA256
    nodes = []
    for node in ast.parse(raw).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id in {"DATASET_HELDINOUT_MAP", "HELDIN_OR_OUT_MAP"}
            for target in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {"reduce_key", "evaluate"}:
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "FalconEvaluator":
            node.body = [
                method
                for method in node.body
                if isinstance(method, ast.FunctionDef)
                and method.name == "compute_metrics_regression"
            ]
            nodes.append(node)
    scope = {
        "np": np,
        "r2_score": r2_score,
        "defaultdict": defaultdict,
        "os": os,
        "pickle": pickle,
        "logger": logging.getLogger("official_scorer_fixture"),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return scope


def _record(file_id, day, targets, predictions=None, mask=None):
    targets = np.asarray(targets, dtype=np.float64)
    if predictions is None:
        predictions = targets.copy()
    if mask is None:
        mask = np.ones(len(targets), dtype=bool)
    return FileEvaluation(file_id, day, np.asarray(predictions), targets, np.asarray(mask))


def test_known_answer_groups_files_before_scoring_and_weights_days_equally():
    first = np.array([[0, 1], [1, 3]], dtype=float)
    second = np.array([[2, 5], [3, 7]], dtype=float)
    records = [
        _record("S2_set_2", "S2", second, second + 1),
        _record("S3_set_1", "S3", [[0, 0], [2, 4]], [[0, 0], [0, 0]]),
        _record("S2_set_1", "S2", first, first + 1),
    ]
    result = score_files(records)
    assert result["per_day"]["S2"]["r2_per_output"] == pytest.approx([0.2, 0.8])
    assert result["per_day"]["S2"]["r2_variance_weighted"] == pytest.approx(0.68)
    assert result["per_day"]["S3"]["r2_variance_weighted"] == -1.0
    assert result["r2_mean"] == pytest.approx(-0.16)
    assert result["r2_population_std"] == pytest.approx(0.84)
    assert result["worst_day"] == "S3"
    assert result["r2_worst_day"] == -1.0
    assert result["num_files"] == 3
    assert result["num_bins"] == result["num_scored_bins"] == 6
    assert result["nominal_duration_seconds"] == pytest.approx(0.12)
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_mask_is_applied_once_to_full_arrays_and_ignores_unscored_targets():
    record = _record(
        "S2_set_1",
        "S2",
        [[0, 0], [np.nan, np.inf], [2, 4], [999, -999]],
        [[0, 0], [-100, 100], [2, 4], [100, -100]],
        [True, False, True, False],
    )
    result = score_files([record])
    assembled = assemble_official_regression([record])
    assert assembled.predictions.shape == (4, 2)
    assert assembled.masked_targets.tolist() == [[0, 0], [2, 4]]
    assert assembled.dset_lens == {"S2": [4]}
    assert result["r2_mean"] == 1.0
    assert result["num_scored_bins"] == result["num_unscored_bins"] == 2
    with pytest.raises(ValueError, match="full targets"):
        score_files(
            [
                FileEvaluation(
                    record.file_id,
                    record.day,
                    record.predictions,
                    record.targets[record.eval_mask],
                    record.eval_mask,
                )
            ]
        )


@pytest.mark.parametrize("all_constant", [False, True])
def test_zero_variance_outputs_match_pinned_helper(official_scorer, all_constant):
    targets = np.array([[0, 7, 8], [1, 7, 8], [2, 7, 8]], dtype=float)
    if all_constant:
        targets[:, 0] = 0
    predictions = targets.copy()
    predictions[:, 2] += 1000
    predictions[:, 0] += 1
    result = regression_metrics(targets, predictions)
    expected = r2_score(targets, predictions, multioutput="raw_values")
    assert result["r2_per_output"] == pytest.approx(expected)
    if all_constant:
        assert result["r2_variance_weighted"] == pytest.approx(1 / 3)
        assert result["zero_variance_outputs"] == [0, 1, 2]
    else:
        assert result["r2_variance_weighted"] == pytest.approx(-0.5)
        assert result["zero_variance_outputs"] == [1, 2]
    official = official_scorer["FalconEvaluator"].compute_metrics_regression(
        predictions,
        targets,
        np.ones(3, dtype=bool),
        {"S2": [3]},
    )
    assert result["r2_variance_weighted"] == pytest.approx(official["R2 Mean"])


def _synthetic_held_out_files():
    """All 14 IDs satisfy the wrapper; S6/S10 intentionally have unequal lengths."""
    rng = np.random.default_rng(918)
    records = []
    for day_number in range(6, 13):
        for set_number in (1, 2):
            length = 2 * day_number + set_number
            targets = rng.normal(size=(length, 7)) * (day_number - 5)
            predictions = targets + (day_number - 5) ** 2 / 20
            mask = np.arange(length) % 3 != 1
            records.append(
                _record(
                    f"S{day_number}_set_{set_number}",
                    f"S{day_number}",
                    targets,
                    predictions,
                    mask,
                )
            )
    return records


def test_full_official_evaluate_wrapper_parity_and_order_trap(official_scorer, tmp_path):
    records = _synthetic_held_out_files()
    result = score_files(reversed(records))
    assembly = assemble_official_regression(records)
    assert result["canonical_day_order"] == ["S10", "S11", "S12", "S6", "S7", "S8", "S9"]
    assert result["chronological_day_order"] == [f"S{day}" for day in range(6, 13)]
    assert assembly.file_order[0] == "S10_set_1"
    assert len(records[0].predictions) != len(records[8].predictions)
    by_id = {record.file_id: record for record in records}
    annotations = {
        "h1": {
            file_id: {"data": record.targets[record.eval_mask], "mask": record.eval_mask}
            for file_id, record in by_id.items()
        }
    }
    submission = {"h1": {"normalized_latency": 0.0}}
    submission["h1"].update(
        {file_id: by_id[file_id].predictions for file_id in assembly.file_order}
    )
    annotation_path = tmp_path / "synthetic_annotations.pkl"
    prediction_path = tmp_path / "synthetic_predictions.pkl"
    annotation_path.write_bytes(pickle.dumps(annotations))
    prediction_path.write_bytes(pickle.dumps(submission))
    official = official_scorer["evaluate"](str(annotation_path), str(prediction_path), "test")
    official = official["result"][0]["test_split_h1"]
    assert result["r2_mean"] == pytest.approx(official["Held Out R2 Mean"], abs=1e-12)
    assert result["r2_population_std"] == pytest.approx(official["Held Out R2 Std."], abs=1e-12)
    # Prove the fixture detects chronological concatenation with lexical boundaries.
    wrong = official_scorer["FalconEvaluator"].compute_metrics_regression(
        np.concatenate([record.predictions for record in records]),
        np.concatenate([record.targets[record.eval_mask] for record in records]),
        np.concatenate([record.eval_mask for record in records]),
        assembly.dset_lens,
    )
    assert abs(wrong["R2 Mean"] - result["r2_mean"]) > 0.001


def test_official_assembly_matches_direct_individual_day_scores(official_scorer):
    records = _synthetic_held_out_files()
    direct = score_files(records)
    for day in direct["canonical_day_order"]:
        assembly = assemble_official_regression(record for record in records if record.day == day)
        actual = official_scorer["FalconEvaluator"].compute_metrics_regression(
            assembly.predictions,
            assembly.masked_targets,
            assembly.eval_mask,
            assembly.dset_lens,
        )
        assert actual["R2 Mean"] == pytest.approx(direct["per_day"][day]["r2_variance_weighted"])


def test_file_and_mask_counts_include_a_fully_unscored_file():
    records = [
        _record("S2_set_1", "S2", [[0], [2]]),
        _record("S2_set_2", "S2", [[5], [6], [7]], mask=[False, False, False]),
    ]
    result = score_files(records)
    assert result["num_files"] == 2
    assert result["num_bins"] == 5
    assert result["num_scored_bins"] == 2
    assert result["r2_mean"] == 1


@pytest.mark.parametrize(
    "records, message",
    [
        ([], "at least one"),
        ([_record("a", "S2", [[0], [1]], mask=[1, 1])], "boolean row mask"),
        ([_record("a", "S02", [[0], [1]])], "canonical S-day"),
        ([_record("a", "S2", [[0], [1]], [[0], [np.inf]])], "predictions must be finite"),
        ([_record("a", "S2", [[0], [np.nan]], [[0], [0]])], "scored targets"),
        ([_record("a", "S2", [[0], [1]], mask=[True, False])], "at least two"),
        ([_record("a", "S2", [[0], [1]], mask=[False, False])], "nonempty"),
        ([_record("a", "S2", [[0], [1]]), _record("a", "S3", [[0], [1]])], "duplicate"),
        ([_record("a", "S2", [[0], [1]]), _record("b", "S3", [[0, 1], [1, 2]])], "output"),
    ],
)
def test_invalid_inputs_fail_without_omitting_a_day(records, message):
    with pytest.raises(ValueError, match=message):
        score_files(records)


def test_unscored_prediction_failure_is_not_hidden_by_mask():
    record = _record("a", "S2", [[0], [1], [2]], [[0], [np.inf], [2]], [True, False, True])
    with pytest.raises(ValueError, match="including unscored"):
        score_files([record])


def test_large_unrepresentable_errors_fail_instead_of_returning_infinite_metrics():
    with pytest.raises(ValueError, match="overflowed"):
        regression_metrics(np.array([[1e308], [-1e308]]), np.array([[0.0], [0.0]]))
