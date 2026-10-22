"""Evaluator-only H1 regression metrics with audited FALCON grouping semantics.

The predictor must never receive these records: they contain behavioral targets
and scoring masks. Public inputs to :func:`score_files` retain every time bin;
only :func:`assemble_official_regression` converts targets to the already-masked
form required by the pinned upstream helper.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

FALCON_SOURCE_PIN = "770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f"
FALCON_EVALUATOR_SHA256 = "2b848f84ef620eac82ce53d331d03727d9ae4ea9a5f2e5edb6e6557537ce4418"


@dataclass(frozen=True)
class FileEvaluation:
    """One complete released recording; arrays have shapes (T, D), (T, D), (T,).

    ``file_id`` uniquely identifies a recording, and ``day`` is its audited S-day
    identity. Both must come from the data manifest. A mask selects whole rows.
    Trial boundaries do not split a day score; file boundaries are retained for
    accounting even though the day's scored rows are concatenated for R².
    """

    file_id: str
    day: str
    predictions: np.ndarray
    targets: np.ndarray
    eval_mask: np.ndarray


@dataclass(frozen=True)
class OfficialRegressionInputs:
    """Canonical inputs for the pinned ``compute_metrics_regression`` helper."""

    predictions: np.ndarray
    masked_targets: np.ndarray
    eval_mask: np.ndarray
    dset_lens: dict[str, list[int]]
    file_order: tuple[str, ...]


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 2 or not all(value.shape) or value.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a nonempty two-dimensional real numeric array")
    return value.astype(np.float64, copy=False)


def _validated(records: Iterable[FileEvaluation]) -> list[FileEvaluation]:
    checked = []
    file_ids: set[str] = set()
    output_dim = None
    for record in records:
        if not isinstance(record, FileEvaluation):
            raise ValueError("records must contain FileEvaluation instances")
        if not isinstance(record.file_id, str) or not record.file_id.strip():
            raise ValueError("file_id must be a nonempty string")
        if record.file_id in file_ids:
            raise ValueError(f"duplicate file_id: {record.file_id}")
        if not isinstance(record.day, str) or not re.fullmatch(r"S(?:0|[1-9]\d*)", record.day):
            raise ValueError("day must be a canonical S-day identifier, for example S6")
        predictions = _matrix(record.predictions, "predictions")
        targets = _matrix(record.targets, "targets")
        mask = np.asarray(record.eval_mask)
        if targets.shape != predictions.shape:
            raise ValueError("full targets and predictions must have identical shapes")
        if mask.dtype.kind != "b" or mask.shape != (len(predictions),):
            raise ValueError("eval_mask must be a boolean row mask with one entry per full bin")
        if not np.isfinite(predictions).all():
            raise ValueError("predictions must be finite, including unscored bins")
        if not np.isfinite(targets[mask]).all():
            raise ValueError("scored targets must be finite")
        if output_dim is not None and predictions.shape[1] != output_dim:
            raise ValueError("every file must use the same output dimension and order")
        output_dim = predictions.shape[1]
        file_ids.add(record.file_id)
        checked.append(FileEvaluation(record.file_id, record.day, predictions, targets, mask))
    if not checked:
        raise ValueError("at least one file is required")
    return sorted(checked, key=lambda record: (record.day, record.file_id))


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    """Direct float64 R² for already-selected rows, matching sklearn's default.

    An exactly constant output receives 1 for perfect predictions and 0 otherwise.
    It gets zero weight when any output varies. When every output is constant,
    the aggregate is the uniform mean of those finite per-output scores. Negative
    scores remain negative. Fewer than two scored rows fail explicitly rather than
    propagating upstream NaN into a purportedly completed report.
    """
    targets = _matrix(targets, "targets")
    predictions = _matrix(predictions, "predictions")
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have identical shapes")
    if len(targets) < 2:
        raise ValueError("a day requires at least two scored rows for R2")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("scored targets and predictions must be finite")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            sse = np.square(targets - predictions).sum(axis=0)
            sst = np.square(targets - targets.mean(axis=0)).sum(axis=0)
            varying = sst != 0
            per_output = np.ones(targets.shape[1], dtype=np.float64)
            per_output[varying] = 1.0 - sse[varying] / sst[varying]
            per_output[~varying & (sse != 0)] = 0.0
            aggregate = np.average(per_output, weights=sst) if varying.any() else per_output.mean()
        except FloatingPointError as exc:
            raise ValueError("R2 arithmetic overflowed; inspect target/prediction scaling") from exc
    return {
        "r2_variance_weighted": float(aggregate),
        "r2_per_output": per_output.tolist(),
        "sse_per_output": sse.tolist(),
        "sst_per_output": sst.tolist(),
        "zero_variance_outputs": np.flatnonzero(~varying).tolist(),
        "num_scored_bins": len(targets),
    }


def assemble_official_regression(records: Iterable[FileEvaluation]) -> OfficialRegressionInputs:
    """Assemble lexical day/file order, full predictions, and *masked* targets.

    Upstream sorts day keys but concatenates arrays in submission insertion order.
    Sorting all three arrays together here avoids the unequal-length S6/S10 offset
    bug. This assembly order has no bearing on chronological predictor replay.
    """
    records = _validated(records)
    dset_lens: dict[str, list[int]] = {}
    for record in records:
        dset_lens.setdefault(record.day, []).append(len(record.predictions))
    return OfficialRegressionInputs(
        predictions=np.concatenate([record.predictions for record in records]),
        masked_targets=np.concatenate([record.targets[record.eval_mask] for record in records]),
        eval_mask=np.concatenate([record.eval_mask for record in records]),
        dset_lens=dset_lens,
        file_order=tuple(record.file_id for record in records),
    )


def score_files(records: Iterable[FileEvaluation], *, bin_seconds: float = 0.02) -> dict:
    """Score every supplied day, then take an equally weighted mean over days.

    The returned dictionary is strict-JSON compatible. Duration fields count bins
    at the declared nominal interval; they do not claim elapsed acquisition time
    across timestamp gaps. No day is silently dropped, weighted by its length, or
    clipped to zero. This is a local scoring calculation, not an official result.
    """
    records = _validated(records)
    if not np.isfinite(bin_seconds) or bin_seconds <= 0:
        raise ValueError("bin_seconds must be finite and positive")
    grouped: dict[str, list[FileEvaluation]] = {}
    for record in records:
        grouped.setdefault(record.day, []).append(record)
    per_day = {}
    for day, day_files in grouped.items():
        targets = np.concatenate([record.targets[record.eval_mask] for record in day_files])
        predictions = np.concatenate([record.predictions[record.eval_mask] for record in day_files])
        try:
            metrics = regression_metrics(targets, predictions)
        except ValueError as exc:
            raise ValueError(f"{day}: {exc}") from exc
        num_bins = sum(len(record.predictions) for record in day_files)
        metrics.update(
            {
                "num_bins": num_bins,
                "num_unscored_bins": num_bins - len(targets),
                "num_files": len(day_files),
                "file_ids": [record.file_id for record in day_files],
                "nominal_duration_seconds": float(num_bins * bin_seconds),
                "scored_duration_seconds": float(len(targets) * bin_seconds),
            }
        )
        per_day[day] = metrics
    day_scores = np.array([value["r2_variance_weighted"] for value in per_day.values()])
    worst_day = min(per_day, key=lambda day: per_day[day]["r2_variance_weighted"])
    num_bins = sum(value["num_bins"] for value in per_day.values())
    num_scored_bins = sum(value["num_scored_bins"] for value in per_day.values())
    return {
        "schema_version": 1,
        "metric": "mean_day_variance_weighted_r2",
        "evaluation_kind": "local_evaluation",
        "upstream_pin": FALCON_SOURCE_PIN,
        "canonical_day_order": list(per_day),
        "chronological_day_order": sorted(per_day, key=lambda day: int(day[1:])),
        "num_days": len(per_day),
        "num_files": len(records),
        "num_outputs": records[0].predictions.shape[1],
        "num_bins": num_bins,
        "num_scored_bins": num_scored_bins,
        "num_unscored_bins": num_bins - num_scored_bins,
        "nominal_bin_seconds": float(bin_seconds),
        "nominal_duration_seconds": float(num_bins * bin_seconds),
        "scored_duration_seconds": float(num_scored_bins * bin_seconds),
        "r2_mean": float(day_scores.mean()),
        "r2_population_std": float(day_scores.std(ddof=0)),
        "r2_worst_day": float(day_scores.min()),
        "worst_day": worst_day,
        "per_day": per_day,
    }
