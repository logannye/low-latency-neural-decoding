"""Causal V9 readouts and completed-trial calibration.

Behavior is restricted to purchased training and offline scoring. Streaming
prediction receives neural features only. Private receipts retain every trial
selection and window; public publication must allowlist aggregate fields.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neural_experiments.v3.study import score
from neural_experiments.v4 import model as core

DT = 0.02
WARMUP = 1500
STARTUP_STOP = 7500
MAX_LABEL_STOP = 19500
DEPLOY = 19600
MAX_BINS = 30000
BURN_BINS = 62
WINDOW = 15
RTS = (0.2, 0.8)
CONFIG_KEYS = {
    "representation",
    "policy",
    "training",
    "target",
    "weighting",
    "rt_seconds",
    "ridge",
    "control",
}
BASELINE = {
    "representation": "ewma200",
    "policy": "startup120",
    "training": "continuous",
    "target": "instantaneous",
    "weighting": "uniform",
    "rt_seconds": 0.2,
    "ridge": 1.0,
    "control": "neural",
}


def digest_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def validate_config(config):
    if (
        not isinstance(config, dict)
        or set(config) != CONFIG_KEYS
        or config["representation"] not in ("ewma200", "boxcar300")
        or config["policy"] not in ("startup120", "duration360", "quota24")
        or config["training"] not in ("continuous", "trial")
        or config["target"] not in ("instantaneous", "window_mean", "endpoint")
        or config["weighting"] not in ("uniform", "sector")
        or isinstance(config["rt_seconds"], (bool, np.bool_))
        or config["rt_seconds"] not in RTS
        or isinstance(config["ridge"], (bool, np.bool_))
        or config["ridge"] not in (0.1, 1.0)
        or config["control"] not in ("neural", "zero", "intercept", "trial_permutation")
    ):
        raise ValueError("exact prespecified V9 configuration required")
    if config["training"] == "continuous":
        if (
            config["representation"] != "ewma200"
            or config["target"] != "instantaneous"
            or config["weighting"] != "uniform"
            or config["ridge"] != 1.0
            or config["control"] == "trial_permutation"
        ):
            raise ValueError("continuous controls preserve the original EWMA ridge recipe")
    elif (
        config["representation"] != "boxcar300"
        or config["target"] == "instantaneous"
        or config["control"] in ("zero", "intercept")
    ):
        raise ValueError("trial readouts require matched boxcar representation and targets")


def representation(features, segments, config):
    validate_config(config)
    transformed = core._transformed(features, {"family": "tx45_sbp"}, 2)
    core._segments(segments, len(transformed))
    if config["representation"] == "ewma200":
        return core._exponential(transformed, segments, 0.2)
    output = np.empty_like(transformed)
    # Use the same recurrence as the streaming implementation. At a block start,
    # average the available samples; no preceding block or zero padding enters.
    for a, b in segments:
        total = np.zeros(transformed.shape[1])
        for i in range(a, b):
            if i - a >= WINDOW:
                total -= transformed[i - WINDOW]
            total += transformed[i]
            output[i] = total / min(i - a + 1, WINDOW)
    return output


def floors(channels):
    return np.r_[np.full(channels, 0.1), np.full(channels, 0.01)]


@dataclass
class Decoder:
    config: dict
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        validate_config(self.config)
        self.config = json.loads(json.dumps(self.config))
        self.mean, self.scale, self.weights = (
            np.asarray(value, np.float64).copy() for value in (self.mean, self.scale, self.weights)
        )
        dim = len(self.mean)
        if (
            self.mean.ndim != 1
            or dim % 2
            or not 1 <= dim // 2 <= 512
            or self.scale.shape != (dim,)
            or self.weights.shape != (dim + 1, 2)
            or np.any(self.scale <= 0)
            or not all(np.isfinite(v).all() for v in (self.mean, self.scale, self.weights))
        ):
            raise ValueError("finite matching portable decoder geometry required")

    @property
    def channels(self):
        return len(self.mean) // 2

    def predict_raw(self, raw):
        raw = np.asarray(raw, np.float64)
        if raw.ndim != 2 or raw.shape[1] != len(self.mean) or not np.isfinite(raw).all():
            raise ValueError("finite matching neural representation required")
        z = np.clip((raw - self.mean) / self.scale, -8, 8)
        return core.project_direction(z @ self.weights[:-1] + self.weights[-1])

    def predict(self, features, segments):
        return self.predict_raw(representation(features, segments, self.config))

    def save(self, path):
        np.savez_compressed(
            path,
            version=np.asarray(9, np.int64),
            config=np.asarray(json.dumps(self.config, sort_keys=True)),
            mean=self.mean,
            scale=self.scale,
            weights=self.weights,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != {"version", "config", "mean", "scale", "weights"}:
                raise ValueError("unexpected decoder artifact fields")
            if (
                z["version"].shape != ()
                or z["version"].dtype != np.dtype("int64")
                or z["version"].item() != 9
                or z["config"].shape != ()
                or z["config"].dtype.kind != "U"
            ):
                raise ValueError("V9 decoder required")
            return cls(json.loads(z["config"].item()), z["mean"], z["scale"], z["weights"])


class StreamingDecoder:
    def __init__(self, decoder):
        self.decoder = decoder
        self.decay = float(np.exp(-DT / 0.2))
        self.buffer = np.zeros((WINDOW, len(decoder.mean)))
        self.state = np.zeros(len(decoder.mean))
        self.reset()

    def reset(self):
        self.count = 0
        self.position = 0
        self.buffer.fill(0)
        self.state.fill(0)

    def step(self, features):
        value = core._transformed(features, {"family": "tx45_sbp"}, 1)
        if value.shape != self.state.shape:
            raise ValueError("streaming channel count changed")
        if self.decoder.config["representation"] == "ewma200":
            self.state[:] = (
                value if not self.count else (self.decay * self.state + (1 - self.decay) * value)
            )
            self.count = 1
            raw = self.state
        else:
            if self.count == WINDOW:
                self.state -= self.buffer[self.position]
            self.state += value
            self.buffer[self.position] = value
            self.position = (self.position + 1) % WINDOW
            self.count = min(self.count + 1, WINDOW)
            raw = self.state / self.count
        return self.decoder.predict_raw(raw[None])[0]


def _context(case):
    context, n = case["context"], len(case["y"])
    for key in (
        "trial_id",
        "trial_age_seconds",
        "trial_onset_seconds",
        "trial_start_bin",
        "trial_stop_bin",
        "trial_right_censored",
        "reference40_distance",
        "reference40_available",
    ):
        if np.asarray(context[key]).shape != (n,):
            raise ValueError("context must match observed prefix")
    for key in ("reference40_direction", "target_position", "cursor_position"):
        if np.asarray(context[key]).shape != (n, 2):
            raise ValueError("two-dimensional bounded geometric context required")
    if np.asarray(context["trial_id"]).dtype.kind not in "iu":
        raise ValueError("integer trial identities required")
    return context


def base_mask(case):
    n = len(case["y"])
    if n > MAX_BINS or np.asarray(case["y"]).shape != (n, 2):
        raise ValueError("bounded two-dimensional targets required")
    return core.training_mask(np.asarray(case["eval_mask"]), case["segments"])


def score_mask(case):
    context = _context(case)
    ids = np.asarray(context["trial_id"])
    # A go can start at389.99s while its first complete paired bin is390.00s.
    # Exact onset and conservative observed completion cover this boundary.
    intersects = (
        (ids >= 0)
        & (np.asarray(context["trial_onset_seconds"]) < MAX_LABEL_STOP * DT)
        & (np.asarray(context["trial_stop_bin"]) + 1 > WARMUP)
    )
    bought = np.unique(ids[intersects])
    mask = base_mask(case) & (ids >= 0) & ~np.isin(ids, bought)
    mask[:DEPLOY] = False
    return mask, bought


def eligible_trials(case, *, limit_bin, training=True, required_mask=None):
    """Yield completed valid trials in causal completion order.

    The generator stops inspecting behavior when a quota caller stops iteration.
    Trial bounds are split metadata; eligibility reads only completed, purchased
    common spans. The exact paired-time convention uses integer centiseconds.
    """
    context = _context(case)
    ids, n = np.asarray(context["trial_id"]), len(case["y"])
    limit = min(int(limit_bin), n)
    base = base_mask(case)
    block = np.full(n, -1, np.int64)
    for number, (a, b) in enumerate(case["segments"]):
        block[a:b] = number
    cutoff = float(case["item"]["far_distance"])
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("positive fixed provider distance cutoff required")
    candidates = []
    for identity in np.unique(ids[:limit]):
        if identity < 0:
            continue
        rows = np.flatnonzero(ids[:limit] == identity)
        first = int(rows[0])
        completion = int(context["trial_stop_bin"][first]) + 1
        if (
            completion > limit
            or context["trial_right_censored"][first]
            or (training and context["trial_onset_seconds"][first] < WARMUP * DT)
        ):
            continue
        candidates.append((completion, first, int(identity), rows))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    for completion, first, identity, rows in candidates:
        distance = float(context["reference40_distance"][first])
        ref = np.asarray(context["reference40_direction"][first], np.float64)
        if (
            not context["reference40_available"][first]
            or not np.isfinite(distance)
            or distance <= cutoff
            or not np.isfinite(ref).all()
            or abs(float(np.linalg.norm(ref)) - 1.0) > 1e-6
        ):
            continue
        ages = np.asarray(context["trial_age_seconds"])[rows]
        if not np.isfinite(ages).all() or np.any(np.abs(ages * 100 - np.rint(ages * 100)) > 1e-8):
            raise ValueError("trial ages must lie on the raw10ms clock")
        centiseconds = np.rint(ages * 100).astype(np.int64)
        windows = []
        for rt in RTS:
            possible = rows[centiseconds - 1 >= round(rt * 100)]
            if not len(possible):
                break
            start = int(possible[0])
            windows.append({"rt_seconds": rt, "start_bin": start, "stop_bin": start + WINDOW})
        if len(windows) != 2:
            continue
        a, b = windows[0]["start_bin"], windows[-1]["stop_bin"]
        if (
            b > limit
            or b - a != 45
            or not np.all(ids[a:b] == identity)
            or not np.all(block[a:b] == block[a])
            or not base[a:b].all()
            or (required_mask is not None and not np.asarray(required_mask)[a:b].all())
        ):
            continue
        target = np.asarray(context["target_position"])[a:b].astype(np.float64)
        delta = target - np.asarray(context["cursor_position"])[a:b].astype(np.float64)
        norms = np.linalg.norm(delta, axis=1)
        if (
            not np.isfinite(delta).all()
            or not np.isfinite(target).all()
            or np.any(norms <= 1e-12)
            or not np.isfinite(np.asarray(case["y"])[a:b]).all()
        ):
            continue
        coherence = float(np.linalg.norm((delta / norms[:, None]).mean(0)))
        excursion = float(np.linalg.norm(target - target[0], axis=1).max())
        if coherence < 0.95 or excursion > 0.01 * distance:
            continue
        for window in windows:
            start, stop = window["start_bin"], window["stop_bin"]
            value = delta[start - a : stop - a].mean(0)
            length = float(np.linalg.norm(value))
            if not np.isfinite(length) or length <= 1e-12:
                break
            window["window_mean"] = (value / length).tolist()
            window["endpoint"] = np.asarray(case["y"])[stop - 1].astype(float).tolist()
            window["first_raw_age_seconds"] = float(context["trial_age_seconds"][start] - 0.01)
        else:
            yield {
                "trial_id": identity,
                "completion_bin": completion,
                "sector": int(np.floor((np.arctan2(ref[1], ref[0]) + np.pi / 4) / (np.pi / 2))) % 4,
                "common_start_bin": a,
                "common_stop_bin": b,
                "directional_coherence": coherence,
                "target_excursion": excursion,
                "reference40_distance": distance,
                "windows": windows,
            }


def calibration(case, policy):
    if policy not in ("startup120", "duration360", "quota24"):
        raise ValueError("unknown purchased-label policy")
    endpoint = STARTUP_STOP if policy == "startup120" else MAX_LABEL_STOP
    records, counts, met = [], np.zeros(4, np.int64), False
    for trial in eligible_trials(case, limit_bin=endpoint):
        records.append(trial)
        counts[trial["sector"]] += 1
        if policy == "quota24" and len(records) >= 24 and counts.min() >= 4:
            endpoint, met = trial["completion_bin"], True
            break
    n = len(case["y"])
    reason = (
        "quota_met"
        if met
        else (
            "recording_ended"
            if n < endpoint
            else "deadline"
            if policy == "quota24"
            else "fixed_endpoint"
        )
    )
    return records, {
        "policy": policy,
        "label_start_bin": WARMUP,
        "label_stop_bin": endpoint,
        "deployment_bin": endpoint + 100,
        "quota_met": met if policy == "quota24" else None,
        "stop_reason": reason,
        "eligible_trials": len(records),
        "sector_trial_counts": counts.tolist(),
        "trial_deficit": max(0, 24 - len(records)),
        "sector_deficits": np.maximum(0, 4 - counts).tolist(),
        "planned_label_seconds": 120.0 if policy == "startup120" else 360.0,
        "requested_label_seconds": max(0, endpoint - WARMUP) * DT,
        "available_label_seconds": max(0, min(n, endpoint) - WARMUP) * DT,
    }


def fit(case, config, *, raw=None):
    validate_config(config)
    n, channels = len(case["y"]), np.asarray(case["features"]["x_tx45"]).shape[1]
    raw = (
        representation(case["features"], case["segments"], config)
        if raw is None
        else np.asarray(raw)
    )
    if raw.shape != (n, channels * 2):
        raise ValueError("matching observed neural representation required")
    if config["control"] == "zero":
        trials = []
        purchase = {
            "policy": config["policy"],
            "label_start_bin": WARMUP,
            "label_stop_bin": MAX_LABEL_STOP,
            "deployment_bin": DEPLOY,
            "quota_met": None,
            "stop_reason": "zero_control",
            "eligible_trials": 0,
            "sector_trial_counts": [0, 0, 0, 0],
            "trial_deficit": 24,
            "sector_deficits": [4, 4, 4, 4],
            "planned_label_seconds": 0.0,
            "requested_label_seconds": 0.0,
            "available_label_seconds": 0.0,
        }
    else:
        trials, purchase = calibration(case, config["policy"])
    stop = purchase["label_stop_bin"]
    if config["control"] == "zero":
        mask = np.zeros(n, bool)
        indices = np.empty(0, np.int64)
        labels = np.empty((0, 2), float)
        trial_ids, sectors = np.empty(0, np.int64), np.empty(0, np.int64)
    elif config["training"] == "continuous":
        mask = base_mask(case) & (np.asarray(case["context"]["trial_id"]) >= 0)
        mask[:WARMUP], mask[stop:] = False, False
        indices = np.flatnonzero(mask)
        labels = np.asarray(case["y"])[indices].astype(np.float64)
        trial_ids = np.asarray(case["context"]["trial_id"])[indices].astype(np.int64)
        sectors = np.full(len(indices), -1, np.int64)
    else:
        position = RTS.index(config["rt_seconds"])
        indices = np.asarray([t["windows"][position]["stop_bin"] - 1 for t in trials], np.int64)
        labels = np.asarray(
            [t["windows"][position][config["target"]] for t in trials], float
        ).reshape(-1, 2)
        trial_ids = np.asarray([t["trial_id"] for t in trials], np.int64)
        sectors = np.asarray([t["sector"] for t in trials], np.int64)
        mask = np.zeros(n, bool)
        mask[indices] = True
    if config["control"] == "zero":
        mask[:] = False
        indices, labels, trial_ids, sectors = indices[:0], labels[:0], trial_ids[:0], sectors[:0]
    weights = np.ones(len(indices), float)
    if config["weighting"] == "sector" and len(indices):
        counts = np.bincount(sectors, minlength=4)
        weights = 1.0 / counts[sectors]
        weights *= len(weights) / weights.sum()
    permutation = []
    if config["control"] == "trial_permutation" and len(labels):
        donors = np.roll(np.arange(len(labels)), -1)
        labels = labels[donors].copy()
        permutation = [
            {"recipient_trial": int(a), "donor_trial": int(b)}
            for a, b in zip(trial_ids, trial_ids[donors], strict=True)
        ]
    minimum = 50 if config["training"] == "continuous" else 1
    fallback = n < stop or len(indices) < minimum
    fallback |= config["control"] == "trial_permutation" and len(indices) < 2
    if not np.isfinite(labels).all() or not np.isfinite(raw[: min(n, stop)]).all():
        raise ValueError("finite purchased labels and observed neural representation required")
    if n >= WARMUP:
        mean, scale = raw[:WARMUP].mean(0), np.maximum(raw[:WARMUP].std(0), floors(channels))
    else:
        mean, scale = np.zeros(channels * 2), floors(channels)
    coefficients = np.zeros((channels * 2 + 1, 2))
    total = float(weights.sum())
    if not fallback and config["control"] != "zero":
        if config["control"] == "intercept":
            coefficients[-1] = (labels * weights[:, None]).sum(0) / total
        else:
            z = np.clip((raw[indices] - mean) / scale, -8, 8)
            x = np.column_stack((z, np.ones(len(z))))
            gram, rhs = (x.T * weights) @ x / total, (x.T * weights) @ labels / total
            penalty = np.full(x.shape[1], config["ridge"])
            penalty[-1] = 0
            coefficients = np.linalg.solve(gram + np.diag(penalty), rhs)
    receipt = {
        "config": config,
        "normalization_start_bin": 0,
        "normalization_stop_bin": WARMUP,
        **purchase,
        "training_rows": len(indices),
        "training_trials": len(np.unique(trial_ids)),
        "last_training_bin": int(indices[-1]) if len(indices) else None,
        "training_mask_sha256": digest_array(mask),
        "training_indices_sha256": digest_array(indices),
        "training_weights_sha256": digest_array(weights),
        "training_labels_sha256": digest_array(labels),
        "weight_sum": total,
        "effective_sample_size": total**2 / float(weights @ weights) if total else 0.0,
        "fallback_zero": bool(fallback),
        "feature_penalty": config["ridge"],
        "free_intercept": True,
        "private_trials": trials,
        "private_permutation": permutation,
    }
    arrays = {
        "training_mask": mask,
        "training_indices": indices,
        "training_weights": weights,
        "training_labels": labels,
        "training_trial_ids": trial_ids,
        "training_sectors": sectors,
    }
    return Decoder(config, mean, scale, coefficients), receipt, arrays


def metrics(y, prediction, mask):
    result = score(np.asarray(y), prediction, mask)
    result["summed_mse"] = (
        float(np.square(np.asarray(y, float)[mask] - prediction[mask]).sum(1).mean())
        if result["r2"] is not None
        else None
    )
    return result


def trial_metrics(y, prediction):
    actual, estimated = (
        np.asarray(y, float).reshape(-1, 2),
        np.asarray(prediction, float).reshape(-1, 2),
    )
    if len(actual) != len(estimated):
        raise ValueError("matched trial target and predictions required")
    if len(actual) < 2:
        return {
            "trials": len(actual),
            "r2": None,
            "summed_mse": None,
            "angular_error_degrees": None,
        }
    if not np.isfinite(actual).all() or not np.isfinite(estimated).all():
        raise ValueError("finite trial target and prediction required")
    sse, sst = np.square(actual - estimated).sum(), np.square(actual - actual.mean(0)).sum()
    norm = np.linalg.norm(actual, axis=1) * np.linalg.norm(estimated, axis=1)
    cosine = np.divide(
        (actual * estimated).sum(1), norm, out=np.zeros(len(actual)), where=norm > 1e-12
    )
    return {
        "trials": len(actual),
        "r2": float(1 - sse / sst) if sst > 1e-12 else None,
        "summed_mse": float(sse / len(actual)),
        "angular_error_degrees": float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).mean()),
    }


def evaluate_session(case, configurations, *, artifact_dir=None):
    if not configurations or any(
        not re.fullmatch(r"[A-Za-z0-9_]+", name) for name in configurations
    ):
        raise ValueError("nonempty safely named configurations required")
    primary, purchased_ids = score_mask(case)
    secondary_trials = list(
        eligible_trials(case, limit_bin=len(primary), training=False, required_mask=primary)
    )
    output = Path(artifact_dir) if artifact_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    rows, receipts, refs, cache = [], {}, {}, {}
    traces = {"primary_mask": primary, "y": np.asarray(case["y"])}
    item, n = case["item"], len(primary)
    for method, config in configurations.items():
        validate_config(config)
        key = config["representation"]
        if key not in cache:
            cache[key] = representation(case["features"], case["segments"], config)
        raw = cache[key]
        fitted, receipt, fit_arrays = fit(case, config, raw=raw)
        expected = fitted.predict_raw(raw)
        ref = None
        if output is not None:
            path = output / (method + ".model.npz")
            fitted.save(path)
            restored = Decoder.load(path)
            for field in ("mean", "scale", "weights"):
                np.testing.assert_array_equal(getattr(fitted, field), getattr(restored, field))
            if fitted.config != restored.config:
                raise ValueError("export changed configuration")
            fitted = restored
            ref = {
                "model_path": str(path.resolve()),
                "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        prediction = fitted.predict_raw(raw)
        np.testing.assert_array_equal(prediction, expected)
        prediction[: receipt["deployment_bin"]] = 0
        receipt["export_prediction_max_error"] = 0.0
        receipts[method], refs[method] = receipt, ref
        traces[method + "__prediction"] = prediction
        for name, value in fit_arrays.items():
            traces[method + "__" + name] = value
        position = RTS.index(config["rt_seconds"])
        secondary_indices = np.asarray(
            [t["windows"][position]["stop_bin"] - 1 for t in secondary_trials], np.int64
        )
        secondary_mask = np.zeros(n, bool)
        secondary_mask[secondary_indices] = True
        secondary_targets = {
            target: np.asarray(
                [t["windows"][position][target] for t in secondary_trials], float
            ).reshape(-1, 2)
            for target in ("window_mean", "endpoint")
        }
        traces[method + "__secondary_indices"] = secondary_indices
        traces[method + "__secondary_trial_ids"] = np.asarray(
            [t["trial_id"] for t in secondary_trials], np.int64
        )
        for target, values in secondary_targets.items():
            traces[method + "__secondary_" + target] = values
        rows.append(
            {
                "participant": item["participant_id"],
                "session_id": item["session_id"],
                "post_implant_day": item["post_implant_day"],
                "method": method,
                "config": config,
                "n_bins": n,
                "metrics": metrics(case["y"], prediction, primary),
                "score_mask_sha256": digest_array(primary),
                "valid_rows": int(primary.sum()),
                "secondary": {
                    **{
                        target: trial_metrics(values, prediction[secondary_indices])
                        for target, values in secondary_targets.items()
                    },
                    "score_mask_sha256": digest_array(secondary_mask),
                    "trials": len(secondary_trials),
                    "prediction_definition": (
                        "one_causal_prediction_at_each_complete_eligible_trial_window_end"
                    ),
                },
                "calibration": {
                    "policy": receipt["policy"],
                    "planned_label_seconds": receipt["planned_label_seconds"],
                    "requested_label_seconds": receipt["requested_label_seconds"],
                    "observed_label_seconds": receipt["available_label_seconds"],
                    "stop_bin": receipt["label_stop_bin"],
                    "quota_attained": receipt["quota_met"],
                    "timed_out": receipt["stop_reason"] == "deadline",
                    "eligible_trials": receipt["eligible_trials"],
                    "sector_counts": receipt["sector_trial_counts"],
                    "sector_deficits": receipt["sector_deficits"],
                },
                **{
                    name: receipt[name]
                    for name in (
                        "planned_label_seconds",
                        "requested_label_seconds",
                        "available_label_seconds",
                        "quota_met",
                        "stop_reason",
                        "eligible_trials",
                        "sector_trial_counts",
                        "trial_deficit",
                        "sector_deficits",
                        "label_stop_bin",
                        "deployment_bin",
                    )
                },
                "training": receipt,
                "excluded_purchased_trial_count": len(purchased_ids),
                "model": ref,
            }
        )
    trace_ref = None
    if output is not None:
        path = output / "traces.npz"
        np.savez_compressed(path, **traces)
        trace_ref = {
            "trace_path": str(path.resolve()),
            "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    for row in rows:
        row["trace"] = trace_ref
    return {
        "rows": rows,
        "traces": traces,
        "fit_receipts": receipts,
        "model_refs": refs,
        "secondary_trials": secondary_trials,
    }
