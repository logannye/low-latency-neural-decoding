"""Causal V8 factorial decoders with separated label and inference interfaces.

The original instantaneous direction remains the primary endpoint. Context can
select purchased training labels and offline scoring strata, but never enters a
decoder's prediction function. The stable reference40 target is a distinct head.
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
LABEL_STOP = 7500
DEPLOY = 7600
MAX_BINS = 30000
MIN_LABELS = 50
BURN_BINS = 62
CONFIG_KEYS = {"family", "taus_seconds", "ridge", "phase", "weighting", "target", "control"}
BASELINE = {
    "family": "tx45_sbp",
    "taus_seconds": [0.2],
    "ridge": 1.0,
    "phase": "all",
    "weighting": "uniform",
    "target": "instantaneous",
    "control": "neural",
}


def digest_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def validate_config(config):
    if (
        not isinstance(config, dict)
        or set(config) != CONFIG_KEYS
        or config["family"] not in ("tx45_sbp", "tx3_sbp")
        or config["taus_seconds"] not in ([0.05], [0.2], [0.5], [0.05, 0.5])
        or isinstance(config["ridge"], (bool, np.bool_))
        or config["ridge"] not in (1.0, 10.0)
        or config["phase"] not in ("all", "clean")
        or config["weighting"] not in ("uniform", "equal_trial")
        or config["target"] not in ("instantaneous", "onset")
        or config["control"] not in ("neural", "zero", "intercept", "trial_permutation")
        or (config["target"] == "onset" and config["phase"] != "clean")
    ):
        raise ValueError("exact prespecified V8 configuration required")


def representation(features, segments, config):
    """Per-scale causal EWMA of sqrt counts and log1p power, reset per block."""
    validate_config(config)
    value = core._transformed(features, config, 2)
    core._segments(segments, len(value))
    return np.concatenate(
        [core._exponential(value, segments, tau) for tau in config["taus_seconds"]], axis=1
    )


def floors(channels, config):
    return np.tile(
        np.r_[np.full(channels, 0.1), np.full(channels, 0.01)], len(config["taus_seconds"])
    )


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
            np.asarray(value, dtype=np.float64).copy()
            for value in (self.mean, self.scale, self.weights)
        )
        dim = len(self.mean)
        if (
            self.mean.ndim != 1
            or self.scale.shape != (dim,)
            or dim % (2 * len(self.config["taus_seconds"]))
            or not 1 <= dim // (2 * len(self.config["taus_seconds"])) <= 512
            or self.weights.shape != (dim + 1, 2)
            or not all(np.isfinite(v).all() for v in (self.mean, self.scale, self.weights))
            or np.any(self.scale <= 0)
        ):
            raise ValueError("finite matching portable decoder geometry required")

    @property
    def channels(self):
        return len(self.mean) // (2 * len(self.config["taus_seconds"]))

    def predict_raw(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != len(self.mean) or not np.isfinite(raw).all():
            raise ValueError("finite matching neural representation required")
        z = np.clip((raw - self.mean) / self.scale, -8, 8)
        z /= np.sqrt(len(self.config["taus_seconds"]))
        return core.project_direction(z @ self.weights[:-1] + self.weights[-1])

    def predict(self, features, segments):
        return self.predict_raw(representation(features, segments, self.config))

    def save(self, path):
        np.savez_compressed(
            path,
            version=np.asarray(8),
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
                or z["version"].item() != 8
                or z["config"].shape != ()
                or z["config"].dtype.kind != "U"
            ):
                raise ValueError("V8 decoder required")
            return cls(json.loads(z["config"].item()), z["mean"], z["scale"], z["weights"])


class StreamingDecoder:
    """Constant-state, current-and-past neural observations only."""

    def __init__(self, decoder):
        self.decoder = decoder
        self.decay = np.exp(-DT / np.asarray(decoder.config["taus_seconds"]))
        self.state = np.zeros((len(self.decay), 2 * decoder.channels))
        self.reset()

    def reset(self):
        self.initialized = False
        self.state.fill(0)

    def step(self, features):
        value = core._transformed(features, self.decoder.config, 1)
        if len(value) != self.state.shape[1]:
            raise ValueError("streaming channel count changed")
        if not self.initialized:
            self.state[:] = value
            self.initialized = True
        else:
            self.state[:] = self.decay[:, None] * self.state + (1 - self.decay[:, None]) * value
        return self.decoder.predict_raw(self.state.reshape(1, -1))[0]


def _context(case):
    context, n = case["context"], len(case["y"])
    for key in (
        "trial_id",
        "trial_age_seconds",
        "trial_start_bin",
        "trial_onset_seconds",
        "reference40_distance",
        "reference40_available",
    ):
        if np.asarray(context[key]).shape != (n,):
            raise ValueError("context vectors must match the exposed prefix")
    if np.asarray(context["reference40_direction"]).shape != (n, 2):
        raise ValueError("reference40 direction must match target geometry")
    if np.asarray(context["trial_id"]).dtype.kind not in "iu":
        raise ValueError("integer trial identities required")
    if np.asarray(context["reference40_available"]).dtype != bool:
        raise ValueError("boolean reference40 availability required")
    return context


def _base_mask(case):
    mask = np.asarray(case.get("mask", case.get("eval_mask")))
    n = len(case["y"])
    if mask.dtype != bool or mask.shape != (n,) or n > MAX_BINS:
        raise ValueError("bounded boolean target mask required")
    result = core.training_mask(mask, case["segments"])
    if np.asarray(case["y"]).shape != (n, 2):
        raise ValueError("two-dimensional direction targets required")
    return result


def clean_mask(case):
    """A fixed early-phase, far-at40ms stratum; this never changes primary scoring."""
    context = _context(case)
    item = case.get("item", case.get("metadata"))
    cutoff = float(item["far_distance"])
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("positive fixed participant distance cutoff required")
    age = np.asarray(context["trial_age_seconds"])
    return (
        np.asarray(context["reference40_available"])
        & (age >= 0.2)
        & (age < 1.0)
        & (np.asarray(context["reference40_distance"]) > cutoff)
        & (np.asarray(context["trial_id"]) >= 0)
    )


def score_masks(case):
    base, context = _base_mask(case), _context(case)
    ids = np.asarray(context["trial_id"])
    # Split metadata, not inference information: no trial having any purchased
    # coverage may be scored, even if its post-150s samples were never labeled.
    purchased_ids = np.unique(ids[WARMUP : min(LABEL_STOP, len(ids))])
    purchased_ids = purchased_ids[purchased_ids >= 0]
    primary = base & (ids >= 0) & ~np.isin(ids, purchased_ids)
    primary[:DEPLOY] = False
    return primary, primary & clean_mask(case), purchased_ids


def training_mask(case, config):
    validate_config(config)
    selected, context = _base_mask(case), _context(case)
    selected[:WARMUP] = False
    selected[LABEL_STOP:] = False
    selected &= np.asarray(context["trial_id"]) >= 0
    if config["phase"] == "clean":
        selected &= clean_mask(case) & (np.asarray(context["trial_onset_seconds"]) >= 30.0)
    return selected


def _permuted_labels(labels, ids):
    """One fixed cyclic whole-trial donor rotation, resampled within each trial.

    Only purchased selected labels enter this diagnostic. It is not a random
    permutation significance test. Unequal-length donor trajectories use a
    deterministic relative-row nearest-left mapping without interpolation.
    """
    unique = np.unique(ids)
    unique = sorted(unique, key=lambda value: int(np.flatnonzero(ids == value)[0]))
    result, mapping = labels.copy(), []
    for recipient, donor in zip(unique, unique[1:] + unique[:1], strict=True):
        take, give = np.flatnonzero(ids == recipient), np.flatnonzero(ids == donor)
        selected = np.arange(len(take)) * len(give) // len(take)
        result[take] = labels[give[selected]]
        mapping.append(
            {
                "recipient_trial": int(recipient),
                "donor_trial": int(donor),
                "recipient_rows": len(take),
                "donor_rows": len(give),
            }
        )
    return result, mapping


def fit(case, config, *, raw=None):
    """Fit only purchased30–150s rows; future labels never enter arithmetic."""
    validate_config(config)
    n = len(case["y"])
    if raw is None:
        raw = representation(case["features"], case["segments"], config)
    raw = np.asarray(raw)
    channels = np.asarray(case["features"]["x_tx45"]).shape[1]
    dimension = channels * 2 * len(config["taus_seconds"])
    if raw.shape != (n, dimension) or not np.isfinite(raw[: min(n, LABEL_STOP)]).all():
        raise ValueError("finite observed representation with matching geometry required")
    selected = training_mask(case, config)
    indices = np.flatnonzero(selected)
    ids = np.asarray(case["context"]["trial_id"])[indices]
    target = (
        case["y"]
        if config["target"] == "instantaneous"
        else case["context"]["reference40_direction"]
    )
    labels = np.asarray(target)[indices].astype(np.float64)
    if not np.isfinite(labels).all():
        raise ValueError("selected purchased labels must be finite")
    source_labels = labels.copy()
    unique, counts = np.unique(ids, return_counts=True)
    sample_weights = np.ones(len(indices), dtype=np.float64)
    if config["weighting"] == "equal_trial" and len(indices):
        count_map = dict(zip(unique, counts, strict=True))
        sample_weights = np.asarray([1 / count_map[value] for value in ids], dtype=np.float64)
        sample_weights *= len(indices) / sample_weights.sum()
    permutation = []
    if config["control"] == "trial_permutation":
        labels, permutation = _permuted_labels(labels, ids)
    # A truncated recording remains in the cohort; it never deploys a model
    # fitted before the declared complete150s calibration endpoint.
    fallback = n < LABEL_STOP or len(indices) < MIN_LABELS
    fallback |= config["control"] == "trial_permutation" and len(unique) < 2
    if n >= WARMUP:
        mean = raw[:WARMUP].mean(0)
        scale = np.maximum(raw[:WARMUP].std(0), floors(channels, config))
    else:
        mean, scale = np.zeros(dimension), floors(channels, config)
    weights = np.zeros((dimension + 1, 2), dtype=np.float64)
    total = float(sample_weights.sum())
    if not fallback and config["control"] != "zero":
        if config["control"] == "intercept":
            weights[-1] = (labels * sample_weights[:, None]).sum(0) / total
        else:
            z = np.clip((raw[indices] - mean) / scale, -8, 8)
            z /= np.sqrt(len(config["taus_seconds"]))
            x = np.column_stack((z, np.ones(len(z))))
            gram, rhs = (x.T * sample_weights) @ x / total, (x.T * sample_weights) @ labels / total
            penalty = np.full(dimension + 1, config["ridge"])
            penalty[-1] = 0
            weights = np.linalg.solve(gram + np.diag(penalty), rhs)
    trial_totals = np.asarray([sample_weights[ids == value].sum() for value in unique])
    directions = np.arctan2(source_labels[:, 1], source_labels[:, 0]) if len(labels) else []
    octants = np.floor((np.asarray(directions) + np.pi) / (np.pi / 4)).astype(int) % 8
    receipt = {
        "config": config,
        "normalization_start_bin": 0,
        "normalization_stop_bin": WARMUP,
        "label_start_bin": WARMUP,
        "label_stop_bin": LABEL_STOP,
        "last_training_bin": int(indices[-1]) if len(indices) else None,
        "training_rows": len(indices),
        "training_trials": len(unique),
        "training_mask_sha256": digest_array(selected),
        "training_weights_sha256": digest_array(sample_weights),
        "training_labels_sha256": digest_array(labels),
        "weight_sum": total,
        "effective_sample_size": total**2 / float(sample_weights @ sample_weights)
        if total
        else 0.0,
        "effective_trials": total**2 / float(trial_totals @ trial_totals) if total else 0.0,
        "direction_octant_rows": np.bincount(octants, minlength=8).tolist(),
        "direction_octants_with_at_least50_rows": int(
            (np.bincount(octants, minlength=8) >= 50).sum()
        ),
        "permutation": permutation,
        "fallback_zero": bool(fallback),
        "feature_penalty": config["ridge"],
        "free_intercept": True,
        "multiscale_design_divisor": float(np.sqrt(len(config["taus_seconds"]))),
    }
    return Decoder(config, mean, scale, weights), receipt, selected, sample_weights


def _metrics(y, prediction, mask):
    result = score(np.asarray(y), prediction, mask)
    result["summed_mse"] = (
        float(np.square(np.asarray(y)[mask] - prediction[mask]).sum(1).mean())
        if result["r2"] is not None
        else None
    )
    return result


def _trial_metrics(y, prediction, mask, trial_ids):
    """Equal trial vectors after averaging on exactly the common secondary bins.

    Averaging changes the endpoint; these descriptive scores cannot satisfy the
    primary gate. Two distinct trial vectors are the minimum for this diagnostic.
    """
    y, prediction = np.asarray(y, np.float64), np.asarray(prediction, np.float64)
    ids = np.unique(np.asarray(trial_ids)[mask])
    actual = np.asarray([np.asarray(y)[mask & (trial_ids == value)].mean(0) for value in ids])
    estimated = np.asarray([prediction[mask & (trial_ids == value)].mean(0) for value in ids])
    if len(ids) < 2:
        return {"trials": len(ids), "r2": None, "summed_mse": None, "angular_error_degrees": None}
    residual = np.square(actual - estimated).sum()
    variation = np.square(actual - actual.mean(0)).sum()
    norm = np.linalg.norm(actual, axis=1) * np.linalg.norm(estimated, axis=1)
    cosine = np.divide(
        (actual * estimated).sum(1), norm, out=np.zeros(len(ids)), where=norm > 1e-12
    )
    return {
        "trials": len(ids),
        "r2": float(1 - residual / variation) if variation > 1e-12 else None,
        "summed_mse": float(residual / len(ids)),
        "angular_error_degrees": float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).mean()),
    }


def evaluate_session(case, configurations, *, artifact_dir=None):
    """Evaluate fixed configurations; all primary masks and time budgets match.

    Optional artifacts contain private sample-level traces. Every saved model is
    reloaded and verified exactly before producing the scored predictions.
    """
    if not configurations or any(not re.fullmatch(r"[A-Za-z0-9_]+", k) for k in configurations):
        raise ValueError("nonempty safely named configurations required")
    item = case.get("item", case.get("metadata"))
    primary, secondary, purchased_ids = score_masks(case)
    n = len(primary)
    output = Path(artifact_dir) if artifact_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    rows, receipts, refs, cache = [], {}, {}, {}
    traces = {
        "primary_mask": primary,
        "secondary_mask": secondary,
        "y": np.asarray(case["y"]),
        "onset_y": np.asarray(case["context"]["reference40_direction"]),
    }
    for method, config in configurations.items():
        validate_config(config)
        representation_key = (config["family"], tuple(config["taus_seconds"]))
        if representation_key not in cache:
            cache[representation_key] = representation(case["features"], case["segments"], config)
        raw = cache[representation_key]
        fitted, receipt, selected, sample_weights = fit(case, config, raw=raw)
        reference = fitted.predict_raw(raw)
        model_ref = None
        if output is not None:
            path = output / (method + ".model.npz")
            fitted.save(path)
            restored = Decoder.load(path)
            for field in ("mean", "scale", "weights"):
                np.testing.assert_array_equal(getattr(restored, field), getattr(fitted, field))
            if restored.config != fitted.config:
                raise ValueError("export changed decoder configuration")
            fitted = restored
            model_ref = {
                "model_path": str(path.resolve()),
                "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        prediction = fitted.predict_raw(raw)
        np.testing.assert_array_equal(reference, prediction)
        prediction[: min(n, DEPLOY)] = 0
        receipt["export_prediction_max_error"] = 0.0
        receipts[method], refs[method] = receipt, model_ref
        traces[method + "__prediction"] = prediction
        traces[method + "__training_mask"] = selected
        traces[method + "__training_weights"] = sample_weights
        row = {
            "participant": item["participant_id"],
            "session_id": item["session_id"],
            "post_implant_day": item["post_implant_day"],
            "method": method,
            "n_bins": n,
            "config": config,
            "target": config["target"],
            "metrics": _metrics(case["y"], prediction, primary),
            "score_mask_sha256": digest_array(primary),
            "valid_rows": int(primary.sum()),
            "secondary": {
                "instantaneous": _metrics(case["y"], prediction, secondary),
                "onset": _metrics(case["context"]["reference40_direction"], prediction, secondary),
                "score_mask_sha256": digest_array(secondary),
                "valid_rows": int(secondary.sum()),
                "endpoint": "fixed40ms_trial_direction_on_early_far_trial_bins",
                "trial_averaged": {
                    "instantaneous": _trial_metrics(
                        case["y"], prediction, secondary, case["context"]["trial_id"]
                    ),
                    "onset": _trial_metrics(
                        case["context"]["reference40_direction"],
                        prediction,
                        secondary,
                        case["context"]["trial_id"],
                    ),
                },
            },
            "planned_label_seconds": 120.0,
            "available_label_seconds": max(0, min(n, LABEL_STOP) - WARMUP) * DT,
            "training": receipt,
            "excluded_purchased_trial_count": len(purchased_ids),
            "model": model_ref,
        }
        rows.append(row)
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
    return {"rows": rows, "traces": traces, "fit_receipts": receipts, "model_refs": refs}
