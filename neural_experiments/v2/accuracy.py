"""Locked H1 V2 development study. Frozen V1 production modules are never modified.

The small TemporalCore/fit_model API is also usable by independently planned
transfer experiments. This module's CLI only opens the S0-S4 development index.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits
from torch import nn
from torch.nn import functional as F

from neural_decoder.artifacts import environment, sha256, write_json
from neural_decoder.metrics import FileEvaluation, score_files
from neural_decoder.train import checked_development_index


class TemporalCore(nn.Module):
    """Causal channels-last temporal features with a small linear readout."""

    def __init__(self, input_channels, output_channels, dilations=(1, 2, 4), width=32):
        super().__init__()
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.dilations = tuple(int(value) for value in dilations)
        self.width = int(width)
        if (
            not self.dilations
            or min(self.input_channels, self.output_channels, self.width, *self.dilations) < 1
        ):
            raise ValueError("positive architecture dimensions required")
        self.receptive_field = 1 + 2 * sum(self.dilations)
        self.convs = nn.ModuleList(
            nn.Conv1d(channels, width, 3, dilation=dilation)
            for channels, dilation in zip(
                (input_channels,) + (width,) * (len(self.dilations) - 1),
                self.dilations,
                strict=True,
            )
        )
        self.head = nn.Linear(width, output_channels)

    def encode(self, features):
        if features.ndim != 3 or features.shape[-1] != self.input_channels or not features.shape[1]:
            raise ValueError("expected nonempty [batch,time,input_channels] features")
        hidden = features.transpose(1, 2)
        for conv, dilation in zip(self.convs, self.dilations, strict=True):
            hidden = F.relu(conv(F.pad(hidden, (2 * dilation, 0))))
        return hidden.transpose(1, 2)

    def forward(self, features):
        return self.head(self.encode(features))


def training_arrays(training, *, receptive_field, chunk_scored_bins=256, minimum_loss_start=None):
    """Use real preceding context and one loss occurrence per eligible training bin."""
    context = receptive_field - 1
    start_at = max(context, context if minimum_loss_start is None else minimum_loss_start)
    inputs = np.concatenate([record[0] for record in training]).astype(np.float64)
    targets = np.concatenate([y[mask] for _, y, mask in training]).astype(np.float64)
    if not len(targets):
        raise ValueError("training has no labeled rows")
    transforms = {
        "input_mean": inputs.mean(0).astype(np.float32),
        "input_scale": np.maximum(inputs.std(0), 1e-6).astype(np.float32),
        "target_mean": targets.mean(0).astype(np.float32),
        "target_scale": np.maximum(targets.std(0), 1e-6).astype(np.float32),
    }
    xs, ys, masks = [], [], []
    for counts, target, supplied_mask in training:
        x = (counts.astype(np.float32) - transforms["input_mean"]) / transforms["input_scale"]
        y = (target.astype(np.float32) - transforms["target_mean"]) / transforms["target_scale"]
        for start in range(start_at, len(x), chunk_scored_bins):
            stop = min(start + chunk_scored_bins, len(x))
            length = stop - start + context
            chunk_x = np.zeros((context + chunk_scored_bins, x.shape[1]), dtype=np.float32)
            chunk_y = np.zeros((len(chunk_x), y.shape[1]), dtype=np.float32)
            chunk_mask = np.zeros(len(chunk_x), dtype=bool)
            chunk_x[:length] = x[start - context : stop]
            chunk_y[:length] = y[start - context : stop]
            chunk_mask[context:length] = supplied_mask[start:stop]
            if chunk_mask.any():
                xs.append(chunk_x)
                ys.append(chunk_y)
                masks.append(chunk_mask)
    if not xs:
        raise ValueError("training has no eligible causal loss chunks")
    return transforms, tuple(torch.from_numpy(np.asarray(values)) for values in (xs, ys, masks))


def loss_weights(transforms, loss_kind):
    if loss_kind == "standardized":
        return np.ones_like(transforms["target_scale"])
    if loss_kind != "source_units":
        raise ValueError("unknown loss kind")
    weights = transforms["target_scale"].astype(np.float64) ** 2
    return (weights / weights.mean()).astype(np.float32)


def fit_model(
    training,
    *,
    dilations=(1, 2, 4),
    loss_kind="standardized",
    seed=0,
    epochs=20,
    checkpoint_epochs=(),
    batch_chunks=16,
    chunk_scored_bins=256,
    width=32,
    minimum_loss_start=None,
    learning_rate=0.001,
    channel_dropout=0.1,
):
    """Return model, train-only transforms, epoch losses, and numeric checkpoint states.

    Callers own thread limits. No validation/evaluation labels enter this function.
    All checkpoints are predeclared epochs, never selected inside this fitter.
    """
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    model = TemporalCore(training[0][0].shape[1], training[0][1].shape[1], dilations, width)
    transforms, (x, y, mask) = training_arrays(
        training,
        receptive_field=model.receptive_field,
        chunk_scored_bins=chunk_scored_bins,
        minimum_loss_start=minimum_loss_start,
    )
    weighting = torch.from_numpy(loss_weights(transforms, loss_kind))
    scale = torch.from_numpy(transforms["target_scale"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    losses, checkpoints = [], {}
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        order = rng.permutation(len(x))
        objective_sum = standardized_sum = source_sum = 0.0
        elements = 0
        for offset in range(0, len(order), batch_chunks):
            chosen = order[offset : offset + batch_chunks]
            batch = x[chosen]
            keep = torch.rand((len(chosen), 1, batch.shape[-1])) >= channel_dropout
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch * keep)
            error = prediction[mask[chosen]] - y[chosen][mask[chosen]]
            squared = error.square()
            objective = (squared * weighting).mean()
            if not torch.isfinite(objective):
                raise ValueError("nonfinite training loss")
            objective.backward()
            optimizer.step()
            elements += error.numel()
            objective_sum += float(objective.detach()) * error.numel()
            standardized_sum += float(squared.detach().sum())
            source_sum += float((squared.detach() * scale.square()).sum())
        losses.append(
            {
                "epoch": epoch,
                "objective": objective_sum / elements,
                "standardized_mse": standardized_sum / elements,
                "source_unit_mse": source_sum / elements,
                "scored_rows": elements // model.output_channels,
                "wall_seconds": time.perf_counter() - started,
            }
        )
        if epoch in checkpoint_epochs:
            checkpoints[epoch] = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
    return model.eval(), transforms, losses, checkpoints


def predict_with_features(model, transforms, counts):
    values = (np.asarray(counts, dtype=np.float32) - transforms["input_mean"]) / transforms[
        "input_scale"
    ]
    with torch.inference_mode():
        latent = model.encode(torch.from_numpy(values)[None])
        output = model.head(latent)[0].numpy()
    return output * transforms["target_scale"] + transforms["target_mean"], latent[0].numpy()


def predict(model, transforms, counts):
    return predict_with_features(model, transforms, counts)[0]


def save_model(path, model, transforms, **extra_arrays):
    arrays = {
        "format_version": np.asarray(1, dtype=np.int64),
        "architecture": np.asarray(
            [model.input_channels, model.output_channels, model.width], dtype=np.int64
        ),
        "dilations": np.asarray(model.dilations, dtype=np.int64),
        **transforms,
        **extra_arrays,
    }
    arrays.update(
        {f"state__{name}": value.detach().numpy() for name, value in model.state_dict().items()}
    )
    with Path(path).open("xb") as stream:
        np.savez(stream, **arrays)
    return sha256(path)


def load_model(path):
    with np.load(path, allow_pickle=False) as bundle:
        arrays = {name: bundle[name] for name in bundle.files}
    if (
        arrays["format_version"].shape != ()
        or arrays["format_version"].dtype != np.int64
        or arrays["format_version"].item() != 1
    ):
        raise ValueError("unsupported model version")
    dims = arrays["architecture"]
    if (
        dims.shape != (3,)
        or dims.dtype != np.int64
        or arrays["dilations"].dtype != np.int64
        or arrays["dilations"].ndim != 1
    ):
        raise ValueError("invalid architecture")
    with torch.random.fork_rng():
        model = TemporalCore(int(dims[0]), int(dims[1]), tuple(arrays["dilations"]), int(dims[2]))
    state = {}
    for name, value in model.state_dict().items():
        array = arrays.pop(f"state__{name}")
        if (
            array.shape != tuple(value.shape)
            or array.dtype != np.float32
            or not np.isfinite(array).all()
        ):
            raise ValueError("invalid model tensor")
        state[name] = torch.from_numpy(array)
    model.load_state_dict(state, strict=True)
    transforms = {
        name: arrays.pop(name)
        for name in ("input_mean", "input_scale", "target_mean", "target_scale")
    }
    for name, dimension in (("input", int(dims[0])), ("target", int(dims[1]))):
        for suffix in ("mean", "scale"):
            array = transforms[f"{name}_{suffix}"]
            if (
                array.shape != (dimension,)
                or array.dtype != np.float32
                or not np.isfinite(array).all()
            ):
                raise ValueError("invalid affine transform")
        if (transforms[f"{name}_scale"] <= 0).any():
            raise ValueError("nonpositive affine scale")
    allowed = {"format_version", "architecture", "dilations", "latent_mean", "latent_scale"}
    if set(arrays) - allowed:
        raise ValueError("unexpected model arrays")
    return model.eval(), transforms, arrays


class CausalStream:
    """Feature-only same-weight reference stream with zero latent startup history."""

    def __init__(self, model, transforms):
        self.transforms = transforms
        self.dilations = model.dilations
        self.weights = [conv.weight.detach().numpy().copy() for conv in model.convs]
        self.biases = [conv.bias.detach().numpy().copy() for conv in model.convs]
        self.head_weight = model.head.weight.detach().numpy().copy()
        self.head_bias = model.head.bias.detach().numpy().copy()
        self.reset()

    def reset(self):
        self.histories = [
            np.zeros((2 * d, w.shape[1]), dtype=np.float32)
            for d, w in zip(self.dilations, self.weights, strict=True)
        ]

    def step(self, counts):
        raw = np.asarray(counts)
        if (
            raw.shape != self.transforms["input_mean"].shape
            or not np.isfinite(raw).all()
            or (raw < 0).any()
        ):
            raise ValueError("invalid neural count vector")
        current = (raw.astype(np.float32) - self.transforms["input_mean"]) / self.transforms[
            "input_scale"
        ]
        for i, (dilation, weight, bias) in enumerate(
            zip(self.dilations, self.weights, self.biases, strict=True)
        ):
            sequence = np.concatenate((self.histories[i], current[None]))
            hidden = bias.copy()
            for tap in range(3):
                hidden += weight[:, :, tap] @ sequence[tap * dilation]
            self.histories[i] = sequence[1:]
            current = np.maximum(hidden, 0)
        result = self.head_weight @ current + self.head_bias
        return result * self.transforms["target_scale"] + self.transforms["target_mean"]

    def predict_chunk(self, counts):
        return np.asarray([self.step(row) for row in counts], dtype=np.float32)


def fit_calibration(
    latent_prefix, prediction_prefix, target_prefix, mask_prefix, *, method, ridge_lambda=1
):
    """Inputs contain only the declared prefix; no suffix labels are accepted."""
    if method not in {"unadapted", "intercept", "ridge_delta_head"} or ridge_lambda <= 0:
        raise ValueError("invalid calibration settings")
    if (
        len(latent_prefix) != len(target_prefix)
        or prediction_prefix.shape != target_prefix.shape
        or np.asarray(mask_prefix).shape != (len(target_prefix),)
    ):
        raise ValueError("prefix arrays must align")
    coefficient = np.zeros((latent_prefix.shape[1], target_prefix.shape[1]), dtype=np.float64)
    intercept = np.zeros(target_prefix.shape[1], dtype=np.float64)
    if method == "unadapted" or not len(target_prefix):
        return coefficient, intercept
    if not np.asarray(mask_prefix).any():
        raise ValueError("positive calibration budget has no scored labels")
    residual = target_prefix[mask_prefix].astype(np.float64) - prediction_prefix[
        mask_prefix
    ].astype(np.float64)
    intercept = residual.mean(0)
    if method == "intercept":
        return coefficient, intercept
    if method != "ridge_delta_head":
        raise ValueError("unknown calibration method")
    features = latent_prefix[mask_prefix].astype(np.float64)
    mean = features.mean(0)
    centered = features - mean
    coefficient = np.linalg.solve(
        centered.T @ centered / len(features) + ridge_lambda * np.eye(features.shape[1]),
        centered.T @ (residual - intercept) / len(features),
    )
    return coefficient, intercept - mean @ coefficient


def match_exposure(same_day_record, earlier_records, fraction, *, minimum_loss_start=62):
    """Match raw duration and eligible label counts without crossing file histories."""
    item, counts, targets, mask = same_day_record
    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0,1]")
    budget = math.floor(len(counts) * fraction)
    same = [(counts[:budget].copy(), targets[:budget].copy(), mask[:budget].copy())]
    same_sources = [{"file_id": item["file_id"], "start": 0, "stop": budget}]
    remaining, other, sources = budget, [], []
    for prior_item, x, y, m in reversed(earlier_records):
        count = min(remaining, len(x))
        if count:
            other.insert(0, (x[-count:].copy(), y[-count:].copy(), m[-count:].copy()))
            sources.insert(
                0, {"file_id": prior_item["file_id"], "start": len(x) - count, "stop": len(x)}
            )
            remaining -= count
        if not remaining:
            break
    if remaining:
        raise ValueError("insufficient earlier-day matched exposure")
    for records in (same, other):
        for _, _, selected_mask in records:
            selected_mask[:minimum_loss_start] = False
    label_count = min(sum(int(m.sum()) for _, _, m in records) for records in (same, other))
    if not label_count:
        raise ValueError("matched exposure has no labels")
    for records in (same, other):
        retained = 0
        for _, _, selected_mask in records:
            positions = np.flatnonzero(selected_mask)
            keep = min(len(positions), label_count - retained)
            selected_mask[positions[keep:]] = False
            retained += keep
    return {"same_day": same, "earlier_days": other}, {
        "raw_bins_each": budget,
        "eligible_labels_each": label_count,
        "same_day_sources": same_sources,
        "earlier_days_sources": sources,
    }


def load_records(index_path, index, file_ids=None):
    records = []
    for item in sorted(index["files"], key=lambda row: (int(row["day"][1:]), row["file_id"])):
        if file_ids is not None and item["file_id"] not in file_ids:
            continue
        with np.load(Path(index_path).parent / item["features"], allow_pickle=False) as data:
            x = data["counts"]
            np.testing.assert_array_equal(data["sequence_id"], np.arange(item["n_bins"]))
            np.testing.assert_array_equal(data["channel_ids"], np.arange(176))
        with np.load(Path(index_path).parent / item["targets"], allow_pickle=False) as data:
            y, mask = data["targets"], data["eval_mask"]
        if (
            x.shape != (item["n_bins"], 176)
            or y.shape != (len(x), 7)
            or mask.shape != (len(x),)
            or mask.dtype != np.bool_
        ):
            raise ValueError("invalid development arrays")
        records.append((item, x, y, mask))
    return records


def fit_arguments(plan, arm, seed, epochs):
    settings = plan["training"]
    return {
        "dilations": tuple(arm["dilations"]),
        "loss_kind": arm["loss_kind"],
        "seed": seed,
        "epochs": epochs,
        **{
            key: settings[key]
            for key in (
                "width",
                "learning_rate",
                "channel_dropout",
                "chunk_scored_bins",
                "batch_chunks",
                "minimum_loss_start",
            )
        },
    }


def evaluate(model, transforms, records, folder, *, full_stream_check=False, suffix_start=0):
    evaluations, evidence, max_error = [], {}, 0.0
    for item, x, y, mask in records:
        predictions = predict(model, transforms, x)
        if full_stream_check:
            actual = CausalStream(model, transforms).predict_chunk(x)
            np.testing.assert_allclose(actual, predictions, atol=1e-7, rtol=3e-5)
            max_error = max(max_error, float(np.max(np.abs(actual - predictions))))
        path = folder / f"{item['file_id']}.npy"
        np.save(path, predictions, allow_pickle=False)
        scored = mask & (np.arange(len(mask)) >= suffix_start)
        evaluations.append(FileEvaluation(item["file_id"], item["day"], predictions, y, scored))
        evidence[item["file_id"]] = {
            "path": str(path),
            "sha256": sha256(path),
            "bins": len(x),
            "scored_bins": int(scored.sum()),
        }
    return score_files(evaluations), evidence, max_error


def artifact_identity(path, output):
    return {"path": str(Path(path).relative_to(output)), "sha256": sha256(path)}


def relative_predictions(predictions, output):
    for item in predictions.values():
        item["path"] = str(Path(item["path"]).relative_to(output))
    return predictions


def run_schedule(plan, index_path, index, output):
    schedule = plan["schedule_check"]
    file_ids = schedule["training_files"] + [schedule["validation_file"]]
    nested = load_records(index_path, index, file_ids)
    training = [
        (x, y, mask) for item, x, y, mask in nested if item["file_id"] in schedule["training_files"]
    ]
    validation = [
        record for record in nested if record[0]["file_id"] == schedule["validation_file"]
    ]
    results = []
    for arm in plan["arms"]:
        folder = output / "schedule" / arm["name"]
        folder.mkdir(parents=True)
        model, transforms, losses, checkpoints = fit_model(
            training, **fit_arguments(plan, arm, 0, 40), checkpoint_epochs=(20, 40)
        )
        write_json(folder / "losses.json", losses)
        scores = {}
        for epoch in (20, 40):
            checkpoint_folder = folder / f"epoch-{epoch}"
            checkpoint_folder.mkdir()
            model.load_state_dict(checkpoints[epoch])
            save_model(checkpoint_folder / "model.npz", model, transforms)
            metric, predictions, _ = evaluate(model, transforms, validation, checkpoint_folder)
            scores[str(epoch)] = {
                "scores": metric,
                "model": artifact_identity(checkpoint_folder / "model.npz", output),
                "predictions": relative_predictions(predictions, output),
            }
        results.append(
            {
                "arm": arm["name"],
                "checkpoints": scores,
                "delta_r2": scores["40"]["scores"]["r2_mean"] - scores["20"]["scores"]["r2_mean"],
            }
        )
        print(
            json.dumps(
                {"phase": "schedule", "arm": arm["name"], "delta_r2": results[-1]["delta_r2"]}
            ),
            flush=True,
        )
    deltas = [row["delta_r2"] for row in results]
    chosen = 40 if np.mean(deltas) >= 0.005 and min(deltas) >= -0.01 else 20
    decision = {
        "selected_epochs": chosen,
        "mean_gain_40_vs20": float(np.mean(deltas)),
        "worst_arm_gain": min(deltas),
        "results": results,
        "validation_file": schedule["validation_file"],
        "outer_days_scored_before_decision": [],
    }
    write_json(output / "schedule-decision.json", decision)
    return decision


def run_calibration(model, transforms, training, evaluation, plan, folder, output, identity):
    all_latent = np.concatenate(
        [predict_with_features(model, transforms, x)[1] for x, _, _ in training]
    ).astype(np.float64)
    latent_mean, latent_scale = all_latent.mean(0), np.maximum(all_latent.std(0), 1e-6)
    np.savez(folder / "latent-normalizer.npz", latent_mean=latent_mean, latent_scale=latent_scale)
    del all_latent
    prepared = []
    for item, x, y, mask in evaluation:
        prediction, latent = predict_with_features(model, transforms, x)
        prepared.append((item, prediction, (latent - latent_mean) / latent_scale, y, mask))
    rows = []
    settings = plan["calibration"]
    for budget in settings["budgets_seconds_per_file"]:
        boundary = round(budget / settings["bin_seconds"])
        for method in settings["methods"]:
            subfolder = folder / f"{method}-{budget}s"
            subfolder.mkdir()
            scores, files = [], {}
            for item, base, latent, y, mask in prepared:
                coefficient, intercept = fit_calibration(
                    latent[:boundary],
                    base[:boundary],
                    y[:boundary],
                    mask[:boundary],
                    method=method,
                    ridge_lambda=settings["ridge_lambda"],
                )
                prediction = base.astype(np.float64)
                prediction[boundary:] += latent[boundary:] @ coefficient + intercept
                file_id = item["file_id"]
                path = subfolder / f"{file_id}.npy"
                np.save(path, prediction, allow_pickle=False)
                np.savez(
                    subfolder / f"{file_id}.calibration.npz",
                    coefficient=coefficient,
                    intercept=intercept,
                )
                suffix = mask & (np.arange(len(mask)) >= settings["common_suffix_start_bin"])
                scores.append(FileEvaluation(file_id, item["day"], prediction, y, suffix))
                files[file_id] = {
                    **artifact_identity(path, output),
                    "prefix_bins": boundary,
                    "prefix_scored_labels": int(mask[:boundary].sum())
                    if method != "unadapted"
                    else 0,
                    "available_prefix_scored_labels": int(mask[:boundary].sum()),
                    "suffix_scored_bins": int(suffix.sum()),
                    "bins": len(mask),
                }
            rows.append(
                {
                    **identity,
                    "method": method,
                    "budget_seconds": budget,
                    "common_suffix_start_bin": settings["common_suffix_start_bin"],
                    "predictions": files,
                    "scores": score_files(scores),
                }
            )
    return rows


def aggregate(main, calibration):
    arms = sorted({row["arm"] for row in main})
    baseline = {
        (row["seed"], row["day"]): row["scores"]["r2_mean"]
        for row in main
        if row["arm"] == "rf15_standardized"
    }
    comparison = []
    for arm in arms:
        rows = [row for row in main if row["arm"] == arm]
        seeds = [
            float(np.mean([r["scores"]["r2_mean"] for r in rows if r["seed"] == seed]))
            for seed in (0, 1, 2)
        ]
        day_means = {
            day: float(np.mean([r["scores"]["r2_mean"] for r in rows if r["day"] == day]))
            for day in ("S2", "S3", "S4")
        }
        comparison.append(
            {
                "arm": arm,
                "mean_r2": float(np.mean(seeds)),
                "seed_means": seeds,
                "seed_mean_population_std": float(np.std(seeds)),
                "day_seed_means": day_means,
                "mean_delta_vs_rf15_standardized": float(
                    np.mean(
                        [r["scores"]["r2_mean"] - baseline[(r["seed"], r["day"])] for r in rows]
                    )
                ),
            }
        )
    calibration_means = []
    for arm in arms:
        for method in ("unadapted", "intercept", "ridge_delta_head"):
            for budget in (0, 10, 30):
                rows = [
                    r
                    for r in calibration
                    if r["arm"] == arm and r["method"] == method and r["budget_seconds"] == budget
                ]
                calibration_means.append(
                    {
                        "arm": arm,
                        "method": method,
                        "budget_seconds": budget,
                        "mean_r2": float(np.mean([r["scores"]["r2_mean"] for r in rows])),
                    }
                )
    return comparison, calibration_means


def select_candidate(main, plan):
    """Locked paired development gate; select an arm, never a favorable seed."""
    reference = {
        (r["seed"], r["day"]): r["scores"]["r2_mean"]
        for r in main
        if r["arm"] == "rf15_standardized"
    }
    expected = {(seed, fold["evaluate"]) for seed in plan["seeds"] for fold in plan["folds"]}
    if set(reference) != expected:
        raise ValueError("candidate selection requires every baseline seed/day")
    decisions = []
    for order, arm in enumerate(plan["arms"]):
        rows = [r for r in main if r["arm"] == arm["name"]]
        if len(rows) != len(expected) or {(r["seed"], r["day"]) for r in rows} != expected:
            raise ValueError("candidate selection requires every arm seed/day")
        deltas = {
            key: next(r["scores"]["r2_mean"] for r in rows if (r["seed"], r["day"]) == key)
            - reference[key]
            for key in expected
        }
        day_deltas = {
            fold["evaluate"]: float(
                np.mean([value for (seed, day), value in deltas.items() if day == fold["evaluate"]])
            )
            for fold in plan["folds"]
        }
        mean_delta = float(np.mean(list(deltas.values())))
        decisions.append(
            {
                "arm": arm["name"],
                "mean_r2": float(np.mean([r["scores"]["r2_mean"] for r in rows])),
                "mean_gain": mean_delta,
                "day_mean_gains": day_deltas,
                "eligible": mean_delta >= 0.005 and min(day_deltas.values()) >= -0.01,
                "parameter_count": rows[0]["parameter_count"],
                "plan_order": order,
            }
        )
    eligible = [row for row in decisions if row["eligible"]]
    selected = (
        min(eligible, key=lambda row: (-row["mean_r2"], row["parameter_count"], row["plan_order"]))[
            "arm"
        ]
        if eligible
        else "rf15_standardized"
    )
    candidate = next(
        r for r in main if r["arm"] == selected and r["seed"] == 0 and r["day"] == "S4"
    )
    return {
        "selected_arm": selected,
        "decisions": decisions,
        "runtime_candidate": {
            "arm": selected,
            "seed": 0,
            "day": "S4",
            "model": candidate["model"],
            "receptive_field": candidate["receptive_field"],
            "parameter_count": candidate["parameter_count"],
        },
    }


def snapshot(root, output, plan_path):
    paths = [Path(__file__).resolve(), plan_path.resolve(), root / "pyproject.toml"]
    paths.extend(sorted((root / "src/neural_decoder").glob("*.py")))
    identities = {}
    for path in paths:
        relative = path.relative_to(root)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        identities[str(relative)] = sha256(destination)
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "files": identities,
    }


def run(plan_path, index_path, output, root):
    plan_path, index_path, output, root = map(Path, (plan_path, index_path, output, root))
    plan = json.loads(plan_path.read_text())
    if plan["role"] != "development_only; S0-S4; published S5-S12 never opened or used":
        raise ValueError("unsupported data role")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "plan_sha256": sha256(plan_path),
        "index_sha256": sha256(index_path),
        "source": snapshot(root.resolve(), output, plan_path),
        "environment": environment(),
        "scope": "development-only H1 V2; no latency claims; all failed/negative results retained",
    }
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "plan-lock.json",
        {
            "plan_sha256": manifest["plan_sha256"],
            "source": manifest["source"],
            "index_sha256": manifest["index_sha256"],
            "training_started": False,
        },
    )
    print(
        json.dumps(
            {"phase": "plan_locked", "plan_sha256": manifest["plan_sha256"], "output": str(output)}
        ),
        flush=True,
    )
    started = time.perf_counter()
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(plan["threads"])
        with threadpool_limits(limits=plan["threads"]):
            manifest["threadpools"] = threadpool_info()
            index = checked_development_index(index_path)
            write_json(output / "input_index.json", index)
            decision = run_schedule(plan, index_path, index, output)
            manifest["schedule_decision_sha256"] = sha256(output / "schedule-decision.json")
            write_json(output / "manifest.json", manifest)
            records = load_records(index_path, index)
            main, calibration, diagnostics = [], [], []
            epochs = decision["selected_epochs"]
            for arm in plan["arms"]:
                for seed in plan["seeds"]:
                    for fold in plan["folds"]:
                        day = fold["evaluate"]
                        selected = [r for r in records if r[0]["day"] in fold["train"]]
                        training = [(x, y, mask) for _, x, y, mask in selected]
                        evaluation = [r for r in records if r[0]["day"] == day]
                        folder = output / "main" / arm["name"] / f"seed-{seed}" / day
                        folder.mkdir(parents=True)
                        before = time.perf_counter()
                        model, transforms, losses, _ = fit_model(
                            training, **fit_arguments(plan, arm, seed, epochs)
                        )
                        write_json(folder / "losses.json", losses)
                        save_model(folder / "model.npz", model, transforms)
                        model, transforms, _ = load_model(folder / "model.npz")
                        scores, predictions, stream_error = evaluate(
                            model,
                            transforms,
                            evaluation,
                            folder,
                            full_stream_check=seed == 0 and day == "S2",
                        )
                        row = {
                            "arm": arm["name"],
                            "seed": seed,
                            "day": day,
                            "epochs": epochs,
                            "receptive_field": model.receptive_field,
                            "parameter_count": sum(p.numel() for p in model.parameters()),
                            "training_file_ids": [r[0]["file_id"] for r in selected],
                            "training_bins": sum(len(x) for x, _, _ in training),
                            "training_loss_rows": losses[-1]["scored_rows"],
                            "fit_and_evaluation_seconds": time.perf_counter() - before,
                            "model": artifact_identity(folder / "model.npz", output),
                            "losses": artifact_identity(folder / "losses.json", output),
                            "predictions": relative_predictions(predictions, output),
                            "scores": scores,
                            "full_stream_check": seed == 0 and day == "S2",
                            "stream_max_absolute_error": stream_error
                            if seed == 0 and day == "S2"
                            else None,
                        }
                        main.append(row)
                        calibration_folder = folder / "calibration"
                        calibration_folder.mkdir()
                        calibration.extend(
                            run_calibration(
                                model,
                                transforms,
                                training,
                                evaluation,
                                plan,
                                calibration_folder,
                                output,
                                {"arm": arm["name"], "seed": seed, "day": day},
                            )
                        )
                        write_json(output / "main.partial.json", main)
                        write_json(output / "calibration.partial.json", calibration)
                        print(
                            json.dumps(
                                {
                                    "phase": "main",
                                    "arm": arm["name"],
                                    "seed": seed,
                                    "day": day,
                                    "r2": scores["r2_mean"],
                                    "seconds": row["fit_and_evaluation_seconds"],
                                }
                            ),
                            flush=True,
                        )
            baseline = plan["arms"][0]
            for day in plan["diagnostics"]["days"]:
                day_records = [r for r in records if r[0]["day"] == day]
                earlier = [r for r in records if int(r[0]["day"][1:]) < int(day[1:])]
                for fraction in plan["diagnostics"]["fractions"]:
                    conditions, exposure = match_exposure(day_records[0], earlier, fraction)
                    for condition, training in conditions.items():
                        for seed in plan["seeds"]:
                            folder = (
                                output
                                / "diagnostics"
                                / day
                                / f"fraction-{fraction:g}"
                                / condition
                                / f"seed-{seed}"
                            )
                            folder.mkdir(parents=True)
                            model, transforms, losses, _ = fit_model(
                                training, **fit_arguments(plan, baseline, seed, epochs)
                            )
                            write_json(folder / "losses.json", losses)
                            save_model(folder / "model.npz", model, transforms)
                            scores, predictions, _ = evaluate(
                                model, transforms, day_records[-1:], folder
                            )
                            diagnostics.append(
                                {
                                    "day": day,
                                    "fraction": fraction,
                                    "condition": condition,
                                    "seed": seed,
                                    "epochs": epochs,
                                    "exposure": exposure,
                                    "scores": scores,
                                    "training_loss_rows": losses[-1]["scored_rows"],
                                    "model": artifact_identity(folder / "model.npz", output),
                                    "losses": artifact_identity(folder / "losses.json", output),
                                    "predictions": relative_predictions(predictions, output),
                                }
                            )
                    write_json(output / "diagnostics.partial.json", diagnostics)
                    print(
                        json.dumps(
                            {
                                "phase": "diagnostics",
                                "day": day,
                                "fraction": fraction,
                                "fits_done": len(diagnostics),
                            }
                        ),
                        flush=True,
                    )
            comparisons, calibration_means = aggregate(main, calibration)
            summary = {
                "schema_version": 1,
                "role": plan["role"],
                "plan_sha256": manifest["plan_sha256"],
                "index_sha256": manifest["index_sha256"],
                "source": manifest["source"],
                "selected_epochs": epochs,
                "schedule_decision": decision,
                "main": main,
                "arm_comparison": comparisons,
                "selection": select_candidate(main, plan),
                "diagnostics": diagnostics,
                "calibration": calibration,
                "calibration_means": calibration_means,
                "claims": (
                    "development comparisons only; not untouched H1 assessment "
                    "or cross-participant transfer"
                ),
            }
            write_json(output / "summary.json", summary)
            # Bind all private models/traces/curves, including closed-form calibration coefficients.
            artifacts = {
                str(path.relative_to(output)): sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name not in ("manifest.json", "artifact-hashes.json")
            }
            write_json(output / "artifact-hashes.json", artifacts)
            manifest.update(
                status="complete",
                summary_sha256=sha256(output / "summary.json"),
                artifacts_sha256=sha256(output / "artifact-hashes.json"),
                wall_seconds=time.perf_counter() - started,
            )
    except BaseException as error:
        manifest.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            wall_seconds=time.perf_counter() - started,
        )
        raise
    finally:
        torch.set_num_threads(old_threads)
        write_json(output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("accuracy_plan.json"))
    parser.add_argument("--index", type=Path, default=Path("data/neural_h1/prepared/index.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.index, args.output, Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    main()
