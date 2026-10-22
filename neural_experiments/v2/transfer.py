"""Participant-held-out, few-shot causal decoding with explicit recording adapters.

Data/labels are separate on disk. Training and evaluation have separate APIs;
the incremental decoder consumes one feature vector and never receives labels.
The plan is frozen in the run manifest before fitting. V1 code is not modified.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mask_digest(mask):
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=bool).tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def resolve(path):
    return ROOT / path if not Path(path).is_absolute() else Path(path)


@dataclass
class Recording:
    metadata: dict
    x: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    segments: list[tuple[int, int]]

    @property
    def session_id(self):
        return self.metadata["session_id"]

    @property
    def participant_id(self):
        return self.metadata["participant_id"]

    def prefix(self, stop):
        stop = min(int(stop), len(self.x))
        return Recording(
            self.metadata,
            self.x[:stop],
            self.y[:stop],
            self.mask[:stop],
            [(start, min(end, stop)) for start, end in self.segments if start < stop],
        )


def load_recordings(index_path):
    from neural_experiments.v2.data import load_features, load_index

    index = load_index(index_path)
    cache = Path(index_path).resolve().parent
    records = []
    for item in index["sessions"]:
        feature_path, target_path = cache / item["feature_path"], cache / item["target_path"]
        for path, key in ((feature_path, "feature_sha256"), (target_path, "target_sha256")):
            if digest(path) != item[key]:
                raise ValueError(f"changed prepared artifact: {path}")
        x = load_features(feature_path)["x"]
        with np.load(target_path, allow_pickle=False) as values:
            y, mask = values["y"].astype(np.float32), values["eval_mask"].astype(bool)
        if x.ndim != 2 or y.shape != (len(x), 2) or mask.shape != (len(x),):
            raise ValueError("invalid prepared recording shape")
        if not np.isfinite(x).all() or np.any(x < 0):
            raise ValueError("neural counts must be finite and nonnegative")
        if not np.isfinite(y[mask]).all():
            raise ValueError("scored targets must be finite")
        segments = [(int(s["start"]), int(s["end"])) for s in item["segments"]]
        if not segments or segments[0][0] != 0 or segments[-1][1] != len(x):
            raise ValueError("recording blocks must cover the complete prepared sequence")
        if any(a > b for a, b in segments) or any(
            a[1] != b[0] for a, b in zip(segments, segments[1:])
        ):
            raise ValueError("invalid or overlapping recording blocks")
        records.append(Recording(item, x, y, mask, segments))
    return index, records


def fit_transform(recordings):
    """Caller supplies training-only records or an already clipped calibration prefix."""
    values = np.concatenate([np.sqrt(r.x) for r in recordings]).astype(np.float64)
    if not len(values):
        raise ValueError("normalization requires an observed prefix")
    return values.mean(0).astype(np.float32), np.maximum(values.std(0), 0.1).astype(np.float32)


def transform(x, stats):
    return (np.sqrt(x) - stats[0]) / stats[1]


class AdapterTCN(nn.Module):
    """Participant adapter -> shared causal temporal core -> participant readout."""

    def __init__(self, channels, width=32, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        self.width, self.dilations = width, tuple(dilations)
        self.adapter = nn.Linear(channels, width)
        self.core = nn.ModuleList([nn.Conv1d(width, width, 3, dilation=d) for d in dilations])
        self.head = nn.Linear(width, 2)

    @property
    def receptive_field(self):
        return 1 + 2 * sum(self.dilations)

    def encode(self, x):
        hidden = F.relu(self.adapter(x)).transpose(1, 2)
        for layer, dilation in zip(self.core, self.dilations, strict=True):
            hidden = F.relu(hidden + layer(F.pad(hidden, (2 * dilation, 0))))
        return hidden.transpose(1, 2)

    def forward(self, x):
        return self.head(self.encode(x))


class SharedTCN(nn.Module):
    def __init__(self, channel_counts, width, dilations):
        super().__init__()
        self.models = nn.ModuleDict(
            {person: AdapterTCN(n, width, dilations) for person, n in channel_counts.items()}
        )
        first = next(iter(self.models.values()))
        for model in self.models.values():
            model.core = first.core


def make_chunks(recordings, stats, burn=62, length=256):
    """Each loss chunk has real preceding context from exactly one recording block."""
    xs, ys, masks = [], [], []
    for record in recordings:
        values = transform(record.x, stats)
        for block_start, block_end in record.segments:
            for start in range(block_start + burn, block_end, length):
                stop = min(start + length, block_end)
                valid = stop - start + burn
                x = np.zeros((burn + length, values.shape[1]), dtype=np.float32)
                y = np.zeros((burn + length, 2), dtype=np.float32)
                mask = np.zeros(burn + length, dtype=bool)
                x[:valid] = values[start - burn : stop]
                y[:valid] = record.y[start - burn : stop]
                mask[burn:valid] = record.mask[start:stop]
                if mask.any():
                    xs.append(x)
                    ys.append(y)
                    masks.append(mask)
    if not xs:
        raise ValueError("no scored training chunks after causal burn-in")
    return tuple(torch.from_numpy(np.stack(v)) for v in (xs, ys, masks))


def training_mask(record, burn):
    mask = record.mask.copy()
    for start, stop in record.segments:
        mask[start : min(start + burn, stop)] = False
    return mask


def train_batch(model, chunks, indices, optimizer, dropout):
    x, y, mask = (v[indices] for v in chunks)
    optimizer.zero_grad(set_to_none=True)
    keep = (torch.rand((len(x), 1, x.shape[2])) >= dropout).to(x.dtype)
    prediction = model(x * keep)
    error = prediction[mask] - y[mask]
    loss = error.square().mean()
    if not torch.isfinite(loss):
        raise ValueError("nonfinite optimization loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach()), int(mask.sum())


@torch.inference_mode()
def predict(model, x, segments, stats):
    """Feature-only inference; neither targets nor scoring masks enter this API."""
    model.eval()
    output = np.empty((len(x), 2), dtype=np.float32)
    burn = model.receptive_field - 1
    values = transform(x, stats)
    # Bounded inference workspace, with true context at each partition boundary.
    for block_start, block_end in segments:
        for start in range(block_start, block_end, 4096):
            stop = min(start + 4096, block_end)
            context = max(block_start, start - burn)
            output[start:stop] = model(torch.from_numpy(values[context:stop])[None])[0].numpy()[
                start - context :
            ]
    return output


def score(y, prediction, mask):
    actual, estimated = np.asarray(y[mask], float), np.asarray(prediction[mask], float)
    if not len(actual):
        return {
            "r2": None,
            "per_output_r2": [None, None],
            "n": 0,
            "angular_error_degrees": None,
            "zero_prediction_count": 0,
        }
    if not np.isfinite(actual).all() or not np.isfinite(estimated).all():
        raise ValueError("nonfinite scored values")
    sse = np.square(actual - estimated).sum(0)
    sst = np.square(actual - actual.mean(0)).sum(0)
    pred_norm, actual_norm = np.linalg.norm(estimated, axis=1), np.linalg.norm(actual, axis=1)
    denominator = pred_norm * actual_norm
    cosine = np.divide(
        (actual * estimated).sum(1),
        denominator,
        out=np.zeros(len(actual)),
        where=denominator > 1e-12,
    )
    return {
        "r2": float(1 - sse.sum() / sst.sum()) if sst.sum() > 1e-12 else None,
        "per_output_r2": [
            float(1 - e / v) if v > 1e-12 else None for e, v in zip(sse, sst, strict=True)
        ],
        "n": len(actual),
        "angular_error_degrees": float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).mean()),
        "zero_prediction_count": int(np.count_nonzero(pred_norm <= 1e-12)),
    }


def pretrain(training, validation, plan, seed, output):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    people = sorted(training)
    stats = {p: fit_transform(training[p]) for p in people}
    width, dilations = plan["model"]["width"], plan["model"]["dilations"]
    models = SharedTCN({p: training[p][0].x.shape[1] for p in people}, width, dilations)
    cfg = plan["source_training"]
    chunks = {
        p: make_chunks(training[p], stats[p], 2 * sum(dilations), cfg["chunk_bins"]) for p in people
    }
    optimizer = torch.optim.AdamW(
        models.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    losses, checkpoints = [], []
    best_state, best_value, best_step = None, math.inf, None
    for step in range(1, cfg["steps"] + 1):
        person = people[(step - 1) % len(people)]
        available = len(chunks[person][0])
        indices = rng.choice(available, size=min(cfg["batch_chunks"], available), replace=False)
        loss, samples = train_batch(
            models.models[person],
            chunks[person],
            indices,
            optimizer,
            plan["model"]["channel_dropout"],
        )
        losses.append({"step": step, "participant": person, "loss": loss, "n": samples})
        if step in cfg["checkpoints"]:
            values = []
            for p in people:
                errors = []
                for record in validation.get(p, []):
                    prediction = predict(models.models[p], record.x, record.segments, stats[p])
                    if record.mask.any():
                        errors.append(
                            float(np.square(prediction[record.mask] - record.y[record.mask]).mean())
                        )
                if errors:
                    values.append(float(np.mean(errors)))
            value = float(np.mean(values)) if values else float(loss)
            checkpoints.append(
                {
                    "step": step,
                    "validation_equal_person_mse": value,
                    "validation_people": len(values),
                }
            )
            if value < best_value:
                best_value, best_step = value, step
                best_state = copy.deepcopy(next(iter(models.models.values())).core.state_dict())
            models.train()
    if best_state is None:
        raise ValueError("no prespecified checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "core.npz", **{k: v.numpy() for k, v in best_state.items()})
    np.savez(
        output / "source_normalization.npz",
        **{
            f"{person}__{name}": values
            for person in people
            for name, values in zip(("input_mean", "input_scale"), stats[person], strict=True)
        },
    )
    write_json(
        output / "training.json",
        {
            "seed": seed,
            "source_people": people,
            "source_train_session_ids": [r.session_id for rr in training.values() for r in rr],
            "source_validation_session_ids": [
                r.session_id for rr in validation.values() for r in rr
            ],
            "source_normalization": {
                person: [
                    {"session_id": r.session_id, "start_bin": 0, "stop_bin": len(r.x)}
                    for r in training[person]
                ]
                for person in people
            },
            "source_normalization_path": relative(output / "source_normalization.npz"),
            "source_normalization_sha256": digest(output / "source_normalization.npz"),
            "losses": losses,
            "checkpoints": checkpoints,
            "selected_step": best_step,
            "core_sha256": digest(output / "core.npz"),
        },
    )
    return best_state


def adapt(prefix, plan, seed, method, core_state):
    torch.manual_seed(seed + 10000)
    rng = np.random.default_rng(seed + 10000)
    model = AdapterTCN(prefix.x.shape[1], plan["model"]["width"], plan["model"]["dilations"])
    if method != "scratch":
        model.core.load_state_dict(core_state)
    if method == "pretrained_frozen_core":
        for parameter in model.core.parameters():
            parameter.requires_grad_(False)
    stats = fit_transform([prefix])
    cfg = plan["adaptation"]
    chunks = make_chunks([prefix], stats, model.receptive_field - 1, cfg["chunk_bins"])
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    history = []
    for epoch in range(cfg["epochs"]):
        model.train()
        permutation = rng.permutation(len(chunks[0]))
        weighted_loss, count = 0.0, 0
        for start in range(0, len(permutation), cfg["batch_chunks"]):
            loss, n = train_batch(
                model,
                chunks,
                permutation[start : start + cfg["batch_chunks"]],
                optimizer,
                plan["model"]["channel_dropout"],
            )
            weighted_loss += loss * n
            count += n
        history.append({"epoch": epoch + 1, "mse": weighted_loss / count})
    return model.eval(), stats, history


def export_model(model, stats, path):
    arrays = {key: value.detach().numpy() for key, value in model.state_dict().items()}
    arrays.update(
        input_mean=stats[0],
        input_scale=stats[1],
        dilations=np.asarray(model.dilations, dtype=np.int64),
        format_version=np.asarray(2, dtype=np.int64),
    )
    with Path(path).open("xb") as stream:
        np.savez(stream, **arrays)


def load_model(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    version = arrays.pop("format_version")
    if version.shape != () or version.dtype != np.int64 or version.item() != 2:
        raise ValueError("unknown model artifact")
    stats = (arrays.pop("input_mean"), arrays.pop("input_scale"))
    dilation_array = arrays.pop("dilations")
    if dilation_array.dtype != np.int64 or dilation_array.ndim != 1:
        raise ValueError("invalid dilation metadata")
    dilations = tuple(int(d) for d in dilation_array)
    if (
        not dilations
        or any(d < 1 for d in dilations)
        or len(dilations) > 8
        or max(dilations) > 128
        or any(a.dtype != np.float32 or not np.isfinite(a).all() for a in arrays.values())
    ):
        raise ValueError("invalid model artifact")
    width, channels = arrays["adapter.weight"].shape
    if (
        not 1 <= width <= 128
        or not 1 <= channels <= 512
        or any(
            a.shape != (channels,) or a.dtype != np.float32 or not np.isfinite(a).all()
            for a in stats
        )
        or not np.all(stats[1] >= 0.1)
    ):
        raise ValueError("invalid model normalization or geometry")
    with torch.random.fork_rng(devices=[]):
        model = AdapterTCN(channels, width, dilations)
    model.load_state_dict({k: torch.from_numpy(v) for k, v in arrays.items()}, strict=True)
    return model.eval(), stats


class IncrementalDecoder:
    """Bounded ring state; complete step includes validation, transforms and readout."""

    def __init__(self, model, stats):
        self.stats = tuple(np.asarray(a, dtype=np.float32).copy() for a in stats)
        self.arrays = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}
        self.dilations = model.dilations
        self.rings = [np.zeros((2 * d + 1, model.width), dtype=np.float32) for d in self.dilations]
        self.reset()

    def reset(self):
        for ring in self.rings:
            ring.fill(0)
        self.cursors = [0] * len(self.rings)
        self.failed = False

    @property
    def state_nbytes(self):
        return sum(r.nbytes for r in self.rings)

    def step(self, x):
        if self.failed:
            raise RuntimeError("reset required after decoder failure")
        x = np.asarray(x)
        if x.shape != self.stats[0].shape or not np.isfinite(x).all() or np.any(x < 0):
            raise ValueError("expected finite nonnegative channel-count vector")
        a = self.arrays
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                h = np.maximum(
                    a["adapter.weight"] @ transform(x.astype(np.float32), self.stats)
                    + a["adapter.bias"],
                    0,
                )
                for i, (d, ring) in enumerate(zip(self.dilations, self.rings, strict=True)):
                    cursor = self.cursors[i]
                    ring[cursor] = h
                    w = a[f"core.{i}.weight"]
                    h = np.maximum(
                        h
                        + w[:, :, 0] @ ring[(cursor - 2 * d) % len(ring)]
                        + w[:, :, 1] @ ring[(cursor - d) % len(ring)]
                        + w[:, :, 2] @ h
                        + a[f"core.{i}.bias"],
                        0,
                    )
                    self.cursors[i] = (cursor + 1) % len(ring)
                y = a["head.weight"] @ h + a["head.bias"]
                if not np.isfinite(y).all():
                    raise FloatingPointError("nonfinite prediction")
                return y
        except FloatingPointError:
            self.failed = True
            raise


def filtered(x, segments, stats, config):
    from scipy.signal import lfilter

    weights = np.exp(-np.arange(config["fir_taps"]) / config["fir_tau_bins"])
    weights /= weights.sum()
    output = np.empty_like(x)
    values = transform(x, stats)
    for start, stop in segments:
        if start < stop:
            output[start:stop] = lfilter(weights, [1.0], values[start:stop], axis=0)
    return output


def fit_ridge(prefix, config, burn=62):
    from sklearn.linear_model import Ridge

    stats = fit_transform([prefix])
    x = filtered(prefix.x, prefix.segments, stats, config)
    model = Ridge(alpha=config["alpha"], solver="svd")
    mask = training_mask(prefix, burn)
    model.fit(x[mask], prefix.y[mask])
    return model, stats


def evaluate(model, stats, records, calibration_id, plan, destination, method):
    traces = []
    cap = round(plan["evaluation_session_cap_seconds"] / plan["bin_seconds"])
    suffix = round(plan["common_suffix_start_seconds"] / plan["bin_seconds"])
    for original in records:
        record = original.prefix(cap)
        start = suffix if record.session_id == calibration_id else 0
        partition = "same_session_suffix" if start else "future_day"
        start = min(start, len(record.x))
        mask = record.mask.copy()
        mask[:start] = False
        if not len(record.x):
            prediction = np.empty((0, 2), dtype=np.float32)
        elif method == "ridge":
            prediction = model.predict(filtered(record.x, record.segments, stats, plan["ridge"]))
        else:
            prediction = predict(model, record.x, record.segments, stats)
        path = destination / f"{record.session_id}.npy"
        np.save(path, prediction, allow_pickle=False)
        traces.append(
            {
                "session_id": record.session_id,
                "partition": partition,
                "start_bin": start,
                "stop_bin": len(record.x),
                "prediction_path": relative(path),
                "prediction_sha256": digest(path),
                "score_mask_sha256": mask_digest(mask),
                "metrics": score(record.y, prediction, mask)
                if partition == "future_day" or mask.sum() >= plan["minimum_suffix_scored_bins"]
                else {
                    "r2": None,
                    "per_output_r2": [None, None],
                    "angular_error_degrees": None,
                    "zero_prediction_count": 0,
                    "n": int(mask.sum()),
                    "unavailable_reason": "fewer than minimum scored evaluation bins",
                },
            }
        )
    return traces


def summarize(manifest, plan):
    rows = []
    for reference in manifest["models"]:
        identity = json.loads(resolve(reference["identity_path"]).read_text())
        for partition in ("same_session_suffix", "future_day"):
            scores = [
                t["metrics"]["r2"]
                for t in identity["traces"]
                if t["partition"] == partition and t["metrics"]["r2"] is not None
            ]
            if scores:
                rows.append(
                    {
                        "participant": identity["target_participant"],
                        "method": identity["method"],
                        "seed": identity["seed"],
                        "budget_seconds": identity["budget_seconds"],
                        "partition": partition,
                        "mean_day_r2": float(np.mean(scores)),
                        "n_days": len(scores),
                    }
                )
    groups = []
    for partition in ("same_session_suffix", "future_day"):
        for budget in plan["calibration_budgets_seconds"]:
            for method in plan["methods"]:
                people = []
                for person in plan["participants"]:
                    selected = [
                        r["mean_day_r2"]
                        for r in rows
                        if r["participant"] == person
                        and r["method"] == method
                        and r["budget_seconds"] == budget
                        and r["partition"] == partition
                    ]
                    if selected:
                        people.append(
                            {
                                "participant": person,
                                "r2": float(np.mean(selected)),
                                "n_seeds": len(selected),
                            }
                        )
                groups.append(
                    {
                        "partition": partition,
                        "budget_seconds": budget,
                        "method": method,
                        "people": people,
                        "mean_person_r2": float(np.mean([p["r2"] for p in people]))
                        if people
                        else None,
                    }
                )
    effects = []
    for person in plan["participants"]:
        values = {}
        for method in ("scratch", "pretrained_frozen_core"):
            selected = [
                r["mean_day_r2"]
                for r in rows
                if r["participant"] == person
                and r["method"] == method
                and r["budget_seconds"] == 120
                and r["partition"] == "future_day"
            ]
            if len(selected) == len(plan["seeds"]):
                values[method] = float(np.mean(selected))
        if len(values) == 2:
            effects.append(
                {
                    "participant": person,
                    "difference": values["pretrained_frozen_core"] - values["scratch"],
                }
            )
    inference = {"effects": effects, "n_people": len(effects), "mean_difference": None}
    if effects:
        d = np.array([e["difference"] for e in effects])
        rng = np.random.default_rng(20261022)
        bootstrap = rng.choice(d, (10000, len(d)), replace=True).mean(1)
        signs = 2 * ((np.arange(2 ** len(d))[:, None] >> np.arange(len(d))) & 1) - 1
        p = np.mean(np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12)
        inference.update(
            mean_difference=float(d.mean()),
            bootstrap_percentile_95=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            exact_two_sided_sign_flip_p=float(p),
            positive_people=int(np.sum(d > 0)),
            target_gain_met=bool(d.mean() >= plan["success_target"]["absolute_r2_gain"]),
        )
    return {
        "status": manifest["status"],
        "rows": rows,
        "groups": groups,
        "primary_contrast": inference,
        "unavailable": manifest["unavailable"],
        "manifest_sha256": manifest.get("manifest_sha256"),
    }


def mark_unavailable(manifest, plan, person, budget, seed, reason):
    for method in plan["methods"]:
        if method == "ridge" and seed != plan["seeds"][0]:
            continue
        manifest["unavailable"].append(
            {
                "target_participant": person,
                "method": method,
                "seed": seed,
                "budget_seconds": budget,
                "reason": reason,
            }
        )


def run(index_path, plan_path, output):
    plan = json.loads(Path(plan_path).read_text())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    index, recordings = load_recordings(index_path)
    by_person = {
        p: sorted(
            [r for r in recordings if r.participant_id == p],
            key=lambda r: (r.metadata["post_implant_day"], r.session_id),
        )
        for p in plan["participants"]
    }
    manifest = {
        "status": "running",
        "created_unix": time.time(),
        "config_path": relative(plan_path),
        "config_sha256": digest(plan_path),
        "index_path": relative(index_path),
        "index_sha256": digest(index_path),
        "source": {
            relative(p): digest(p) for p in (Path(__file__), Path(__file__).with_name("data.py"))
        },
        "source_snapshots": {},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "threads": 2,
        },
        "models": [],
        "unavailable": [],
        "pretraining": [],
    }
    for original in manifest["source"]:
        snapshot = output / "source" / Path(original).name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resolve(original), snapshot)
        manifest["source_snapshots"][original] = relative(snapshot)
    write_json(output / "manifest.json", manifest)
    write_json(output / "plan.json", plan)
    write_json(output / "input_index.json", index)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    cap = round(plan["source_session_cap_seconds"] / plan["bin_seconds"])
    with threadpool_limits(limits=2):
        for target in plan["participants"]:
            if not by_person[target]:
                for seed in plan["seeds"]:
                    for budget in plan["calibration_budgets_seconds"]:
                        mark_unavailable(
                            manifest, plan, target, budget, seed, "no acquired recordings"
                        )
                continue
            target_records = by_person[target]
            calibration = target_records[0]
            training, validation = {}, {}
            for person, records in by_person.items():
                if person == target or not records:
                    continue
                channel_counts = {r.x.shape[1] for r in records}
                if len(channel_counts) != 1:
                    raise ValueError(f"channel schema changed within participant {person}")
                training[person] = [r.prefix(cap) for r in (records[:-1] or records)]
                validation[person] = [records[-1].prefix(cap)] if len(records) > 1 else []
            if len(training) < 1:
                raise ValueError("participant transfer requires a different source person")
            for seed in plan["seeds"]:
                pretrain_folder = output / "pretrain" / target / f"seed-{seed}"
                started = time.perf_counter()
                core = pretrain(training, validation, plan, seed, pretrain_folder)
                manifest["pretraining"].append(
                    {
                        "target_participant": target,
                        "seed": seed,
                        "training_seconds": time.perf_counter() - started,
                        "receipt_path": relative(pretrain_folder / "training.json"),
                        "receipt_sha256": digest(pretrain_folder / "training.json"),
                    }
                )
                for budget in plan["calibration_budgets_seconds"]:
                    stop = round(budget / plan["bin_seconds"])
                    if stop > len(calibration.x):
                        mark_unavailable(
                            manifest,
                            plan,
                            target,
                            budget,
                            seed,
                            "earliest session shorter than prefix",
                        )
                        continue
                    prefix = calibration.prefix(stop)
                    effective_mask = training_mask(prefix, 2 * sum(plan["model"]["dilations"]))
                    if effective_mask.sum() < 50:
                        mark_unavailable(
                            manifest,
                            plan,
                            target,
                            budget,
                            seed,
                            "fewer than50 scored prefix bins after common burn-in",
                        )
                        continue
                    for method in plan["methods"]:
                        if method == "ridge" and seed != plan["seeds"][0]:
                            continue
                        model_id = f"{target}-{method}-{budget}s-seed{seed}"
                        folder = output / "models" / model_id
                        folder.mkdir(parents=True, exist_ok=False)
                        started = time.perf_counter()
                        if method == "ridge":
                            model, stats = fit_ridge(
                                prefix, plan["ridge"], 2 * sum(plan["model"]["dilations"])
                            )
                            history = []
                            np.savez(
                                folder / "model.npz",
                                coefficient=model.coef_,
                                intercept=model.intercept_,
                                input_mean=stats[0],
                                input_scale=stats[1],
                            )
                        else:
                            model, stats, history = adapt(prefix, plan, seed, method, core)
                            export_model(model, stats, folder / "model.npz")
                            # Score the deployed artifact, not a separate in-memory training object.
                            model, stats = load_model(folder / "model.npz")
                        elapsed = time.perf_counter() - started
                        write_json(folder / "training.json", history)
                        identity = {
                            "model_id": model_id,
                            "target_participant": target,
                            "method": method,
                            "seed": seed,
                            "budget_seconds": budget,
                            "source_participants": sorted(training)
                            if method.startswith("pretrained")
                            else [],
                            "source_train_session_ids": [
                                r.session_id for rr in training.values() for r in rr
                            ]
                            if method.startswith("pretrained")
                            else [],
                            "source_validation_session_ids": [
                                r.session_id for rr in validation.values() for r in rr
                            ]
                            if method.startswith("pretrained")
                            else [],
                            "calibration": {
                                "session_id": calibration.session_id,
                                "start_bin": 0,
                                "stop_bin": stop,
                                "eligible_count": int(prefix.mask.sum()),
                                "mask_sha256": mask_digest(prefix.mask),
                                "training_count": int(effective_mask.sum()),
                                "training_mask_sha256": mask_digest(effective_mask),
                            },
                            "normalization": {
                                "session_id": calibration.session_id,
                                "start_bin": 0,
                                "stop_bin": stop,
                            },
                            "model_path": relative(folder / "model.npz"),
                            "model_sha256": digest(folder / "model.npz"),
                            "core_sha256": digest(pretrain_folder / "core.npz")
                            if method.startswith("pretrained")
                            else None,
                            "training_seconds": elapsed,
                            "parameter_count": int(sum(p.numel() for p in model.parameters()))
                            if method != "ridge"
                            else int(model.coef_.size + model.intercept_.size),
                            "traces": evaluate(
                                model,
                                stats,
                                target_records,
                                calibration.session_id,
                                plan,
                                folder,
                                method,
                            ),
                        }
                        write_json(folder / "identity.json", identity)
                        manifest["models"].append(
                            {
                                "identity_path": relative(folder / "identity.json"),
                                "identity_sha256": digest(folder / "identity.json"),
                            }
                        )
                        write_json(output / "manifest.json", manifest)
                        write_json(output / "summary.partial.json", summarize(manifest, plan))
                        print(
                            json.dumps(
                                {
                                    "model": model_id,
                                    "training_seconds": elapsed,
                                    "traces": len(identity["traces"]),
                                }
                            ),
                            flush=True,
                        )
    manifest["status"] = "complete"
    manifest["finished_unix"] = time.time()
    write_json(output / "manifest.json", manifest)
    summary = summarize(manifest, plan)
    summary["manifest_sha256"] = digest(output / "manifest.json")
    write_json(output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--plan", default=str(Path(__file__).with_name("transfer_plan.json")))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.index, args.plan, args.output)


if __name__ == "__main__":
    main()
