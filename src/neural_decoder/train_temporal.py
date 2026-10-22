"""One fixed small TCN recipe; forward-day development and bounded timing pilot."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from .artifacts import environment, sha256, source_identity, write_json
from .metrics import FileEvaluation, score_files
from .temporal import CausalTCN, TCNStream, export_tcn
from .train import FOLDS, load_development_files


def training_arrays(training, config):
    """Every loss chunk has 14 preceding real bins from the same file.

    First 14 bins/file are excluded from training loss only. Right padding is
    unscored. Every real eligible bin appears once/epoch; no file is concatenated.
    """
    x_all = np.concatenate([x for x, _, _ in training]).astype(np.float64)
    y_all = np.concatenate([y[m] for _, y, m in training]).astype(np.float64)
    transforms = {
        "input_mean": x_all.mean(0).astype(np.float32),
        "input_scale": np.maximum(x_all.std(0), 1e-6).astype(np.float32),
        "target_mean": y_all.mean(0).astype(np.float32),
        "target_scale": np.maximum(y_all.std(0), 1e-6).astype(np.float32),
    }
    burn, chunk = config["burn_in_bins"], config["chunk_scored_bins"]
    if burn != 14 or chunk < 1:
        raise ValueError("this TCN requires exactly 14 burn-in bins and positive chunks")
    xs, ys, masks = [], [], []
    for x, y, mask in training:
        x = (np.asarray(x, dtype=np.float32) - transforms["input_mean"]) / transforms["input_scale"]
        y = (np.asarray(y, dtype=np.float32) - transforms["target_mean"]) / transforms[
            "target_scale"
        ]
        for start in range(burn, len(x), chunk):
            stop = min(start + chunk, len(x))
            length = stop - start + burn
            chunk_x = np.zeros((burn + chunk, 176), dtype=np.float32)
            chunk_y = np.zeros((burn + chunk, 7), dtype=np.float32)
            chunk_mask = np.zeros(burn + chunk, dtype=bool)
            chunk_x[:length] = x[start - burn : stop]
            chunk_y[:length] = y[start - burn : stop]
            chunk_mask[burn:length] = mask[start:stop]
            if chunk_mask.any():
                xs.append(chunk_x)
                ys.append(chunk_y)
                masks.append(chunk_mask)
    if not xs:
        raise ValueError("no eligible training chunks")
    return transforms, tuple(torch.from_numpy(np.array(v)) for v in (xs, ys, masks))


def fit_temporal(training, config, seed, epochs):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    generator = np.random.default_rng(seed)
    transforms, (x, y, mask) = training_arrays(training, config)
    model = CausalTCN()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    losses = []
    for epoch in range(epochs):
        model.train()
        order = generator.permutation(len(x))
        sum_loss, samples = 0.0, 0
        for offset in range(0, len(x), config["batch_chunks"]):
            indices = order[offset : offset + config["batch_chunks"]]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x[indices])
            selected = mask[indices]
            error = prediction[selected] - y[indices][selected]
            loss = error.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite TCN training loss")
            loss.backward()
            optimizer.step()
            sum_loss += loss.item() * error.numel()
            samples += error.numel()
        losses.append({"epoch": epoch + 1, "masked_standardized_mse": sum_loss / samples})
    return model.eval(), transforms, losses


def evaluate_temporal(model, frozen, files, destination):
    records, parity = [], {}
    with torch.inference_mode():
        for item, x, y, mask in files:
            decoder = TCNStream(frozen, implementation="incremental")
            predictions = decoder.predict_chunk(x)
            values = (
                np.asarray(x, dtype=np.float32) - frozen.arrays["input_mean"]
            ) / frozen.arrays["input_scale"]
            reference = model(torch.from_numpy(values)[None])[0].numpy()
            reference = reference * frozen.arrays["target_scale"] + frozen.arrays["target_mean"]
            np.testing.assert_allclose(predictions, reference, rtol=3e-5, atol=3e-5)
            parity[item["file_id"]] = float(np.max(np.abs(predictions - reference)))
            np.save(destination / f"{item['file_id']}.npy", predictions, allow_pickle=False)
            records.append(FileEvaluation(item["file_id"], item["day"], predictions, y, mask))
    return records, parity


def run_temporal(index_path, config_path, output, *, root, pilot=False):
    config = json.loads(Path(config_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "pilot": pilot,
        "configuration": config,
        "environment": environment(),
        "source": source_identity(root, output),
        "index_sha256": sha256(index_path),
        "config_sha256": sha256(config_path),
    }
    write_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config["threads"])
        index, files = load_development_files(index_path)
        write_json(output / "input_index.json", index)
        summaries = []
        with threadpool_limits(limits=config["threads"]):
            manifest["threadpools"] = threadpool_info()
            for seed in config["seeds"][:1] if pilot else config["seeds"]:
                combined = []
                for train_days, day in FOLDS[:1] if pilot else FOLDS:
                    folder = output / f"seed-{seed}" / day
                    folder.mkdir(parents=True)
                    training = [(x, y, m) for item, x, y, m in files if item["day"] in train_days]
                    training_start = time.perf_counter()
                    model, transforms, losses = fit_temporal(
                        training, config, seed, 1 if pilot else config["epochs"]
                    )
                    training_seconds = time.perf_counter() - training_start
                    frozen = export_tcn(model, folder / "model.npz", **transforms)
                    write_json(folder / "losses.json", losses)
                    identity = {
                        "train_days": train_days,
                        "evaluation_day": None if pilot else day,
                        "model_sha256": sha256(folder / "model.npz"),
                        "training_seconds": training_seconds,
                        "parameter_count": frozen.parameter_count,
                    }
                    if not pilot:
                        evaluation = [record for record in files if record[0]["day"] == day]
                        records, parity = evaluate_temporal(model, frozen, evaluation, folder)
                        scores = score_files(records)
                        write_json(folder / "scores.json", scores)
                        identity["parity_max_absolute_error"] = parity
                        identity["prediction_sha256"] = {
                            r.file_id: sha256(folder / f"{r.file_id}.npy") for r in records
                        }
                        combined.extend(records)
                    write_json(folder / "identity.json", identity)
                    print(
                        json.dumps(
                            {
                                "seed": seed,
                                "day": day,
                                "pilot": pilot,
                                "training_seconds": training_seconds,
                                "r2": None if pilot else scores["r2_mean"],
                            }
                        ),
                        flush=True,
                    )
                if not pilot:
                    summaries.append({"seed": seed, "scores": score_files(combined)})
                    write_json(output / "summary.partial.json", summaries)
        write_json(
            output / "summary.json",
            {
                "seeds": summaries,
                "pilot": pilot,
                "sealed_days_scored": [],
                "selection_role": "development_only",
            },
        )
        manifest.update(
            {
                "status": "complete",
                "wall_seconds": time.perf_counter() - started,
                "summary_sha256": sha256(output / "summary.json"),
            }
        )
    except BaseException as exc:
        manifest.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        torch.set_num_threads(previous_threads)
        write_json(output / "manifest.json", manifest)
    return output
