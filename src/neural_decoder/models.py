"""Small safe numeric artifacts for target-mean and causal ridge decoders."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve

from .features import CausalFeatures, filter_file, fir_weights, history_matrix


@dataclass
class LinearModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    coefficient: np.ndarray
    intercept: np.ndarray
    history: int
    kind: str = "ridge"

    def validate(self):
        CausalFeatures(self.mean, self.scale, self.history, self.weights)
        if self.kind not in ("ridge", "constant"):
            raise ValueError("unsupported linear model kind")
        if self.coefficient.shape != (self.history * len(self.mean), len(self.intercept)):
            raise ValueError("coefficient shape disagrees with feature contract")
        if self.intercept.ndim != 1 or not len(self.intercept):
            raise ValueError("invalid intercept")
        if not np.isfinite(self.coefficient).all() or not np.isfinite(self.intercept).all():
            raise ValueError("nonfinite model")
        return self

    def save(self, path):
        self.validate()
        np.savez(
            path,
            schema_version=np.array(1),
            kind=np.array(self.kind),
            mean=self.mean,
            scale=self.scale,
            weights=self.weights,
            coefficient=self.coefficient,
            intercept=self.intercept,
            history=np.array(self.history),
        )


def load_model(path):
    with np.load(Path(path), allow_pickle=False) as item:
        expected = {
            "schema_version",
            "kind",
            "mean",
            "scale",
            "weights",
            "coefficient",
            "intercept",
            "history",
        }
        if set(item.files) != expected or item["schema_version"].item() != 1:
            raise ValueError("unsupported model format")
        return LinearModel(
            mean=item["mean"],
            scale=item["scale"],
            weights=item["weights"],
            coefficient=item["coefficient"],
            intercept=item["intercept"],
            history=int(item["history"].item()),
            kind=str(item["kind"].item()),
        ).validate()


def fit_linear(files, *, history=1, regularization=0.01, constant=False):
    """files are authorized training tuples (counts, targets, boolean loss mask).

    Minimize mean squared residual plus regularization * squared coefficient norm.
    Fit normalization to all training neural bins; fit regression to scored bins.
    No target-conditioned still-period removal or per-evaluation-file statistics.
    """
    if not files or regularization <= 0:
        raise ValueError("training files and positive regularization required")
    weights = fir_weights()
    filtered = [filter_file(x, weights) for x, _, _ in files]
    all_filtered = np.concatenate(filtered)
    mean = all_filtered.mean(axis=0)
    scale = np.maximum(all_filtered.std(axis=0), 1e-6)
    xs, ys = [], []
    for values, (_, target, mask) in zip(filtered, files, strict=True):
        if mask.dtype != np.bool_ or mask.shape != (len(values),):
            raise ValueError("boolean training mask must align with counts")
        if target.ndim != 2 or len(target) != len(values) or not np.isfinite(target).all():
            raise ValueError("invalid training targets")
        xs.append(history_matrix(values, mean, scale, history)[mask])
        ys.append(target[mask])
    x, y = np.concatenate(xs), np.concatenate(ys)
    if len(x) < 2:
        raise ValueError("insufficient training samples")
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    if constant:
        coef = np.zeros((x.shape[1], y.shape[1]))
    else:
        x -= x_mean
        centered_y = y - y_mean
        gram = x.T @ x / len(x)
        gram.flat[:: len(gram) + 1] += regularization
        coef = solve(gram, x.T @ centered_y / len(x), assume_a="pos")
    intercept = y_mean - x_mean @ coef
    return LinearModel(
        mean, scale, weights, coef, intercept, history, "constant" if constant else "ridge"
    ).validate()


class LinearDecoder:
    """Feature-only batch-one decoder. Trials never reset this state."""

    def __init__(self, model):
        self.model = model.validate()
        self.features = CausalFeatures(model.mean, model.scale, model.history, model.weights)

    def reset(self):
        self.features.reset()

    def step(self, counts, sequence_id=None):
        x = self.features.step(counts, sequence_id)
        return x @ self.model.coefficient + self.model.intercept

    def predict_chunk(self, counts):
        counts = np.asarray(counts)
        if counts.ndim != 2:
            raise ValueError("chunk must be time by channel")
        return np.asarray([self.step(row) for row in counts]).reshape(-1, len(self.model.intercept))
