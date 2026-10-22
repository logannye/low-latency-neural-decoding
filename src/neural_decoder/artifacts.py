"""Compact experiment identity and failure-safe result metadata."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_identity(root, destination):
    """Retain the exact exploratory source bytes; git HEAD alone is insufficient."""
    root, destination = Path(root), Path(destination)
    snapshot = destination / "source"
    files = [root / "pyproject.toml"]
    for directory in ("src/neural_decoder", "neural_experiments/configs"):
        files.extend(
            p for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts
        )
    files.extend(p for p in (root / "neural_experiments").glob("*.md"))
    files.extend(p for p in (root / "neural_experiments").glob("*.py"))
    identities = {}
    for path in sorted(files):
        relative = path.relative_to(root)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        identities[str(relative)] = sha256(target)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
    (destination / "source.diff").write_bytes(diff)
    return {
        "head": revision,
        "files": identities,
        "diff_sha256": sha256(destination / "source.diff"),
    }


def environment():
    dependencies = {}
    for name in (
        "numpy",
        "scipy",
        "scikit-learn",
        "torch",
        "h5py",
        "pynwb",
        "threadpoolctl",
        "onnx",
        "onnxruntime",
        "psutil",
    ):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return {
        "started_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "dependencies": dependencies,
        "thread_environment": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }
