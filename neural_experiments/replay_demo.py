"""Build a private, offline replay from a completed frozen assessment.

This viewer consumes saved predictions; it does not execute inference. Real-data
outputs must remain in an ignored results/ or work/ directory and must not be
published. --smoke exercises the standalone viewer with synthetic series only.

Example after the final assessment has been explicitly released and completed:
    PYTHONPATH=src .venv/bin/python neural_experiments/replay_demo.py \
      --index data/neural_h1/prepared_assessment/later-DIGEST/index.json \
      --freeze results/neural-h1/later-freeze.json --sha256 FULL_FREEZE_SHA256 \
      --primary-run results/neural-h1/later-evaluation \
      --output results/neural-h1/replay-demo.html

Synthetic, with no H1 access or scientific-library imports:
    python neural_experiments/replay_demo.py --smoke \
      --output work/neural/replay-smoke.html
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path

OUTPUT_NAMES = ["tx", "ty", "tz", "rx", "g1", "g2", "g3"]


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def resolve_input(path, root):
    path = Path(path)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def private_output(path, root):
    """Refuse publication directories, symlink escapes, and silent overwrites."""
    path = resolve_input(path, root)
    if not path.is_relative_to(root):
        raise ValueError("Replay output must stay inside this repository")
    relative = path.relative_to(root)
    if relative.parts[0] not in ("results", "work") or path.suffix != ".html":
        raise ValueError("Replay output must be an HTML file under private results/ or work/")
    receipt = path.with_suffix(".receipt.json")
    for candidate in (path, receipt):
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", "--", str(candidate.relative_to(root))],
            cwd=root,
            check=False,
        )
        if ignored.returncode != 0:
            raise ValueError("Replay and receipt must both be ignored by Git")
        if candidate.exists():
            raise FileExistsError(candidate)
    return path, receipt


def synthetic_payload():
    """Deterministic plumbing data with no physiological interpretation."""
    times = [index * 0.02 for index in range(1000)]
    targets = [[math.sin(t * (0.35 + j * 0.09) + j * 0.4) for j in range(7)] for t in times]
    predictions = [
        [
            0.85 * math.sin((t - 0.08) * (0.35 + j * 0.09) + j * 0.4) + 0.05 * math.cos(t * 2.3 + j)
            for j in range(7)
        ]
        for t in times
    ]
    return {
        "mode": "synthetic",
        "title": "Synthetic plumbing demonstration",
        "file_id": "synthetic-seven-output-stream",
        "day": "synthetic",
        "model_name": "deterministic synthetic curves",
        "output_names": OUTPUT_NAMES,
        "units": "synthetic arbitrary units",
        "timestamps_s": times,
        "targets": targets,
        "predictions": predictions,
        "eval_mask": [not (250 <= i < 300) for i in range(len(times))],
        "identities": {"data": "synthetic; no H1 files read", "freeze": "not applicable"},
    }


def runtime_context(path):
    """Read aggregate service measurements; never time browser playback."""
    aggregate = read_json(path)
    if not isinstance(aggregate, dict) or not isinstance(aggregate.get("runs"), list):
        raise ValueError("Runtime summary must contain the measured run list")
    names = {
        "numpy_incremental": "NumPy incremental",
        "torch_window": "PyTorch window",
        "onnx_incremental": "ONNX incremental",
    }
    expected = {
        (name, mode, repeat)
        for name in names
        for mode, repeats in (("fast", 3), ("paced", 2))
        for repeat in range(repeats)
    }
    seen = set()
    for row in aggregate["runs"]:
        if not isinstance(row, dict):
            raise ValueError("Runtime run must be an object")
        identity = (row.get("implementation"), row.get("mode"), row.get("repeat"))
        if identity not in expected or identity in seen or type(row.get("repeat")) is not int:
            raise ValueError("Runtime implementation/mode/repeat inventory is invalid")
        seen.add(identity)
        bins, misses = row.get("bins"), row.get("missed_deadlines")
        if type(bins) is not int or type(misses) is not int or not 0 <= misses <= bins:
            raise ValueError("Runtime bin/deadline counts must be valid integers")
        if bins != (112488 if row["mode"] == "fast" else 3001):
            raise ValueError("Runtime workload differs from the declared H1 study")
        service = row.get("step_service")
        if not isinstance(service, dict):
            raise ValueError("Complete-step service metrics are missing")
        values = [service.get(key) for key in ("p50_ms", "p95_ms", "p99_ms", "max_ms")]
        if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Runtime service metrics must be finite nonnegative numbers")
        if values != sorted(values):
            raise ValueError("Runtime service percentiles are not ordered")
    if seen != expected:
        raise ValueError("Runtime summary omits a required measurement block")
    rows = []
    for name, label in names.items():
        runs = [row for row in aggregate["runs"] if row["implementation"] == name]
        fast = [row["step_service"]["p99_ms"] for row in runs if row["mode"] == "fast"]
        paced = [row for row in runs if row["mode"] == "paced"]
        rows.append(
            {
                "implementation": label,
                "fast_median_p99_ms": statistics.median(fast),
                "paced_min_p99_ms": min(row["step_service"]["p99_ms"] for row in paced),
                "paced_max_p99_ms": max(row["step_service"]["p99_ms"] for row in paced),
                "paced_missed_deadlines": sum(row["missed_deadlines"] for row in paced),
                "paced_bins": sum(row["bins"] for row in paced),
            }
        )
    return {"summary_sha256": digest(path), "rows": rows}


def assessment_payload(args, root):
    """Validate release, source, inventory, and completion before opening arrays."""
    import numpy as np

    from neural_decoder.assessment_data import load_assessment_index, validate_freeze

    index_path = resolve_input(args.index, root)
    freeze_path = resolve_input(args.freeze, root)
    cache = resolve_input(args.cache, root)
    run = resolve_input(args.primary_run, root)
    freeze = validate_freeze(freeze_path, args.sha256, root=root, cache=cache)
    index = load_assessment_index(index_path, freeze_path, args.sha256, root=root, cache=cache)
    manifest_path, summary_path = run / "manifest.json", run / "summary.json"
    manifest = read_json(manifest_path)
    index_sha = digest(index_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("freeze_sha256") != args.sha256
        or manifest.get("index_sha256") != index_sha
        or manifest.get("summary_sha256") != digest(summary_path)
    ):
        raise ValueError("Primary run is incomplete or differs from the frozen assessment")
    summary = read_json(summary_path)
    if (
        summary.get("freeze_sha256") != args.sha256
        or summary.get("index_sha256") != index_sha
        or summary.get("stage") != freeze["stage"]
        or summary.get("regime") != "frozen_zero_new_day_calibration"
        or summary.get("models") != manifest.get("models")
        or [entry["model"] for entry in manifest["models"]] != freeze["models"]
    ):
        raise ValueError("Primary summary and exact frozen model inventory disagree")
    streams = {item["file_id"] for item in index["files"]}
    if any(set(entry["prediction_sha256"]) != streams for entry in manifest["models"]):
        raise ValueError("Primary prediction inventory differs from assessment inventory")
    models = [entry for entry in manifest["models"] if entry["model"]["name"] == args.model]
    if len(models) != 1:
        raise ValueError(f"Frozen model not found: {args.model}")
    selected = models[0]
    records = sorted(index["files"], key=lambda item: (int(item["day"][1:]), item["file_id"]))
    if args.file:
        records = [record for record in records if record["file_id"] == args.file]
    if not records:
        raise ValueError(f"Authorized stream not found: {args.file}")
    record = records[0]
    prediction_path = run / args.model / f"{record['file_id']}.npy"
    if not prediction_path.resolve().is_relative_to(run):
        raise ValueError("Prediction path escapes the completed primary run")
    prediction_sha = selected["prediction_sha256"][record["file_id"]]
    if digest(prediction_path) != prediction_sha:
        raise ValueError("Saved prediction identity changed")
    # The gate has already checked every feature/target artifact hash and role.
    # Only this viewer joins saved predictions to scorer-side comparison targets.
    with np.load(index_path.parent / record["features"], allow_pickle=False) as features:
        times = features["timestamps"].copy()
    with np.load(index_path.parent / record["targets"], allow_pickle=False) as target_file:
        target = target_file["targets"].copy()
        mask = target_file["eval_mask"].copy()
    prediction = np.load(prediction_path, allow_pickle=False)
    length = record["n_bins"]
    if (
        times.shape != (length,)
        or mask.shape != (length,)
        or mask.dtype != np.bool_
        or target.shape != (length, 7)
        or prediction.shape != target.shape
        or length < 2
        or not np.isfinite(times).all()
        or not np.all(np.diff(times) > 0)
        or not np.isfinite(target).all()
        or not np.isfinite(prediction).all()
    ):
        raise ValueError("Replay arrays have invalid shape, ordering, or finite-value contract")
    identities = {
        "stage": freeze["stage"],
        "model": args.model,
        "model_sha256": selected["model"]["sha256"],
        "model_training_days": selected["model"]["training_days"],
        "regime": selected["model"]["regime"],
        "file_id": record["file_id"],
        "day": record["day"],
        "provider_split": record["provider_split"],
        "provider_asset_id": record["asset_id"],
        "raw_sha256": record["raw_sha256"],
        "feature_sha256": record["features_sha256"],
        "target_sha256": record["targets_sha256"],
        "prediction_sha256": prediction_sha,
        "index_sha256": index_sha,
        "freeze_sha256": args.sha256,
        "frozen_source_revision": freeze["source"]["git_revision"],
        "upstream_revision": freeze["data"]["upstream_pin"],
        "primary_manifest_sha256": digest(manifest_path),
        "primary_summary_sha256": digest(summary_path),
        "first_recording_timestamp_s": float(times[0]),
        "bins": length,
        "scored_bins": int(mask.sum()),
    }
    return {
        "mode": "private_h1_recording",
        "title": "H1 saved prediction replay",
        "file_id": record["file_id"],
        "day": record["day"],
        "model_name": args.model,
        "output_names": OUTPUT_NAMES,
        "units": "provider arbitrary units",
        "timestamps_s": (times - times[0]).tolist(),
        "targets": target.tolist(),
        "predictions": prediction.tolist(),
        "eval_mask": mask.tolist(),
        "identities": identities,
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy"
  content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'">
<title>Recorded neural decoding replay</title>
<style>
:root {font: 16px/1.5 system-ui, sans-serif; color: #172e3b; background: #f3f5f6}
* {box-sizing: border-box} body {max-width: 1100px; margin: auto; padding: 28px 20px 48px}
h1 {font-size: 1.8rem; line-height: 1.2; margin: 0 0 12px} p {margin: 8px 0}
.notice {border-left: 4px solid #b46b12; padding: 10px 14px; background: #fff7e8}
#identity-line {font-weight: 600} .controls {background: white; border: 1px solid #ced9df;
  padding: 14px; border-radius: 7px; margin: 20px 0 14px; position: sticky; top: 0; z-index: 2}
.control-row {display: flex; flex-wrap: wrap; align-items: center; gap: 12px}
button, select {font: inherit; padding: 7px 12px; color: inherit; background: white;
  border: 1px solid #9aacb6; border-radius: 5px} button {cursor: pointer; min-width: 86px}
button:focus-visible, input:focus-visible, select:focus-visible {outline: 3px solid #15858c}
#play {background: #116a73; color: white; border-color: #116a73}
#scrub {width: 100%; margin: 15px 0 4px; accent-color: #116a73}
#status, .muted {color: #526773; font-size: .9rem}
.legend {display: flex; gap: 20px; flex-wrap: wrap}
.legend span::before {content: ''; display: inline-block; width: 26px; height: 3px;
  margin: 0 7px 4px 0; background: #6b7680}.legend .prediction::before {background: #087e8b}
.chart {background: white; border: 1px solid #d9e1e5; border-radius: 6px; padding: 10px 14px;
  margin-top: 10px}
.chart-header {display: flex; align-items: baseline; justify-content: space-between;
  flex-wrap: wrap; gap: 8px} .chart h2 {font-size: 1rem; margin: 0}
.readout {font-size: .85rem; font-variant-numeric: tabular-nums; color: #526773}
table {border-collapse: collapse; width: 100%; font-size: .85rem}
th, td {padding: 7px 9px; text-align: left; border-bottom: 1px solid #e3e9ec}
.table-wrap {overflow-x: auto}
canvas {display: block; width: 100%; height: 120px} details {margin-top: 24px}
pre {overflow-wrap: anywhere; white-space: pre-wrap; padding: 14px; background: white;
  border: 1px solid #ced9df; font-size: .78rem}
@media(max-width:600px) {body {padding: 16px 10px} h1 {font-size: 1.4rem} .chart {padding: 8px}}
</style>
</head>
<body>
<h1>Recorded replay, not live control</h1>
<p id="identity-line"></p>
<p class="notice" id="notice"></p>
<p>These are saved decoder predictions. Playback does not run inference or control a device.
Target traces are scorer-side comparisons; they were not input to the frozen primary decoder.</p>
<section id="runtime-context" class="chart" hidden>
<h2>Measured runtime context</h2>
<p class="muted">These separate complete-step service measurements use the study's pinned
original development weights, not the final refit or synthetic curves displayed here.
Browser playback performance is unmeasured. These are counts-ready CPU measurements,
not acquisition or live-control latency.</p>
<div class="table-wrap"><table>
<thead><tr><th>Implementation</th><th>Fast median p99 (ms)</th>
<th>Paced p99 range (ms)</th><th>Paced deadline misses / bins</th></tr></thead>
<tbody id="runtime-rows"></tbody>
</table></div>
<p class="muted">Three fast repeats and two 60-second paced blocks per implementation.
The source aggregate SHA-256 is included with the artifact identities below.</p>
</section>
<div class="controls">
  <div class="control-row">
    <button id="play" type="button">Play</button><button id="reset" type="button">Reset</button>
    <label for="speed">Playback speed</label><select id="speed">
      <option value="0.5">0.5×</option><option value="1" selected>1×</option>
      <option value="2">2×</option><option value="4">4×</option></select>
    <output id="position" aria-live="off"></output>
  </div>
  <label for="scrub" class="muted">Recording position</label>
  <input id="scrub" type="range" min="0" value="0" step="1">
  <div id="status"></div>
</div>
<div class="legend"><span>Target (scorer comparison)</span>
<span class="prediction">Saved prediction</span></div>
<p class="muted">All seven provider output components are shown in arbitrary units.
Axes scale separately.
Unscored bins are shaded pale orange. The cursor shows recording time,
not measured inference latency.</p>
<div id="charts"></div>
<details><summary>Model, recording, freeze, and artifact identities</summary>
<pre id="identities"></pre></details>
<script id="replay-data" type="application/json">__PAYLOAD__</script>
<script>
'use strict';
const data = JSON.parse(document.getElementById('replay-data').textContent);
const times = data.timestamps_s, n = times.length, duration = times[n-1];
const $ = id => document.getElementById(id);
$('identity-line').textContent = `${data.title} · ${data.file_id} · ${data.model_name}`;
$('notice').textContent = data.mode === 'synthetic'
  ? 'Synthetic plumbing demonstration. These curves contain no H1 recording data '
    + 'and establish no decoder performance.'
  : 'Private recording-derived artifact. Keep this HTML local; it embeds sample targets '
    + 'and predictions and must not be redistributed.';
$('identities').textContent = JSON.stringify(data.identities, null, 2);
if (data.runtime_context) {
  $('runtime-context').hidden = false;
  for (const row of data.runtime_context.rows) {
    const tr = document.createElement('tr');
    const values = [row.implementation, row.fast_median_p99_ms.toFixed(6),
      `${row.paced_min_p99_ms.toFixed(6)}–${row.paced_max_p99_ms.toFixed(6)}`,
      `${row.paced_missed_deadlines} / ${row.paced_bins.toLocaleString('en-US')}`];
    for (const value of values) {
      const td = document.createElement('td'); td.textContent = value; tr.append(td);
    }
    $('runtime-rows').append(tr);
  }
}
$('scrub').max = n - 1;
const plots = [], W = 1000, H = 120, L = 63, R = 15, TOP = 8, BOTTOM = 22;
const plotWidth = W-L-R, plotHeight = H-TOP-BOTTOM;
const xp = index => L + times[index] / duration * plotWidth;
const valueText = value => {
  const magnitude = Math.abs(value);
  return magnitude > 0 && (magnitude < 0.01 || magnitude >= 1000)
    ? value.toExponential(2) : value.toFixed(3);
};
for (let dim = 0; dim < 7; dim++) {
  const section = document.createElement('section'); section.className = 'chart';
  const header = document.createElement('div'); header.className = 'chart-header';
  const title = document.createElement('h2'); title.textContent = data.output_names[dim];
  const readout = document.createElement('span'); readout.className = 'readout';
  header.append(title, readout);
  const canvas = document.createElement('canvas'); canvas.width = W; canvas.height = H;
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label',
    `${data.output_names[dim]} target and saved prediction versus recording time`);
  section.append(header, canvas); $('charts').append(section);
  const background = document.createElement('canvas'); background.width = W; background.height = H;
  const lines = document.createElement('canvas'); lines.width = W; lines.height = H;
  const bg = background.getContext('2d'), trace = lines.getContext('2d');
  let low = Infinity, high = -Infinity;
  for (let i = 0; i < n; i++) {
    low = Math.min(low, data.targets[i][dim], data.predictions[i][dim]);
    high = Math.max(high, data.targets[i][dim], data.predictions[i][dim]);
  }
  const pad = Math.max((high-low)*0.08, 1e-9); low -= pad; high += pad;
  const yp = value => TOP + (high-value)/(high-low)*plotHeight;
  bg.fillStyle = '#fff'; bg.fillRect(0, 0, W, H);
  bg.fillStyle = '#fff2de';
  for (let i = 0; i < n; i++) if (!data.eval_mask[i]) {
    bg.fillRect(xp(i), TOP, Math.max(1, xp(Math.min(i+1, n-1))-xp(i)), plotHeight);
  }
  bg.font = '11px system-ui'; bg.textAlign = 'right'; bg.textBaseline = 'middle';
  for (const fraction of [0, 0.5, 1]) {
    const y = TOP + fraction*plotHeight;
    bg.strokeStyle = '#e3e9ec'; bg.beginPath(); bg.moveTo(L,y); bg.lineTo(W-R,y); bg.stroke();
    bg.fillStyle = '#526773'; bg.fillText(valueText(high-fraction*(high-low)), L-8, y);
  }
  bg.textBaseline = 'bottom'; bg.textAlign = 'center';
  for (const fraction of [0, 0.5, 1]) {
    bg.fillText(`${(duration*fraction).toFixed(1)} s`, L+fraction*plotWidth, H-1);
  }
  for (const [series, color] of [[data.targets, '#6b7680'], [data.predictions, '#087e8b']]) {
    trace.strokeStyle = color; trace.lineWidth = 1.5; trace.beginPath();
    for (let i = 0; i < n; i++) {
      if (i === 0) trace.moveTo(xp(i), yp(series[i][dim]));
      else trace.lineTo(xp(i), yp(series[i][dim]));
    }
    trace.stroke();
  }
  plots.push({canvas, background, lines, readout, dim});
}
let index = 0, playing = false, anchorWall = 0, anchorTime = 0, frame = null;
function render() {
  const cursor = xp(index);
  $('scrub').value = index;
  $('position').textContent = `${times[index].toFixed(2)} / ${duration.toFixed(2)} s`;
  $('status').textContent = `Bin ${index+1} of ${n} · ${data.eval_mask[index]
    ? 'Included in official scoring mask' : 'Excluded from official scoring mask'} · ${data.units}`;
  for (const plot of plots) {
    const ctx = plot.canvas.getContext('2d');
    ctx.clearRect(0,0,W,H); ctx.drawImage(plot.background,0,0);
    ctx.save(); ctx.beginPath(); ctx.rect(L,TOP,Math.max(1,cursor-L),plotHeight); ctx.clip();
    ctx.drawImage(plot.lines,0,0); ctx.restore();
    ctx.strokeStyle = '#b46b12'; ctx.lineWidth = 1.5; ctx.beginPath();
    ctx.moveTo(cursor,TOP); ctx.lineTo(cursor,H-BOTTOM); ctx.stroke();
    plot.readout.textContent = `Target ${valueText(data.targets[index][plot.dim])}`
      + ` · Prediction ${valueText(data.predictions[index][plot.dim])}`;
  }
}
function anchor() {anchorWall = performance.now(); anchorTime = times[index];}
function pause() {
  playing = false; cancelAnimationFrame(frame); $('play').textContent = 'Play';
}
function tick(now) {
  if (!playing) return;
  const desired = anchorTime + (now-anchorWall)/1000*Number($('speed').value);
  while (index+1 < n && times[index+1] <= desired) index++;
  render();
  if (index === n-1) pause(); else frame = requestAnimationFrame(tick);
}
$('play').addEventListener('click', () => {
  if (playing) {pause(); return;}
  if (index === n-1) index = 0;
  playing = true; $('play').textContent = 'Pause'; anchor(); frame = requestAnimationFrame(tick);
});
$('reset').addEventListener('click', () => {pause(); index = 0; render();});
$('scrub').addEventListener('input', event => {
  index = Number(event.target.value); anchor(); render();
});
$('speed').addEventListener('change', anchor);
render();
</script>
</body></html>
"""


def write_demo(payload, output, receipt_path):
    payload["identities"]["generator_sha256"] = digest(__file__)
    payload["identities"]["generated_at"] = datetime.now(UTC).isoformat()
    serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    html = HTML.replace("__PAYLOAD__", serialized)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(html)
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "designation": "recorded replay, not live control",
        "contains_private_sample_data": payload["mode"] != "synthetic",
        "html_sha256": digest(output),
        "html_file": output.name,
        "identities": payload["identities"],
    }
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Use synthetic series; never access H1"
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cache", type=Path, default=Path("data/neural_h1"))
    parser.add_argument("--index", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--sha256", help="Externally authorized immutable freeze SHA256")
    parser.add_argument("--primary-run", type=Path)
    parser.add_argument("--model", default="tcn-seed-0")
    parser.add_argument("--file", help="Exact authorized file ID; defaults to earliest day/file")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--runtime-summary", type=Path, help="Optional aggregate same-weight runtime context"
    )
    args = parser.parse_args()
    if args.smoke and any((args.index, args.freeze, args.sha256, args.primary_run, args.file)):
        parser.error("--smoke cannot be mixed with recording input arguments")
    if not args.smoke and not all((args.index, args.freeze, args.sha256, args.primary_run)):
        parser.error("Real replay requires --index, --freeze, --sha256, and --primary-run")
    root = args.root.resolve()
    destination = args.output or Path(
        "work/neural/replay-smoke.html" if args.smoke else "results/neural-h1/replay-demo.html"
    )
    output, receipt_path = private_output(destination, root)
    payload = synthetic_payload() if args.smoke else assessment_payload(args, root)
    if args.runtime_summary:
        context = runtime_context(resolve_input(args.runtime_summary, root))
        payload["runtime_context"] = context
        payload["identities"]["runtime_summary_sha256"] = context["summary_sha256"]
    receipt = write_demo(payload, output, receipt_path)
    print(
        json.dumps(
            {
                "html": str(output),
                "receipt": str(receipt_path),
                "html_sha256": receipt["html_sha256"],
                "mode": payload["mode"],
            }
        )
    )


if __name__ == "__main__":
    main()
