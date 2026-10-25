"""A terminal reporter failure remains failed even after writing partial output."""

import json

import pytest

from neural_experiments import h1_final_closure_report as reporter


def test_partial_report_failure_remains_failed_and_cannot_overwrite(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        (kwargs["output"] / "quality.json").write_text('{"partial": true}')
        raise RuntimeError("injected terminal formatting failure")

    monkeypatch.setattr(reporter, "_export_report", broken)
    output = tmp_path / "report"
    with pytest.raises(RuntimeError, match="terminal"):
        reporter.export_report(output=output)
    status = json.loads((output / "manifest.json").read_text())
    assert status["status"] == "failed"
    assert "quality.json" in status["artifacts"]
    with pytest.raises(FileExistsError):
        reporter.export_report(output=output)
