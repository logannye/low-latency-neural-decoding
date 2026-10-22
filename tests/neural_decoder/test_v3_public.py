"""Only exact audited artifacts and aggregate-safe strings may be published."""

import pytest

from neural_experiments.v3 import public


def test_publication_requires_exact_passed_binding():
    public.check_receipt({"status": "passed", "manifest_sha256": "a"}, manifest_sha256="a")
    for receipt in (
        {"status": "failed", "manifest_sha256": "a"},
        {"status": "passed", "manifest_sha256": "b"},
        {},
    ):
        with pytest.raises(ValueError, match="bound"):
            public.check_receipt(receipt, manifest_sha256="a")


@pytest.mark.parametrize(
    "private",
    [
        "/Users/private/file",
        "/home/private/file",
        "https://example.test/?X-Amz-Signature=placeholder",
    ],
)
def test_private_paths_and_signed_credentials_are_rejected(tmp_path, private):
    with pytest.raises(ValueError, match="private"):
        public.safe_write(tmp_path / "public.json", {"nested": [private]})
    assert not (tmp_path / "public.json").exists()


def test_nonfinite_metrics_are_not_published(tmp_path):
    with pytest.raises(ValueError):
        public.safe_write(tmp_path / "public.json", {"r2": float("nan")})
