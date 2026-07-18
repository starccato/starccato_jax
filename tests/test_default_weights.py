from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from starccato_jax.data import default_weights


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_verified_weight_is_reused_without_download(tmp_path, monkeypatch):
    payload = b"immutable model"
    model_dir = tmp_path / "ccsne"
    model_dir.mkdir()
    (model_dir / "model.h5").write_bytes(payload)
    monkeypatch.setattr(default_weights, "DATA_DIR", f"{tmp_path}/")
    monkeypatch.setitem(
        default_weights.WEIGHT_SPECS,
        "ccsne",
        ("https://example.invalid/model.h5", sha256(payload)),
    )

    def fail_download(*args, **kwargs):
        raise AssertionError("valid immutable cache should not be refreshed")

    monkeypatch.setattr(
        default_weights, "download_with_progress", fail_download
    )
    assert default_weights.get_default_weights_dir("ccsne") == str(model_dir)


def test_bad_cached_weight_is_atomically_replaced(tmp_path, monkeypatch):
    payload = b"verified replacement"
    model_dir = tmp_path / "blip"
    model_dir.mkdir()
    destination = model_dir / "model.h5"
    destination.write_bytes(b"corrupt")
    monkeypatch.setattr(default_weights, "DATA_DIR", f"{tmp_path}/")
    monkeypatch.setitem(
        default_weights.WEIGHT_SPECS,
        "blip",
        ("https://example.invalid/model.h5", sha256(payload)),
    )

    def fake_download(url: str, output_path: str):
        Path(output_path).write_bytes(payload)

    monkeypatch.setattr(
        default_weights, "download_with_progress", fake_download
    )
    default_weights.get_default_weights_dir("blip")
    assert destination.read_bytes() == payload
    assert not list(model_dir.glob("*.download"))


def test_checksum_failure_preserves_existing_cache(tmp_path, monkeypatch):
    model_dir = tmp_path / "ccsne"
    model_dir.mkdir()
    destination = model_dir / "model.h5"
    destination.write_bytes(b"old")
    monkeypatch.setattr(default_weights, "DATA_DIR", f"{tmp_path}/")
    monkeypatch.setitem(
        default_weights.WEIGHT_SPECS,
        "ccsne",
        ("https://example.invalid/model.h5", sha256(b"expected")),
    )

    def fake_download(url: str, output_path: str):
        Path(output_path).write_bytes(b"wrong")

    monkeypatch.setattr(
        default_weights, "download_with_progress", fake_download
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        default_weights.get_default_weights_dir("ccsne")
    assert destination.read_bytes() == b"old"


def test_unknown_default_weight_dataset_is_rejected():
    with pytest.raises(ValueError, match="Unknown default-weight dataset"):
        default_weights.get_default_weights_dir("typo")
