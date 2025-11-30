import json
import shutil
from pathlib import Path

import h5py
import jax.numpy as jnp
from flax.training.train_state import TrainState

from starccato_jax.vae.core.io import (
    MODEL_FNAME,
    load_model,
    save_model,
    get_model_version,
)
from starccato_jax.vae.core.data_containers import TrainValMetrics, Losses, Gradients
from starccato_jax.vae.config import Config
from starccato_jax import __version__ as CURRENT_VERSION


def _minimal_metrics():
    return TrainValMetrics(
        train_metrics=Losses(0.0, 0.0, 0.0, 0.0),
        val_metrics=Losses(0.0, 0.0, 0.0, 0.0),
        gradient_norms=Gradients(0),
    )


def test_save_includes_version(tmp_path):
    savedir = tmp_path / "artifact"
    savedir.mkdir()

    # Monkeypatch Config.__post_init__ to avoid heavy dataset loading.
    # We only need latent_dim and data_dim.
    def _fake_post_init(self):  # noqa: D401
        self.beta_schedule = []
        # Ensure data_dim is set (simulate dataset inference)
        self.data_dim = self.data_dim or 4
        self.data = type("Dummy", (), {"train": jnp.zeros((2, self.data_dim))})()
        self._batch_size_check = lambda: None

    original_post_init = Config.__post_init__
    Config.__post_init__ = _fake_post_init  # type: ignore
    try:
        config = Config(latent_dim=3, data_dim=4, epochs=1, batch_size=1)
    finally:
        Config.__post_init__ = original_post_init  # restore

    # Minimal params tree
    params = {"layer": {"w": jnp.zeros((3, 4))}}
    state = TrainState(
        step=0,
        apply_fn=lambda *a, **k: None,
        params=params,
        tx=None,
        opt_state=None,
    )
    save_model(state, config, _minimal_metrics(), str(savedir))

    version = get_model_version(str(savedir))
    assert version == CURRENT_VERSION, "Stored version attribute should match current library version"


def test_version_mismatch_error_message(tmp_path):
    # Copy existing test artifact model.h5 to temp, then corrupt config and version.
    source_dir = Path(__file__).parent / "test_output-main"
    target_dir = tmp_path / "copy"
    shutil.copytree(source_dir, target_dir)
    model_file = target_dir / MODEL_FNAME

    # Inject mismatching version and corrupt config to trigger error
    with h5py.File(model_file, "r+") as f:
        f.attrs["library_version"] = "0.0.0"  # simulate old version
        cfg = json.loads(f["config"][()].decode("utf-8"))
        cfg["unexpected_field"] = 123  # force TypeError in Config(**cfg)
        new_cfg = json.dumps(cfg)
        del f["config"]
        f.create_dataset("config", data=new_cfg)

    try:
        load_model(str(target_dir))
    except RuntimeError as e:
        msg = str(e)
        assert "weights saved with starccato_jax" in msg
        assert "0.0.0" in msg and CURRENT_VERSION in msg
    else:  # pragma: no cover
        raise AssertionError("Expected RuntimeError due to corrupted config JSON")
