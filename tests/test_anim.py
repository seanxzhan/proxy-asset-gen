"""Round-trip tests against the windblown_data_gen.py output schema.

Fabricates a tiny dataset directory in tmp_path and verifies AnimationData /
load_dataset return the right shapes / dtypes / sharing semantics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pag.anim import AnimationData, load_dataset


def _write_dataset(
    out_dir: Path,
    n_p: int = 6,
    n_f: int = 4,
    t_train: int = 5,
    t_test: int = 3,
    n_pin: int = 2,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    V0 = rng.standard_normal((n_p, 3)).astype(np.float64)
    F = np.array([[i, (i + 1) % n_p, (i + 2) % n_p] for i in range(n_f)], dtype=np.int64)
    pinned = np.arange(n_pin, dtype=np.int64)
    np.savez(out_dir / "mesh.npz", V0=V0, F=F, pinned=pinned)

    def _split(t: int):
        X = rng.standard_normal((t, n_p, 3)).astype(np.float32)
        azimuth_deg = rng.uniform(0.0, 360.0, size=t)
        magnitude = rng.uniform(0.1, 0.5, size=t)
        theta = np.stack([np.deg2rad(azimuth_deg), magnitude], axis=1)  # (t, 2)
        wind_global = rng.standard_normal((t, 3)).astype(np.float64)
        return dict(
            X=X, theta=theta, wind_global=wind_global,
            azimuth_deg=azimuth_deg, magnitude=magnitude,
        )

    np.savez(out_dir / "train.npz", **_split(t_train))
    np.savez(out_dir / "test.npz", **_split(t_test))

    with (out_dir / "metadata.json").open("w") as fh:
        json.dump({"obj": "fake.obj", "train_frames": t_train, "test_frames": t_test}, fh)


def test_load_split_shapes_and_dtypes(tmp_path: Path):
    _write_dataset(tmp_path, n_p=6, t_train=5)
    a = AnimationData.load_dir(tmp_path, "train")
    assert a.V_proxy_rest.shape == (6, 3) and a.V_proxy_rest.dtype == np.float64
    assert a.F_proxy.dtype == np.int64
    assert a.X.shape == (5, 6, 3) and a.X.dtype == np.float32
    assert a.theta.shape == (5, 2) and a.theta.dtype == np.float64
    assert a.wind_global.shape == (5, 3)
    assert a.n_frames == 5 and a.n_proxy == 6


def test_load_dataset_shares_mesh(tmp_path: Path):
    _write_dataset(tmp_path)
    train, test, meta = load_dataset(tmp_path)
    np.testing.assert_array_equal(train.V_proxy_rest, test.V_proxy_rest)
    np.testing.assert_array_equal(train.F_proxy, test.F_proxy)
    assert meta["obj"] == "fake.obj"
    assert train.n_frames == 5 and test.n_frames == 3


def test_theta_and_azimuth_consistent(tmp_path: Path):
    _write_dataset(tmp_path, t_train=4)
    a = AnimationData.load_dir(tmp_path, "train")
    np.testing.assert_allclose(a.theta[:, 0], np.deg2rad(a.azimuth_deg))
    np.testing.assert_allclose(a.theta[:, 1], a.magnitude)


def test_missing_split_raises(tmp_path: Path):
    (tmp_path / "mesh.npz").write_bytes(b"")  # placeholder
    with pytest.raises(FileNotFoundError):
        AnimationData.load_dir(tmp_path, "train")


def test_real_dataset_loads():
    """The committed dataset at pbd/data/9423122485_cleaned_proxy/ — sanity load.

    Skips if the path doesn't exist on this machine (CI may not have it)."""
    real_dir = Path("/Users/szhan/projects/pbd/data/9423122485_cleaned_proxy")
    if not real_dir.exists():
        pytest.skip(f"{real_dir} not present")
    train, test, meta = load_dataset(real_dir)
    assert train.V_proxy_rest.shape[1] == 3
    assert train.X.shape[1:] == train.V_proxy_rest.shape
    assert test.X.shape[1:] == train.V_proxy_rest.shape
    assert "metrics" in meta or "obj" in meta
