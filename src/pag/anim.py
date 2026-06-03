"""Loader for the windblown_data_gen.py PBD pre-simulation output.

Stage 2 (skinning weight optimization) consumes a directory written by
``pbd/examples/windblown_data_gen.py``. The directory holds:

  - mesh.npz  : V0 (N_p, 3) f64, F (M_p, 3) i64, pinned (n_pin,) i64
  - train.npz : X (T, N_p, 3) f32, theta (T, 2) f64, wind_global (T, 3) f64,
                azimuth_deg (T,) f64, magnitude (T,) f64
  - test.npz  : same keys as train, T_test = held-out OOD frames
  - metadata.json : the CLI args + OOD metrics — passed back unchanged

Field names mirror the .npz keys exactly, so the contract with the pbd repo
is just ``np.load`` + attribute access.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AnimationData:
    """One split (train OR test) from windblown_data_gen.py.

    Field names are 1:1 with the .npz keys it writes.
    """
    V_proxy_rest: np.ndarray   # (N_p, 3) f64 — mesh.npz V0
    F_proxy: np.ndarray        # (M_p, 3) i64 — mesh.npz F
    pinned: np.ndarray         # (n_pin,) i64 — mesh.npz pinned
    X: np.ndarray              # (T, N_p, 3) f32
    theta: np.ndarray          # (T, 2) f64 — (azimuth_rad, magnitude)
    wind_global: np.ndarray    # (T, 3) f64
    azimuth_deg: np.ndarray    # (T,) f64
    magnitude: np.ndarray      # (T,) f64

    @property
    def n_frames(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_proxy(self) -> int:
        return int(self.V_proxy_rest.shape[0])

    @classmethod
    def load_dir(cls, dir: str | Path, split: str = "train") -> "AnimationData":
        """Read mesh.npz + {split}.npz from a windblown_data_gen.py output directory."""
        d = Path(dir)
        mesh_path = d / "mesh.npz"
        split_path = d / f"{split}.npz"
        if not mesh_path.exists():
            raise FileNotFoundError(f"missing {mesh_path}")
        if not split_path.exists():
            raise FileNotFoundError(f"missing {split_path}")

        m = np.load(mesh_path)
        s = np.load(split_path)

        V0 = np.ascontiguousarray(m["V0"], dtype=np.float64)
        F = np.ascontiguousarray(m["F"], dtype=np.int64)
        pinned = np.ascontiguousarray(m["pinned"], dtype=np.int64)
        X = np.ascontiguousarray(s["X"], dtype=np.float32)

        if X.ndim != 3 or X.shape[1] != V0.shape[0] or X.shape[2] != 3:
            raise ValueError(
                f"{split_path}: X has shape {X.shape}; expected "
                f"(T, {V0.shape[0]}, 3) to match mesh V0"
            )

        return cls(
            V_proxy_rest=V0,
            F_proxy=F,
            pinned=pinned,
            X=X,
            theta=np.ascontiguousarray(s["theta"], dtype=np.float64),
            wind_global=np.ascontiguousarray(s["wind_global"], dtype=np.float64),
            azimuth_deg=np.ascontiguousarray(s["azimuth_deg"], dtype=np.float64),
            magnitude=np.ascontiguousarray(s["magnitude"], dtype=np.float64),
        )


def load_dataset(
    dir: str | Path,
) -> tuple[AnimationData, AnimationData, dict]:
    """Load both splits + metadata from a windblown_data_gen.py output directory.

    Returns
    -------
    train : AnimationData
    test  : AnimationData (shares V_proxy_rest, F_proxy, pinned with train)
    metadata : dict — contents of metadata.json (CLI args + OOD metrics)
    """
    d = Path(dir)
    train = AnimationData.load_dir(d, "train")
    test = AnimationData.load_dir(d, "test")
    meta_path = d / "metadata.json"
    metadata: dict = {}
    if meta_path.exists():
        with meta_path.open() as fh:
            metadata = json.load(fh)
    # Sanity: both splits must reference the same proxy mesh byte-for-byte.
    if not np.array_equal(train.V_proxy_rest, test.V_proxy_rest):
        raise ValueError("train and test V_proxy_rest disagree — corrupt dataset")
    if not np.array_equal(train.F_proxy, test.F_proxy):
        raise ValueError("train and test F_proxy disagree — corrupt dataset")
    return train, test, metadata
