"""Generic precompute runner for scenario-based skinning evaluation.

A "scenario" is a `pbd.System` plus two callbacks:

  per_frame(t, sys)     # mutate kinematic colliders / apply forces
  obstacles_at(t)       # snapshot the renderer-facing obstacle geometry

`run_proxy_sim` settles the cloth from its rest pose for `n_settle` un-logged
frames, then steps `n_frames` more, snapshotting `sys.X` and the obstacle log
on each. The output `(X_p, obstacles_per_frame)` plus `lbs_drive(...)` is all
the side-by-side viewer needs.

Scenarios live in `scripts/eval_scenarios/`; this module is the shared library
they import.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import scipy.sparse


# ---------------------------------------------------------------- Obstacle

@dataclass
class Obstacle:
    """Renderer-facing snapshot of one obstacle on one frame.

    All fields are world-space; the viewer mirrors them into the visual pane
    by adding its own +x offset.
    """
    name: str
    kind: Literal["sphere", "plane", "mesh"]
    V: Optional[np.ndarray] = None       # (M, 3) for kind="mesh"
    F: Optional[np.ndarray] = None       # (K, 3) for kind="mesh"
    center: Optional[np.ndarray] = None  # (3,) for "sphere" / "plane" anchor
    radius: Optional[float] = None       # for "sphere"
    normal: Optional[np.ndarray] = None  # (3,) for "plane"
    color: tuple[float, float, float] = (0.85, 0.35, 0.35)


# ------------------------------------------------------------------ I/O

def _load_weights_npz(
    path: Path,
) -> tuple[scipy.sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Reconstruct (W, B, s) from the .npz that get_skin_weights.py --out writes.

    Mirrors scripts/skinning_playback.py:_load_weights — duplicated rather than
    refactored so the playback script stays untouched.
    """
    z = np.load(path)
    shape = tuple(int(x) for x in z["W_shape"])
    W = scipy.sparse.csr_matrix(
        (z["W_data"], z["W_indices"], z["W_indptr"]), shape=shape,
    )
    return W, z["B"], z["s"]


def load_eval_inputs(
    visual_path: str | Path,
    anim_dir: str | Path,
    weights_path: str | Path,
) -> tuple[
    np.ndarray, np.ndarray,    # V_visual, F_visual
    np.ndarray, np.ndarray,    # V_p0, F_p
    np.ndarray,                # pinned
    np.ndarray, np.ndarray,    # s, B
]:
    """Load the visual mesh, proxy rest mesh, and learned weights for eval.

    Reads `mesh.npz` from `anim_dir` directly — does not require `train.npz` /
    `test.npz` to exist, so eval can be run from any directory that contains
    just the rest-pose proxy.
    """
    from pag.io import load_obj

    V_visual, F_visual = load_obj(Path(visual_path))

    m = np.load(Path(anim_dir) / "mesh.npz")
    V_p0 = np.ascontiguousarray(m["V0"], dtype=np.float64)
    F_p = np.ascontiguousarray(m["F"], dtype=np.int64)
    pinned = np.ascontiguousarray(m["pinned"], dtype=np.int64)

    W, B, s = _load_weights_npz(Path(weights_path))

    if W.shape[0] != V_visual.shape[0]:
        raise ValueError(
            f"visual |V|={V_visual.shape[0]} does not match weights "
            f"N_v={W.shape[0]} — is --visual the mesh you trained on?"
        )
    if W.shape[1] != V_p0.shape[0]:
        raise ValueError(
            f"proxy N_p={V_p0.shape[0]} does not match weights "
            f"N_p={W.shape[1]} — is --anim-dir from the same Stage 1 pair?"
        )

    return V_visual, F_visual, V_p0, F_p, pinned, s, B


# -------------------------------------------------------------- sim loop

def run_proxy_sim(
    sys,                                          # pbd.System
    n_frames: int,
    per_frame_fn: Callable[[int, "object"], None],
    obstacles_at: Callable[[int], list[Obstacle]],
    *,
    dt: float = 1.0 / 60.0,
    iters: int = 15,
    k_damp: float = 0.05,
    friction: float = 0.0,
    restitution: float = 0.0,
    contact_skin: float = 0.0,
    n_settle: int = 30,
    solver: str = "jacobi",
) -> tuple[np.ndarray, list[list[Obstacle]]]:
    """Settle then run a logged trajectory.

    Parameters
    ----------
    sys
        A fully configured `pbd.System` — constraints added, colliders added,
        pinning applied.
    n_frames
        Number of frames to log AFTER settling.
    per_frame_fn
        Called as `per_frame_fn(t, sys)` BEFORE the t-th step. Mutate
        kinematic colliders here. Not called during settling.
    obstacles_at
        Called as `obstacles_at(t)` AFTER the t-th step. Returns the obstacle
        snapshot for the renderer.

    Returns
    -------
    X_p : (n_frames, N_p, 3) float32
    obstacles_per_frame : list of length n_frames, each a list[Obstacle]
    """
    step_kw = dict(
        dt=dt, iters=iters, k_damp=k_damp,
        friction=friction, restitution=restitution,
        contact_skin=contact_skin, solver=solver,
    )
    for _ in range(n_settle):
        sys.step(**step_kw)

    n_p = sys.X.shape[0]
    X_p = np.zeros((n_frames, n_p, 3), dtype=np.float32)
    obs_log: list[list[Obstacle]] = []
    for t in range(n_frames):
        per_frame_fn(t, sys)
        sys.step(**step_kw)
        X_p[t] = sys.X.astype(np.float32, copy=False)
        obs_log.append(obstacles_at(t))
    return X_p, obs_log


# ------------------------------------------------------------------ LBS

def lbs_drive(
    s: np.ndarray,        # (N_v, k_B)
    B: np.ndarray,        # (N_v, k_B) int64
    V_v0: np.ndarray,     # (N_v, 3)
    V_p0: np.ndarray,     # (N_p, 3)
    X_p: np.ndarray,      # (T, N_p, 3)
) -> np.ndarray:
    """Numpy-out wrapper around `pag.skinning_lbs.simplified_lbs`.

    Returns (T, N_v, 3) float32.
    """
    import torch
    from pag.skinning_lbs import simplified_lbs

    s_t = torch.as_tensor(s, dtype=torch.float32)
    B_t = torch.as_tensor(B, dtype=torch.long)
    V_v0_t = torch.as_tensor(V_v0, dtype=torch.float32)
    V_p0_t = torch.as_tensor(V_p0, dtype=torch.float32)
    X_t = torch.as_tensor(X_p, dtype=torch.float32)
    with torch.no_grad():
        V_recon = simplified_lbs(s_t, B_t, V_v0_t, V_p0_t, X_t)
    return V_recon.cpu().numpy()
