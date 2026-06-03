"""End-to-end tests for optimize_skinning_weights().

Synthetic configurations whose answer we can hand-derive:
  - Rigid-body translation (visual+proxy = same sphere) → solver must recover
    weights that perfectly track the proxy motion.
  - Determinism: identical seeds → identical s.
  - Sparse W shape and row-sum invariants.
"""
from __future__ import annotations

import numpy as np
import torch

from pag.anim import AnimationData
from pag.skinning import optimize_skinning_weights


def _icosphere(radius: float = 1.0, subdivs: int = 2):
    t = (1.0 + 5.0 ** 0.5) / 2.0
    V = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    F = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=np.int64)
    cache: dict[tuple[int, int], int] = {}
    Vlist = [v.copy() for v in V]

    def mid(a, b):
        key = (min(a, b), max(a, b))
        if key in cache:
            return cache[key]
        m = 0.5 * (Vlist[a] + Vlist[b])
        idx = len(Vlist); Vlist.append(m); cache[key] = idx
        return idx

    for _ in range(subdivs):
        nF = []
        for tri in F:
            a, b, c = tri
            ab = mid(a, b); bc = mid(b, c); ca = mid(c, a)
            nF.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        F = np.array(nF, dtype=np.int64)
    Vf = np.array(Vlist)
    Vf = radius * Vf / np.linalg.norm(Vf, axis=1, keepdims=True)
    return Vf, F


def _make_anim(V_proxy: np.ndarray, frames: np.ndarray, seed: int = 0) -> AnimationData:
    """Wrap a (T, N_p, 3) trajectory in a stub AnimationData."""
    T = frames.shape[0]
    rng = np.random.default_rng(seed)
    return AnimationData(
        V_proxy_rest=np.ascontiguousarray(V_proxy, dtype=np.float64),
        F_proxy=np.zeros((0, 3), dtype=np.int64),
        pinned=np.zeros((0,), dtype=np.int64),
        X=np.ascontiguousarray(frames, dtype=np.float32),
        theta=np.zeros((T, 2), dtype=np.float64),
        wind_global=rng.standard_normal((T, 3)),
        azimuth_deg=np.zeros(T),
        magnitude=np.zeros(T),
    )


# ----------------- rigid-body translation -----------------

def test_translation_only_proxy_drives_visual_perfectly():
    """Visual = proxy; pure translations of the proxy → simplified LBS with
    *any* row-stochastic w trivially recovers the translation. Loss should be
    ~zero from epoch 0, and the reconstructed visual must equal the proxy
    frames within float32 noise."""
    V, F = _icosphere(radius=1.0, subdivs=1)
    T = 6
    rng = np.random.default_rng(0)
    deltas = rng.standard_normal((T, 3)) * 0.3
    frames = V[None] + deltas[:, None, :]                 # (T, N, 3)
    train = _make_anim(V, frames)

    res = optimize_skinning_weights(
        V, F, train, k_B=4, k_K=4, epochs=30, lr=1e-2, seed=0,
    )

    # Reconstruction check (use the optimized s on the train frames).
    s = torch.tensor(res.s)
    B = torch.tensor(res.B, dtype=torch.long)
    V_v0 = torch.tensor(V, dtype=torch.float32)
    V_p0 = torch.tensor(V, dtype=torch.float32)
    X = torch.tensor(frames, dtype=torch.float32)
    from pag.skinning_lbs import simplified_lbs
    V_recon = simplified_lbs(s, B, V_v0, V_p0, X).numpy()
    err = np.abs(V_recon - frames).max()
    assert err < 1e-4, f"recon error {err} too large for rigid translation"


def test_loss_decreases_during_optimization():
    """Synthetic deformation scenario: a low-frequency twist of the proxy
    expressed via sin-wave per-frame perturbation. L_total should decrease
    by ≥ 30% over 50 epochs — sanity that AdamW is actually training."""
    V, F = _icosphere(radius=1.0, subdivs=2)
    T = 12
    t = np.linspace(0, 1, T)[:, None, None]
    deform = 0.2 * np.sin(2 * np.pi * t * V[None, :, 0:1])
    frames = V[None] + np.concatenate([deform, np.zeros_like(deform), np.zeros_like(deform)], axis=-1)
    train = _make_anim(V, frames)

    res = optimize_skinning_weights(
        V, F, train, k_B=6, k_K=6, epochs=50, lr=5e-2, seed=0,
    )
    initial = res.losses["total"][0]
    final = res.losses["total"][-1]
    assert final < 0.7 * initial, (
        f"L_total only dropped {initial:.3e} → {final:.3e} — optimizer may be stuck"
    )


# ----------------- determinism + W shape -----------------

def test_determinism_same_seed():
    V, F = _icosphere(radius=1.0, subdivs=1)
    T = 4
    rng = np.random.default_rng(0)
    frames = V[None] + 0.1 * rng.standard_normal((T, 1, 3))
    train = _make_anim(V, frames)
    a = optimize_skinning_weights(V, F, train, k_B=4, k_K=4, epochs=10, seed=0)
    b = optimize_skinning_weights(V, F, train, k_B=4, k_K=4, epochs=10, seed=0)
    np.testing.assert_allclose(a.s, b.s, atol=1e-6, rtol=1e-6)


def test_sparse_W_row_sums_to_one():
    V, F = _icosphere(radius=1.0, subdivs=1)
    T = 3
    rng = np.random.default_rng(1)
    frames = V[None] + 0.05 * rng.standard_normal((T, 1, 3))
    train = _make_anim(V, frames)
    res = optimize_skinning_weights(V, F, train, k_B=5, k_K=5, epochs=5, seed=0)

    N_v, N_p = res.W.shape
    assert N_v == V.shape[0]
    assert N_p == V.shape[0]
    # Row sums should equal 1 within fp32 tolerance.
    row_sums = np.asarray(res.W.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6, rtol=1e-6)
    # Each row has exactly k_B non-zeros (the bone indices may collide if
    # cKDTree returns duplicates, but for distinct vertices they don't).
    assert res.W.nnz == N_v * 5


def test_held_out_test_eval_recorded():
    """test=AnimationData → eval dict is populated."""
    V, F = _icosphere(radius=1.0, subdivs=1)
    rng = np.random.default_rng(2)
    train_frames = V[None] + 0.1 * rng.standard_normal((4, 1, 3))
    test_frames = V[None] + 0.1 * rng.standard_normal((3, 1, 3))
    train = _make_anim(V, train_frames)
    test = _make_anim(V, test_frames)
    res = optimize_skinning_weights(
        V, F, train, test=test, k_B=4, k_K=4, epochs=5, seed=0,
    )
    for k in ("test_L_r", "test_L_c", "test_L_a", "test_chamfer_to_proxy",
              "train_chamfer_to_proxy"):
        assert k in res.eval and np.isfinite(res.eval[k])
    # On the matched-mesh sphere case, the chamfer-to-proxy should be tiny.
    assert res.eval["test_chamfer_to_proxy"] < 0.5


def test_timings_recorded():
    V, F = _icosphere(radius=1.0, subdivs=1)
    rng = np.random.default_rng(3)
    frames = V[None] + 0.05 * rng.standard_normal((3, 1, 3))
    train = _make_anim(V, frames)
    res = optimize_skinning_weights(V, F, train, k_B=4, k_K=4, epochs=3, seed=0)
    for k in ("setup", "optimize", "total"):
        assert k in res.timings and res.timings[k] >= 0.0
