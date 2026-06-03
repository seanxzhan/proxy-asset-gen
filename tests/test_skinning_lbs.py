"""Unit tests for the simplified-LBS forward pass and neighborhood builders."""
from __future__ import annotations

import numpy as np
import torch

from pag.skinning_lbs import (
    build_bone_indices,
    build_knn_indices,
    build_topological_1ring,
    reparam_weights,
    simplified_lbs,
)


# ------------------------- forward pass -------------------------

def test_identity_lbs_recovers_proxy_motion():
    """Visual = proxy, B(i) = i, w_i,0 = 1 → V_recon == X_pt for every frame."""
    rng = np.random.default_rng(0)
    n = 7; T = 4
    V0 = rng.standard_normal((n, 3))
    X = rng.standard_normal((T, n, 3))

    s = torch.zeros(n, 1)
    s[:, 0] = 1.0  # all weight on bone 0
    B = torch.arange(n, dtype=torch.long)[:, None]   # (n, 1)
    V_v0 = torch.tensor(V0, dtype=torch.float64)
    V_p0 = torch.tensor(V0, dtype=torch.float64)
    X_pt = torch.tensor(X, dtype=torch.float64)

    out = simplified_lbs(s.double(), B, V_v0, V_p0, X_pt)
    assert out.shape == (T, n, 3)
    torch.testing.assert_close(out, X_pt, atol=1e-12, rtol=1e-12)


def test_translation_equivariance():
    """Shift every proxy frame by Δ → every visual recon shifts by Δ. Weights
    sum to 1, so a uniform translation flows through identically."""
    rng = np.random.default_rng(1)
    N_v, N_p, k_B, T = 5, 4, 3, 6
    V_visual = rng.standard_normal((N_v, 3))
    V_proxy = rng.standard_normal((N_p, 3))
    X = rng.standard_normal((T, N_p, 3))
    delta = np.array([0.3, -1.2, 0.7])
    X_shifted = X + delta

    B_np = build_bone_indices(V_visual, V_proxy, k_B=k_B)
    s = torch.tensor(rng.standard_normal((N_v, k_B)), dtype=torch.float64)
    B = torch.tensor(B_np)
    V_v0 = torch.tensor(V_visual)
    V_p0 = torch.tensor(V_proxy)
    out = simplified_lbs(s, B, V_v0, V_p0, torch.tensor(X))
    out_shift = simplified_lbs(s, B, V_v0, V_p0, torch.tensor(X_shifted))
    torch.testing.assert_close(out_shift - out, torch.tensor(delta).expand_as(out), atol=1e-10, rtol=1e-10)


def test_reparam_row_stochastic():
    rng = np.random.default_rng(2)
    s = torch.tensor(rng.standard_normal((50, 8)), dtype=torch.float64)
    w = reparam_weights(s)
    assert (w >= 0).all()
    torch.testing.assert_close(w.sum(dim=-1), torch.ones(50, dtype=torch.float64), atol=1e-12, rtol=1e-12)


def test_reparam_zero_row_does_not_nan():
    """All-zero row should produce a finite (uniform-ish or zero) weight row."""
    s = torch.zeros(3, 4)
    w = reparam_weights(s)
    assert torch.isfinite(w).all()


def test_reparam_gradient_flows():
    """Backward through the weight reparam — needed for the AdamW loop."""
    s = torch.randn(5, 3, requires_grad=True)
    w = reparam_weights(s)
    w.sum().backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


# ------------------------- neighborhood builders -------------------------

def test_build_bone_indices_grid():
    """2D grid visual + grid proxy: B(i) is the 4 nearest proxy verts."""
    grid = np.array([[x, y, 0.0] for y in range(4) for x in range(4)])
    V_visual = grid + np.array([0.1, 0.1, 0.0])
    B = build_bone_indices(V_visual, grid, k_B=4)
    assert B.shape == (16, 4)
    # For a grid offset by (0.1, 0.1) the closest 4 are the surrounding cell corners.
    # Vertex 0 of V_visual is (0.1, 0.1) — closest proxy verts are 0, 1, 4, 5.
    assert set(B[0].tolist()) == {0, 1, 4, 5}


def test_build_knn_excludes_self():
    """K(i) skips the trivial self-distance-0 hit."""
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 2, 2]], dtype=float)
    K = build_knn_indices(V, k_K=2)
    assert K.shape == (5, 2)
    for i, row in enumerate(K):
        assert i not in row.tolist()


def test_build_topological_1ring_triangle():
    F = np.array([[0, 1, 2]], dtype=np.int64)
    N1, mask = build_topological_1ring(F, 3)
    assert N1.shape == (3, 2) and mask.shape == (3, 2)
    assert mask.all()
    # vertex 0 neighbors {1, 2}
    assert set(N1[0].tolist()) == {1, 2}
    assert set(N1[1].tolist()) == {0, 2}
    assert set(N1[2].tolist()) == {0, 1}


def test_build_topological_1ring_padded():
    """Two triangles sharing an edge → vertex degree varies; pad with -1."""
    F = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    N1, mask = build_topological_1ring(F, 4)
    # Vertex 1 and 2 have 3 neighbors; vertex 0, 3 have 2.
    assert mask[0].sum() == 2 and mask[3].sum() == 2
    assert mask[1].sum() == 3 and mask[2].sum() == 3
    # Padding slots are -1
    pad_slots = N1[mask == False]
    assert (pad_slots == -1).all()
