"""Differentiable simplified LBS forward + neighborhood builders.

Implements paper §4 Eq. 6: per-frame visual mesh reconstruction by displacing
each visual vertex by a weighted sum of its bone-proxy displacements.

    v_visual^{i,t} = v_visual^{i,0}
                   + Σ_{j ∈ B(i)} w_ij · (v_proxy^{j,t} − v_proxy^{j,0})

Constraint ``w_ij ≥ 0, Σ_j w_ij = 1`` is enforced by reparameterization
``w_ij = |s_ij| / Σ_k |s_ik|`` so AdamW can optimize ``s`` unconstrained.

Pure torch — no scipy / numpy in the forward path so gradients flow cleanly
through ``s``. Neighborhood builders run in numpy at setup and are converted
to long tensors before the optimizer loop.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from scipy.spatial import cKDTree


# -------------------------- forward pass (Eq. 6) --------------------------

def reparam_weights(s: torch.Tensor) -> torch.Tensor:
    """Map unconstrained ``s`` (N, k_B) to row-stochastic ``w`` ≥ 0, Σ_j = 1.

    Paper §4: ``w_ij = |s_ij| / Σ_k |s_ik|``. Adds a tiny eps to the denominator
    only as a NaN guard for the all-zero row degenerate case (AdamW won't drive
    every entry to exactly 0, but we shouldn't blow up if it did)."""
    abs_s = s.abs()
    denom = abs_s.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return abs_s / denom


def simplified_lbs(
    s: torch.Tensor,         # (N_v, k_B) trainable
    B: torch.Tensor,         # (N_v, k_B) long — bone indices into proxy verts
    V_v0: torch.Tensor,      # (N_v, 3)   visual rest pose
    V_p0: torch.Tensor,      # (N_p, 3)   proxy rest pose
    X_pt: torch.Tensor,      # (T, N_p, 3) per-frame proxy positions
) -> torch.Tensor:
    """Reconstruct the visual mesh on every frame by simplified LBS (Eq. 6).

    Returns (T, N_v, 3).
    """
    w = reparam_weights(s)                       # (N_v, k_B)
    # disp[t, i, j, :] = X_pt[t, B[i, j]] - V_p0[B[i, j]]
    disp_rest = V_p0[B]                          # (N_v, k_B, 3)
    disp_t = X_pt[:, B]                          # (T, N_v, k_B, 3)
    disp = disp_t - disp_rest                    # (T, N_v, k_B, 3) — broadcasts over T
    weighted = (w.unsqueeze(-1) * disp).sum(dim=2)  # (T, N_v, 3)
    return V_v0 + weighted


# ------------------------ neighborhood builders ------------------------

def build_bone_indices(
    V_visual: np.ndarray, V_proxy: np.ndarray, k_B: int
) -> np.ndarray:
    """B(i) — for each visual vertex, the k_B nearest proxy vertices at rest.

    Fixed across the entire optimization run (paper §4 — kNN at rest pose).
    """
    if k_B > V_proxy.shape[0]:
        raise ValueError(
            f"k_B={k_B} exceeds proxy vertex count {V_proxy.shape[0]}"
        )
    tree = cKDTree(V_proxy)
    _, B = tree.query(V_visual, k=k_B)
    if k_B == 1:
        B = B[:, None]
    return np.ascontiguousarray(B, dtype=np.int64)


def build_knn_indices(
    V_visual: np.ndarray, k_K: int
) -> np.ndarray:
    """K(i) — for each visual vertex, the k_K nearest *other* visual vertices.

    Used by L_c (kNN-ARAP collision) and L_a (attachment), both at rest pose.
    The query asks for k_K + 1 and drops index 0 (which is the vertex itself).
    """
    n = V_visual.shape[0]
    if k_K >= n:
        raise ValueError(f"k_K={k_K} must be < N_v={n}")
    tree = cKDTree(V_visual)
    _, K = tree.query(V_visual, k=k_K + 1)
    return np.ascontiguousarray(K[:, 1:], dtype=np.int64)


def build_topological_1ring(
    F_visual: np.ndarray, n_verts: int
) -> tuple[np.ndarray, np.ndarray]:
    """Topological 1-ring N(i) padded to a (N_v, k_max) matrix with -1 sentinels.

    Returns
    -------
    N1 : (N_v, k_max) int64 — neighbor indices per vertex; -1 = padding
    mask : (N_v, k_max) bool — True where N1 holds a valid neighbor

    Used by L_r (ARAP topological). Padding handled in arap_loss with the
    returned mask: invalid slots contribute 0 energy.
    """
    adj: dict[int, set[int]] = defaultdict(set)
    F = np.asarray(F_visual, dtype=np.int64)
    for tri in F:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        adj[a].add(b); adj[a].add(c)
        adj[b].add(a); adj[b].add(c)
        adj[c].add(a); adj[c].add(b)
    if n_verts == 0:
        return np.zeros((0, 0), dtype=np.int64), np.zeros((0, 0), dtype=bool)

    k_max = max((len(adj[i]) for i in range(n_verts)), default=0)
    if k_max == 0:
        # No edges at all — preserve a (N_v, 1) shape with all-padding so
        # callers don't have to special-case empty-neighborhood meshes.
        return (
            -np.ones((n_verts, 1), dtype=np.int64),
            np.zeros((n_verts, 1), dtype=bool),
        )

    N1 = -np.ones((n_verts, k_max), dtype=np.int64)
    mask = np.zeros((n_verts, k_max), dtype=bool)
    for i in range(n_verts):
        nbrs = sorted(adj[i])
        if not nbrs:
            continue
        N1[i, : len(nbrs)] = nbrs
        mask[i, : len(nbrs)] = True
    return N1, mask
