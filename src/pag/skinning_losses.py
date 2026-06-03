"""ARAP and attachment losses for skinning weight optimization (paper §4).

Three losses share two helpers (`arap_loss`, `attachment_loss`):

  L_r  — ARAP over the visual mesh's topological 1-ring (smoothness / fabric)
  L_c  — same form, but over kNN(K(i)) (collision / inter-layer penetration)
  L_a  — attachment: ‖V − (I − K) V‖² with rest-distance-weighted K

Rotations for ARAP are recovered closed-form via per-vertex SVD on the
cross-covariance ``S = Σ_j e_0_{ij} ⊗ e_t_{ij}``. The standard alternating-min
trick: with ``R`` fixed (under ``no_grad``), the residual ``‖e_t − R e_0‖²``
is differentiable w.r.t. the trainable ``s`` only via ``e_t``.

Attachment uses the algebraic identity

    L_a = ‖V − (I − K) V‖² = ‖K V‖²

(paper Eq. 9). We compute the long-form ``V − weighted_neigh`` to mirror the
paper's notation exactly — algebraically equivalent but easier to read.
"""
from __future__ import annotations

import torch


# --------------------------- ARAP ---------------------------

def arap_loss(
    V_t: torch.Tensor,         # (T, N, 3) — current positions
    V_0: torch.Tensor,         # (N, 3)    — rest positions
    neigh: torch.Tensor,       # (N, k) long — neighbor indices, -1 = padding
    mask: torch.Tensor | None = None,   # (N, k) bool — True where neigh is valid
) -> torch.Tensor:
    """ARAP loss with closed-form per-vertex rotation.

    For each vertex i and frame t,

        L_i^t = min_{R ∈ SO(3)}  Σ_{j ∈ N(i)} ‖(v_t^i − v_t^j) − R (v_0^i − v_0^j)‖²

    The optimal R is recovered from the SVD of S = Σ_j e_0_{ij} ⊗ e_t_{ij}
    with a det-correction so R is a proper rotation (det = +1). R is treated
    as a constant for backward — gradients flow only through ``V_t``.

    Padding (``-1`` slots in ``neigh``) is masked out before the squared-norm
    sum so it cannot leak energy into nonexistent neighbors.
    """
    T, N, _ = V_t.shape
    k = neigh.shape[1]
    if mask is None:
        mask = neigh >= 0
    # Replace -1 indices with 0 so the gather is safe; mask zeros out their
    # contribution before any reduction.
    safe_neigh = neigh.clamp_min(0)

    # e_0[i, j, :] = V_0[neigh[i, j]] - V_0[i]      (N, k, 3)
    e_0 = V_0[safe_neigh] - V_0[:, None, :]
    # e_t[t, i, j, :] = V_t[t, neigh[i, j]] - V_t[t, i]      (T, N, k, 3)
    e_t = V_t[:, safe_neigh, :] - V_t[:, :, None, :]

    m = mask.to(V_t.dtype)                              # (N, k)
    e_0_m = e_0 * m[..., None]                          # zero out padding
    e_t_m = e_t * m[None, :, :, None]

    # Cross-covariance S^{t,i} = Σ_j (e_0_{ij}) (e_t_{ij})^T  ∈ R^{3x3}
    # Use einsum: 'nki,tnkj->tnij'
    S = torch.einsum("nki,tnkj->tnij", e_0_m, e_t_m)    # (T, N, 3, 3)

    with torch.no_grad():
        # SVD in float32 on float32 input is fine; for tests we run float64.
        U, _, Vh = torch.linalg.svd(S)
        R = (Vh.transpose(-1, -2) @ U.transpose(-1, -2))     # (T, N, 3, 3)
        det = torch.linalg.det(R)                             # (T, N)
        # Det correction: flip the sign of the last row of Vh where det < 0
        # so R becomes a proper rotation. This is the standard ARAP-SVD recipe.
        sign = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
        Vh_fix = Vh.clone()
        Vh_fix[..., -1, :] = Vh_fix[..., -1, :] * sign.unsqueeze(-1)
        R = (Vh_fix.transpose(-1, -2) @ U.transpose(-1, -2))  # (T, N, 3, 3)

    # rotated[t, i, j, :] = R[t, i] @ e_0[i, j]          (T, N, k, 3)
    rotated = torch.einsum("tnij,nkj->tnki", R, e_0_m)
    diff = (e_t_m - rotated) * m[None, :, :, None]
    return (diff ** 2).sum()


# --------------------------- Attachment ---------------------------

def attachment_loss(
    V_t: torch.Tensor,         # (T, N, 3)
    V_0: torch.Tensor,         # (N, 3)
    K_idx: torch.Tensor,       # (N, k) long — kNN at rest pose, no padding
    eps_z: float = 1e-3,
) -> torch.Tensor:
    """Paper Eq. 9: ``L_a = Σ_t ‖V_t − (I − K) V_t‖²``.

    K is row-stochastic with weights ``1 / (‖v_0^i − v_0^j‖ + ε)``, normalized
    so each row sums to 1. Geometrically, this softly glues each vertex to a
    rest-distance-weighted average of its kNN — disconnected components that
    happen to be close at rest get pulled toward each other.
    """
    rest_dist = (V_0[K_idx] - V_0[:, None, :]).norm(dim=-1) + eps_z   # (N, k)
    K_w = 1.0 / rest_dist
    K_w = K_w / K_w.sum(dim=-1, keepdim=True)                          # row-stochastic
    weighted_neigh = (K_w[None, ..., None] * V_t[:, K_idx, :]).sum(dim=2)
    return ((V_t - weighted_neigh) ** 2).sum()
