"""Top-level Stage 2 entry point: optimize_skinning_weights().

Drives an AdamW loop on the unconstrained reparameterization
``w_ij = |s_ij| / Σ_k |s_ik|`` (paper §4) under three losses:

  L_r — ARAP over the visual mesh's topological 1-ring (smoothness)
  L_c — ARAP over kNN(K(i)) (collision)
  L_a — attachment, ‖V − (I − K) V‖² with rest-distance-weighted K

Returns a :class:`SkinningResult` with both the dense ``s`` matrix that the
optimizer worked on and a sparse ``(N_v, N_p)`` runtime-friendly W.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
import scipy.sparse
import torch

from pag.anim import AnimationData
from pag.skinning_lbs import (
    build_bone_indices,
    build_knn_indices,
    build_topological_1ring,
    reparam_weights,
    simplified_lbs,
)
from pag.skinning_losses import arap_loss, attachment_loss


@dataclass
class SkinningResult:
    """Output of :func:`optimize_skinning_weights`.

    Attributes
    ----------
    W : scipy.sparse.csr_matrix
        Runtime-friendly skinning matrix, shape ``(N_v, N_p)``, row-stochastic
        with at most ``k_B`` non-zeros per row.
    B : (N_v, k_B) int64
        Bone (proxy-vertex) indices per visual vertex, fixed at rest pose.
    s : (N_v, k_B) float32
        Raw reparameterization values from AdamW. Useful for resume / debug.
    losses : dict
        Per-epoch lists of L_r, L_c, L_a, total — all measured on the train split.
    eval : dict
        Held-out test-split metrics (paper §5.2 internal-consistency cue).
        Empty if ``test`` was None.
    timings : dict
        Wall-clock seconds: ``setup``, ``optimize``, ``eval``, ``total``.
    knobs : dict
        Hyperparameters used, for reproduction.
    """
    W: scipy.sparse.csr_matrix
    B: np.ndarray
    s: np.ndarray
    losses: dict[str, list[float]] = field(default_factory=dict)
    eval: dict[str, float] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    knobs: dict = field(default_factory=dict)


def _to_sparse_W(s: np.ndarray, B: np.ndarray, n_proxy: int) -> scipy.sparse.csr_matrix:
    """Bake the dense (N_v, k_B) reparam values into a sparse (N_v, N_p) W."""
    N_v, k_B = s.shape
    abs_s = np.abs(s)
    denom = abs_s.sum(axis=1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    w_dense = abs_s / denom                                  # (N_v, k_B)
    rows = np.repeat(np.arange(N_v), k_B)
    cols = B.ravel()
    data = w_dense.ravel()
    M = scipy.sparse.csr_matrix(
        (data, (rows, cols)), shape=(N_v, n_proxy)
    )
    M.sum_duplicates()
    return M


def _chamfer_to_proxy(V_recon: torch.Tensor, X_proxy: torch.Tensor) -> torch.Tensor:
    """One-sided chamfer: per-frame mean nearest-proxy distance for every
    reconstructed visual vertex. Cheap proxy-side sanity for OOD blow-up.

    V_recon : (T, N_v, 3); X_proxy : (T, N_p, 3). Returns scalar (mean over t, i).
    """
    # (T, N_v, N_p) pairwise — fine for our 600 × ~10k × 128 worst case.
    diff = V_recon[:, :, None, :] - X_proxy[:, None, :, :]
    d2 = (diff ** 2).sum(dim=-1)
    return d2.min(dim=-1).values.sqrt().mean()


def optimize_skinning_weights(
    V_visual: np.ndarray,
    F_visual: np.ndarray,
    train: AnimationData,
    *,
    test: AnimationData | None = None,
    k_B: int = 8,
    k_K: int = 8,
    lambda_r: float = 1.0,
    lambda_c: float = 1.0,
    lambda_a: float = 1.0,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    eps_z: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
    log_every: int = 20,
    verbose: bool = False,
) -> SkinningResult:
    """Solve for visual-mesh LBS skinning weights via differentiable simplified LBS.

    Parameters
    ----------
    V_visual : (N_v, 3) float
    F_visual : (M_v, 3) int
    train : AnimationData — required, used as gradient signal
    test : AnimationData, optional — held-out OOD wind directions; if given,
        an internal-consistency eval is computed and stored on the result.

    k_B : bones per visual vertex (paper §4 silent — default 8)
    k_K : kNN size for L_c, L_a (paper §4 silent — default 8)
    lambda_r, lambda_c, lambda_a : loss weights (paper "set manually")
    epochs : 200 paper-pinned
    lr : 1e-3 paper-pinned
    weight_decay : AdamW default (paper-silent — kept at torch default's spirit)
    eps_z : attachment regularizer (paper Eq. 10 ε_z)

    device : "cpu" | "mps" | "cuda"
    seed : torch / numpy seed for reproducibility
    log_every : print frequency when ``verbose=True``

    Returns
    -------
    SkinningResult
    """
    knobs = dict(
        k_B=k_B, k_K=k_K,
        lambda_r=lambda_r, lambda_c=lambda_c, lambda_a=lambda_a,
        epochs=epochs, lr=lr, weight_decay=weight_decay,
        eps_z=eps_z, device=device, seed=seed,
    )
    timings: dict[str, float] = {}
    t_start = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)

    N_v = V_visual.shape[0]
    V_p0_np = train.V_proxy_rest
    N_p = V_p0_np.shape[0]

    # ---------- setup: neighborhoods (numpy / scipy) ----------
    t0 = time.perf_counter()
    B_np = build_bone_indices(V_visual, V_p0_np, k_B=k_B)            # (N_v, k_B)
    K_np = build_knn_indices(V_visual, k_K=k_K)                      # (N_v, k_K)
    N1_np, mask_np = build_topological_1ring(F_visual, N_v)          # (N_v, k_max)

    dev = torch.device(device)
    dtype = torch.float32       # AdamW + 600-frame batches → fp32 for memory
    V_v0 = torch.tensor(V_visual, dtype=dtype, device=dev)
    V_p0 = torch.tensor(V_p0_np, dtype=dtype, device=dev)
    X_train = torch.tensor(train.X, dtype=dtype, device=dev)         # (T, N_p, 3)
    B_t = torch.tensor(B_np, dtype=torch.long, device=dev)
    K_t = torch.tensor(K_np, dtype=torch.long, device=dev)
    N1_t = torch.tensor(N1_np, dtype=torch.long, device=dev)
    mask_t = torch.tensor(mask_np, dtype=torch.bool, device=dev)

    # Initial s: ones → uniform 1/k_B weights, a neutral starting point.
    s = torch.ones(N_v, k_B, dtype=dtype, device=dev, requires_grad=True)
    opt = torch.optim.AdamW([s], lr=lr, weight_decay=weight_decay)
    timings["setup"] = time.perf_counter() - t0

    # ---------- optimization loop ----------
    losses: dict[str, list[float]] = {"L_r": [], "L_c": [], "L_a": [], "total": []}
    t0 = time.perf_counter()
    for epoch in range(epochs):
        opt.zero_grad()
        V_vt = simplified_lbs(s, B_t, V_v0, V_p0, X_train)           # (T, N_v, 3)
        l_r = arap_loss(V_vt, V_v0, N1_t, mask_t)
        l_c = arap_loss(V_vt, V_v0, K_t)
        l_a = attachment_loss(V_vt, V_v0, K_t, eps_z=eps_z)
        
        total = lambda_r * l_r + lambda_c * l_c + lambda_a * l_a
        total.backward()
        opt.step()

        losses["L_r"].append(l_r.detach().item())
        losses["L_c"].append(l_c.detach().item())
        losses["L_a"].append(l_a.detach().item())
        losses["total"].append(total.detach().item())
        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            print(
                f"  epoch {epoch:4d}/{epochs}  "
                f"L_r={losses['L_r'][-1]:.4e}  L_c={losses['L_c'][-1]:.4e}  "
                f"L_a={losses['L_a'][-1]:.4e}  total={losses['total'][-1]:.4e}"
            )
    timings["optimize"] = time.perf_counter() - t0

    # ---------- held-out test eval ----------
    eval_metrics: dict[str, float] = {}
    if test is not None:
        t0 = time.perf_counter()
        with torch.no_grad():
            X_test = torch.tensor(test.X, dtype=dtype, device=dev)
            V_vt_test = simplified_lbs(s, B_t, V_v0, V_p0, X_test)
            eval_metrics["test_L_r"] = float(arap_loss(V_vt_test, V_v0, N1_t, mask_t))
            eval_metrics["test_L_c"] = float(arap_loss(V_vt_test, V_v0, K_t))
            eval_metrics["test_L_a"] = float(attachment_loss(V_vt_test, V_v0, K_t, eps_z=eps_z))
            eval_metrics["test_chamfer_to_proxy"] = float(
                _chamfer_to_proxy(V_vt_test, X_test)
            )
            # Train-side chamfer for ratio-based regression detection.
            with torch.no_grad():
                V_vt_train = simplified_lbs(s, B_t, V_v0, V_p0, X_train)
                eval_metrics["train_chamfer_to_proxy"] = float(
                    _chamfer_to_proxy(V_vt_train, X_train)
                )
        timings["eval"] = time.perf_counter() - t0

    s_np = s.detach().cpu().numpy()
    W = _to_sparse_W(s_np, B_np, N_p)
    timings["total"] = time.perf_counter() - t_start

    return SkinningResult(
        W=W,
        B=B_np,
        s=s_np,
        losses=losses,
        eval=eval_metrics,
        timings=timings,
        knobs=knobs,
    )
