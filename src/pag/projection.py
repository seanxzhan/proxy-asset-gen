"""§3.2 — Project ``M_iso`` onto ``M_visual`` with vector Adam.

``M_iso`` is topologically clean (watertight, manifold) but loose: every vertex
sits ~one voxel off the input. We tighten it by minimising

    Σ_i ||v_i − p_visual^i||²  +  λ_L · Σ_i ||v_i − μ_i||²              (Eq §3.2)

where

  - ``p_visual^i`` is the closest point on ``M_visual``  (libigl AABB query)
  - ``μ_i = (1/|N(i)|) Σ_{j ∈ N(i)} v_j``  is the uniform-Laplacian centroid

Topology is unchanged — only ``V`` moves; ``F`` and ``edges`` are reused.

Solver: **Vector Adam** [Ling 2022, "Vector-Adam"]. Standard Adam normalises
each Cartesian coordinate by its own ``√v̂``, which biases steps toward the
coordinate axes (a sphere does not stay round). Vector Adam keeps the first
moment per-vertex as a 3-vector but accumulates the *scalar* ``‖g‖²`` for the
second moment, making the per-vertex update rotation-equivariant. Defaults
β₁=0.9, β₂=0.999, ε=1e-8 mirror torch's Adam.

Self-intersection is allowed by design — runtime PBD in games disables
self-collision anyway, and CCD here would dominate the §3 wall-clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import igl
import numpy as np

from pag.mesh import Mesh, build_mesh


@dataclass
class ProjectionResult:
    """Output of :func:`project_to_visual`.

    ``history`` is empty unless ``record_history=True``. When populated it
    contains length-``n_iters`` arrays for each energy term, useful for
    diagnostics and the ``test_data_energy_decreases`` regression test.
    """
    mesh: Mesh
    n_iters: int
    knobs: dict[str, float] = field(default_factory=dict)
    history: dict[str, np.ndarray] = field(default_factory=dict)


def project_to_visual(
    iso_mesh: Mesh,
    V_visual: np.ndarray,
    F_visual: np.ndarray,
    *,
    n_iters: int = 200,
    lr: float = 1e-2,
    lambda_L: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    record_history: bool = False,
) -> ProjectionResult:
    """Pull each ``M_iso`` vertex toward its closest point on ``M_visual``.

    Parameters
    ----------
    iso_mesh : Mesh
        Output of :func:`pag.udf.isosurface_from_udf`.
    V_visual, F_visual : ndarray
        The original (possibly non-manifold) visual mesh.
    n_iters : int, default 200
        Vector-Adam iterations. Paper-silent — 200 fits a sphere to <0.1× voxel
        size; 100 is usually enough but the reference number used downstream.
    lr : float, default 1e-2
        Adam step size.
    lambda_L : float, default 0.1
        Smoothness weight. λ_L = 0 → tightest fit but lumpy; large λ_L →
        smooth but stays loose.
    beta1, beta2, eps : float
        Adam moment / numerical defaults.
    record_history : bool, default False
        If True, ``result.history`` is filled with per-iter ``data``,
        ``smooth``, ``total`` energies (handy for diagnostics, ~1% slower).

    Returns
    -------
    ProjectionResult
    """
    if n_iters < 0:
        raise ValueError(f"n_iters must be ≥ 0; got {n_iters}")

    V = np.array(iso_mesh.V, dtype=np.float64, copy=True)
    F = iso_mesh.F
    edges = iso_mesh.edges                          # (E, 2) undirected, unique
    n = V.shape[0]

    V_v = np.ascontiguousarray(V_visual, dtype=np.float64)
    F_v = np.ascontiguousarray(F_visual, dtype=np.int64)

    # Per-vertex degree on the M_iso 1-ring (used by the uniform Laplacian).
    deg = np.zeros(n, dtype=np.int64)
    np.add.at(deg, edges[:, 0], 1)
    np.add.at(deg, edges[:, 1], 1)
    inv_deg = np.zeros(n, dtype=np.float64)
    inv_deg[deg > 0] = 1.0 / deg[deg > 0]

    # Vector-Adam state. Per-vertex first moment is a 3-vector; per-vertex
    # second moment is a *scalar* (||g||²) — the whole point of vector Adam.
    m = np.zeros_like(V)
    s = np.zeros(n, dtype=np.float64)

    hist_data: list[float] = []
    hist_smooth: list[float] = []
    hist_total: list[float] = []

    for t in range(1, n_iters + 1):
        # Closest point on M_visual for each iso vertex.
        sqrD, _, C = igl.point_mesh_squared_distance(V, V_v, F_v)

        # Uniform-Laplacian centroids μ_i = (1/deg(i)) Σ_{j ∈ N(i)} v_j.
        mu = np.zeros_like(V)
        np.add.at(mu, edges[:, 0], V[edges[:, 1]])
        np.add.at(mu, edges[:, 1], V[edges[:, 0]])
        mu *= inv_deg[:, None]
        delta = V - mu                              # (N, 3) residual

        # ∇_v_i Σ_k ||v_k − μ_k||²  =  2 δ_i − 2 Σ_{k ∈ N(i)} δ_k / deg(k).
        # (See module docstring; this is the *exact* gradient of the
        # uniform-Laplacian quadratic, not the heuristic δ_i alone.)
        weighted = delta * inv_deg[:, None]
        pulled = np.zeros_like(V)
        np.add.at(pulled, edges[:, 0], weighted[edges[:, 1]])
        np.add.at(pulled, edges[:, 1], weighted[edges[:, 0]])
        grad_smooth = 2.0 * delta - 2.0 * pulled

        grad_data = 2.0 * (V - C)                   # ∇ Σ ||v − p||²
        g = grad_data + lambda_L * grad_smooth

        # ----- Vector Adam (Ling 2022) -----------------------------------
        m = beta1 * m + (1.0 - beta1) * g
        s = beta2 * s + (1.0 - beta2) * np.einsum("ij,ij->i", g, g)
        m_hat = m / (1.0 - beta1 ** t)
        s_hat = s / (1.0 - beta2 ** t)
        V = V - lr * m_hat / (np.sqrt(s_hat)[:, None] + eps)

        if record_history:
            data = float(sqrD.sum())
            smooth = float((delta * delta).sum())
            hist_data.append(data)
            hist_smooth.append(smooth)
            hist_total.append(data + lambda_L * smooth)

    history: dict[str, np.ndarray] = {}
    if record_history:
        history = {
            "data": np.asarray(hist_data),
            "smooth": np.asarray(hist_smooth),
            "total": np.asarray(hist_total),
        }

    return ProjectionResult(
        mesh=build_mesh(V, F),
        n_iters=n_iters,
        knobs={
            "lr": lr, "lambda_L": lambda_L,
            "beta1": beta1, "beta2": beta2, "eps": eps,
        },
        history=history,
    )
