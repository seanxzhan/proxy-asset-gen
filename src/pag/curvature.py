"""§3.3 helper — per-vertex absolute principal curvature.

|κ_i| = max(|κ₁_i|, |κ₂_i|), feeding the §3.3 smoothness weight

    w_s^{ij} = 1 − (max(|κ_i|, |κ_j|) / κ̄)⁴ ∈ [0, 1]

where κ̄ is the per-mesh mean of |κ|. Ridges (high |κ|) → low w_s → cheap to
cut → the layer boundary naturally settles on a curvature ridge.
"""
from __future__ import annotations

import igl
import numpy as np

from pag.mesh import Mesh


def vertex_abs_curvature(mesh: Mesh, *, ring_radius: int = 3) -> np.ndarray:
    """Per-vertex |κ| via libigl quadric fit over a vertex k-ring.

    Parameters
    ----------
    mesh : Mesh
        Triangle mesh — typically ``M_proj`` post-§3.2.
    ring_radius : int, default 3
        k-ring size for the quadric fit. Larger → smoother κ but slower.
        ``radius=1`` would fit each quadric on the immediate 1-ring; the libigl
        default is 5. We pick 3 as a reasonable middle ground.

    Returns
    -------
    abs_kappa : (N,) float64
        ``max(|κ₁_i|, |κ₂_i|)`` per vertex.
    """
    V = np.ascontiguousarray(mesh.V, dtype=np.float64)
    F = np.ascontiguousarray(mesh.F, dtype=np.int64)
    # libigl returns (PD1, PD2, PV1, PV2, bad_vertices); we only need PV1, PV2.
    _, _, pv1, pv2, _ = igl.principal_curvature(V, F, radius=ring_radius)
    return np.maximum(np.abs(np.asarray(pv1)), np.abs(np.asarray(pv2)))
