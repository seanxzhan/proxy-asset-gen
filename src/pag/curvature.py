"""§3.3 helper — per-vertex absolute mean curvature.

The paper defines ``κ_i`` as "the approximate mean curvature around ``v_i``,
computed by locally fitting a quadric function [Gatzke and Grimm 2006]", so

    |κ_i| = |H_i| = |(κ₁_i + κ₂_i) / 2|,

feeding the §3.3 smoothness weight

    w_s^{ij} = 1 − (max(|κ_i|, |κ_j|) / κ̄)⁴ ∈ [0, 1]

with ``κ̄ = max_j |κ_j|`` the per-mesh **maximum** absolute curvature (paper
§3.3, computed in ``guide_graph``). Ridges (high |κ|) → low w_s → cheap to cut →
the layer boundary naturally settles on a curvature ridge.
"""
from __future__ import annotations

import igl
import numpy as np

from pag.mesh import Mesh


def vertex_abs_curvature(mesh: Mesh, *, ring_radius: int = 3) -> np.ndarray:
    """Per-vertex |mean curvature| via libigl quadric fit over a vertex k-ring.

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
        ``|（κ₁_i + κ₂_i) / 2|`` per vertex — the paper's ``|κ_i|`` (mean curvature).
    """
    V = np.ascontiguousarray(mesh.V, dtype=np.float64)
    F = np.ascontiguousarray(mesh.F, dtype=np.int64)
    # libigl returns (PD1, PD2, PV1, PV2, bad_vertices); we only need PV1, PV2.
    # PV1 is max, PV2 is min
    _, _, pv1, pv2, _ = igl.principal_curvature(V, F, radius=ring_radius)
    # return the average
    # return np.maximum(np.asarray(pv1), np.asarray(pv2))
    return np.abs(0.5 * (np.asarray(pv1) + np.asarray(pv2))), np.abs(pv1).max()
