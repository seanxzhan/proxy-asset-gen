"""§3.3 — Guide graph for single-layer extraction.

For each vertex ``v_i`` on ``M_proj`` with vertex normal ``n_i``:

  1. Cast a ray ``(v_i, −n_i)``. The first triangle hit is a candidate
     "opposite layer". It survives if both filters pass:
       - hit distance ``< 2 · voxel_size``  (paper §3.3 — anything farther
         can't be the matching iso layer).
       - hit-triangle face normal nearly antiparallel to ``n_i``:
         ``n_face · n_i < −1 + ε_o``.
     If it survives, pair ``i`` with the *closest* vertex of the hit triangle
     and record ``(i, j) ∈ E_o`` (canonicalised so ``i < j``, deduped).

  2. Cast a ray ``(v_i, +n_i)``. ``d_+^i`` = first hit distance, or ``∞`` if no
     hit. Outer-layer verts have ``d_+ = ∞`` since their ``+n`` ray escapes;
     inner-layer verts hit the opposite shell from inside. The §3.3 ILP bias
     uses ``sign(d_+^i − d_+^j)`` to break ties toward the outer label.

Smoothness weights on every mesh edge:

    w_s^{ij} = 1 − ( max(|κ_i|, |κ_j|) / κ̄ )⁴   ∈ [0, 1]

so layer boundaries settle on curvature ridges.

The procedure is intentionally permissive — duplicate, mismatched, or missing
opposite edges are tolerated. The §3.3 ILP cleans up.
"""
from __future__ import annotations

from dataclasses import dataclass

import igl
import numpy as np

from pag.curvature import vertex_abs_curvature
from pag.mesh import Mesh


@dataclass
class GuideGraph:
    """Inputs to :func:`pag.ilp.solve_layer_ilp`.

    Attributes
    ----------
    mesh_edges : (E_s, 2) int64
        Unique undirected edges on ``M_proj`` — these carry the smoothness
        term in the ILP.
    smoothness_w : (E_s,) float64
        Per-edge smoothness weight ``w_s ∈ [0, 1]`` (low = cheap to cut).
    opposite_edges : (E_o, 2) int64
        Canonicalised, deduplicated opposite-vertex pairs. ``i < j`` per row.
    d_plus : (N,) float64
        Per-vertex distance to the first ``+n_i``-ray hit; ``∞`` for misses.
    """
    mesh_edges: np.ndarray
    smoothness_w: np.ndarray
    opposite_edges: np.ndarray
    d_plus: np.ndarray


def build_guide_graph(
    mesh: Mesh,
    *,
    voxel_size: float,
    abs_curvature: np.ndarray | None = None,
    eps_normal: float = 0.1,
    eps_origin: float = 1e-6,
    curvature_radius: int = 3,
) -> GuideGraph:
    """Construct the §3.3 guide graph from ``M_proj``.

    Parameters
    ----------
    mesh : Mesh
        Typically ``M_proj`` (post-§3.2). Need not be closed.
    voxel_size : float
        The §3.1 ``D / N_v``. Sets the opposite-edge distance filter
        (``< 2 · voxel_size``).
    abs_curvature : (N,) optional
        Pre-computed ``max(|κ₁|, |κ₂|)`` per vertex. Computed via libigl if
        not provided.
    eps_normal : float, default 0.1
        Opposite-edge filter: ``n_face · n_i < −1 + eps_normal``.
    eps_origin : float, default 1e-6
        Ray-origin offset along the ray direction, to skip self-hits at the
        source vertex's own incident triangles. Should be ≪ ``voxel_size``.
    curvature_radius : int, default 3
        k-ring size if ``abs_curvature`` is computed here.

    Returns
    -------
    GuideGraph
    """
    V = np.ascontiguousarray(mesh.V, dtype=np.float64)
    F = np.ascontiguousarray(mesh.F, dtype=np.int64)
    N = V.shape[0]

    if abs_curvature is None:
        abs_curvature = vertex_abs_curvature(mesh, ring_radius=curvature_radius)
    abs_curvature = np.asarray(abs_curvature, dtype=np.float64)

    # ---- smoothness weights w_s = 1 − (max(κ_i, κ_j)/κ̄)⁴, capped to [0, 1].
    kappa_mean = float(abs_curvature.mean()) if abs_curvature.size else 0.0
    if kappa_mean > 0.0:
        ratio = np.minimum(abs_curvature / kappa_mean, 1.0)
        w_per_vert = 1.0 - ratio ** 4
    else:
        # Degenerate (all-flat) mesh — every edge is "smoothness-protected".
        w_per_vert = np.ones(N, dtype=np.float64)
    edges = mesh.edges
    # max(κ_i, κ_j) → min(w_per_vert_i, w_per_vert_j)  since w decreases in κ.
    smoothness_w = np.minimum(w_per_vert[edges[:, 0]], w_per_vert[edges[:, 1]])

    # ---- ray cast both ±n. Build AABB once; query both directions.
    n_vert = mesh.vertex_normals()
    f_normals = mesh.face_normals(normalize=True)

    tree = igl.AABB()
    tree.init(V, F)

    # +n cast: d_plus
    src_pos = V + eps_origin * n_vert
    idx_pos, t_pos, _ = tree.intersect_ray_first(V, F, src_pos, n_vert)
    d_plus = np.full(N, np.inf, dtype=np.float64)
    hit_pos = (idx_pos >= 0)
    d_plus[hit_pos] = np.asarray(t_pos)[hit_pos]

    # -n cast: opposite candidates
    dir_neg = -n_vert
    src_neg = V + eps_origin * dir_neg
    idx_neg, t_neg, _ = tree.intersect_ray_first(V, F, src_neg, dir_neg)
    hit_neg = (idx_neg >= 0)
    # Hit world positions reconstructed from t (libigl returns barycentrics).
    locs_neg = src_neg + np.asarray(t_neg)[:, None] * dir_neg

    opposite_pairs: set[tuple[int, int]] = set()
    for i in np.flatnonzero(hit_neg):
        d_hit = float(t_neg[i])
        if d_hit >= 2.0 * voxel_size:
            continue
        tri = int(idx_neg[i])
        n_face = f_normals[tri]
        # Filter: face normal nearly antiparallel to vertex normal.
        if float(n_face @ n_vert[i]) >= -1.0 + eps_normal:
            continue
        # Pick closest vertex of the hit triangle as the opposite partner.
        tri_vs = F[tri]
        d_tri = np.linalg.norm(V[tri_vs] - locs_neg[i], axis=1)
        j = int(tri_vs[d_tri.argmin()])
        if j == int(i):
            continue
        a, b = (int(i), j) if i < j else (j, int(i))
        opposite_pairs.add((a, b))

    if opposite_pairs:
        opposite_edges = np.array(sorted(opposite_pairs), dtype=np.int64)
    else:
        opposite_edges = np.zeros((0, 2), dtype=np.int64)

    return GuideGraph(
        mesh_edges=edges,
        smoothness_w=smoothness_w,
        opposite_edges=opposite_edges,
        d_plus=d_plus,
    )
