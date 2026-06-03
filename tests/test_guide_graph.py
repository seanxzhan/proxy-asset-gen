"""§3.3 — guide graph construction.

Validation:
  - smoothness weights ∈ [0, 1] and shape matches mesh.edges
  - opposite edges canonical (i < j) and deduplicated
  - thin parallel slab → opposite edges pair the two layers; all d_+^outer = ∞
  - distance filter: layers > 2·voxel_size apart yield no opposite edges
  - normal filter: same-orientation parallel layers yield no opposite edges
"""
from __future__ import annotations

import numpy as np

from pag.guide_graph import build_guide_graph
from pag.mesh import build_mesh


# -------------------- helpers --------------------

def _two_parallel_quads(
    sep: float, *, n: int = 6, side: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Two flat quads at z=±sep/2, normals pointing AWAY from each other.

    This mirrors the §3.3 double-cover geometry of ``M_iso`` for a thin sheet:
    the two layers' outward normals each point away from the other, so the −n
    ray from one layer crosses the slab and lands on the matching layer.

      - Bottom (z=−sep/2): normals point −z
      - Top    (z=+sep/2): normals point +z
    """
    xs = np.linspace(-side / 2, side / 2, n)
    ys = np.linspace(-side / 2, side / 2, n)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    V_b = np.stack([xx.ravel(), yy.ravel(), np.full(n * n, -sep / 2)], axis=1)
    V_t = V_b.copy()
    V_t[:, 2] = sep / 2

    # Top quad: CCW from +z → normal +z (away from bottom).
    F_t_rel: list[list[int]] = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i; b = a + 1; c = a + n; d = c + 1
            F_t_rel.append([a, b, d])
            F_t_rel.append([a, d, c])
    # Bottom quad: opposite winding → normal −z (away from top).
    F_b_rel = [[f[0], f[2], f[1]] for f in F_t_rel]

    F_b = F_b_rel
    F_t = [[v + n * n for v in f] for f in F_t_rel]

    V = np.vstack([V_b, V_t])
    F = np.array(F_b + F_t, dtype=np.int64)
    return V, F


def _two_same_facing_quads(
    sep: float, *, n: int = 6, side: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Two quads at z=±sep/2, both facing +z. Normals are aligned, not antiparallel,
    so the ε_normal filter must reject every opposite-edge candidate."""
    xs = np.linspace(-side / 2, side / 2, n)
    ys = np.linspace(-side / 2, side / 2, n)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    V_b = np.stack([xx.ravel(), yy.ravel(), np.full(n * n, -sep / 2)], axis=1)
    V_t = V_b.copy()
    V_t[:, 2] = sep / 2

    F_b: list[list[int]] = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b = a + 1
            c = a + n
            d = c + 1
            F_b.append([a, b, d])
            F_b.append([a, d, c])
    F_t = [[f[0] + n * n, f[1] + n * n, f[2] + n * n] for f in F_b]

    V = np.vstack([V_b, V_t])
    F = np.array(F_b + F_t, dtype=np.int64)
    return V, F


# -------------------- tests --------------------

def test_smoothness_in_unit_interval():
    V, F = _two_parallel_quads(sep=0.05, n=4)
    mesh = build_mesh(V, F)
    g = build_guide_graph(mesh, voxel_size=0.05)
    assert g.smoothness_w.shape == (mesh.edges.shape[0],)
    assert (g.smoothness_w >= 0.0).all()
    assert (g.smoothness_w <= 1.0).all()
    # Flat mesh — curvature is ~zero everywhere → weights pinned at 1.
    assert np.median(g.smoothness_w) > 0.95


def test_opposite_edges_canonical_and_deduped():
    V, F = _two_parallel_quads(sep=0.05, n=5)
    mesh = build_mesh(V, F)
    g = build_guide_graph(mesh, voxel_size=0.05)
    if g.opposite_edges.size:
        a = g.opposite_edges[:, 0]
        b = g.opposite_edges[:, 1]
        assert (a < b).all(), "opposite_edges must be canonicalised (a < b)"
        # Dedup: each (a,b) appears once.
        as_tuples = {tuple(r) for r in g.opposite_edges}
        assert len(as_tuples) == g.opposite_edges.shape[0]


def test_thin_slab_pairs_layers():
    """Two parallel quads with antiparallel normals → many opposite pairs.

    Each pair must connect a bottom-layer vertex to a top-layer vertex (i.e.
    cross the slab). The exact count depends on which rays land on the
    matching grid vertex vs. an adjacent triangle, but should be > N/4.
    """
    V, F = _two_parallel_quads(sep=0.05, n=6)
    mesh = build_mesh(V, F)
    n_per_layer = 36
    g = build_guide_graph(mesh, voxel_size=0.05)
    assert g.opposite_edges.shape[0] > n_per_layer // 4
    # Every edge crosses layers (one endpoint < n_per_layer, other ≥ n_per_layer).
    crosses = (g.opposite_edges[:, 0] < n_per_layer) & (g.opposite_edges[:, 1] >= n_per_layer)
    assert crosses.all(), "every opposite edge must cross the slab"


def test_distance_filter_rejects_far_apart():
    """If layers are > 2·voxel_size apart, opposite-edge filter rejects them."""
    sep = 0.20
    voxel = 0.05  # 2*voxel = 0.10 < sep
    V, F = _two_parallel_quads(sep=sep, n=5)
    mesh = build_mesh(V, F)
    g = build_guide_graph(mesh, voxel_size=voxel)
    assert g.opposite_edges.shape[0] == 0


def test_normal_filter_rejects_same_orientation():
    """Both quads facing +z → no antiparallel hit → no opposite edges."""
    V, F = _two_same_facing_quads(sep=0.05, n=5)
    mesh = build_mesh(V, F)
    g = build_guide_graph(mesh, voxel_size=0.05)
    assert g.opposite_edges.shape[0] == 0


def test_d_plus_infinite_on_outer_layers():
    """Bottom layer's +n ray escapes downward — d_+ = ∞.
    Top layer's +n ray escapes upward — d_+ = ∞.
    The thin slab is too thin for any +n ray to hit anything.
    """
    V, F = _two_parallel_quads(sep=0.05, n=4)
    mesh = build_mesh(V, F)
    g = build_guide_graph(mesh, voxel_size=0.05)
    # All vertices' +n rays escape (since the +n direction always points away from
    # the other layer in this geometry).
    assert np.isinf(g.d_plus).all()


def test_d_plus_shape_matches_n_verts():
    V, F = _two_parallel_quads(sep=0.05, n=4)
    mesh = build_mesh(V, F)
    g = build_guide_graph(mesh, voxel_size=0.05)
    assert g.d_plus.shape == (mesh.n_verts,)
    assert g.d_plus.dtype == np.float64


def test_smoothness_weight_matches_paper_formula():
    """w_s^{ij} = 1 − (max(|κ_i|, |κ_j|) / κ̄)⁴, clamped to [0, 1].

    Pass abs_curvature directly so the formula isolates from libigl. Compute
    expected w by hand for a 3-vertex triangle and assert exact agreement.
    """
    # Triangle with vertices (0,1,2). Edges (sorted): (0,1), (0,2), (1,2).
    V = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=np.float64)
    F = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = build_mesh(V, F)

    # Choose curvatures with mean 1/6 (so ratios are clean).
    abs_kappa = np.array([0.0, 0.1, 0.4])  # κ̄ = 0.5/3 ≈ 0.16667
    g = build_guide_graph(mesh, voxel_size=1.0, abs_curvature=abs_kappa)

    kbar = abs_kappa.mean()
    # Per-vertex ratio clamped to 1 → per-vertex w = 1 - ratio⁴ ∈ [0, 1].
    ratio = np.minimum(abs_kappa / kbar, 1.0)
    w_per_vert = 1.0 - ratio ** 4
    # mesh.edges is sorted ascending → expect [(0,1), (0,2), (1,2)].
    expected = np.array([
        min(w_per_vert[0], w_per_vert[1]),
        min(w_per_vert[0], w_per_vert[2]),
        min(w_per_vert[1], w_per_vert[2]),
    ])
    np.testing.assert_allclose(g.smoothness_w, expected, atol=1e-12)


def test_smoothness_clamped_when_kappa_above_mean():
    """When max(|κ_i|, |κ_j|) ≫ κ̄, the paper formula yields negative w_s.
    Our impl clamps the per-vertex ratio to 1, giving w_s = 0 instead.

    A negative weight would *reward* cuts in the ILP (a sign error). This
    test pins down the clamping behaviour as intentional, not a bug.
    """
    V = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=np.float64)
    F = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = build_mesh(V, F)
    abs_kappa = np.array([0.0, 0.0, 10.0])  # κ̄ = 10/3, vertex 2 ratio > 1
    g = build_guide_graph(mesh, voxel_size=1.0, abs_curvature=abs_kappa)
    # Edges (0,2) and (1,2) involve the high-κ vertex → w=0 (clamped).
    # Edge (0,1) has κ̄-only verts on both sides → w=1.
    np.testing.assert_allclose(g.smoothness_w, [1.0, 0.0, 0.0], atol=1e-12)


def test_jacket_pipeline_runs():
    """End-to-end §3.1 + §3.2 + §3.3 guide-graph build on jacket.obj."""
    from pag import isosurface_from_udf, load_obj, project_to_visual
    V, F = load_obj("/Users/szhan/projects/proxy-asset-gen/data/jacket.obj")
    iso = isosurface_from_udf(V, F, n_v=32)
    proj = project_to_visual(iso.mesh, V, F, n_iters=50, lr=5e-3, lambda_L=0.1)
    g = build_guide_graph(proj.mesh, voxel_size=iso.voxel_size)
    # Cloth doubles up under UDF, so opposite edges should be found.
    assert g.opposite_edges.shape[0] > 0
    # Some +n rays land on the opposite shell (inner verts), some escape (outer).
    assert np.isfinite(g.d_plus).any()
    assert np.isinf(g.d_plus).any()
