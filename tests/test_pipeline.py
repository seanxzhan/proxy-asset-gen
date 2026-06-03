"""End-to-end §3.1 + §3.2 + §3.3 pipeline tests on synthetic shapes.

Where unit tests pin formulas and isolated components, these tests check that
the *composition* of all three stages produces topologically sensible output
on shapes whose answer we can hand-derive from the paper.

Key assertion pattern: for a closed shape, M_iso has 2 shells (inner+outer).
After §3.3, the kept set should map almost perfectly onto the *outer* shell —
the one whose +n ray escapes to infinity (d_+ = ∞), which the §3.3 bias
prefers. We record the pre-projection shell membership (radii > input radius)
and verify post-ILP labels match.
"""
from __future__ import annotations

import numpy as np

from pag import (
    build_guide_graph,
    isosurface_from_udf,
    project_to_visual,
    solve_layer_ilp,
)


# -------------------- helpers --------------------

def _icosphere(
    radius: float = 1.0, subdivs: int = 3, center: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    t = (1.0 + 5.0 ** 0.5) / 2.0
    V = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=float)
    F = np.array([
        [0,11, 5],[0, 5, 1],[0, 1, 7],[0, 7,10],[0,10,11],
        [1, 5, 9],[5,11, 4],[11,10, 2],[10, 7, 6],[7, 1, 8],
        [3, 9, 4],[3, 4, 2],[3, 2, 6],[3, 6, 8],[3, 8, 9],
        [4, 9, 5],[2, 4,11],[6, 2,10],[8, 6, 7],[9, 8, 1],
    ], dtype=np.int64)
    cache: dict[tuple[int, int], int] = {}
    V_list = [v.copy() for v in V]

    def _mid(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key in cache:
            return cache[key]
        m = 0.5 * (V_list[a] + V_list[b])
        idx = len(V_list)
        V_list.append(m)
        cache[key] = idx
        return idx

    for _ in range(subdivs):
        new_F = []
        for tri in F:
            a, b, c = tri
            ab = _mid(a, b); bc = _mid(b, c); ca = _mid(c, a)
            new_F.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        F = np.array(new_F, dtype=np.int64)

    Vf = np.array(V_list)
    Vf = radius * Vf / np.linalg.norm(Vf, axis=1, keepdims=True)
    if center is not None:
        Vf = Vf + np.asarray(center, dtype=np.float64)
    return Vf, F


def _hemisphere_open(
    radius: float = 1.0, subdivs: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Open hemisphere (z >= 0). M_iso wraps both sides → 1 closed component."""
    V, F = _icosphere(radius, subdivs)
    keep_face = (V[F][:, :, 2] >= -1e-9).all(axis=1)
    F2 = F[keep_face]
    used = np.unique(F2.ravel())
    remap = -np.ones(V.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.size)
    return V[used], remap[F2]


def _full_pipeline(
    V: np.ndarray, F: np.ndarray, *, n_v: int = 24,
    n_iters: int = 200, lr: float = 5e-3, lambda_L: float = 0.05,
    lambda_s: float = 1.0, lambda_o: float = 1.0, lambda_bias: float = 0.5,
):
    """Run §3.1 + §3.2 + §3.3 and return all intermediate state."""
    iso = isosurface_from_udf(V, F, n_v=n_v)
    proj = project_to_visual(
        iso.mesh, V, F, n_iters=n_iters, lr=lr, lambda_L=lambda_L,
    )
    guide = build_guide_graph(proj.mesh, voxel_size=iso.voxel_size)
    res = solve_layer_ilp(
        guide,
        lambda_s=lambda_s, lambda_o=lambda_o, lambda_bias=lambda_bias,
        time_limit=120.0,
    )
    return iso, proj, guide, res


# -------------------- tests --------------------

def test_closed_sphere_keeps_outer_shell():
    """The §3.3 bias prefers d_+=∞ → outer shell. After §3.3, ≥90% of outer
    iso-verts should be kept and ≥90% of inner iso-verts should be dropped.

    The "shell" of a vertex is determined BEFORE projection (when iso radii
    cluster around 1±voxel_size), since post-projection both shells overlap.
    """
    V, F = _icosphere(radius=1.0, subdivs=3)
    iso, proj, guide, res = _full_pipeline(V, F, n_v=24)

    radii_iso = np.linalg.norm(iso.mesh.V, axis=1)
    outer_mask = radii_iso > 1.0
    inner_mask = ~outer_mask
    assert outer_mask.any() and inner_mask.any(), "iso must have both shells"

    keep = res.labels == 1
    outer_kept = (keep & outer_mask).sum() / max(outer_mask.sum(), 1)
    inner_dropped = (~keep & inner_mask).sum() / max(inner_mask.sum(), 1)

    assert outer_kept > 0.90, (
        f"expected ≥90% of outer kept; got {100 * outer_kept:.1f}%"
    )
    assert inner_dropped > 0.90, (
        f"expected ≥90% of inner dropped; got {100 * inner_dropped:.1f}%"
    )


def test_closed_sphere_kept_count_is_half():
    """Total kept ≈ outer-shell vertex count (half the iso). Loose tolerance."""
    V, F = _icosphere(radius=1.0, subdivs=3)
    iso, proj, guide, res = _full_pipeline(V, F, n_v=24)
    keep_frac = res.n_keep / iso.mesh.n_verts
    assert 0.40 < keep_frac < 0.60, (
        f"kept {100 * keep_frac:.1f}% — expected ~50% (one of two shells)"
    )


def test_two_disjoint_spheres_yield_two_kept_components():
    """Two well-separated spheres → 4 iso shells → §3.3 keeps 2 (outer of each).

    We don't (yet) build a Mesh from the kept set — we just count connected
    components of the kept-vertex induced subgraph from M_proj's mesh edges.
    """
    Va, Fa = _icosphere(radius=0.5, subdivs=2, center=np.array([-1.5, 0, 0]))
    Vb, Fb = _icosphere(radius=0.5, subdivs=2, center=np.array([+1.5, 0, 0]))
    V = np.vstack([Va, Vb])
    F = np.vstack([Fa, Fb + Va.shape[0]])

    iso, proj, guide, res = _full_pipeline(V, F, n_v=24)
    keep = res.labels == 1
    kept_idx = set(np.flatnonzero(keep).tolist())

    # Component count on the induced kept-subgraph of mesh edges.
    parent = {i: i for i in kept_idx}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in proj.mesh.edges:
        a, b = int(a), int(b)
        if a in kept_idx and b in kept_idx:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    n_components = len({find(i) for i in kept_idx})
    assert n_components == 2, (
        f"expected 2 kept components (one per sphere); got {n_components}"
    )


def test_open_hemisphere_keeps_outer_shell():
    """Open hemisphere → M_iso is one closed wrap (both sides + thin rim).

    The §3.3 bias still picks the d_+=∞ side → ~half kept.
    """
    V, F = _hemisphere_open(radius=1.0, subdivs=3)
    iso, proj, guide, res = _full_pipeline(V, F, n_v=20)
    # M_iso is a single closed component for an open hemi (UDF wraps both sides).
    assert iso.mesh.is_closed
    keep_frac = res.n_keep / iso.mesh.n_verts
    assert 0.30 < keep_frac < 0.70, (
        f"kept {100 * keep_frac:.1f}% — expected ~50% (one wrap side)"
    )


def test_pipeline_determinism():
    """Same input → identical labels. No hidden RNG anywhere in §3.1–§3.3."""
    V, F = _icosphere(radius=1.0, subdivs=2)
    _, _, _, res1 = _full_pipeline(V, F, n_v=18)
    _, _, _, res2 = _full_pipeline(V, F, n_v=18)
    np.testing.assert_array_equal(res1.labels, res2.labels)
    assert abs(res1.energy - res2.energy) < 1e-9


def test_jacket_pipeline_topology():
    """End-to-end on the supplied jacket. Loose checks (extract.py + voronoi
    are still TODO so M_single will have spurious holes), but the kept set
    should be roughly half (one shell) and the ILP should report HiGHS-optimal.
    """
    from pag import load_obj
    V, F = load_obj("/Users/szhan/projects/proxy-asset-gen/data/jacket.obj")
    iso, proj, guide, res = _full_pipeline(V, F, n_v=32)
    assert "Optimal" in res.solver_status
    keep_frac = res.n_keep / iso.mesh.n_verts
    assert 0.30 < keep_frac < 0.70, (
        f"jacket kept {100 * keep_frac:.1f}% — expected ~50% (one shell)"
    )


# -------------------- generate_proxy_mesh — full end-to-end --------------------

def test_generate_proxy_mesh_sphere():
    """Full §3.1–§3.4 chain on a sphere → ~n_p verts, single component."""
    from pag import generate_proxy_mesh
    V, F = _icosphere(radius=1.0, subdivs=3)
    out = generate_proxy_mesh(V, F, n_v=24, n_p=128, keep_intermediates=False)
    assert 0.95 * 128 <= out.mesh.n_verts <= 1.05 * 128
    assert out.mesh.n_components() == 1
    # All required timings present.
    for k in ("udf", "projection", "guide_graph", "ilp", "extract", "voronoi", "total"):
        assert k in out.timings and out.timings[k] >= 0.0
    # voxel_size and knobs recorded.
    assert abs(out.voxel_size - 2.0 / 24) < 1e-9
    assert out.knobs["n_v"] == 24 and out.knobs["n_p"] == 128


def test_generate_proxy_mesh_keeps_intermediates():
    """keep_intermediates=True populates M_iso/M_proj/M_single, False clears them."""
    from pag import generate_proxy_mesh
    V, F = _icosphere(radius=1.0, subdivs=2)
    out_keep = generate_proxy_mesh(V, F, n_v=18, n_p=64, keep_intermediates=True)
    assert out_keep.M_iso is not None
    assert out_keep.M_proj is not None
    assert out_keep.M_single is not None
    out_drop = generate_proxy_mesh(V, F, n_v=18, n_p=64, keep_intermediates=False)
    assert out_drop.M_iso is None
    assert out_drop.M_proj is None
    assert out_drop.M_single is None


def test_generate_proxy_mesh_jacket_table_1():
    """Paper Table 1 acceptance check on jacket.obj — *loose* targets while
    extract.py hole-repair and §3.3 horn-handling are still TODO.

    Paper averages: components=1.0, boundary loops=2.0, |V|=128, runtime ~8.4s
    (i9-12900K, Mosek). What we currently get: |V|≈128, components in {1,2,3}
    (jacket horns sometimes detach from the main body when their tips lack
    opposite-edge anchors), boundary loops 5–15 (spurious holes from the
    jagged §3.3 cut). HiGHS is slower than Mosek so the runtime budget is
    looser.

    These bounds will tighten as horn-handling and hole-repair land.
    """
    from pag import generate_proxy_mesh, load_obj
    V, F = load_obj("/Users/szhan/projects/proxy-asset-gen/data/jacket.obj")
    out = generate_proxy_mesh(V, F, n_v=32, n_p=128)
    n_comp = out.mesh.n_components()
    assert n_comp <= 3, f"expected ≤3 components; got {n_comp}"
    # Plan §3.4 acceptance: |V| ∈ [115, 130]. pyacvd drops clusters during
    # the manifold-cleanup pass, so the lower bound is the practical floor.
    assert 115 <= out.mesh.n_verts <= 135, (
        f"|V_proxy|={out.mesh.n_verts} — expected [115, 135]"
    )
    n_loops = out.mesh.n_boundary_loops()
    assert n_loops <= 15, f"boundary loops={n_loops} — expected ≤15 pre-hole-repair"
    assert out.timings["total"] < 90.0, (
        f"runtime {out.timings['total']:.1f}s exceeds 90s budget"
    )
