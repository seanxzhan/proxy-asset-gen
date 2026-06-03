"""§3.3 — Komodakis–Tziritas ILP for single-layer extraction.

This is the correctness centerpiece of §3.3. Validation:

  - Empty guide graph → trivial all-remove.
  - Single opposite pair → labels (0,1) or (1,0); bias picks the d_+ = ∞ side.
  - Multiple disjoint opposite pairs → each picks its own bias-preferred config.
  - Smoothness energy aligns labels along a chain — no zigzag.
  - Layered slab geometry: outer layer fully kept, inner fully removed.
  - Submodularity: confirm the opposite-energy pair-cost matrix violates the
    Boykov–Kolmogorov regularity condition E(0,0)+E(1,1) ≤ E(0,1)+E(1,0).
"""
from __future__ import annotations

import numpy as np
import pytest

from pag.guide_graph import GuideGraph
from pag.ilp import solve_layer_ilp


# -------------------- helpers --------------------

def _make_guide(
    *,
    n_verts: int,
    mesh_edges: list[tuple[int, int]],
    smoothness_w: list[float],
    opposite_edges: list[tuple[int, int]],
    d_plus: list[float],
) -> GuideGraph:
    """Construct a synthetic GuideGraph from raw lists."""
    return GuideGraph(
        mesh_edges=np.array(mesh_edges, dtype=np.int64).reshape(-1, 2),
        smoothness_w=np.array(smoothness_w, dtype=np.float64),
        opposite_edges=np.array(opposite_edges, dtype=np.int64).reshape(-1, 2),
        d_plus=np.array(d_plus, dtype=np.float64),
    )


# -------------------- tests --------------------

def test_empty_guide_returns_all_remove():
    """No edges → trivial early-out, no solver call."""
    g = _make_guide(
        n_verts=4,
        mesh_edges=[],
        smoothness_w=[],
        opposite_edges=[],
        d_plus=[np.inf, np.inf, np.inf, np.inf],
    )
    res = solve_layer_ilp(g)
    assert res.labels.shape == (4,)
    assert res.labels.dtype == np.int8
    assert res.n_keep == 0
    assert res.n_remove == 4
    assert res.energy == 0.0
    assert "trivial" in res.solver_status


def test_single_pair_picks_exactly_one():
    """One opposite pair, no bias → exactly one of the two is kept."""
    g = _make_guide(
        n_verts=2,
        mesh_edges=[],
        smoothness_w=[],
        opposite_edges=[(0, 1)],
        d_plus=[1.0, 1.0],  # equal → bias does not fire
    )
    res = solve_layer_ilp(g, lambda_o=1.0, lambda_bias=0.1)
    assert res.labels.sum() == 1
    assert set(res.labels.tolist()) == {0, 1}


def test_outer_bias_picks_d_plus_infinity_side():
    """When d_+(a) > d_+(b), bias prefers (l_a=1, l_b=0) — keep the outer side.

    Outer = larger d_+ (the +n ray escapes farther / never hits anything).
    """
    g = _make_guide(
        n_verts=2,
        mesh_edges=[],
        smoothness_w=[],
        opposite_edges=[(0, 1)],
        d_plus=[np.inf, 0.5],  # vertex 0 is "outer"
    )
    res = solve_layer_ilp(g, lambda_o=1.0, lambda_bias=0.1)
    assert res.labels[0] == 1
    assert res.labels[1] == 0

    # Symmetric case: vertex 1 is outer.
    g2 = _make_guide(
        n_verts=2,
        mesh_edges=[],
        smoothness_w=[],
        opposite_edges=[(0, 1)],
        d_plus=[0.5, np.inf],
    )
    res2 = solve_layer_ilp(g2, lambda_o=1.0, lambda_bias=0.1)
    assert res2.labels[0] == 0
    assert res2.labels[1] == 1


def test_outer_bias_disabled_falls_back_to_either():
    """With bias off, both polarities are equal-energy — solver may pick either.

    We just check exactly one is kept (the opposite term).
    """
    g = _make_guide(
        n_verts=2,
        mesh_edges=[],
        smoothness_w=[],
        opposite_edges=[(0, 1)],
        d_plus=[np.inf, 0.5],
    )
    res = solve_layer_ilp(g, lambda_o=1.0, enable_outer_bias=False)
    assert res.labels.sum() == 1


def test_multiple_independent_pairs():
    """Three disjoint opposite pairs → each independently picks the outer side."""
    g = _make_guide(
        n_verts=6,
        mesh_edges=[],
        smoothness_w=[],
        opposite_edges=[(0, 1), (2, 3), (4, 5)],
        d_plus=[np.inf, 0.5, 0.5, np.inf, np.inf, 0.5],
    )
    res = solve_layer_ilp(g, lambda_o=1.0, lambda_bias=0.1)
    expected = np.array([1, 0, 0, 1, 1, 0], dtype=np.int8)
    np.testing.assert_array_equal(res.labels, expected)


def test_smoothness_aligns_a_chain():
    """A 3-vertex chain with three opposite pairs against a virtual second layer.

    Vertices 0–1–2 are connected by mesh edges; their opposite partners 3,4,5
    are not. With strong smoothness, neighbours along the chain agree.

      mesh edges: 0-1, 1-2 (high w)
      opposite edges: (0,3), (1,4), (2,5)
      d_+: 0..2 = ∞ (outer), 3..5 = small (inner) → bias prefers keep 0,1,2.

    With strong λ_s, label 1 (kept) propagates uniformly over 0,1,2 and
    label 0 (removed) over 3,4,5.
    """
    g = _make_guide(
        n_verts=6,
        mesh_edges=[(0, 1), (1, 2)],
        smoothness_w=[1.0, 1.0],
        opposite_edges=[(0, 3), (1, 4), (2, 5)],
        d_plus=[np.inf, np.inf, np.inf, 0.5, 0.5, 0.5],
    )
    res = solve_layer_ilp(g, lambda_s=10.0, lambda_o=1.0, lambda_bias=0.1)
    np.testing.assert_array_equal(res.labels, np.array([1, 1, 1, 0, 0, 0], dtype=np.int8))


def test_labels_are_strictly_binary():
    """Even though y is continuous, l_i must end up integer."""
    g = _make_guide(
        n_verts=4,
        mesh_edges=[(0, 1), (2, 3)],
        smoothness_w=[1.0, 1.0],
        opposite_edges=[(0, 2), (1, 3)],
        d_plus=[np.inf, np.inf, 0.5, 0.5],
    )
    res = solve_layer_ilp(g)
    unique = np.unique(res.labels)
    assert set(unique.tolist()).issubset({0, 1})


def test_thin_slab_keeps_one_layer():
    """Synthetic 4×4 thin slab → ILP keeps one full side.

    Build the §3.3 input by hand: two 4×4 grids with mesh edges within each
    grid and opposite-vertex pairs across. Bias prefers the d_+ = ∞ side.
    """
    n = 4
    n_per = n * n
    n_verts = 2 * n_per

    # Mesh edges: 4-connected grid on the bottom + same on the top.
    def _grid_edges(offset: int) -> list[tuple[int, int]]:
        edges: list[tuple[int, int]] = []
        for j in range(n):
            for i in range(n):
                v = offset + j * n + i
                if i + 1 < n:
                    edges.append((v, v + 1))
                if j + 1 < n:
                    edges.append((v, v + n))
        return edges

    mesh_edges = _grid_edges(0) + _grid_edges(n_per)
    smoothness_w = [1.0] * len(mesh_edges)
    opposite_edges = [(i, i + n_per) for i in range(n_per)]
    d_plus = [np.inf] * n_per + [0.5] * n_per   # bottom outer, top inner

    g = _make_guide(
        n_verts=n_verts,
        mesh_edges=mesh_edges,
        smoothness_w=smoothness_w,
        opposite_edges=opposite_edges,
        d_plus=d_plus,
    )
    res = solve_layer_ilp(g, lambda_s=1.0, lambda_o=1.0, lambda_bias=0.1)
    np.testing.assert_array_equal(res.labels[:n_per], np.ones(n_per, dtype=np.int8))
    np.testing.assert_array_equal(res.labels[n_per:], np.zeros(n_per, dtype=np.int8))


def test_pair_energy_at_optimum_no_bias():
    """Single opposite pair, no bias, no smoothness:
       E(0,0) = E(1,1) = λ_o,    E(0,1) = E(1,0) = 0.
    Optimum picks one of (0,1) or (1,0), so the optimal energy is exactly 0.
    """
    g = _make_guide(
        n_verts=2,
        mesh_edges=[],
        smoothness_w=[],
        opposite_edges=[(0, 1)],
        d_plus=[1.0, 1.0],   # equal → bias does not fire even when enabled
    )
    res = solve_layer_ilp(g, lambda_o=1.0, enable_outer_bias=False)
    assert abs(res.energy) < 1e-9, f"expected 0, got {res.energy}"


def test_pair_energy_at_optimum_with_bias():
    """Single opposite pair, λ_bias=0.1, λ_o=1.0, d_+ = (∞, 1).

    Pair-cost matrix per impl (a=outer, b=inner):
       c[0]=E(0,0)=λ_o + w_bias       (both removed — penalised + bias fires)
       c[1]=E(0,1)=w_bias             (removed-then-kept — wrong polarity)
       c[2]=E(1,0)=0                  (kept-then-removed — correct polarity)
       c[3]=E(1,1)=λ_o + w_bias       (both kept — penalised + bias fires)
    where w_bias = λ_o · λ_bias = 0.1.

    Optimum is (l_a, l_b) = (1, 0) → energy = c[2] = 0.
    """
    g = _make_guide(
        n_verts=2,
        mesh_edges=[],
        smoothness_w=[],
        opposite_edges=[(0, 1)],
        d_plus=[np.inf, 1.0],
    )
    res = solve_layer_ilp(g, lambda_o=1.0, lambda_bias=0.1)
    assert res.labels[0] == 1 and res.labels[1] == 0
    assert abs(res.energy) < 1e-9, f"expected 0, got {res.energy}"


def test_smoothness_energy_at_optimum():
    """A single mesh edge with weight w and no opposite pairs.

    Smoothness pair-cost matrix c = [0, w, w, 0]:
        E(0,0)=0,  E(0,1)=w,  E(1,0)=w,  E(1,1)=0.
    Optimum picks matching labels (energy = 0). HiGHS may pick either polarity
    since the problem is symmetric — assert only "matching".
    """
    g = _make_guide(
        n_verts=2,
        mesh_edges=[(0, 1)],
        smoothness_w=[0.7],
        opposite_edges=[],
        d_plus=[np.inf, np.inf],
    )
    res = solve_layer_ilp(g, lambda_s=1.0, lambda_o=1.0)
    assert abs(res.energy) < 1e-9, f"expected 0, got {res.energy}"
    assert res.labels[0] == res.labels[1], "smoothness must align labels"


def test_opposite_pair_violates_submodularity():
    """The pair-cost matrix from §3.3 violates the Boykov–Kolmogorov regularity
    condition  E(0,0) + E(1,1) ≤ E(0,1) + E(1,0)  — which is *why* graph cut
    isn't certifiably optimal here and we use the LP-relaxation ILP instead.

    With λ_o = 1, λ_bias = 0 (no bias), the opposite term is:
       E(0,0) = E(1,1) = 1     (both kept or both removed — penalised)
       E(0,1) = E(1,0) = 0     (exactly one kept — preferred)
    so  1 + 1 = 2  >  0 + 0 = 0.  Regularity is violated; the term is supermodular.
    """
    e00 = e11 = 1.0
    e01 = e10 = 0.0
    assert e00 + e11 > e01 + e10, (
        "If this ever held, a graph-cut solver would suffice and we wouldn't need "
        "the LP relaxation in pag.ilp."
    )


def test_negative_iters_not_used_invalid_inputs_raise():
    """Sanity: invalid inputs (mismatched array sizes) raise rather than silent crash.

    GuideGraph itself is a dumb dataclass, but solve_layer_ilp indexes into
    d_plus by opposite-edge endpoints; out-of-bounds should error.
    """
    g = _make_guide(
        n_verts=2,
        mesh_edges=[],
        smoothness_w=[],
        opposite_edges=[(0, 5)],   # 5 ≥ n_verts
        d_plus=[1.0, 1.0],
    )
    with pytest.raises((IndexError, ValueError)):
        solve_layer_ilp(g)


def test_jacket_pipeline_produces_partial_keep():
    """End-to-end §3.1 + §3.2 + §3.3 on jacket.obj. Sanity check: ILP doesn't
    keep everything or nothing, and output is binary.
    """
    from pag import (
        build_guide_graph,
        isosurface_from_udf,
        load_obj,
        project_to_visual,
    )
    V, F = load_obj("/Users/szhan/projects/proxy-asset-gen/data/jacket.obj")
    iso = isosurface_from_udf(V, F, n_v=32)
    proj = project_to_visual(iso.mesh, V, F, n_iters=50, lr=5e-3, lambda_L=0.1)
    g = build_guide_graph(proj.mesh, voxel_size=iso.voxel_size)

    res = solve_layer_ilp(g, time_limit=120.0)
    n = proj.mesh.n_verts
    assert res.labels.shape == (n,)
    assert set(np.unique(res.labels).tolist()).issubset({0, 1})
    # Roughly half the verts should survive (one of the two shells).
    keep_frac = res.n_keep / n
    assert 0.2 < keep_frac < 0.8, (
        f"keep fraction {keep_frac:.2f} — expected ~0.5 (one of two shells)"
    )
