"""§3.2 vector-Adam projection: M_iso → M_proj.

Validation:
  - sphere → mean radial error after projection ≪ voxel_size (loose iso → tight)
  - vertex / face / edge counts unchanged (projection only moves vertices)
  - data energy decreases (post Adam warmup): final ≪ initial
  - λ_L = 0 fits visual mesh tighter than λ_L > 0 (regularisation trade-off)
  - jacket sanity: a few iters drop closest-point distance below the input voxel
"""
from __future__ import annotations

import numpy as np
import pytest

from pag.projection import project_to_visual
from pag.udf import isosurface_from_udf


# -------------------- helpers --------------------

def _icosphere(radius: float = 1.0, subdivs: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Match test_udf._icosphere — a recursively subdivided icosahedron."""
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
    return radius * Vf / np.linalg.norm(Vf, axis=1, keepdims=True), F


# -------------------- tests --------------------

def test_sphere_projection_pulls_to_input():
    """The iso starts ~one voxel off the sphere; projection should snap it tight.

    Both the inner and outer shells get pulled to r=1, since closest-point is
    side-agnostic.
    """
    V, F = _icosphere(radius=1.0, subdivs=3)
    iso = isosurface_from_udf(V, F, n_v=24)
    res = project_to_visual(
        iso.mesh, V, F, n_iters=200, lr=5e-3, lambda_L=0.05,
    )
    err = np.abs(np.linalg.norm(res.mesh.V, axis=1) - 1.0)
    assert err.mean() < iso.voxel_size / 5.0, (
        f"mean radial error {err.mean():.4g} (voxel={iso.voxel_size:.4g})"
    )


def test_projection_preserves_topology():
    V, F = _icosphere(radius=1.0, subdivs=2)
    iso = isosurface_from_udf(V, F, n_v=16)
    res = project_to_visual(iso.mesh, V, F, n_iters=10)
    assert res.mesh.n_verts == iso.mesh.n_verts
    assert res.mesh.n_faces == iso.mesh.n_faces
    assert res.mesh.n_edges == iso.mesh.n_edges
    np.testing.assert_array_equal(res.mesh.F, iso.mesh.F)


def test_data_energy_decreases():
    """Adam isn't strictly monotonic, but the last quarter must be ≪ first quarter."""
    V, F = _icosphere(radius=1.0, subdivs=3)
    iso = isosurface_from_udf(V, F, n_v=20)
    res = project_to_visual(
        iso.mesh, V, F, n_iters=120, lr=5e-3, lambda_L=0.05,
        record_history=True,
    )
    h = res.history["data"]
    n = h.size
    early = h[:n // 4].mean()
    late = h[3 * n // 4:].mean()
    assert late < 0.1 * early, f"data energy: early {early:.4g} → late {late:.4g}"


def test_lambda_L_zero_fits_tighter_than_smooth():
    V, F = _icosphere(radius=1.0, subdivs=3)
    iso = isosurface_from_udf(V, F, n_v=20)
    res_none = project_to_visual(
        iso.mesh, V, F, n_iters=200, lr=5e-3, lambda_L=0.0, record_history=True,
    )
    res_smooth = project_to_visual(
        iso.mesh, V, F, n_iters=200, lr=5e-3, lambda_L=0.5, record_history=True,
    )
    # Without smoothness regularisation, the data term gets to zero faster.
    assert res_none.history["data"][-1] < res_smooth.history["data"][-1]


def test_record_history_off_by_default():
    V, F = _icosphere(radius=1.0, subdivs=2)
    iso = isosurface_from_udf(V, F, n_v=16)
    res = project_to_visual(iso.mesh, V, F, n_iters=5)
    assert res.history == {}


def test_zero_iters_returns_input():
    V, F = _icosphere(radius=1.0, subdivs=2)
    iso = isosurface_from_udf(V, F, n_v=16)
    res = project_to_visual(iso.mesh, V, F, n_iters=0)
    np.testing.assert_array_equal(res.mesh.V, iso.mesh.V)


def test_negative_iters_raises():
    V, F = _icosphere(radius=1.0, subdivs=2)
    iso = isosurface_from_udf(V, F, n_v=16)
    with pytest.raises(ValueError):
        project_to_visual(iso.mesh, V, F, n_iters=-1)


def test_jacket_obj_runs():
    """End-to-end §3.1 + §3.2 on the supplied visual mesh."""
    from pag import load_obj
    V, F = load_obj("/Users/szhan/projects/proxy-asset-gen/data/jacket.obj")
    iso = isosurface_from_udf(V, F, n_v=32)
    res = project_to_visual(
        iso.mesh, V, F, n_iters=50, lr=5e-3, lambda_L=0.1, record_history=True,
    )
    assert res.mesh.n_verts == iso.mesh.n_verts
    # Closest-point energy must drop substantially over a real visual mesh.
    assert res.history["data"][-1] < 0.1 * res.history["data"][0]
