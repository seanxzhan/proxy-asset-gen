"""§3.3 helper — per-vertex |κ|.

Validation:
  - sphere of radius r → |κ| ≈ 1/r everywhere (both principal curvatures equal 1/r)
  - flat plane → |κ| ≈ 0 (zero principal curvatures)
  - shape matches mesh.V
"""
from __future__ import annotations

import numpy as np

from pag.curvature import vertex_abs_curvature
from pag.mesh import build_mesh


def _icosphere(radius: float = 1.0, subdivs: int = 3) -> tuple[np.ndarray, np.ndarray]:
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


def _flat_grid(n: int = 16, side: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """A z=0 grid; principal curvatures are zero."""
    xs = np.linspace(-side / 2, side / 2, n)
    ys = np.linspace(-side / 2, side / 2, n)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    V = np.stack([xx.ravel(), yy.ravel(), np.zeros(n * n)], axis=1)
    F = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b = a + 1
            c = a + n
            d = c + 1
            F.append([a, b, d])
            F.append([a, d, c])
    return V, np.array(F, dtype=np.int64)


def test_sphere_kappa_is_inverse_radius():
    R = 2.5
    V, F = _icosphere(radius=R, subdivs=3)
    mesh = build_mesh(V, F)
    abs_k = vertex_abs_curvature(mesh, ring_radius=3)
    assert abs_k.shape == (V.shape[0],)
    # Drop libigl's NaN/inf on poorly-conditioned verts (rare); rest should be ≈ 1/R.
    finite = np.isfinite(abs_k)
    assert finite.sum() > 0.95 * V.shape[0]
    err = np.abs(abs_k[finite] - 1.0 / R) / (1.0 / R)
    assert err.mean() < 0.10, f"mean rel err {err.mean():.3f} (expected ≪ 1)"


def test_flat_grid_kappa_near_zero():
    V, F = _flat_grid(n=12, side=1.0)
    mesh = build_mesh(V, F)
    abs_k = vertex_abs_curvature(mesh, ring_radius=2)
    finite = np.isfinite(abs_k)
    # Flat plane: any non-zero κ is numerical jitter relative to a grid that's
    # been planar-fit to within machine precision.
    assert np.median(abs_k[finite]) < 1e-3


def test_output_shape_matches_n_verts():
    V, F = _icosphere(radius=1.0, subdivs=2)
    mesh = build_mesh(V, F)
    abs_k = vertex_abs_curvature(mesh)
    assert abs_k.shape == (mesh.n_verts,)
    assert abs_k.dtype == np.float64
