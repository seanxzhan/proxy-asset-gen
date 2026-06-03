"""§3.4 — voronoi_simplify (ACVD via pyacvd).

Validation:
  - output vertex count ≈ n_p (within ±5% — pyacvd may merge isolated clusters)
  - output is edge-manifold (validated by build_mesh on the way out)
  - sphere → output approximates the sphere (mean radius ≈ 1.0)
  - input not mutated
  - n_p < 4 raises ValueError
  - small input → auto-subdivide kicks in
  - paper §3.4 quality cues: min face angle > 20°, mean aspect ratio > 0.6
"""
from __future__ import annotations

import numpy as np
import pytest

from pag.mesh import build_mesh
from pag.voronoi import voronoi_simplify


# -------------------- helpers --------------------

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


def _min_face_angle_deg(V: np.ndarray, F: np.ndarray) -> float:
    """Smallest interior angle across all triangles (degrees)."""
    a = V[F[:, 0]]; b = V[F[:, 1]]; c = V[F[:, 2]]

    def angle(u, v):
        cu = np.einsum("ij,ij->i", u, v)
        nu = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        return np.degrees(np.arccos(np.clip(cu / np.maximum(nu, 1e-30), -1.0, 1.0)))

    angles = np.stack([angle(b - a, c - a), angle(a - b, c - b), angle(a - c, b - c)], axis=1)
    return float(angles.min())


def _mean_aspect_ratio(V: np.ndarray, F: np.ndarray) -> float:
    """Inradius/circumradius ratio averaged over faces. 1.0 = equilateral."""
    a = V[F[:, 0]]; b = V[F[:, 1]]; c = V[F[:, 2]]
    ea = np.linalg.norm(b - c, axis=1)
    eb = np.linalg.norm(a - c, axis=1)
    ec = np.linalg.norm(a - b, axis=1)
    s = 0.5 * (ea + eb + ec)
    area = np.sqrt(np.maximum(s * (s - ea) * (s - eb) * (s - ec), 0.0))
    inradius = area / np.maximum(s, 1e-30)
    circ = (ea * eb * ec) / np.maximum(4.0 * area, 1e-30)
    return float((2.0 * inradius / np.maximum(circ, 1e-30)).mean())


# -------------------- tests --------------------

def test_sphere_output_count_close_to_n_p():
    V, F = _icosphere(radius=1.0, subdivs=3)
    mesh = build_mesh(V, F)
    out = voronoi_simplify(mesh, n_p=128)
    # Within ±5% — pyacvd may drop a few isolated clusters.
    assert 0.95 * 128 <= out.n_verts <= 1.05 * 128


def test_sphere_output_lies_on_input_surface():
    """moveclus=True snaps cluster centroids to the closest input surface point.
    Verify by bounding the radius spread on a unit sphere.
    """
    V, F = _icosphere(radius=1.0, subdivs=3)
    mesh = build_mesh(V, F)
    out = voronoi_simplify(mesh, n_p=128)
    radii = np.linalg.norm(out.V, axis=1)
    assert abs(radii.mean() - 1.0) < 0.01
    assert radii.std() < 0.02


def test_sphere_output_is_manifold_and_closed():
    V, F = _icosphere(radius=1.0, subdivs=3)
    mesh = build_mesh(V, F)
    out = voronoi_simplify(mesh, n_p=128)
    # build_mesh would have raised on >2 faces per edge — manifold by construction.
    assert out.is_closed, "ACVD'ing a closed input must yield a closed output"
    assert out.n_components() == 1


def test_paper_quality_cues_on_sphere():
    """Paper Fig. 10: min face angle ≈ 47.9° ± 6.6, mean aspect ≈ 0.76 ± 0.13.
    Loose check that we're in the ballpark — tight enough to catch a regression
    where ACVD output goes degenerate (slivers, near-zero-area triangles).
    """
    V, F = _icosphere(radius=1.0, subdivs=3)
    mesh = build_mesh(V, F)
    out = voronoi_simplify(mesh, n_p=128)
    assert _min_face_angle_deg(out.V, out.F) > 20.0
    assert _mean_aspect_ratio(out.V, out.F) > 0.6


def test_input_not_mutated():
    V, F = _icosphere(radius=1.0, subdivs=2)
    mesh = build_mesh(V, F)
    V_orig = mesh.V.copy()
    F_orig = mesh.F.copy()
    _ = voronoi_simplify(mesh, n_p=64)
    np.testing.assert_array_equal(mesh.V, V_orig)
    np.testing.assert_array_equal(mesh.F, F_orig)


def test_n_p_below_minimum_raises():
    V, F = _icosphere(radius=1.0, subdivs=2)
    mesh = build_mesh(V, F)
    with pytest.raises(ValueError):
        voronoi_simplify(mesh, n_p=3)


def test_auto_subdivide_handles_small_input():
    """Tiny input + n_p > |V| → auto-subdivide must kick in or pyacvd fails."""
    V, F = _icosphere(radius=1.0, subdivs=1)   # 42 verts
    mesh = build_mesh(V, F)
    out = voronoi_simplify(mesh, n_p=128)
    assert 0.90 * 128 <= out.n_verts <= 1.10 * 128
