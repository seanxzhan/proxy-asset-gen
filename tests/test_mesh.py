"""Mesh topology + connectivity utilities."""
from __future__ import annotations

import numpy as np
import pytest

from pag import Mesh, NonManifoldError, build_mesh


# -------------------- helpers --------------------

def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    F = np.array([
        [0, 2, 1],
        [0, 1, 3],
        [0, 3, 2],
        [1, 2, 3],
    ])
    return V, F


def _open_quad() -> tuple[np.ndarray, np.ndarray]:
    """Two triangles sharing one edge — single open patch."""
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    F = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ])
    return V, F


def _two_disconnected_triangles() -> tuple[np.ndarray, np.ndarray]:
    V = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [5.0, 1.0, 0.0],
    ])
    F = np.array([[0, 1, 2], [3, 4, 5]])
    return V, F


# -------------------- closed mesh basics --------------------

def test_tetrahedron_topology():
    V, F = _tetrahedron()
    m = build_mesh(V, F)
    assert m.n_verts == 4
    assert m.n_faces == 4
    assert m.n_edges == 6
    assert m.is_closed
    assert not m.boundary_mask.any()
    assert m.n_boundary_loops() == 0
    assert m.n_components() == 1


def test_tetrahedron_face_areas_sum():
    V, F = _tetrahedron()
    m = build_mesh(V, F)
    # Three legs are right triangles area 0.5; the slanted face has area sqrt(3)/2.
    expected = np.array([0.5, 0.5, 0.5, np.sqrt(3.0) / 2.0])
    np.testing.assert_allclose(np.sort(m.face_areas()), np.sort(expected), atol=1e-12)


def test_face_normals_outward_on_tetra():
    V, F = _tetrahedron()
    m = build_mesh(V, F)
    fn = m.face_normals()
    centroid = V.mean(axis=0)
    face_centroids = V[F].mean(axis=1)
    # All normals should point away from the centroid (outward orientation).
    assert np.all(np.einsum("ij,ij->i", fn, face_centroids - centroid) > 0)


def test_vertex_normals_unit_length():
    V, F = _tetrahedron()
    m = build_mesh(V, F)
    vn = m.vertex_normals()
    np.testing.assert_allclose(np.linalg.norm(vn, axis=1), 1.0, atol=1e-12)


# -------------------- boundary detection --------------------

def test_open_quad_boundary():
    V, F = _open_quad()
    m = build_mesh(V, F)
    assert not m.is_closed
    assert int(m.boundary_mask.sum()) == 4   # outer ring
    assert m.n_boundary_loops() == 1
    assert m.n_components() == 1


def test_disconnected_triangles_components():
    V, F = _two_disconnected_triangles()
    m = build_mesh(V, F)
    assert m.n_components() == 2
    assert m.n_boundary_loops() == 2
    assert int(m.boundary_mask.sum()) == 6


# -------------------- guardrails --------------------

def test_non_manifold_raises():
    """Three triangles sharing the same edge → edge has 3 incident faces."""
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    F = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 1, 4],
    ])
    with pytest.raises(NonManifoldError):
        build_mesh(V, F)


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        build_mesh(np.zeros((3, 2)), np.zeros((1, 3), dtype=np.int64))
    with pytest.raises(ValueError):
        build_mesh(np.zeros((3, 3)), np.zeros((1, 4), dtype=np.int64))


# -------------------- repeated build is deterministic --------------------

def test_build_is_deterministic():
    V, F = _tetrahedron()
    m1 = build_mesh(V, F)
    m2 = build_mesh(V, F)
    np.testing.assert_array_equal(m1.edges, m2.edges)
    np.testing.assert_array_equal(m1.boundary_mask, m2.boundary_mask)


# -------------------- Mesh dataclass should not mutate inputs --------------------

def test_build_does_not_mutate_inputs():
    V, F = _tetrahedron()
    V0, F0 = V.copy(), F.copy()
    _ = build_mesh(V, F)
    np.testing.assert_array_equal(V, V0)
    np.testing.assert_array_equal(F, F0)
