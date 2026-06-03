"""§3.3 — extract_single_layer (apply ILP labels → M_single).

Validation:
  - drops face if any vertex has l = 0
  - compacts vertex indices (no dangling, no gaps)
  - vertex positions preserved bit-exact (we only filter; nothing moves)
  - shape mismatch raises ValueError
  - all-drop → empty Mesh (no crash)
  - input is non-mutated
"""
from __future__ import annotations

import numpy as np
import pytest

from pag.extract import extract_single_layer
from pag.mesh import build_mesh


def _square(n: int) -> tuple[np.ndarray, np.ndarray]:
    """A z=0 n×n grid, fan-triangulated, manifold."""
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    V = np.stack([xx.ravel(), yy.ravel(), np.zeros(n * n)], axis=1)
    F = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i; b = a + 1; c = a + n; d = c + 1
            F.append([a, b, d])
            F.append([a, d, c])
    return V, np.array(F, dtype=np.int64)


def test_keep_all_returns_identical_topology():
    V, F = _square(4)
    mesh = build_mesh(V, F)
    labels = np.ones(mesh.n_verts, dtype=np.int8)
    out = extract_single_layer(mesh, labels)
    assert out.n_verts == mesh.n_verts
    assert out.n_faces == mesh.n_faces
    np.testing.assert_array_equal(out.V, mesh.V)
    np.testing.assert_array_equal(out.F, mesh.F)


def test_drop_all_returns_empty():
    V, F = _square(4)
    mesh = build_mesh(V, F)
    labels = np.zeros(mesh.n_verts, dtype=np.int8)
    out = extract_single_layer(mesh, labels)
    assert out.n_verts == 0
    assert out.n_faces == 0


def test_face_dropped_when_any_vertex_dropped():
    """Drop one corner vertex → all faces incident to it disappear."""
    V, F = _square(3)   # 9 verts, 8 faces
    mesh = build_mesh(V, F)
    labels = np.ones(mesh.n_verts, dtype=np.int8)
    labels[4] = 0   # interior vertex of a 3x3 grid → many incident faces
    n_incident = int((F == 4).any(axis=1).sum())
    out = extract_single_layer(mesh, labels)
    assert out.n_faces == mesh.n_faces - n_incident


def test_vertex_indices_compacted():
    """Surviving faces must reference only [0, n_kept) — no gaps."""
    V, F = _square(4)
    mesh = build_mesh(V, F)
    labels = np.zeros(mesh.n_verts, dtype=np.int8)
    labels[[0, 1, 4, 5]] = 1   # one corner quad's 4 verts
    out = extract_single_layer(mesh, labels)
    assert out.n_verts > 0
    assert out.F.min() == 0
    assert out.F.max() == out.n_verts - 1


def test_kept_vertex_positions_preserved():
    """Filtering doesn't move anything — kept verts have bit-exact positions."""
    V, F = _square(4)
    mesh = build_mesh(V, F)
    labels = np.ones(mesh.n_verts, dtype=np.int8)
    labels[[3, 7, 11, 15]] = 0   # drop the rightmost column
    out = extract_single_layer(mesh, labels)
    # The first row of the grid (x=0) should appear unchanged in V_out.
    kept_idx = np.flatnonzero(labels)
    np.testing.assert_array_equal(out.V, mesh.V[kept_idx])


def test_shape_mismatch_raises():
    V, F = _square(3)
    mesh = build_mesh(V, F)
    with pytest.raises(ValueError):
        extract_single_layer(mesh, np.ones(mesh.n_verts + 1, dtype=np.int8))


def test_input_not_mutated():
    V, F = _square(3)
    mesh = build_mesh(V, F)
    V_orig = mesh.V.copy()
    F_orig = mesh.F.copy()
    labels = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=np.int8)
    _ = extract_single_layer(mesh, labels)
    np.testing.assert_array_equal(mesh.V, V_orig)
    np.testing.assert_array_equal(mesh.F, F_orig)


def test_isolated_kept_vertex_dropped():
    """A vertex labelled keep but with no kept-face neighbours → not in output."""
    V, F = _square(3)
    mesh = build_mesh(V, F)
    # Keep only the center vertex (index 4); drop all corners. No face is
    # all-kept, so the center should not appear in M_single either.
    labels = np.zeros(mesh.n_verts, dtype=np.int8)
    labels[4] = 1
    out = extract_single_layer(mesh, labels)
    assert out.n_verts == 0
    assert out.n_faces == 0
