"""§3.1 UDF + marching cubes → M_iso.

Validation is geometric:
  - sphere → M_iso radius ≈ r + voxel_size, ±5 % of voxel size, watertight
  - thin disk → M_iso has ≥ 2× input vertex count (double cover)
  - open hemisphere → M_iso is closed (UDF wraps both sides)
  - voxel_size and grid metadata are exposed for downstream stages
"""
from __future__ import annotations

import numpy as np
import pytest

from pag.udf import isosurface_from_udf


# -------------------- helpers --------------------

def _icosphere(radius: float = 1.0, subdivs: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A simple icosphere (build by recursive midpoint subdivision)."""
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

    def _mid(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key in cache:
            return cache[key]
        m = 0.5 * (V_list[a] + V_list[b])
        idx = len(V_list)
        V_list.append(m)
        cache[key] = idx
        return idx

    V_list = [v.copy() for v in V]
    for _ in range(subdivs):
        new_F = []
        for tri in F:
            a, b, c = tri
            ab = _mid(a, b); bc = _mid(b, c); ca = _mid(c, a)
            new_F.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        F = np.array(new_F, dtype=np.int64)

    Vf = np.array(V_list)
    Vf = radius * Vf / np.linalg.norm(Vf, axis=1, keepdims=True)
    return Vf, F


def _hemisphere_open(radius: float = 1.0, subdivs: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """An open hemisphere (z >= 0): drop all faces that have any vertex with z < 0."""
    V, F = _icosphere(radius, subdivs)
    keep_face = (V[F][:, :, 2] >= -1e-9).all(axis=1)
    F2 = F[keep_face]
    used = np.unique(F2.ravel())
    remap = -np.ones(V.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.size)
    return V[used], remap[F2]


def _flat_disk(radius: float = 1.0, n_ring: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """A z=0 disk fan-triangulated from the centre."""
    theta = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    rim = np.stack([radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)], axis=1)
    V = np.vstack([[[0.0, 0.0, 0.0]], rim])
    F = np.array([[0, 1 + i, 1 + (i + 1) % n_ring] for i in range(n_ring)], dtype=np.int64)
    return V, F


# -------------------- tests --------------------

def test_sphere_iso_is_closed_and_offset():
    """Closed sphere → UDF iso has *two* components: outer shell at r ≈ 1 + v,
    inner shell at r ≈ 1 − v. Both watertight, by construction. This is exactly
    why §3.3 needs the single-layer extraction step downstream."""
    V, F = _icosphere(radius=1.0, subdivs=3)
    iso = isosurface_from_udf(V, F, n_v=32)
    assert iso.mesh.is_closed, "M_iso must be watertight"
    assert iso.mesh.n_components() in (1, 2)

    radii = np.linalg.norm(iso.mesh.V, axis=1)
    outer = radii[radii > 1.0]
    inner = radii[radii < 1.0]
    assert outer.size > 0 and inner.size > 0, "expected both inner and outer shells"
    err_outer = np.abs(outer - (1.0 + iso.voxel_size)).mean() / iso.voxel_size
    err_inner = np.abs(inner - (1.0 - iso.voxel_size)).mean() / iso.voxel_size
    assert err_outer < 0.10, f"outer shell radial error {err_outer:.3f} voxel_sizes"
    assert err_inner < 0.10, f"inner shell radial error {err_inner:.3f} voxel_sizes"


def test_sphere_grid_metadata():
    V, F = _icosphere(radius=1.0, subdivs=2)
    iso = isosurface_from_udf(V, F, n_v=32)
    # bbox diag of unit sphere is 2.0
    assert abs(iso.bbox_diag - 2.0) < 1e-9
    assert abs(iso.voxel_size - 2.0 / 32) < 1e-12
    nx, ny, nz = iso.grid_shape
    # Grid must be at least N_v + 2*pad in each axis.
    for n in (nx, ny, nz):
        assert n >= 32 + 2 * 3


def test_thin_disk_double_covers():
    V, F = _flat_disk(radius=1.0, n_ring=64)
    iso = isosurface_from_udf(V, F, n_v=24)
    assert iso.mesh.is_closed, "open disk → closed M_iso (double cover)"
    # The wrap is approximately a flattened cylinder: top + bottom + thin rim.
    # Vertex count must comfortably exceed the input's: at minimum it doubles.
    assert iso.mesh.n_verts >= 2 * V.shape[0], (
        f"Expected ≥ {2*V.shape[0]} verts, got {iso.mesh.n_verts}"
    )


def test_open_hemisphere_iso_is_closed():
    V, F = _hemisphere_open(radius=1.0, subdivs=3)
    iso = isosurface_from_udf(V, F, n_v=24)
    assert iso.mesh.is_closed, "M_iso wraps even open inputs"


def test_increasing_n_v_increases_vert_count():
    V, F = _icosphere(radius=1.0, subdivs=2)
    iso_lo = isosurface_from_udf(V, F, n_v=16)
    iso_hi = isosurface_from_udf(V, F, n_v=48)
    assert iso_hi.mesh.n_verts > iso_lo.mesh.n_verts


def test_degenerate_input_raises():
    V = np.zeros((3, 3))
    F = np.array([[0, 1, 2]], dtype=np.int64)
    with pytest.raises(ValueError):
        isosurface_from_udf(V, F)


def test_jacket_obj_runs():
    """Sanity: the real visual mesh produces a closed iso with ≥ 1 component.

    Marker for §3.1 working end-to-end on the supplied input.
    """
    from pag import load_obj
    V, F = load_obj("/Users/szhan/projects/proxy-asset-gen/data/jacket.obj")
    iso = isosurface_from_udf(V, F, n_v=32)
    assert iso.mesh.is_closed
    # Jacket has at least one connected wrap (likely 1; test loose to start).
    assert iso.mesh.n_components() >= 1
    assert iso.mesh.n_verts > V.shape[0]  # iso is finer than input here
