"""§3.2 IPC + CCD collision-free projection (``collision_free=True``).

The paper's vector-Adam projection lets the two iso sheets collapse through each
other; that self-intersection corrupts the §3.3 guide graph and holes M_single.
The IPC solver guarantees the output is self-intersection-free. Validation:

  - on a sphere iso (two nested shells pulled to the same radius), vector Adam
    self-intersects but IPC does not
  - topology is preserved (projection only moves vertices)
  - the data fit still drops by orders of magnitude from the loose iso
  - the two sheets settle a positive gap apart (not collapsed)
  - the start must be intersection-free (guard raises otherwise)
  - the reshape(N,3) C-order convention the solver relies on matches ipctk
"""
from __future__ import annotations

import numpy as np
import pytest

ipctk = pytest.importorskip("ipctk")

from pag.mesh import build_mesh
from pag.projection import project_to_visual
from pag.udf import isosurface_from_udf


# -------------------- helpers --------------------

def _icosphere(radius: float = 1.0, subdivs: int = 2) -> tuple[np.ndarray, np.ndarray]:
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
    V_list = [v.copy() for v in V]
    cache: dict[tuple[int, int], int] = {}

    def _mid(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key in cache:
            return cache[key]
        V_list.append(0.5 * (V_list[a] + V_list[b]))
        cache[key] = len(V_list) - 1
        return cache[key]

    for _ in range(subdivs):
        new_F = []
        for a, b, c in F:
            ab, bc, ca = _mid(a, b), _mid(b, c), _mid(c, a)
            new_F.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        F = np.array(new_F, dtype=np.int64)
    Vf = np.array(V_list)
    return radius * Vf / np.linalg.norm(Vf, axis=1, keepdims=True), F


def _self_intersects(V: np.ndarray, F: np.ndarray, E: np.ndarray) -> bool:
    cmesh = ipctk.CollisionMesh(
        np.asfortranarray(np.ascontiguousarray(V, np.float64)),
        np.asfortranarray(E.astype(np.int32)),
        np.asfortranarray(F.astype(np.int32)),
    )
    return bool(ipctk.has_intersections(
        cmesh, np.asfortranarray(np.ascontiguousarray(V, np.float64))
    ))


# -------------------- tests --------------------

def test_ipc_resolves_self_intersection_that_adam_creates():
    """The load-bearing claim: vector Adam collapses the two shells into each
    other (self-intersecting), the IPC solver does not."""
    V, F = _icosphere(radius=1.0, subdivs=2)
    iso = isosurface_from_udf(V, F, n_v=16)
    # Topology is preserved by projection, so the iso mesh's own F/E describe
    # both projected results (the visual F/E are a different, coarser topology).
    iso_F, iso_E = iso.mesh.F, iso.mesh.edges

    adam = project_to_visual(iso.mesh, V, F, n_iters=200, lr=5e-3, lambda_L=0.05)
    ipc = project_to_visual(iso.mesh, V, F, n_iters=60, lambda_L=0.05,
                            collision_free=True)

    assert _self_intersects(adam.mesh.V, iso_F, iso_E), (
        "premise failed: vector Adam was expected to self-intersect here"
    )
    assert not _self_intersects(ipc.mesh.V, iso_F, iso_E), (
        "IPC projection must be self-intersection-free"
    )


def test_ipc_preserves_topology():
    V, F = _icosphere(radius=1.0, subdivs=2)
    iso = isosurface_from_udf(V, F, n_v=16)
    ipc = project_to_visual(iso.mesh, V, F, n_iters=40, lambda_L=0.05,
                            collision_free=True)
    assert ipc.mesh.n_verts == iso.mesh.n_verts
    assert ipc.mesh.n_faces == iso.mesh.n_faces
    assert ipc.mesh.n_edges == iso.mesh.n_edges
    np.testing.assert_array_equal(ipc.mesh.F, iso.mesh.F)


def test_ipc_data_fit_drops_orders_of_magnitude():
    """Even though the barrier holds the sheets ~dhat off the surface, the
    closest-point fit still drops by orders of magnitude from the loose iso."""
    V, F = _icosphere(radius=1.0, subdivs=2)
    iso = isosurface_from_udf(V, F, n_v=16)
    ipc = project_to_visual(iso.mesh, V, F, n_iters=60, lambda_L=0.05,
                            collision_free=True, record_history=True)
    h = ipc.history["data"]
    assert h[-1] < 1e-2 * h[0], f"data fit {h[0]:.3e} -> {h[-1]:.3e}"


def test_ipc_holds_sheets_on_opposite_sides():
    """The two shells should straddle the surface (one inside, one outside),
    separated by a positive gap — not collapsed onto a single radius."""
    V, F = _icosphere(radius=1.0, subdivs=2)
    iso = isosurface_from_udf(V, F, n_v=16)
    ipc = project_to_visual(iso.mesh, V, F, n_iters=60, lambda_L=0.05,
                            collision_free=True)
    r = np.linalg.norm(ipc.mesh.V, axis=1)
    inner, outer = r[r < 1.0], r[r > 1.0]
    assert inner.size > 0 and outer.size > 0, "shells collapsed to one side"
    gap = outer.mean() - inner.mean()
    assert gap > 0.5 * ipc.knobs["dhat"], (
        f"shells nearly collapsed: gap {gap:.4g}, dhat {ipc.knobs['dhat']:.4g}"
    )


def test_ipc_rejects_intersecting_start():
    """Two interpenetrating tetrahedra are edge-manifold but self-intersecting;
    the IPC solver needs an intersection-free start and must say so."""
    tet = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    tet_F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    V = np.vstack([tet, tet + 0.25])                     # second tet overlaps first
    F = np.vstack([tet_F, tet_F + 4])
    mesh = build_mesh(V, F)
    assert _self_intersects(V, F, mesh.edges), "premise: the two tets must overlap"
    with pytest.raises(ValueError, match="intersect"):
        project_to_visual(mesh, V, F, n_iters=5, collision_free=True)


def test_ipc_barrier_gradient_is_C_order():
    """Lock the convention the solver relies on: ipctk's barrier gradient
    reshapes to (N,3) in C-order (DOF = 3*i + c). A finite-difference check on a
    few DOFs guards against an ipctk upgrade silently changing the layout."""
    V, F = _icosphere(radius=1.0, subdivs=1)
    iso = isosurface_from_udf(V, F, n_v=12)
    Vv = np.ascontiguousarray(iso.mesh.V, np.float64)
    E_i = np.asfortranarray(iso.mesh.edges.astype(np.int32))
    F_i = np.asfortranarray(iso.mesh.F.astype(np.int32))
    n = Vv.shape[0]
    cmesh = ipctk.CollisionMesh(np.asfortranarray(Vv), E_i, F_i)
    diag = float(ipctk.world_bbox_diagonal_length(np.asfortranarray(Vv)))
    dhat = 5e-3 * diag
    barrier = ipctk.BarrierPotential(dhat)

    def energy(Vq):
        nc = ipctk.NormalCollisions()
        nc.build(cmesh, np.asfortranarray(Vq), dhat)
        return 0.0 if nc.empty() else float(barrier(nc, cmesh, np.asfortranarray(Vq)))

    nc = ipctk.NormalCollisions()
    nc.build(cmesh, np.asfortranarray(Vv), dhat)
    if nc.empty():
        pytest.skip("no active collisions on this iso; nothing to check")
    g = barrier.gradient(nc, cmesh, np.asfortranarray(Vv)).reshape(n, 3)

    rng = np.random.default_rng(0)
    eps = 1e-6
    for i in rng.integers(0, n, size=4):
        for c in range(3):
            Vp = Vv.copy(); Vp[i, c] += eps
            Vm = Vv.copy(); Vm[i, c] -= eps
            fd = (energy(Vp) - energy(Vm)) / (2 * eps)
            assert abs(fd - g[i, c]) < 1e-3 * (1 + abs(g[i, c])), (
                f"gradient layout mismatch at ({i},{c}): fd={fd:.4e} g={g[i, c]:.4e}"
            )
