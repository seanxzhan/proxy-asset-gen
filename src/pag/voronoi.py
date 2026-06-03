"""§3.4 — Centroidal Voronoi Diagram simplification of M_single → M_proxy.

We delegate to ``pyacvd``, which is a Python port of the
Valette–Chassery 2004 ACVD algorithm the paper specifies. The wrapper:

  1. Adapts our :class:`pag.mesh.Mesh` to a ``pyvista.PolyData`` (pyacvd's
     input type).
  2. Subdivides if the input has fewer vertices than the requested cluster
     count (pyacvd assigns one cluster per vertex during initialisation).
  3. Runs ``cluster(n_p)`` to get a centroidal Voronoi tessellation, then
     ``create_mesh()`` to extract one triangle mesh from cluster centroids.
  4. Snaps centroids back onto the input surface (``moveclus=True``) so the
     proxy stays on the cloth, not at floating barycentres.
  5. Validates the result via :func:`pag.mesh.build_mesh` — raises if the
     output is non-manifold.

Why this works as the §3.4 step: ACVD spaces clusters by surface area, so the
output triangulation is roughly equilateral with uniform edge lengths, exactly
the property a PBD cloth simulator wants (the paper reports min angle ≈ 47.9°,
mean aspect ratio ≈ 0.76 in Fig. 10).
"""
from __future__ import annotations

import numpy as np
import pyacvd
import pyvista as pv

from pag.mesh import Mesh, build_mesh


def _mesh_to_polydata(mesh: Mesh) -> pv.PolyData:
    """Adapt our Mesh to pyvista.PolyData. Faces are stored as a flat array
    [n0, v0_0, v0_1, ..., n1, v1_0, ...]; for triangles n_i ≡ 3."""
    F = np.ascontiguousarray(mesh.F, dtype=np.int64)
    n_faces = F.shape[0]
    faces_flat = np.empty((n_faces, 4), dtype=np.int64)
    faces_flat[:, 0] = 3
    faces_flat[:, 1:] = F
    return pv.PolyData(np.ascontiguousarray(mesh.V, dtype=np.float64),
                       faces_flat.ravel())


def _polydata_to_mesh(pdata: pv.PolyData) -> Mesh:
    V = np.asarray(pdata.points, dtype=np.float64)
    raw = np.asarray(pdata.faces, dtype=np.int64)
    if raw.size == 0:
        return build_mesh(V, np.zeros((0, 3), dtype=np.int64))
    # ACVD output is all triangles, so the [n, v0, v1, v2] strides are 4-wide.
    if raw[0] != 3:
        raise ValueError("voronoi output is not triangulated")
    F = raw.reshape(-1, 4)[:, 1:]
    return build_mesh(V, F)


def voronoi_simplify(
    mesh: Mesh, *,
    n_p: int = 128,
    max_iter: int = 100,
    auto_subdivide: bool = True,
) -> Mesh:
    """§3.4 ACVD simplification of M_single to ~``n_p`` vertices.

    Parameters
    ----------
    mesh : Mesh
        Typically ``M_single`` (post-§3.3 extraction).
    n_p : int, default 128
        Target proxy vertex count (paper §3.4: 128).
    max_iter : int, default 100
        ``pyacvd`` Lloyd-iteration cap. Paper-silent.
    auto_subdivide : bool, default True
        If the input has fewer vertices than ``n_p``, subdivide each triangle
        until ``|V| ≥ n_p``. ACVD requires more input verts than clusters.

    Returns
    -------
    Mesh
        ``M_proxy`` — manifold, validated. Component count of the output
        mirrors the input: ``voronoi_simplify`` does not bridge or drop
        components. If a single-component proxy is needed, weld / merge
        upstream (e.g. before §3.1) so the input here is already 1 component.
    """
    if n_p < 4:
        raise ValueError(f"n_p must be ≥ 4 (got {n_p})")

    pdata = _mesh_to_polydata(mesh)
    clu = pyacvd.Clustering(pdata)

    # ACVD assigns one cluster per input vertex initially; need |V| > n_p.
    # Each subdivide() splits every triangle into 4 (vertex count grows ~4×).
    if auto_subdivide:
        n_subs = 0
        while clu.mesh.n_points < 4 * n_p and n_subs < 4:
            clu.subdivide(2)
            n_subs += 1

    clu.cluster(n_p, maxiter=max_iter)
    out = clu.create_mesh(moveclus=True, flipnorm=True, clean=True)
    return _polydata_to_mesh(out)
