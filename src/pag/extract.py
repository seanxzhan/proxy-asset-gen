"""§3.3 — apply ILP labels: M_proj → M_single.

Drop every face that has at least one ``l = 0`` vertex (the cut runs along edges
between kept and dropped verts). Compact the surviving vertex indices to
``[0, N_kept)``.

Holes from a jagged §3.3 cut are not patched here — §3.4 Voronoi simplification
resamples to ~128 verts (≫ voxel size), which collapses sub-voxel topology
noise as a side-effect. If you need a clean, hole-repaired M_single (for
inspection or as input to a non-§3.4 step), call this and patch boundary loops
shorter than some threshold separately.

The paper also calls for "remove non-manifold faces / split non-manifold
vertices". We delegate edge-manifold checks to ``build_mesh`` (which raises on
>2 faces per edge) and accept that vertex-non-manifold cases (a single kept
vertex shared by two disjoint kept fans) survive — pyacvd handles them in §3.4.
"""
from __future__ import annotations

import numpy as np

from pag.mesh import Mesh, build_mesh


def extract_single_layer(mesh: Mesh, labels: np.ndarray) -> Mesh:
    """Drop labelled-out vertices and faces.

    Parameters
    ----------
    mesh : Mesh
        Typically ``M_proj`` (post-§3.2).
    labels : (N,) int / bool
        Per-vertex keep mask from :func:`pag.ilp.solve_layer_ilp`. ``1`` = keep.

    Returns
    -------
    Mesh
        ``M_single`` — the surviving sub-mesh, validated edge-manifold.

    Raises
    ------
    ValueError
        If ``labels.shape != (mesh.n_verts,)``.
    pag.mesh.NonManifoldError
        If the cut leaves an edge shared by >2 faces (shouldn't happen for a
        manifold M_proj input but we surface it rather than silently corrupt).
    """
    labels = np.asarray(labels)
    if labels.shape != (mesh.n_verts,):
        raise ValueError(
            f"labels shape {labels.shape} != (n_verts={mesh.n_verts},)"
        )
    keep = labels.astype(bool)
    keep_face = keep[mesh.F].all(axis=1)
    F_kept = mesh.F[keep_face]

    # Compact vertex indexing — only keep verts referenced by surviving faces.
    used = np.unique(F_kept.ravel())
    if used.size == 0:
        # Pathological: ILP dropped everything. Return an empty mesh-shaped Mesh.
        return build_mesh(
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int64),
        )
    remap = -np.ones(mesh.V.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.size)
    return build_mesh(mesh.V[used], remap[F_kept])
