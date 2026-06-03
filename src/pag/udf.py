"""§3.1 — Isosurface from unsigned distance field.

Build a UDF on a voxel grid of spacing ``D / N_v`` (where ``D`` is the maximum
extent of the visual mesh's bounding box), then extract the ``D / N_v``-isosurface
with marching cubes. By construction the result is a watertight ``M_iso`` that
wraps a thin double cover around any open / non-manifold structure in the input.

Robust to:
  - non-manifold input (UDF doesn't care about edge multiplicity)
  - disconnected components (each gets its own wrap)
  - open surfaces (single sheet → double-covered shell)

Paper §3.1: "we extract the isosurface with iso-value at D/N_v, as it is a small
iso-value required to extract a watertight iso-surface from the UDF containing
M_visual."
"""
from __future__ import annotations

from dataclasses import dataclass

import igl
import numpy as np
from skimage import measure as skmeasure

from pag.mesh import Mesh, build_mesh


@dataclass
class IsoExtraction:
    """Container for the isosurface extraction result + grid metadata.

    The grid metadata is kept around because §3.2 / §3.3 need ``voxel_size``
    (i.e. ``D / N_v``) to define the opposite-pair distance threshold and the
    iso-value used during projection.
    """
    mesh: Mesh
    voxel_size: float       # D / N_v
    bbox_diag: float        # D = max(bbox extents)
    grid_origin: np.ndarray  # (3,) world coords of grid voxel (0,0,0)
    grid_shape: tuple[int, int, int]


def isosurface_from_udf(
    V_visual: np.ndarray,
    F_visual: np.ndarray,
    n_v: int = 32,
    pad_voxels: int = 3,
) -> IsoExtraction:
    """Extract ``M_iso`` from the UDF of a visual mesh.

    Parameters
    ----------
    V_visual : (N, 3) float
    F_visual : (M, 3) int
    n_v : int
        Voxelization resolution (paper §3.1 default: 32). The voxel size is
        ``D / n_v`` where ``D`` is the maximum side of the visual mesh's bbox.
    pad_voxels : int
        Number of voxels of padding around the bbox. Three is enough to fully
        contain an iso-surface placed one voxel out from the input.

    Returns
    -------
    IsoExtraction
    """
    V = np.ascontiguousarray(V_visual, dtype=np.float64)
    F = np.ascontiguousarray(F_visual, dtype=np.int64)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"V_visual must be (N, 3); got {V.shape}")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"F_visual must be (M, 3); got {F.shape}")

    bbox_min = V.min(axis=0)
    bbox_max = V.max(axis=0)
    extents = bbox_max - bbox_min
    bbox_diag = float(extents.max())
    if bbox_diag == 0.0:
        raise ValueError("Visual mesh is degenerate (zero bounding box).")

    voxel_size = bbox_diag / float(n_v)
    pad = pad_voxels * voxel_size

    grid_min = bbox_min - pad
    grid_max = bbox_max + pad

    nx = int(np.ceil((grid_max[0] - grid_min[0]) / voxel_size)) + 1
    ny = int(np.ceil((grid_max[1] - grid_min[1]) / voxel_size)) + 1
    nz = int(np.ceil((grid_max[2] - grid_min[2]) / voxel_size)) + 1

    xs = grid_min[0] + voxel_size * np.arange(nx)
    ys = grid_min[1] + voxel_size * np.arange(ny)
    zs = grid_min[2] + voxel_size * np.arange(nz)

    # P shape (nx*ny*nz, 3) — keep ('xy' indexing → axis order x, y, z).
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

    # libigl returns (S, I, C, N); UDF mode → S already non-negative.
    S, _, _, _ = igl.signed_distance(
        P, V, F, igl.SIGNED_DISTANCE_TYPE_UNSIGNED
    )
    udf = np.asarray(S, dtype=np.float64).reshape(nx, ny, nz)

    # paper §3.1 — iso-value is exactly one voxel size
    level = voxel_size
    if udf.min() > level or udf.max() < level:
        raise RuntimeError(
            f"UDF range [{udf.min():.4g}, {udf.max():.4g}] does not bracket "
            f"iso-level {level:.4g}; increase pad_voxels or check input."
        )

    verts, faces, _, _ = skmeasure.marching_cubes(
        udf, level=level, spacing=(voxel_size, voxel_size, voxel_size),
    )
    # skimage gives verts in grid-local frame [0, extent]; shift to world.
    verts = verts + grid_min[None, :]
    faces = faces.astype(np.int64, copy=False)

    # skimage may emit coincident vertices when the iso-surface clips through a
    # grid corner; weld at sub-voxel tolerance so downstream connectivity is sane.
    verts, faces = _weld_vertices(verts, faces, tol=voxel_size * 1e-4)

    mesh = build_mesh(verts, faces)
    return IsoExtraction(
        mesh=mesh,
        voxel_size=voxel_size,
        bbox_diag=bbox_diag,
        grid_origin=grid_min,
        grid_shape=(nx, ny, nz),
    )


def _weld_vertices(
    V: np.ndarray, F: np.ndarray, tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Merge vertices closer than ``tol`` and drop degenerate faces."""
    keys = np.round(V / tol).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    n_new = inv.max() + 1
    V_new = np.zeros((n_new, 3), dtype=np.float64)
    counts = np.zeros(n_new, dtype=np.int64)
    np.add.at(V_new, inv, V)
    np.add.at(counts, inv, 1)
    V_new /= counts[:, None]

    F_new = inv[F]
    # Drop triangles that collapsed to a degenerate edge/point.
    keep = (F_new[:, 0] != F_new[:, 1]) & (F_new[:, 1] != F_new[:, 2]) & (F_new[:, 0] != F_new[:, 2])
    return V_new, np.ascontiguousarray(F_new[keep], dtype=np.int64)
