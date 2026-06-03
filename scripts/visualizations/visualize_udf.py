"""Visualize the §3.1 UDF and extracted iso-surface(s).

Loads a visual mesh, runs ``isosurface_from_udf``, and shows in polyscope:

  - the input visual mesh (``M_visual``)
  - the UDF, as a 3D scalar field on the same voxel grid the iso came off
  - ``M_iso``, split by connected component (each shell its own colour) — a
    closed visual mesh produces *two* shells (inner + outer), which is the
    motivation for §3.3 single-layer extraction.

The volume grid is sampled at the same xs/ys/zs as ``isosurface_from_udf`` so
the iso-surface lines up exactly with the level set the user can scrub on the
volume grid's slice plane.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pag import isosurface_from_udf, load_obj


def _component_labels(n_verts: int, edges: np.ndarray) -> np.ndarray:
    """Per-vertex connected-component ID via union-find on edges. (N,) int."""
    parent = np.arange(n_verts, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb

    roots = np.array([find(i) for i in range(n_verts)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)


def _evaluate_udf_on_grid(
    V: np.ndarray, F: np.ndarray, grid_origin: np.ndarray,
    voxel_size: float, grid_shape: tuple[int, int, int],
) -> np.ndarray:
    """Re-evaluate the UDF on the same grid the iso was extracted from. (nx,ny,nz)."""
    import igl
    nx, ny, nz = grid_shape
    xs = grid_origin[0] + voxel_size * np.arange(nx)
    ys = grid_origin[1] + voxel_size * np.arange(ny)
    zs = grid_origin[2] + voxel_size * np.arange(nz)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    S, _, _, _ = igl.signed_distance(
        np.ascontiguousarray(P, dtype=np.float64),
        np.ascontiguousarray(V, dtype=np.float64),
        np.ascontiguousarray(F, dtype=np.int64),
        igl.SIGNED_DISTANCE_TYPE_UNSIGNED,
    )
    return np.asarray(S, dtype=np.float64).reshape(nx, ny, nz)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input", type=str,
        default="/Users/szhan/projects/proxy-asset-gen/data/jacket.obj",
        help="Path to visual mesh OBJ.",
    )
    ap.add_argument("--n-v", type=int, default=32,
                    help="Voxel resolution along the longest bbox axis (paper §3.1: 32).")
    ap.add_argument("--pad", type=int, default=3,
                    help="Padding voxels around the bbox.")
    ap.add_argument("--smoke", action="store_true",
                    help="Run headless and print stats; skip polyscope window.")
    args = ap.parse_args()

    path = Path(args.input)
    print(f"loading {path} ...")
    V_visual, F_visual = load_obj(path)
    print(f"  |V|={V_visual.shape[0]}  |F|={F_visual.shape[0]}")

    t0 = time.time()
    iso = isosurface_from_udf(V_visual, F_visual, n_v=args.n_v, pad_voxels=args.pad)
    t_iso = time.time() - t0

    labels = _component_labels(iso.mesh.n_verts, iso.mesh.edges)
    n_comp = int(labels.max()) + 1 if labels.size else 0
    print(f"isosurface_from_udf: {t_iso:.3f}s")
    print(f"  voxel_size={iso.voxel_size:.5f}  bbox_diag={iso.bbox_diag:.5f}")
    print(f"  grid_shape={iso.grid_shape}  ({np.prod(iso.grid_shape):,} voxels)")
    print(f"  M_iso: |V|={iso.mesh.n_verts}  |F|={iso.mesh.n_faces}  "
          f"closed={iso.mesh.is_closed}  components={n_comp}")

    if args.smoke:
        return

    t0 = time.time()
    udf = _evaluate_udf_on_grid(
        V_visual, F_visual, iso.grid_origin, iso.voxel_size, iso.grid_shape,
    )
    print(f"UDF re-eval for viz: {time.time() - t0:.3f}s")

    import polyscope as ps
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_front_dir("z_front")
    ps.set_ground_plane_mode("none")

    visual = ps.register_surface_mesh(
        "M_visual", V_visual, F_visual,
        color=(0.65, 0.65, 0.7), transparency=0.35,
    )
    visual.set_smooth_shade(True)

    iso_ps = ps.register_surface_mesh("M_iso", iso.mesh.V, iso.mesh.F)
    iso_ps.set_smooth_shade(True)
    if n_comp > 1:
        iso_ps.add_scalar_quantity(
            "component", labels.astype(np.float64),
            defined_on="vertices", cmap="turbo", enabled=True,
        )

    # Volume grid sampled at the same xs/ys/zs as the marching-cubes input.
    # Polyscope's volume grid takes (origin, upper_corner, dims) — upper_corner
    # is the world-space position of the *last* sample, i.e. origin + (dim-1)*vs.
    nx, ny, nz = iso.grid_shape
    upper = iso.grid_origin + iso.voxel_size * (np.array([nx, ny, nz]) - 1)
    grid = ps.register_volume_grid(
        "UDF", (nx, ny, nz),
        bound_low=tuple(iso.grid_origin),
        bound_high=tuple(upper),
    )
    # skimage marching_cubes uses (i, j, k) → (x, y, z) with 'ij' meshgrid; the
    # UDF array we built is in the same axis order, so it can be passed straight
    # in (polyscope's volume grid expects shape (nx, ny, nz) with x fastest? —
    # we hand it the array as-is and trust the iso lining up to confirm).
    grid.add_scalar_quantity(
        "distance", udf, defined_on="nodes",
        cmap="viridis",
        isolines_enabled=True,
        enabled=True,
    )
    # Slice plane the user can drag through the volume.
    ps.add_scene_slice_plane()

    print("opening polyscope viewer (Ctrl+C in the terminal to exit) ...")
    ps.show()


if __name__ == "__main__":
    main()
