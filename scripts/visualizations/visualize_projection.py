"""Visualize §3.2 mesh projection: M_visual, M_iso, and M_proj together.

Loads a visual mesh, runs §3.1 (UDF + marching cubes → M_iso) and §3.2
(vector-Adam projection → M_proj), then shows all three in one polyscope
window with per-vertex closest-point distance to ``M_visual`` so the tightening
is visible at a glance:

  - ``M_iso`` distances cluster near ``voxel_size`` (one voxel off, by design).
  - ``M_proj`` distances should be ≈ 0 (data term minimised).

Toggle meshes individually in the polyscope sidebar to compare. Each mesh also
carries a ``component`` scalar — for closed visual inputs, M_iso/M_proj have
two coincident shells (inner + outer), the very thing §3.3 strips next.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pag import isosurface_from_udf, load_obj, project_to_visual


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


def _closest_point_distance(
    V_query: np.ndarray, V: np.ndarray, F: np.ndarray
) -> np.ndarray:
    """Per-query unsigned distance to the closest point on (V, F)."""
    import igl
    sqrD, _, _ = igl.point_mesh_squared_distance(
        np.ascontiguousarray(V_query, dtype=np.float64),
        np.ascontiguousarray(V, dtype=np.float64),
        np.ascontiguousarray(F, dtype=np.int64),
    )
    return np.sqrt(np.asarray(sqrD, dtype=np.float64))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input", type=str,
        default="/Users/szhan/projects/proxy-asset-gen/data/jacket.obj",
        help="Path to visual mesh OBJ.",
    )
    ap.add_argument("--n-v", type=int, default=32,
                    help="Voxel resolution along the longest bbox axis (paper §3.1: 32).")
    ap.add_argument("--n-iters", type=int, default=200,
                    help="Vector-Adam iterations for §3.2 projection.")
    ap.add_argument("--lr", type=float, default=5e-3,
                    help="Adam step size.")
    ap.add_argument("--lambda-L", type=float, default=0.1,
                    help="Laplacian smoothness weight.")
    ap.add_argument("--smoke", action="store_true",
                    help="Run headless and print stats; skip polyscope window.")
    args = ap.parse_args()

    path = Path(args.input)
    print(f"loading {path} ...")
    V_visual, F_visual = load_obj(path)
    print(f"  |V|={V_visual.shape[0]}  |F|={F_visual.shape[0]}")

    t0 = time.time()
    iso = isosurface_from_udf(V_visual, F_visual, n_v=args.n_v)
    t_iso = time.time() - t0
    labels = _component_labels(iso.mesh.n_verts, iso.mesh.edges)
    n_comp = int(labels.max()) + 1 if labels.size else 0
    print(f"§3.1 isosurface_from_udf: {t_iso:.3f}s")
    print(f"  M_iso: |V|={iso.mesh.n_verts} |F|={iso.mesh.n_faces} "
          f"voxel={iso.voxel_size:.4g} components={n_comp}")

    t0 = time.time()
    proj = project_to_visual(
        iso.mesh, V_visual, F_visual,
        n_iters=args.n_iters, lr=args.lr, lambda_L=args.lambda_L,
        record_history=True,
    )
    t_proj = time.time() - t0
    d0 = proj.history["data"][0]
    df = proj.history["data"][-1]
    print(f"§3.2 project_to_visual:    {t_proj:.3f}s  ({args.n_iters} iters)")
    print(f"  data energy: {d0:.4g} → {df:.4g}  ({d0 / max(df, 1e-30):.1f}× tighter)")

    # Per-vertex closest-point distance to M_visual — for both iso and proj.
    d_iso = _closest_point_distance(iso.mesh.V, V_visual, F_visual)
    d_proj = _closest_point_distance(proj.mesh.V, V_visual, F_visual)
    print(f"  closest-point distance to M_visual:")
    print(f"    M_iso:  mean={d_iso.mean():.4g}  max={d_iso.max():.4g}  "
          f"(voxel_size={iso.voxel_size:.4g})")
    print(f"    M_proj: mean={d_proj.mean():.4g}  max={d_proj.max():.4g}")

    if args.smoke:
        return

    import polyscope as ps
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_front_dir("z_front")
    ps.set_ground_plane_mode("none")

    # M_visual — semi-transparent grey, the unchanging reference.
    visual = ps.register_surface_mesh(
        "M_visual", V_visual, F_visual,
        color=(0.65, 0.65, 0.7), transparency=0.4,
    )
    visual.set_smooth_shade(True)

    # Use a common color scale across both iso & proj so the contrast is honest.
    vmax = float(d_iso.max())

    iso_ps = ps.register_surface_mesh("M_iso", iso.mesh.V, iso.mesh.F)
    iso_ps.set_smooth_shade(True)
    iso_ps.set_transparency(0.7)
    iso_ps.add_scalar_quantity(
        "dist to M_visual", d_iso, defined_on="vertices",
        cmap="viridis", vminmax=(0.0, vmax), enabled=True,
    )
    if n_comp > 1:
        iso_ps.add_scalar_quantity(
            "component", labels.astype(np.float64),
            defined_on="vertices", cmap="turbo",
        )

    proj_ps = ps.register_surface_mesh("M_proj", proj.mesh.V, proj.mesh.F)
    proj_ps.set_smooth_shade(True)
    proj_ps.add_scalar_quantity(
        "dist to M_visual", d_proj, defined_on="vertices",
        cmap="viridis", vminmax=(0.0, vmax), enabled=True,
    )
    if n_comp > 1:
        proj_ps.add_scalar_quantity(
            "component", labels.astype(np.float64),
            defined_on="vertices", cmap="turbo",
        )

    # Disp vectors from M_iso → M_proj (per-vertex), so the projection motion
    # is visible as arrows. Disabled by default; flip on in the sidebar.
    disp = proj.mesh.V - iso.mesh.V
    iso_ps.add_vector_quantity(
        "→ M_proj", disp, defined_on="vertices", color=(0.9, 0.4, 0.2),
    )

    print("opening polyscope viewer (Ctrl+C in the terminal to exit) ...")
    ps.show()


if __name__ == "__main__":
    main()
