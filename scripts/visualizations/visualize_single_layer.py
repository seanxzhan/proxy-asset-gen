"""Visualize all four §3.1–§3.3 stages: M_visual, M_iso, M_proj, M_single.

Pipeline (mirrors visualize_projection.py + adds the §3.3 ILP layer):

  1. Load visual mesh (e.g. data/jacket.obj).
  2. §3.1 isosurface_from_udf       → M_iso  (double-cover wrap)
  3. §3.2 project_to_visual         → M_proj (tightened to M_visual)
  4. §3.3 build_guide_graph + solve_layer_ilp
                                    → labels (1 = keep, 0 = remove)
  5. Drop faces with any removed vertex → M_single
     (a hole-repair pass would normally come next; this script keeps it simple
      so the raw ILP cut is visually obvious).

Each registered mesh carries a "dist to M_visual" scalar so you can compare
surface fit at a glance, plus the raw labels on M_proj so the cut is colored.
Toggle individual meshes in the polyscope sidebar.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pag import (
    build_guide_graph,
    isosurface_from_udf,
    load_obj,
    project_to_visual,
    solve_layer_ilp,
)


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


def _filter_to_kept(
    V: np.ndarray, F: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop faces with any l=0 vertex; remap surviving verts to a compact range.

    Returns (V_single, F_single). Used purely for visual inspection — the real
    extract.py (TBD) will follow with a manifold-cleanup + hole-repair pass.
    """
    keep_v = labels.astype(bool)
    keep_face = keep_v[F].all(axis=1)
    F_kept = F[keep_face]

    # Compact vertex indexing.
    used = np.unique(F_kept.ravel())
    remap = -np.ones(V.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.size)
    return V[used], remap[F_kept]


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
    ap.add_argument("--lr", type=float, default=5e-3, help="Adam step size.")
    ap.add_argument("--lambda-L", type=float, default=0.1,
                    help="§3.2 Laplacian smoothness weight.")
    ap.add_argument("--lambda-s", type=float, default=1.0,
                    help="§3.3 smoothness energy weight.")
    ap.add_argument("--lambda-o", type=float, default=1.0,
                    help="§3.3 opposite-energy weight.")
    ap.add_argument("--lambda-bias", type=float, default=0.1,
                    help="§3.3 outer-layer bias weight (multiplied by λ_o).")
    ap.add_argument("--no-bias", action="store_true",
                    help="Disable §3.3 outer-layer bias (debug knob).")
    ap.add_argument("--ilp-time", type=float, default=120.0,
                    help="HiGHS time limit (s).")
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
    print(f"§3.1 isosurface_from_udf: {t_iso:.3f}s")
    print(f"  M_iso: |V|={iso.mesh.n_verts} |F|={iso.mesh.n_faces} "
          f"voxel={iso.voxel_size:.4g}")

    t0 = time.time()
    proj = project_to_visual(
        iso.mesh, V_visual, F_visual,
        n_iters=args.n_iters, lr=args.lr, lambda_L=args.lambda_L,
        record_history=True,
    )
    t_proj = time.time() - t0
    d0 = proj.history["data"][0]
    df = proj.history["data"][-1]
    print(f"§3.2 project_to_visual:   {t_proj:.3f}s ({args.n_iters} iters)")
    print(f"  data energy: {d0:.4g} → {df:.4g}  ({d0 / max(df, 1e-30):.1f}× tighter)")

    t0 = time.time()
    guide = build_guide_graph(proj.mesh, voxel_size=iso.voxel_size)
    t_guide = time.time() - t0
    print(f"§3.3 build_guide_graph:   {t_guide:.3f}s")
    print(f"  |E_smooth|={guide.mesh_edges.shape[0]}  "
          f"|E_opposite|={guide.opposite_edges.shape[0]}  "
          f"d_+ finite={int(np.isfinite(guide.d_plus).sum())}/{guide.d_plus.shape[0]}")

    t0 = time.time()
    res = solve_layer_ilp(
        guide,
        lambda_s=args.lambda_s, lambda_o=args.lambda_o,
        lambda_bias=args.lambda_bias,
        enable_outer_bias=not args.no_bias,
        time_limit=args.ilp_time,
    )
    t_ilp = time.time() - t0
    print(f"§3.3 solve_layer_ilp:     {t_ilp:.3f}s  (HiGHS: {res.solver_status})")
    print(f"  keep {res.n_keep}/{proj.mesh.n_verts}  "
          f"({100.0 * res.n_keep / proj.mesh.n_verts:.1f}%)  energy={res.energy:.4g}")

    V_single, F_single = _filter_to_kept(proj.mesh.V, proj.mesh.F, res.labels)
    print(f"  M_single: |V|={V_single.shape[0]}  |F|={F_single.shape[0]} "
          f"(after face-drop; hole repair TODO §3.3 extract)")

    # Per-vertex closest-point distance to M_visual — for the iso/proj/single trio.
    d_iso = _closest_point_distance(iso.mesh.V, V_visual, F_visual)
    d_proj = _closest_point_distance(proj.mesh.V, V_visual, F_visual)
    d_single = _closest_point_distance(V_single, V_visual, F_visual) if V_single.size else np.zeros(0)
    print(f"  closest-point distance to M_visual:")
    print(f"    M_iso:    mean={d_iso.mean():.4g}  max={d_iso.max():.4g}  "
          f"(voxel_size={iso.voxel_size:.4g})")
    print(f"    M_proj:   mean={d_proj.mean():.4g}  max={d_proj.max():.4g}")
    if d_single.size:
        print(f"    M_single: mean={d_single.mean():.4g}  max={d_single.max():.4g}")

    if args.smoke:
        return

    import polyscope as ps
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_front_dir("z_front")
    ps.set_ground_plane_mode("none")

    # Common viridis scale across iso/proj/single so the contrast is honest.
    vmax = float(d_iso.max())

    visual = ps.register_surface_mesh(
        "M_visual", V_visual, F_visual,
        color=(0.65, 0.65, 0.7), transparency=0.4,
    )
    visual.set_smooth_shade(True)

    iso_ps = ps.register_surface_mesh("M_iso", iso.mesh.V, iso.mesh.F)
    iso_ps.set_smooth_shade(True)
    iso_ps.set_transparency(0.7)
    iso_ps.add_scalar_quantity(
        "dist to M_visual", d_iso, defined_on="vertices",
        cmap="viridis", vminmax=(0.0, vmax),
    )

    proj_ps = ps.register_surface_mesh("M_proj", proj.mesh.V, proj.mesh.F)
    proj_ps.set_smooth_shade(True)
    proj_ps.set_transparency(0.7)
    proj_ps.add_scalar_quantity(
        "dist to M_visual", d_proj, defined_on="vertices",
        cmap="viridis", vminmax=(0.0, vmax),
    )
    # ILP labels — make the §3.3 cut visible directly on M_proj.
    proj_ps.add_scalar_quantity(
        "ILP keep (1=keep, 0=remove)",
        res.labels.astype(np.float64), defined_on="vertices",
        cmap="coolwarm", vminmax=(0.0, 1.0), enabled=True,
    )
    # d_+ where finite (∞ becomes vmax for color); useful for spot-checking the bias.
    d_plus_view = np.where(np.isfinite(guide.d_plus), guide.d_plus, np.nan)
    proj_ps.add_scalar_quantity(
        "d_+ (finite only; NaN = ∞)", d_plus_view, defined_on="vertices",
        cmap="magma",
    )
    # Opposite edges as a curve network — the ILP's cross-layer pairings.
    if guide.opposite_edges.shape[0] > 0:
        ps.register_curve_network(
            "E_opposite (§3.3)", proj.mesh.V, guide.opposite_edges,
            color=(0.95, 0.55, 0.15), radius=0.001,
        )

    if F_single.shape[0] > 0:
        single_ps = ps.register_surface_mesh("M_single", V_single, F_single)
        single_ps.set_smooth_shade(True)
        if d_single.size:
            single_ps.add_scalar_quantity(
                "dist to M_visual", d_single, defined_on="vertices",
                cmap="viridis", vminmax=(0.0, vmax), enabled=True,
            )

    print("opening polyscope viewer (Ctrl+C in the terminal to exit) ...")
    ps.show()


if __name__ == "__main__":
    main()
