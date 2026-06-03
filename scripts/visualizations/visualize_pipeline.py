"""Visualize the entire Stage 1 pipeline: M_visual → M_iso → M_proj → M_single → M_proxy.

End-to-end inspector. Runs ``generate_proxy_mesh`` with intermediates retained
and registers all five meshes in one polyscope window. Each non-input mesh
carries a per-vertex closest-point distance to ``M_visual`` on a common viridis
scale so surface fit is comparable at a glance:

  - ``M_iso``    distances cluster near ``voxel_size`` (one voxel off, by design).
  - ``M_proj``   distances should be ≈ 0 (data term minimised).
  - ``M_single`` same as M_proj on the surviving verts (it's just a face filter).
  - ``M_proxy``  distances stay small but become non-zero where ACVD merges/moves.

M_proj also carries the §3.3 ILP label (1=keep / 0=remove), the d_+ ray distance
(NaN where +n escapes to ∞), and the opposite-edge curve network so the cut
the ILP made is visible directly on the projected mesh.

Toggle individual meshes / quantities in the polyscope sidebar.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pag import build_guide_graph, generate_proxy_mesh, load_obj


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input", type=str,
        default="/Users/szhan/projects/proxy-asset-gen/data/jacket.obj",
        help="Path to visual mesh OBJ.",
    )
    ap.add_argument("--n-v", type=int, default=32,
                    help="§3.1 voxel resolution (paper: 32).")
    ap.add_argument("--n-p", type=int, default=128,
                    help="§3.4 target proxy vertex count (paper: 128).")
    ap.add_argument("--proj-iters", type=int, default=200,
                    help="§3.2 vector-Adam iterations.")
    ap.add_argument("--lr", type=float, default=5e-3, help="§3.2 Adam step size.")
    ap.add_argument("--lambda-L", type=float, default=0.1,
                    help="§3.2 Laplacian smoothing weight.")
    ap.add_argument("--lambda-s", type=float, default=1.0,
                    help="§3.3 smoothness weight.")
    ap.add_argument("--lambda-o", type=float, default=1.0,
                    help="§3.3 opposite-energy weight.")
    ap.add_argument("--lambda-bias", type=float, default=0.5,
                    help="§3.3 outer-layer bias (multiplied by λ_o).")
    ap.add_argument("--no-bias", action="store_true",
                    help="Disable §3.3 outer-layer bias (debug knob).")
    ap.add_argument("--ilp-time", type=float, default=120.0,
                    help="HiGHS time limit (s).")
    ap.add_argument("--smoke", action="store_true",
                    help="Headless; print stats and exit.")
    args = ap.parse_args()

    path = Path(args.input)
    print(f"loading {path} ...")
    V_visual, F_visual = load_obj(path)
    print(f"  |V|={V_visual.shape[0]}  |F|={F_visual.shape[0]}")

    out = generate_proxy_mesh(
        V_visual, F_visual,
        n_v=args.n_v,
        n_p=args.n_p,
        proj_iters=args.proj_iters,
        proj_lr=args.lr,
        lambda_L=args.lambda_L,
        lambda_s=args.lambda_s,
        lambda_o=args.lambda_o,
        lambda_bias=args.lambda_bias,
        enable_outer_bias=not args.no_bias,
        ilp_time_limit=args.ilp_time,
        keep_intermediates=True,
        verbose=True,
    )

    n_comp = out.mesh.n_components()
    n_loops = out.mesh.n_boundary_loops()
    print()
    print("=== M_proxy summary ===")
    print(f"  |V|={out.mesh.n_verts}  |F|={out.mesh.n_faces}")
    print(f"  components={n_comp}  boundary loops={n_loops}")
    print(f"  total runtime: {out.timings['total']:.3f}s")

    iso, proj, single, proxy = out.M_iso, out.M_proj, out.M_single, out.mesh

    # Per-stage distance to M_visual on a common viridis scale.
    d_iso = _closest_point_distance(iso.V, V_visual, F_visual)
    d_proj = _closest_point_distance(proj.V, V_visual, F_visual)
    d_single = (
        _closest_point_distance(single.V, V_visual, F_visual)
        if single.n_verts else np.zeros(0)
    )
    d_proxy = _closest_point_distance(proxy.V, V_visual, F_visual)

    print()
    print("closest-point distance to M_visual:")
    print(f"  M_iso:    mean={d_iso.mean():.4g}  max={d_iso.max():.4g}  "
          f"(voxel_size={out.voxel_size:.4g})")
    print(f"  M_proj:   mean={d_proj.mean():.4g}  max={d_proj.max():.4g}")
    if d_single.size:
        print(f"  M_single: mean={d_single.mean():.4g}  max={d_single.max():.4g}")
    print(f"  M_proxy:  mean={d_proxy.mean():.4g}  max={d_proxy.max():.4g}")

    if args.smoke:
        return

    # Recompute the guide graph + labels so M_proj can carry the ILP
    # diagnostics (labels, d_+, opposite edges). Cheap relative to the
    # rest of the pipeline.
    guide = build_guide_graph(proj, voxel_size=out.voxel_size)
    # Reconstruct kept-mask on M_proj from M_single's vertex positions:
    # extract_single_layer keeps positions bit-exact, so we can match by
    # nearest-vertex equality.
    keep_mask = np.zeros(proj.n_verts, dtype=bool)
    if single.n_verts:
        # Build a tiny lookup — coords are bit-exact from extract.py.
        proj_keys = {tuple(v): i for i, v in enumerate(proj.V)}
        for v in single.V:
            i = proj_keys.get(tuple(v))
            if i is not None:
                keep_mask[i] = True

    import polyscope as ps
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_front_dir("z_front")
    ps.set_ground_plane_mode("none")

    # Common viridis scale across iso/proj/single/proxy so the contrast is honest.
    vmax = float(d_iso.max())

    visual_ps = ps.register_surface_mesh(
        "M_visual", V_visual, F_visual,
        color=(0.65, 0.65, 0.7), transparency=0.4,
    )
    visual_ps.set_smooth_shade(True)

    iso_ps = ps.register_surface_mesh("M_iso (§3.1)", iso.V, iso.F)
    iso_ps.set_smooth_shade(True)
    iso_ps.set_transparency(0.7)
    iso_ps.add_scalar_quantity(
        "dist to M_visual", d_iso, defined_on="vertices",
        cmap="viridis", vminmax=(0.0, vmax),
    )

    proj_ps = ps.register_surface_mesh("M_proj (§3.2)", proj.V, proj.F)
    proj_ps.set_smooth_shade(True)
    proj_ps.set_transparency(0.7)
    proj_ps.add_scalar_quantity(
        "dist to M_visual", d_proj, defined_on="vertices",
        cmap="viridis", vminmax=(0.0, vmax),
    )
    proj_ps.add_scalar_quantity(
        "ILP keep (1=keep, 0=remove)",
        keep_mask.astype(np.float64), defined_on="vertices",
        cmap="coolwarm", vminmax=(0.0, 1.0), enabled=True,
    )
    d_plus_view = np.where(np.isfinite(guide.d_plus), guide.d_plus, np.nan)
    proj_ps.add_scalar_quantity(
        "d_+ (finite only; NaN = ∞)", d_plus_view, defined_on="vertices",
        cmap="magma",
    )
    if guide.opposite_edges.shape[0] > 0:
        ps.register_curve_network(
            "E_opposite (§3.3)", proj.V, guide.opposite_edges,
            color=(0.95, 0.55, 0.15), radius=0.001,
        )

    if single.n_faces > 0:
        single_ps = ps.register_surface_mesh(
            "M_single (§3.3)", single.V, single.F
        )
        single_ps.set_smooth_shade(True)
        single_ps.add_scalar_quantity(
            "dist to M_visual", d_single, defined_on="vertices",
            cmap="viridis", vminmax=(0.0, vmax),
        )

    proxy_ps = ps.register_surface_mesh("M_proxy (§3.4)", proxy.V, proxy.F)
    proxy_ps.set_smooth_shade(True)
    proxy_ps.set_edge_width(1.0)
    proxy_ps.add_scalar_quantity(
        "dist to M_visual", d_proxy, defined_on="vertices",
        cmap="viridis", vminmax=(0.0, vmax), enabled=True,
    )

    print("opening polyscope viewer (close window or Ctrl+C to exit) ...")
    ps.show()


if __name__ == "__main__":
    main()
