"""Visualize the §3.3 smoothness weight ``w_s`` on ``M_proj``: OLD vs CORRECTED.

The §3.3 smoothness term ``E_s = Σ_{ij∈E_proj} w_s^{ij} · 𝟙[l_i ≠ l_j]`` steers the
single-layer boundary. ``w_s`` is built from per-vertex curvature:

    w_i = 1 − ( |κ_i| / κ̄ )⁴ ,     w_s^{ij} = min(w_i, w_j) ∈ [0, 1]

Low ``w_s`` = cheap to cut = a seam/hole attractor; high ``w_s`` = protected. Two
implementations differ in BOTH the curvature measure and the normalizer ``κ̄``:

  OLD  (hole-prone):  |κ_i| = max(|κ₁|,|κ₂|)  (max principal),  κ̄ = mean_i|κ_i|,
                      ratio clamped to 1.   → many edges with w_s ≈ 0 (free to cut)
  CORRECTED (clean):  |κ_i| = |(κ₁+κ₂)/2|    (mean curvature),   κ̄ = max_i|κ_i|,
                      no clamp.             → w_s ≈ 1 almost everywhere (protected)

Both fields are computed from the SAME principal curvatures on the SAME ``M_proj``;
only the formula differs. The figure renders each weight field on the mesh side by
side, plus the per-edge ``w_s`` histograms, so you can see *why* OLD shatters
``M_single`` into holes (broad cheap-to-cut regions let label noise carve a ragged
seam) while CORRECTED over-regularizes into a watertight — but ridge-blind — seam.

Examples
--------
$ python examples/smoothness_weight_old_vs_new.py
$ python examples/smoothness_weight_old_vs_new.py --mesh data/jacket.obj --n-v 48
$ python examples/smoothness_weight_old_vs_new.py --collision-free --out /tmp/w.png
$ python examples/smoothness_weight_old_vs_new.py --polyscope   # interactive viewer
$ python examples/smoothness_weight_old_vs_new.py --smoke      # stats only, no render
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import igl
import numpy as np

from pag import isosurface_from_udf, load_obj, project_to_visual
from pag.mesh import Mesh


@dataclass
class WeightField:
    """A per-vertex/per-edge ``w_s`` field produced by one implementation."""
    tag: str
    kappa: np.ndarray       # per-vertex |κ_i| actually used
    w_vert: np.ndarray      # per-vertex w_i = 1 − (|κ_i|/κ̄)⁴
    w_edge: np.ndarray      # per-edge   w_s = min(w_i, w_j)
    kbar: float             # the normalizer κ̄


def _principal_curvatures(mesh: Mesh, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex principal curvatures (κ₁, κ₂) via libigl quadric fit.

    Indexed (not unpacked) so it tolerates the 4- or 5-tuple igl returns.
    """
    V = np.ascontiguousarray(mesh.V, dtype=np.float64)
    F = np.ascontiguousarray(mesh.F, dtype=np.int64)
    res = igl.principal_curvature(V, F, radius=radius)
    return np.asarray(res[2], dtype=np.float64), np.asarray(res[3], dtype=np.float64)


def _edge_weights(w_vert: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # max(κ_i, κ_j) → min(w_i, w_j), since w = 1 − (·)⁴ decreases in κ.
    return np.minimum(w_vert[edges[:, 0]], w_vert[edges[:, 1]])


def weights_old(pv1: np.ndarray, pv2: np.ndarray, edges: np.ndarray) -> WeightField:
    """max-principal curvature ÷ MEAN, ratio clamped to [0, 1] (the buggy variant)."""
    kappa = np.maximum(np.abs(pv1), np.abs(pv2))
    kbar = float(kappa.mean()) if kappa.size else 0.0
    if kbar > 0.0:
        ratio = np.minimum(kappa / kbar, 1.0)
        w_vert = 1.0 - ratio ** 1
    else:
        w_vert = np.ones_like(kappa)
    return WeightField("OLD: max|κ| ÷ mean (clamped)", kappa, w_vert,
                       _edge_weights(w_vert, edges), kbar)


def weights_new(pv1: np.ndarray, pv2: np.ndarray, edges: np.ndarray) -> WeightField:
    """mean curvature ÷ MAX, no clamp (the corrected, paper-normalizer variant)."""
    kappa = np.abs(0.5 * (pv1 + pv2))
    kbar = float(kappa.max()) if kappa.size else 0.0
    if kbar > 0.0:
        ratio = kappa / kbar                      # ∈ [0, 1] by construction
        w_vert = 1.0 - ratio ** 1
    else:
        w_vert = np.ones_like(kappa)
    return WeightField("CORRECTED: mean curv ÷ max", kappa, w_vert,
                       _edge_weights(w_vert, edges), kbar)


def _stats_line(wf: WeightField) -> str:
    w = wf.w_edge
    return (f"{wf.tag:30}  κ̄={wf.kbar:8.3f}  "
            f"median w_s={np.median(w):.3f}  mean={w.mean():.3f}  "
            f"frac(w_s<0.1)={np.mean(w < 0.1):5.1%}  frac(w_s≈0)={np.mean(w < 1e-6):5.1%}")


def render(mesh: Mesh, old: WeightField, new: WeightField, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    V, F = mesh.V, mesh.F
    cmap = "RdYlGn"   # 0 (free to cut) → red, 1 (protected) → green

    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 2, height_ratios=[2.4, 1.0], hspace=0.16, wspace=0.04)

    def surf(ax, wf: WeightField):
        face_vals = wf.w_vert[F].mean(axis=1)     # per-vertex field averaged to faces
        c = ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F,
                            cmap=cmap, shade=False, linewidth=0.0, antialiased=False)
        c.set_array(face_vals)
        c.set_clim(0.0, 1.0)
        ax.set_title(f"{wf.tag}\nmedian w_s={np.median(wf.w_edge):.3f},  "
                    f"{np.mean(wf.w_edge < 0.1):.0%} of edges w_s<0.1", fontsize=10)
        ax.set_axis_off()
        try:
            ax.set_box_aspect([max(np.ptp(V[:, k]), 1e-3) for k in range(3)])
        except Exception:
            pass
        return c

    ax_o = fig.add_subplot(gs[0, 0], projection="3d")
    ax_n = fig.add_subplot(gs[0, 1], projection="3d")
    surf(ax_o, old)
    c = surf(ax_n, new)
    cb = fig.colorbar(c, ax=[ax_o, ax_n], shrink=0.6, pad=0.02)
    cb.set_label("smoothness weight  w_s   (0 = free to cut → hole-prone,   1 = protected)")

    ax_h = fig.add_subplot(gs[1, :])
    bins = np.linspace(0.0, 1.0, 51)
    ax_h.hist(old.w_edge, bins=bins, color="tab:red", alpha=0.55, label=old.tag)
    ax_h.hist(new.w_edge, bins=bins, color="tab:green", alpha=0.55, label=new.tag)
    ax_h.axvline(np.median(old.w_edge), color="tab:red", ls="--", lw=1.2)
    ax_h.axvline(np.median(new.w_edge), color="tab:green", ls="--", lw=1.2)
    ax_h.set_yscale("log")
    ax_h.set_xlim(0.0, 1.0)
    ax_h.set_xlabel("per-edge smoothness weight  w_s = min(w_i, w_j)")
    ax_h.set_ylabel("edge count (log)")
    ax_h.set_title("per-edge w_s — OLD spreads toward 0 (cheap cuts → ragged seam → holes);  "
                  "CORRECTED piles at ~1 (near-uniform regularizer)", fontsize=10)
    ax_h.legend(loc="upper center", fontsize=9)

    fig.suptitle("§3.3 smoothness weight on M_proj — old (hole-prone) vs corrected", fontsize=13)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  wrote {path}")


def view_polyscope(mesh: Mesh, old: WeightField, new: WeightField,
                   *, cmap: str = "viridis", offset_frac: float = 1.35) -> None:
    """Interactive side-by-side viewer: OLD weight field (left) vs CORRECTED (right).

    Two copies of ``M_proj`` are placed next to each other and colored by THEIR
    own per-vertex weight on a shared ``[0, 1]`` scale, so the colors are directly
    comparable. Each also carries, registered but disabled (toggle in the GUI):

      - ``w_s`` as a per-face quantity (the face-min = its cheapest incident edge),
      - the ``|κ|`` that fed the weight, and
      - a per-EDGE ``w_s`` curve network — exactly the quantity ``E_s`` sums over.
    """
    try:
        import polyscope as ps
    except ImportError as e:  # pragma: no cover - optional dependency
        raise SystemExit("polyscope not installed — run `pip install polyscope`") from e

    V = np.ascontiguousarray(mesh.V, dtype=np.float64)
    F = np.ascontiguousarray(mesh.F, dtype=np.int64)
    edges = np.ascontiguousarray(mesh.edges, dtype=np.int64)
    span_x = float(np.ptp(V[:, 0])) if V.shape[0] else 1.0
    dx = np.array([offset_frac * max(span_x, 1e-6), 0.0, 0.0])

    ps.init()
    ps.set_ground_plane_mode("none")    # floating comparison — no reflective floor

    for wf, shift in [(old, np.zeros(3)), (new, dx)]:
        wf: WeightField
        nodes = V + shift
        sm = ps.register_surface_mesh(f"M_proj — {wf.tag}", nodes, F, smooth_shade=True)
        print(wf.w_vert.min(), wf.w_vert.max())
        sm.add_scalar_quantity(
            "w  (1 = protected, 0 = free to cut)", wf.w_vert, defined_on="vertices",
            vminmax=(0.5, wf.w_vert.max()), cmap=cmap, enabled=True, onscreen_colorbar_enabled=True,
        )
        # sm.add_scalar_quantity(
        #     "w_s  (face-min = cheapest incident edge)", wf.w_vert[F].min(axis=1),
        #     defined_on="faces", vminmax=(0.0, 1.0), cmap=cmap,
        # )
        sm.add_scalar_quantity("|kappa| used", wf.kappa, defined_on="vertices", cmap="reds")
        # Per-edge w_s — exactly what E_s sums over. Structure disabled by default;
        # toggle it on (and the surface off) in the GUI to see the actual cut costs.
        cn = ps.register_curve_network(f"E_proj  w_s — {wf.tag}", nodes, edges, enabled=False)
        cn.add_scalar_quantity(
            "w_s = min(w_i, w_j)", wf.w_edge, defined_on="edges",
            vminmax=(wf.w_edge.min(), wf.w_edge.max()), cmap=cmap, enabled=True,
        )

    print("  launching polyscope — left = OLD (hole-prone), right = CORRECTED. "
          "Close the window to exit.")
    ps.show()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", default="data/jacket.obj")
    ap.add_argument("--n-v", type=int, default=32, help="UDF marching-cubes resolution.")
    ap.add_argument("--proj-iters", type=int, default=50)
    ap.add_argument("--proj-lr", type=float, default=5e-3)
    ap.add_argument("--proj-lambda-L", type=float, default=0.1)
    ap.add_argument("--collision-free", action="store_true",
                    help="Use the IPC self-intersection-free projection for M_proj.")
    ap.add_argument("--curvature-radius", type=int, default=3,
                    help="k-ring radius for the libigl curvature fit (guide_graph default = 3).")
    ap.add_argument("--out", default="/tmp/smoothness_weight_old_vs_new.png")
    ap.add_argument("--polyscope", action="store_true",
                    help="Launch the interactive polyscope viewer instead of saving a PNG.")
    ap.add_argument("--cmap", default="viridis",
                    help="polyscope colormap (e.g. viridis, coolwarm, reds, spectral).")
    ap.add_argument("--smoke", action="store_true", help="Print stats only; skip rendering.")
    args = ap.parse_args()

    V, F = load_obj(args.mesh)
    iso = isosurface_from_udf(V, F, n_v=args.n_v)
    proj_kwargs = dict(n_iters=args.proj_iters, lr=args.proj_lr, lambda_L=args.proj_lambda_L)
    if args.collision_free:
        proj_kwargs["collision_free"] = True
    proj = project_to_visual(iso.mesh, V, F, **proj_kwargs, verbose=True)
    mesh = proj.mesh

    edges = mesh.edges
    pv1, pv2 = _principal_curvatures(mesh, args.curvature_radius)
    old = weights_old(pv1, pv2, edges)
    new = weights_new(pv1, pv2, edges)

    print(f"mesh={args.mesh}  M_proj: V={mesh.n_verts} F={mesh.n_faces} "
          f"E={edges.shape[0]}  (collision_free={args.collision_free})")
    print(f"  {_stats_line(old)}")
    print(f"  {_stats_line(new)}")
    print("  low w_s = cheap to cut = hole-prone; OLD has broad cheap regions, "
          "CORRECTED is ~uniformly protected.")

    if args.smoke:
        return
    if args.polyscope:
        view_polyscope(mesh, old, new, cmap=args.cmap)
        return
    render(mesh, old, new, args.out)


if __name__ == "__main__":
    main()
