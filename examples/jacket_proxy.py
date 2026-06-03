"""Run the Stage 1 pipeline on a visual cloth mesh.

Defaults to ``data/jacket.obj`` (the paper's poster example). Run headless
with ``--smoke`` for CI / one-line sanity, or omit it to launch a polyscope
window showing all five stages side-by-side.

Examples
--------
$ python examples/jacket_proxy.py --smoke
$ python examples/jacket_proxy.py --input data/9423122485.obj
$ python examples/jacket_proxy.py --n-v 24 --n-p 96 --out /tmp/proxy.obj
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pag import generate_proxy_mesh, load_obj, save_obj


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
    ap.add_argument("--lambda-L", type=float, default=0.1,
                    help="§3.2 Laplacian smoothing weight.")
    ap.add_argument("--lambda-bias", type=float, default=0.5,
                    help="§3.3 outer-layer bias (multiplied by λ_o).")
    ap.add_argument("--no-bias", action="store_true",
                    help="Disable §3.3 outer-layer bias (debug knob).")
    ap.add_argument("--out", type=str, default=None,
                    help="Optional path to save M_proxy as OBJ.")
    ap.add_argument("--smoke", action="store_true",
                    help="Headless run; print stats and exit.")
    args = ap.parse_args()

    path = Path(args.input)
    print(f"loading {path} ...")
    V, F = load_obj(path)
    print(f"  |V|={V.shape[0]}  |F|={F.shape[0]}")

    out = generate_proxy_mesh(
        V, F,
        n_v=args.n_v,
        n_p=args.n_p,
        proj_iters=args.proj_iters,
        lambda_L=args.lambda_L,
        lambda_bias=args.lambda_bias,
        enable_outer_bias=not args.no_bias,
        keep_intermediates=not args.smoke,
        verbose=True,
    )

    n_comp = out.mesh.n_components()
    n_loops = out.mesh.n_boundary_loops()
    print()
    print("=== M_proxy summary ===")
    print(f"  |V|={out.mesh.n_verts}  |F|={out.mesh.n_faces}")
    print(f"  components={n_comp}  boundary loops={n_loops}")
    print(f"  total runtime: {out.timings['total']:.3f}s")
    print(f"    paper Table 1: components=1.0  boundary loops=2.0  |V|=128")

    if args.out:
        save_obj(args.out, out.mesh.V, out.mesh.F)
        print(f"  wrote {args.out}")

    if args.smoke:
        return

    from pag.viz import show_stages
    print("opening polyscope viewer (close window or Ctrl+C to exit) ...")
    show_stages(out, (V, F))


if __name__ == "__main__":
    main()
