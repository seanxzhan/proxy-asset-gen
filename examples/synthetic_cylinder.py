"""Stage 1 pipeline on a programmatic thin cylindrical shell.

A teaching/debugging aid: every step has a hand-derivable answer.

  - M_visual: open cylinder (single-layer tube, no caps).
  - §3.1 UDF wraps both sides → M_iso has roughly 2× the vert count of
    the input, arranged as inner+outer shells.
  - §3.3 ILP should keep ~half — the +n-escapes-to-infinity (outer) shell —
    leaving M_single as a single tube with two boundary loops (top, bottom).
  - §3.4 ACVD resamples to ``n_p`` near-equilateral verts.

Useful when poking the ILP weights — much faster than the jacket and the
correct answer is obvious by symmetry.
"""
from __future__ import annotations

import argparse

import numpy as np

from pag import generate_proxy_mesh, save_obj


def make_cylinder(
    radius: float = 1.0, height: float = 2.0,
    n_theta: int = 48, n_h: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Open cylinder (no caps) along y, centred at origin. y-up.

    Triangulated as a regular grid wrapped on the θ axis. Manifold with two
    boundary loops (y = ±height/2).
    """
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    ys = np.linspace(-0.5 * height, 0.5 * height, n_h)
    TT, YY = np.meshgrid(thetas, ys, indexing="xy")  # (n_h, n_theta)
    V = np.stack([
        radius * np.cos(TT).ravel(),
        YY.ravel(),
        radius * np.sin(TT).ravel(),
    ], axis=1)

    F: list[list[int]] = []
    for j in range(n_h - 1):
        for i in range(n_theta):
            i1 = (i + 1) % n_theta
            a = j * n_theta + i
            b = j * n_theta + i1
            c = (j + 1) * n_theta + i
            d = (j + 1) * n_theta + i1
            F.append([a, b, d])
            F.append([a, d, c])
    return V, np.array(F, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--height", type=float, default=2.0)
    ap.add_argument("--n-theta", type=int, default=48,
                    help="Resolution around the tube.")
    ap.add_argument("--n-h", type=int, default=24,
                    help="Resolution along the tube axis.")
    ap.add_argument("--n-v", type=int, default=24,
                    help="§3.1 voxel resolution (smaller for fast iteration).")
    ap.add_argument("--n-p", type=int, default=64,
                    help="§3.4 target proxy vertex count.")
    ap.add_argument("--lambda-bias", type=float, default=0.5,
                    help="§3.3 outer-layer bias.")
    ap.add_argument("--no-bias", action="store_true",
                    help="Disable §3.3 outer-layer bias.")
    ap.add_argument("--out", type=str, default=None,
                    help="Optional path to save M_proxy as OBJ.")
    ap.add_argument("--smoke", action="store_true",
                    help="Headless; print stats and exit.")
    args = ap.parse_args()

    V, F = make_cylinder(
        radius=args.radius, height=args.height,
        n_theta=args.n_theta, n_h=args.n_h,
    )
    print(f"M_visual cylinder: |V|={V.shape[0]}  |F|={F.shape[0]}  "
          f"r={args.radius} h={args.height}")

    out = generate_proxy_mesh(
        V, F,
        n_v=args.n_v,
        n_p=args.n_p,
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
    print(f"  expected: 1 component, 2 boundary loops (top + bottom)")

    if args.out:
        save_obj(args.out, out.mesh.V, out.mesh.F)
        print(f"  wrote {args.out}")

    if args.smoke:
        return

    from pag.viz import show_stages
    print("opening polyscope viewer ...")
    show_stages(out, (V, F), spacing=0.8)


if __name__ == "__main__":
    main()
