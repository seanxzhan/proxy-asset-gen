#! /Users/szhan/miniforge3/envs/proxyasset/bin/python
"""Standalone §3.4 ACVD Voronoi simplification of an input mesh.

Loads a triangle OBJ, runs :func:`pag.voronoi.voronoi_simplify`, prints
before/after stats, and optionally writes the result + opens a polyscope viewer.

Examples
--------
$ python scripts/voronoi_simplify.py data/jacket.obj
$ python scripts/voronoi_simplify.py data/jacket.obj --n-p 256 --out /tmp/proxy.obj
$ python scripts/voronoi_simplify.py mesh.obj --smoke              # no viewer
$ python scripts/voronoi_simplify.py mesh.obj --no-auto-subdivide  # debug knob

Notes
-----
The input must be edge-manifold (no edge shared by >2 faces) — the same
constraint :func:`pag.mesh.build_mesh` enforces. Run §3.3 ``extract_single_layer``
first if you're starting from a multi-layer mesh like a UDF isosurface.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pag import build_mesh, load_obj, save_obj, voronoi_simplify


def _min_face_angle_deg(V: np.ndarray, F: np.ndarray) -> float:
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]

    def angle(u, v):
        cu = np.einsum("ij,ij->i", u, v)
        nu = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        return np.degrees(np.arccos(np.clip(cu / np.maximum(nu, 1e-30), -1.0, 1.0)))

    angles = np.stack(
        [angle(b - a, c - a), angle(a - b, c - b), angle(a - c, b - c)], axis=1
    )
    return float(angles.min())


def _mean_aspect_ratio(V: np.ndarray, F: np.ndarray) -> float:
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    ea = np.linalg.norm(b - c, axis=1)
    eb = np.linalg.norm(a - c, axis=1)
    ec = np.linalg.norm(a - b, axis=1)
    s = 0.5 * (ea + eb + ec)
    area = np.sqrt(np.maximum(s * (s - ea) * (s - eb) * (s - ec), 0.0))
    inradius = area / np.maximum(s, 1e-30)
    circ = (ea * eb * ec) / np.maximum(4.0 * area, 1e-30)
    return float((2.0 * inradius / np.maximum(circ, 1e-30)).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=str, help="Path to input OBJ.")
    ap.add_argument("--n-p", type=int, default=128,
                    help="Target proxy vertex count (paper §3.4: 128).")
    ap.add_argument("--max-iter", type=int, default=100,
                    help="pyacvd Lloyd-iteration cap.")
    ap.add_argument("--no-auto-subdivide", action="store_true",
                    help="Disable auto-subdivide of small inputs (will fail if |V| < n_p).")
    ap.add_argument("--out", type=str, default=None,
                    help="Optional path to write simplified mesh as OBJ.")
    ap.add_argument("--smoke", action="store_true",
                    help="Headless; print stats and exit.")
    args = ap.parse_args()

    path = Path(args.input)
    print(f"loading {path} ...")
    V, F = load_obj(path)
    mesh = build_mesh(V, F)
    print(f"  M_in:  |V|={mesh.n_verts}  |F|={mesh.n_faces}  "
          f"components={mesh.n_components()}  "
          f"boundary_loops={mesh.n_boundary_loops()}")
    print(f"         min_angle={_min_face_angle_deg(mesh.V, mesh.F):.2f}°  "
          f"mean_aspect={_mean_aspect_ratio(mesh.V, mesh.F):.3f}")

    t0 = time.time()
    out = voronoi_simplify(
        mesh,
        n_p=args.n_p,
        max_iter=args.max_iter,
        auto_subdivide=not args.no_auto_subdivide,
    )
    elapsed = time.time() - t0
    print(f"voronoi_simplify: {elapsed:.3f}s  (n_p={args.n_p}, max_iter={args.max_iter})")
    print(f"  M_out: |V|={out.n_verts}  |F|={out.n_faces}  "
          f"components={out.n_components()}  "
          f"boundary_loops={out.n_boundary_loops()}")
    print(f"         min_angle={_min_face_angle_deg(out.V, out.F):.2f}°  "
          f"mean_aspect={_mean_aspect_ratio(out.V, out.F):.3f}  "
          f"(paper Fig. 10: ≈47.9°, ≈0.76)")

    if args.out:
        save_obj(args.out, out.V, out.F)
        print(f"  wrote {args.out}")

    if args.smoke:
        return

    import polyscope as ps
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_front_dir("z_front")
    ps.set_ground_plane_mode("none")

    in_ps = ps.register_surface_mesh(
        "M_in", mesh.V, mesh.F, color=(0.65, 0.65, 0.7), transparency=0.4
    )
    in_ps.set_smooth_shade(True)

    out_ps = ps.register_surface_mesh(
        "M_simplified", out.V, out.F, color=(0.90, 0.45, 0.45)
    )
    out_ps.set_smooth_shade(True)
    out_ps.set_edge_width(1.0)

    print("opening polyscope viewer (close window or Ctrl+C to exit) ...")
    ps.show()


if __name__ == "__main__":
    main()
