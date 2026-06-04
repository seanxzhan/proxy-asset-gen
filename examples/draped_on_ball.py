"""Drape a cloth onto a floor (and optionally a sphere obstacle).

Coordinate convention: y-up. Cloth starts horizontal in the xz-plane at
y = 1.5 and falls under gravity onto the floor at y = 0.
"""
from __future__ import annotations

import argparse

import numpy as np

from pbd import Bend, Stretch, System, build_mesh
from pbd.viz import Viewer


def make_grid_mesh(n: int = 25, side: float = 1.0):
    """Cloth in the xz-plane at y=1.5 (above the floor)."""
    xs = np.linspace(-side, side, n)
    zs = np.linspace(-side, side, n)
    XX, ZZ = np.meshgrid(xs, zs, indexing="xy")
    V = np.stack([XX.ravel(), np.ones(n * n) * 1.5, ZZ.ravel()], axis=1)
    F = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b = j * n + (i + 1)
            c = (j + 1) * n + i
            d = (j + 1) * n + (i + 1)
            if (i + j) % 2 == 0:
                F.extend([[a, b, d], [a, d, c]])
            else:
                F.extend([[a, b, c], [b, d, c]])
    return build_mesh(V, np.array(F, dtype=np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--dt", type=float, default=1.0 / 60)
    ap.add_argument("--no-sphere", action="store_true",
                    help="omit the sphere obstacle (cloth just lands flat on the floor)")
    ap.add_argument("--solver", choices=["jacobi", "gauss-seidel"],
                    default="jacobi",
                    help="Constraint solver: Jacobi (default) or graph-colored Gauss-Seidel")
    ap.add_argument("--sphere-visual-inset", type=float, default=0.01,
                    help="Shrink rendered sphere radius by this much to hide the "
                         "chord-vs-arc dip between contact verts (physics unchanged). "
                         "Try ~L^2/(8·r) where L is cell side.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-frames", type=int, default=120)
    args = ap.parse_args()

    mesh = make_grid_mesh(args.n)
    sys = System.from_mesh(mesh, density=1.0, gravity=(0.0, -9.81, 0.0))
    sys.add_constraint(Stretch.from_mesh(mesh, k=0.99))
    sys.add_constraint(Bend.from_mesh(mesh, k=0.3))

    if args.smoke:
        # Manually wire colliders without the viewer (no GUI deps).
        from pbd.constraints.collision import Plane, Sphere
        sys.add_collider(Plane(normal=(0.0, 1.0, 0.0), offset=0.0))
        if not args.no_sphere:
            sys.add_collider(Sphere(center=np.array([0.0, 0.5, 0.0]), radius=0.4))

        import time
        t0 = time.time()
        for _ in range(args.smoke_frames):
            sys.step(dt=args.dt, iters=args.iters, k_damp=0.05,
                     restitution=0.0, friction=0.4, solver=args.solver)
        elapsed = time.time() - t0
        fps = args.smoke_frames / elapsed
        print(f"{args.smoke_frames} frames in {elapsed:.3f}s ({fps:.1f} fps) "
              f"[solver={args.solver}]")
        print(f"final y range: [{sys.X[:, 1].min():.3f}, {sys.X[:, 1].max():.3f}]")
        return

    viewer = Viewer(sys, mesh.F, name="cloth")
    viewer.add_floor(y=0.0)
    if not args.no_sphere:
        viewer.add_sphere_obstacle(
            center=(0.0, 0.5, 0.0),
            radius=0.4,
            visual_inset=args.sphere_visual_inset,
        )

    viewer.run(dt=args.dt, iters=args.iters, k_damp=0.05,
               restitution=0.0, friction=0.8, solver=args.solver,
               contact_skin=0.002)


if __name__ == "__main__":
    main()
