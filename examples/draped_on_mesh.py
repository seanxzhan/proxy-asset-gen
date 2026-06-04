"""Drape a cloth onto an arbitrary triangle-mesh obstacle.

Demonstrates the ``TriangleMesh`` collider, which follows the paper §3.4
detection recipe: ray-cast each predicted move ``X_i → P_i`` for first-
hit (continuous collision); for particles whose ray missed but whose
predicted position still ended up inside the obstacle, fall back to
the closest point on the surface and that face's normal. The result is
the same half-space constraint ``n_c · (p − q_c) ≥ 0`` as the analytic
Plane / Sphere colliders.

Default obstacle is a 20-face icosahedron floating above the floor —
a small closed surface so the closest-point fallback path is
exercised, and the cloth gets a clearly faceted thing to drape over.

Pass ``--obj`` to swap in any obstacle mesh. Use ``--scale`` and
``--translate`` to fit it into the scene without editing the file.

Mesh winding must be consistently outward — most OBJ exporters produce
this; ``io.fix_winding`` repairs the rest.

Non-convex obstacles: TriangleMesh's closest-point fallback assumes the
host face's plane normal is the SDF gradient, which only holds on
convex meshes. On concave geometry it can mis-fire (phantom contacts
that yank the cloth around). Pass ``--convex-hull`` to substitute the
obstacle's convex hull as the collision proxy; the original mesh is
still drawn but physics uses the simpler hull.
"""
from __future__ import annotations

import argparse

import numpy as np

from pbd import (
    Bend,
    Plane,
    Stretch,
    System,
    TriangleMesh,
    build_mesh,
    convex_hull,
    load_obj,
)
from pbd.viz import Viewer


def make_grid_mesh(n: int = 25, side: float = 1.0, y: float = 1.5):
    """Cloth in the xz-plane at height y."""
    xs = np.linspace(-side, side, n)
    zs = np.linspace(-side, side, n)
    XX, ZZ = np.meshgrid(xs, zs, indexing="xy")
    V = np.stack([XX.ravel(), np.ones(n * n) * y, ZZ.ravel()], axis=1)
    F = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b = j * n + (i + 1)
            c = (j + 1) * n + i
            d = (j + 1) * n + (i + 1)
            F.extend([[a, b, d], [a, d, c]])
    return build_mesh(V, np.array(F, dtype=np.int64))


def make_icosphere(center=(0.0, 0.7, 0.0), radius: float = 0.4):
    """Regular icosahedron — 12 vertices, 20 outward-wound faces.

    A nice "minimum viable" closed mesh for the TriangleMesh demo:
    cheap, faceted enough to look obviously triangulated, and closed
    so the closest-point fallback is the path that catches a cloth
    vertex which crept inside.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    V = np.array([
        [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1],
    ], dtype=np.float64)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)        # unit sphere
    V = V * radius + np.asarray(center)
    F = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    return V, F


def make_tilted_pad(
    center=(0.0, 0.6, 0.0),
    side: float = 1.2,
    tilt_deg: float = 20.0,
    n_subdiv: int = 4,
):
    """Square pad in the xz-plane, lifted to ``center`` and tilted around
    the x-axis by ``tilt_deg`` degrees. Triangulated into 2*n_subdiv²
    faces so the cloth sees a triangulated (not analytic) obstacle.

    Tilt makes the cloth slide off if friction is too low — a useful
    parameter for showing friction effects with a non-trivial geometry.
    """
    s = side
    xs = np.linspace(-s, s, n_subdiv + 1)
    zs = np.linspace(-s, s, n_subdiv + 1)
    XX, ZZ = np.meshgrid(xs, zs, indexing="xy")
    V = np.stack([XX.ravel(), np.zeros((n_subdiv + 1) ** 2), ZZ.ravel()], axis=1)
    # Tilt around x-axis: y' = z·sin θ + y·cos θ, z' = z·cos θ - y·sin θ.
    th = np.deg2rad(tilt_deg)
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(th), -np.sin(th)],
        [0.0, np.sin(th), np.cos(th)],
    ])
    V = V @ R.T + np.asarray(center)

    F = []
    n = n_subdiv + 1
    for j in range(n_subdiv):
        for i in range(n_subdiv):
            a = j * n + i
            b = j * n + (i + 1)
            c = (j + 1) * n + i
            d = (j + 1) * n + (i + 1)
            # Wind so the face normal points along +y in the un-tilted
            # frame (i.e. up out of the pad's top face).
            F.extend([[a, c, d], [a, d, b]])
    return V, np.asarray(F, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default=None,
                    help="OBJ obstacle to drape onto; default = icosphere")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="scale obstacle uniformly around origin "
                         "(applied before --translate)")
    ap.add_argument("--translate", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    metavar=("X", "Y", "Z"),
                    help="translate obstacle by (X, Y, Z) after scaling")
    ap.add_argument("--convex-hull", action="store_true",
                    help="replace the obstacle with its convex hull for "
                         "collision (visual still shows the original mesh). "
                         "Use this for non-convex meshes — TriangleMesh's "
                         "closest-point fallback assumes convexity and can "
                         "fire phantom contacts on concave geometry.")
    ap.add_argument("--n", type=int, default=25,
                    help="cloth grid resolution")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--dt", type=float, default=1.0 / 60)
    ap.add_argument("--solver", choices=["jacobi", "gauss-seidel"],
                    default="jacobi")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-frames", type=int, default=120)
    args = ap.parse_args()

    if args.obj:
        Vobs, Fobs = load_obj(args.obj)
    else:
        Vobs, Fobs = make_icosphere()
    if args.scale != 1.0:
        Vobs = Vobs * args.scale
    if any(t != 0.0 for t in args.translate):
        Vobs = Vobs + np.asarray(args.translate)

    # Choose physics geometry: original mesh, or its convex hull for
    # non-convex obstacles. Visual mesh always stays as the original.
    if args.convex_hull:
        Vphys, Fphys = convex_hull(Vobs)
        print(f"convex-hull proxy: {Vphys.shape[0]} verts, {Fphys.shape[0]} faces "
              f"(visual mesh keeps {Vobs.shape[0]}/{Fobs.shape[0]})")
    else:
        Vphys, Fphys = Vobs, Fobs

    cloth = make_grid_mesh(args.n)
    sys = System.from_mesh(cloth, density=1.0, gravity=(0.0, -9.81, 0.0))
    sys.add_constraint(Stretch.from_mesh(cloth, k=0.99))
    sys.add_constraint(Bend.from_mesh(cloth, k=0.3))
    sys.add_collider(Plane(normal=(0.0, 1.0, 0.0), offset=0.0))
    obstacle = TriangleMesh(Vphys, Fphys)
    sys.add_collider(obstacle)
    print(f"cloth verts={cloth.n_verts}  obstacle faces={Fphys.shape[0]}")

    if args.smoke:
        import time
        t0 = time.time()
        for _ in range(args.smoke_frames):
            sys.step(dt=args.dt, iters=args.iters, k_damp=0.05,
                     restitution=0.0, friction=0.6, solver=args.solver,
                     contact_skin=0.005)
        elapsed = time.time() - t0
        fps = args.smoke_frames / elapsed
        print(f"{args.smoke_frames} frames in {elapsed:.3f}s ({fps:.1f} fps) "
              f"[solver={args.solver}]")
        print(f"final y range: [{sys.X[:, 1].min():.3f}, "
              f"{sys.X[:, 1].max():.3f}]")
        return

    viewer = Viewer(sys, cloth.F, name="cloth")
    viewer.add_floor(y=0.0)
    # Draw the obstacle as its own surface mesh; physics already wired.
    import polyscope as ps
    import polyscope.imgui as psim
    obs = ps.register_surface_mesh("obstacle", Vobs, Fobs)
    obs.set_color((0.7, 0.3, 0.3))
    obs.set_smooth_shade(False)

    # When the collision proxy differs from the visual mesh, also draw
    # the proxy so you can see what the cloth is actually colliding
    # with. Translucent wireframe-ish look so the original mesh stays
    # legible behind it.
    hull_viz = None
    if args.convex_hull:
        # Enable transparency globally so the original mesh shows
        # through the hull overlay; default is "none" (opaque).
        ps.set_transparency_mode("pretty")
        hull_viz = ps.register_surface_mesh("collision proxy (hull)", Vphys, Fphys)
        hull_viz.set_color((0.2, 0.6, 0.9))
        hull_viz.set_transparency(0.35)
        hull_viz.set_edge_width(1.0)
        hull_viz.set_smooth_shade(False)

    # Live obstacle toggle: removes/re-adds the TriangleMesh collider
    # AND hides/shows its visual mesh. Useful to compare the cloth's
    # behavior with vs. without the obstacle in the same starting state.
    state = {"obstacle_on": True}

    def on_ui():
        changed, new_on = psim.Checkbox("Obstacle active", state["obstacle_on"])
        if changed:
            state["obstacle_on"] = new_on
            if new_on:
                if obstacle not in sys.colliders:
                    sys.colliders.append(obstacle)
                obs.set_enabled(True)
                if hull_viz is not None:
                    hull_viz.set_enabled(True)
            else:
                if obstacle in sys.colliders:
                    sys.colliders.remove(obstacle)
                obs.set_enabled(False)
                if hull_viz is not None:
                    hull_viz.set_enabled(False)

    viewer.run(dt=args.dt, iters=args.iters, k_damp=0.05,
               restitution=0.0, friction=0.8, solver=args.solver,
               contact_skin=0.005, on_ui=on_ui)


if __name__ == "__main__":
    main()
