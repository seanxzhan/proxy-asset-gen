"""Evaluate learned skinning weights by translating a sphere through the
pinned proxy cloth and driving the visual mesh via LBS.

The sphere walks along x from `start_x_frac * diag_x` to `end_x_frac * diag_x`
at the cloth's centroid y, where `diag_x` is the x-extent of the proxy bounding
box. The cloth is pinned at the same top-y vertices used during training so it
deflects without flying away. After the run the proxy and the LBS-driven visual
mesh are shown side-by-side with the sphere drawn in both panes.

Example
-------
$ python scripts/eval_scenarios/moving_sphere.py \\
      --visual data/9423122485_cleaned.obj \\
      --anim-dir data/9423122485_cleaned_proxy \\
      --weights results/9423122485_cleaned_proxy_skinning.npz \\
      --frames 240
"""
from __future__ import annotations

import argparse

import numpy as np

import pbd

from pag.eval_runner import (
    Obstacle,
    lbs_drive,
    load_eval_inputs,
    run_proxy_sim,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visual", required=True,
                    help="Visual mesh OBJ — must match the training mesh.")
    ap.add_argument("--anim-dir", required=True,
                    help="Directory containing mesh.npz (rest pose + pinned).")
    ap.add_argument("--weights", required=True,
                    help="Skinning weights .npz from get_skin_weights.py --out.")
    ap.add_argument("--frames", type=int, default=240,
                    help="Number of logged frames during the moving phase "
                         "(sphere lerps from start to end over these frames).")
    ap.add_argument("--tail-frames", type=int, default=0,
                    help="Extra logged frames AFTER the sphere reaches `end`, "
                         "with the sphere held in place. Useful for showing "
                         "the cloth settling against the stopped sphere.")
    ap.add_argument("--radius-frac", type=float, default=0.10,
                    help="Sphere radius as a fraction of the proxy bbox diagonal.")
    ap.add_argument("--start-x-frac", type=float, default=1.5,
                    help="Sphere start x as a fraction of half-x-extent. "
                         "1.0 = at the cloth's +x bbox edge; >1.0 = outside.")
    ap.add_argument("--end-x-frac", type=float, default=-1.5,
                    help="Sphere end x as a fraction of half-x-extent. "
                         "-1.0 = at the cloth's -x bbox edge; <-1.0 = outside.")
    # Sim defaults match training (gen_windblown_data.py).
    ap.add_argument("--dt", type=float, default=1.0 / 60.0)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--k-damp", type=float, default=0.05)
    ap.add_argument("--k-stretch", type=float, default=0.99)
    ap.add_argument("--k-bend", type=float, default=0.1)
    ap.add_argument("--friction", type=float, default=0.4,
                    help="Tangential friction at contact. 0 = frictionless slide; "
                         "~0.4 = matches the draped_on_ball.py viewer default.")
    ap.add_argument("--restitution", type=float, default=0.0,
                    help="Normal bounce on contact. 0 = inelastic (cloth stays put), "
                         "1 = perfectly elastic.")
    ap.add_argument("--n-settle", type=int, default=30,
                    help="Un-logged steps before the trajectory begins, so the "
                         "cloth swings from its rest pose to a hanging steady "
                         "state under gravity+pinning before the sphere arrives.")
    ap.add_argument("--no-viz", action="store_true",
                    help="Skip the polyscope viewer (for smoke tests).")
    args = ap.parse_args()

    print(f"loading inputs:")
    print(f"  visual : {args.visual}")
    print(f"  anim   : {args.anim_dir}")
    print(f"  weights: {args.weights}")
    V_visual, F_visual, V_p0, F_p, pinned, s, B = load_eval_inputs(
        args.visual, args.anim_dir, args.weights,
    )
    print(f"  visual |V|={V_visual.shape[0]}  proxy |V|={V_p0.shape[0]}  "
          f"|pinned|={pinned.shape[0]}  k_B={B.shape[1]}")

    # Sphere trajectory derived from the proxy bbox (so the same script works
    # on any proxy/visual pair without retuning).
    bbox_min = V_p0.min(axis=0)
    bbox_max = V_p0.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    park_scaling_offset = 0.4
    half_x = park_scaling_offset * (bbox_max[0] - bbox_min[0])
    cy = float(park_scaling_offset * (bbox_max[1] + bbox_min[1]))
    z_offset = 0.1
    start = np.array([args.start_x_frac * half_x, cy, z_offset], dtype=np.float64)
    end = np.array([args.end_x_frac * half_x, cy, z_offset], dtype=np.float64)
    radius = args.radius_frac * diag
    print(f"sphere: r={radius:.3f}  start={start}  end={end}  "
          f"T_move={args.frames}  T_tail={args.tail_frames}")

    # System setup (mirrors examples/draped_on_ball.py).
    mesh = pbd.build_mesh(V_p0, F_p)
    sys = pbd.System.from_mesh(mesh, density=1.0, gravity=(0.0, -9.81, 0.0))
    sys.add_constraint(pbd.Stretch.from_mesh(mesh, k=args.k_stretch))
    sys.add_constraint(pbd.Bend.from_mesh(mesh, k=args.k_bend))
    sys.pin(pinned)

    # Park the sphere far off-stage during the settle phase so the cloth
    # reaches a clean hanging rest under gravity+pinning before the run
    # begins. per_frame(0) below snaps it to `start` for the first logged
    # step, so frame 0 shows "just-arrived" contact, not pre-draped cloth.
    park = start + np.array([10.0 * diag, 0.0, 0.0])
    sphere = pbd.Sphere(center=park.copy(), radius=radius)
    sys.add_collider(sphere)

    T_move = args.frames
    T = T_move + args.tail_frames

    def per_frame(t: int, _sys) -> None:
        # Clamp t at T_move - 1 so frames past the moving phase keep the
        # sphere parked at `end` while the cloth keeps simulating.
        t_clamped = min(t, T_move - 1)
        u = t_clamped / max(T_move - 1, 1)
        sphere.center[:] = (1.0 - u) * start + u * end

    def obstacles_at(_t: int) -> list[Obstacle]:
        return [Obstacle(
            name="sphere",
            kind="sphere",
            center=sphere.center.copy(),
            radius=float(sphere.radius),
        )]

    print(f"simulating: {args.n_settle} settle + {T} logged frames")
    X_p, obs_log = run_proxy_sim(
        sys, T, per_frame, obstacles_at,
        dt=args.dt, iters=args.iters, k_damp=args.k_damp,
        friction=args.friction, restitution=args.restitution,
        n_settle=args.n_settle,
    )
    print(f"  proxy y range over run: "
          f"[{X_p[..., 1].min():.3f}, {X_p[..., 1].max():.3f}]")

    print("driving visual via LBS")
    V_recon = lbs_drive(s, B, V_visual, V_p0, X_p)
    print(f"  V_recon shape={V_recon.shape}")

    if args.no_viz:
        return

    from pag.eval_viz import show_eval
    show_eval(V_visual, F_visual, V_p0, F_p, X_p, V_recon, obs_log)


if __name__ == "__main__":
    main()
