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
from pathlib import Path

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
    ap.add_argument("--contact-skin", type=float, default=0.025,
                    help="Safety margin between cloth and collider, in world "
                         "units. Helps prevent tunneling and stabilizes "
                         "contact. Matches draped_on_ball.py default.")
    # Optional ground-truth comparison: simulate the visual mesh directly with
    # PBD in a third pane. Higher-resolution meshes need more iterations and
    # often different stiffnesses than the proxy, so all four sim params are
    # exposed independently.
    ap.add_argument("--sim-full", action="store_true",
                    help="Also simulate the visual mesh directly with PBD "
                         "and show it in a third pane (proxy | LBS recon | "
                         "full sim). Ground-truth comparison for the LBS "
                         "approximation.")
    ap.add_argument("--full-iters", type=int, default=100,
                    help="Solver iterations for the full-mesh sim. Higher "
                         "than --iters because constraints take longer to "
                         "propagate across denser edge graphs.")
    ap.add_argument("--full-k-damp", type=float, default=0.05)
    ap.add_argument("--full-k-stretch", type=float, default=0.999)
    ap.add_argument("--full-k-bend", type=float, default=0.5)
    ap.add_argument("--full-contact-skin", type=float, default=0.025,
                    help="Contact skin for the full-mesh sim. Smaller than "
                         "--contact-skin because the visual mesh's shorter "
                         "edges otherwise produce visible stair-stepping at "
                         "the contact band.")
    ap.add_argument("--full-solver", choices=["jacobi", "gauss-seidel"],
                    default="gauss-seidel",
                    help="PBD solver for the full-mesh sim. Gauss-Seidel "
                         "converges roughly 2x faster than Jacobi at the "
                         "same iters because each constraint sees up-to-date "
                         "particle positions — a big win on dense meshes "
                         "where stretch corrections must propagate across "
                         "long edge chains.")
    ap.add_argument("--n-settle", type=int, default=120,
                    help="Un-logged steps before the trajectory begins, so the "
                         "cloth swings from its rest pose to a hanging steady "
                         "state under gravity+pinning before the sphere arrives.")
    ap.add_argument("--cache", type=str, default=None,
                    help="Path to a .npz cache. If the file exists and the "
                         "shapes match, load X_p / V_recon / V_full_sim from "
                         "it instead of re-simulating; otherwise simulate and "
                         "write to this path. Shape checks catch frame-count "
                         "/ mesh-resolution mismatches but NOT sim-param or "
                         "trajectory changes — delete the file when those "
                         "change, or pass --recompute.")
    ap.add_argument("--recompute", action="store_true",
                    help="Force re-simulation and overwrite --cache, even if "
                         "the cache file already exists.")
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

    sphere_color = (0.78, 0.65, 0.46)  # warm sand-dune tan

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
            color=sphere_color,
        )]

    cache_path = Path(args.cache) if args.cache else None
    load_from_cache = (cache_path is not None
                       and cache_path.exists()
                       and not args.recompute)

    if load_from_cache:
        print(f"loading cache: {cache_path}")
        z = np.load(cache_path, allow_pickle=False)
        X_p = np.ascontiguousarray(z["X_p"])
        V_recon = np.ascontiguousarray(z["V_recon"])
        V_full_sim = (np.ascontiguousarray(z["V_full_sim"])
                      if "V_full_sim" in z.files else None)
        sphere_centers_log = z["sphere_centers"]
        sphere_radius_log = float(z["sphere_radius"])

        # Shape sanity. Sim params and trajectory aren't tracked — user
        # must --recompute (or delete the file) when those change.
        def _check(name: str, got, expected) -> None:
            if got != expected:
                raise SystemExit(
                    f"cache {name} shape {got} != expected {expected}; "
                    f"delete {cache_path} or pass --recompute."
                )
        _check("X_p", X_p.shape, (T, V_p0.shape[0], 3))
        _check("V_recon", V_recon.shape, (T, V_visual.shape[0], 3))
        _check("sphere_centers", sphere_centers_log.shape, (T, 3))
        if args.sim_full:
            if V_full_sim is None:
                raise SystemExit(
                    f"--sim-full requested but {cache_path} has no "
                    f"V_full_sim entry. Pass --recompute to regenerate."
                )
            _check("V_full_sim", V_full_sim.shape, (T, V_visual.shape[0], 3))

        obs_log = [
            [Obstacle(name="sphere", kind="sphere",
                      center=sphere_centers_log[t].copy(),
                      radius=sphere_radius_log,
                      color=sphere_color)]
            for t in range(T)
        ]
        print(f"  X_p={X_p.shape}  V_recon={V_recon.shape}"
              + (f"  V_full_sim={V_full_sim.shape}"
                 if V_full_sim is not None else ""))
    else:
        print(f"simulating: {args.n_settle} settle + {T} logged frames")
        X_p, obs_log = run_proxy_sim(
            sys, T, per_frame, obstacles_at,
            dt=args.dt, iters=args.iters, k_damp=args.k_damp,
            friction=args.friction, restitution=args.restitution,
            contact_skin=args.contact_skin,
            n_settle=args.n_settle,
        )
        print(f"  proxy y range over run: "
              f"[{X_p[..., 1].min():.3f}, {X_p[..., 1].max():.3f}]")

        print("driving visual via LBS")
        V_recon = lbs_drive(s, B, V_visual, V_p0, X_p)
        print(f"  V_recon shape={V_recon.shape}")

        V_full_sim = None
        if args.sim_full:
            # Pin visual verts at or above the lowest y of the proxy's pinned
            # region — geometric rather than topological correspondence, so
            # it works without a vert-level proxy↔visual map.
            y_pin_thresh = float(V_p0[pinned, 1].min())
            pinned_full = np.where(V_visual[:, 1] >= y_pin_thresh)[0].astype(np.int64)
            print(f"full-mesh sim: |V|={V_visual.shape[0]}  "
                  f"|pinned|={pinned_full.shape[0]}  iters={args.full_iters}  "
                  f"k_stretch={args.full_k_stretch}  k_bend={args.full_k_bend}  "
                  f"k_damp={args.full_k_damp}  solver={args.full_solver}")
            if pinned_full.shape[0] == 0:
                raise SystemExit(
                    "no visual verts above the proxy's pinned y-threshold "
                    f"({y_pin_thresh:.4f}); cloth would fall away — refusing "
                    "to simulate. Check that --visual and --anim-dir share a "
                    "frame."
                )
            mesh_full = pbd.build_mesh(V_visual, F_visual.astype(np.int64))
            sys_full = pbd.System.from_mesh(
                mesh_full, density=1.0, gravity=(0.0, -9.81, 0.0),
            )
            sys_full.add_constraint(pbd.Stretch.from_mesh(mesh_full, k=args.full_k_stretch))
            sys_full.add_constraint(pbd.Bend.from_mesh(mesh_full, k=args.full_k_bend))
            sys_full.pin(pinned_full)
            sphere_full = pbd.Sphere(center=park.copy(), radius=radius)
            sys_full.add_collider(sphere_full)

            def per_frame_full(t: int, _sys) -> None:
                t_clamped = min(t, T_move - 1)
                u = t_clamped / max(T_move - 1, 1)
                sphere_full.center[:] = (1.0 - u) * start + u * end

            def obstacles_full(_t: int) -> list[Obstacle]:
                return [Obstacle(
                    name="sphere", kind="sphere",
                    center=sphere_full.center.copy(),
                    radius=float(sphere_full.radius),
                    color=sphere_color,
                )]

            print(f"simulating full mesh: {args.n_settle} settle + {T} "
                  f"logged frames")
            X_full, _ = run_proxy_sim(
                sys_full, T, per_frame_full, obstacles_full,
                dt=args.dt, iters=args.full_iters, k_damp=args.full_k_damp,
                friction=args.friction, restitution=args.restitution,
                contact_skin=args.full_contact_skin,
                n_settle=args.n_settle, solver=args.full_solver,
            )
            V_full_sim = X_full
            print(f"  full y range: "
                  f"[{V_full_sim[..., 1].min():.3f}, "
                  f"{V_full_sim[..., 1].max():.3f}]")

        if cache_path is not None:
            sphere_centers_log = np.stack(
                [obs_log[t][0].center for t in range(T)]
            ).astype(np.float64)
            save_kw = dict(
                X_p=X_p, V_recon=V_recon,
                sphere_centers=sphere_centers_log,
                sphere_radius=np.float64(radius),
            )
            if V_full_sim is not None:
                save_kw["V_full_sim"] = V_full_sim
            np.savez(cache_path, **save_kw)
            print(f"saved cache: {cache_path}")

    if args.no_viz:
        return

    from pag.eval_viz import show_eval
    show_eval(
        V_visual, F_visual, V_p0, F_p, X_p, V_recon, obs_log,
        V_full_sim=V_full_sim,
    )


if __name__ == "__main__":
    main()
