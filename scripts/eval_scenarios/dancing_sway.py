"""Evaluate learned skinning weights by oscillating the pinned vertices
laterally, mimicking the hip sway of a character dancing in a dress.

The pinned (top) vertices swing side-to-side in x (and optionally z) on a
sinusoidal trajectory. The free cloth below swings, lags, and settles like
fabric on a moving body. After the run the proxy and the LBS-driven visual
mesh are shown side-by-side.

Example
-------
$ python scripts/eval_scenarios/dancing_sway.py \
      --visual data/9423122485_cleaned.obj \
      --anim-dir data/9423122485_cleaned_proxy \
      --weights results/9423122485_cleaned_proxy_skinning.npz \
      --frames 360 --cycles 3
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
    ap.add_argument("--frames", type=int, default=360,
                    help="Number of logged frames during the sway phase.")
    ap.add_argument("--tail-frames", type=int, default=60,
                    help="Extra frames after sway ends with pins held still, "
                         "letting the cloth settle to rest.")
    ap.add_argument("--cycles", type=float, default=3.0,
                    help="Number of full left-right oscillation cycles over "
                         "--frames. More cycles = faster dancing.")
    ap.add_argument("--amplitude-x", type=float, default=0.3,
                    help="Peak lateral (x) displacement of the pinned vertices "
                         "as a fraction of the proxy bbox x-extent.")
    ap.add_argument("--amplitude-z", type=float, default=0.0,
                    help="Peak fore/aft (z) displacement of the pinned vertices "
                         "as a fraction of the proxy bbox z-extent. "
                         "Non-zero adds an elliptical hip-circle motion "
                         "(z leads x by 90 degrees).")
    ap.add_argument("--phase-z", type=float, default=0.25,
                    help="Phase offset of z-oscillation relative to x, in "
                         "fractions of a full cycle. 0.25 = 90 degrees (circular).")
    ap.add_argument("--sharpness", type=float, default=4.0,
                    help="Waveform sharpness. 1.0 = smooth sine; higher values "
                         "snap quickly between extremes and hold there, like "
                         "jerky dance steps. Internally raises |sin| to 1/s, "
                         "so s=4 gives near-square-wave snap with brief holds.")
    # Sim parameters (match training defaults).
    ap.add_argument("--dt", type=float, default=1.0 / 60.0)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--k-damp", type=float, default=0.05)
    ap.add_argument("--k-stretch", type=float, default=0.99)
    ap.add_argument("--k-bend", type=float, default=0.1)
    ap.add_argument("--friction", type=float, default=0.4)
    ap.add_argument("--restitution", type=float, default=0.0)
    ap.add_argument("--contact-skin", type=float, default=0.025)
    # Full-mesh ground-truth sim (same as moving_sphere.py).
    ap.add_argument("--sim-full", action="store_true",
                    help="Also simulate the visual mesh directly with PBD "
                         "in a third pane (proxy | LBS recon | full sim).")
    ap.add_argument("--full-iters", type=int, default=100)
    ap.add_argument("--full-k-damp", type=float, default=0.05)
    ap.add_argument("--full-k-stretch", type=float, default=0.999)
    ap.add_argument("--full-k-bend", type=float, default=0.5)
    ap.add_argument("--full-contact-skin", type=float, default=0.025)
    ap.add_argument("--full-solver", choices=["jacobi", "gauss-seidel"],
                    default="gauss-seidel")
    ap.add_argument("--n-settle", type=int, default=120,
                    help="Un-logged frames to let the cloth drape under gravity "
                         "before the sway begins.")
    ap.add_argument("--cache", type=str, default=None,
                    help="Path to a .npz cache file (same semantics as "
                         "moving_sphere.py --cache).")
    ap.add_argument("--recompute", action="store_true",
                    help="Force re-simulation even if --cache exists.")
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

    # Sway amplitude derived from the proxy bbox.
    bbox_min = V_p0.min(axis=0)
    bbox_max = V_p0.max(axis=0)
    extent_x = bbox_max[0] - bbox_min[0]
    extent_z = bbox_max[2] - bbox_min[2]
    amp_x = args.amplitude_x * extent_x
    amp_z = args.amplitude_z * extent_z

    T_sway = args.frames
    T = T_sway + args.tail_frames
    omega = 2.0 * np.pi * args.cycles / max(T_sway - 1, 1)
    phase_z = 2.0 * np.pi * args.phase_z

    sharpness = args.sharpness
    print(f"sway: amp_x={amp_x:.4f}  amp_z={amp_z:.4f}  "
          f"cycles={args.cycles}  sharpness={sharpness}  "
          f"T_sway={T_sway}  T_tail={args.tail_frames}")

    # Record rest positions of pinned vertices for offset computation.
    pin_rest = V_p0[pinned].copy()  # (|pinned|, 3)

    # System setup.
    mesh = pbd.build_mesh(V_p0, F_p)
    sys = pbd.System.from_mesh(mesh, density=1.0, gravity=(0.0, -9.81, 0.0))
    sys.add_constraint(pbd.Stretch.from_mesh(mesh, k=args.k_stretch))
    sys.add_constraint(pbd.Bend.from_mesh(mesh, k=args.k_bend))
    sys.pin(pinned)

    def _sharp_sin(theta: float) -> float:
        """sin shaped by sharpness: sign(sin) * |sin|^(1/s).

        s=1 is a normal sine. Higher s flattens near ±1 and snaps
        through zero — quick jerky direction changes with holds at
        the extremes.
        """
        v = np.sin(theta)
        return float(np.sign(v) * np.abs(v) ** (1.0 / sharpness))

    def per_frame(t: int, sys) -> None:
        t_clamped = min(t, T_sway - 1)
        dx = amp_x * _sharp_sin(omega * t_clamped)
        dz = amp_z * _sharp_sin(omega * t_clamped + phase_z)
        sys.X[pinned, 0] = pin_rest[:, 0] + dx
        sys.X[pinned, 2] = pin_rest[:, 2] + dz

    def obstacles_at(_t: int) -> list[Obstacle]:
        return []

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

        def _check(name: str, got, expected) -> None:
            if got != expected:
                raise SystemExit(
                    f"cache {name} shape {got} != expected {expected}; "
                    f"delete {cache_path} or pass --recompute."
                )
        _check("X_p", X_p.shape, (T, V_p0.shape[0], 3))
        _check("V_recon", V_recon.shape, (T, V_visual.shape[0], 3))
        if args.sim_full:
            if V_full_sim is None:
                raise SystemExit(
                    f"--sim-full requested but {cache_path} has no "
                    f"V_full_sim entry. Pass --recompute to regenerate."
                )
            _check("V_full_sim", V_full_sim.shape, (T, V_visual.shape[0], 3))

        obs_log = [[] for _ in range(T)]
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
            y_pin_thresh = float(V_p0[pinned, 1].min())
            pinned_full = np.where(V_visual[:, 1] >= y_pin_thresh)[0].astype(np.int64)
            print(f"full-mesh sim: |V|={V_visual.shape[0]}  "
                  f"|pinned|={pinned_full.shape[0]}  iters={args.full_iters}  "
                  f"k_stretch={args.full_k_stretch}  k_bend={args.full_k_bend}  "
                  f"k_damp={args.full_k_damp}  solver={args.full_solver}")
            if pinned_full.shape[0] == 0:
                raise SystemExit(
                    "no visual verts above the proxy's pinned y-threshold "
                    f"({y_pin_thresh:.4f}); cloth would fall away."
                )
            mesh_full = pbd.build_mesh(V_visual, F_visual.astype(np.int64))
            sys_full = pbd.System.from_mesh(
                mesh_full, density=1.0, gravity=(0.0, -9.81, 0.0),
            )
            sys_full.add_constraint(pbd.Stretch.from_mesh(mesh_full, k=args.full_k_stretch))
            sys_full.add_constraint(pbd.Bend.from_mesh(mesh_full, k=args.full_k_bend))
            sys_full.pin(pinned_full)

            pin_rest_full = V_visual[pinned_full].copy()

            def per_frame_full(t: int, sys_full) -> None:
                t_clamped = min(t, T_sway - 1)
                dx = amp_x * _sharp_sin(omega * t_clamped)
                dz = amp_z * _sharp_sin(omega * t_clamped + phase_z)
                sys_full.X[pinned_full, 0] = pin_rest_full[:, 0] + dx
                sys_full.X[pinned_full, 2] = pin_rest_full[:, 2] + dz

            def obstacles_full(_t: int) -> list[Obstacle]:
                return []

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
            save_kw = dict(X_p=X_p, V_recon=V_recon)
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
