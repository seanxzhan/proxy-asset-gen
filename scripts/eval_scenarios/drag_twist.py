"""Evaluate proxy skinning with the baseline drag/twist pin trajectories.

This is the proxy-asset counterpart of
``baselines/cpp/python/demo_drag_twist_contact.py``.  It prescribes the same
sinusoidal translation or rotation to the proxy's pinned vertices, simulates
the proxy with PBD, and drives the visual mesh from the result with learned
LBS weights.  The baseline's optional stationary sphere contact is supported
as well.

Examples
--------
$ python scripts/eval_scenarios/drag_twist.py \
      --visual data/9423122485_cleaned.obj \
      --anim-dir data/9423122485_cleaned_proxy \
      --weights results/9423122485_cleaned_proxy_skinning.npz \
      --demo drag

$ python scripts/eval_scenarios/drag_twist.py \
      --visual data/9423122485_cleaned.obj \
      --anim-dir data/9423122485_cleaned_proxy \
      --weights results/9423122485_cleaned_proxy_skinning.npz \
      --demo twist --collision
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import pbd

from pag.eval_runner import Obstacle, lbs_drive, load_eval_inputs, run_proxy_sim


def rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return the 3-D rotation used by the baseline drag/twist demo."""
    x, y, z = np.asarray(axis, dtype=np.float64)
    c = np.cos(angle)
    s = np.sin(angle)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [
                c + x * x * one_minus_c,
                x * y * one_minus_c - z * s,
                x * z * one_minus_c + y * s,
            ],
            [
                y * x * one_minus_c + z * s,
                c + y * y * one_minus_c,
                y * z * one_minus_c - x * s,
            ],
            [
                z * x * one_minus_c - y * s,
                z * y * one_minus_c + x * s,
                c + z * z * one_minus_c,
            ],
        ],
        dtype=np.float64,
    )


def pin_targets(
    pin_rest: np.ndarray,
    pin_center: np.ndarray,
    *,
    demo: str,
    time: float,
    drag_axis: int,
    drag_amp: float,
    drag_omega: float,
    twist_axis: int,
    twist_amp: float,
    twist_omega: float,
) -> np.ndarray:
    """Compute prescribed pins exactly as the baseline's ``pin_target``."""
    if demo == "drag":
        target = pin_rest.copy()
        target[:, drag_axis] += drag_amp * np.sin(drag_omega * time)
        return target
    if demo != "twist":
        raise ValueError(f"unknown demo: {demo!r}")

    axis = np.zeros(3, dtype=np.float64)
    axis[twist_axis] = 1.0
    angle = np.deg2rad(twist_amp) * np.sin(twist_omega * time)
    rotation = rodrigues(axis, angle)
    return (pin_rest - pin_center) @ rotation.T + pin_center


def _add_sim_arguments(ap: argparse.ArgumentParser) -> None:
    # Baseline trajectory and time-step defaults. PBD stiffness lies in [0, 1]
    # and therefore cannot directly use the C++ solver's energy weights.
    ap.add_argument("--dt", type=float, default=0.03)
    ap.add_argument("--density", type=float, default=0.1)
    ap.add_argument("--gravity", type=float, default=1.0,
                    help="Downward gravitational acceleration magnitude.")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--k-damp", type=float, default=0.05)
    ap.add_argument("--k-stretch", type=float, default=0.99)
    ap.add_argument("--k-bend", type=float, default=0.1)
    ap.add_argument("--friction", type=float, default=0.4)
    ap.add_argument("--restitution", type=float, default=0.0)
    ap.add_argument(
        "--contact-skin", "--margin", dest="contact_skin", type=float,
        default=0.03,
        help="Sphere contact margin (the baseline calls this --margin).",
    )
    ap.add_argument("--solver", choices=["jacobi", "gauss-seidel"],
                    default="jacobi")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visual", required=True,
                    help="Visual mesh OBJ; must match the training mesh.")
    ap.add_argument("--anim-dir", required=True,
                    help="Directory containing proxy rest-pose mesh.npz.")
    ap.add_argument("--weights", required=True,
                    help="Skinning weights .npz from get_skin_weights.py.")
    ap.add_argument("--demo", choices=["drag", "twist"], default="drag")
    ap.add_argument("--frames", type=int, default=240,
                    help="Number of frames following the sinusoidal motion.")
    ap.add_argument("--tail-frames", type=int, default=0,
                    help="Extra frames with pins held at their final target.")
    ap.add_argument("--drag-axis", "--drag_axis", dest="drag_axis", type=int,
                    choices=[0, 2], default=0)
    ap.add_argument("--drag-amp", "--drag_amp", dest="drag_amp", type=float,
                    default=0.5,
                    help="Peak drag displacement in world units.")
    ap.add_argument("--drag-omega", "--drag_omega", dest="drag_omega",
                    type=float, default=2.0,
                    help="Drag angular frequency in radians/second.")
    ap.add_argument("--twist-axis", "--twist_axis", dest="twist_axis",
                    type=int, choices=[0, 1, 2], default=1)
    ap.add_argument("--twist-amp", "--twist_amp", dest="twist_amp",
                    type=float, default=45.0,
                    help="Peak twist angle in degrees.")
    ap.add_argument("--twist-omega", "--twist_omega", dest="twist_omega",
                    type=float, default=2.0,
                    help="Twist angular frequency in radians/second.")

    ap.add_argument(
        "--collision", action=argparse.BooleanOptionalAction, default=False,
        help="Enable the baseline's stationary analytic sphere contact.",
    )
    ap.add_argument("--sphere-radius", "--sphere_radius", dest="sphere_radius",
                    type=float, default=None,
                    help="Sphere radius; default chooses a non-intersecting radius.")
    ap.add_argument("--sphere-y-offset", "--sphere_y_offset",
                    dest="sphere_y_offset", type=float, default=-0.25,
                    help="Offset from the proxy bbox's vertical midpoint.")

    _add_sim_arguments(ap)
    ap.add_argument("--n-settle", type=int, default=120,
                    help="Unlogged frames to settle before moving the pins.")
    ap.add_argument("--sim-full", action="store_true",
                    help="Also directly simulate the visual mesh in a third pane.")
    ap.add_argument("--full-iters", type=int, default=100)
    ap.add_argument("--full-k-damp", type=float, default=0.05)
    ap.add_argument("--full-k-stretch", type=float, default=0.999)
    ap.add_argument("--full-k-bend", type=float, default=0.5)
    ap.add_argument("--full-contact-skin", type=float, default=None,
                    help="Full-mesh contact margin; defaults to --contact-skin.")
    ap.add_argument("--full-solver", choices=["jacobi", "gauss-seidel"],
                    default="gauss-seidel")
    ap.add_argument("--cache", type=str, default=None,
                    help="Optional .npz simulation cache.")
    ap.add_argument("--recompute", action="store_true",
                    help="Ignore and overwrite an existing cache.")
    ap.add_argument("--no-viz", action="store_true",
                    help="Skip the Polyscope viewer.")
    ap.add_argument("--export-dir", default="out/drag_twist_obj",
                    help="Root directory used by the viewer's OBJ export button.")
    return ap


def _validate_args(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.frames < 1:
        ap.error("--frames must be positive")
    if args.tail_frames < 0 or args.n_settle < 0:
        ap.error("--tail-frames and --n-settle must be nonnegative")
    if args.dt <= 0.0 or args.density <= 0.0:
        ap.error("--dt and --density must be positive")
    if args.iters < 1 or args.full_iters < 1:
        ap.error("--iters and --full-iters must be positive")
    if args.gravity < 0.0 or args.contact_skin < 0.0:
        ap.error("--gravity and --contact-skin must be nonnegative")
    if args.full_contact_skin is not None and args.full_contact_skin < 0.0:
        ap.error("--full-contact-skin must be nonnegative")
    if args.sphere_radius is not None and args.sphere_radius <= 0.0:
        ap.error("--sphere-radius must be positive")
    for name in ("k_damp", "k_stretch", "k_bend", "full_k_damp",
                 "full_k_stretch", "full_k_bend"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            ap.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.friction < 0.0 or not 0.0 <= args.restitution <= 1.0:
        ap.error("--friction must be nonnegative and --restitution in [0, 1]")


def _make_system(
    vertices: np.ndarray,
    faces: np.ndarray,
    pinned: np.ndarray,
    *,
    density: float,
    gravity: float,
    k_stretch: float,
    k_bend: float,
    sphere_center: np.ndarray,
    sphere_radius: float,
    collision: bool,
):
    mesh = pbd.build_mesh(vertices, faces)
    system = pbd.System.from_mesh(
        mesh, density=density, gravity=(0.0, -gravity, 0.0),
    )
    system.add_constraint(pbd.Stretch.from_mesh(mesh, k=k_stretch))
    system.add_constraint(pbd.Bend.from_mesh(mesh, k=k_bend))
    system.pin(pinned)
    if collision:
        system.add_collider(
            pbd.Sphere(center=sphere_center.copy(), radius=sphere_radius)
        )
    return system


def _cache_array(
    z: np.lib.npyio.NpzFile,
    name: str,
    expected_shape: tuple[int, ...],
    cache_path: Path,
) -> np.ndarray:
    if name not in z.files:
        raise SystemExit(
            f"cache {cache_path} has no {name} entry; pass --recompute."
        )
    value = np.ascontiguousarray(z[name])
    if value.shape != expected_shape:
        raise SystemExit(
            f"cache {name} shape {value.shape} != expected {expected_shape}; "
            f"delete {cache_path} or pass --recompute."
        )
    return value


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    _validate_args(ap, args)

    print("loading inputs:")
    print(f"  visual : {args.visual}")
    print(f"  anim   : {args.anim_dir}")
    print(f"  weights: {args.weights}")
    V_visual, F_visual, V_p0, F_p, pinned, s, B = load_eval_inputs(
        args.visual, args.anim_dir, args.weights,
    )
    if pinned.size == 0:
        raise SystemExit("proxy has no pinned vertices; drag/twist needs a handle")
    print(
        f"  visual |V|={V_visual.shape[0]}  proxy |V|={V_p0.shape[0]}  "
        f"|pinned|={pinned.shape[0]}  k_B={B.shape[1]}"
    )

    lo = V_p0.min(axis=0)
    hi = V_p0.max(axis=0)
    sphere_center = np.array(
        [
            V_p0[:, 0].mean(),
            0.5 * (lo[1] + hi[1]) + args.sphere_y_offset,
            V_p0[:, 2].mean(),
        ],
        dtype=np.float64,
    )
    center_distances = np.linalg.norm(V_p0 - sphere_center, axis=1)
    safe_radius = max(
        1e-3,
        float(center_distances.min()) - args.contact_skin - 0.02,
    )
    sphere_radius = (
        min(0.5, safe_radius)
        if args.sphere_radius is None
        else args.sphere_radius
    )
    initial_clearance = float(
        center_distances.min() - sphere_radius - args.contact_skin
    )
    if args.collision and initial_clearance < 0.0:
        print(
            "warning: sphere initially intersects the contact margin by "
            f"{-initial_clearance:.4g}"
        )

    print(
        f"demo={args.demo} dt={args.dt:g} frames={args.frames} "
        f"tail={args.tail_frames}"
    )
    if args.demo == "drag":
        print(
            f"drag: axis={args.drag_axis} amp={args.drag_amp:g} "
            f"omega={args.drag_omega:g} rad/s"
        )
    else:
        print(
            f"twist: axis={args.twist_axis} amp={args.twist_amp:g} deg "
            f"omega={args.twist_omega:g} rad/s"
        )
    print(
        f"sphere: enabled={args.collision} center={np.round(sphere_center, 4)} "
        f"radius={sphere_radius:.4g} margin={args.contact_skin:g} "
        f"initial clearance={initial_clearance:.4g}"
    )

    pin_rest = V_p0[pinned].copy()
    pin_center = pin_rest.mean(axis=0)
    total_frames = args.frames + args.tail_frames

    def target_at(t: int, rest: np.ndarray, center: np.ndarray) -> np.ndarray:
        # The baseline increments time before each solve. Clamp the frame index
        # during the tail so its last prescribed target is held stationary.
        motion_frame = min(t, args.frames - 1)
        time = (motion_frame + 1) * args.dt
        return pin_targets(
            rest,
            center,
            demo=args.demo,
            time=time,
            drag_axis=args.drag_axis,
            drag_amp=args.drag_amp,
            drag_omega=args.drag_omega,
            twist_axis=args.twist_axis,
            twist_amp=args.twist_amp,
            twist_omega=args.twist_omega,
        )

    sphere_color = (0.65, 0.68, 0.75)

    def obstacles_at(_t: int) -> list[Obstacle]:
        if not args.collision:
            return []
        return [
            Obstacle(
                name="sphere",
                kind="sphere",
                center=sphere_center.copy(),
                radius=float(sphere_radius),
                color=sphere_color,
            )
        ]

    cache_path = Path(args.cache) if args.cache else None
    load_from_cache = (
        cache_path is not None and cache_path.exists() and not args.recompute
    )

    if load_from_cache:
        print(f"loading cache: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as z:
            X_p = _cache_array(
                z, "X_p", (total_frames, V_p0.shape[0], 3), cache_path,
            )
            V_recon = _cache_array(
                z,
                "V_recon",
                (total_frames, V_visual.shape[0], 3),
                cache_path,
            )
            V_full_sim = None
            if args.sim_full:
                V_full_sim = _cache_array(
                    z,
                    "V_full_sim",
                    (total_frames, V_visual.shape[0], 3),
                    cache_path,
                )
        obs_log = [obstacles_at(t) for t in range(total_frames)]
        print(
            f"  X_p={X_p.shape}  V_recon={V_recon.shape}"
            + (
                f"  V_full_sim={V_full_sim.shape}"
                if V_full_sim is not None
                else ""
            )
        )
    else:
        system = _make_system(
            V_p0,
            F_p,
            pinned,
            density=args.density,
            gravity=args.gravity,
            k_stretch=args.k_stretch,
            k_bend=args.k_bend,
            sphere_center=sphere_center,
            sphere_radius=sphere_radius,
            collision=args.collision,
        )

        def per_frame(t: int, current_system) -> None:
            current_system.X[pinned] = target_at(t, pin_rest, pin_center)

        print(f"simulating: {args.n_settle} settle + {total_frames} logged frames")
        X_p, obs_log = run_proxy_sim(
            system,
            total_frames,
            per_frame,
            obstacles_at,
            dt=args.dt,
            iters=args.iters,
            k_damp=args.k_damp,
            friction=args.friction,
            restitution=args.restitution,
            contact_skin=args.contact_skin,
            n_settle=args.n_settle,
            solver=args.solver,
        )
        print(
            f"  proxy y range over run: "
            f"[{X_p[..., 1].min():.3f}, {X_p[..., 1].max():.3f}]"
        )

        print("driving visual via LBS")
        V_recon = lbs_drive(s, B, V_visual, V_p0, X_p)
        print(f"  V_recon shape={V_recon.shape}")

        V_full_sim = None
        if args.sim_full:
            y_pin_threshold = float(V_p0[pinned, 1].min())
            pinned_full = np.where(V_visual[:, 1] >= y_pin_threshold)[0].astype(
                np.int64
            )
            if pinned_full.size == 0:
                raise SystemExit(
                    "no visual vertices are above the proxy pin threshold "
                    f"({y_pin_threshold:.4f}); check that the inputs share a frame"
                )
            pin_rest_full = V_visual[pinned_full].copy()
            pin_center_full = pin_rest_full.mean(axis=0)
            system_full = _make_system(
                V_visual,
                F_visual.astype(np.int64),
                pinned_full,
                density=args.density,
                gravity=args.gravity,
                k_stretch=args.full_k_stretch,
                k_bend=args.full_k_bend,
                sphere_center=sphere_center,
                sphere_radius=sphere_radius,
                collision=args.collision,
            )

            def per_frame_full(t: int, current_system) -> None:
                current_system.X[pinned_full] = target_at(
                    t, pin_rest_full, pin_center_full
                )

            full_contact_skin = (
                args.contact_skin
                if args.full_contact_skin is None
                else args.full_contact_skin
            )
            print(
                f"simulating full mesh: |V|={V_visual.shape[0]} "
                f"|pinned|={pinned_full.shape[0]}  {args.n_settle} settle + "
                f"{total_frames} logged frames"
            )
            V_full_sim, _ = run_proxy_sim(
                system_full,
                total_frames,
                per_frame_full,
                obstacles_at,
                dt=args.dt,
                iters=args.full_iters,
                k_damp=args.full_k_damp,
                friction=args.friction,
                restitution=args.restitution,
                contact_skin=full_contact_skin,
                n_settle=args.n_settle,
                solver=args.full_solver,
            )
            print(
                f"  full y range: [{V_full_sim[..., 1].min():.3f}, "
                f"{V_full_sim[..., 1].max():.3f}]"
            )

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            save_values = {"X_p": X_p, "V_recon": V_recon}
            if V_full_sim is not None:
                save_values["V_full_sim"] = V_full_sim
            np.savez(cache_path, **save_values)
            print(f"saved cache: {cache_path}")

    if args.no_viz:
        return

    from pag.eval_viz import show_eval

    show_eval(
        V_visual,
        F_visual,
        V_p0,
        F_p,
        X_p,
        V_recon,
        obs_log,
        V_full_sim=V_full_sim,
        fps=1.0 / args.dt,
        export_dir=args.export_dir,
    )


if __name__ == "__main__":
    main()
