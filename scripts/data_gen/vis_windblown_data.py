"""Real-time PBD cloth in a refactored wind field, with live UI knobs.

Same scene shape as windblown_proxy.py — load (or grid-generate) a
triangle mesh, pin the topmost slice of vertices, run cloth dynamics
under gravity + wind. The difference is in the wind model:

    F_v(t) = magnitude · d̂(azimuth) + ε_v(t)

where ε is per-vertex Gaussian noise smoothed in time by an EMA on
``coherence``. windblown_proxy.py drew an INDEPENDENT random direction
at every vertex, so the bulk wind across the cloth was ~N(0, 1/√V) ≈ 0
and "direction" wasn't a meaningful parameter. Here the global wind
vector is a single 3-D thing the cloth actually feels, with turbulence
on top as a perturbation. This is the same wind model windblown_data_gen.py
uses to produce datasets — this script is the live, interactive twin.

Live UI knobs (in addition to Viewer's stiffness / damp / friction):

* azimuth (deg)   — global wind direction in the xz plane (0 = +x, 90 = +z)
* magnitude       — global wind force scale
* turbulence_std  — std of per-vertex Gaussian noise force
* coherence       — EMA smoothing on turbulence (0=white, 1=frozen)

A short green line at the cloth centroid points along the wind, length
∝ magnitude — useful for sanity-checking the direction slider.
"""
from __future__ import annotations

import argparse

import numpy as np

from pbd import Bend, Stretch, System, build_mesh, load_obj
from pbd.viz import Viewer


def make_default_mesh(n: int = 20, side: float = 1.0):
    """Cloth-style grid in the xy-plane (z=0). y is up."""
    xs = np.linspace(-side, side, n)
    ys = np.linspace(-side, side, n)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    V = np.stack([XX.ravel(), YY.ravel(), np.zeros(n * n)], axis=1)
    F = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b = j * n + (i + 1)
            c = (j + 1) * n + i
            d = (j + 1) * n + (i + 1)
            F.extend([[a, b, d], [a, d, c]])
    return build_mesh(V, np.array(F, dtype=np.int64))


def pick_top_fraction(V: np.ndarray, frac: float) -> np.ndarray:
    """Indices of the top ``frac`` of vertices, ranked by y."""
    n = V.shape[0]
    n_pin = max(1, int(np.ceil(frac * n)))
    return np.argsort(V[:, 1])[-n_pin:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default=None,
                    help="OBJ to load; default = 20x20 cloth grid")
    ap.add_argument("--n", type=int, default=20,
                    help="default grid resolution when --obj is not set")
    ap.add_argument("--pin-fraction", type=float, default=0.10)

    # Initial values for the wind sliders.
    ap.add_argument("--azimuth", type=float, default=90.0,
                    help="initial wind azimuth (deg). 0 = +x, 90 = +z. "
                         "Default 90° pushes perpendicular to the default "
                         "cloth plane for an obvious response.")
    ap.add_argument("--magnitude", type=float, default=4.0,
                    help="initial global wind magnitude")
    ap.add_argument("--turbulence-std", type=float, default=1.0,
                    help="initial per-vertex turbulence std")
    ap.add_argument("--coherence", type=float, default=0.95,
                    help="initial EMA coherence on turbulence (0=white, 1=frozen)")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--dt", type=float, default=1.0 / 60)
    ap.add_argument("--k-stretch", type=float, default=0.99)
    ap.add_argument("--k-bend", type=float, default=0.3)
    ap.add_argument("--k-damp", type=float, default=0.05)
    ap.add_argument("--solver", choices=["jacobi", "gauss-seidel"], default="jacobi")
    args = ap.parse_args()

    if args.obj:
        V, F = load_obj(args.obj)
        mesh = build_mesh(V, F)
    else:
        mesh = make_default_mesh(args.n)

    sys = System.from_mesh(mesh, density=1.0, gravity=(0.0, -9.81, 0.0))
    sys.add_constraint(Stretch.from_mesh(mesh, k=args.k_stretch))
    sys.add_constraint(Bend.from_mesh(mesh, k=args.k_bend))

    pinned = pick_top_fraction(mesh.V, args.pin_fraction)
    sys.pin(pinned)
    print(f"verts={mesh.n_verts}  faces={mesh.n_faces}  "
          f"pinned={len(pinned)} (top {100*args.pin_fraction:.1f}%)")

    rng = np.random.default_rng(args.seed)
    free = sys.W > 0.0

    # Mutable wind state — the UI sliders write into this dict directly,
    # and apply_wind reads it every step. That way dragging the slider
    # propagates immediately to the next physics step (max 1-frame lag).
    wind = {
        "azimuth_deg": float(args.azimuth),
        "magnitude": float(args.magnitude),
        "turbulence_std": float(args.turbulence_std),
        "coherence": float(args.coherence),
    }
    turb_state = np.zeros_like(sys.X)

    def apply_wind():
        az = np.deg2rad(wind["azimuth_deg"])
        direction = np.array([np.cos(az), 0.0, np.sin(az)])
        global_force = wind["magnitude"] * direction
        # Per-vertex turbulence with EMA coherence.
        kick = wind["turbulence_std"] * rng.standard_normal(turb_state.shape)
        coh = wind["coherence"]
        turb_state[:] = coh * turb_state + (1.0 - coh) * kick
        F_total = global_force[None, :] + turb_state
        sys.V[free] += args.dt * sys.W[free, None] * F_total[free]
        # sys.V[free] += args.dt * F_total[free]

    viewer = Viewer(sys, mesh.F, name="cloth")

    import polyscope as ps
    import polyscope.imgui as psim

    # Pinned verts highlighted in red so the boundary condition is visible.
    pin_pc = ps.register_point_cloud("pinned", sys.X[pinned])
    pin_pc.set_radius(0.012, relative=False)
    pin_pc.set_color((0.9, 0.2, 0.2))

    # Wind direction indicator: a line from the cloth centroid pointing
    # along the wind, length scaled by magnitude.
    wind_curve = ps.register_curve_network(
        "wind", np.zeros((2, 3)), np.array([[0, 1]], dtype=np.int64),
    )
    wind_curve.set_radius(0.012, relative=False)
    wind_curve.set_color((0.1, 0.9, 0.3))

    def update_wind_arrow():
        centroid = sys.X[free].mean(axis=0) if free.any() else sys.X.mean(axis=0)
        az = np.deg2rad(wind["azimuth_deg"])
        direction = np.array([np.cos(az), 0.0, np.sin(az)])
        length = wind["magnitude"] * 0.1
        wind_curve.update_node_positions(
            np.stack([centroid, centroid + direction * length])
        )

    def on_step(_frame):
        # Wind kick is hooked into on_step (post-step) following
        # windblown_proxy.py — there's a 1-frame lag, imperceptible
        # at 60 Hz. Same place we refresh the visual wind arrow.
        apply_wind()
        update_wind_arrow()

    def on_ui():
        _, wind["azimuth_deg"] = psim.SliderFloat(
            "azimuth (deg)", float(wind["azimuth_deg"]), 0.0, 360.0,
        )
        _, wind["magnitude"] = psim.SliderFloat(
            "magnitude", float(wind["magnitude"]), 0.0, 10.0,
        )
        _, wind["turbulence_std"] = psim.SliderFloat(
            "turbulence std", float(wind["turbulence_std"]), 0.0, 3.0,
        )
        _, wind["coherence"] = psim.SliderFloat(
            "coherence", float(wind["coherence"]), 0.0, 1.0,
        )
        az = float(wind["azimuth_deg"])
        mag = float(wind["magnitude"])
        ar = np.deg2rad(az)
        psim.Text(f"wind vector: ({mag*np.cos(ar):+.2f}, 0.00, {mag*np.sin(ar):+.2f})")

    viewer.run(
        dt=args.dt, iters=args.iters, k_damp=args.k_damp, solver=args.solver,
        on_step=on_step, on_ui=on_ui,
    )


if __name__ == "__main__":
    main()
