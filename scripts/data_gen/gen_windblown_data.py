"""Generate train/test cloth-under-wind frame datasets with controllable
OOD splits in wind-condition space.

Wind model
----------
Each split is a SINGLE long simulation. At every frame, a fresh wind
condition ``θ = (azimuth, magnitude)`` is sampled (direction restricted
to horizontal, y-up); per-vertex force on free vertices is

    F_v(t) = magnitude(t) · d̂(azimuth(t)) + ε_v(t)

with ε a per-vertex Gaussian turbulence (std ``--turbulence-std``)
exponentially smoothed in time by ``--coherence``. Turbulence is white
in space, smoothed in time. Cloth state carries over from frame to
frame — so each frame's positions reflect the entire history of θ's
sampled before it (this is by design: the dataset is a continuous
trajectory through condition space, not i.i.d. snapshots).

This is a deliberate refactor of windblown_proxy.py: there, every
vertex got an INDEPENDENT random direction so the bulk wind across
the cloth was ~N(0, 1/√V) ≈ 0 and "direction" wasn't a meaningful
axis. Here, direction is a global parameter the cloth actually feels.

Train/test split
----------------
``--axis`` selects which condition components define the OOD boundary:

* ``direction`` — train/test get disjoint AZIMUTH ranges, fixed magnitude
* ``magnitude`` — train/test get disjoint MAGNITUDE ranges, fixed azimuth
* ``both``      — both axes split

Per-axis ranges have sensible defaults but every range is overridable
via ``--azimuth-train LO HI`` / ``--azimuth-test`` / ``--mag-train`` /
``--mag-test`` (degrees / force units). Setting ``LO == HI`` pins that
axis to a constant; setting the *same* range for train and test means
that axis varies freely in both sets (no split on that axis) — handy
for "both sets see all directions, only magnitude differs" experiments.

Deviation metrics
-----------------
Computed in θ-feature space (azimuth → (cos, sin) so the circle wraps
correctly; magnitude → min-max-normalized scalar). Reported:

* Nearest-neighbour distance test → train: min / median / max
* RBF MMD² with median-heuristic bandwidth

Outputs
-------
``<out>/train.npz``, ``<out>/test.npz``  — X (T, V, 3) f32, theta (T, 2),
                                           wind_global (T, 3),
                                           azimuth_deg (T,), magnitude (T,)
``<out>/mesh.npz``                        — V0, F, pinned
``<out>/metadata.json``                   — all CLI args + metrics
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pbd import Bend, Stretch, System, build_mesh, load_obj


# Per-axis defaults: (azimuth_train, azimuth_test, mag_train, mag_test).
# Azimuth in degrees, magnitude in same units as windblown_proxy.py's
# --wind-mean. The defaults give a clean OOD split in each mode.
DEFAULTS = {
    # Wedge holdout on azimuth, magnitude pinned at 4.0.
    "direction": dict(
        az_train=(0.0, 270.0), az_test=(270.0, 360.0),
        mag_train=(4.0, 4.0), mag_test=(4.0, 4.0),
    ),
    # Magnitude extrapolation (test stronger than train), azimuth pinned at +x.
    "magnitude": dict(
        az_train=(0.0, 0.0), az_test=(0.0, 0.0),
        mag_train=(1.0, 4.0), mag_test=(4.0, 6.0),
    ),
    # Both axes split.
    "both": dict(
        az_train=(0.0, 270.0), az_test=(270.0, 360.0),
        mag_train=(1.0, 4.0), mag_test=(4.0, 6.0),
    ),
}


def pick_top_fraction(V: np.ndarray, frac: float) -> np.ndarray:
    """Indices of the top ``frac`` of vertices, ranked by y. At least one."""
    n = V.shape[0]
    n_pin = max(1, int(np.ceil(frac * n)))
    return np.argsort(V[:, 1])[-n_pin:]


def azimuth_to_dir(az_rad: float) -> np.ndarray:
    """Horizontal unit vector (y-up). az=0 → +x, az=π/2 → +z."""
    return np.array([np.cos(az_rad), 0.0, np.sin(az_rad)], dtype=np.float64)


def thetas_to_wind_global(thetas: np.ndarray) -> np.ndarray:
    """(T, 2) θ → (T, 3) global-wind force vectors (horizontal, y-up)."""
    az, mag = thetas[:, 0], thetas[:, 1]
    return np.column_stack([
        mag * np.cos(az),
        np.zeros_like(mag),
        mag * np.sin(az),
    ])


# ----------------------------------------------------------- single trajectory


def simulate_varying(
    mesh,
    pinned_idx: np.ndarray,
    thetas: np.ndarray,
    dt: float,
    iters: int,
    k_stretch: float,
    k_bend: float,
    k_damp: float,
    turbulence_std: float,
    coherence: float,
    seed: int,
    gravity: tuple[float, float, float],
    solver: str,
    progress_label: str = "",
    progress_every: int = 50,
) -> np.ndarray:
    """Run ONE simulation of ``len(thetas)`` frames whose global wind
    direction and magnitude change per frame.

    ``thetas`` is (T, 2): each row is (azimuth_rad, magnitude) used for
    that one frame's external force. Cloth state and the turbulence EMA
    both carry across frames — this is a single continuous trajectory,
    not a batch of independent runs.

    Returns (T, V, 3) float32 vertex positions after each step.
    """
    sys = System.from_mesh(mesh, density=1.0, gravity=gravity)
    sys.add_constraint(Stretch.from_mesh(mesh, k=k_stretch))
    sys.add_constraint(Bend.from_mesh(mesh, k=k_bend))
    sys.pin(pinned_idx)

    rng = np.random.default_rng(seed)
    turb_state = np.zeros((mesh.n_verts, 3), dtype=np.float64)
    free = sys.W > 0.0

    n_frames = thetas.shape[0]
    out = np.zeros((n_frames, mesh.n_verts, 3), dtype=np.float32)

    t_total = time.time()
    for t in range(n_frames):
        az, mag = float(thetas[t, 0]), float(thetas[t, 1])
        global_force = mag * azimuth_to_dir(az)

        # Turbulence EMA — stateful across frames so coherence>0 gives
        # gusts that span multiple frames even though θ is i.i.d. per frame.
        kick = turbulence_std * rng.standard_normal(turb_state.shape)
        turb_state[:] = (
            coherence * turb_state + (1.0 - coherence) * kick
        )

        F = global_force[None, :] + turb_state
        sys.V[free] += dt * sys.W[free, None] * F[free]
        # sys.V[free] += dt * F[free]
        sys.step(dt=dt, iters=iters, k_damp=k_damp, solver=solver)
        out[t] = sys.X

        if progress_label and (t + 1) % progress_every == 0:
            elapsed = time.time() - t_total
            sim_fps = (t + 1) / elapsed
            print(f"  {progress_label}: frame {t+1:5d}/{n_frames}  "
                  f"({sim_fps:5.1f} sim-fps)")
    if progress_label:
        print(f"  {progress_label} total: {time.time() - t_total:.1f}s")
    return out


# --------------------------------------------------------------- θ utilities


def sample_thetas(
    n: int,
    az_range_deg: tuple[float, float],
    mag_range: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n θ = (azimuth_rad, magnitude) uniformly in given boxes.

    A range with LO == HI degenerates to a constant, which is what we
    want when the user pins that axis."""
    az_deg = rng.uniform(az_range_deg[0], az_range_deg[1], size=n)
    mag = rng.uniform(mag_range[0], mag_range[1], size=n)
    return np.column_stack([np.deg2rad(az_deg), mag])


def features(
    theta_train: np.ndarray,
    theta_test: np.ndarray,
    axis: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Map θ to a Euclidean feature space for distance metrics.

    * Azimuth → (cos, sin) so the circle wraps correctly (357° ≈ 3°).
    * Magnitude → min-max normalized to [0, 1] over (train ∪ test) so
      its contribution is commensurate with the (cos, sin) pair.

    The ``axis`` argument selects which components enter the feature
    vector — components held *constant* across train/test would just
    inflate dimension without contributing to the test↔train distance,
    so we drop them."""
    az_train, mag_train = theta_train[:, 0], theta_train[:, 1]
    az_test, mag_test = theta_test[:, 0], theta_test[:, 1]

    parts_train, parts_test = [], []
    if axis in ("direction", "both"):
        parts_train += [np.cos(az_train), np.sin(az_train)]
        parts_test += [np.cos(az_test), np.sin(az_test)]
    if axis in ("magnitude", "both"):
        all_mag = np.concatenate([mag_train, mag_test])
        m_lo, m_hi = float(all_mag.min()), float(all_mag.max())
        scale = max(m_hi - m_lo, 1e-12)
        parts_train.append((mag_train - m_lo) / scale)
        parts_test.append((mag_test - m_lo) / scale)
    return (
        np.column_stack(parts_train),
        np.column_stack(parts_test),
    )


def nn_distance(A: np.ndarray, B: np.ndarray) -> tuple[float, float, float]:
    """For each row of B, distance to its nearest neighbour in A.
    Returns (min, median, max) of those nearest-neighbour distances."""
    d = np.linalg.norm(B[:, None] - A[None, :], axis=-1)  # (nB, nA)
    nn = d.min(axis=1)
    return float(nn.min()), float(np.median(nn)), float(nn.max())


def rbf_mmd_squared(X: np.ndarray, Y: np.ndarray) -> float:
    """Biased squared MMD with RBF kernel; bandwidth from median heuristic.

    MMD² = E[k(x,x')] + E[k(y,y')] - 2 E[k(x,y)]. With RBF, MMD² ≥ 0
    and equals 0 iff the distributions are identical (in the universal-
    kernel sense)."""
    Z = np.vstack([X, Y])
    diff = Z[:, None, :] - Z[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)                     # (n+m, n+m)
    triu = d2[np.triu_indices_from(d2, k=1)]
    sig2 = float(np.median(triu)) if triu.size and triu.max() > 0 else 1.0
    K = np.exp(-d2 / (2.0 * sig2))
    n, m = X.shape[0], Y.shape[0]
    Kxx = K[:n, :n]
    Kyy = K[n:, n:]
    Kxy = K[:n, n:]
    return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())


# ---------------------------------------------------------------- playback


def play_dataset(
    X_train: np.ndarray,
    X_test: np.ndarray,
    theta_train: np.ndarray,
    theta_test: np.ndarray,
    F: np.ndarray,
    pinned: np.ndarray,
):
    """Polyscope replay UI: pick split + frame, scrub or play.

    The dataset is already on disk by the time this runs; nothing here
    re-simulates. Frames just stream out of the in-memory ``X_*`` arrays
    so playback is realtime regardless of mesh size.
    """
    import polyscope as ps
    import polyscope.imgui as psim

    if not getattr(ps, "_pbd_initialized", False):
        ps.init()
        ps._pbd_initialized = True
    ps.set_up_dir("y_up")
    ps.set_front_dir("z_front")
    ps.set_ground_plane_mode("none")

    F_int = np.ascontiguousarray(F, dtype=np.int64)
    state = {"split": "train", "frame": 0, "playing": True}

    def src() -> tuple[np.ndarray, np.ndarray]:
        return ((X_train, theta_train) if state["split"] == "train"
                else (X_test, theta_test))

    Xs0, _ = src()
    mesh = ps.register_surface_mesh("cloth", Xs0[0], F_int, edge_width=1.0)
    mesh.set_smooth_shade(True)
    pin_pc = ps.register_point_cloud("pinned", Xs0[0, pinned])
    pin_pc.set_radius(0.012, relative=False)
    pin_pc.set_color((0.9, 0.2, 0.2))

    def tick():
        Xs, ths = src()
        T = Xs.shape[0]
        if state["playing"]:
            state["frame"] = (state["frame"] + 1) % T
        if state["frame"] >= T:
            state["frame"] = T - 1
        Xf = Xs[state["frame"]]
        mesh.update_vertex_positions(Xf)
        pin_pc.update_point_positions(Xf[pinned])

        # ---- UI
        if psim.RadioButton("Train", state["split"] == "train"):
            if state["split"] != "train":
                state["split"] = "train"
                state["frame"] = 0
        psim.SameLine()
        if psim.RadioButton("Test", state["split"] == "test"):
            if state["split"] != "test":
                state["split"] = "test"
                state["frame"] = 0

        _, state["playing"] = psim.Checkbox("Play", state["playing"])
        _, state["frame"] = psim.SliderInt(
            f"frame (0..{T - 1})", int(state["frame"]), 0, T - 1,
        )

        th = ths[state["frame"]]
        az_deg = float(np.rad2deg(float(th[0])))
        mag = float(th[1])
        wx, wz = mag * float(np.cos(th[0])), mag * float(np.sin(th[0]))
        psim.Text(f"split: {state['split']}   "
                  f"az: {az_deg:6.1f}°   mag: {mag:5.2f}")
        psim.Text(f"wind vector: ({wx:+.2f}, 0.00, {wz:+.2f})")

    ps.set_user_callback(tick)
    print("\nopening playback viewer; close the window to exit")
    ps.show()


# --------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obj", required=True,
                    help="OBJ mesh to use as the cloth/proxy")
    ap.add_argument("--out", default="data/wind_dataset",
                    help="output directory (created if missing)")
    ap.add_argument("--axis", choices=["direction", "magnitude", "both"], required=True,
                    help="which condition axis defines the train/test split. "
                         "Per-axis defaults can be overridden with the range "
                         "flags below.")

    ap.add_argument("--train-frames", type=int, default=600,
                    help="number of frames in the train split (one continuous "
                         "simulation)")
    ap.add_argument("--test-frames", type=int, default=200,
                    help="number of frames in the test split (one continuous "
                         "simulation)")

    # Range overrides (None ⇒ use --axis defaults).
    ap.add_argument("--azimuth-train", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="azimuth range (deg) for train; LO==HI pins it")
    ap.add_argument("--azimuth-test", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="azimuth range (deg) for test; LO==HI pins it")
    ap.add_argument("--mag-train", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="magnitude range for train; LO==HI pins it")
    ap.add_argument("--mag-test", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="magnitude range for test; LO==HI pins it")

    # Wind hyperparameters held constant across both splits.
    ap.add_argument("--turbulence-std", type=float, default=1.0,
                    help="std of per-vertex turbulence force (units of force)")
    ap.add_argument("--coherence", type=float, default=0.95,
                    help="EMA coherence on turbulence (0=white, 1=frozen)")

    # Cloth dynamics — defaults match windblown_proxy.py.
    ap.add_argument("--pin-fraction", type=float, default=0.10)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--dt", type=float, default=1.0 / 60)
    ap.add_argument("--k-stretch", type=float, default=0.99)
    ap.add_argument("--k-bend", type=float, default=0.1)
    ap.add_argument("--k-damp", type=float, default=0.05)
    ap.add_argument("--gravity", type=float, nargs=3, default=[0.0, -9.81, 0.0])
    ap.add_argument("--solver", choices=["jacobi", "gauss-seidel"], default="jacobi")

    ap.add_argument("--seed", type=int, default=0,
                    help="master seed; θ sampling and turbulence RNG seeds "
                         "are derived from it deterministically.")
    ap.add_argument("--viz", action="store_true",
                    help="after generation, open a polyscope replay window. "
                         "Pick a split + frame from the imgui sidebar; θ and "
                         "the global wind vector are shown alongside.")
    args = ap.parse_args()

    # Resolve per-axis range defaults if user didn't override.
    d = DEFAULTS[args.axis]
    az_train = tuple(args.azimuth_train) if args.azimuth_train else d["az_train"]
    az_test = tuple(args.azimuth_test) if args.azimuth_test else d["az_test"]
    mag_train = tuple(args.mag_train) if args.mag_train else d["mag_train"]
    mag_test = tuple(args.mag_test) if args.mag_test else d["mag_test"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- mesh + pinning
    Vobj, Fobj = load_obj(args.obj)
    mesh = build_mesh(Vobj, Fobj)
    pinned = pick_top_fraction(mesh.V, args.pin_fraction)
    print(f"mesh: V={mesh.n_verts}  F={mesh.n_faces}  pinned={len(pinned)} "
          f"(top {100*args.pin_fraction:.1f}%)")

    # ---------- per-frame θ sampling + per-split turbulence seeds
    rng = np.random.default_rng(args.seed)
    theta_train = sample_thetas(args.train_frames, az_train, mag_train, rng)
    theta_test = sample_thetas(args.test_frames, az_test, mag_test, rng)
    seed_train = int(rng.integers(0, 2**31 - 1))
    seed_test = int(rng.integers(0, 2**31 - 1))

    # ---------- summary BEFORE simulating, so user can ctrl-C if ranges look wrong
    print(f"\naxis={args.axis}  train_frames={args.train_frames}  "
          f"test_frames={args.test_frames}  dt={args.dt}")
    print(f"  train: az ∈ [{az_train[0]:.1f}°, {az_train[1]:.1f}°]   "
          f"mag ∈ [{mag_train[0]:.2f}, {mag_train[1]:.2f}]")
    print(f"  test:  az ∈ [{az_test[0]:.1f}°, {az_test[1]:.1f}°]   "
          f"mag ∈ [{mag_test[0]:.2f}, {mag_test[1]:.2f}]")
    print(f"  turbulence_std={args.turbulence_std}  coherence={args.coherence}")

    # Estimate dataset size up front (float32, 4 bytes/element).
    bytes_per_split = lambda n: n * mesh.n_verts * 3 * 4
    print(f"  estimated size: train {bytes_per_split(args.train_frames)/1e6:.0f} MB, "
          f"test {bytes_per_split(args.test_frames)/1e6:.0f} MB")

    # ---------- simulate each split (one continuous run per split)
    print("\ngenerating train ...")
    X_train = simulate_varying(
        mesh, pinned, theta_train, args.dt, args.iters,
        args.k_stretch, args.k_bend, args.k_damp,
        args.turbulence_std, args.coherence, seed_train,
        tuple(args.gravity), args.solver, progress_label="train",
    )
    print("generating test ...")
    X_test = simulate_varying(
        mesh, pinned, theta_test, args.dt, args.iters,
        args.k_stretch, args.k_bend, args.k_damp,
        args.turbulence_std, args.coherence, seed_test,
        tuple(args.gravity), args.solver, progress_label="test",
    )

    wind_train = thetas_to_wind_global(theta_train)
    wind_test = thetas_to_wind_global(theta_test)

    # ---------- deviation metrics in θ-feature space
    F_train, F_test = features(theta_train, theta_test, args.axis)
    nn_min, nn_med, nn_max = nn_distance(F_train, F_test)
    mmd2 = rbf_mmd_squared(F_train, F_test)

    # Domain-classifier sanity (1-NN leave-one-out on the joint set).
    # ~0.5 accuracy ⇒ leaky split; ~1.0 ⇒ cleanly disjoint.
    Z = np.vstack([F_train, F_test])
    y = np.concatenate([np.zeros(len(F_train)), np.ones(len(F_test))])
    d_full = np.linalg.norm(Z[:, None] - Z[None, :], axis=-1)
    np.fill_diagonal(d_full, np.inf)
    pred = y[d_full.argmin(axis=1)]
    nn_class_acc = float(np.mean(pred == y))

    # ---------- save
    np.savez(
        out_dir / "train.npz",
        X=X_train,
        theta=theta_train.astype(np.float64),
        wind_global=wind_train.astype(np.float64),
        azimuth_deg=np.rad2deg(theta_train[:, 0]),
        magnitude=theta_train[:, 1].astype(np.float64),
    )
    np.savez(
        out_dir / "test.npz",
        X=X_test,
        theta=theta_test.astype(np.float64),
        wind_global=wind_test.astype(np.float64),
        azimuth_deg=np.rad2deg(theta_test[:, 0]),
        magnitude=theta_test[:, 1].astype(np.float64),
    )
    np.savez(
        out_dir / "mesh.npz",
        V0=mesh.V.astype(np.float64),
        F=mesh.F,
        pinned=pinned.astype(np.int64),
    )
    metadata = {
        "obj": str(args.obj),
        "n_verts": int(mesh.n_verts),
        "n_faces": int(mesh.n_faces),
        "n_pinned": int(len(pinned)),
        "pin_fraction": args.pin_fraction,
        "axis": args.axis,
        "train_frames": args.train_frames,
        "test_frames": args.test_frames,
        "dt": args.dt,
        "ranges": {
            "azimuth_train_deg": list(az_train),
            "azimuth_test_deg": list(az_test),
            "mag_train": list(mag_train),
            "mag_test": list(mag_test),
        },
        "wind_hyperparams": {
            "turbulence_std": args.turbulence_std,
            "coherence": args.coherence,
        },
        "physics": {
            "k_stretch": args.k_stretch,
            "k_bend": args.k_bend,
            "k_damp": args.k_damp,
            "iters": args.iters,
            "solver": args.solver,
            "gravity": list(args.gravity),
        },
        "seed": args.seed,
        "split_seeds": {"train": seed_train, "test": seed_test},
        "metrics": {
            "feature_space": {
                "direction": "(cos az, sin az)",
                "magnitude": "(magnitude min-max-normalized to [0,1])",
                "both": "(cos az, sin az, mag min-max-normalized)",
            }[args.axis],
            "feature_dim": int(F_train.shape[1]),
            "nn_distance_test_to_train": {
                "min": nn_min, "median": nn_med, "max": nn_max,
            },
            "rbf_mmd_squared": mmd2,
            "domain_classifier_1nn_accuracy": nn_class_acc,
        },
    }
    with (out_dir / "metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\nWrote {out_dir}/")
    print(f"  train.npz  X {X_train.shape} ({X_train.nbytes/1e6:.1f} MB)")
    print(f"  test.npz   X {X_test.shape} ({X_test.nbytes/1e6:.1f} MB)")
    print(f"  mesh.npz   V {mesh.V.shape} F {mesh.F.shape}")
    print(f"  metadata.json")
    print(f"\nDeviation in θ-feature space "
          f"({metadata['metrics']['feature_space']}):")
    print(f"  NN distance test→train:  min={nn_min:.4f}  "
          f"median={nn_med:.4f}  max={nn_max:.4f}")
    print(f"  RBF MMD² (median bw):    {mmd2:.4f}")
    print(f"  1-NN domain classifier:  {nn_class_acc:.3f}  "
          f"(0.5 = overlapping, 1.0 = cleanly disjoint)")

    if args.viz:
        play_dataset(X_train, X_test, theta_train, theta_test, mesh.F, pinned)


if __name__ == "__main__":
    main()
