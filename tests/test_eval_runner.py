"""Unit tests for pag.eval_runner — the scenario-evaluation precompute loop."""
from __future__ import annotations

import numpy as np
import pytest

from pag.eval_runner import Obstacle, lbs_drive, run_proxy_sim


def _grid_mesh(n: int = 4, side: float = 1.0):
    """Flat grid in the xz plane at y=1.0, returned as (V, F) for pbd.build_mesh."""
    xs = np.linspace(-side, side, n)
    zs = np.linspace(-side, side, n)
    XX, ZZ = np.meshgrid(xs, zs, indexing="xy")
    V = np.stack(
        [XX.ravel(), np.ones(n * n, dtype=np.float64), ZZ.ravel()], axis=1,
    )
    F = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b = j * n + (i + 1)
            c = (j + 1) * n + i
            d = (j + 1) * n + (i + 1)
            F.extend([[a, b, d], [a, d, c]])
    return V, np.asarray(F, dtype=np.int64)


def test_run_proxy_sim_shapes_and_callbacks():
    """run_proxy_sim must return (T, N_p, 3) f32, call per_frame T times,
    call obstacles_at T times. Settle frames must NOT trigger per_frame."""
    import pbd

    V, F = _grid_mesh(n=4)
    mesh = pbd.build_mesh(V, F)
    sys = pbd.System.from_mesh(mesh, density=1.0, gravity=(0., -9.81, 0.))
    sys.add_constraint(pbd.Stretch.from_mesh(mesh, k=0.99))
    sys.pin([0])  # pin one corner so it doesn't fly off into infinity

    per_frame_calls: list[int] = []
    obstacles_calls: list[int] = []

    def per_frame(t, _sys):
        per_frame_calls.append(t)

    def obstacles_at(t):
        obstacles_calls.append(t)
        return [Obstacle(name="probe", kind="sphere",
                         center=np.zeros(3), radius=0.1)]

    n_frames = 5
    n_settle = 3
    X_p, obs = run_proxy_sim(
        sys, n_frames, per_frame, obstacles_at, n_settle=n_settle,
    )

    assert X_p.shape == (n_frames, V.shape[0], 3)
    assert X_p.dtype == np.float32
    assert per_frame_calls == list(range(n_frames))   # not called during settle
    assert obstacles_calls == list(range(n_frames))
    assert len(obs) == n_frames
    assert obs[0][0].name == "probe"


def test_lbs_drive_identity_recovers_proxy_motion():
    """Visual = proxy, B(i) = i, all weight on bone 0 → V_recon == X_p exactly.

    This pins the contract that scenarios test: under identity weights, the
    visual mesh follows the proxy frame-for-frame, so any deviation is purely
    LBS approximation error and not eval-pipeline noise.
    """
    rng = np.random.default_rng(0)
    n, T = 8, 5
    V0 = rng.standard_normal((n, 3))
    X_p = rng.standard_normal((T, n, 3)).astype(np.float32)

    s = np.zeros((n, 1), dtype=np.float32)
    s[:, 0] = 1.0
    B = np.arange(n, dtype=np.int64)[:, None]   # (n, 1)

    V_recon = lbs_drive(s, B, V0, V0, X_p)
    assert V_recon.shape == (T, n, 3)
    np.testing.assert_allclose(V_recon, X_p, atol=1e-6, rtol=0)


def test_run_proxy_sim_per_frame_can_mutate_collider():
    """Mutating a collider in per_frame must affect the next step (proves the
    moving_sphere scenario's design works: keep a reference, mutate .center)."""
    import pbd

    V, F = _grid_mesh(n=4)
    mesh = pbd.build_mesh(V, F)
    sys = pbd.System.from_mesh(mesh, density=1.0, gravity=(0., -9.81, 0.))
    sys.add_constraint(pbd.Stretch.from_mesh(mesh, k=0.99))
    sys.pin([0, 3, 12, 15])  # pin all four corners — cloth hangs taut

    sphere = pbd.Sphere(center=np.array([10.0, 0.5, 0.0]), radius=0.2)
    sys.add_collider(sphere)

    centers_seen: list[np.ndarray] = []

    def per_frame(t, _sys):
        # Walk the sphere along -x.
        sphere.center[:] = [10.0 - 0.1 * t, 0.5, 0.0]
        centers_seen.append(sphere.center.copy())

    X_p, _ = run_proxy_sim(
        sys, n_frames=5, per_frame_fn=per_frame,
        obstacles_at=lambda t: [], n_settle=2,
    )
    assert X_p.shape == (5, V.shape[0], 3)
    assert len(centers_seen) == 5
    # Each per_frame call left the sphere at a different x coord.
    xs = [c[0] for c in centers_seen]
    assert xs == sorted(xs, reverse=True)
