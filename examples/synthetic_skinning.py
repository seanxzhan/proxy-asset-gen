"""Standalone Stage 2 demo on a synthetic cylinder — no PBD dependency.

A two-layer cylinder = M_visual; a single-layer cylinder = M_proxy. Generate
20 frames of a known proxy bend in pure numpy (no pre-simulation), optimize
skinning weights, and (interactively) show the skinned visual tracking the
proxy in polyscope.

Examples
--------
$ python examples/synthetic_skinning.py --smoke
$ python examples/synthetic_skinning.py
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from pag import AnimationData, optimize_skinning_weights


def _cylinder_mesh(
    n_ring: int, n_height: int, radius: float, height: float, layers: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """A simple ribbed cylinder. layers=2 produces two concentric cylinders
    (inner+outer with a small radius offset) — a stand-in for a "double-cover"
    visual mesh."""
    V_list = []
    F_list = []
    for layer in range(layers):
        r = radius * (1.0 + 0.04 * layer)
        for j in range(n_height):
            z = (j / max(n_height - 1, 1) - 0.5) * height
            for i in range(n_ring):
                a = 2 * np.pi * i / n_ring
                V_list.append([r * np.cos(a), z, r * np.sin(a)])

    layer_size = n_ring * n_height
    for layer in range(layers):
        off = layer * layer_size
        for j in range(n_height - 1):
            for i in range(n_ring):
                ip = (i + 1) % n_ring
                a = off + j * n_ring + i
                b = off + j * n_ring + ip
                c = off + (j + 1) * n_ring + i
                d = off + (j + 1) * n_ring + ip
                F_list.append([a, b, d])
                F_list.append([a, d, c])
    return np.asarray(V_list, dtype=np.float64), np.asarray(F_list, dtype=np.int64)


def _make_bend_frames(V_proxy: np.ndarray, T: int = 20) -> np.ndarray:
    """Bend the cylinder around the +y axis; tip-to-base sin sweep."""
    z_min, z_max = V_proxy[:, 1].min(), V_proxy[:, 1].max()
    h = z_max - z_min
    frames = np.zeros((T, V_proxy.shape[0], 3), dtype=np.float32)
    for t in range(T):
        phase = t / max(T - 1, 1)
        amp = 0.4 * np.sin(2 * np.pi * phase)
        z = (V_proxy[:, 1] - z_min) / h
        dx = amp * z ** 2
        frames[t] = V_proxy.astype(np.float32) + np.stack(
            [dx, np.zeros_like(dx), np.zeros_like(dx)], axis=1
        ).astype(np.float32)
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--k-B", type=int, default=4)
    ap.add_argument("--k-K", type=int, default=4)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--smoke", action="store_true",
                    help="Headless run: print stats and exit.")
    args = ap.parse_args()

    print("building synthetic cylinder ...")
    V_visual, F_visual = _cylinder_mesh(n_ring=24, n_height=12, radius=1.0, height=2.0, layers=1)
    V_proxy, F_proxy = _cylinder_mesh(n_ring=12, n_height=8, radius=1.0, height=2.0, layers=1)
    print(f"  M_visual: |V|={V_visual.shape[0]}  |F|={F_visual.shape[0]}")
    print(f"  M_proxy : |V|={V_proxy.shape[0]}   |F|={F_proxy.shape[0]}")

    frames = _make_bend_frames(V_proxy, T=20)
    train = AnimationData(
        V_proxy_rest=V_proxy, F_proxy=F_proxy,
        pinned=np.array([], dtype=np.int64),
        X=frames, theta=np.zeros((20, 2)),
        wind_global=np.zeros((20, 3)),
        azimuth_deg=np.zeros(20), magnitude=np.zeros(20),
    )

    t0 = time.perf_counter()
    print(f"\noptimizing skinning weights ({args.epochs} epochs) ...")
    res = optimize_skinning_weights(
        V_visual, F_visual, train,
        k_B=args.k_B, k_K=args.k_K,
        epochs=args.epochs, lr=args.lr,
        device=args.device, seed=0, verbose=True, log_every=10,
    )
    dt = time.perf_counter() - t0
    print()
    print(f"  L_total: {res.losses['total'][0]:.3e} -> {res.losses['total'][-1]:.3e}")
    print(f"  total runtime: {dt:.2f}s")
    print(f"  W shape: {res.W.shape}  nnz={res.W.nnz}")

    if args.smoke:
        return

    import polyscope as ps
    from pag.skinning_lbs import simplified_lbs
    import torch

    s = torch.tensor(res.s)
    B = torch.tensor(res.B, dtype=torch.long)
    V_v0 = torch.tensor(V_visual, dtype=torch.float32)
    V_p0 = torch.tensor(V_proxy, dtype=torch.float32)
    X = torch.tensor(frames, dtype=torch.float32)
    with torch.no_grad():
        V_recon = simplified_lbs(s, B, V_v0, V_p0, X).numpy()

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    diag = float(np.linalg.norm(V_visual.max(0) - V_visual.min(0)))
    spacing = 1.5 * diag

    proxy_curve = ps.register_surface_mesh(
        "M_proxy(t)", V_proxy + np.array([0, 0, 0]), F_proxy, color=(0.45, 0.65, 0.85),
    )
    visual_recon = ps.register_surface_mesh(
        "M_visual_recon(t)", V_visual + np.array([spacing, 0, 0]), F_visual,
        color=(0.90, 0.45, 0.45),
    )
    ps.register_surface_mesh(
        "M_visual rest", V_visual + np.array([2 * spacing, 0, 0]), F_visual,
        color=(0.65, 0.65, 0.65), transparency=0.4,
    )

    state = {"frame": 0}

    def callback() -> None:
        import polyscope.imgui as psim
        changed, state["frame"] = psim.SliderInt("frame", state["frame"], 0, frames.shape[0] - 1)
        if changed or psim.Button("step"):
            f = state["frame"]
            proxy_curve.update_vertex_positions(frames[f] + np.array([0, 0, 0]))
            visual_recon.update_vertex_positions(V_recon[f] + np.array([spacing, 0, 0]))

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
