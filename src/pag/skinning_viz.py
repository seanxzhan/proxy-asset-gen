"""Shared polyscope viewer for Stage 2 skinning results.

Lays out four side-by-side meshes along +x:

  x = 0      M_proxy(t)            — animated proxy
  x = sp     M_visual_recon(t)     — LBS reconstruction of the visual mesh
  x = 2*sp   M_visual rest         — faded reference of the un-deformed visual
  x = 3*sp   M_visual weights      — visual mesh colored by w(:, j) for the
                                      currently selected proxy vertex j

UI:
  - "train" / "test" toggle (test only when test frames are available)
  - "play" / "pause" toggle + fps slider (auto-advances the frame slider)
  - frame slider
  - bone-j slider that re-paints the weights mesh and moves a green dot to
    proxy vertex j (so you can see which proxy vertex you're inspecting)
"""
from __future__ import annotations

import time

import numpy as np


def show_skinning(
    V_visual: np.ndarray,
    F_visual: np.ndarray,
    V_p0: np.ndarray,
    F_proxy: np.ndarray,
    X_train: np.ndarray,
    V_recon_train: np.ndarray,
    W,                                 # scipy.sparse or dense (N_v, N_p)
    *,
    X_test: np.ndarray | None = None,
    V_recon_test: np.ndarray | None = None,
) -> None:
    import polyscope as ps
    import polyscope.imgui as psim

    n_train = X_train.shape[0]
    n_test = 0 if X_test is None else X_test.shape[0]
    n_p = V_p0.shape[0]

    # The weights viz needs random column access; densify once. For the paper
    # scale (N_v ≤ ~50k, N_p = 128) this is a few MB — well within budget.
    W_dense = np.asarray(W.todense()) if hasattr(W, "todense") else np.asarray(W)

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    diag = float(np.linalg.norm(V_visual.max(0) - V_visual.min(0)))
    sp = 1.1 * diag
    off_recon = np.array([sp, 0, 0])
    off_rest = np.array([2 * sp, 0, 0])
    off_w = np.array([3 * sp, 0, 0])

    state = {
        "frame": 0,
        "split": "train",
        "bone": 0,
        "playing": False,
        "fps": 30.0,
        "last_tick": 0.0,    # perf_counter timestamp of the last auto-advance
    }

    proxy_mesh = ps.register_surface_mesh(
        "M_proxy(t)", V_p0, F_proxy,
        color=(0.45, 0.65, 0.85), edge_width=1.0,
    )
    recon_mesh = ps.register_surface_mesh(
        "M_visual_recon(t)", V_visual + off_recon,
        F_visual, color=(0.90, 0.45, 0.45), edge_width=1.0,
    )
    ps.register_surface_mesh(
        "M_visual rest", V_visual + off_rest,
        F_visual, color=(0.65, 0.65, 0.65),
        transparency=1.0, edge_width=1.0,
    )
    weights_mesh = ps.register_surface_mesh(
        "M_visual weights", V_visual + off_w,
        F_visual, color=(0.7, 0.7, 0.7), edge_width=1.0,
    )
    weights_mesh.add_scalar_quantity(
        "w(:, j)", W_dense[:, 0], enabled=True, cmap="reds",
    )
    bone_marker = ps.register_point_cloud(
        "bone j", (V_p0[0:1] + off_w).copy(),
        color=(0.10, 0.85, 0.10),
    )
    bone_marker.set_radius(0.01, relative=True)

    def callback() -> None:
        now = time.perf_counter()
        n_frames = n_train if state["split"] == "train" else n_test
        if state["frame"] >= n_frames:
            state["frame"] = max(n_frames - 1, 0)

        changed_split = False
        if psim.Button("train"):
            state["split"] = "train"; state["frame"] = 0; changed_split = True
        if n_test > 0:
            psim.SameLine()
            if psim.Button("test"):
                state["split"] = "test"; state["frame"] = 0; changed_split = True
        psim.Text(f"split: {state['split']}  frames: {n_frames}")

        # Play/pause + fps. We tick the frame ourselves rather than relying on
        # the polyscope callback rate (~vsync) so playback speed is the same
        # whether the window is in foreground or partially occluded.
        play_label = "pause" if state["playing"] else "play"
        if psim.Button(play_label):
            state["playing"] = not state["playing"]
            state["last_tick"] = now
        psim.SameLine()
        _, state["fps"] = psim.SliderFloat("fps", state["fps"], 1.0, 60.0)

        advanced = False
        if state["playing"] and n_frames > 0:
            dt = 1.0 / max(state["fps"], 1e-3)
            if now - state["last_tick"] >= dt:
                state["frame"] = (state["frame"] + 1) % n_frames
                state["last_tick"] = now
                advanced = True

        changed_frame, state["frame"] = psim.SliderInt(
            "frame", state["frame"], 0, max(n_frames - 1, 0)
        )
        changed_bone, state["bone"] = psim.SliderInt(
            "bone j", state["bone"], 0, n_p - 1
        )

        if changed_frame or changed_split or advanced:
            X = X_train if state["split"] == "train" else X_test
            R = V_recon_train if state["split"] == "train" else V_recon_test
            f = state["frame"]
            proxy_mesh.update_vertex_positions(X[f])
            recon_mesh.update_vertex_positions(R[f] + off_recon)

        if changed_bone:
            j = state["bone"]
            weights_mesh.add_scalar_quantity(
                "w(:, j)", W_dense[:, j], enabled=True, cmap="reds",
            )
            bone_marker.update_point_positions(V_p0[j:j + 1] + off_w)

    ps.set_user_callback(callback)
    ps.show()
