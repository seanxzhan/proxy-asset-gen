"""Side-by-side polyscope viewer for scenario-based skinning evaluation.

Lays out two meshes along +x:

  x = 0    M_proxy(t)            — animated proxy from the scenario sim
  x = sp   M_visual_recon(t)     — LBS reconstruction of the visual mesh

Obstacles (sphere / plane / mesh) are drawn in BOTH panes so collisions in
proxy space and the resulting deformation in visual space are visible together.
Spheres render as polyscope point-cloud sphere impostors (GPU ray-traced —
effectively an SDF render — so they're smooth at any zoom level, not faceted).

UI:
  - play / pause toggle + fps slider
  - frame slider

Modeled on `pag.skinning_viz.show_skinning` but stripped of the bone selector,
the train/test split toggle, and the rest/weights panes — all of which apply to
the wind-data viewer, not to scenario evaluation.
"""
from __future__ import annotations

import time

import numpy as np

from pag.eval_runner import Obstacle


# ------------------------------------------------------------ small geometry

def _plane_quad(center: np.ndarray, normal: np.ndarray, size: float):
    """A square quad sized `2*size` per side, centered at `center`, lying in
    the plane with the given `normal`. Two triangles."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / max(np.linalg.norm(n), 1e-12)
    # Build an orthonormal basis (u, v) on the plane.
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper); u /= max(np.linalg.norm(u), 1e-12)
    v = np.cross(n, u)
    c = np.asarray(center, dtype=np.float64)
    V = np.stack([
        c - size * u - size * v,
        c + size * u - size * v,
        c + size * u + size * v,
        c - size * u + size * v,
    ])
    F = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return V, F


# ------------------------------------------------------------ obstacle handles

class _ObstacleHandle:
    """One obstacle, registered twice with polyscope (one per pane).

    Spheres render as a 1-point polyscope point cloud in `'sphere'` render
    mode — GPU-ray-traced impostors with absolute world-space radius. Smooth
    at any zoom level; no triangulated facets to clean up.
    """

    def __init__(self, ps, ob: Obstacle, slot: int, off_visual: np.ndarray):
        self.kind = ob.kind
        self.off_visual = off_visual

        if ob.kind == "sphere":
            center = np.asarray(ob.center, dtype=np.float64).reshape(1, 3)
            self._h_proxy = ps.register_point_cloud(
                f"obs{slot}_{ob.name}_proxy", center,
                point_render_mode="sphere", color=ob.color,
            )
            self._h_proxy.set_radius(float(ob.radius), relative=False)
            self._h_visual = ps.register_point_cloud(
                f"obs{slot}_{ob.name}_visual", center + off_visual,
                point_render_mode="sphere", color=ob.color,
            )
            self._h_visual.set_radius(float(ob.radius), relative=False)
        elif ob.kind == "plane":
            # Planes are static and rendered as a large finite quad; if a
            # scenario ever needs a sized or moving plane, add a `size` field
            # to Obstacle and update() here.
            V_p, F_p = _plane_quad(ob.center, ob.normal, size=10.0)
            self._h_proxy = ps.register_surface_mesh(
                f"obs{slot}_{ob.name}_proxy", V_p, F_p,
                color=ob.color, transparency=0.6, edge_width=0.0,
            )
            self._h_visual = ps.register_surface_mesh(
                f"obs{slot}_{ob.name}_visual", V_p + off_visual, F_p,
                color=ob.color, transparency=0.6, edge_width=0.0,
            )
        elif ob.kind == "mesh":
            V_p = np.asarray(ob.V, dtype=np.float64)
            F_p = np.asarray(ob.F, dtype=np.int64)
            self._h_proxy = ps.register_surface_mesh(
                f"obs{slot}_{ob.name}_proxy", V_p, F_p,
                color=ob.color, edge_width=0.5,
            )
            self._h_visual = ps.register_surface_mesh(
                f"obs{slot}_{ob.name}_visual", V_p + off_visual, F_p,
                color=ob.color, edge_width=0.5,
            )
        else:
            raise ValueError(f"unknown obstacle kind: {ob.kind!r}")

    def update(self, ob: Obstacle) -> None:
        """Refresh both handles for frame t. No-op for static planes.

        Sphere radius is fixed at construction. If a scenario needs a sphere
        whose radius varies per frame, also call set_radius() here.
        """
        if ob.kind == "sphere":
            center = np.asarray(ob.center, dtype=np.float64).reshape(1, 3)
            self._h_proxy.update_point_positions(center)
            self._h_visual.update_point_positions(center + self.off_visual)
        elif ob.kind == "mesh":
            V_p = np.asarray(ob.V, dtype=np.float64)
            self._h_proxy.update_vertex_positions(V_p)
            self._h_visual.update_vertex_positions(V_p + self.off_visual)
        # plane: nothing to update


# --------------------------------------------------------------- show_eval

def show_eval(
    V_visual: np.ndarray,
    F_visual: np.ndarray,
    V_p0: np.ndarray,
    F_proxy: np.ndarray,
    X_p: np.ndarray,            # (T, N_p, 3)
    V_recon: np.ndarray,        # (T, N_v, 3)
    obstacles_per_frame: list[list[Obstacle]],
    *,
    fps: float = 60.0,
) -> None:
    import polyscope as ps
    import polyscope.imgui as psim

    n_frames = X_p.shape[0]
    if V_recon.shape[0] != n_frames:
        raise ValueError(
            f"X_p has {n_frames} frames but V_recon has {V_recon.shape[0]}"
        )
    if len(obstacles_per_frame) != n_frames:
        raise ValueError(
            f"obstacles_per_frame has {len(obstacles_per_frame)} entries; "
            f"expected {n_frames}"
        )

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    diag = float(np.linalg.norm(V_visual.max(0) - V_visual.min(0)))
    sp = 1.1 * diag
    off_visual = np.array([sp, 0.0, 0.0])

    proxy_mesh = ps.register_surface_mesh(
        "M_proxy(t)", X_p[0], F_proxy,
        color=(0.45, 0.65, 0.85), edge_width=1.0,
    )
    recon_mesh = ps.register_surface_mesh(
        "M_visual_recon(t)", V_recon[0] + off_visual, F_visual,
        color=(0.90, 0.45, 0.45), edge_width=1.0,
    )

    # One handle per obstacle slot, allocated from frame 0's obstacle list.
    # Scenarios are expected to return a stable obstacle layout (same names,
    # same kinds) across frames; only the geometry is updated.
    obs0 = obstacles_per_frame[0]
    handles = [_ObstacleHandle(ps, ob, i, off_visual) for i, ob in enumerate(obs0)]

    state = {
        "frame": 0,
        "playing": False,
        "fps": float(fps),
        "last_tick": 0.0,
    }

    def callback() -> None:
        now = time.perf_counter()

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
            "frame", state["frame"], 0, max(n_frames - 1, 0),
        )

        if changed_frame or advanced:
            f = state["frame"]
            proxy_mesh.update_vertex_positions(X_p[f])
            recon_mesh.update_vertex_positions(V_recon[f] + off_visual)
            for h, ob in zip(handles, obstacles_per_frame[f]):
                h.update(ob)

    ps.set_user_callback(callback)
    ps.show()
