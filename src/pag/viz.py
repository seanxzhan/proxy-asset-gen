"""Polyscope viewer for the Stage 1 pipeline.

Side-by-side display of the five meshes that flow through ``generate_proxy_mesh``:

  M_visual → M_iso → M_proj → M_single → M_proxy

Each is shifted along +x by a multiple of the input bbox diagonal so all five
sit on a horizontal strip. Toggle individually in polyscope's sidebar.

Polyscope is imported lazily (inside ``StagesViewer.__init__``) so the package
stays importable in headless contexts (CI, ``--smoke`` runs, unit tests).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from pag.mesh import Mesh, VisualMesh
from pag.pipeline import ProxyMesh


def _bbox_diag(V: np.ndarray) -> float:
    if V.size == 0:
        return 1.0
    return float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))


class StagesViewer:
    """Register the five Stage-1 meshes side-by-side in polyscope.

    Parameters
    ----------
    proxy : ProxyMesh
        Output of :func:`pag.generate_proxy_mesh`. Must have been produced with
        ``keep_intermediates=True`` for the M_iso/M_proj/M_single panels to show.
    visual : VisualMesh or (V, F) tuple
        The original input mesh (typically ``M_visual``).
    spacing : float, optional
        Horizontal gap between adjacent meshes, as a multiple of the visual
        mesh's bbox diagonal. Default 1.2 (small breathing room).
    """

    def __init__(
        self,
        proxy: ProxyMesh,
        visual: VisualMesh | tuple[np.ndarray, np.ndarray],
        *,
        spacing: float = 1.2,
    ):
        import polyscope as ps

        self._ps = ps
        if not getattr(ps, "_pag_initialized", False):
            ps.init()
            ps._pag_initialized = True
        ps.set_up_dir("y_up")
        ps.set_front_dir("z_front")
        ps.set_ground_plane_mode("none")

        V_visual = np.asarray(visual[0], dtype=np.float64)
        F_visual = np.asarray(visual[1], dtype=np.int64)
        diag = _bbox_diag(V_visual)
        step = spacing * diag

        self._registered: list[str] = []

        # Stage 0 — input visual mesh.
        self._register("M_visual", V_visual, F_visual, dx=0 * step,
                       color=(0.65, 0.65, 0.7), transparency=0.35)

        # Stages 1..3 — intermediates (only if keep_intermediates was set).
        if proxy.M_iso is not None:
            self._register("M_iso  (§3.1 UDF)", proxy.M_iso.V, proxy.M_iso.F,
                           dx=1 * step, color=(0.45, 0.65, 0.85))
        if proxy.M_proj is not None:
            self._register("M_proj (§3.2 projection)", proxy.M_proj.V, proxy.M_proj.F,
                           dx=2 * step, color=(0.55, 0.80, 0.55))
        if proxy.M_single is not None:
            self._register("M_single (§3.3 ILP cut)", proxy.M_single.V, proxy.M_single.F,
                           dx=3 * step, color=(0.95, 0.75, 0.35))

        # Stage 4 — proxy. Always present.
        self._register("M_proxy (§3.4 ACVD)", proxy.mesh.V, proxy.mesh.F,
                       dx=4 * step, color=(0.90, 0.45, 0.45))

    def _register(
        self,
        name: str,
        V: np.ndarray,
        F: np.ndarray,
        *,
        dx: float,
        color: tuple[float, float, float],
        transparency: float = 1.0,
    ) -> None:
        if V.shape[0] == 0 or F.shape[0] == 0:
            return
        Vs = V.copy()
        Vs[:, 0] += dx
        m = self._ps.register_surface_mesh(name, Vs, F, color=color)
        m.set_smooth_shade(True)
        if transparency < 1.0:
            m.set_transparency(transparency)
        self._registered.append(name)

    def show(self) -> None:
        """Block on the polyscope event loop."""
        self._ps.show()


def show_stages(
    proxy: ProxyMesh,
    visual: VisualMesh | tuple[np.ndarray, np.ndarray],
    *,
    spacing: float = 1.2,
) -> None:
    """Convenience: build a :class:`StagesViewer` and call ``show()``."""
    StagesViewer(proxy, visual, spacing=spacing).show()
