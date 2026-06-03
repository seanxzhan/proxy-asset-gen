"""End-to-end Stage 1 pipeline: M_visual → M_iso → M_proj → M_single → M_proxy.

Single entry point :func:`generate_proxy_mesh` chains §3.1 through §3.4.
Per-stage timings are captured in :attr:`ProxyMesh.timings`; all paper-silent
knobs are recorded in :attr:`ProxyMesh.knobs` so a run is reproducible.

Set ``keep_intermediates=True`` to retain ``M_iso``, ``M_proj``, ``M_single``
on the result — useful for debugging and visualisation. Off by default to
keep memory bounded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from pag.extract import extract_single_layer
from pag.guide_graph import build_guide_graph
from pag.ilp import solve_layer_ilp
from pag.mesh import Mesh
from pag.projection import project_to_visual
from pag.udf import isosurface_from_udf
from pag.voronoi import voronoi_simplify


@dataclass
class ProxyMesh:
    """Output of :func:`generate_proxy_mesh`.

    Attributes
    ----------
    mesh : Mesh
        ``M_proxy`` — the deliverable, typically ~``n_p`` vertices.
    timings : dict[str, float]
        Wall-clock seconds per stage. Keys: ``udf``, ``projection``, ``guide_graph``,
        ``ilp``, ``extract``, ``voronoi``, ``total``.
    knobs : dict[str, float]
        All non-default parameter values used, for reproduction.
    M_iso, M_proj, M_single : Mesh, optional
        Intermediate stages, retained iff ``keep_intermediates=True``.
    voxel_size : float
        ``§3.1 D / N_v`` — exposed so downstream code can scale relative to it.
    n_keep_ilp : int
        Vertex count surviving §3.3 (before §3.4 resampling). Useful for sanity.
    """
    mesh: Mesh
    timings: dict[str, float] = field(default_factory=dict)
    knobs: dict[str, float] = field(default_factory=dict)
    M_iso: Mesh | None = None
    M_proj: Mesh | None = None
    M_single: Mesh | None = None
    voxel_size: float = 0.0
    n_keep_ilp: int = 0


def generate_proxy_mesh(
    V_visual: np.ndarray,
    F_visual: np.ndarray,
    *,
    # §3.1
    n_v: int = 32,
    # §3.2
    proj_iters: int = 200,
    proj_lr: float = 5e-3,
    lambda_L: float = 0.1,
    # §3.3
    eps_normal: float = 0.1,
    lambda_s: float = 1.0,
    lambda_o: float = 1.0,
    lambda_bias: float = 0.5,
    enable_outer_bias: bool = True,
    ilp_time_limit: float = 120.0,
    curvature_radius: int = 3,
    # §3.4
    n_p: int = 128,
    cvd_max_iter: int = 100,
    # bookkeeping
    keep_intermediates: bool = False,
    verbose: bool = False,
) -> ProxyMesh:
    """Run the full Stage 1 pipeline on a visual cloth mesh.

    Parameters
    ----------
    V_visual : (N_in, 3) float
        Input visual mesh vertex positions. May be non-manifold.
    F_visual : (M_in, 3) int
        Input triangles.

    n_v : int, default 32
        §3.1 voxel resolution along the longest bbox axis (paper-pinned).
    proj_iters, proj_lr, lambda_L : §3.2 vector-Adam knobs (paper-silent).
    eps_normal : §3.3 opposite-edge normal threshold (paper-silent).
    lambda_s, lambda_o, lambda_bias : §3.3 ILP energy weights.
    enable_outer_bias : §3.3 — disable to debug bias-driven label flips.
    ilp_time_limit : HiGHS time cap, seconds.
    curvature_radius : k-ring for §3.3 curvature computation.

    n_p : int, default 128
        §3.4 target proxy vertex count (paper-pinned).
    cvd_max_iter : §3.4 ACVD Lloyd-iteration cap.

    keep_intermediates : if True, retain M_iso, M_proj, M_single on the result.
    verbose : print per-stage timings as we go.

    Returns
    -------
    ProxyMesh
    """
    knobs = {
        "n_v": n_v, "proj_iters": proj_iters, "proj_lr": proj_lr,
        "lambda_L": lambda_L, "eps_normal": eps_normal,
        "lambda_s": lambda_s, "lambda_o": lambda_o, "lambda_bias": lambda_bias,
        "enable_outer_bias": float(enable_outer_bias),
        "n_p": n_p, "cvd_max_iter": cvd_max_iter,
        "curvature_radius": curvature_radius,
    }
    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    iso = isosurface_from_udf(V_visual, F_visual, n_v=n_v)
    timings["udf"] = time.perf_counter() - t0
    if verbose:
        print(f"§3.1 udf:         {timings['udf']:.3f}s  "
              f"|V|={iso.mesh.n_verts} |F|={iso.mesh.n_faces}")

    t0 = time.perf_counter()
    proj = project_to_visual(
        iso.mesh, V_visual, F_visual,
        n_iters=proj_iters, lr=proj_lr, lambda_L=lambda_L,
    )
    timings["projection"] = time.perf_counter() - t0
    if verbose:
        print(f"§3.2 projection:  {timings['projection']:.3f}s  ({proj_iters} iters)")

    t0 = time.perf_counter()
    guide = build_guide_graph(
        proj.mesh, voxel_size=iso.voxel_size,
        eps_normal=eps_normal, curvature_radius=curvature_radius,
    )
    timings["guide_graph"] = time.perf_counter() - t0
    if verbose:
        print(f"§3.3 guide_graph: {timings['guide_graph']:.3f}s  "
              f"|E_o|={guide.opposite_edges.shape[0]}")

    t0 = time.perf_counter()
    labels = solve_layer_ilp(
        guide,
        lambda_s=lambda_s, lambda_o=lambda_o, lambda_bias=lambda_bias,
        enable_outer_bias=enable_outer_bias,
        time_limit=ilp_time_limit,
    )
    timings["ilp"] = time.perf_counter() - t0
    if verbose:
        print(f"§3.3 ilp:         {timings['ilp']:.3f}s  "
              f"keep {labels.n_keep}/{proj.mesh.n_verts}")

    t0 = time.perf_counter()
    single = extract_single_layer(proj.mesh, labels.labels)
    timings["extract"] = time.perf_counter() - t0
    if verbose:
        print(f"§3.3 extract:     {timings['extract']:.3f}s  "
              f"|V|={single.n_verts} |F|={single.n_faces}")

    t0 = time.perf_counter()
    proxy = voronoi_simplify(single, n_p=n_p, max_iter=cvd_max_iter)
    timings["voronoi"] = time.perf_counter() - t0
    if verbose:
        print(f"§3.4 voronoi:     {timings['voronoi']:.3f}s  "
              f"|V|={proxy.n_verts} |F|={proxy.n_faces} "
              f"components={proxy.n_components()} bdry_loops={proxy.n_boundary_loops()}")

    timings["total"] = time.perf_counter() - t_start
    if verbose:
        print(f"total:            {timings['total']:.3f}s")

    return ProxyMesh(
        mesh=proxy,
        timings=timings,
        knobs=knobs,
        M_iso=iso.mesh if keep_intermediates else None,
        M_proj=proj.mesh if keep_intermediates else None,
        M_single=single if keep_intermediates else None,
        voxel_size=iso.voxel_size,
        n_keep_ilp=labels.n_keep,
    )
