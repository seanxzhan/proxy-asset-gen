"""pag — Proxy Asset Generation for Cloth Simulation.

Stage 1 (proxy mesh generation) and Stage 2 (skinning weights) of
Zheng et al. 2024 (ACM TOG 43(4) Article 73). See docs/Proxy-Asset-Generation.md
for the algorithmic overview.

Skinning symbols (``optimize_skinning_weights``, ``SkinningResult``) and the
voronoi step (``voronoi_simplify``) are lazy-loaded on first attribute access.
The reason: ``pag.skinning`` imports torch and ``pag.voronoi`` imports
pyvista/pyacvd → vtk; both ship their own OpenMP runtime on macOS and
co-importing them at module load can segfault inside torch fancy-indexing.
The lazy split lets a Stage 1-only or Stage 2-only consumer avoid the
collision entirely; mixed users see whichever runtime they touched first.
"""
from __future__ import annotations

from pag.anim import AnimationData, load_dataset
from pag.curvature import vertex_abs_curvature
from pag.extract import extract_single_layer
from pag.guide_graph import GuideGraph, build_guide_graph
from pag.ilp import LayerLabels, solve_layer_ilp
from pag.io import load_obj, save_obj
from pag.mesh import Mesh, NonManifoldError, VisualMesh, build_mesh
from pag.pipeline import ProxyMesh, generate_proxy_mesh
from pag.projection import ProjectionResult, project_to_visual
from pag.udf import IsoExtraction, isosurface_from_udf

__all__ = [
    "AnimationData",
    "GuideGraph",
    "IsoExtraction",
    "LayerLabels",
    "Mesh",
    "NonManifoldError",
    "ProjectionResult",
    "ProxyMesh",
    "SkinningResult",
    "VisualMesh",
    "build_guide_graph",
    "build_mesh",
    "extract_single_layer",
    "generate_proxy_mesh",
    "isosurface_from_udf",
    "load_dataset",
    "load_obj",
    "optimize_skinning_weights",
    "project_to_visual",
    "save_obj",
    "solve_layer_ilp",
    "vertex_abs_curvature",
    "voronoi_simplify",
]


_LAZY = {
    "SkinningResult": ("pag.skinning", "SkinningResult"),
    "optimize_skinning_weights": ("pag.skinning", "optimize_skinning_weights"),
    "voronoi_simplify": ("pag.voronoi", "voronoi_simplify"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        mod_name, attr = _LAZY[name]
        mod = importlib.import_module(mod_name)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'pag' has no attribute {name!r}")
