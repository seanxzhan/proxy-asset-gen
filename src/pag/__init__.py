"""pag — Proxy Asset Generation for Cloth Simulation.

Implementation of Stage 1 (proxy mesh generation) from
Zheng et al. 2024 (ACM TOG 43(4) Article 73). See docs/Proxy-Asset-Generation.md
for the algorithmic overview.
"""
from __future__ import annotations

from pag.curvature import vertex_abs_curvature
from pag.extract import extract_single_layer
from pag.guide_graph import GuideGraph, build_guide_graph
from pag.ilp import LayerLabels, solve_layer_ilp
from pag.io import load_obj, save_obj
from pag.mesh import Mesh, NonManifoldError, VisualMesh, build_mesh
from pag.pipeline import ProxyMesh, generate_proxy_mesh
from pag.projection import ProjectionResult, project_to_visual
from pag.udf import IsoExtraction, isosurface_from_udf
from pag.voronoi import voronoi_simplify

__all__ = [
    "GuideGraph",
    "IsoExtraction",
    "LayerLabels",
    "Mesh",
    "NonManifoldError",
    "ProjectionResult",
    "ProxyMesh",
    "VisualMesh",
    "build_guide_graph",
    "build_mesh",
    "extract_single_layer",
    "generate_proxy_mesh",
    "isosurface_from_udf",
    "load_obj",
    "project_to_visual",
    "save_obj",
    "solve_layer_ilp",
    "vertex_abs_curvature",
    "voronoi_simplify",
]
