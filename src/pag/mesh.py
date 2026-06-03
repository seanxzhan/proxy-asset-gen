"""Triangle mesh with topology extraction and validation.

Topology pieces we need downstream:
  - unique undirected edges (E, 2)        → smoothness term in §3.3
  - boundary mask                         → boundary-loop counting (Table 1)
  - vertex→face adjacency (CSR-like)      → component counting + 1-rings
  - is_closed flag                        → trivially is_closed = no boundary

This module mirrors pbd/mesh.py but drops the bend-quad extraction (no PBD bending
constraint here) and adds connectivity utilities for proxy-quality reporting.

A separate ``VisualMesh`` namedtuple is used for the *raw input* visual mesh,
which the paper allows to be non-manifold; ``build_mesh`` enforces edge-manifoldness
for downstream operands (M_iso, M_proj, M_single, M_proxy).
"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np


VisualMesh = namedtuple("VisualMesh", ["V", "F"])
"""Raw input mesh — may be non-manifold, layered, or have disconnected components."""


class NonManifoldError(ValueError):
    """Raised when a mesh has an edge shared by more than two faces."""


@dataclass
class Mesh:
    V: np.ndarray            # (N, 3) float64 — vertex positions
    F: np.ndarray            # (M, 3) int64 — triangle vertex indices

    edges: np.ndarray        # (E, 2) int64 — unique undirected edges, sorted ascending
    boundary_mask: np.ndarray  # (E,) bool — true on boundary (1-incident) edges
    edge_face_count: np.ndarray = field(repr=False)  # (E,) int — 1 or 2

    @property
    def n_verts(self) -> int:
        return int(self.V.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.F.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    @property
    def is_closed(self) -> bool:
        return not bool(self.boundary_mask.any())

    def face_areas(self) -> np.ndarray:
        v0 = self.V[self.F[:, 0]]
        v1 = self.V[self.F[:, 1]]
        v2 = self.V[self.F[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    def face_normals(self, normalize: bool = True) -> np.ndarray:
        v0 = self.V[self.F[:, 0]]
        v1 = self.V[self.F[:, 1]]
        v2 = self.V[self.F[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        if normalize:
            length = np.linalg.norm(n, axis=1, keepdims=True)
            length[length == 0.0] = 1.0
            n = n / length
        return n

    def vertex_normals(self) -> np.ndarray:
        """Area-weighted average of incident face normals. (N, 3) unit vectors."""
        # Weight by face area (so larger triangles dominate) — implicit via un-normalized normals.
        fn = self.face_normals(normalize=False)  # (M, 3) — magnitude == 2 * face area
        vn = np.zeros_like(self.V)
        np.add.at(vn, self.F[:, 0], fn)
        np.add.at(vn, self.F[:, 1], fn)
        np.add.at(vn, self.F[:, 2], fn)
        length = np.linalg.norm(vn, axis=1, keepdims=True)
        length[length == 0.0] = 1.0
        return vn / length

    def n_components(self) -> int:
        """Number of connected components by triangle adjacency (union-find on edges)."""
        parent = np.arange(self.n_verts, dtype=np.int64)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return int(x)

        for a, b in self.edges:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[ra] = rb
        # Only count roots of vertices that actually appear in F (isolated verts ignored).
        used = np.unique(self.F.ravel())
        roots = {find(int(v)) for v in used}
        return len(roots)

    def n_boundary_loops(self) -> int:
        """Number of distinct boundary loops. Walks boundary edges until each is consumed.

        A manifold mesh's boundary is a disjoint union of simple cycles, so we
        traverse via a vertex→next-boundary-vertex adjacency.
        """
        bedges = self.edges[self.boundary_mask]
        if bedges.size == 0:
            return 0
        # Build adjacency: each boundary vertex has exactly two boundary neighbours
        # in a manifold mesh; we walk it as an undirected graph and count components.
        from collections import defaultdict
        adj: dict[int, list[int]] = defaultdict(list)
        for a, b in bedges:
            adj[int(a)].append(int(b))
            adj[int(b)].append(int(a))

        visited: set[int] = set()
        loops = 0
        for start in adj:
            if start in visited:
                continue
            loops += 1
            stack = [start]
            while stack:
                v = stack.pop()
                if v in visited:
                    continue
                visited.add(v)
                stack.extend(adj[v])
        return loops


def build_mesh(V: np.ndarray, F: np.ndarray) -> Mesh:
    """Construct a Mesh from vertex / face arrays, extracting topology.

    Raises
    ------
    NonManifoldError
        If any edge is shared by more than two faces.
    """
    V = np.ascontiguousarray(V, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.int64)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"V must be (N, 3); got {V.shape}")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"F must be (M, 3); got {F.shape}")

    raw = np.sort(F[:, [[0, 1], [1, 2], [2, 0]]], axis=2).reshape(-1, 2)
    edges, inv = np.unique(raw, axis=0, return_inverse=True)
    counts = np.bincount(inv, minlength=edges.shape[0])

    if (counts > 2).any():
        bad = np.where(counts > 2)[0]
        raise NonManifoldError(
            f"Non-manifold mesh: {bad.size} edge(s) shared by >2 faces "
            f"(first: vertices {edges[bad[0]].tolist()}, "
            f"shared by {int(counts[bad[0]])} faces)"
        )

    boundary_mask = counts == 1

    return Mesh(
        V=V,
        F=F,
        edges=edges,
        boundary_mask=boundary_mask,
        edge_face_count=counts.astype(np.int64),
    )
