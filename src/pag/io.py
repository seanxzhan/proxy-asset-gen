"""Minimal OBJ loader/saver. Numpy only, no trimesh dep.

Mirrors pbd/io.py — kept deliberately small. For richer formats reach for
trimesh; for the cloth pipeline we only need triangle V/F.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an OBJ file as (V, F).

    Returns
    -------
    V : (N, 3) float64
    F : (M, 3) int64

    Only ``v`` and ``f`` lines are parsed. Polygons with >3 vertices are
    triangulated as a fan from the first vertex. Texture and normal
    references on ``f`` lines (e.g. ``1/2/3``) are ignored.
    """
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with Path(path).open() as fh:
        for line in fh:
            tok = line.split()
            if not tok:
                continue
            head = tok[0]
            if head == "v":
                verts.append([float(x) for x in tok[1:4]])
            elif head == "f":
                idx = [int(t.split("/", 1)[0]) - 1 for t in tok[1:]]
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return (
        np.asarray(verts, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
    )


def save_obj(path: str | Path, V: np.ndarray, F: np.ndarray) -> None:
    """Write (V, F) as a triangle OBJ. 1-indexed faces, no normals/UVs."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"V must be (N, 3); got {V.shape}")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"F must be (M, 3); got {F.shape}")

    with Path(path).open("w") as fh:
        for x, y, z in V:
            fh.write(f"v {x:.8g} {y:.8g} {z:.8g}\n")
        for a, b, c in F + 1:
            fh.write(f"f {a} {b} {c}\n")
