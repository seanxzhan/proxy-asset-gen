#! /Users/szhan/miniforge3/envs/proxyasset/bin/python
"""
Visualize visual meshes and their proxy meshes in a polyscope grid.

Visual meshes are shown on the top row, proxy meshes on the bottom row.
Meshes without a corresponding proxy (i.e. a .txt failure file instead of
.obj) are skipped entirely.

Usage:
    python scripts/visualize_proxy_grid.py data/Skirt_caps_removed_cleaned results/Skirt_caps_removed_cleaned_proxy
    python scripts/visualize_proxy_grid.py /path/to/visual_dir /path/to/proxy_dir --cols 6
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import polyscope as ps
import trimesh


def _alphanumeric_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    return np.array(mesh.vertices), np.array(mesh.faces)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualize visual + proxy meshes in a polyscope grid."
    )
    ap.add_argument("visual_dir", type=str, help="Directory of visual meshes (.obj)")
    ap.add_argument("proxy_dir", type=str, help="Directory of proxy meshes (.obj / .txt)")
    ap.add_argument("--cols", type=int, default=None,
                    help="Number of columns in the grid (default: auto)")
    args = ap.parse_args()

    visual_files = sorted(
        [f for f in os.listdir(args.visual_dir) if f.lower().endswith(".obj")],
        key=lambda f: _alphanumeric_key(Path(f).stem),
    )

    entries = []
    for vf in visual_files:
        stem = Path(vf).stem
        proxy_path = os.path.join(args.proxy_dir, f"{stem}.obj")
        has_proxy = os.path.isfile(proxy_path)
        entries.append((os.path.join(args.visual_dir, vf), proxy_path if has_proxy else None, stem))

    if not entries:
        print("No visual meshes found.")
        return

    n = len(entries)
    n_ok = sum(1 for _, p, _ in entries if p is not None)
    n_fail = n - n_ok
    print(f"Found {n} visual meshes ({n_ok} with proxy, {n_fail} failed)")
    cols = args.cols or min(n, 8)
    print(f"Displaying {n} meshes in grid ({cols} cols)")

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("y_up")

    # Every mesh is normalized to a unit bbox diagonal (centered, scaled) before
    # placement, so the grid reads uniformly no matter how the source garments
    # are scaled. Spacing is therefore in unit-mesh multiples, not tied to any
    # one mesh's size.
    spacing_pair = -0.7  # visual/proxy gap within a cell
    spacing_x = 1.5       # gap between grid columns
    spacing_z = 1.2      # gap between grid rows

    for i, (visual_path, proxy_path, stem) in enumerate(entries):
        V_vis, F_vis = load_mesh(visual_path)

        lo, hi = V_vis.min(0), V_vis.max(0)
        centroid_vis = (lo + hi) / 2
        diag_vis = float(np.linalg.norm(hi - lo))
        scale = 1.0 / diag_vis if diag_vis > 0 else 1.0
        V_vis_n = (V_vis - centroid_vis) * scale

        col = i % cols
        row = i // cols

        offset_x = col * spacing_x
        offset_y = -row * spacing_z

        if proxy_path is not None:
            V_prx, F_prx = load_mesh(proxy_path)
            centroid_prx = (V_prx.max(0) + V_prx.min(0)) / 2
            # Scale the proxy by the *visual's* factor so their relative size is
            # preserved within the cell.
            V_prx_n = (V_prx - centroid_prx) * scale

            V_vis_shifted = V_vis_n + np.array([offset_x - spacing_pair / 2, offset_y, 0])
            V_prx_shifted = V_prx_n + np.array([offset_x + spacing_pair / 2, offset_y, 0])

            ps.register_surface_mesh(f"visual/{stem}", V_vis_shifted, F_vis,
                                     color=(0.7, 0.7, 0.7), smooth_shade=False, edge_width=1.0)
            ps.register_surface_mesh(f"proxy/{stem}", V_prx_shifted, F_prx,
                                     color=(0.3, 0.6, 0.9), smooth_shade=False, edge_width=1.0)
        else:
            V_vis_shifted = V_vis_n + np.array([offset_x, offset_y, 0])
            ps.register_surface_mesh(f"FAILED/{stem}", V_vis_shifted, F_vis,
                                     color=(0.9, 0.3, 0.3), smooth_shade=False, edge_width=1.0)

    print("Opening polyscope viewer (close window or Ctrl+C to exit) ...")
    ps.show()


if __name__ == "__main__":
    main()
