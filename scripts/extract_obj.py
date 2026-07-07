#! /Users/szhan/miniforge3/envs/proxyasset/bin/python
"""
Extract non-cage meshes from .gltf/.glb files and save as .obj.

For each file in the input directory, loads all geometries, drops any whose
name contains "cage" (case-insensitive), combines the rest, and writes a
single .obj to the output folder.

Usage:
    python extract_obj.py /path/to/input_dir --output /path/to/output_dir
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import trimesh


CAGE_RE = re.compile(r"cage", re.IGNORECASE)


def _alphanumeric_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_non_cage_mesh(path: str) -> trimesh.Trimesh:
    """Load a gltf/glb and return all non-'cage' geometries concatenated."""
    loaded = trimesh.load(path, process=False)

    if isinstance(loaded, trimesh.Trimesh):
        return loaded

    if not isinstance(loaded, trimesh.Scene):
        raise ValueError(f"Unsupported load result type: {type(loaded)}")

    keep_names = []
    for name in loaded.geometry.keys():
        if not CAGE_RE.search(name):
            keep_names.append(name)

    cage_geometries = set()
    node_geometries = {}
    for node_name in loaded.graph.nodes_geometry:
        try:
            _, geom_name = loaded.graph[node_name]
        except (KeyError, ValueError):
            continue
        if geom_name is None:
            continue
        node_geometries.setdefault(geom_name, []).append(node_name)

    for geom_name, node_names in node_geometries.items():
        if all(CAGE_RE.search(n) for n in node_names):
            cage_geometries.add(geom_name)

    keep_names = [n for n in keep_names if n not in cage_geometries]

    if not keep_names:
        raise ValueError(f"No non-cage geometry found in {path}")

    sub = trimesh.Scene()
    for node_name in loaded.graph.nodes_geometry:
        try:
            transform, geom_name = loaded.graph[node_name]
        except (KeyError, ValueError):
            continue
        if geom_name not in keep_names:
            continue
        geom = loaded.geometry[geom_name]
        sub.add_geometry(geom, node_name=node_name, transform=transform)

    if not sub.geometry:
        meshes = [loaded.geometry[n] for n in keep_names]
        return trimesh.util.concatenate(meshes)

    combined = trimesh.util.concatenate(sub.dump())
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Extract non-cage meshes from gltf/glb files to .obj."
    )
    parser.add_argument("input_dir", type=str, help="Directory containing .gltf/.glb files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: <input_dir_name>_obj/ in CWD)")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output or f"{Path(input_dir).name}_obj"

    files = sorted(
        [os.path.join(input_dir, f) for f in os.listdir(input_dir)
         if f.lower().endswith((".gltf", ".glb"))],
        key=lambda p: _alphanumeric_key(Path(p).stem),
    )
    if not files:
        print(f"No .gltf/.glb files in {input_dir}")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Extracting {len(files)} files to {output_dir}/")

    for i, path in enumerate(files, 1):
        stem = Path(path).stem
        out_path = os.path.join(output_dir, f"{stem}.obj")
        try:
            mesh = load_non_cage_mesh(path)
            mesh.export(out_path)
            nv, nf = len(mesh.vertices), len(mesh.faces)
            print(f"  [{i}/{len(files)}] ok: {stem}.obj (V:{nv:,} F:{nf:,})")
        except Exception as e:
            print(f"  [{i}/{len(files)}] ERR: {stem} — {e}")

    print(f"\nDone. Output in {output_dir}/")


if __name__ == "__main__":
    main()
