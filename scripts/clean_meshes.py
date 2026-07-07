#! /Users/szhan/miniforge3/envs/proxyasset/bin/python
"""
Remove unreferenced vertices and faces from .obj files using PyMeshLab.

Usage:
    python scripts/clean_meshes.py /path/to/input_dir --output /path/to/output_dir
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pymeshlab


def _alphanumeric_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Remove unreferenced vertices/faces from .obj files."
    )
    ap.add_argument("input_dir", type=str, help="Directory containing .obj files")
    ap.add_argument("--output", type=str, default=None,
                    help="Output directory (default: <input_dir_name>_cleaned/ in CWD)")
    args = ap.parse_args()

    input_dir = args.input_dir
    output_dir = args.output or f"{Path(input_dir).name}_cleaned"

    files = sorted(
        [os.path.join(input_dir, f) for f in os.listdir(input_dir)
         if f.lower().endswith(".obj")],
        key=lambda p: _alphanumeric_key(Path(p).stem),
    )
    if not files:
        print(f"No .obj files in {input_dir}")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Cleaning {len(files)} files -> {output_dir}/")

    for i, path in enumerate(files, 1):
        stem = Path(path).stem
        out_path = os.path.join(output_dir, f"{stem}.obj")
        try:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(path)
            v_before = ms.current_mesh().vertex_number()
            f_before = ms.current_mesh().face_number()

            ms.apply_filter("meshing_remove_duplicate_vertices")
            ms.apply_filter("meshing_remove_duplicate_faces")
            ms.apply_filter("meshing_remove_unreferenced_vertices")
            ms.apply_filter("meshing_remove_null_faces")

            v_after = ms.current_mesh().vertex_number()
            f_after = ms.current_mesh().face_number()

            ms.save_current_mesh(out_path)
            removed_v = v_before - v_after
            removed_f = f_before - f_after
            print(f"  [{i}/{len(files)}] {stem}: "
                  f"V {v_before}->{v_after} (-{removed_v})  "
                  f"F {f_before}->{f_after} (-{removed_f})")
        except Exception as e:
            print(f"  [{i}/{len(files)}] ERR: {stem} — {e}")

    print(f"\nDone. Output in {output_dir}/")


if __name__ == "__main__":
    main()
