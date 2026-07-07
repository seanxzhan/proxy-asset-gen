#! /Users/szhan/miniforge3/envs/proxyasset/bin/python
"""
Batch proxy generation over a directory of .obj files.

For each .obj in the input directory, runs generate_proxy_mesh and saves the
result to the output folder. On failure, writes a .txt file with the error.

Usage:
    python scripts/batch_proxy.py /path/to/objs --output /path/to/output
    python scripts/batch_proxy.py /path/to/objs --n-v 32 --n-p 128 --collision-free
"""
from __future__ import annotations

import argparse
import os
import re
import traceback
from pathlib import Path

from pag import generate_proxy_mesh, load_obj, save_obj


def _alphanumeric_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch proxy generation over a directory of .obj files."
    )
    ap.add_argument("input_dir", type=str, help="Directory containing .obj files")
    ap.add_argument("--output", type=str, default=None,
                    help="Output directory (default: <input_dir_name>_proxy/ in CWD)")
    ap.add_argument("--n-v", type=int, default=32,
                    help="§3.1 voxel resolution (paper: 32).")
    ap.add_argument("--n-p", type=int, default=128,
                    help="§3.4 target proxy vertex count (paper: 128).")
    ap.add_argument("--proj-iters", type=int, default=200,
                    help="§3.2 vector-Adam iterations.")
    ap.add_argument("--lambda-L", type=float, default=0.1,
                    help="§3.2 Laplacian smoothing weight.")
    ap.add_argument("--lambda-bias", type=float, default=0.5,
                    help="§3.3 outer-layer bias.")
    ap.add_argument("--no-bias", action="store_true",
                    help="Disable §3.3 outer-layer bias.")
    ap.add_argument("--collision-free", action="store_true",
                    help="§3.2: IPC + CCD solver for intersection-free projection.")
    ap.add_argument("--proj-dhat-frac", type=float, default=1e-3,
                    help="§3.2 IPC barrier activation distance fraction.")
    ap.add_argument("--verbose", action="store_true",
                    help="Pass verbose=True to generate_proxy_mesh.")
    args = ap.parse_args()

    input_dir = args.input_dir
    output_dir = args.output or f"{Path(input_dir).name}_proxy"

    files = sorted(
        [os.path.join(input_dir, f) for f in os.listdir(input_dir)
         if f.lower().endswith(".obj")],
        key=lambda p: _alphanumeric_key(Path(p).stem),
    )
    if not files:
        print(f"No .obj files in {input_dir}")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(files)} files -> {output_dir}/")

    n_ok, n_err, n_skip = 0, 0, 0
    for i, path in enumerate(files, 1):
        stem = Path(path).stem
        out_path = os.path.join(output_dir, f"{stem}.obj")
        err_path = os.path.join(output_dir, f"{stem}.txt")
        if os.path.isfile(out_path):
            print(f"  [{i}/{len(files)}] skip: {stem} (already generated)")
            n_skip += 1
            continue
        if os.path.isfile(err_path):
            print(f"  [{i}/{len(files)}] skip: {stem} (previously failed)")
            n_skip += 1
            continue
        print(f"\n[{i}/{len(files)}] {stem}")
        try:
            V, F = load_obj(path)
            print(f"  input |V|={V.shape[0]}  |F|={F.shape[0]}")

            out = generate_proxy_mesh(
                V, F,
                n_v=args.n_v,
                n_p=args.n_p,
                proj_iters=args.proj_iters,
                lambda_L=args.lambda_L,
                proj_collision_free=args.collision_free,
                proj_dhat_frac=args.proj_dhat_frac,
                lambda_bias=args.lambda_bias,
                enable_outer_bias=not args.no_bias,
                keep_intermediates=False,
                verbose=args.verbose,
            )

            out_path = os.path.join(output_dir, f"{stem}.obj")
            save_obj(out_path, out.mesh.V, out.mesh.F)
            print(f"  ok: |V|={out.mesh.n_verts}  |F|={out.mesh.n_faces}  "
                  f"components={out.mesh.n_components()}  "
                  f"boundary_loops={out.mesh.n_boundary_loops()}")
            n_ok += 1

        except Exception as e:
            err_path = os.path.join(output_dir, f"{stem}.txt")
            tb = traceback.format_exc()
            with open(err_path, "w") as f:
                f.write(tb)
            print(f"  FAILED: {e}")
            print(f"  traceback written to {err_path}")
            n_err += 1

    print(f"\nDone. {n_ok} succeeded, {n_err} failed, {n_skip} skipped. Output in {output_dir}/")


if __name__ == "__main__":
    main()
