"""Stage 2 — optimize skinning weights for a real asset.

Inputs:
  - --visual : the visual mesh OBJ (Stage 1 input)
  - --proxy  : the proxy mesh OBJ (Stage 1 output) — only used for sanity prints;
               the rest pose actually consumed comes from --anim-dir/mesh.npz
  - --anim-dir : windblown_data_gen.py output directory (mesh.npz, train.npz,
                 test.npz, metadata.json)

Examples
--------
$ python scripts/get_skin_weights.py --smoke
$ python scripts/get_skin_weights.py --epochs 200 --device cpu --out /tmp/W.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pag import (
    load_dataset,
    load_obj,
    optimize_skinning_weights,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--visual", type=str,
        default="data/9423122485_cleaned.obj",
    )
    ap.add_argument(
        "--proxy", type=str,
        default="data/9423122485_cleaned_proxy.obj",
    )
    ap.add_argument(
        "--anim-dir", type=str,
        default="/Users/szhan/projects/pbd/data/9423122485_cleaned_proxy",
    )
    ap.add_argument("--epochs", type=int, default=200, help="paper §4: 200")
    ap.add_argument("--lr", type=float, default=1e-3, help="paper §4: 1e-3")
    ap.add_argument("--k-B", type=int, default=8)
    ap.add_argument("--k-K", type=int, default=8)
    ap.add_argument("--lambda-r", type=float, default=1.0)
    ap.add_argument("--lambda-c", type=float, default=1.0)
    ap.add_argument("--lambda-a", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cpu",
                    choices=["cpu", "mps", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None,
                    help="Optional output .npz path for W, B, s, eval.")
    ap.add_argument("--smoke", action="store_true",
                    help="Headless run; print stats and exit.")
    args = ap.parse_args()

    visual_path = Path(args.visual)
    print(f"loading visual mesh: {visual_path}")
    V_visual, F_visual = load_obj(visual_path)
    print(f"  |V|={V_visual.shape[0]}  |F|={F_visual.shape[0]}")

    print(f"loading animation data: {args.anim_dir}")
    train, test, meta = load_dataset(args.anim_dir)
    print(f"  proxy: |V|={train.n_proxy}  |F|={train.F_proxy.shape[0]}")
    print(f"  train: T={train.n_frames}  test: T={test.n_frames}")
    if "metrics" in meta:
        print(f"  OOD metrics (from windblown_data_gen.py): {meta['metrics']}")

    # Sanity: the proxy OBJ should agree with mesh.npz V0 byte-for-byte if both
    # come from the same Stage 1 / pre-sim pair.
    proxy_path = Path(args.proxy)
    if proxy_path.exists():
        V_proxy_obj, _ = load_obj(proxy_path)
        if V_proxy_obj.shape == train.V_proxy_rest.shape:
            err = float(np.abs(V_proxy_obj - train.V_proxy_rest).max())
            print(f"  proxy.obj ↔ mesh.npz V0 max abs diff: {err:.3e}")

    print(f"\noptimizing skinning weights ({args.epochs} epochs, device={args.device}) ...")
    res = optimize_skinning_weights(
        V_visual, F_visual, train, test=test,
        k_B=args.k_B, k_K=args.k_K,
        lambda_r=args.lambda_r, lambda_c=args.lambda_c, lambda_a=args.lambda_a,
        epochs=args.epochs, lr=args.lr,
        device=args.device, seed=args.seed,
        verbose=True, log_every=20,
    )

    print()
    print("=== Stage 2 summary ===")
    print(f"  L_total: {res.losses['total'][0]:.3e} -> {res.losses['total'][-1]:.3e}")
    print(f"  L_r:     {res.losses['L_r'][0]:.3e} -> {res.losses['L_r'][-1]:.3e}")
    print(f"  L_c:     {res.losses['L_c'][0]:.3e} -> {res.losses['L_c'][-1]:.3e}")
    print(f"  L_a:     {res.losses['L_a'][0]:.3e} -> {res.losses['L_a'][-1]:.3e}")
    print(f"  setup:    {res.timings['setup']:.2f}s")
    print(f"  optimize: {res.timings['optimize']:.2f}s")
    print(f"  total:    {res.timings['total']:.2f}s  (paper budget: ~9 min on i9)")
    print(f"  W shape={res.W.shape}  nnz={res.W.nnz}")
    print(f"  held-out test eval: {res.eval}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            W_data=res.W.data, W_indices=res.W.indices, W_indptr=res.W.indptr,
            W_shape=np.asarray(res.W.shape),
            B=res.B, s=res.s,
        )
        print(f"  wrote {out_path}")

    if args.smoke:
        return

    import torch
    from pag.skinning_lbs import simplified_lbs
    from pag.skinning_viz import show_skinning

    s = torch.tensor(res.s)
    B = torch.tensor(res.B, dtype=torch.long)
    V_v0 = torch.tensor(V_visual, dtype=torch.float32)
    V_p0 = torch.tensor(train.V_proxy_rest, dtype=torch.float32)
    X_train = torch.tensor(train.X, dtype=torch.float32)
    X_test = torch.tensor(test.X, dtype=torch.float32)
    with torch.no_grad():
        V_recon_train = simplified_lbs(s, B, V_v0, V_p0, X_train).numpy()
        V_recon_test = simplified_lbs(s, B, V_v0, V_p0, X_test).numpy()

    show_skinning(
        V_visual, F_visual,
        train.V_proxy_rest, train.F_proxy,
        train.X, V_recon_train, res.W,
        X_test=test.X, V_recon_test=V_recon_test,
    )


if __name__ == "__main__":
    main()
