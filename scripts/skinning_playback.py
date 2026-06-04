"""Replay a saved Stage 2 skinning result without retraining.

Loads the visual mesh, animation data, and the .npz produced by
``get_skin_weights.py --out``. Reconstructs the visual mesh on every frame via
:func:`pag.skinning_lbs.simplified_lbs` and hands everything to the same
polyscope viewer used by ``get_skin_weights.py``.

Examples
--------
$ python scripts/skinning_playback.py \\
      --visual data/9423122485_cleaned.obj \\
      --anim-dir /Users/szhan/projects/pbd/data/9423122485_cleaned_proxy \\
      --weights /tmp/W.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse

from pag import load_dataset, load_obj


def _load_weights(path: Path) -> tuple[scipy.sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Reconstruct (W, B, s) from the .npz that ``get_skin_weights.py --out`` writes."""
    z = np.load(path)
    shape = tuple(int(x) for x in z["W_shape"])
    W = scipy.sparse.csr_matrix(
        (z["W_data"], z["W_indices"], z["W_indptr"]), shape=shape,
    )
    return W, z["B"], z["s"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visual", type=str, required=True,
                    help="Visual mesh OBJ — must match the one used at training time.")
    ap.add_argument("--proxy", type=str, default=None,
                    help="Proxy mesh OBJ (optional, used only for sanity prints).")
    ap.add_argument("--anim-dir", type=str, required=True,
                    help="windblown_data_gen.py output directory.")
    ap.add_argument("--weights", type=str, required=True,
                    help="Path to the .npz saved by get_skin_weights.py --out.")
    args = ap.parse_args()

    visual_path = Path(args.visual)
    print(f"loading visual mesh: {visual_path}")
    V_visual, F_visual = load_obj(visual_path)
    print(f"  |V|={V_visual.shape[0]}  |F|={F_visual.shape[0]}")

    print(f"loading animation data: {args.anim_dir}")
    train, test, _meta = load_dataset(args.anim_dir)
    print(f"  proxy: |V|={train.n_proxy}  |F|={train.F_proxy.shape[0]}")
    print(f"  train: T={train.n_frames}  test: T={test.n_frames}")

    weights_path = Path(args.weights)
    print(f"loading weights: {weights_path}")
    W, B_np, s_np = _load_weights(weights_path)
    print(f"  W shape={W.shape}  nnz={W.nnz}  k_B={B_np.shape[1]}")

    if W.shape[0] != V_visual.shape[0]:
        raise SystemExit(
            f"visual |V|={V_visual.shape[0]} does not match weights N_v={W.shape[0]} "
            "— is --visual the mesh you trained on?"
        )
    if W.shape[1] != train.V_proxy_rest.shape[0]:
        raise SystemExit(
            f"anim N_p={train.V_proxy_rest.shape[0]} does not match weights "
            f"N_p={W.shape[1]} — is --anim-dir from the same Stage 1 pair?"
        )

    if args.proxy:
        proxy_path = Path(args.proxy)
        if proxy_path.exists():
            V_proxy_obj, _ = load_obj(proxy_path)
            if V_proxy_obj.shape == train.V_proxy_rest.shape:
                err = float(np.abs(V_proxy_obj - train.V_proxy_rest).max())
                print(f"  proxy.obj ↔ mesh.npz V0 max abs diff: {err:.3e}")

    import torch
    from pag.skinning_lbs import simplified_lbs
    from pag.skinning_viz import show_skinning

    s = torch.tensor(s_np)
    B = torch.tensor(B_np, dtype=torch.long)
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
        train.X, V_recon_train, W,
        X_test=test.X, V_recon_test=V_recon_test,
    )


if __name__ == "__main__":
    main()
