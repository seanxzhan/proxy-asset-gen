"""Fourier vs Wavelet: why localized sharp events need different bases.

Demonstrates:
1. A smooth signal is sparse in BOTH Fourier and wavelet bases.
2. A signal with a localized sharp fold is DENSE in Fourier but SPARSE in wavelet.
3. Reconstruction quality vs. number of coefficients kept.

This is the core argument for why eigenmodes (global smooth) can't efficiently
represent localized deformation features (folds, wrinkles) — and why a
localized basis (wavelets, or a learned dictionary of local atoms) can.

Run:
    python scripts/visualizations/fourier_vs_wavelet.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def make_signals(n: int = 1024):
    """Create two signals: one smooth, one with a localized sharp fold."""
    x = np.linspace(0, 1, n, endpoint=False)

    # Signal A: smooth (sum of a few low-frequency sinusoids)
    smooth = 0.5 * np.sin(2 * np.pi * 3 * x) + 0.3 * np.cos(2 * np.pi * 7 * x)

    # Signal B: smooth everywhere EXCEPT one localized sharp fold
    fold_center = 0.6
    fold_width = 0.02
    fold_depth = 1.5
    fold = -fold_depth * np.exp(-((x - fold_center) / fold_width) ** 2)
    # Add a sharp crease (triangle wave localized)
    crease = np.zeros_like(x)
    mask = np.abs(x - fold_center) < 0.03
    crease[mask] = -fold_depth * (1.0 - np.abs(x[mask] - fold_center) / 0.03)

    with_fold = smooth + crease

    return x, smooth, with_fold


# ============================================================================
# Biorthogonal 6.8 wavelet via PyWavelets
#
# bior6.8 has smooth, symmetric basis functions (linear phase) — used in
# image/signal processing for its near-optimal time-frequency localization.
# Reconstruction is smooth (no Haar blockiness).
# ============================================================================

import pywt

_WAVELET = 'bior6.8'
_MODE = 'periodization'  # periodic boundary (signal wraps around)


def bior68_wavelet_transform(signal: np.ndarray) -> list[np.ndarray]:
    """Multi-level DWT using bior6.8.

    Returns pywt coefficient list: [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    (approx at coarsest level, then details coarsest-to-finest).
    """
    max_level = pywt.dwt_max_level(len(signal), _WAVELET)
    coeffs = pywt.wavedec(signal, _WAVELET, mode=_MODE, level=max_level)
    return coeffs


def inverse_bior68(coeffs: list[np.ndarray]) -> np.ndarray:
    """Inverse multi-level DWT using bior6.8."""
    return pywt.waverec(coeffs, _WAVELET, mode=_MODE)


def wavelet_to_flat(coeffs: list[np.ndarray]) -> np.ndarray:
    """Flatten wavelet coefficients into a single array."""
    return np.concatenate(coeffs)


def flat_to_wavelet(flat: np.ndarray, level_sizes: list[int]) -> list[np.ndarray]:
    """Unflatten back to list-of-arrays using stored level sizes."""
    result = []
    idx = 0
    for size in level_sizes:
        result.append(flat[idx:idx + size])
        idx += size
    return result


def sparse_reconstruct_fourier(signal: np.ndarray, k: int) -> np.ndarray:
    """Keep only the k largest Fourier coefficients, reconstruct."""
    F = np.fft.fft(signal)
    magnitudes = np.abs(F)
    # Keep k largest
    threshold = np.sort(magnitudes)[::-1][min(k, len(magnitudes) - 1)]
    F_sparse = F.copy()
    F_sparse[magnitudes < threshold] = 0
    # Count actual nonzeros (may differ from k due to ties)
    return np.real(np.fft.ifft(F_sparse)), np.sum(magnitudes >= threshold)


def sparse_reconstruct_wavelet(signal: np.ndarray, k: int) -> np.ndarray:
    """Keep only the k largest wavelet coefficients, reconstruct."""
    coeffs = bior68_wavelet_transform(signal)
    level_sizes = [len(c) for c in coeffs]
    flat = wavelet_to_flat(coeffs)
    magnitudes = np.abs(flat)
    threshold = np.sort(magnitudes)[::-1][min(k, len(magnitudes) - 1)]
    flat_sparse = flat.copy()
    flat_sparse[magnitudes < threshold] = 0
    coeffs_sparse = flat_to_wavelet(flat_sparse, level_sizes)
    return inverse_bior68(coeffs_sparse), np.sum(magnitudes >= threshold)


def main():
    n = 1024
    x, smooth, with_fold = make_signals(n)

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(6, 2, figure=fig, hspace=0.55, wspace=0.3,
                  top=0.95, bottom=0.04)

    # --- Row 1: The two signals ---
    ax_smooth = fig.add_subplot(gs[0, 0])
    ax_smooth.plot(x, smooth, 'k', linewidth=1)
    ax_smooth.set_title("Signal A: Smooth\n(like a draped cape)", fontsize=10)
    ax_smooth.set_xlim(0, 1)
    ax_smooth.set_ylabel("displacement")

    ax_fold = fig.add_subplot(gs[0, 1])
    ax_fold.plot(x, with_fold, 'k', linewidth=1)
    ax_fold.axvspan(0.57, 0.63, alpha=0.15, color='red', label='fold region')
    ax_fold.set_title("Signal B: Smooth + localized sharp fold\n(like a cape with one crease)", fontsize=10)
    ax_fold.set_xlim(0, 1)
    ax_fold.legend()

    # --- Row 2: Fourier coefficients (both signals) ---
    F_smooth = np.abs(np.fft.fft(smooth))[:n // 2]
    F_fold = np.abs(np.fft.fft(with_fold))[:n // 2]
    freqs = np.arange(n // 2)

    ax_fc_smooth = fig.add_subplot(gs[1, 0])
    ax_fc_smooth.stem(freqs[:50], F_smooth[:50], linefmt='b-', markerfmt='b.', basefmt='k-')
    ax_fc_smooth.set_title("Fourier coefficients (smooth signal)", fontsize=9)
    ax_fc_smooth.set_xlabel("frequency index")
    ax_fc_smooth.set_ylabel("|coefficient|")

    ax_fc_fold = fig.add_subplot(gs[1, 1])
    ax_fc_fold.stem(freqs[:50], F_fold[:50], linefmt='r-', markerfmt='r.', basefmt='k-')
    ax_fc_fold.set_title("Fourier coefficients (fold signal)", fontsize=9)
    ax_fc_fold.set_xlabel("frequency index")
    ax_fc_fold.set_ylabel("|coefficient|")

    # --- Row 3: Wavelet coefficients (both signals) ---
    W_smooth_coeffs = bior68_wavelet_transform(smooth)
    W_fold_coeffs = bior68_wavelet_transform(with_fold)
    W_smooth_flat = wavelet_to_flat(W_smooth_coeffs)
    W_fold_flat = wavelet_to_flat(W_fold_coeffs)

    ax_wc_smooth = fig.add_subplot(gs[2, 0])
    ax_wc_smooth.stem(np.arange(len(W_smooth_flat[:100])), np.abs(W_smooth_flat[:100]),
                      linefmt='b-', markerfmt='b.', basefmt='k-')
    ax_wc_smooth.set_title("Wavelet (bior6.8) coefficients (smooth signal)", fontsize=9)
    ax_wc_smooth.set_xlabel("coefficient index (coarse→fine)")
    ax_wc_smooth.set_ylabel("|coefficient|")

    ax_wc_fold = fig.add_subplot(gs[2, 1])
    ax_wc_fold.stem(np.arange(len(W_fold_flat[:100])), np.abs(W_fold_flat[:100]),
                    linefmt='r-', markerfmt='r.', basefmt='k-')
    ax_wc_fold.set_title("Wavelet (bior6.8) coefficients (fold signal)",
                         fontsize=9)
    ax_wc_fold.set_xlabel("coefficient index (coarse→fine)")
    ax_wc_fold.set_ylabel("|coefficient|")

    # --- Row 4: Fourier reconstruction ---
    ks = [5, 10, 20, 50]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(ks)))

    ax_recon_fourier = fig.add_subplot(gs[3, :])
    ax_recon_fourier.plot(x, with_fold, 'k-', linewidth=2, alpha=0.3, label='ground truth')
    for k, color in zip(ks, colors):
        recon, actual_k = sparse_reconstruct_fourier(with_fold, k)
        err = np.sqrt(np.mean((recon - with_fold) ** 2))
        ax_recon_fourier.plot(x, recon, color=color, linewidth=1,
                             label=f'k={actual_k}, RMSE={err:.3f}')
    ax_recon_fourier.set_title("Fourier reconstruction of fold signal (keep top-k coefficients)",
                               fontsize=9)
    ax_recon_fourier.legend(fontsize=8, ncol=3)
    ax_recon_fourier.set_xlim(0, 1)
    ax_recon_fourier.set_ylabel("displacement")

    # --- Row 5: Wavelet reconstruction ---
    ax_recon_wavelet = fig.add_subplot(gs[4, :])
    ax_recon_wavelet.plot(x, with_fold, 'k-', linewidth=2, alpha=0.3, label='ground truth')
    for k, color in zip(ks, colors):
        recon, actual_k = sparse_reconstruct_wavelet(with_fold, k)
        err = np.sqrt(np.mean((recon - with_fold) ** 2))
        ax_recon_wavelet.plot(x, recon, color=color, linewidth=1,
                             label=f'k={actual_k}, RMSE={err:.3f}')
    ax_recon_wavelet.set_title("Wavelet (bior6.8) reconstruction of fold signal (keep top-k coefficients)",
                               fontsize=9)
    ax_recon_wavelet.legend(fontsize=8, ncol=3)
    ax_recon_wavelet.set_xlim(0, 1)
    ax_recon_wavelet.set_ylabel("displacement")

    # --- Row 6: Error vs. k comparison (the punchline) ---
    ax_err = fig.add_subplot(gs[5, :])
    ks_range = np.arange(2, 80)
    errs_fourier_smooth = []
    errs_fourier_fold = []
    errs_wavelet_smooth = []
    errs_wavelet_fold = []

    for k in ks_range:
        r, _ = sparse_reconstruct_fourier(smooth, k)
        errs_fourier_smooth.append(np.sqrt(np.mean((r - smooth) ** 2)))
        r, _ = sparse_reconstruct_fourier(with_fold, k)
        errs_fourier_fold.append(np.sqrt(np.mean((r - with_fold) ** 2)))
        r, _ = sparse_reconstruct_wavelet(smooth, k)
        errs_wavelet_smooth.append(np.sqrt(np.mean((r - smooth) ** 2)))
        r, _ = sparse_reconstruct_wavelet(with_fold, k)
        errs_wavelet_fold.append(np.sqrt(np.mean((r - with_fold) ** 2)))

    ax_err.semilogy(ks_range, errs_fourier_smooth, 'b--', linewidth=1.5,
                    label='Fourier × smooth signal')
    ax_err.semilogy(ks_range, errs_fourier_fold, 'r--', linewidth=2,
                    label='Fourier × fold signal ← SLOW DECAY (eigenmodes fail here)')
    ax_err.semilogy(ks_range, errs_wavelet_smooth, 'b-', linewidth=1.5,
                    label='Wavelet × smooth signal')
    ax_err.semilogy(ks_range, errs_wavelet_fold, 'r-', linewidth=2,
                    label='Wavelet × fold signal ← FAST DECAY (localized basis wins)')
    ax_err.set_xlabel("Number of coefficients kept (k)", fontsize=11)
    ax_err.set_ylabel("Reconstruction RMSE", fontsize=11)
    ax_err.set_title("eigenmodes (≈ Fourier) vs. "
                     "localized basis (≈ wavelets)",
                     fontsize=10, fontweight='bold')
    ax_err.legend(fontsize=9, loc='lower left')
    ax_err.axhline(0.01, color='gray', linestyle=':', alpha=0.5)
    ax_err.text(70, 0.012, 'visual threshold', fontsize=8, color='gray')
    ax_err.set_xlim(2, 80)

    # plt.suptitle("subspace eigenmodes can't represent sharp local features efficiently",
    #              fontsize=12, fontweight='bold', y=0.99)
    plt.savefig("fourier_vs_wavelet.png", dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
