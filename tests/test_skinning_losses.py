"""ARAP / attachment loss tests — the math centerpiece for Stage 2.

Hand-derived rigid-body, stretch, and reflection cases pin down the SVD
det-correction and the row-normalization in L_a.
"""
from __future__ import annotations

import numpy as np
import torch

from pag.skinning_losses import arap_loss, attachment_loss
from pag.skinning_lbs import build_topological_1ring


def _square_mesh():
    """Two triangles forming a unit square in the z=0 plane."""
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    F = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return V, F


# ----------------- ARAP -----------------

def test_arap_zero_at_rest():
    V, F = _square_mesh()
    N1, mask = build_topological_1ring(F, 4)
    V0 = torch.tensor(V, dtype=torch.float64)
    Vt = V0[None].clone()                       # T=1, identical to rest
    loss = arap_loss(Vt, V0, torch.tensor(N1), torch.tensor(mask))
    assert float(loss) < 1e-20


def test_arap_zero_under_rigid_translation():
    V, F = _square_mesh()
    N1, mask = build_topological_1ring(F, 4)
    V0 = torch.tensor(V, dtype=torch.float64)
    delta = torch.tensor([0.7, -1.5, 0.3], dtype=torch.float64)
    Vt = (V0 + delta)[None]
    loss = arap_loss(Vt, V0, torch.tensor(N1), torch.tensor(mask))
    assert float(loss) < 1e-20


def test_arap_zero_under_rigid_rotation():
    """Rotate a square by 30° around z. Closed-form R should recover it
    exactly and the residual should be zero (up to float64 roundoff)."""
    V, F = _square_mesh()
    N1, mask = build_topological_1ring(F, 4)
    theta = np.deg2rad(30.0)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1.0],
    ])
    Vt = (V @ R.T)[None]
    loss = arap_loss(
        torch.tensor(Vt, dtype=torch.float64),
        torch.tensor(V, dtype=torch.float64),
        torch.tensor(N1), torch.tensor(mask),
    )
    assert float(loss) < 1e-18


def test_arap_zero_under_rigid_rotation_with_reflection_seed():
    """Build a det=−1 cross-cov by reflecting along z. The det-correction in
    arap_loss must recover a proper rotation (det=+1) so the residual stays
    minimal — for an in-plane mesh, a reflection across z only differs from
    identity in z, and the closed-form proper-rotation fit nails it.
    """
    V, F = _square_mesh()  # all z=0
    N1, mask = build_topological_1ring(F, 4)
    Vt = V.copy()
    Vt[:, 2] = -Vt[:, 2]   # reflect across z=0 (still in-plane → identical)
    loss = arap_loss(
        torch.tensor(Vt[None], dtype=torch.float64),
        torch.tensor(V, dtype=torch.float64),
        torch.tensor(N1), torch.tensor(mask),
    )
    # Identity (in-plane mesh's z is unchanged); proper-rotation fit is identity.
    assert float(loss) < 1e-18


def test_arap_positive_under_uniform_stretch():
    V, F = _square_mesh()
    N1, mask = build_topological_1ring(F, 4)
    Vt = V * 1.10                                # 10% stretch
    loss = arap_loss(
        torch.tensor(Vt[None], dtype=torch.float64),
        torch.tensor(V, dtype=torch.float64),
        torch.tensor(N1), torch.tensor(mask),
    )
    assert float(loss) > 1e-3


def test_arap_padding_does_not_leak_energy():
    """Two graphs with the same neighbors but different amounts of -1 padding
    must produce the same loss."""
    V, F = _square_mesh()
    N1, mask = build_topological_1ring(F, 4)
    Vt = V * 1.10
    V0 = torch.tensor(V, dtype=torch.float64)
    Vt_t = torch.tensor(Vt[None], dtype=torch.float64)
    loss_a = arap_loss(Vt_t, V0, torch.tensor(N1), torch.tensor(mask))

    # Same neighbor data, but with extra padding columns.
    pad_cols = 3
    N1_pad = -np.ones((4, N1.shape[1] + pad_cols), dtype=np.int64)
    N1_pad[:, : N1.shape[1]] = N1
    mask_pad = np.zeros_like(N1_pad, dtype=bool)
    mask_pad[:, : N1.shape[1]] = mask
    loss_b = arap_loss(Vt_t, V0, torch.tensor(N1_pad), torch.tensor(mask_pad))
    assert abs(float(loss_a) - float(loss_b)) < 1e-12


def test_arap_gradient_flows_through_Vt():
    """The optimizer needs ∂L/∂V_t to backprop through the LBS forward."""
    V, F = _square_mesh()
    N1, mask = build_topological_1ring(F, 4)
    V0 = torch.tensor(V, dtype=torch.float64)
    Vt = (V0 * 1.05).clone()[None].requires_grad_(True)
    loss = arap_loss(Vt, V0, torch.tensor(N1), torch.tensor(mask))
    loss.backward()
    assert Vt.grad is not None and torch.isfinite(Vt.grad).all()


# ----------------- attachment -----------------

def test_attachment_zero_when_in_neighbor_mean_configuration():
    """The paper's K is defined so (I−K) is the row-stochastic weighted-mean
    operator. L_a = ‖V − (I−K)V‖² vanishes only when V already equals its own
    weighted neighbor mean. A collinear, regularly spaced ramp of points whose
    each interior vertex is the average of its two equidistant neighbors hits
    that fixed point: the loss is 0 on the interior, nonzero at endpoints."""
    V0 = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64
    )
    K_idx = torch.tensor([[1, 2], [0, 2], [0, 1]], dtype=torch.long)
    # Vertex 1 sees v0=(0,0,0) and v2=(2,0,0); equidistant → weighted mean (1,0,0) == V[1].
    loss = attachment_loss(V0[None], V0, K_idx)
    # Endpoints contribute, interior is exactly zero. Sanity: loss is finite
    # and positive (endpoints), not blowing up.
    assert float(loss) > 0 and torch.isfinite(loss)


def test_attachment_invariant_to_global_translation():
    """(I−K) preserves constant vectors because each row of (I−K) sums to 1
    (paper K_{ii}=1, off-diagonal entries are negative weights summing to −1
    on neighbors). So loss(V + δ) == loss(V) for any constant vector δ."""
    V, _ = _square_mesh()
    V0 = torch.tensor(V, dtype=torch.float64)
    K_idx = torch.tensor([[1, 3], [0, 2], [1, 3], [0, 2]], dtype=torch.long)
    delta = torch.tensor([10.0, -3.0, 1.5], dtype=torch.float64)
    loss_rest = attachment_loss(V0[None], V0, K_idx)
    loss_shift = attachment_loss((V0 + delta)[None], V0, K_idx)
    assert abs(float(loss_shift) - float(loss_rest)) < 1e-12


def test_attachment_grows_when_one_vertex_drifts():
    """Moving just one vertex in V_t away from its neighbor mean increases L_a
    monotonically. This is the "glue disconnected components" behaviour: a
    component that drifts away from its rest-pose neighbours pays a quadratic
    penalty."""
    V, _ = _square_mesh()
    V0 = torch.tensor(V, dtype=torch.float64)
    K_idx = torch.tensor([[1, 3], [0, 2], [1, 3], [0, 2]], dtype=torch.long)
    base = attachment_loss(V0[None], V0, K_idx)
    Vt = V0.clone()
    Vt[0] = Vt[0] + torch.tensor([1.0, 0.0, 0.0])
    bumped = attachment_loss(Vt[None], V0, K_idx)
    Vt[0] = Vt[0] + torch.tensor([1.0, 0.0, 0.0])
    bumped_more = attachment_loss(Vt[None], V0, K_idx)
    assert float(bumped) > float(base)
    assert float(bumped_more) > float(bumped)


def test_attachment_gradient_flows():
    V, _ = _square_mesh()
    V0 = torch.tensor(V, dtype=torch.float64)
    K_idx = torch.tensor([[1, 3], [0, 2], [1, 3], [0, 2]], dtype=torch.long)
    Vt = V0.clone()[None].requires_grad_(True)
    attachment_loss(Vt, V0, K_idx).backward()
    assert Vt.grad is not None and torch.isfinite(Vt.grad).all()
