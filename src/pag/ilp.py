"""§3.3 — Single-layer extraction as Integer Linear Program.

We label every vertex of ``M_proj`` with ``l_i ∈ {0, 1}`` (1 = keep, 0 = remove)
to minimise

    λ_s · Σ_{ij ∈ E_proj}  w_s^{ij} · 𝟙[l_i ≠ l_j]                  (smoothness)
  + λ_o · Σ_{ij ∈ E_o} (𝟙[l_i = l_j]
                       + λ_bias · 𝟙[d_i ≠ d_j ∧ (l_i − l_j) ≠ sign(d_i − d_j)])
                                                                   (opposite + bias)

The opposite term *wants* disagreement (the matching layer should be removed),
so the Boykov–Kolmogorov regularity condition

    E(0,0) + E(1,1) ≤ E(1,0) + E(0,1)

is **violated** — graph cut would not certify optimality. Komodakis–Tziritas
(2007) reformulate the joint-label term as an LP over auxiliary variables

    y^p_{ab} ∈ [0, 1]  for p ∈ pairs, a ∈ {0,1}, b ∈ {0,1}

with marginal-consistency constraints

    y^p_{0·} = 1 − l_i       y^p_{1·} = l_i
    y^p_{·0} = 1 − l_j       y^p_{·1} = l_j

(one of the four is redundant; we keep all four — HiGHS handles redundancy.)
The objective becomes linear:

    Σ_p Σ_{ab} c_{p,ab} · y^p_{ab}

so a generic MILP solver (HiGHS via ``scipy.optimize.milp``) returns the
integer optimum. The LP relaxation polytope has integer extreme points when
the ``l`` are integer, so HiGHS solves few branch nodes in practice.

Variable layout (length ``N + 4·P``):
  - ``z[0:N]``: ``l_0, …, l_{N-1}``  (binary)
  - ``z[N + 4k + 0..3]``: ``y^k_{00}, y^k_{01}, y^k_{10}, y^k_{11}``  (continuous)

Constraint layout (4 per pair):
  Row ``4k + 0``:  y^k_{00} + y^k_{01} + l_a       = 1   (l_a = 0 case)
  Row ``4k + 1``:  y^k_{10} + y^k_{11} − l_a       = 0   (l_a = 1 case)
  Row ``4k + 2``:  y^k_{00} + y^k_{10} + l_b       = 1   (l_b = 0 case)
  Row ``4k + 3``:  y^k_{01} + y^k_{11} − l_b       = 0   (l_b = 1 case)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from pag.guide_graph import GuideGraph


@dataclass
class LayerLabels:
    """Output of :func:`solve_layer_ilp`.

    Attributes
    ----------
    labels : (N,) int8
        ``1`` = keep, ``0`` = remove.
    energy : float
        Value of the objective at the integer optimum.
    n_keep, n_remove : int
        Convenience counts.
    solver_status : str
        HiGHS status message — surfaced to make solver issues debuggable.
    """
    labels: np.ndarray
    energy: float
    n_keep: int
    n_remove: int
    solver_status: str


def solve_layer_ilp(
    guide: GuideGraph,
    *,
    lambda_s: float = 1.0,
    lambda_o: float = 1.0,
    lambda_bias: float = 0.1,
    enable_outer_bias: bool = True,
    time_limit: float = 60.0,
    verbose: bool = False,
) -> LayerLabels:
    """Solve the §3.3 single-layer ILP via HiGHS.

    Parameters
    ----------
    guide : GuideGraph
        Output of :func:`pag.guide_graph.build_guide_graph`.
    lambda_s, lambda_o : float, default 1.0
        Smoothness / opposite-energy weights.
    lambda_bias : float, default 0.1
        Outer-layer-bias weight, applied multiplicatively inside ``λ_o``.
    enable_outer_bias : bool, default True
        Disable to test the un-biased opposite-energy formulation (useful for
        confirming that bias-driven label flips are reproducible).
    time_limit : float, default 60.0
        HiGHS time limit in seconds.
    verbose : bool, default False
        Pass ``disp=True`` to HiGHS — prints solver progress.

    Returns
    -------
    LayerLabels
    """
    N = int(guide.d_plus.shape[0])

    # ------------------------------------------------------------------
    # Build a unique pair set, accumulating costs from both edge sources.
    # Costs are stored in canonical (a < b) order: c[0]=E(0,0), c[1]=E(0,1),
    # c[2]=E(1,0), c[3]=E(1,1).
    # ------------------------------------------------------------------
    pair_costs: dict[tuple[int, int], np.ndarray] = {}

    def _add(a: int, b: int, c: np.ndarray) -> None:
        # Caller guarantees c is in (a, b) canonical order with a < b.
        key = (a, b)
        if key not in pair_costs:
            pair_costs[key] = np.zeros(4)
        pair_costs[key] += c

    # Smoothness: cost on (0,1) and (1,0) only — symmetric in i, j.
    for k in range(guide.mesh_edges.shape[0]):
        i = int(guide.mesh_edges[k, 0])
        j = int(guide.mesh_edges[k, 1])
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        w = lambda_s * float(guide.smoothness_w[k])
        _add(a, b, np.array([0.0, w, w, 0.0]))

    # Opposite: cost on (0,0) and (1,1) (paper's "exactly one kept"), plus
    # optional outer-layer bias.
    for k in range(guide.opposite_edges.shape[0]):
        i = int(guide.opposite_edges[k, 0])
        j = int(guide.opposite_edges[k, 1])
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)

        c = np.zeros(4)
        c[0] = lambda_o      # E(0,0)  — both removed
        c[3] = lambda_o      # E(1,1)  — both kept

        if enable_outer_bias:
            d_a = float(guide.d_plus[a])
            d_b = float(guide.d_plus[b])
            # NaN comparisons are False, so np.isnan-safe is unnecessary here.
            if d_a != d_b:
                w_bias = lambda_o * lambda_bias
                # Penalty fires when (l_a − l_b) ≠ sign(d_a − d_b). Equivalently:
                # if d_a > d_b, prefer (l_a, l_b) = (1, 0) → penalty on the rest.
                if d_a > d_b:
                    c[0] += w_bias  # (0,0)
                    c[1] += w_bias  # (0,1)
                    c[3] += w_bias  # (1,1)
                else:  # d_a < d_b
                    c[0] += w_bias  # (0,0)
                    c[2] += w_bias  # (1,0)
                    c[3] += w_bias  # (1,1)
        _add(a, b, c)

    pairs = sorted(pair_costs.keys())
    P = len(pairs)

    # Trivial early-out: no constraints, no objective. Pick all-zeros.
    if P == 0:
        labels = np.zeros(N, dtype=np.int8)
        return LayerLabels(
            labels=labels, energy=0.0, n_keep=0, n_remove=N,
            solver_status="trivial — empty guide graph",
        )

    n_vars = N + 4 * P

    # ------------------------------------------------------------------
    # Cost vector
    # ------------------------------------------------------------------
    c_vec = np.zeros(n_vars)
    for k, p in enumerate(pairs):
        c_vec[N + 4 * k : N + 4 * (k + 1)] = pair_costs[p]

    # ------------------------------------------------------------------
    # Equality constraints: 4 per pair (see module docstring).
    # ------------------------------------------------------------------
    n_rows = 4 * P
    # 12 nonzeros per pair (3 per row).
    rows = np.empty(12 * P, dtype=np.int64)
    cols = np.empty(12 * P, dtype=np.int64)
    data = np.empty(12 * P, dtype=np.float64)
    rhs = np.empty(n_rows, dtype=np.float64)

    for k, (a, b) in enumerate(pairs):
        y0, y1, y2, y3 = (N + 4 * k + r for r in range(4))
        base = 12 * k

        # Row 4k+0: y00 + y01 + l_a = 1
        rows[base : base + 3] = 4 * k
        cols[base : base + 3] = (y0, y1, a)
        data[base : base + 3] = (1.0, 1.0, 1.0)
        rhs[4 * k] = 1.0

        # Row 4k+1: y10 + y11 − l_a = 0
        rows[base + 3 : base + 6] = 4 * k + 1
        cols[base + 3 : base + 6] = (y2, y3, a)
        data[base + 3 : base + 6] = (1.0, 1.0, -1.0)
        rhs[4 * k + 1] = 0.0

        # Row 4k+2: y00 + y10 + l_b = 1
        rows[base + 6 : base + 9] = 4 * k + 2
        cols[base + 6 : base + 9] = (y0, y2, b)
        data[base + 6 : base + 9] = (1.0, 1.0, 1.0)
        rhs[4 * k + 2] = 1.0

        # Row 4k+3: y01 + y11 − l_b = 0
        rows[base + 9 : base + 12] = 4 * k + 3
        cols[base + 9 : base + 12] = (y1, y3, b)
        data[base + 9 : base + 12] = (1.0, 1.0, -1.0)
        rhs[4 * k + 3] = 0.0

    A = csr_matrix((data, (rows, cols)), shape=(n_rows, n_vars))

    # ------------------------------------------------------------------
    # Bounds + integrality
    # ------------------------------------------------------------------
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    integrality = np.zeros(n_vars, dtype=np.int64)
    integrality[:N] = 1   # l ∈ {0, 1}; y stays continuous (integer at LP vtx).

    constraints = LinearConstraint(A, lb=rhs, ub=rhs)
    bounds = Bounds(lb=lb, ub=ub)

    res = milp(
        c=c_vec,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": float(time_limit), "disp": bool(verbose)},
    )
    if not res.success:
        raise RuntimeError(f"MILP failed (status {res.status}): {res.message}")

    labels = np.round(res.x[:N]).astype(np.int8)
    return LayerLabels(
        labels=labels,
        energy=float(res.fun),
        n_keep=int(labels.sum()),
        n_remove=int(N - labels.sum()),
        solver_status=str(res.message),
    )
