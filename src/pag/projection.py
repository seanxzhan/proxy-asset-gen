"""§3.2 — Project ``M_iso`` onto ``M_visual`` with vector Adam.

``M_iso`` is topologically clean (watertight, manifold) but loose: every vertex
sits ~one voxel off the input. We tighten it by minimising

    Σ_i ||v_i − p_visual^i||²  +  λ_L · Σ_i ||v_i − μ_i||²              (Eq §3.2)

where

  - ``p_visual^i`` is the closest point on ``M_visual``  (libigl AABB query)
  - ``μ_i = (1/|N(i)|) Σ_{j ∈ N(i)} v_j``  is the uniform-Laplacian centroid

Topology is unchanged — only ``V`` moves; ``F`` and ``edges`` are reused.

Two solvers are available, selected by ``collision_free``:

**Vector Adam** (default) [Ling 2022, "Vector-Adam"]. Standard Adam normalises
each Cartesian coordinate by its own ``√v̂``, which biases steps toward the
coordinate axes (a sphere does not stay round). Vector Adam keeps the first
moment per-vertex as a 3-vector but accumulates the *scalar* ``‖g‖²`` for the
second moment, making the per-vertex update rotation-equivariant. Defaults
β₁=0.9, β₂=0.999, ε=1e-8 mirror torch's Adam. **Self-intersection is allowed** —
fast, but the two iso sheets collapse through each other, which corrupts the
§3.3 guide graph and punches holes in ``M_single`` (see
``docs/Self-Intersection-Free-Projection.md``).

**IPC** (``collision_free=True``) — projected-Newton minimisation of the same
data + Laplacian energy plus an Incremental Potential Contact barrier
[Li 2020], with a continuous-collision-detection (CCD) filtered line search.
``M_iso`` is intersection-free by construction, so a CCD-clamped trajectory
keeps every intermediate mesh intersection-free: the two sheets are guaranteed
never to cross. The barrier activation distance ``dhat`` is a *small* fraction
of the bbox diagonal (``≪`` edge length) — its only job is to stop crossing;
the sheets settle ~``dhat`` apart, well within the §3.3 distance filter. This
is the fix for the paper-silent self-intersection check the authors confirmed
by email is required.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import igl
import numpy as np

from pag.mesh import Mesh, build_mesh


@dataclass
class ProjectionResult:
    """Output of :func:`project_to_visual`.

    ``history`` is empty unless ``record_history=True``. When populated it
    contains length-``n_iters`` arrays for each energy term, useful for
    diagnostics and the ``test_data_energy_decreases`` regression test.
    """
    mesh: Mesh
    n_iters: int
    knobs: dict[str, float] = field(default_factory=dict)
    history: dict[str, np.ndarray] = field(default_factory=dict)


def project_to_visual(
    iso_mesh: Mesh,
    V_visual: np.ndarray,
    F_visual: np.ndarray,
    *,
    n_iters: int = 200,
    lr: float = 1e-2,
    lambda_L: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    record_history: bool = False,
    # ----- IPC collision-free solver (collision_free=True) -----
    collision_free: bool = False,
    dhat: float | None = None,
    dhat_frac: float = 1e-3,
    barrier_stiffness: float | None = None,
    ccd_safety: float = 0.9,
    c_armijo: float = 1e-4,
    tol: float = 1e-5,
    tol_rel: float = 1e-3,
    max_backtracks: int = 16,
    verbose: bool = False,
) -> ProjectionResult:
    """Pull each ``M_iso`` vertex toward its closest point on ``M_visual``.

    Parameters
    ----------
    iso_mesh : Mesh
        Output of :func:`pag.udf.isosurface_from_udf`.
    V_visual, F_visual : ndarray
        The original (possibly non-manifold) visual mesh.
    n_iters : int, default 200
        Iteration cap. Vector Adam runs exactly ``n_iters``; the IPC solver
        treats it as a Newton-iteration cap and early-stops on ``tol``.
    lr : float, default 1e-2
        Adam step size. *(Vector Adam only.)*
    lambda_L : float, default 0.1
        Smoothness weight. λ_L = 0 → tightest fit but lumpy; large λ_L →
        smooth but stays loose. *(Both solvers.)*
    beta1, beta2, eps : float
        Adam moment / numerical defaults. *(Vector Adam only.)*
    record_history : bool, default False
        If True, ``result.history`` is filled with per-iter energies (handy for
        diagnostics). Vector Adam records ``data``/``smooth``/``total``; IPC
        additionally records ``barrier``/``min_dist``/``alpha``.

    collision_free : bool, default False
        If True, use the IPC + CCD solver instead of Vector Adam. Guarantees the
        output is self-intersection-free (the two iso sheets never cross), which
        §3.3 single-layer extraction needs. The Adam-only parameters above are
        ignored.
    dhat : float, optional
        IPC barrier activation distance (absolute). Defaults to
        ``dhat_frac * bbox_diagonal``. Keep it ``≪`` edge length: it only needs
        to stop crossing, not to enforce a thick gap.
    dhat_frac : float, default 1e-3
        Fraction of the bbox diagonal used when ``dhat`` is None (IPC standard).
    barrier_stiffness : float, optional
        Fixed barrier weight κ. ``None`` (default) uses IPC's adaptive stiffness
        (``initial_barrier_stiffness`` + ``update_barrier_stiffness``).
    ccd_safety : float, default 0.9
        Fraction of the CCD time-of-impact taken as the max step (``<1`` keeps a
        margin off exact contact).
    c_armijo : float, default 1e-4
        Armijo sufficient-decrease constant for the backtracking line search.
    tol : float, default 1e-5
        Strict convergence threshold: stop when ``max|Δv| < tol · bbox_diagonal``.
    tol_rel : float, default 1e-3
        Plateau threshold: stop once the barrier-free objective fails to improve
        by this relative amount for a few consecutive steps. This is what
        normally ends the loop — the data term converges in a handful of Newton
        steps and the remainder is sub-voxel jitter.
    max_backtracks : int, default 16
        Max line-search halvings per Newton step.
    verbose : bool, default False
        Print per-iteration IPC diagnostics.

    Returns
    -------
    ProjectionResult
    """
    if n_iters < 0:
        raise ValueError(f"n_iters must be ≥ 0; got {n_iters}")

    if collision_free:
        return _project_ipc(
            iso_mesh, V_visual, F_visual,
            n_iters=n_iters, lambda_L=lambda_L,
            dhat=dhat, dhat_frac=dhat_frac, barrier_stiffness=barrier_stiffness,
            ccd_safety=ccd_safety, c_armijo=c_armijo, tol=tol, tol_rel=tol_rel,
            max_backtracks=max_backtracks, record_history=record_history,
            verbose=verbose,
        )

    V = np.array(iso_mesh.V, dtype=np.float64, copy=True)
    F = iso_mesh.F
    edges = iso_mesh.edges                          # (E, 2) undirected, unique
    n = V.shape[0]

    V_v = np.ascontiguousarray(V_visual, dtype=np.float64)
    F_v = np.ascontiguousarray(F_visual, dtype=np.int64)

    # Per-vertex degree on the M_iso 1-ring (used by the uniform Laplacian).
    deg = np.zeros(n, dtype=np.int64)
    np.add.at(deg, edges[:, 0], 1)
    np.add.at(deg, edges[:, 1], 1)
    inv_deg = np.zeros(n, dtype=np.float64)
    inv_deg[deg > 0] = 1.0 / deg[deg > 0]

    # Vector-Adam state. Per-vertex first moment is a 3-vector; per-vertex
    # second moment is a *scalar* (||g||²) — the whole point of vector Adam.
    m = np.zeros_like(V)
    s = np.zeros(n, dtype=np.float64)

    hist_data: list[float] = []
    hist_smooth: list[float] = []
    hist_total: list[float] = []

    for t in range(1, n_iters + 1):
        # Closest point on M_visual for each iso vertex.
        sqrD, _, C = igl.point_mesh_squared_distance(V, V_v, F_v)

        # Uniform-Laplacian centroids μ_i = (1/deg(i)) Σ_{j ∈ N(i)} v_j.
        mu = np.zeros_like(V)
        np.add.at(mu, edges[:, 0], V[edges[:, 1]])
        np.add.at(mu, edges[:, 1], V[edges[:, 0]])
        mu *= inv_deg[:, None]
        delta = V - mu                              # (N, 3) residual

        # ∇_v_i Σ_k ||v_k − μ_k||²  =  2 δ_i − 2 Σ_{k ∈ N(i)} δ_k / deg(k).
        # (See module docstring; this is the *exact* gradient of the
        # uniform-Laplacian quadratic, not the heuristic δ_i alone.)
        weighted = delta * inv_deg[:, None]
        pulled = np.zeros_like(V)
        np.add.at(pulled, edges[:, 0], weighted[edges[:, 1]])
        np.add.at(pulled, edges[:, 1], weighted[edges[:, 0]])
        grad_smooth = 2.0 * delta - 2.0 * pulled

        grad_data = 2.0 * (V - C)                   # ∇ Σ ||v − p||²
        g = grad_data + lambda_L * grad_smooth

        # ----- Vector Adam (Ling 2022) -----------------------------------
        m = beta1 * m + (1.0 - beta1) * g
        s = beta2 * s + (1.0 - beta2) * np.einsum("ij,ij->i", g, g)
        m_hat = m / (1.0 - beta1 ** t)
        s_hat = s / (1.0 - beta2 ** t)
        V = V - lr * m_hat / (np.sqrt(s_hat)[:, None] + eps)

        if record_history:
            data = float(sqrD.sum())
            smooth = float((delta * delta).sum())
            hist_data.append(data)
            hist_smooth.append(smooth)
            hist_total.append(data + lambda_L * smooth)

    history: dict[str, np.ndarray] = {}
    if record_history:
        history = {
            "data": np.asarray(hist_data),
            "smooth": np.asarray(hist_smooth),
            "total": np.asarray(hist_total),
        }

    return ProjectionResult(
        mesh=build_mesh(V, F),
        n_iters=n_iters,
        knobs={
            "lr": lr, "lambda_L": lambda_L,
            "beta1": beta1, "beta2": beta2, "eps": eps,
        },
        history=history,
    )


# ======================================================================
# IPC + CCD collision-free solver (collision_free=True)
# ======================================================================

# Early-stop guards: the data term converges in a few Newton steps; the tail is
# sub-voxel jitter as the barrier resolves marching-cubes slivers. Stop once the
# barrier-free objective stops improving for _IPC_PATIENCE consecutive steps,
# but never before _IPC_MIN_ITERS.
_IPC_PATIENCE = 3
_IPC_MIN_ITERS = 3


def _uniform_laplacian(edges: np.ndarray, n: int):
    """Sparse uniform-Laplacian operator ``L = I − A`` and its normal ``LᵀL``.

    ``A`` is the row-stochastic 1-ring averaging matrix (``A_ij = 1/deg(i)`` for
    ``j ∈ N(i)``), so the smoothness energy ``Σ_i ‖v_i − μ_i‖² = ‖L V‖²_F`` and
    its gradient / Hessian are ``2 LᵀL V`` / ``2 LᵀL`` — identical in spirit to
    the explicit uniform-Laplacian gradient the Vector-Adam path computes.
    """
    import scipy.sparse as sp

    i = edges[:, 0]
    j = edges[:, 1]
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    deg = np.bincount(rows, minlength=n).astype(np.float64)
    deg[deg == 0] = 1.0
    data = 1.0 / deg[rows]
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    L = (sp.identity(n, format="csr") - A).tocsc()
    LtL = (L.T @ L).tocsc()
    return L, LtL


def _project_ipc(
    iso_mesh: Mesh,
    V_visual: np.ndarray,
    F_visual: np.ndarray,
    *,
    n_iters: int,
    lambda_L: float,
    dhat: float | None,
    dhat_frac: float,
    barrier_stiffness: float | None,
    ccd_safety: float,
    c_armijo: float,
    tol: float,
    tol_rel: float,
    max_backtracks: int,
    record_history: bool,
    verbose: bool,
) -> ProjectionResult:
    """Projected-Newton projection with an IPC self-contact barrier and a
    CCD-filtered line search. See :func:`project_to_visual` for parameters.

    Energy (per Newton step, with closest points ``C`` frozen — Gauss–Newton):

        E(V) = Σ‖v_i − C_i‖²  +  λ_L ‖L V‖²  +  κ · B(V; dhat)

    Hessian = ``2I  +  2 λ_L LᵀL  +  κ ∇²B`` (barrier block PSD-projected), solved
    for a descent direction ``p``; the step ``α`` is capped at ``ccd_safety`` ×
    the CCD time-of-impact and Armijo-backtracked. Because ``M_iso`` starts
    intersection-free and ``α`` never crosses a collision, every iterate — and
    the result — is intersection-free.
    """
    import ipctk
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    V = np.ascontiguousarray(iso_mesh.V, dtype=np.float64)
    F = iso_mesh.F
    E = iso_mesh.edges
    n = V.shape[0]

    V_v = np.ascontiguousarray(V_visual, dtype=np.float64)
    F_v = np.ascontiguousarray(F_visual, dtype=np.int64)

    # ipctk wants fortran-contiguous int32 connectivity / float64 vertices.
    E_i = np.asfortranarray(E.astype(np.int32))
    F_i = np.asfortranarray(F.astype(np.int32))
    cmesh = ipctk.CollisionMesh(np.asfortranarray(V), E_i, F_i)
    if cmesh.ndof != 3 * n:                       # safety: we assume identity DOF map
        raise RuntimeError(
            f"CollisionMesh DOF map is non-identity (ndof={cmesh.ndof}, 3N={3 * n})"
        )

    diag = float(ipctk.world_bbox_diagonal_length(np.asfortranarray(V)))
    if dhat is None:
        dhat = dhat_frac * diag
    dhat = float(dhat)

    if ipctk.has_intersections(cmesh, np.asfortranarray(V)):
        raise ValueError(
            "M_iso is already self-intersecting; the IPC solver needs an "
            "intersection-free start. Check the §3.1 isosurface."
        )

    barrier = ipctk.BarrierPotential(dhat)
    L, LtL = _uniform_laplacian(E, n)
    # Constant Hessian blocks (vertex-major DOF order 3i+c, matching the
    # C-order reshape of (N,3) gradients — verified against ipctk by FD).
    I3 = sp.identity(3, format="csc")
    H_const = (2.0 * sp.identity(3 * n, format="csc")          # data: 2I
               + 2.0 * lambda_L * sp.kron(LtL, I3, format="csc"))  # smoothness

    def build_collisions(Vq: np.ndarray):
        nc = ipctk.NormalCollisions()
        nc.build(cmesh, np.asfortranarray(Vq), dhat)
        return nc

    def barrier_energy(Vq: np.ndarray, nc) -> float:
        if nc.empty():
            return 0.0
        return float(barrier(nc, cmesh, np.asfortranarray(Vq)))

    def total_energy(Vq: np.ndarray, C: np.ndarray, kappa: float) -> float:
        data = float(((Vq - C) ** 2).sum())
        res = L @ Vq
        smooth = float((res * res).sum())
        eb = barrier_energy(Vq, build_collisions(Vq))
        return data + lambda_L * smooth + kappa * eb

    kappa = barrier_stiffness
    kappa_max = np.inf
    kappa_set = kappa is not None
    prev_min_d = np.inf

    hist: dict[str, list[float]] = {
        "data": [], "smooth": [], "barrier": [], "total": [],
        "min_dist": [], "alpha": [],
    }

    best_data = np.inf
    stale = 0
    it_done = 0
    for it in range(n_iters):
        it_done = it + 1
        Vf = np.asfortranarray(V)

        # Data term: closest point on M_visual (frozen within this Newton step).
        sqrD, _, C = igl.point_mesh_squared_distance(V, V_v, F_v)
        g_data = 2.0 * (V - C)                              # (N, 3)
        g_smooth = 2.0 * (LtL @ V)                          # (N, 3)
        res0 = L @ V
        data_e = float(sqrD.sum())
        smooth_e = float((res0 * res0).sum())
        obj = data_e + lambda_L * smooth_e                  # barrier-free objective

        nc = build_collisions(V)
        if nc.empty():
            min_d = np.inf
            e_bar = 0.0
            g_bar = np.zeros((n, 3))
            H_bar = sp.csc_matrix((3 * n, 3 * n))
        else:
            min_d = float(nc.compute_minimum_distance(cmesh, Vf))
            e_bar = float(barrier(nc, cmesh, Vf))
            g_bar = barrier.gradient(nc, cmesh, Vf).reshape(n, 3)
            H_bar = barrier.hessian(nc, cmesh, Vf, ipctk.PSDProjectionMethod.CLAMP)

        # Barrier stiffness κ: IPC adaptive scheme (set on first active step,
        # then nudged as the min distance shrinks toward dhat).
        if not kappa_set and not nc.empty():
            g_e = (g_data + lambda_L * g_smooth).reshape(-1, 1)
            g_b = g_bar.reshape(-1, 1)
            kappa, kappa_max = ipctk.initial_barrier_stiffness(
                diag, barrier.barrier, dhat, 1.0, g_e, g_b
            )
            kappa_set = True
            prev_min_d = min_d
        elif kappa_set and barrier_stiffness is None and not nc.empty():
            kappa = ipctk.update_barrier_stiffness(
                prev_min_d, min_d, kappa_max, kappa, diag
            )
            prev_min_d = min_d
        k = kappa if kappa_set else 1.0

        g_flat = (g_data + lambda_L * g_smooth + k * g_bar).reshape(-1)

        # Newton direction: (2I + 2λ_L LᵀL + κ ∇²B + εI) p = −g.
        H = (H_const + k * H_bar + 1e-9 * sp.identity(3 * n, format="csc")).tocsc()
        p = spsolve(H, -g_flat).reshape(n, 3)

        # CCD-filtered, Armijo-backtracked line search. E0 reuses the energies
        # already evaluated at V (no extra collision build).
        alpha = min(1.0, ccd_safety * float(ipctk.compute_collision_free_stepsize(
            cmesh, Vf, np.asfortranarray(V + p))))
        E0 = obj + k * e_bar
        gTp = float(g_flat @ p.reshape(-1))                 # < 0 (descent)
        for _ in range(max_backtracks):
            if total_energy(V + alpha * p, C, k) <= E0 + c_armijo * alpha * gTp:
                break
            alpha *= 0.5
        V = V + alpha * p
        step_inf = float(alpha * np.abs(p).max()) if p.size else 0.0

        if record_history:
            hist["data"].append(data_e)
            hist["smooth"].append(smooth_e)
            hist["barrier"].append(e_bar)
            hist["total"].append(obj)
            hist["min_dist"].append(min_d)
            hist["alpha"].append(alpha)
        if verbose:
            print(
                f"  ipc it {it:3d}  data={data_e:.4e}  min_d={min_d:.3e}  "
                f"κ={k:.3e}  α={alpha:.3e}  |Δv|∞={step_inf:.3e}"
            )

        # Convergence: strict step bound, or fit plateau (patience). The plateau
        # is measured on the *data* term — the closest-point fit is what we care
        # about; the smoothness term drifts down forever (Laplacian never fully
        # stops), so including it would mask the plateau.
        if data_e < best_data * (1.0 - tol_rel):
            best_data = data_e
            stale = 0
        else:
            stale += 1
        if step_inf < tol * diag:
            break
        if stale >= _IPC_PATIENCE and it_done >= _IPC_MIN_ITERS:
            break

    history: dict[str, np.ndarray] = {}
    if record_history:
        history = {key: np.asarray(val) for key, val in hist.items()}

    return ProjectionResult(
        mesh=build_mesh(V, F),
        n_iters=it_done,
        knobs={
            "solver": "ipc", "lambda_L": lambda_L, "dhat": dhat,
            "dhat_frac": dhat_frac,
            "barrier_stiffness": (-1.0 if barrier_stiffness is None
                                  else barrier_stiffness),
            "ccd_safety": ccd_safety, "c_armijo": c_armijo, "tol": tol,
            "tol_rel": tol_rel,
        },
        history=history,
    )
