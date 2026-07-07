"""Subspace Dynamics: interactive tutorial with polyscope.

Demonstrates the key ideas of reduced-order simulation on a simple beam:

1. FULL-SPACE simulation: PBD with all N vertices as DOFs.
2. SUBSPACE simulation: project into m eigenmodes, integrate in reduced space,
   reconstruct full positions. Show that m=6 smooth modes capture the global
   swing but miss local features.
3. COMPARISON: side-by-side, with residual (what the subspace can't represent)
   shown as a color map.

Controls (polyscope GUI):
  - "num_modes" slider: change how many eigenmodes the subspace uses (2..32)
  - "apply_force" checkbox: toggle an external point load
  - "reset" button: restart both simulations

The beam is a simple 2D grid extruded slightly in z (thin plate), pinned at
the left edge. Gravity + optional point load at the tip.

Subspace construction:
  - Build the mesh Laplacian (cotangent weights)
  - Compute the first m generalized eigenvectors of (L, M)
  - These are the "skinning eigenmodes" (same idea as FreeForm/Simplicits)

Subspace dynamics:
  - Project forces into subspace: f_reduced = W^T @ f_full
  - Integrate reduced DOFs: z_{t+1} via implicit Euler (small system, direct solve)
  - Reconstruct: x = x_rest + W @ z

Run:
    python scripts/visualizations/subspace_dynamics_tutorial.py
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import polyscope as ps
import polyscope.imgui as psim


# ============================================================================
# Mesh generation
# ============================================================================

def make_beam_mesh(nx: int = 30, ny: int = 6, length: float = 2.0,
                   width: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """Create a flat rectangular beam mesh (triangulated)."""
    xs = np.linspace(0, length, nx)
    ys = np.linspace(-width / 2, width / 2, ny)
    V = np.zeros((nx * ny, 3))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            idx = i * ny + j
            V[idx] = [x, y, 0.0]

    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            v00 = i * ny + j
            v10 = (i + 1) * ny + j
            v01 = i * ny + (j + 1)
            v11 = (i + 1) * ny + (j + 1)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    return V, np.array(faces, dtype=np.int64)


# ============================================================================
# Laplacian & mass matrix (for eigenmode computation)
# ============================================================================

def cotangent_laplacian(V: np.ndarray, F: np.ndarray):
    """Compute cotangent Laplacian L and lumped mass matrix M."""
    n = V.shape[0]
    L = sp.lil_matrix((n, n), dtype=np.float64)
    M = np.zeros(n)

    for tri in F:
        i, j, k = tri
        vi, vj, vk = V[i], V[j], V[k]

        # Edge vectors
        eij = vj - vi
        eik = vk - vi
        ejk = vk - vj

        # Cotangent weights
        area = 0.5 * np.linalg.norm(np.cross(eij, eik))
        if area < 1e-12:
            continue

        cot_i = np.dot(eij, eik) / (2.0 * area)
        cot_j = np.dot(-eij, ejk) / (2.0 * area)
        cot_k = np.dot(-eik, -ejk) / (2.0 * area)

        # Assemble (j,k) edge with weight cot_i, etc.
        L[j, k] -= cot_i
        L[k, j] -= cot_i
        L[i, k] -= cot_j
        L[k, i] -= cot_j
        L[i, j] -= cot_k
        L[j, i] -= cot_k

        L[i, i] += cot_j + cot_k
        L[j, j] += cot_i + cot_k
        L[k, k] += cot_i + cot_j

        # Lumped mass (area / 3 per vertex)
        M[i] += area / 3.0
        M[j] += area / 3.0
        M[k] += area / 3.0

    return sp.csc_matrix(L), sp.diags(M)


# ============================================================================
# Subspace construction
# ============================================================================

def compute_eigenmodes(V: np.ndarray, F: np.ndarray, pinned: np.ndarray,
                       num_modes: int = 32):
    """Compute the first `num_modes` non-trivial eigenmodes of (L, M).

    Returns W: (n, num_modes) matrix of per-vertex scalar weights.
    For 3D displacement, we apply each mode independently to x, y, z
    (translation-only subspace, like Zheng et al. / FreeForm).
    """
    n = V.shape[0]
    L, M = cotangent_laplacian(V, F)

    # Pin boundary: zero out rows/cols for pinned vertices
    free = np.ones(n, dtype=bool)
    free[pinned] = False
    free_idx = np.where(free)[0]

    L_free = L[np.ix_(free_idx, free_idx)]
    M_free = sp.diags(M.diagonal()[free_idx])

    # Solve generalized eigenvalue problem: L @ v = λ M @ v
    # We want the smallest non-trivial eigenvalues (lowest energy modes)
    k = min(num_modes, len(free_idx) - 2)
    eigenvalues, eigenvectors = spla.eigsh(L_free, k=k, M=M_free, which='SM',
                                           sigma=0.01)

    # Expand back to full vertex set (pinned vertices get zero weight)
    W = np.zeros((n, k))
    W[free_idx, :] = eigenvectors

    # Normalize each mode
    M_diag = M.diagonal()
    for i in range(k):
        norm = np.sqrt(np.sum(W[:, i] ** 2 * M_diag))
        if norm > 1e-10:
            W[:, i] /= norm

    return W, eigenvalues


# ============================================================================
# Full-space dynamics (explicit Euler + damping, simple for tutorial)
# ============================================================================

class FullSpaceSim:
    def __init__(self, V: np.ndarray, F: np.ndarray, pinned: np.ndarray,
                 dt: float = 0.005, damping: float = 0.98, stiffness: float = 50.0):
        self.V_rest = V.copy()
        self.V = V.copy()
        self.vel = np.zeros_like(V)
        self.F = F
        self.pinned = pinned
        self.dt = dt
        self.damping = damping
        self.stiffness = stiffness
        self.n = V.shape[0]

        # Precompute rest-length edges for spring forces
        edges = set()
        for tri in F:
            for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                edges.add((min(a, b), max(a, b)))
        self.edges = np.array(list(edges), dtype=np.int64)
        self.rest_lengths = np.linalg.norm(
            V[self.edges[:, 0]] - V[self.edges[:, 1]], axis=1)

    def step(self, gravity: np.ndarray, ext_force: np.ndarray | None = None):
        forces = np.zeros_like(self.V)
        forces[:] = gravity[None, :]

        # Spring forces (Hookean)
        for idx, (i, j) in enumerate(self.edges):
            d = self.V[j] - self.V[i]
            length = np.linalg.norm(d)
            if length < 1e-10:
                continue
            rest = self.rest_lengths[idx]
            f = self.stiffness * (length - rest) * d / length
            forces[i] += f
            forces[j] -= f

        if ext_force is not None:
            forces += ext_force

        # Pin constraints
        forces[self.pinned] = 0.0

        # Integrate
        self.vel += forces * self.dt
        self.vel *= self.damping
        self.vel[self.pinned] = 0.0
        self.V += self.vel * self.dt

    def reset(self):
        self.V = self.V_rest.copy()
        self.vel = np.zeros_like(self.V)


# ============================================================================
# Subspace dynamics
# ============================================================================

class SubspaceSim:
    """Reduced-order simulation using m eigenmodes.

    Deformation model (translation-only, like Zheng et al.):
        x_i = x_rest_i + sum_j W_ij * z_j   (for each coordinate independently)

    Here z ∈ R^(m*3) are the reduced DOFs (m modes × 3 coordinates).
    """

    def __init__(self, V: np.ndarray, F: np.ndarray, pinned: np.ndarray,
                 W: np.ndarray, eigenvalues: np.ndarray,
                 dt: float = 0.005, damping: float = 0.98, stiffness: float = 50.0):
        self.V_rest = V.copy()
        self.V = V.copy()
        self.F = F
        self.pinned = pinned
        self.dt = dt
        self.damping = damping
        self.stiffness = stiffness
        self.n = V.shape[0]

        self.W = W  # (n, m)
        self.m = W.shape[1]
        self.eigenvalues = eigenvalues

        # Reduced DOFs: z ∈ R^(m, 3) — one coefficient per mode per coordinate
        self.z = np.zeros((self.m, 3))
        self.z_vel = np.zeros((self.m, 3))

        # Precompute edges (same as full-space)
        edges = set()
        for tri in F:
            for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                edges.add((min(a, b), max(a, b)))
        self.edges = np.array(list(edges), dtype=np.int64)
        self.rest_lengths = np.linalg.norm(
            V[self.edges[:, 0]] - V[self.edges[:, 1]], axis=1)

    def reconstruct(self) -> np.ndarray:
        """x = x_rest + W @ z (per coordinate)."""
        return self.V_rest + self.W @ self.z

    def step(self, gravity: np.ndarray, ext_force: np.ndarray | None = None):
        # Compute full-space forces at current configuration
        self.V = self.reconstruct()
        forces = np.zeros_like(self.V)
        forces[:] = gravity[None, :]

        # Spring forces
        for idx, (i, j) in enumerate(self.edges):
            d = self.V[j] - self.V[i]
            length = np.linalg.norm(d)
            if length < 1e-10:
                continue
            rest = self.rest_lengths[idx]
            f = self.stiffness * (length - rest) * d / length
            forces[i] += f
            forces[j] -= f

        if ext_force is not None:
            forces += ext_force

        forces[self.pinned] = 0.0

        # PROJECT forces into subspace: f_reduced = W^T @ f_full
        f_reduced = self.W.T @ forces  # (m, 3)

        # Integrate in reduced space
        self.z_vel += f_reduced * self.dt
        self.z_vel *= self.damping
        self.z += self.z_vel * self.dt

        # Reconstruct for visualization
        self.V = self.reconstruct()

    def reset(self):
        self.z = np.zeros((self.m, 3))
        self.z_vel = np.zeros((self.m, 3))
        self.V = self.V_rest.copy()

    def update_modes(self, W: np.ndarray, eigenvalues: np.ndarray):
        """Swap in a new set of modes (when user changes slider)."""
        self.W = W
        self.m = W.shape[1]
        self.eigenvalues = eigenvalues
        self.z = np.zeros((self.m, 3))
        self.z_vel = np.zeros((self.m, 3))
        self.V = self.V_rest.copy()


# ============================================================================
# Main
# ============================================================================

def main():
    # Build mesh
    nx, ny = 25, 5
    V, F = make_beam_mesh(nx=nx, ny=ny, length=2.0, width=0.3)
    n = V.shape[0]

    # Pin left edge
    pinned = np.array([i * ny + j for i in range(2) for j in range(ny)],
                      dtype=np.int64)

    # Compute eigenmodes (precompute a large set, use a subset at runtime)
    max_modes = 32
    print(f"Computing {max_modes} eigenmodes for {n} vertices...")
    W_all, eigs_all = compute_eigenmodes(V, F, pinned, num_modes=max_modes)
    print(f"  eigenvalues: {eigs_all[:6].round(4)}")

    # Create simulators
    dt = 0.005
    gravity = np.array([0.0, -9.81, 0.0])

    full_sim = FullSpaceSim(V, F, pinned, dt=dt)

    num_modes = 6
    sub_sim = SubspaceSim(V, F, pinned, W_all[:, :num_modes],
                          eigs_all[:num_modes], dt=dt)

    # Polyscope setup
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    # Register meshes side by side
    offset = np.array([0.0, 0.0, -0.8])
    offset2 = np.array([0.0, 0.0, 0.8])

    mesh_full = ps.register_surface_mesh("full_space", V + offset, F,
                                         edge_width=1.0, smooth_shade=True)
    mesh_full.set_color((0.2, 0.5, 0.9))

    mesh_sub = ps.register_surface_mesh("subspace", V + offset2, F,
                                        edge_width=1.0, smooth_shade=True)
    mesh_sub.set_color((0.9, 0.4, 0.2))

    # Mode visualization
    mesh_modes = ps.register_surface_mesh("modes_viz", V, F,
                                          edge_width=0.5, smooth_shade=True,
                                          enabled=False)

    # State
    state = {
        "num_modes": num_modes,
        "running": True,
        "apply_force": False,
        "force_vertex": n - ny // 2,  # tip center
        "show_residual": True,
        "show_modes": False,
        "mode_to_show": 0,
    }

    def callback():
        nonlocal num_modes

        psim.TextUnformatted("=== Subspace Dynamics Tutorial ===")
        psim.Separator()
        psim.TextUnformatted(
            f"Blue (back): FULL-SPACE sim ({n} DOFs)\n"
            f"Orange (front): SUBSPACE sim ({state['num_modes']} modes = "
            f"{state['num_modes'] * 3} DOFs)"
        )
        psim.Separator()

        # Mode count slider
        changed, new_val = psim.SliderInt("num_modes", state["num_modes"],
                                          v_min=2, v_max=max_modes)
        if changed:
            state["num_modes"] = new_val
            sub_sim.update_modes(W_all[:, :new_val], eigs_all[:new_val])

        # Controls
        _, state["running"] = psim.Checkbox("running", state["running"])
        _, state["apply_force"] = psim.Checkbox("apply_force (tip load)",
                                                state["apply_force"])
        _, state["show_residual"] = psim.Checkbox("show_residual",
                                                  state["show_residual"])

        psim.Separator()
        _, state["show_modes"] = psim.Checkbox("show eigenmodes",
                                               state["show_modes"])
        if state["show_modes"]:
            changed_m, state["mode_to_show"] = psim.SliderInt(
                "mode index", state["mode_to_show"], v_min=0,
                v_max=state["num_modes"] - 1)
            mesh_modes.set_enabled(True)
            mode_vals = W_all[:, state["mode_to_show"]]
            mesh_modes.add_scalar_quantity("mode_weight", mode_vals,
                                          defined_on='vertices', cmap='coolwarm',
                                          enabled=True)
        else:
            mesh_modes.set_enabled(False)

        if psim.Button("reset"):
            full_sim.reset()
            sub_sim.reset()

        # Step simulation
        if state["running"]:
            ext_force = None
            if state["apply_force"]:
                ext_force = np.zeros((n, 3))
                # Point load at the tip
                ext_force[state["force_vertex"]] = [0.0, -50.0, 0.0]

            for _ in range(4):  # substeps per frame
                full_sim.step(gravity, ext_force)
                sub_sim.step(gravity, ext_force)

        # Update visualization
        mesh_full.update_vertex_positions(full_sim.V + offset)
        mesh_sub.update_vertex_positions(sub_sim.V + offset2)

        # Residual: what subspace can't represent
        if state["show_residual"]:
            residual = np.linalg.norm(full_sim.V - sub_sim.V, axis=1)
            mesh_sub.add_scalar_quantity("residual (vs full sim)", residual,
                                        defined_on='vertices', cmap='coolwarm',
                                        enabled=True)
            mesh_full.add_scalar_quantity("residual (vs full sim)", residual,
                                         defined_on='vertices', cmap='coolwarm',
                                         enabled=True)

        # Info text
        psim.Separator()
        max_disp_full = np.max(np.linalg.norm(full_sim.V - V, axis=1))
        max_disp_sub = np.max(np.linalg.norm(sub_sim.V - V, axis=1))
        max_residual = np.max(np.linalg.norm(full_sim.V - sub_sim.V, axis=1))
        psim.TextUnformatted(
            f"Max displacement: full={max_disp_full:.4f}  sub={max_disp_sub:.4f}\n"
            f"Max residual (full - sub): {max_residual:.4f}\n"
            f"Eigenvalues [{state['num_modes']} modes]: "
            f"{eigs_all[:state['num_modes']].round(2)}"
        )

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
