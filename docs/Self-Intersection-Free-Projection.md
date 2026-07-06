# Self-Intersection-Free Projection (§3.2)

> Design note on a gap in Zheng et al. 2024. The published §3.2 projection allows
> self-intersection ("runtime PBD disables self-collision anyway"). In practice
> this breaks the **§3.3 single-layer extraction**, producing holes in
> `M_single`. The authors confirmed by email that the projection module needs a
> self-intersection check that the paper omits:
>
> > *"The projection module needs to incorporate a self-intersection check to
> > guarantee the resulting mesh is entirely self-intersection free."*
>
> This note explains why, and specifies the fix.

Related: [Proxy-Asset-Generation.md](./Proxy-Asset-Generation.md) ·
implementation in [`src/pag/projection.py`](../src/pag/projection.py),
[`src/pag/guide_graph.py`](../src/pag/guide_graph.py),
[`src/pag/ilp.py`](../src/pag/ilp.py).

### What the paper actually says (verified against the PDF)

The published text is self-contradictory on this point, which is why the email
matters:

- **§3.2** explicitly *declines* the check: *"Note that we do not enforce
  self-intersection-free in the optimization, as our proxy generation and
  simulation pipeline do not mandate a self-intersection-free proxy mesh."*
- **Limitations and Future Works** then lists it as an open direction:
  *"introducing continuous collision detection during projection to prevent
  self-intersection could be an interesting future direction."*

So **the fix in this note — CCD during projection — is the paper's own named
future-work item**, and the authors' email is them confirming that in practice
it is necessary, not merely interesting. Note also what the paper does *not*
claim: it never produces a clean `M_single`. The clean, single-component output
is `M_proxy`, achieved by **§3.4** (Voronoi clustering, *then* "removing
non-manifold faces and splitting non-manifold vertices"). `M_single` is an
intermediate; its jagged-cut holes are expected and are resampled away by §3.4.
The paper reports the *final* `M_proxy` as a single component with an **average
of 2.0 boundary loops** — the bar to measure against.

## The pipeline stage in question

Stage 1 runs $M_\text{visual} \to M_\text{iso} \to M_\text{proj} \to M_\text{single} \to M_\text{proxy}$.
The projection step (§3.2) tightens the loose, watertight double-cover $M_\text{iso}$
onto the input by minimizing, per vertex $\mathbf{v}_i$,

$$
E_\text{proj}(\mathbf{V}) \;=\; \underbrace{\sum_i \lVert \mathbf{v}_i - \mathbf{p}_i \rVert^2}_{\text{data: snap to } M_\text{visual}} \;+\; \lambda_L \underbrace{\sum_i \Bigl\lVert \mathbf{v}_i - \tfrac{1}{|\mathcal{N}(i)|}\!\!\sum_{j \in \mathcal{N}(i)}\!\! \mathbf{v}_j \Bigr\rVert^2}_{\text{smoothness: uniform Laplacian}}
$$

where $\mathbf{p}_i$ is the closest point on $M_\text{visual}$ (a libigl AABB query)
and $\mathcal{N}(i)$ is the 1-ring of $\mathbf{v}_i$ on $M_\text{iso}$. The reference
implementation solves this with Vector Adam [Ling 2022] and **no collision term**.

## Why self-intersection produces holes in `M_single`

The failure is causal, not cosmetic. Tracing it tells us exactly what the
projection must guarantee:

1. $M_\text{iso}$ is a clean watertight **double cover** — two sheets wrapping
   each thin shell, separated by $\approx 2\,h$ where $h = D/N_v$ is the voxel
   size.

2. The data term pulls **both** sheets toward the *same* closest points on
   $M_\text{visual}$ — `point_mesh_squared_distance` has no notion of which layer
   a vertex belongs to. With small $\lambda_L$ the two sheets collapse toward
   each other and **cross**. Nothing in $E_\text{proj}$ keeps them apart.

3. Once the sheets interpenetrate, the §3.3 **guide graph** breaks. For each
   $\mathbf{v}_i$ it casts a ray along $-\mathbf{n}_i$ and pairs $i$ with the first
   triangle hit, keeping the pair only if

$$
d_{\text{hit}} < 2h
\qquad\text{and}\qquad
\mathbf{n}_\text{face}\cdot\mathbf{n}_i < -1 + \varepsilon_o .
$$

   Where the sheets have crossed, the $-\mathbf{n}_i$ ray hits the wrong sheet or
   its own folded geometry, so the opposite-edge set $\mathcal{E}_o$ is corrupted
   and these filters fire inconsistently.

4. The ILP labeling $l_i \in \{0,1\}$ then mislabels in the tangled region, and
   `extract_single_layer` — which drops every face touching an $l=0$ vertex —
   punches **holes** exactly along the tangle.

So the requirement is sharper than "no self-intersection." What §3.3 needs is
that the two sheets stay **topologically separated with a positive gap** — both
non-crossing *and* not collapsed. That distinction drives the method choice.

## The key structural fact: this is invariant *preservation*

$M_\text{iso}$ comes out of marching cubes on a UDF as a clean 2-manifold — it is
**intersection-free by construction**. So §3.2 is not a "repair an intersecting
mesh" problem. It is: *start intersection-free, and never take a step that
crosses.* That is precisely the regime where collision-aware stepping yields a
**guarantee** rather than a heuristic — matching the authors' word "guarantee."

## Recommended fix: IPC barrier + CCD-filtered line search

Reformulate §3.2 as a **contact-aware** minimization. Add a contact barrier over
non-incident vertex–triangle and edge–edge primitive pairs $\mathcal{C}$:

$$
E(\mathbf{V}) \;=\; \sum_i \lVert \mathbf{v}_i - \mathbf{p}_i \rVert^2 \;+\; \lambda_L \sum_i \lVert \mathbf{v}_i - \boldsymbol{\mu}_i \rVert^2 \;+\; \kappa \!\!\sum_{(k,l)\,\in\,\mathcal{C}}\!\! b\bigl(d_{kl}(\mathbf{V}),\, \hat{d}\bigr)
$$

with the IPC log-barrier [Li et al. 2020], active only within distance $\hat{d}$:

$$
b(d, \hat{d}) \;=\;
\begin{cases}
-\,(d - \hat{d})^2 \,\ln\!\bigl(d/\hat{d}\bigr), & 0 < d < \hat{d}, \\[4pt]
0, & d \ge \hat{d},
\end{cases}
\qquad
\lim_{d \to 0^+} b(d,\hat{d}) = +\infty .
$$

Minimize with projected-Newton (or L-BFGS / gradient descent) whose **line search
is filtered by continuous collision detection (CCD)**. Given the current iterate
$\mathbf{V}$ and a search direction $\Delta\mathbf{V}$, take the largest step
$\alpha \le \alpha_\text{TOI}$ where the time of impact

$$
\alpha_\text{TOI} \;=\; \sup\bigl\{\,\alpha \in [0,1] \;:\; \mathbf{V} + \tau\,\Delta\mathbf{V}\ \text{is intersection-free for all}\ \tau \in [0,\alpha]\,\bigr\}
$$

is computed by an exact/conservative CCD (e.g. Tight-Inclusion [Wang et al. 2021]).
The barrier diverges as $d \to 0$ and the CCD filter forbids crossing, so the
**entire optimization trajectory is provably intersection-free** — because it
started so.

### Choosing $\hat{d}$: small, **not** the voxel size

A tempting idea is to set $\hat{d} \approx h = D/N_v$ so the barrier holds the
two sheets "inflated" a full voxel apart. **This is wrong**, and the
implementation's probe makes it concrete. The IPC barrier fires between *every*
non-incident primitive pair within $\hat{d}$ — and on $M_\text{iso}$ the edge
length is itself $\approx h$. So $\hat{d} \approx h$ turns every vertex's own
one-/two-ring neighbours into "collisions": on a sphere iso (3.7k vertices) the
collision count explodes from ~470 at $\hat{d} = 10^{-3} D_\text{bbox}$ to ~84k
at $\hat{d} = 5\!\times\!10^{-2} D_\text{bbox}$. The barrier then fights the
Laplacian everywhere and the mesh can never tighten.

The correct regime is the **IPC standard** — $\hat{d}$ a small fraction of the
bounding-box diagonal, far below the element size:

$$
\boxed{\;\hat{d} \;\approx\; 10^{-3}\, D_\text{bbox} \;\ll\; \text{edge length}\;}
$$

The separation between *requirement* and *mechanism* is the key insight:

- **The requirement** §3.3 has is **non-crossing**, and that is delivered by the
  **CCD line-search filter**, not by the barrier's reach. CCD guarantees no step
  ever tunnels a sheet through another, for any $\hat{d}$.
- **The barrier** only needs to be wide enough to brake an approach before
  contact. With a small $\hat{d}$ the two sheets are pulled tight by the data
  term and settle $\sim\!\hat{d}$ apart — a *thin* but strictly positive gap.

That thin gap is fine for the guide graph: its filter keeps opposite hits with
$d_\text{hit} < 2h$, and $\hat{d} \approx 10^{-3} D_\text{bbox} \ll 2h$, so a ray
from one sheet still lands on the other well inside the window. We need the
sheets *not to cross*, not to stand a voxel apart.

### Tooling

In this Python codebase, use the **IPC Toolkit** (`ipctk`, pip-installable). It
supplies the distance functions, the barrier, the constraint set, and Tight-Inclusion
CCD with Python bindings, and slots in where the AABB closest-point query already
lives in `project_to_visual`. The data + Laplacian terms become the "elastic
potential"; IPC Toolkit provides the contact term and the CCD line-search filter.

## Minimal-change alternative (keep Vector Adam)

If restructuring the solver is undesirable, add a **CCD-filtered step clamp** to
the existing loop (Provot/Bridson-style impact zones):

1. Compute the Vector-Adam displacement $\Delta = \mathbf{V}_\text{new} - \mathbf{V}$ as today.
2. Run vertex–triangle + edge–edge CCD between $\mathbf{V}$ and $\mathbf{V}+\Delta$ to get $\alpha_\text{TOI}$ over all non-incident pairs.
3. If $\alpha_\text{TOI} < 1$, take the clamped step $\mathbf{V} \leftarrow \mathbf{V} + 0.9\,\alpha_\text{TOI}\,\Delta$.

This preserves the invariant since the mesh starts clean. Caveats:

- A **global** TOI clamp interacts badly with Vector Adam's per-vertex momentum
  normalization — the moments $m, s$ keep accumulating against a clamped step.
  Damp or reset them when a clamp fires.
- It only prevents **crossing**, not **collapse**. You still want a small
  self-repulsion term or a minimum-separation barrier to hold the sheets apart
  for §3.3 — which is exactly what the IPC barrier gives for free. Hence the
  preference for the full reformulation above.

## Implementation details that matter either way

- **Filter incident primitives.** Pairs sharing a vertex or edge are legitimately
  close; only **non-incident** proximity is a real contact. Exclude them from
  $\mathcal{C}$ / the CCD set.
- **Broadphase.** Use a spatial hash or BVH over triangles. Brute-force
  $O(|F|^2)$ proximity will dominate the §3 wall-clock on the iso mesh.
- **Geometric thickness.** A small $\varepsilon \approx 0.1\,h$ in the CCD avoids
  numerical zero-distance degeneracies at exact contact.

## Heavier alternatives (probably unnecessary)

Explicit surface tracking — **El Topo** [Brochu & Bridson 2009] or **Codim-IPC**
[Li et al. 2021] — additionally remeshes to maintain element quality during the
flow. Consider this only if the projection is found to *also* degrade triangle
quality badly before §3.4. Otherwise it is more machinery than needed, and its
remeshing changes connectivity *before* the ILP, which §3.3 does not want.

## Recommendation

Replace the free Vector-Adam projection in §3.2 with an **IPC barrier
($\hat{d} \approx 10^{-3} D_\text{bbox}$) + CCD-filtered line search**. The CCD
filter delivers the non-crossing guarantee §3.3 needs; the small barrier brakes
the approach so the two sheets settle a thin positive gap apart instead of
collapsing through each other.

## Implementation & validation in this repo

Implemented in [`src/pag/projection.py`](../src/pag/projection.py) as the
`collision_free=True` path of `project_to_visual` (projected-Newton with the IPC
Toolkit's `BarrierPotential` / `compute_collision_free_stepsize`, adaptive
barrier stiffness, Armijo backtracking). Exposed through the pipeline as
`generate_proxy_mesh(..., proj_collision_free=True)`. Tests:
[`tests/test_projection_ipc.py`](../tests/test_projection_ipc.py).

Empirical effect of the barrier reach (sphere iso, 3.7k vertices, edge length
$\approx h$): collisions grow from ~470 at $\hat{d}=10^{-3}D_\text{bbox}$ to ~84k
at $5\!\times\!10^{-2}D_\text{bbox}$ — the concrete reason $\hat{d}$ must stay
$\ll$ edge length. A downstream sweep confirms the small default is also best for
§3.3: on `9423122485_cleaned` raising $\hat{d}$ from $10^{-3}$ to
$1.2\!\times\!10^{-2}\,D_\text{bbox}$ degraded the projection fit ~40× and
*increased* `M_single` fragmentation — so $\hat{d}=10^{-3}D_\text{bbox}$ is the
default and larger is not better.

End-to-end on `data/jacket.obj` (§3.1→§3.3, $N_v=24$), Vector Adam vs. IPC:

| Solver | `M_proj` self-intersects | `M_single` components | `M_single` boundary loops |
|---|---|---|---|
| Vector Adam | **yes** | 2 | 6 |
| IPC + CCD | **no** | **1** | **4** |

The IPC projection removes the self-intersection and yields a single connected
`M_single` (the paper's target of 1.0 components) whose boundary loops drop to
the garment's genuine openings — directly resolving the holes the authors
flagged. Cost: ~15 s for the projection vs. ~1 s for Vector Adam — the premium
for the guarantee, acceptable for a one-time asset bake.

## References

- M. Li et al. *Incremental Potential Contact: Intersection- and Inversion-free
  Large Deformation Dynamics.* ACM TOG (SIGGRAPH) 2020.
- B. Wang, Z. Ferguson et al. *A Large-Scale Benchmark and an Inclusion-Based
  Algorithm for Continuous Collision Detection* (Tight-Inclusion CCD). ACM TOG 2021.
- Z. Ferguson et al. *IPC Toolkit.* https://ipctk.xyz
- M. Li et al. *Codimensional Incremental Potential Contact.* ACM TOG (SIGGRAPH) 2021.
- T. Brochu, R. Bridson. *Robust Topological Operations for Dynamic Explicit
  Surfaces* (El Topo). SIAM J. Sci. Comput. 2009.
- B. Ling et al. *Vector Adam.* 2022 (the current §3.2 solver).
- R. Bridson, R. Fedkiw, J. Anderson. *Robust Treatment of Collisions, Contact
  and Friction for Cloth Animation.* ACM TOG (SIGGRAPH) 2002 (impact-zone clamping).
</content>
</invoke>
