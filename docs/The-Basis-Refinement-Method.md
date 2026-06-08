# The Basis Refinement Method

> Eitan Grinspun (Caltech). PhD thesis, defended 16 May 2003. Advisor: Peter Schröder. Committee: Peter
> Schröder, Mathieu Desbrun, Al Barr, Petr Krysl, Jerry Marsden.
> PDF: [`Grinspun - The Basis Refinement Method.pdf`](./Grinspun%20-%20The%20Basis%20Refinement%20Method.pdf)

## TL;DR

Adaptive finite-element solvers buy accuracy at fixed cost by refining the discretization where the error
indicator says it matters. The textbook approach — *split the mesh element where it's too coarse* — turns out
to be the wrong abstraction: it creates T-vertices that violate compatibility, requires bespoke fix-up rules
per element type and dimension, and **doesn't even apply** to discretizations whose basis functions span more
than one ring of elements (B-splines of degree ≥ 2, Loop subdivision, etc.).

The thesis flips the abstraction: **refine basis functions, not mesh elements**. The basic move is a sequence
of nested approximation spaces `V(0) ⊂ V(1) ⊂ V(2) ⊂ ...` whose bases are *refinable* — every level-`p`
scaling function is a fixed linear combination of its level-`(p+1)` children:

```
φ_i^(p)(x) = Σ_k a_ik^(p+1) φ_k^(p+1)(x)
```

Refinement enriches the active basis `B` either by **adding details** (a function from `D(p) = V(p+1) \ V(p)`)
or by **substitution** (deactivate one parent, activate its children). Both operations live in spaces, not in
geometry — and incompatibility never arises because the basis is always globally defined.

The contribution is a sequence of pragmatic claims:

1. **A minimal mathematical framework** — nested spaces + Riesz bases + compact, diminishing support — that
   makes no assumption about domain dimension, element type (triangle/quad/tet/hex), basis order, or
   smoothness. The only requirement is that basis functions be refinable.
2. **Two short, provably-correct algorithms** — `Activate(φ)` and `Deactivate(φ)` — that maintain three data
   structures (active basis `B`, active elements `E`, active tiles `T`) plus per-element integration tables
   (`Bs`, `Ba`) under arbitrary (un)refinement. Substitution refinement is just one `Deactivate` composed with
   `N` `Activate` calls. Krysl implemented and debugged the core algorithms in a day; extending to 2D and 3D
   was three hours each.
3. **A unified mapping** — wavelets, multiwavelets, finite elements, B-splines, and subdivision schemes
   (Loop, Doo-Sabin, Catmull-Clark, √3, Butterfly) all fit the framework. The same algorithms drive all of
   them.
4. **Working applications** across graphics, biomechanics, and medicine: thin-shell simulation (Loop
   subdivision, fourth-order PDE), brain-volume deformation under surgery (linear tetrahedra), human-mandible
   stress analysis (trilinear hexahedra), and ECG potential fields (linear tetrahedra on the torso).

The umbrella acronym is **CHARMS**: Conforming Hierarchical Adaptive Refinement MethodS.

## The problem

Adaptive solvers want to allocate degrees of freedom where the error indicator says they matter and remove
them where it doesn't. Two perspectives on how to do this:

- **Element point of view** (classical FE). The approximation is described element-by-element. To refine,
  bisect/quadrisect/octasect the element. Local, per-element, no theory needed beyond the element itself.
- **Basis function point of view**. The approximation is `Σ u_i φ_i(x)`. To refine, enlarge the spanning set.

In one dimension with linear basis functions the two are equivalent. As soon as you leave that toy setting,
element refinement starts paying costs the basis view doesn't:

- **T-vertices break compatibility.** Quadrisecting one triangle in a 2D mesh leaves T-vertices on shared
  edges with neighbors that *weren't* split. The function becomes discontinuous across those edges. Fixes
  exist — red/green retriangulation, longest-edge bisection, Lagrange multipliers, penalty methods — but each
  is element-type-specific, each adds bookkeeping, and each is increasingly cumbersome in 3D and beyond.
  The literature documents specialized algorithms for triangles, tetrahedra, and hexahedra separately.
- **Higher-order bases don't admit element-isolated refinement.** A quadratic B-spline is supported on three
  intervals. Bisect just one and the function is no longer in the span — the new space doesn't contain the
  old one. Subdivision basis functions (Loop's two-ring support, Catmull-Clark's larger support) have the
  same problem amplified. For these schemes, refining one element in isolation is *impossible* in the strict
  sense of nested spaces.
- **Hermite cubics dodge the issue but cost DOFs.** They're locally defined and refinable per-element, but
  they introduce non-displacement degrees of freedom (tangents) that engineers and artists don't think in,
  and they generalize awkwardly to bivariate arbitrary-topology surfaces.

The thesis's bet: meshless methods avoid the compatibility problem because they have no mesh, and their
natural refinement strategy *is* basis enrichment. Bring that idea back to mesh-based discretizations — keep
the mesh (it makes basis functions and quadrature simple) but stop treating it as the thing being refined.

## Method

### Stage 1 — Nested spaces and the refinement relation (§2.1–2.2)

Start with an infinite sequence of nested spaces, `V(0) ⊂ V(1) ⊂ V(2) ⊂ ...`, dense in the solution space.
Each `V(p)` has a Riesz basis of **scaling functions** `{φ_i^(p)}`. The detail space `D(p)` complements `V(p)`
in `V(p+1)`, with its own Riesz basis of **detail functions** `{ψ_i^(p)}` (orthogonality to `V(p)` is nice
but not required).

Because `V(p) ⊂ V(p+1)`, every coarse scaling function is a finite linear combination of fine ones — the
**refinement relation**. Compactly-supported scaling functions plus *diminishing* support (every child's
support is a fixed factor `K < 1` smaller than the parent's) produce a multiresolution analysis: refinement
becomes a parametrically-local operation. All the algorithms assume compact diminishing support; the theory
goes through without it but loses efficiency.

The active set `B ⊂ ⋃_p V(p) ∪ D(p)` is the spanning set of the current trial space `S`. Two atomic
refinement primitives:

- **Detail refinement**: `B := B ∪ {ψ}` for some inactive detail. Trivially preserves linear independence
  (details from different levels live in disjoint spaces).
- **Substitution refinement**: `B := B \ {φ_j^(p)} ∪ C(φ_j^(p))` — deactivate a scaling function, activate
  its children. Does *not* in general preserve linear independence (parent and children live in nested
  spaces). Kraft's construction (§2.4.2) restores it: after every substitution, recursively substitute any
  function whose support is fully covered by finer active elements. Geometrically: stack subdomains by
  level and let scaling functions fall through holes under gravity.

### Stage 2 — Elements, tiles, and the tile coloring problem (§2.3)

Numerical quadrature needs a partition of the domain over which every active basis function has a "simple
form" (typically piecewise polynomial). Two layers of structure:

- **Element tiles** `E(p)` — a partition of the domain at level `p`, chosen so that every level-`p` scaling
  function is simple over every level-`p` element. The natural support set `S(φ_i^(p))` is the minimal set
  of level-`p` elements containing the function's parametric support.
- **Resolving tiles** — fillers between levels. Because the active set draws from multiple levels, the
  required partition needs sub-pieces that resolve overlaps between coarse and fine elements. For schemes
  with strictly nested element partitions (Loop, FE quadrisection), level-`p` resolving tiles are just
  level-`(p+1)` elements; for schemes like √3 subdivision they're genuinely separate.

The integration partition is computed by solving the **tile coloring problem (TCP)** (§4.2.6). Every tile in
the (infinite) hierarchy gets one of three colors:

- **Black** — too coarse: has an active descendant, or is the resolving tile beneath a black element.
- **Red** — fits exactly: an active leaf element (active with no active descendants), or a resolving tile
  between a black coarser-link and a white finer-link.
- **White** — too fine: descended from red, or resolved away by finer red tiles.

**Theorem 4** proves the red tiles form a *minimal* partition that resolves every active element. The
incremental coloring algorithm (`UpdateTilesOnElementActivation`, `UpdateTilesOnElementDeactivation`)
maintains it under any sequence of (de)activations — colors only ever move "up" on activation (white → red →
black) and "down" on deactivation, so updates are local to changed elements and their ancestors.

### Stage 3 — The activation/deactivation algorithms (§4.3–4.4)

Three system invariants define a **consistent snapshot**:

- **I1 (active elements)**: `E = ⋃_{φ ∈ B} S(φ)` — every active function's support is covered by active
  elements; every active element supports an active function.
- **I2 (native table)**: `Bs(ε) = B ∩ S*(ε)` — element `ε`'s native table lists exactly the same-level
  active functions whose support overlaps it.
- **I3 (ancestral table)**: `Ba(ε) = ⋃_{ε' ∈ D*(ε)} Bs(ε')` if `Bs(ε) ≠ ∅`, else `∅` — element `ε`'s
  ancestral table lists all coarser-level active functions overlapping it (cleared when inactive).

`Activate(φ)` and `Deactivate(φ)` are the smallest possible algorithms that preserve all three invariants
under the corresponding change to `B`. The thesis gives a **hierarchical Lamport-style proof** that they do
(Chapter 4 Appendix), translating each proof step directly into a code line. The implementations are five
or six lines of pseudocode each (§4.4.2–4.4.3).

Once you have these two, every higher-level operation is composition:

- `Refine(φ)` (substitution) — `Activate` each child, then `Deactivate` the parent (with coefficient
  redistribution: `u_j += a_ij u_i`).
- `ComputeStiffness(E)` — iterate over active elements, for each consider every same-level pair from
  `Bs(ε) × Bs(ε)` and every cross-level pair from `Bs(ε) × Ba(ε)`. Each pair-element triple is visited
  exactly once at the finest level that captures the interaction (§4.4.5).

The portable insight: traditional FE codes already iterate elements and assemble per-element contributions.
Adding basis refinement is replacing the per-element loop body with one that consults `Bs` and `Ba`.

### Stage 4 — Mapping concrete discretizations onto the framework (Chapter 3)

The framework is empty without instances. Chapter 3 maps:

- **Wavelets** (§3.2) — scale-invariant, shift-invariant nested spaces with a canonical scaling function and
  mother wavelet. Haar system as the canonical small example.
- **Multiwavelets** (§3.3) — `L > 1` canonical scaling functions per level; matrix dilation/wavelet
  equations. "Haar's Hats" example with mean+slope scaling functions and symmetric/antisymmetric mother
  wavelets, recovering smoothness Daubechies proved a single mother wavelet can't have.
- **Finite elements** (§3.4) — piecewise polynomials with element-boundary compatibility. Reframed via
  subdivision: dyadically split every element globally, treat the resulting children as the level-`(p+1)`
  basis. Element refinement *is* basis refinement when you stop thinking element-locally.
- **Subdivision** (§3.5) — the most general case. A subdivision scheme is a topological refinement
  operator (e.g., quadrisection) plus a coefficient refinement operator (a stencil). Loop's scheme worked
  through in detail (§3.5.2) — basis functions are 2-ring-supported, regular elements use box-spline
  quadrature, all elements are treated as regular for integration without breaking convergence in practice.
  Doo-Sabin handled as a dual scheme (§3.5.3).

Each instance specifies: scaling functions, detail functions (or substitution-only), elements, resolving
tiles, natural support sets, descendants/ancestors. Once those are defined, Chapter 4's algorithms run
unchanged.

### Stage 5 — Maintaining linear independence (§2.4, §4.2.7)

Detail-only refinement keeps the active set linearly independent automatically (different levels of the
multiresolution analysis are disjoint). Substitution refinement does not — parent and children both live in
`V(p+1)`. **Kraft's construction** (§2.4.2) gives a clean fix: after each substitution, automatically
substitute every function whose support is covered by finer active elements; iterate to fixpoint. Equivalent
to "let coarse functions fall through holes punched by fine subdomains until they get stuck."

For some applications (the thesis's explicit thin-shell time-stepping among them) the linear-dependence
issue can simply be ignored — the solution process stays well-behaved.

## Applications (Chapter 5)

| Application | PDE order | Discretization | Refinement | Result |
|---|---|---|---|---|
| Inflating metal-foil balloon | 4th-order elliptic | Loop subdivision | substitution | 50 → 1000 active fns over 6 levels; fine functions concentrate at wrinkles |
| Crushing aluminum cylinder | 4th-order | Loop subdivision | substitution | Captures buckling mode + sharp folds |
| Pillow inflation | 4th-order | Loop subdivision | substitution | Fine wrinkle structure with thicker (cloth) material |
| Brain volume deformation post-resection | 2nd-order linear elasticity | linear tetrahedra | substitution | 5526 → 64,905 DOFs adapted vs. ~300k DOFs uniformly; 38s on 600MHz P III |
| Human mandible bite stress | 2nd-order linear elasticity | trilinear hexahedra (octasection) | details + substitution | 1700 → 4200 DOFs; stress concentration captured at bite point and thin extremities |
| ECG torso potential field | generalized Laplace | linear tetrahedra | substitution | 900 → 8500 nodes over 4 levels; isopotential surfaces of epicardial dipole |

The thin-shell results are the strongest argument: classical FE mesh refinement *cannot apply* to
subdivision-element discretizations because Loop basis functions span the 2-ring of a vertex. Element-isolated
refinement breaks the nested-spaces property. Basis refinement runs unchanged.

## Limitations

- **Per-function error indicators are weakly developed.** The implementation lumps standard per-element
  error estimators onto basis functions via `∫ φ_i(x) γ(x) dx` where `γ(x)` is piecewise-constant element
  error density. Effective in practice but the thesis flags a per-function indicator as desirable future
  work.
- **Unrefinement is inherently lossy.** Deactivating a function projects onto a smaller space. The
  implementation uses *interpolated unrefinement* — fade `u_i → 0` linearly over a deactivation delay — to
  avoid temporal `C^0` discontinuities. Velocity discontinuities remain noticeable; future work suggests
  decay curves with zero initial/final time derivatives.
- **No asynchronous time integration.** Spatially-adaptive discretizations have spatially-varying CFL limits
  but the thesis uses synchronous explicit time-stepping, dragged by the finest function. Asynchronous
  variational integrators (AVIs) are flagged as theoretically orthogonal and worth combining.
- **No explicit GPU/parallel implementation.** All examples run serial CPU. The 38s brain-deformation solve
  on a 600MHz Pentium III is "fast enough for surgical interventions on a two- or four-CPU PC" — i.e.,
  marginal, and presented before commodity GPU compute existed.
- **Compact support assumed for efficiency.** Theory generalizes to weakly-localized basis functions but
  the algorithms' performance arguments depend on compact support.
- **Basis-property maintenance for substitution requires care.** Without Kraft's construction (or moral
  equivalent) substitution refinement produces linearly-dependent active sets — fine for explicit dynamics,
  fatal for implicit solves with stiffness matrices that need to be inverted.
- **Tagged subdivision is future work.** Sharp creases, semi-sharp creases, and other tag-modulated
  discretizations break the simple nested-space structure into a 2D space-of-spaces (level × sharpness);
  the thesis sketches multi-nesting refinement as worthwhile but doesn't implement it.

## Relationship to neighboring work

- **Generalizes hierarchical splines** [Forsey & Bartels 1988] and **wavelet adaptive solvers**
  [Gortler & Cohen 1995, Gortler et al. 1993]. Both are recast as instances of basis refinement; the
  earlier "buffer regions of frozen control points" hack disappears once you stop thinking in terms of
  control points and start thinking in terms of which spanning functions are active.
- **Replaces classical mesh refinement** [Bey 1995/2000, Rivara & Inostroza 1997, Arnold et al. 2001] for
  every discretization where it applies, and *enables* refinement for discretizations where mesh
  refinement breaks (B-splines of degree ≥ 2, all subdivision schemes with support beyond one-ring).
- **Built on subdivision theory** [Loop 1987, Catmull & Clark 1978, Doo & Sabin 1978, Kobbelt 2000a,
  Zorin & Schröder 2000]. Subdivision provides the refinement relation; CHARMS turns refinement relations
  into adaptive solver algorithms.
- **Specializes Harten's discrete framework** [Harten 1993, 1996] (§2.5). Harten's index-set formulation
  with restriction/prolongation operators is more general; CHARMS adds the locality assumptions needed for
  efficient implementation.
- **Capell et al. 2002** built an interactive skeleton-driven elasticity tool on basis refinement, citing
  this work directly. The connection to skinning/proxy-driven simulation pipelines is direct: a coarse
  proxy mesh + adaptive enrichment is structurally the same idea as a coarse simulation space adaptively
  refined by the error indicator. The thesis's framework is the principled foundation for that intuition.
- **Future link to model reduction and control** (§6.2) — sketched but not implemented. Reduce a complex
  system to a Galerkin-projected coarse model for controller design, then re-refine the basis when the
  controller's performance evaluator says coarseness is hurting. Same `Activate`/`Deactivate` machinery,
  different error indicator.
- **Could compose with proxy-driven cloth pipelines** — the [proxy-asset-generation
  pipeline](./Proxy-Asset-Generation.md) produces a fixed ~128-vertex proxy that drives the visual mesh via
  LBS. CHARMS suggests a different axis: rather than committing to a fixed proxy resolution, start coarse
  and let an error indicator activate finer basis functions only where folds, contacts, or wrinkles
  demand them. Different runtime tradeoff (LBS skinning is cheaper than recomputing active sets) but the
  same core question — *where do we spend simulation DOFs?* — answered structurally rather than by a
  fixed authoring budget.
