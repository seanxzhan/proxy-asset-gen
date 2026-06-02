# Proxy Asset Generation for Cloth Simulation in Games

> Zhongtian Zheng, Tongtong Wang, Qijia Feng, Zherong Pan, Xifeng Gao, Kui Wu (LightSpeed Studios).
> *ACM Transactions on Graphics 43(4), Article 73, July 2024 (SIGGRAPH 2024).* DOI: [10.1145/3658177](https://doi.org/10.1145/3658177)
> PDF: [`Zheng et al. - 2024 - Proxy Asset Generation for Cloth Simulation in Games.pdf`](./Zheng%20et%20al.%20-%202024%20-%20Proxy%20Asset%20Generation%20for%20Cloth%20Simulation%20in%20Games.pdf)

## TL;DR

Real-time games can't simulate the visual cloth mesh — too many vertices, too few milliseconds. The trick the
industry uses is a low-poly **proxy mesh** that gets simulated, then drives the high-res **visual mesh** via
Linear Blended Skinning (LBS). Hand-authoring the proxy + skinning weights takes a skilled artist days. This
paper automates both steps:

1. **Proxy mesh generation** — extract an isosurface from a UDF around the visual mesh, project it back tight,
   strip one of the two layers via an **ILP** (a graph-cut variant that escapes the regularity condition), then
   Voronoi-cluster down to ~128 vertices.
2. **Skinning weights** — pre-simulate the proxy with [PBD](../../pbd/docs/Position-Based-Dynamics.md), then fit
   visual-mesh LBS weights with **differentiable skinning** under three losses: ARAP (smoothness), kNN-ARAP
   (collision), and an attachment term that softly glues geometrically nearby but topologically disconnected
   parts.

Result: ~10s for the proxy, ~10min for the weights, vs. several days of artist labor — and the output beats a
4-hour artist-painted baseline. Runtime cost: 0.86 ms / PBD iter for a 128-vertex proxy on a Samsung S20 in
Unreal Chaos Cloth.

## The problem

The runtime side of the pipeline is fixed: **simulate proxy → LBS-skin visual → render**. The authoring side
is where everything is hard.

- **Visual cloth meshes are ill-conditioned by nature**: intricate folds, layered structures (a dress with
  inner lining), disconnected components (belts, knots, tassels), and outright non-manifold surfaces. Mesh
  simplification techniques (QEM, instant meshes, edge collapse on animated meshes) assume a clean input
  manifold and explode on this geometry.
- **The output target is harsh**: a *single-layer*, *uniformly meshed*, *simulation-ready* surface with on the
  order of **128 vertices**, because mobile LBS targets ~256 bones and proxy-vertex count is the LBS bone
  count. Voxel-based remeshing (Chen et al. 2023, occluder-gen [Wu 2022]) gets close but doesn't separate
  layers and doesn't produce skinning weights.
- **UDF-based open-surface extractors fail on cloth**: MeshUDF uses a heuristic single-layer pass that breaks
  on complex UDF gradients; LevelSetUDF leaves 15+ components on average; DCUDF's graph-cut layer separation
  depends on a random initial seed and only succeeds ~75% of the time within 20 trials.
- **Skinning-weight prior art needs what we don't have**: ARAPLBS [Thiery & Eisemann 2018] needs a
  well-conditioned mesh to run ARAP on; NeuroSkinning / RigNet need large hand-painted training sets. Neither
  is applicable when the visual mesh itself isn't simulation-ready.

The contribution is a **two-stage pipeline that decouples geometry from weights** and produces both
automatically.

## Method

### Stage 1 — Proxy mesh generation (§3)

Pipeline: `M_visual → M_iso → M_proj → M_single → M_proxy`.

#### 1. Isosurface from UDF (§3.1)

Build the unsigned distance field on a `D/N_v` voxel grid (`D` = bbox max length, `N_v = 32` default). Run
marching cubes at iso-value `D/N_v` (one voxel size — small enough to wrap the original tightly). Output
`M_iso` is **watertight** and double-layered around any thin shell, by construction. Robustness to non-manifold
input is essentially free: the UDF doesn't care.

#### 2. Mesh projection (§3.2)

`M_iso` is topologically clean but loose. Tighten it by minimizing, per vertex `v_i`,

```
Σ_i ||v_proj^i − p_visual^i||² + λ_L Σ_i ||v_proj^i − (1/|N(i)|) Σ_{j∈N(i)} v_proj^j||²
```

where `p_visual^i` is the closest point on `M_visual`. Vector Adam [Ling 2022] solves it. First term snaps to
the input; Laplacian term keeps element uniformity. **Self-intersection is allowed** — runtime cloth in games
disables self-collision for performance anyway.

#### 3. Single-layer extraction via ILP (§3.3) — the technical centerpiece

`M_proj` still has two surfaces wrapping every thin shell. Removing one is a labeling problem `l_i ∈ {0, 1}`
per vertex (keep / remove).

**Guide graph.** For each vertex `v_i`, cast a ray along `−n_i`, find the first triangle hit, and mark its
vertices as candidate **opposite vertices** if they pass two filters:

- distance `< 2D/N_v` (twice the iso-extraction band — anything farther isn't the matching layer);
- normals nearly antiparallel: `n_j · n_i < −1 + ε_o`.

Add an "opposite" edge `(i, j) ∈ E_o` for each pair. Note the procedure can be sloppy — the ILP cleans up
inconsistencies.

**Energy.** Two terms:

```
E_s = Σ_{ij ∈ E_proj} w_s^{ij} · 𝟙[l_i ≠ l_j]    (smoothness on mesh edges)
E_o = Σ_{ij ∈ E_o} 𝟙[l_i = l_j] + λ_bias · 𝟙[d_i ≠ d_j ∧ (l_i − l_j) ≠ sign(d_i − d_j)]
```

- `w_s^{ij} = 1 − (max(|κ_i|, |κ_j|)/κ̄)⁴`. **Low-curvature edges are expensive to cut, high-curvature edges
  are cheap** — layer boundaries naturally sit at curvature ridges.
- `E_o`'s first term wants exactly one of every opposite pair kept. The bias term prefers the **outer** layer
  via `d_i = distance to next +normal hit`: an outer-layer vertex shoots its +normal ray to infinity (`d=∞`),
  so the bias breaks ties toward outer.

**Why ILP, not graph cut?** The Boykov–Kolmogorov polynomial-time algorithm needs **regularity** (submodular
pairwise terms): `E(0,0) + E(1,1) ≤ E(1,0) + E(0,1)`. The opposite-energy term flips this — it *wants*
disagreement — so the standard graph-cut machinery fails. Reformulate per Komodakis–Tziritas (2007): introduce
auxiliary continuous vars `l^{ij}_{00, 01, 10, 11}` for each pair, replace the indicator with

```
l^{ij}_{00} E(0,0) + l^{ij}_{11} E(1,1) + l^{ij}_{10} E(1,0) + l^{ij}_{01} E(0,1)
```

plus consistency constraints `l^{ij}_{00} + l^{ij}_{01} = 1 − l_i`, etc. Mosek solves the resulting ILP in a
few seconds. This is exactly the move that DCUDF [Hou 2023] couldn't make: their layer separation is graph-cut
with smoothness only, plus a random initial label seed — fast but fragile. The ILP gets to **100% success**
where DCUDF gets ~75%.

#### 4. Voronoi simplification (§3.4)

Centroidal Voronoi clustering [Valette & Chassery 2004] reduces `M_single` to exactly `N_p` vertices (default
`N_p = 128`, sized to mobile LBS bone budgets). Centroids → mesh, fix non-manifold artifacts, done. Output mesh
quality is solid: min face angle 47.9°±6.6°, aspect ratio 0.76±0.13 — important because runtime PBD stiffness
is sensitive to triangle quality.

**Optional surface carving** (§5.1): if artists want to preserve separated parts in `M_proxy` so they swing
independently, project `M_visual` onto `M_single` to get a 2D SDF, then carve along an isoline. Off by default.

### Stage 2 — Skinning weight optimization (§4)

Goal: solve for `w_ij` such that the LBS-skinned visual mesh tracks the simulated proxy mesh plausibly.

#### Simplified LBS — translations only

```
v_visual^{i,t} = v_visual^{i,0} + Σ_{j ∈ B(i)} w_ij · (v_proxy^{j,t} − v_proxy^{j,0})
```

with `w_ij ≥ 0`, `Σ_j w_ij = 1` (convex). `B(i)` is fixed at rest pose by kNN. **No per-vertex rotation** —
the runtime cost of integrating rotational DOFs in a real-time PBD step is prohibitive on mobile. This is the
production-pragmatic choice; standard LBS or DQS would slot in if the runtime simulator integrates rotations
too.

#### Pre-simulation

Simulate `M_proxy` with [PBD](../../pbd/docs/Position-Based-Dynamics.md) in PhysX. Two spring types: edge
stretch (per `M_proxy` edge), and bending (across non-shared vertices of each face pair). Pin the top 5% of
vertices, blow with random wind, sample 200 frames at regular intervals. ~0.54s/asset.

#### Three losses

| Loss | Form | Role |
|---|---|---|
| `L_r` (ARAP) | `Σ_t Σ_i min_{R_r} Σ_{j ∈ N(i)} ‖(v_v^{i,t} − v_v^{j,t}) − R_r (v_v^{i,0} − v_v^{j,0})‖²` | Local rigidity over the visual mesh's **topological** 1-ring → preserves shape, enforces near-non-stretchable material |
| `L_c` (collision) | Same form but over **kNN** neighbors `K(i)` | Penalizes geometrically-close vertex pairs that diverge → prevents inter-layer penetration |
| `L_a` (attachment) | `Σ_t ‖V^t − (I − K) V^t‖²` with `K_ij ∝ 1/(‖v^{i,0} − v^{j,0}‖ + ε)` | Softly glues nearby but disconnected components (knots-on-belts, tassels-on-skirts) |

The `L_r` vs. `L_c` split is the load-bearing trick: ARAP on topology preserves *fabric*, ARAP on geometry
prevents *penetration*. Both share the SVD-per-iter rotation estimation; they're cheap to compute together.

`L_a`'s weight kernel is a Laplacian-style operator weighted by *inverse rest-pose distance*: closer pairs
get glued harder. It's distinct from `L_c` because it doesn't penalize relative motion symmetrically — it
specifically pulls disconnected components back into rest-pose proximity.

#### Solver

The convex constraint `w_ij ≥ 0, Σ w_ij = 1` is awkward for AdamW. Reparameterize:

```
w_ij = |s_ij| / Σ_{k ∈ B(i)} |s_ik|
```

Optimize `s_ij` unconstrained. AdamW (lr 1e-3, ≤200 epochs). SVD inside each step recovers the optimal
per-vertex `R_r^{i,t}, R_c^{i,t}` for ARAP closed-form. ~9 min / asset.

## Why decouple the two stages? (§5.2 ablation)

The obvious alternative is **joint optimization**: simplify `M_single` and optimize weights together,
collapsing the proxy edge with the lowest combined visual+proxy quadric energy each step. The authors built
this baseline. It produces non-uniform proxy meshes (tiny triangles where the visual mesh has detail, huge
ones elsewhere), which **breaks PBD** — non-uniform springs give wildly varying stiffness, and the
deformation looks unnatural. Decoupling lets Voronoi enforce uniformity unconditionally; weight optimization
then absorbs the geometric mismatch through the loss landscape.

## Results

100-model dataset of real game garments. 12th-gen i9-12900K + RTX 3090.

**Proxy generation (Table 1, Fig. 17)** — vs. baselines at the same Voronoi target of 128 vertices:

| Method | Success | Components | B-loops | HD | LFD | Time |
|---|---|---|---|---|---|---|
| MeshUDF (32³) | 100% | 2.1 | 3.8 | 0.28 | 4.6e3 | 0.3s |
| LevelSetUDF (32³) | 100% | 15.3 | 15.7 | 0.59 | 1.6e4 | 425s |
| LevelSetUDF (256³) | 100% | 1.8 | 2.3 | 0.63 | 7.1e3 | 491s |
| DCUDF (32³) | 74% | 1.1 | 1.4 | 0.35 | 4.7e3 | 41s |
| DCUDF (256³) | 75% | 2.4 | 3.4 | 0.33 | 4.5e3 | 82s |
| **Ours (32³)** | **100%** | **1.0** | 2.0 | **0.22** | **3.6e3** | 8.4s |

Time breakdown of "Ours": ILP 60%, projection 24%, graph build 12%, isosurface 4%.

**Skinning weights:**
- 0.54s data gen + ~9 min weight optimization per asset.
- Ablation: removing `L_r` → bumpy belt (Fig 12); removing `L_c` → self-collision (Fig 12); removing `L_a` →
  knots detach from darts (Fig 13).
- **User study**: a senior artist (10+ yrs) hand-painted weights for the "Scarf" asset over **4 hours** in
  Unreal — result was clumpy and spiky in transitional regions. The pipeline produces a smoother result in
  **14 minutes**, with smoothly varying weights (Fig. 14). Color-coded weight visualization shows artist's
  hand-paint exhibits abrupt boundaries between bones; differentiable optimization smooths them.
- Generalizes to held-out wind directions (validation video) and to complex motions like waist-joint dance
  sequences (Fig. 1).
- **Runtime**: 0.86 ms per PBD iter for a 128-vertex proxy on Samsung S20 (SD865) under Unreal Chaos Cloth.

## Limitations

- **Voronoi can lose sharp boundaries at extremely small `N_p`**. Future work: hybrid with edge-collapse / flip
  passes for boundary-aware simplification.
- **No CCD during projection** — `M_proj` may self-intersect. Acceptable here because runtime games disable
  self-collision anyway; would matter for higher-fidelity targets.
- **Non-stretchable cloth assumption** is baked into ARAP. Stretchable inputs would need the generalized ARAP
  in [Thiery & Eisemann 2018].
- **Geometry and weights are optimized separately, not jointly.** The decoupling beats their joint baseline
  (§5.2), but a smarter joint formulation might do better — explicitly listed as future work.
- **Voronoi simplification pre-determines `N_p`.** Generating multi-resolution LOD stacks for progressive
  simulation [Zhang 2022] would require running the pipeline multiple times.

## Relationship to neighboring work

- **Builds directly on [PBD (Müller et al. 2007)](../../pbd/docs/Position-Based-Dynamics.md)** for both
  pre-simulation (in PhysX) and target runtime (Unreal Chaos Cloth). The "extremely simplified proxy at 128
  verts" target only works because PBD's stability is iteration-bounded, not stiffness-bounded.
- **ARAPLBS [Thiery & Eisemann 2018]** is the closest skinning-weight ancestor. They pioneered ARAP-based
  weight optimization, but assumed clean meshes. This paper's `L_c` (kNN-ARAP for collision) and `L_a`
  (attachment) extend ARAPLBS to ill-conditioned input.
- **DCUDF [Hou et al. 2023]** is the direct comparison for single-layer extraction — same UDF + double-cover
  + label-cut pipeline shape, but their graph-cut + random-init formulation can't express the opposite-pair
  energy and fails ~25% of the time. The ILP reformulation (via Komodakis–Tziritas 2007) is the exact fix.
- **Robust low-poly meshing [Chen et al. 2023]** uses a similar UDF + voxel + ILP recipe for general 3D
  models; this paper specializes it to cloth (single-layer extraction is the cloth-specific addition) and
  pairs it with a skinning-weight optimizer.
- Could compose with **progressive cloth simulation [Zhang et al. 2022]** to generate LOD stacks of proxies,
  if the pipeline is run at multiple `N_p` targets — listed as future work.
