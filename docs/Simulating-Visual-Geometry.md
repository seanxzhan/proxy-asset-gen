# Simulating Visual Geometry

> Matthias Müller, Nuttapong Chentanez, Miles Macklin (NVIDIA Research).
> *Proceedings of Motion in Games (MiG '16), Burlingame, CA, October 2016.* DOI: [10.1145/2994258.2994260](https://doi.org/10.1145/2994258.2994260)
> PDF: [`Müller et al. - 2016 - Simulating visual geometry.pdf`](./Müller%20et%20al.%20-%202016%20-%20Simulating%20visual%20geometry.pdf)

## TL;DR

Games and films keep three meshes per object — a visual mesh, a simulation mesh, and a convex-decomposition
collision proxy — because no single representation has been cheap *and* faithful enough at runtime. The cost is
authoring effort and visible mismatch (objects penetrate where they shouldn't, deform where they shouldn't).
This paper goes the other way: simulate the visual mesh **directly**, on a representation derived from it
automatically, with a visualization mesh that's identical-up-to-vertex-averaging.

The trick is a sequence of pragmatic choices:

1. **Primitive construction** — extrude each visual face along `−n` by a per-material thickness to form a
   convex polyhedron; optionally fuse enclosed convex regions into single primitives. Connect primitives that
   touch or overlap in rest pose. Output is a general graph, not a manifold mesh, and tolerates non-manifold,
   non-conforming, self-intersecting input.
2. **Simulation** — drive each primitive as an [oriented particle](../../pbd/docs/Position-Based-Dynamics.md)
   ([MC11]), but use the **affine** part of the shape-matching transform (not just rotation) to deform each
   primitive linearly. Linearity preserves convexity, so collision and fracture machinery still work.
3. **Visualization** — vertices in different primitives that share a `global id` get their positions averaged
   each frame, closing gaps without changing the topology. The visual mesh is the simulation mesh, modulo a
   per-frame averaging pass.
4. **Plasticity, fracture, tearing** — extend [MCK13]'s convex-decomposition fracture to the visual-mesh
   setting; treat rigid objects as rigid between impacts and only call the local quasi-static solver on hit
   (the 16k-primitive car runs at 60+ fps because of this).

Result: WYSIWYS — what you see is what you simulate. A car authored in Blender in <1 hr deforms, shatters,
and tears across all of its visible details, with no separate simulation mesh.

## The problem

The standard runtime pipeline carries three representations:

- **Visual mesh** — tuned for rendering, often non-manifold (cup + handle modeled separately and intersecting),
  non-conforming, with overlapping faces. Not simulation-ready.
- **Simulation mesh** — usually a tetrahedral or coarse surface mesh authored separately. Stable, but adds an
  embedding step (FFD-style cage [SP86], tet embedding [MG04, MTG04], or skinning [MTLT]) and complicates
  fracture: splitting the visual mesh in lockstep with the sim mesh requires per-frame booleans or a
  conforming-tet visualization that exposes the tet structure as artifacts ([JML+16]).
- **Collision proxy** — a third decomposition into convex shapes for the rigid-body engine.

Authoring three meshes is expensive, and the gap between them produces visible artifacts: collision contact
that doesn't match silhouette, deformation that doesn't match the visible surface, fracture lines that snap to
the sim mesh's tessellation. The historical justification was performance — game consoles could not afford to
simulate visual-resolution geometry.

The paper's bet is that this constraint is loosening (cloud rendering, more powerful GPUs, VR raising the bar
on fidelity) and a method that **collapses the three representations into one** is now worth the runtime cost.
The contribution is the construction and the simulator that makes one representation viable for all three
roles.

## Method

### Stage 1 — Physical mesh construction (§3.1)

Input: visual mesh with triangle/quad faces, plus per-submesh material parameters (thickness, stiffness,
material type — soft / plastic / brittle). Faces may intersect, and submeshes may be disconnected.

**Per-face extrusion.** Each face is extruded along `−n` by the user-defined thickness to form a convex
polyhedron. These are the primitives. Figure 2 (top row) shows the 2D cut: dark-blue input triangles become
light-blue volumetric prisms.

**Enclosed-region fusion (optional).** When a closed set of visual faces bounds a convex volume (Figure 3),
fuse them into a single primitive instead of extruding each face. Faces not part of an enclosed convex region
fall back to extrusion. This keeps primitive count manageable on sealed parts.

**Connectivity graph.** Connect every pair of primitives that touch or overlap in the rest configuration.
This is what physically glues independent input meshes together: a cup and its handle, modeled as separate
intersecting submeshes, become one connected body for free.

**Joints (optional).** Higher-level structure (car body ↔ wheel) is authored by placing **named boxes** in the
DCC tool — the box's pose defines joint location and extent, the box's name defines joint type and which
submeshes participate (Figure 4). Spatial extent matters later for fracture (joints persist by re-attaching to
new objects whose volume intersects the joint box).

**Soft-on-rigid attachment** is automatic: any soft-body primitive overlapping a rigid body in rest pose is
attached to it. Soft tires attach to rigid wheel hubs without explicit authoring.

### Stage 2 — Simulation via oriented particles with affine deformation (§3.2–3.3)

Each primitive is an [oriented particle](../../pbd/docs/Position-Based-Dynamics.md) — a position + orientation
pair driven by [shape matching](../../pbd/docs/Position-Based-Dynamics.md) over its 1-ring neighborhood in the
connectivity graph. One shape-matching group per primitive.

**Why convex polyhedra instead of ellipsoids ([MC11]'s original choice).** Ellipsoids smooth away flat
surfaces and sharp corners — the things that matter for visual fidelity *and* for collision contact after
fracture (small fragments wedged together need precise face/edge contact, not blurry ellipsoid response).
Convex polyhedra also admit standard collision detection: sweep-and-prune for broad phase, separating-axis
theorem for narrow phase.

**Three-stage broad phase.** Naive AABB sweep on 16k primitives dies. Solution: (1) AABB-test entire object
bounds; (2) for each intersecting object pair, find primitives whose AABBs intersect the *cut volume* of the
two object bounds; (3) sweep-and-prune those primitives, then SAT narrow phase. The cut-volume restriction is
the key — most primitives in a large object aren't candidates for any given collision.

#### Affine deformation of primitives (§3.3) — the technical centerpiece

Standard oriented particles only update **position + rotation** of each primitive, leaving its shape rigid.
This produces visible gaps between neighboring primitives under bending (Figure 2 third row, Figure 5 top).
Deforming primitives arbitrarily would destroy convexity and break collision/fracture downstream.

The fix is to apply the **local affine transform** that shape matching already computes, restricted to its
linear part — linearity preserves convexity.

The local moment matrix, accumulated over a primitive and its neighbors, is

```
A = Σ_i (A_i + m_i · x_i · x̄_i^T) − M · c · c̄^T
```

where `A_i = (1/5) m_i r_i^2 R_i`; `R_i, m_i, r_i, x̄_i, x_i` are orientation, mass, radius, rest position,
current position of primitive `i`; `M` is total neighborhood mass; `c̄, c` are rest and current centers of mass.

`A` alone is what [MC11] uses to extract optimal rotation, but it's *not* the best-fit affine map — see
[MHT05]. To recover the actual affine transform `D`, divide out the rest-pose moment:

```
D = A · Ā⁻¹       where    Ā = Σ_i (Ā_i + m_i · x̄_i · x̄_i^T) − M · c̄ · c̄^T,    Ā_i = (1/5) m_i r_i^2 I
```

`Ā` is a constant, so `Ā⁻¹` is precomputed once. `D` is computed **once per timestep** before collision
handling — not per solver iteration — making it nearly free.

Each primitive's vertices get pushed by `D` (relative to the primitive's origin). Since `D` is linear, the
deformed primitive remains convex; SAT collision and the fracture pipeline stay valid. Figure 5 middle vs.
top: gaps that look like blinds in the curtain shrink to almost nothing.

This idea is called out as portable: any framework with a shape-matching step can apply `D` to its primitives
to close gaps without giving up convexity.

### Stage 3 — Visualization mesh (§3.4)

Even with affine deformation, residual gaps remain at primitive boundaries. The fix is structural: define
the visualization mesh as a *vertex-averaged* version of the simulation mesh, not a separate mesh.

**Global ids.** Every primitive vertex stores a `global id` scoped to its containing body. Vertices with the
same id come from the same source location (a vertex shared by multiple input faces, or two extruded vertices
that landed at the same point).

**Per-frame averaging.** Two-pass:
1. Sum positions of all vertices into an array indexed by `global id`. Track `val(id)` = number of vertices
   contributing to each id.
2. Each vertex reads back `sum(id) / val(id)` as its visual position.

Even in rest pose this produces a watertight inner surface (Figure 2 second row), because the inner-side
vertices of neighboring extruded prisms collapse to a single visual point. Under deformation (Figure 2
bottom), the averaging continues to close gaps automatically.

**Initial id assignment is free.** Use the input visual mesh's vertex indices as global ids. For extruded
inner-side vertices, mirror the same connectivity from the outer side. The visual mesh is then "the same mesh
the artist drew, with averaged positions."

### Stage 4 — Plastic deformation: rigid-between-impacts (§3.5)

Simulating 16k primitives at 60 fps is not feasible if every primitive runs its solver every frame. The
observation: for plastic objects (car body, metal door), the object behaves as a single rigid body **between
deforming impacts** — primitives don't move relative to each other.

So:
- **Between impacts**: simulate the object as one rigid body in PhysX. No per-primitive cost.
- **On impact**: iterate contacts; if relative normal velocity exceeds the material's deformation threshold,
  compute a local displacement `(v_rel · Δt)`, transform into object frame, displace participating primitives
  along the contact normal, then run the oriented-particle solver in **quasi-static** mode (no inertia,
  iterate to equilibrium) with primitives intersecting joint volumes pinned as boundary conditions.
- After deformation: recompute object center of mass and inertia tensor.

This is why **joints are defined with spatial extent**, not as point pivots — they double as boundary
conditions for the local static solver.

Cost: 1–10 ms per hit on the car. Net: 16k primitives, 21 objects, 60+ fps. The static-solver-on-hit pattern
is what makes the WYSIWYS target affordable.

### Stage 5 — Subdivision, fracture, tearing (§3.6)

Builds on [MCK13]'s convex-decomposition fracture: a **fracture pattern** (a connected set of convex cells) is
positioned at the impact, each primitive is cut against each pattern cell, and pieces inside each cell are
combined into a new object.

**Three extensions:**

1. **Deformation patterns (adaptive LOD).** Same fracture-pattern machinery, but **don't separate** resulting
   pieces into independent objects — keep them in the original body. Effect: the mesh is locally subdivided
   around the impact, allowing detailed deformation in originally coarse regions. The metal door in Figure 10
   has only 8 faces per wing initially; deformation patterns introduce thousands of small primitives where
   bullets hit, producing fine-scale dents the original tessellation couldn't represent.

2. **Ductile fracture / tearing.** Mark connections as `tearable`, then break them when distance exceeds
   `strain_limit · rest_length`. **Critical detail**: only tear connections introduced *by the fracture
   pattern*, not those from the initial mesh — otherwise the input tessellation becomes visible along tear
   lines. By making only a subset of pattern-introduced links tearable, the artist controls tear paths
   directly (Figure 6 — pre-defined cracks).

3. **Incremental visual-mesh updates.** The visual mesh is implicit (averaged primitive vertices), so
   fracture/tearing reduces to **updating global ids**, not Boolean mesh ops:

   - **Fracture** (Figure 8): apply patterns in the **undeformed** configuration. Vertices that should share a
     global id are exactly those at the same rest-pose position — found by sorting along x/y/z. New ids are
     assigned trivially. The deformed visual mesh updates automatically next averaging pass.
   - **Tearing** (Figure 9): for each broken link, flood-fill the connectivity graph from each adjacent
     primitive, visiting only neighbors that share the broken link's global id. If the flood-fill set is
     smaller than `val(id)`, the id has split — allocate a new id, replace in visited primitives, decrement
     `val(id)`. Local procedure, runs only on broken-link neighborhoods.
   - **Joints**: when an object splits, find new sub-objects whose volume intersects the joint box, clone the
     joint with the new object reference, delete the original.

The one-to-one face/primitive correspondence is what makes all this cheap. There is no embedding to update
and no Boolean operation on the visual surface — everything is id bookkeeping.

## Results

Single core, Intel Core i7 @ 3.1 GHz. Non-optimized serial; authors expect substantial GPU speedup.

| Scene | Objects | Joints | Primitives | Frame rate | Notes |
|---|---|---|---|---|---|
| Car (Fig 1, 7) | 21 | 28 | 16k | 60+ fps | Authoring <1 hr in Blender; 1–10 ms per static-solve on hit |
| Curtain (Fig 5) | 1 | — | (per fabric) | — | Demonstrates affine-`D` gap closure |
| Metal door (Fig 10) | 4 | 4 | 700 → +1k per hit | 100+ fps between impacts | Deformation patterns add detail at run-time |
| Monster truck (Fig 11 bottom) | — | 9 | 45k | 30 fps | Soft tires (2.5k primitives each) dominate cost |
| Truck on bridge (Fig 11 top) | — | — | +600 +1.5k attach | 20 fps | Mixed soft/rigid + attachments |

The qualitative claims (closed gaps, watertight inner surface, fracture lines that don't expose tessellation,
soft-rigid attachments persisting through tearing) are demonstrated rather than measured numerically — this is
a 2016 MiG short paper, not a benchmark study.

## Limitations

- **Visual primitive count = simulation primitive count.** Every visible face becomes a simulation primitive.
  Acknowledged: small features should probably be folded into normal maps and only larger-than-threshold
  geometry kept in the physical mesh. The paper does not implement this; it's listed as future work.
- **Linear-only primitive deformation.** `D` is a single affine map per primitive — no bending, no
  higher-order deformation within a primitive. Adequate for the mesh densities shown, would limit fidelity at
  very coarse densities.
- **Convexity assumption.** Every primitive must be convex. Concave faces must be triangulated first; complex
  enclosed regions need a convex-decomposition step before the enclosed-region fusion can fire.
- **Static solver requires extended joints.** Joints are spatial volumes, not point pivots, because they're
  reused as boundary conditions for the local plastic-deformation solver. Authoring joints as boxes is mildly
  unnatural and constrains joint placement.
- **Author-controlled fracture-pattern tearability.** Realistic crack propagation is not simulated from
  stress; the artist marks subsets of fracture-pattern links tearable. Trades physical correctness for
  predictable, art-directable cracks — appropriate for games, less so for offline VFX.
- **Serial CPU implementation.** No parallelization; 30 fps on the monster truck is on a single core. Frame
  rates would scale on GPU but the paper doesn't show it.

## Relationship to neighboring work

- **Built on [PBD (Müller et al. 2007)](../../pbd/docs/Position-Based-Dynamics.md) and the unified PBD solver
  [MMCK14]** — the entire simulation runs as PBD constraints (shape matching, contact, joints). Position-based
  rigid bodies follow [DCB14]; the inertia/implicit-Euler extension is [BML+14], damping-reduction via
  second-order velocity update is [BMM15].
- **Extends oriented particles [MC11]** in two ways: convex polyhedra instead of ellipsoids (sharp features +
  precise contact) and applying the affine `D` to deform primitives (closes gaps). The deformation idea is
  flagged as portable to other shape-matching frameworks.
- **Reuses [MCK13]'s convex-decomposition fracture** wholesale, then adds (a) deformation patterns for
  adaptive LOD, (b) tearable subsets, (c) the global-id scheme for incremental visual-mesh updates.
- **Contrast with embedding approaches.** [MG04]/[MTG04] embed visual meshes in tet/grid sim meshes;
  [JML+16] simulates a *conforming* tet mesh as the visualization, which leaks the tet structure into fracture
  lines. The paper's one-mesh-many-roles approach avoids both the embedding cost and the conforming-tet
  artifact.
- **Contrast with shape-matching ancestors** [MHT05, RJ07, FGBP11, MKB+10] — those operate on points or
  control points; this paper carries the same shape-matching machinery to a primitive-with-extent
  representation that doubles as a collision shape.
- **Could compose with proxy-asset pipelines** — when full WYSIWYS is too expensive, the
  [proxy-asset-generation pipeline](./Proxy-Asset-Generation.md) produces a low-poly simulation proxy and
  drives the visual mesh via LBS. The two papers stake out opposite ends of the same tradeoff: one mesh for
  everything (this paper) vs. one mesh for sim, one for visual, with learned skinning between (Zheng 2024).
