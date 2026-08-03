---
title: API Boundary
description: Stable public surface, compatibility-only symbols, and current experimental exclusions.
---

The stable public API is the square-cell JAX EvoGym core:

- top-level voxel constants: `EMPTY`, `RIGID`, `SOFT`, `H_ACT`, `V_ACT`, `FIXED`, `CONTRACTILE`
- world construction: `EvoWorld`, `WorldObject`, `compile_world_template`, `compile_world_templates`, `instantiate_world`
- simulation/runtime types: `SimState`, `PhysicsConstants`, `CollisionData`, `StaticColliderData`, `ActuatorInfo`, `PerAxisActuatorInfo`, `DeformationInfo`, `BuiltWorld`
- simulation helpers, built-in environments, rendering helpers, and the dense-grid JAX build path documented in this site

## Compatibility-only surface

Some symbols remain importable because old fixtures and internal tests need to
load or inspect them. These are not part of the stable release contract:

- slope ids `7` through `30` in `jax_evogym.constants`
- slope mirror/rotation/pairing tables
- `allow_experimental_slopes=True`
- `slope_contact_*` and `slope_grip_*` fields on `PhysicsConstants`
- slope-only stiction scope

`EvoWorld` accepts known legacy slope ids so JSON fixtures can be loaded and
inspected. Stable compilation rejects slopes by default.

## Experimental slopes

Slopes are experimental geometry only. With `allow_experimental_slopes=True`,
the builder may compile legacy slope cells into static polygon collider and
render geometry. Sloped edges receive normal collision response, but tangential
friction, stiction, near-contact support, slope grip, and persistent slope
manifold behavior are disabled.

Slopes are left out of the designer palette, stable examples, and release gates
because they still need a defensible friction/support model before they can be
treated as research-library API.

Flat `FIXED` terrain is different: ground-plane friction, flat static-terrain
tangential friction, and opt-in flat floor stiction are stable behavior.

## JAX-side additions

Compared with original EvoGym, the stable JAX library adds:

- object-separated world compilation instead of one merged grid
- static fixed-terrain colliders omitted from `SimState`
- `CONTRACTILE` per-axis actuation
- deformation observations
- pure-JAX dense-grid builders for vmap-heavy morphology search
- ownership-aware rendering and the browser designer
- documented damping/friction extensions for stable square-cell simulation
