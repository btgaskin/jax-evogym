---
title: Voxel Types
description: Stable cell type constants, type sets, spring constants, and physics constants.
---

## Stable cell type constants

| Constant | Value | Category | Description |
|----------|-------|----------|-------------|
| `EMPTY` | 0 | — | No cell. Disables a grid position. |
| `RIGID` | 1 | Dynamic | Stiff structural material. Highest spring constants. |
| `SOFT` | 2 | Dynamic | Deformable material. Lower spring constants than rigid. |
| `H_ACT` | 3 | Dynamic | Horizontal actuator. Action controls horizontal edge springs. |
| `V_ACT` | 4 | Dynamic | Vertical actuator. Action controls vertical edge springs. |
| `FIXED` | 5 | Static | Anchored, immovable. Used for ground, walls, and platforms. |
| `CONTRACTILE` | 6 | Dynamic | Universal actuator with independent per-axis control at runtime. |

## Unsupported experimental values

Values `7` through `30` are reserved for legacy slope terrain fixtures:

- `SLOPE_*`
- `SLOPE2_*`
- `SLOPE3_*`

These values remain in `jax_evogym.constants` so older fixtures and internal
tests can still be inspected. They are not exported from the package top level,
are not shown in the public designer palette, and are rejected by
`compile_world_template(...)` unless `allow_experimental_slopes=True` is passed.

When explicitly enabled, slopes are geometry-only. Sloped edges receive normal
collision response, but slope friction, slope grip, near-contact support,
stiction, and persistent slope manifold behavior are disabled. They are omitted
from the designer and stable examples because that friction/support model is
still unresolved.

## Type sets

```python
ACTUATOR_TYPES = {H_ACT, V_ACT, CONTRACTILE}

STABLE_STATIC_TERRAIN_TYPES = {FIXED}

DYNAMIC_VOXEL_TYPES = {RIGID, SOFT, H_ACT, V_ACT, CONTRACTILE}

STABLE_WORLD_CELL_TYPES = {EMPTY, FIXED, RIGID, SOFT, H_ACT, V_ACT, CONTRACTILE}
```

## Where each stable type is allowed

| Type | In robots | In terrain | Dynamic | Static |
|------|-----------|------------|---------|--------|
| EMPTY | Yes (disables cell) | Yes | — | — |
| RIGID | Yes | Yes | Yes | No |
| SOFT | Yes | Yes | Yes | No |
| H_ACT | Yes | No | Yes | No |
| V_ACT | Yes | No | Yes | No |
| CONTRACTILE | Yes | No | Yes | No |
| FIXED | No | Yes | In mixed objects | In pure fixed objects |

A terrain object containing only `FIXED` cells is compiled as **static** — it
contributes collider/render geometry but no dynamic points, masses, or springs
to `SimState`. A terrain object containing `SOFT`, `RIGID`, or actuator cells
is compiled as **dynamic**.

## Spring constants

Spring constants are material properties sourced from C++ `ObjectCreator.h`.
They are not scaled by cell size.

| Material | Main K | Diagonal K |
|----------|--------|------------|
| Rigid / Fixed | ~85,714,286 | ~42,857,143 |
| Actuator (`H_ACT`, `V_ACT`, `CONTRACTILE`) | 60,000,000 | 30,000,000 |
| Soft | 50,000,000 | 12,500,000 |

Main springs connect adjacent grid points (horizontal and vertical edges).
Diagonal springs connect corner-to-corner within each cell.

## Physics constants

Most square-cell constants are EvoGym-derived. A few values are documented
JAX-side corrections or extensions, including ground normal velocity damping
and disabled-by-default flat-contact stiction hooks.

| Constant | Value | Notes |
|----------|-------|-------|
| `DT` | 0.0001 | Physics step size. |
| `GRAVITY` | 110.0 | Gravity acceleration. |
| `VISCOUS_DRAG` | 0.1 | Linear velocity drag. |
| `PHYSICS_UPDATES_PER_STEP` | 30 | Substeps per environment step. |
| `CELL_SIZE` | 0.1 | Cell width/height. |
| `POINT_MASS` | 1.0 | Point mass. |
| `ACTUATOR_CONVERGENCE` | 0.006 | Rest-length convergence factor. |
| `COLLISION_CONST_GROUND` | 14,000,000 | Ground normal stiffness. |
| `COLLISION_CONST_OBJ` | 2,100,000 | Object-object normal stiffness. |
| `COLLISION_VEL_DAMPING` | 2,030 | Object-object damping extension. |
| `GROUND_VEL_DAMPING` | 5,238 | Ground normal damping extension. |
| `COLLISION_BASE_DIST` | 0.005 | Collision base distance. |
| `STATIC_FRICTION_CONST` | 0.5 | Static-friction coefficient used by opt-in non-slope terrain stiction paths. |
| `DYNAMIC_FRICTION_CONST` | 0.2 | Dynamic friction coefficient. |
| `FRICTION_CONST` | 1,200 | Object/static-object friction scale. |
| `GROUND_FRICTION_CONST` | 10 | Ground-plane friction scale. |

`SLOPE_CONTACT_*` and `SLOPE_GRIP_*` constants are compatibility values only in
the current stable boundary. They are retained for old configs but do not enable
slope friction/support.
