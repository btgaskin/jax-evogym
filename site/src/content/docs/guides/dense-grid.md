---
title: Dense Grid Pipeline
description: Pure JAX, vmap-able morphology construction for evolutionary search over robot populations.
---

The dense grid pipeline builds simulation data with JAX after a one-time NumPy setup. Its build functions support `jax.jit` and `jax.vmap`, allowing batches of different morphologies to share a fixed grid layout. Batch capacity depends on grid size, collision buffers, and available device memory.

This is the recommended path for **morphology search**: neuroevolution, QD algorithms, or any regime where the robot body is a search variable alongside or instead of the controller.

## How it works

The pipeline has two stages:

1. **Precompute** — compute all static grid infrastructure (spring indices, point topology, cell mappings) for a given `(H, W)` size. This runs once in NumPy and returns a `GridData` object holding JAX arrays.
2. **Build** — given a morphology array `(H, W) int32` and the precomputed `GridData`, build `SimState`, `ActuatorInfo`, `CollisionData`, and optionally `DeformationInfo`. These functions are pure JAX: JIT-able, vmap-able, and designed for massively parallel rollouts.

The key insight is that `GridData` captures all **structural** facts about the grid (which springs connect which points, which cells own which springs) while the morphology array provides all **material** facts (which cells are active, what type they are). Holding the structure constant and vmapping over the morphology is what enables population-level parallelism.

## precompute_grid

```python
from jax_evogym import precompute_grid

grid_data = precompute_grid(H=5, W=5)
```

Call once per grid size. Returns a `GridData` named tuple with JAX arrays for:

- Point positions on the `(H+1) × (W+1)` point lattice
- Spring endpoint indices and rest lengths for all horizontal, vertical, and diagonal springs
- Per-spring cell adjacency (`spring_cell_a`, `spring_cell_b`)
- Per-cell spring index lookups (`cell_horiz_bottom`, `cell_horiz_top`, etc.)
- Per-cell corner point indices

`precompute_grid` runs in NumPy and is not JIT-able. Call it at startup or in your environment initialisation, then capture `grid_data` in the build function, as in the example below. Do not mark the array-containing `GridData` tuple as a hashable JIT static argument.

The total spring count for an `H × W` grid is `(H+1)*W + H*(W+1) + 2*H*W`.

## jax_build_sim_state

```python
from jax_evogym import jax_build_sim_state

sim_state, spring_topology = jax_build_sim_state(morphology, grid_data)
```

Builds the `SimState` and `SpringTopology` for a composited morphology grid.

| Argument | Type | Description |
|----------|------|-------------|
| `morphology` | `(H, W) int32` | Composited grid (terrain + robot, or robot-only). y-up. |
| `grid_data` | `GridData` | Output of `precompute_grid`. |
| `spawn_x` | `float` | Horizontal offset in grid units (default `0.0`). |
| `spawn_y` | `float` | Vertical offset in grid units (default `0.0`). |

Returns `(SimState, SpringTopology)`. `SimState` holds positions, velocities, masses, fixed-point masks, spring constants, and rest lengths for every point and spring in the grid.

Spring constant assignment matches the C++ `ObjectCreator` two-pass algorithm: main springs (horizontal/vertical) default to `RIGID_MAIN_K`; `SOFT` and actuator cells overwrite their adjacent main springs. Diagonal springs take constants from the owning cell.

A point is **fixed** (immovable) if it is not touched by any non-`FIXED` non-`EMPTY` cell. All points touched exclusively by `FIXED` terrain — or untouched ghost points — are fixed.

## jax_build_actuator_info

```python
from jax_evogym import jax_build_actuator_info

actuator_info, actuator_mask = jax_build_actuator_info(morphology, grid_data)
```

Builds `ActuatorInfo`, which maps each actuator cell to the pair of springs it controls.

- `H_ACT` cells control their bottom and top horizontal springs.
- `V_ACT` cells control their left and right vertical springs.
- `CONTRACTILE` cells are **not** handled here — use `jax_build_per_axis_actuator_info` instead.

The returned arrays are padded to `(H*W, 2)`. Non-actuator rows hold spring index `0` and contribute zero action via scatter-add (safe, not a bug). Use `actuator_mask` `(H*W,)` bool to identify active actuator slots.

## jax_build_per_axis_actuator_info

```python
from jax_evogym import jax_build_per_axis_actuator_info

per_axis_info, actuator_mask = jax_build_per_axis_actuator_info(morphology, grid_data)
```

The full actuator info for controllers that need independent per-axis control. Handles `H_ACT`, `V_ACT`, and `CONTRACTILE`:

- `H_ACT` and `CONTRACTILE` → horizontal spring pairs
- `V_ACT` and `CONTRACTILE` → vertical spring pairs

Returns a `PerAxisActuatorInfo` with separate `h_spring_pairs` and `v_spring_pairs` arrays, plus compact index arrays (`h_compact_cell_indices`, `v_compact_cell_indices`) and per-axis actuator counts. Use this when your controller emits separate horizontal and vertical signals per cell.

## jax_build_collision_data

```python
from jax_evogym import jax_build_collision_data

collision_data = jax_build_collision_data(morphology, robot_mask, grid_data)
```

Builds `CollisionData` from a composited grid and a boolean mask identifying which cells belong to the robot.

| Argument | Type | Description |
|----------|------|-------------|
| `morphology` | `(H, W) int32` | Composited grid (robot + terrain). |
| `robot_mask` | `(H, W) bool` | `True` where robot cells are placed. |
| `grid_data` | `GridData` | Output of `precompute_grid`. |
| `max_triples` | `int` | Cap on collision triple buffer size (default `131072`). |
| `max_self_triples` | `int` | Cap on self-collision buffer size (default `16384`). |
| `max_active_boxels` | `int \| None` | Cap on active boxel count for pairwise broadphase (default: uncapped). |

Uses a sparse-active pair path: only non-empty cells participate in broadphase pairwise checks. The result is indexed into fixed-size triple buffers (`triple_i`, `triple_j`, `triple_k`) for efficient collision resolution during simulation.

## jax_build_deformation_info

```python
from jax_evogym import jax_build_deformation_info

deform_info, deform_mask = jax_build_deformation_info(morphology, grid_data, robot_mask)
```

Builds `DeformationInfo`, used by environments that include voxel deformation in their observations. Non-robot rows receive a `-1` sentinel (which maps to a deformation ratio of `1.0` in `compute_deformation`, i.e. no deformation).

If you are working with the dense grid pipeline outside of the built-in environments and do not need deformation observations, this function is optional.

## jax_compose_grid

```python
from jax_evogym import jax_compose_grid

composited, robot_mask = jax_compose_grid(terrain, robot_morph, spawn_y, spawn_x)
```

Places a robot morphology onto a terrain grid and returns the composited grid along with a boolean `robot_mask`.

| Argument | Type | Description |
|----------|------|-------------|
| `terrain` | `(H, W) int32` | Background terrain grid (y-up, `FIXED` voxels). |
| `robot_morph` | `(rH, rW) int32` | Robot morphology (y-up). |
| `spawn_y`, `spawn_x` | `int` | Spawn position in grid coordinates (bottom-left corner of the robot bounding box). |

Returns `(composited, robot_mask)` where `composited` is `(H, W)` with robot cells overwriting terrain cells at the spawn location, and `robot_mask` is `(H, W) bool`.

Use `jax_compose_grid` when you have a shared terrain grid and want to vmap over a batch of robot morphologies, each placed at the same spawn point.

## Limitations vs. the object-separated build

The dense grid pipeline has deliberate restrictions that make vmapping tractable:

| Feature | Dense grid | Object-separated build |
|---------|-----------|----------------------|
| Slope terrain (types 7–30) | Not supported | Unsupported geometry-only experimental path |
| Multiple terrain objects | Single composited grid only | Any number of named objects |
| Static terrain colliders | Terrain encoded as `FIXED` in the composited grid | Dedicated static collider pipeline |
| Mixed robot + terrain in one grid | Yes (via `jax_compose_grid`) | Objects kept separate until compile |
| vmap over morphologies | Yes | No |
| JIT-able | Yes | No (uses NumPy during compilation) |

If your environment includes walls or multiple independent terrain objects, use `compile_world_template` / `instantiate_world` instead. See [Variable Morphology](../../guides/variable-morphology/) for a comparison.

Flat `FIXED` terrain is supported in both paths. Slope friction/support is not
part of either stable path.

## Typical usage pattern

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax_evogym import (
    precompute_grid,
    jax_compose_grid,
    jax_build_sim_state,
    jax_build_actuator_info,
    jax_build_collision_data,
    FIXED, EMPTY, H_ACT, V_ACT, RIGID,
)

# --- One-time setup ---
H, W = 10, 14
grid_data = precompute_grid(H, W)

# Fixed terrain grid (flat ground, 2 rows high)
terrain = np.zeros((H, W), dtype=np.int32)
terrain[0, :] = FIXED
terrain[1, :] = FIXED
terrain_jnp = jnp.array(terrain)

# Spawn position for the robot
SPAWN_Y, SPAWN_X = 2, 4

# --- Per-morphology build (vmap-able) ---
def build_from_morph(robot_morph):
    """Build simulation data for a single (rH, rW) robot morphology."""
    composited, robot_mask = jax_compose_grid(terrain_jnp, robot_morph, SPAWN_Y, SPAWN_X)
    sim_state, topology = jax_build_sim_state(composited, grid_data)
    actuator_info, _ = jax_build_actuator_info(composited, grid_data)
    collision_data = jax_build_collision_data(composited, robot_mask, grid_data)
    return sim_state, topology, actuator_info, collision_data

# --- Evaluate a population with vmap ---
# population: (batch, rH, rW) int32 array of morphologies
rH, rW = 4, 4
population = jnp.array(
    np.random.choice([EMPTY, RIGID, H_ACT, V_ACT], size=(256, rH, rW))
)

build_batch = jax.vmap(build_from_morph)
sim_states, topologies, actuator_infos, collision_datas = build_batch(population)
```

The `build_batch` call executes in parallel across all 256 morphologies. This example uses `vmap` to batch construction. To compile the whole batched function explicitly, use `jax.jit(build_batch)`; the first call includes compilation.

## When to use each pipeline

| Scenario | Recommended approach |
|----------|---------------------|
| Standard task evaluation, fixed robot body | Built-in environments |
| Fixed terrain, changing robot body at rollout time | `robot_override` + `instantiate_world` |
| Terrain contains multiple objects | `compile_world_template` + `instantiate_world` |
| Population-level morphology search with `jax.vmap` | Dense grid pipeline |
| Custom terrain + vmap over robot body | Dense grid pipeline with `jax_compose_grid` |

For the object-separated build (`compile_world_template`), see [Variable Morphology](../../guides/variable-morphology/).
