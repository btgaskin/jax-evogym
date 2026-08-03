---
title: Build API
description: Public runtime types and builder entrypoints for object-separated world compilation.
---

The core builder API is the canonical way to compile and instantiate object-separated EvoGym worlds.

For the stable public surface and compatibility-only symbols, see
[API Boundary](../../reference/api-boundary/).

## Public types

### `ObjectTemplate`

Compiled template for one object. Includes:

- object identity and origin
- local grid
- local point/spring data
- dynamic boxel collision geometry
- generalized cell geometry for rendering and static collision
- robot/deformation metadata
- actuator mappings

### `WorldTemplate`

Compiled world template containing:

- world bounds
- robot name
- ordered `ObjectTemplate` entries
- robot template index

### `WorldTemplateSet`

Paired template wrapper:

- `primary`
- `mirror`
- `mirror_mode`

Use this when you want primary/mirrored templates available together, typically for within-genome evaluation.

### `StaticColliderData`

Precomputed static-terrain collider geometry omitted from dynamic `SimState`.

Key fields:

- `point_positions`
- `cell_vertices`
- `cell_vertex_count`
- `cell_edge_a`
- `cell_edge_b`
- `surface_edge_mask`
- `cell_types`

### `BuiltWorld`

Concrete built runtime payload containing:

- `sim_state`
- `topology`
- `constants`
- `collision_data`
- `static_collider_data`
- `actuator_info`
- `per_axis_info`
- `deformation_info`
- `robot_point_indices`
- `robot_point_mask`
- `dynamic_terrain_point_indices`
- `fixed_terrain_sample_positions`
- `render_info`
- `grid_h`
- `grid_w`

## Public functions

### `compile_world_template`

```python
def compile_world_template(
    world: EvoWorld,
    robot_name: str = "robot",
    spring_stiffness_scale: float = 1.0,
    *,
    allow_experimental_slopes: bool = False,
) -> WorldTemplate:
```

Compile a single `WorldTemplate`. `spring_stiffness_scale` must be greater than
zero. `allow_experimental_slopes=True` opts into the unsupported geometry-only
slope path.

### `compile_world_templates`

```python
def compile_world_templates(
    world: EvoWorld,
    robot_name: str = "robot",
    mirror_mode: str = "none",
    spring_stiffness_scale: float = 1.0,
    *,
    allow_experimental_slopes: bool = False,
) -> WorldTemplateSet:
```

Compile a `WorldTemplateSet`.

Supported `mirror_mode` values:

- `"none"`
- `"paired"`

The stiffness scale and experimental-slope setting apply to both the primary
and mirrored templates.

### `instantiate_world`

```python
def instantiate_world(
    template: WorldTemplate,
    robot_override: np.ndarray | jnp.ndarray | None = None,
    *,
    max_triples: int = C.MAX_TRIPLES,              # 131_072
    max_self_triples: int = C.MAX_SELF_TRIPLES,    # 16_384
    spring_stiffness_scale: float = 1.0,
) -> BuiltWorld:
```

Build a concrete `BuiltWorld` from a compiled template. `max_triples` and
`max_self_triples` set the fixed collision-buffer capacities. The stiffness
scale must be greater than zero and is applied when `robot_override` causes the
robot template to be rebuilt.

## `robot_override`

`robot_override` lets you vary morphology inside the robot’s fixed local frame.

Rules:

- shape must match the robot template grid exactly
- use the same body-array orientation as `add_from_array(...)`
- `EMPTY` disables cells inside that frame
- supported robot voxel types can vary across overrides
- slope values are rejected
- terrain objects are never affected

## Unsupported experimental slope cells

Slope ids remain in `jax_evogym.constants` and in internal legacy fixtures, but
they are not part of the stable public build contract.

By default:

- `compile_world_template(...)` rejects slope cells
- `compile_world_templates(...)` rejects slope cells
- public top-level imports do not export slope constants
- the designer palette exposes only stable square-cell types

For internal slope investigations and legacy fixtures, pass:

```python
compile_world_template(world, robot_name="robot", allow_experimental_slopes=True)
```

The experimental path preserves the old slope geometry/validation code while
slopes remain geometry-only. Sloped edges receive normal collision response, but
slope friction, support, grip, stiction, and persistent manifold behavior are
disabled. Flat `FIXED` terrain friction is unaffected.
