---
title: Types
description: Public simulation and environment pytrees, plus rendering configuration.
---

Simulation and environment state types are `NamedTuple` subclasses, which gives automatic JAX pytree registration for `jax.jit` and `jax.vmap`. `RenderConfig` is a standard Python dataclass because rendering runs outside the JIT-compiled simulation.

**Batching model:** for a fixed morphology, `SimState` is batched while topology, constants, and actuator mappings can be shared. When morphology varies, its material and actuator data may also vary across the batch; see the [dense-grid guide](../../guides/dense-grid/).

---

## Simulation state

### `SimState`

The core mutable state passed through every simulation step. Vmapped across population during training.

| Field | Shape | dtype | Description |
|---|---|---|---|
| `positions` | `(n_points, 2)` | float32 | Current point positions in world space (x, y) |
| `velocities` | `(n_points, 2)` | float32 | Current point velocities |
| `positions_last` | `(n_points, 2)` | float32 | Positions from the previous timestep, used by the integrator |
| `velocities_true` | `(n_points, 2)` | float32 | Finite-difference velocities, used for observation and reward |
| `masses` | `(n_points, 2)` | float32 | Per-point mass (broadcast across x/y) |
| `fixed` | `(n_points, 2)` | bool | Whether each point is pinned (zero velocity applied) |
| `point_mask` | `(n_points,)` | bool | Active point mask; padded points are `False` |
| `spring_rest_length` | `(n_springs,)` | float32 | Current rest length used by the spring force computation |
| `spring_rest_length_goal` | `(n_springs,)` | float32 | Target rest length that actuators drive toward |
| `spring_init_rest_length` | `(n_springs,)` | float32 | Rest length at spawn time; reference for deformation measurement |
| `spring_const` | `(n_springs,)` | float32 | Spring stiffness coefficient |
| `spring_mask` | `(n_springs,)` | bool | Active spring mask; padded springs are `False` |
| `rigid_spring_mask` | `(n_springs,)` | bool | `True` for springs eligible for Phase 2 PBD. Current builders mark `RIGID` springs and `FIXED` springs that remain in dynamic mixed objects. |
| `external_forces` | `(n_points, 2)` | float32 | Per-point external forces applied before integration |
| `tangential_deformation` | `(n_points, 2)` | float32 | Optional tangential memory for non-slope static friction models |
| `friction_anchor` | `(n_points, 2)` | float32 | Optional world-space PBD anchor for non-slope terrain friction |
| `friction_anchor_active` | `(n_points,)` | bool | Active mask for friction anchors |
| `friction_anchor_no_contact_count` | `(n_points,)` | int32 | Persistence counter for friction anchors |
| `static_manifold_*` | varies | mixed | Legacy static-contact memory fields; slope manifold behavior is inactive in the stable boundary |

---

## Shared simulation data

These are constructed once per world and can be shared across rollouts with the same morphology.

### `PhysicsConstants`

Scalar hyperparameters that govern the simulation.

| Field | dtype | Description |
|---|---|---|
| `dt` | float | Timestep duration in seconds |
| `gravity` | float | Gravitational acceleration (positive = downward) |
| `viscous_drag` | float | Linear drag coefficient applied to all velocities |
| `collision_const_ground` | float | Impulse strength for robot-ground collisions |
| `collision_const_obj` | float | Impulse strength for robot-object collisions |
| `collision_vel_damping` | float | Velocity damping applied on collision contact |
| `ground_vel_damping` | float | Ground-plane normal damping extension |
| `collision_base_dist` | float | Minimum separation distance before collision response |
| `dynamic_friction_const` | float | Friction coefficient for dynamic contacts |
| `friction_const` | float | Object-object friction constant (default 1200) |
| `ground_friction_const` | float | Ground friction constant (default 10) |
| `actuator_convergence` | float | Rate at which spring rest length converges toward goal |
| `integrator` | int | Integration scheme: `0` = RK4, `1` = symplectic Euler |
| `static_friction_const` | float | Opt-in non-slope terrain stiction coefficient |
| `terrain_stiction_stiffness` | float | Opt-in non-slope terrain stiction stiffness |
| `ground_stiction_stiffness` | float | Opt-in ground-plane stiction stiffness |
| `floor_static_friction_const` | float | Opt-in flat floor stiction coefficient independent of terrain stiction |
| `floor_stiction_stiffness` | float | Opt-in flat floor stiction stiffness |
| `ground_static_friction_const` | float | Opt-in ground-plane static-friction coefficient |
| `terrain_stiction_scope` | int | Scope selector for non-slope static terrain stiction; slope-only scope is currently inert |
| `friction_model` | int | Compatibility selector for projected friction memory on non-slope stiction contacts |
| `friction_constraint_mode` | int | Optional terrain PBD anchor mode for non-slope stiction contacts |
| `slope_contact_*` | mixed | Compatibility fields retained as no-ops for geometry-only experimental slopes |
| `slope_grip_*` | mixed | Compatibility fields retained as no-ops for geometry-only experimental slopes |

Use `default_physics_constants()` to get the standard configuration. Flat
ground and flat static-terrain friction are stable. Experimental slope terrain
does not use tangential friction/support fields.

### `SpringTopology`

The fixed connectivity graph of the spring-mass system. Shared across batch.

| Field | Shape | dtype | Description |
|---|---|---|---|
| `a_idx` | `(n_springs,)` | int32 | Index of the first endpoint point for each spring |
| `b_idx` | `(n_springs,)` | int32 | Index of the second endpoint point for each spring |

### `ActuatorInfo`

Maps actuator cells to the springs they control. Shared across batch.

| Field | Shape | dtype | Description |
|---|---|---|---|
| `cell_spring_indices` | `(n_actuator_cells, 2)` | int32 | Indices of the two springs controlled by each actuator cell |
| `spring_act_count` | `(n_springs,)` | float32 | Number of actuator cells sharing each spring (for averaging) |
| `actuated_spring_mask` | `(n_springs,)` | bool | `True` for springs that belong to at least one actuator cell |

### `PerAxisActuatorInfo`

Extended actuator bookkeeping that separates horizontal and vertical springs, used by controllers that apply axis-aligned actions.

| Field | Shape | dtype | Description |
|---|---|---|---|
| `h_spring_pairs` | `(H*W, 2)` | int32 | Horizontal spring pair indices for every grid cell |
| `v_spring_pairs` | `(H*W, 2)` | int32 | Vertical spring pair indices for every grid cell |
| `h_compact_cell_indices` | `(N_compact_max,)` | int32 | Indices of cells with active horizontal actuators |
| `v_compact_cell_indices` | `(N_compact_max,)` | int32 | Indices of cells with active vertical actuators |
| `h_compact_spring_pairs` | `(N_compact_max, 2)` | int32 | Horizontal spring pairs for compact actuator cells only |
| `v_compact_spring_pairs` | `(N_compact_max, 2)` | int32 | Vertical spring pairs for compact actuator cells only |
| `h_compact_count` | `()` | int32 | Number of active horizontal actuator cells |
| `v_compact_count` | `()` | int32 | Number of active vertical actuator cells |
| `actuated_spring_mask` | `(n_springs,)` | bool | `True` for springs controlled by any actuator |
| `spring_act_count` | `(n_springs,)` | float32 | Actuator cell share count per spring |

---

## Collision

### `CollisionData`

Precomputed dynamic collision structures for robot-terrain and robot-robot contacts. Built once from the world geometry and reused each step.

| Field | Shape | dtype | Description |
|---|---|---|---|
| `boxel_corners` | `(n_boxels, 4)` | int32 | Corner point indices for each collision boxel |
| `boxel_edge_a` | `(n_boxels, 4)` | int32 | First endpoint index of each boxel edge |
| `boxel_edge_b` | `(n_boxels, 4)` | int32 | Second endpoint index of each boxel edge |
| `boxel_mask` | `(n_boxels,)` | bool | Active boxel mask |
| `surface_edge_mask` | `(n_boxels, 4)` | bool | `True` for edges that face outward (participate in contact) |
| `boxel_object_id` | `(n_boxels,)` | int32 | Object identifier for each boxel |
| `boxel_is_robot` | `(n_boxels,)` | bool | `True` if the boxel belongs to the robot |
| `boxel_world_vy` | `(n_boxels,)` | int32 | Grid row of the boxel in world space |
| `boxel_world_vx` | `(n_boxels,)` | int32 | Grid column of the boxel in world space |
| `corner_edge_indices` | `(4, 2)` | int32 | Lookup table: for each corner, which two edges share it |
| `n_real_points` | int | — | Count of non-padded points |
| `triple_i/j/k` | `(MAX_TRIPLES,)` | int32 | Point-edge-point triples for dynamic collision pairs |
| `triple_active` | `(MAX_TRIPLES,)` | bool | Active mask for each collision triple |
| `triple_pair_slot` | `(MAX_TRIPLES,)` | int32 | Broadphase slot that produced each triple |
| `self_triple_i/j/k` | `(MAX_SELF_TRIPLES,)` | int32 | Self-collision triples (robot against itself) |
| `self_triple_active` | `(MAX_SELF_TRIPLES,)` | bool | Active mask for self-collision triples |
| `candidate_pair_i/j` | `(MAX_TRIPLES,)` | int32 | Broadphase candidate pairs before narrowphase |
| `candidate_pair_active` | `(MAX_TRIPLES,)` | bool | Active mask for candidate pairs |
| `n_triples` | int | — | Diagnostic count of active collision triples |
| `n_self_triples` | int | — | Diagnostic count of active self-collision triples |
| `n_candidate_pairs` | int | — | Diagnostic count of broadphase candidate pairs |
| `n_robot_surface_boxels` | int | — | Count of active robot boxels that have at least one exposed surface edge |

### `StaticColliderData`

Precomputed collision representation of fixed terrain. Points are constant throughout simulation.

| Field | Shape | dtype | Description |
|---|---|---|---|
| `point_positions` | `(n_static_points, 2)` | float32 | Fixed world-space positions of terrain vertices |
| `cell_vertices` | `(n_static_cells, 4)` | int32 | Vertex indices for each static cell (padded to 4) |
| `cell_vertex_count` | `(n_static_cells,)` | int32 | Actual vertex count per cell (3 for triangles, 4 for quads) |
| `cell_edge_a` | `(n_static_cells, 4)` | int32 | First endpoint index of each cell edge |
| `cell_edge_b` | `(n_static_cells, 4)` | int32 | Second endpoint index of each cell edge |
| `surface_edge_mask` | `(n_static_cells, 4)` | bool | `True` for outward-facing (collidable) edges |
| `cell_types` | `(n_static_cells,)` | int32 | Voxel type constant for each static cell |
| `cell_aabb_min` | `(n_static_cells, 2)` | float32 | Static cell AABB minimum |
| `cell_aabb_max` | `(n_static_cells, 2)` | float32 | Static cell AABB maximum |
| `column_cell_indices` | `(n_columns, max_cells_per_column)` | int32 | Column acceleration structure |
| `column_cell_count` | `(n_columns,)` | int32 | Active static cells per column |

---

## Build pipeline

### `ObjectTemplate`

Describes a single object (robot or terrain block) before it is instantiated into a world. Holds 35+ fields covering the grid layout, voxel types, spring topology, mass and stiffness assignments, point masks, and axis-aligned actuator metadata. You generally do not construct this directly — it is produced by the robot/terrain loaders and consumed by the world builder.

### `WorldTemplate`

A complete description of a world layout, ready to be built into a `BuiltWorld`.

| Field | Type | Description |
|---|---|---|
| `grid_h` | int | Total world grid height in voxels |
| `grid_w` | int | Total world grid width in voxels |
| `robot_name` | str | Identifier string for the robot morphology |
| `object_templates` | `tuple[ObjectTemplate, ...]` | Ordered sequence of all objects in the world |
| `robot_template_index` | int | Index into `object_templates` identifying the robot |

### `WorldTemplateSet`

Pairs a primary world with an optional mirror for symmetric training.

| Field | Type | Description |
|---|---|---|
| `primary` | `WorldTemplate` | The canonical world |
| `mirror` | `WorldTemplate \| None` | Horizontally mirrored world, or `None` if unused |
| `mirror_mode` | `Literal["none", "paired"]` | How the mirror is used during training |

### `BuiltWorld`

The fully instantiated, ready-to-simulate world. All arrays are sized and padded; this is what environments hold at construction time.

| Field | Type | Description |
|---|---|---|
| `sim_state` | `SimState` | Initial simulation state at spawn |
| `topology` | `SpringTopology` | Fixed spring connectivity graph |
| `constants` | `PhysicsConstants` | Physics hyperparameters |
| `collision_data` | `CollisionData` | Dynamic collision structures |
| `static_collider_data` | `StaticColliderData` | Static terrain collision data |
| `actuator_info` | `ActuatorInfo` | Actuator-to-spring mapping |
| `per_axis_info` | `PerAxisActuatorInfo` | Per-axis actuator bookkeeping |
| `deformation_info` | `DeformationInfo` | Robot deformation measurement data |
| `robot_point_indices` | `jnp.ndarray` | Indices into `sim_state.positions` for robot points |
| `robot_point_mask` | `jnp.ndarray` | Boolean mask over all points identifying robot points |
| `dynamic_terrain_point_indices` | `jnp.ndarray` | Indices for moveable terrain points |
| `fixed_terrain_sample_positions` | `jnp.ndarray` | Static sample positions used for terrain observations |
| `render_info` | `RenderInfo` | Static rendering metadata |
| `grid_h` | int | World grid height |
| `grid_w` | int | World grid width |

---

## Environment states

Each environment carries a `sim_state` plus the minimal bookkeeping needed to compute rewards and detect episode termination.

### `EnvState`

Used by the locomotion (walker) environment.

| Field | Shape | dtype | Description |
|---|---|---|---|
| `sim_state` | `SimState` | — | Full simulation state |
| `step_count` | `()` | int32 | Steps elapsed in the current episode |
| `done` | `()` | bool | Whether the episode has ended |
| `prev_com_x` | `()` | float32 | Robot centre-of-mass x from the previous step, for velocity reward |

### `ShapeEnvState`

Used by shape (height / wingspan) environments. Rewards are based on the robot's spatial extent rather than locomotion.

| Field | Shape | dtype | Description |
|---|---|---|---|
| `sim_state` | `SimState` | — | Full simulation state |
| `step_count` | `()` | int32 | Steps elapsed |
| `done` | `()` | bool | Episode ended flag |
| `prev_span` | `()` | float32 | Previous step's height or wingspan measurement |

### `OrientedEnvState`

Used by environments that penalise deviating from the robot's initial orientation (e.g. balance or directed locomotion tasks).

| Field | Shape | dtype | Description |
|---|---|---|---|
| `sim_state` | `SimState` | — | Full simulation state |
| `step_count` | `()` | int32 | Steps elapsed |
| `done` | `()` | bool | Episode ended flag |
| `prev_com_x` | `()` | float32 | Previous centre-of-mass x |
| `initial_robot_positions` | `(n_robot_points, 2)` | float32 | Spawn positions used as orientation reference |

### `JumperEnvState`

Used by the jump environment. Tracks both x and y components of the centre of mass.

| Field | Shape | dtype | Description |
|---|---|---|---|
| `sim_state` | `SimState` | — | Full simulation state |
| `step_count` | `()` | int32 | Steps elapsed |
| `done` | `()` | bool | Episode ended flag |
| `prev_com_x` | `()` | float32 | Previous centre-of-mass x |
| `prev_com_y` | `()` | float32 | Previous centre-of-mass y, for vertical reward |

### `ClimberEnvState`

Used by the climb environment. Rewards upward movement only.

| Field | Shape | dtype | Description |
|---|---|---|---|
| `sim_state` | `SimState` | — | Full simulation state |
| `step_count` | `()` | int32 | Steps elapsed |
| `done` | `()` | bool | Episode ended flag |
| `prev_com_y` | `()` | float32 | Previous centre-of-mass y |

---

## Grid data

### `GridData`

The raw grid-aligned spring-mass layout for a single object before it is merged into a world. Produced by the object builder.

| Field | Shape | dtype | Description |
|---|---|---|---|
| `positions` | `((H+1)*(W+1), 2)` | float32 | Initial vertex positions on the grid lattice |
| `spring_a_idx` | `(n_springs,)` | int32 | First endpoint index of each spring |
| `spring_b_idx` | `(n_springs,)` | int32 | Second endpoint index of each spring |
| `spring_rest_length` | `(n_springs,)` | float32 | Rest length of each spring |
| `spring_types` | `(n_springs,)` | int32 | Spring orientation: `0` = horizontal, `1` = vertical, `2` = diagonal |
| `spring_cell_a` | `(n_springs,)` | int32 | Primary cell owning each spring |
| `spring_cell_b` | `(n_springs,)` | int32 | Secondary cell sharing each spring (if any) |
| `cell_corner_indices` | `(H*W, 4)` | int32 | Corner point indices per cell: `[BL, BR, TR, TL]` |
| `cell_grid_vy` | `(H*W,)` | int32 | Grid row of each cell |
| `cell_grid_vx` | `(H*W,)` | int32 | Grid column of each cell |
| `cell_horiz_bottom` | `(H*W,)` | int32 | Bottom horizontal spring index for each cell |
| `cell_horiz_top` | `(H*W,)` | int32 | Top horizontal spring index for each cell |
| `cell_vert_left` | `(H*W,)` | int32 | Left vertical spring index for each cell |
| `cell_vert_right` | `(H*W,)` | int32 | Right vertical spring index for each cell |
| `cell_diag1` | `(H*W,)` | int32 | First diagonal spring index for each cell |
| `cell_diag2` | `(H*W,)` | int32 | Second diagonal spring index for each cell |
| `H` | int | — | Grid height in voxels |
| `W` | int | — | Grid width in voxels |

### `DeformationInfo`

Indices needed to compute per-voxel deformation observations from spring rest lengths.

| Field | Shape | dtype | Description |
|---|---|---|---|
| `voxel_horiz_springs` | `(n_robot_voxels, 2)` | int32 | Indices of the top and bottom horizontal springs per voxel; `-1` if missing |
| `voxel_vert_springs` | `(n_robot_voxels, 2)` | int32 | Indices of the left and right vertical springs per voxel; `-1` if missing |
| `n_robot_voxels` | int | — | Number of active robot voxels |

---

## Rendering

### `RenderInfo`

Static, Python-side metadata built once and used each render call. Contains numpy arrays (not JAX).

| Field | Shape | dtype | Description |
|---|---|---|---|
| `cell_vertices` | `(n_cells, 4)` | int32 | Vertex indices per cell |
| `cell_vertex_count` | `(n_cells,)` | int32 | Actual vertex count (3 or 4) per cell |
| `cell_types` | `(n_cells,)` | int32 | Voxel type constant for each cell |
| `cell_mask` | `(n_cells,)` | bool | Active cell mask |
| `surface_edge_mask` | `(n_cells, 4)` | bool | Outward-facing edge mask for rendering borders |
| `cell_is_robot` | `(n_cells,)` | bool | `True` for robot cells |
| `cell_is_dynamic` | `(n_cells,)` | bool | `True` for non-static cells |
| `cell_object_id` | `(n_cells,)` | int32 | Object identifier per cell |
| `robot_point_indices` | `(n_robot_points,)` | int32 | Which global points belong to the robot |
| `grid_h` | int | — | World grid height |
| `grid_w` | int | — | World grid width |
| `actuator_cell_indices` | `(n_actuator_cells,)` | int32 | Cell indices for actuator colouring |
| `cell_h_spring_pairs` | `(n_cells, 2)` | int32 | Horizontal spring pair indices per cell |
| `cell_v_spring_pairs` | `(n_cells, 2)` | int32 | Vertical spring pair indices per cell |
| `spring_init_rest_length` | `(n_springs,)` | float32 | Initial rest lengths for deformation visualisation |
| `static_point_positions` | `(n_static_points, 2)` | float32 | Fixed terrain vertex positions |

### `RenderStepOutput`

The record captured by `make_render_episode_step` for rendering a rollout.

| Field | Shape | dtype | Description |
|---|---|---|---|
| `positions` | `(n_points, 2)` | float32 | Current point positions |
| `spring_rest_length` | `(n_springs,)` | float32 | Current spring rest lengths (for deformation colour) |
| `reward` | `()` | float32 | Reward received this step |
| `done` | `()` | bool | Whether the episode ended |

### `StepOutput`

The record returned by the `make_*_episode_step` factories for `lax.scan`. An environment’s `step()` itself returns `(obs, state, reward, done)`.

| Field | Shape | dtype | Description |
|---|---|---|---|
| `obs` | `(obs_dim,)` | float32 | Observation vector for the current step |
| `reward` | `()` | float32 | Reward received this step |
| `done` | `()` | bool | Whether the episode ended |
| `positions` | `(n_points, 2)` | float32 | Current point positions (for logging / visualisation) |

### `RenderConfig`

A `@dataclass` (not a `NamedTuple`) controlling the renderer. All fields have defaults.

| Field | Default | Description |
|---|---|---|
| `enabled` | `False` | Whether to render at all |
| `width` | `600` | Output frame width in pixels |
| `height` | `300` | Output frame height in pixels |
| `fps` | `50` | Target playback frame rate |
| `real_time` | `True` | Match physics timing by computing the frame skip automatically |
| `frame_skip` | `7` | Manual frame skip used when `real_time=False` |
| `show_grid` | `False` | Overlay the voxel grid |
| `show_edges` | `True` | Draw cell border edges |
| `dynamic_actuator_colors` | `True` | Colour actuators by current activation level |
| `camera_padding` | `0.3` | World units of padding around the robot |
| `viewport_width` | `4.0` | World-space units visible horizontally |
| `camera_mode` | `"fit_robot"` | Fit the viewport to the robot; any other value uses a fixed viewport width |
| `camera_smoothing` | `0.18` | Camera interpolation factor (`0` is instant, `1` is static) |
| `supersample` | `2` | Anti-aliasing multiplier (render at N× then downscale) |
| `grid_major_every` | `5` | Draw a heavier grid line every N cells |
| `show_minor_grid` | `False` | Draw minor grid lines |
| `edge_width_px` | `2` | Edge stroke width in pixels |
| `output_dir` | `"renders"` | Directory for saved GIFs |
| `target_frames` | `None` | Optional frame budget cap |
| `sim_time_per_step` | `None` | Override the simulated duration of each environment step for real-time pacing |
| `max_colors` | `256` | Palette size cap for GIF optimisation |
| `optimize` | `False` | Apply GIF frame optimisation |
| `dither` | `True` | Enable Floyd-Steinberg dithering in GIF output |

---

## Factory

### `default_physics_constants()`

```python
from jax_evogym import default_physics_constants

constants = default_physics_constants()
```

Returns a `PhysicsConstants` instance with the standard simulation parameters.
Slope contact/grip fields remain present on the tuple for compatibility, but
they are no-ops under the current geometry-only slope boundary. Override
individual fields using `._replace()`:

```python
constants = default_physics_constants()._replace(integrator=1)
```
