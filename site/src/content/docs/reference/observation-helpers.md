---
title: Observation Helpers
description: Pure JAX observation functions used by built-in environments.
---

All observation helpers are pure JAX functions, JIT-compatible, and designed to match the C++ EvoGymBase/BenchmarkBase observation methods. Positions are in physical coordinates (no scaling needed).

Imported from `jax_evogym.environments._obs`.

## get_vel_com_obs

```python
def get_vel_com_obs(
    velocities_true: jnp.ndarray,  # (n_points, 2)
    robot_point_indices: jnp.ndarray,  # (n_robot_points,)
) -> jnp.ndarray:  # (2,)
```

Mean velocity of robot points. Returns `(vx_mean, vy_mean)`.

Uses `velocities_true` (finite-difference velocity from `positions - positions_last`), not the integrated velocity.

## get_relative_pos_obs

```python
def get_relative_pos_obs(
    positions: jnp.ndarray,  # (n_points, 2)
    robot_point_indices: jnp.ndarray,  # (n_robot_points,)
) -> jnp.ndarray:  # (2 * n_robot_points,)
```

Relative positions of robot points to their center of mass.

Returns `[x0, x1, ..., xn, y0, y1, ..., yn]` — the transposed-then-flattened layout matching C++ row-major `(2, n)` format.

## compute_orientation

```python
def compute_orientation(
    current_pos: jnp.ndarray,  # (n_robot_points, 2)
    initial_pos: jnp.ndarray,  # (n_robot_points, 2)
) -> jnp.ndarray:  # scalar in [0, 2*pi]
```

Weighted angular displacement from initial configuration.

Algorithm (matching C++ `Environment.cpp:317-380`):
1. Center both configurations around their COMs
2. Normalize each point vector
3. Per-point angle via `acos(dot(current_norm, initial_norm))`, weighted by initial magnitude
4. Average = `sum(weighted_angles) / sum(magnitudes)`
5. Cross product sum determines CW/CCW: if `cross > 0`, `angle = 2*pi - angle`

Used by `BridgeWalkerV0` and `UpStepperV0` for orientation observation and flip detection.

## get_floor_obs

```python
def get_floor_obs(
    com_pos: jnp.ndarray,  # (2,) robot COM in physical coords
    terrain_positions: jnp.ndarray,  # (max_terrain_points, 2)
    terrain_mask: jnp.ndarray,  # (max_terrain_points,) bool
    sight_dist: int,
    sight_range: float = 5.0,
) -> jnp.ndarray:  # (2 * sight_dist + 1,)
```

Terrain elevation observations below the robot COM.

For each column offset in `[-sight_dist, ..., +sight_dist]`:
- Finds the maximum terrain y-coordinate below the COM in a 0.1-wide column
- Computes `elevation = com_y - max_terrain_y`
- Clamps to `[0, sight_range * 0.1]`

Returns elevations in physical units (scaled by `CELL_SIZE = 0.1`).

Used by `JumperV0` (`sight_dist=2`, 5 columns) and `UpStepperV0` (`sight_dist=5`, 11 columns).

## get_deformation_obs

```python
def get_deformation_obs(
    positions: jnp.ndarray,  # (n_points, 2)
    topology: SpringTopology,
    spring_init_rest_length: jnp.ndarray,  # (n_springs,)
    deformation_info: DeformationInfo,
    spring_rest_length: jnp.ndarray | None = None,
    reference: str = "init",
) -> jnp.ndarray:  # (2 * n_robot_voxels,)
```

Per-voxel spring strain as proprioceptive observation.

Returns `[deform_x_0, deform_y_0, ..., deform_x_n, deform_y_n]`. With the
default `reference="init"`, each value is the ratio of current spring length to
initial rest length:
- `1.0` = no deformation
- `> 1.0` = extension
- `< 1.0` = compression

Use `reference="current"` with `spring_rest_length` to measure against the
current actuated rest lengths instead. `spring_rest_length` is required in that
mode. Other `reference` values raise `ValueError`.

Maps each robot voxel to its horizontal and vertical edge springs via `DeformationInfo`. Missing springs (sentinel `-1`) are padded with `1.0`.

Enabled by default in all environments via `include_deformation=True`.
