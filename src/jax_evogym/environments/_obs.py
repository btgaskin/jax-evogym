"""Observation helper functions — pure JAX, JIT-compatible.

All obs functions take JAX arrays and return JAX arrays.
Designed to match C++ EvoGymBase/BenchmarkBase observation methods.

BenchmarkBase applies *0.1 to all position/velocity outputs, but EvoSim
internally divides by 0.1 — net effect is no scaling. Our JAX positions
are already physical, so no extra scaling is needed.
"""

import jax
import jax.numpy as jnp

from ..deformation import compute_deformation
from ..types import DeformationInfo, SpringTopology


def get_vel_com_obs(
    velocities_true: jnp.ndarray,
    robot_point_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Mean velocity of robot points → (2,).

    Matches C++ EvoGymBase.get_vel_com_obs: mean of object velocities.
    """
    active_vel = velocities_true[robot_point_indices]
    return jnp.mean(active_vel, axis=0)


def get_relative_pos_obs(
    positions: jnp.ndarray,
    robot_point_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Relative positions of robot points → (2*n_robot_points,).

    Returns (pos - com).T.flatten() giving [x0..xn, y0..yn] order,
    matching C++ (2, n) row-major layout.
    """
    active_pos = positions[robot_point_indices]
    com = jnp.mean(active_pos, axis=0)
    rel = (active_pos - com).T.flatten()
    return rel


def compute_orientation(
    current_pos: jnp.ndarray,
    initial_pos: jnp.ndarray,
) -> jnp.ndarray:
    """Weighted angular displacement from initial config → scalar in [0, 2π].

    Matches C++ Environment.cpp:317-380 orientation algorithm:
    1. Center current + initial positions around their COMs
    2. Normalize each point vector
    3. Per-point angle = acos(dot(current_norm, initial_norm)),
       weighted by initial magnitude
    4. average = sum(weighted_angles) / sum(magnitudes)
    5. Cross product sum determines CW/CCW:
       if cross > 0, angle = 2π - angle

    Args:
        current_pos: (n, 2) current robot point positions
        initial_pos: (n, 2) initial robot point positions (from reset)

    Returns:
        Scalar float in [0, 2π]
    """
    # Center around COMs
    com_curr = jnp.mean(current_pos, axis=0)
    com_init = jnp.mean(initial_pos, axis=0)
    curr_centered = current_pos - com_curr
    init_centered = initial_pos - com_init

    # Magnitudes
    curr_mag = jnp.sqrt(jnp.sum(curr_centered ** 2, axis=-1))  # (n,)
    init_mag = jnp.sqrt(jnp.sum(init_centered ** 2, axis=-1))  # (n,)

    # Normalize (avoid division by zero)
    eps = 1e-10
    curr_norm = curr_centered / jnp.maximum(curr_mag[:, None], eps)
    init_norm = init_centered / jnp.maximum(init_mag[:, None], eps)

    # Per-point angle via acos(dot)
    dots = jnp.sum(curr_norm * init_norm, axis=-1)
    dots = jnp.clip(dots, -1.0, 1.0)
    angles = jnp.arccos(dots)  # (n,)

    # Weighted average by initial magnitude
    weighted = angles * init_mag
    total_mag = jnp.sum(init_mag)
    avg_angle = jnp.where(total_mag > eps, jnp.sum(weighted) / total_mag, 0.0)

    # Cross product sum to determine CW/CCW
    cross_sum = jnp.sum(
        curr_norm[:, 0] * init_norm[:, 1] - curr_norm[:, 1] * init_norm[:, 0]
    )
    avg_angle = jnp.where(cross_sum > 0, 2.0 * jnp.pi - avg_angle, avg_angle)

    return avg_angle


def get_floor_obs(
    com_pos: jnp.ndarray,
    terrain_positions: jnp.ndarray,
    terrain_mask: jnp.ndarray,
    sight_dist: int,
    sight_range: float = 5.0,
) -> jnp.ndarray:
    """Terrain elevation observations below robot COM → (2*sight_dist+1,).

    Matches C++ EvoGymBase.get_floor_obs (base.py:276-327):
    - Works in physical coordinates (our positions are already physical)
    - For each column offset in [-sight_dist, ..., +sight_dist]:
      Find max terrain y below COM in a 0.1-wide column
    - elevation = com_y - max_terrain_y, clamped to [0, sight_range * 0.1]
    - BenchmarkBase multiplies by 0.1, but EvoSim divides by 0.1 first,
      so the net output is in grid units * 0.1 = physical units

    Note: C++ EvoGymBase works in grid units internally (divides by VOXEL_SIZE),
    then BenchmarkBase multiplies back. The column width is 1 grid unit = 0.1 physical.
    sight_range is in grid units, output is physical (scaled by 0.1 in BenchmarkBase).

    Args:
        com_pos: (2,) robot center of mass position in physical coords
        terrain_positions: (max_terrain_points, 2) terrain point positions
        terrain_mask: (max_terrain_points,) bool mask for valid terrain points
        sight_dist: number of columns left and right of COM
        sight_range: max distance in grid units (default 5)

    Returns:
        (2*sight_dist+1,) array of elevation observations (physical units)
    """
    cell_size = 0.1
    n_cols = 2 * sight_dist + 1

    # Convert COM to grid units for column computation
    com_x_grid = com_pos[0] / cell_size
    com_y_grid = com_pos[1] / cell_size

    # Terrain positions in grid units
    terrain_x_grid = terrain_positions[:, 0] / cell_size
    terrain_y_grid = terrain_positions[:, 1] / cell_size

    # Filter terrain: within x-range and below COM
    x_range_mask = (
        (terrain_x_grid > (com_x_grid - (sight_dist + 0.5))) &
        (terrain_x_grid < (com_x_grid + (sight_dist + 0.5)))
    )
    below_mask = terrain_y_grid < com_y_grid
    valid = terrain_mask & x_range_mask & below_mask

    def column_elevation(i):
        """Max terrain y in column centered at com_x_grid + i."""
        col_center = com_x_grid + i
        col_mask = (
            (terrain_x_grid > (col_center - 0.5)) &
            (terrain_x_grid < (col_center + 0.5))
        )
        in_column = valid & col_mask
        # Default: com_y_grid - sight_range (no terrain visible)
        default_y = com_y_grid - sight_range
        max_y = jnp.where(
            jnp.any(in_column),
            jnp.max(jnp.where(in_column, terrain_y_grid, -1e10)),
            default_y,
        )
        return max_y

    # Compute per-column max elevation
    offsets = jnp.arange(-sight_dist, sight_dist + 1, dtype=jnp.float32)
    max_ys = jax.vmap(column_elevation)(offsets)

    # Elevation = distance from COM to terrain, in grid units
    elevations = com_y_grid - max_ys
    elevations = jnp.clip(elevations, 0.0, sight_range)

    # Convert to physical units (BenchmarkBase * 0.1)
    return elevations * cell_size


def get_deformation_obs(
    positions: jnp.ndarray,
    topology: SpringTopology,
    spring_init_rest_length: jnp.ndarray,
    deformation_info: DeformationInfo,
    spring_rest_length: jnp.ndarray | None = None,
    reference: str = "init",
) -> jnp.ndarray:
    """Deformation obs → (2 * n_robot_voxels,) flat vector.

    Returns [deform_x_0, deform_y_0, ..., deform_x_n, deform_y_n].
    Values: 1.0 = no deformation, >1 = extension, <1 = compression.
    """
    deform = compute_deformation(
        positions,
        topology,
        spring_init_rest_length,
        deformation_info,
        spring_rest_length=spring_rest_length,
        reference=reference,
    )
    return deform.flatten()
