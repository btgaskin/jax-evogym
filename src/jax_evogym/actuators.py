"""Actuator goal-setting and convergence.

Matches C++ PhysicsEngine.cpp:38-97 (update_actuator_goals)
and PhysicsEngine.cpp:99-119 (update_actuators).
"""

import jax
import jax.numpy as jnp

from jax_evogym.types import (
    ActuatorInfo,
    PhysicsConstants,
    PerAxisActuatorInfo,
    SimState,
    SpringTopology,
)


def _deterministic_sum_by_index(
    indices: jnp.ndarray,
    values: jnp.ndarray,
    *,
    size: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Deterministic index accumulation via dense one-hot reduction."""
    weights = jax.nn.one_hot(indices, size, dtype=dtype)
    return weights.T @ values


def set_actuator_goals(
    state: SimState,
    actuator_info: ActuatorInfo,
    action: jnp.ndarray,
) -> SimState:
    """Set actuator spring rest length goals from per-cell action values.

    Called ONCE per gym step, before the 30-substep loop.
    Matches C++ PhysicsEngine.cpp:38-97 three-pass scatter-add-average.

    action: (n_actuator_cells,) — one action per actuator cell.
    Each cell drives 2 springs. Shared edges get averaged actions.
    goal[spring] = sum_of_actions / count * init_rest_length.
    """
    indices = actuator_info.cell_spring_indices          # (n_cells, 2)
    flat_indices = indices.flatten()                      # (n_cells*2,)
    flat_actions = jnp.repeat(action, 2)                 # (n_cells*2,)

    # Zero actuated spring goals, scatter-add actions
    goals = jnp.where(
        actuator_info.actuated_spring_mask,
        0.0,
        state.spring_rest_length_goal,
    )
    goals = goals + _deterministic_sum_by_index(
        flat_indices,
        flat_actions,
        size=int(goals.shape[0]),
        dtype=goals.dtype,
    )

    # Average by count, multiply by init rest length
    safe_count = jnp.where(actuator_info.spring_act_count > 0,
                           actuator_info.spring_act_count, 1.0)
    goals = jnp.where(actuator_info.actuated_spring_mask,
                      goals * state.spring_init_rest_length / safe_count,
                      goals)
    return state._replace(spring_rest_length_goal=goals)


def update_actuators(
    state: SimState,
    topology: SpringTopology,
    constants: PhysicsConstants,
) -> SimState:
    """Converge spring rest lengths toward goals. Called every substep.

    Operates on ALL springs (not just actuator springs), matching C++ behavior.
    For non-actuator springs, goal == init_rest_length, so this gently
    restores deformed springs toward their original length.

    Formula:
      new_rest_length = current_dist + (goal - current_dist) * actuator_convergence
    """
    a_idx = topology.a_idx
    b_idx = topology.b_idx

    vec = state.positions[a_idx] - state.positions[b_idx]
    current_dist = jnp.sqrt(jnp.sum(vec ** 2, axis=1))

    additive = (state.spring_rest_length_goal - current_dist) * constants.actuator_convergence
    new_rest_length = current_dist + additive

    return state._replace(spring_rest_length=new_rest_length)


def set_per_axis_goals(
    state: SimState,
    info: PerAxisActuatorInfo,
    h_action: jnp.ndarray,
    v_action: jnp.ndarray,
) -> SimState:
    """Set spring rest-length goals from per-cell horizontal/vertical actions.

    Each cell contributes to horizontal and vertical spring pairs independently.
    Shared edges are averaged with spring_act_count, matching actuator semantics.
    """
    h_pairs = info.h_spring_pairs
    v_pairs = info.v_spring_pairs

    h_indices = h_pairs.flatten()
    v_indices = v_pairs.flatten()
    h_values = jnp.repeat(h_action, 2)
    v_values = jnp.repeat(v_action, 2)

    goals = jnp.where(info.actuated_spring_mask, 0.0, state.spring_rest_length_goal)
    n_springs = int(goals.shape[0])
    goals = goals + _deterministic_sum_by_index(
        h_indices,
        h_values,
        size=n_springs,
        dtype=goals.dtype,
    )
    goals = goals + _deterministic_sum_by_index(
        v_indices,
        v_values,
        size=n_springs,
        dtype=goals.dtype,
    )

    safe_count = jnp.where(info.spring_act_count > 0.0, info.spring_act_count, 1.0)
    goals = jnp.where(
        info.actuated_spring_mask,
        goals * state.spring_init_rest_length / safe_count,
        goals,
    )
    return state._replace(spring_rest_length_goal=goals)


def set_per_axis_goals_compact(
    state: SimState,
    info: PerAxisActuatorInfo,
    h_action_compact: jnp.ndarray,
    v_action_compact: jnp.ndarray,
) -> SimState:
    """Set per-axis goals using compact actuator-indexed action vectors."""

    h_pairs = info.h_compact_spring_pairs
    v_pairs = info.v_compact_spring_pairs

    h_valid = jnp.arange(h_pairs.shape[0], dtype=jnp.int32) < info.h_compact_count
    v_valid = jnp.arange(v_pairs.shape[0], dtype=jnp.int32) < info.v_compact_count

    h_repeat_valid = jnp.repeat(h_valid, 2)
    v_repeat_valid = jnp.repeat(v_valid, 2)
    h_indices = jnp.where(h_repeat_valid, h_pairs.flatten(), 0)
    v_indices = jnp.where(v_repeat_valid, v_pairs.flatten(), 0)
    h_values = jnp.where(h_repeat_valid, jnp.repeat(h_action_compact, 2), 0.0)
    v_values = jnp.where(v_repeat_valid, jnp.repeat(v_action_compact, 2), 0.0)

    goals = jnp.where(info.actuated_spring_mask, 0.0, state.spring_rest_length_goal)
    n_springs = int(goals.shape[0])
    goals = goals + _deterministic_sum_by_index(
        h_indices,
        h_values,
        size=n_springs,
        dtype=goals.dtype,
    )
    goals = goals + _deterministic_sum_by_index(
        v_indices,
        v_values,
        size=n_springs,
        dtype=goals.dtype,
    )

    safe_count = jnp.where(info.spring_act_count > 0.0, info.spring_act_count, 1.0)
    goals = jnp.where(
        info.actuated_spring_mask,
        goals * state.spring_init_rest_length / safe_count,
        goals,
    )
    return state._replace(spring_rest_length_goal=goals)
