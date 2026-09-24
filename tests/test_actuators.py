"""Tests for jax_evogym.actuators — goal setting and convergence."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_evogym.types import (
    ActuatorInfo,
    PerAxisActuatorInfo,
    SimState,
    SpringTopology,
    default_physics_constants,
)
from jax_evogym.actuators import (
    set_actuator_goals,
    set_per_axis_goals,
    set_per_axis_goals_compact,
    update_actuators,
)
from jax_evogym import constants as C


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_actuator_system(n_springs=4, cell_spring_pairs=((1, 3),)):
    """System with per-cell actuator mapping. All springs start at rest_length=0.1.

    cell_spring_pairs: tuple of (spring_a, spring_b) per actuator cell.
    """
    n_points = 5  # arbitrary, just need enough
    n_cells = len(cell_spring_pairs)
    positions = jnp.array([
        [0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0],
    ], dtype=jnp.float32)
    rest_len = jnp.ones(n_springs, dtype=jnp.float32) * 0.1

    state = SimState(
        positions=positions,
        velocities=jnp.zeros((n_points, 2), dtype=jnp.float32),
        positions_last=positions,
        velocities_true=jnp.zeros((n_points, 2), dtype=jnp.float32),
        masses=jnp.ones((n_points, 2), dtype=jnp.float32),
        fixed=jnp.zeros((n_points, 2), dtype=bool),
        point_mask=jnp.ones(n_points, dtype=bool),
        spring_rest_length=rest_len,
        spring_rest_length_goal=rest_len,
        spring_init_rest_length=rest_len,
        spring_const=jnp.ones(n_springs, dtype=jnp.float32) * C.ACTUATOR_MAIN_K,
        spring_mask=jnp.ones(n_springs, dtype=bool),
        rigid_spring_mask=jnp.zeros(n_springs, dtype=bool),
        external_forces=jnp.zeros((n_points, 2), dtype=jnp.float32),
        tangential_deformation=jnp.zeros((n_points, 2), dtype=jnp.float32),
        friction_anchor=positions,
        friction_anchor_active=jnp.zeros((n_points,), dtype=bool),
        friction_anchor_no_contact_count=jnp.zeros((n_points,), dtype=jnp.int32),
        static_manifold_cell_idx=jnp.full((n_points,), -1, dtype=jnp.int32),
        static_manifold_edge_idx=jnp.full((n_points,), -1, dtype=jnp.int32),
        static_manifold_active=jnp.zeros((n_points,), dtype=bool),
        static_manifold_no_contact_count=jnp.zeros((n_points,), dtype=jnp.int32),
        static_manifold_normal_force=jnp.zeros((n_points,), dtype=jnp.float32),
        static_manifold_normal=jnp.zeros((n_points, 2), dtype=jnp.float32),
    )
    topology = SpringTopology(
        a_idx=jnp.array([0, 1, 2, 3], dtype=jnp.int32)[:n_springs],
        b_idx=jnp.array([1, 2, 3, 4], dtype=jnp.int32)[:n_springs],
    )

    # Build ActuatorInfo
    cell_indices = jnp.array(list(cell_spring_pairs), dtype=jnp.int32)
    spring_act_count = jnp.zeros(n_springs, dtype=jnp.float32)
    for pair in cell_spring_pairs:
        for idx in pair:
            spring_act_count = spring_act_count.at[idx].add(1.0)
    actuated_mask = spring_act_count > 0

    actuator_info = ActuatorInfo(
        cell_spring_indices=cell_indices,
        spring_act_count=spring_act_count,
        actuated_spring_mask=actuated_mask,
    )
    return state, topology, actuator_info


def _make_per_axis_system():
    """Tiny per-axis system with 2 cells over 6 springs.

    Cell 0:
      h -> (0, 1), v -> (2, 3)
    Cell 1:
      h -> (1, 4), v -> (3, 5)
    Shared springs: 1 (horizontal), 3 (vertical).
    """
    n_points = 7
    n_springs = 7
    positions = jnp.array(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0], [0.5, 0.0], [0.6, 0.0]],
        dtype=jnp.float32,
    )
    rest_len = jnp.ones(n_springs, dtype=jnp.float32) * 0.1
    state = SimState(
        positions=positions,
        velocities=jnp.zeros((n_points, 2), dtype=jnp.float32),
        positions_last=positions,
        velocities_true=jnp.zeros((n_points, 2), dtype=jnp.float32),
        masses=jnp.ones((n_points, 2), dtype=jnp.float32),
        fixed=jnp.zeros((n_points, 2), dtype=bool),
        point_mask=jnp.ones(n_points, dtype=bool),
        spring_rest_length=rest_len,
        spring_rest_length_goal=rest_len,
        spring_init_rest_length=rest_len,
        spring_const=jnp.ones(n_springs, dtype=jnp.float32) * C.ACTUATOR_MAIN_K,
        spring_mask=jnp.ones(n_springs, dtype=bool),
        rigid_spring_mask=jnp.zeros(n_springs, dtype=bool),
        external_forces=jnp.zeros((n_points, 2), dtype=jnp.float32),
        tangential_deformation=jnp.zeros((n_points, 2), dtype=jnp.float32),
        friction_anchor=positions,
        friction_anchor_active=jnp.zeros((n_points,), dtype=bool),
        friction_anchor_no_contact_count=jnp.zeros((n_points,), dtype=jnp.int32),
        static_manifold_cell_idx=jnp.full((n_points,), -1, dtype=jnp.int32),
        static_manifold_edge_idx=jnp.full((n_points,), -1, dtype=jnp.int32),
        static_manifold_active=jnp.zeros((n_points,), dtype=bool),
        static_manifold_no_contact_count=jnp.zeros((n_points,), dtype=jnp.int32),
        static_manifold_normal_force=jnp.zeros((n_points,), dtype=jnp.float32),
        static_manifold_normal=jnp.zeros((n_points, 2), dtype=jnp.float32),
    )

    h_pairs = jnp.array([[0, 1], [1, 4]], dtype=jnp.int32)
    v_pairs = jnp.array([[2, 3], [3, 5]], dtype=jnp.int32)
    spring_act_count = jnp.array([1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
    actuated_spring_mask = spring_act_count > 0
    info = PerAxisActuatorInfo(
        h_spring_pairs=h_pairs,
        v_spring_pairs=v_pairs,
        h_compact_cell_indices=jnp.array([0, 1], dtype=jnp.int32),
        v_compact_cell_indices=jnp.array([0, 1], dtype=jnp.int32),
        h_compact_spring_pairs=h_pairs,
        v_compact_spring_pairs=v_pairs,
        h_compact_count=jnp.int32(2),
        v_compact_count=jnp.int32(2),
        actuated_spring_mask=actuated_spring_mask,
        spring_act_count=spring_act_count,
    )
    return state, info


# ---------------------------------------------------------------------------
# Goal setting tests
# ---------------------------------------------------------------------------

class TestSetActuatorGoals:
    """Test set_actuator_goals."""

    def test_goals_set_correctly(self):
        """action=1.2 for one cell → goal = init_length * 1.2 for both springs."""
        state, topo, act_info = _make_actuator_system()
        action = jnp.array([1.2], dtype=jnp.float32)  # 1 cell
        new_state = set_actuator_goals(state, act_info, action)

        # Actuator springs (idx 1, 3) should have goal = 0.1 * 1.2 = 0.12
        np.testing.assert_allclose(float(new_state.spring_rest_length_goal[1]), 0.12, rtol=1e-5)
        np.testing.assert_allclose(float(new_state.spring_rest_length_goal[3]), 0.12, rtol=1e-5)

    def test_non_actuator_goals_unchanged(self):
        """Non-actuator spring goals remain at init_rest_length."""
        state, topo, act_info = _make_actuator_system()
        action = jnp.array([1.5], dtype=jnp.float32)
        new_state = set_actuator_goals(state, act_info, action)

        # Non-actuator springs (idx 0, 2) should be unchanged
        np.testing.assert_allclose(float(new_state.spring_rest_length_goal[0]), 0.1, rtol=1e-6)
        np.testing.assert_allclose(float(new_state.spring_rest_length_goal[2]), 0.1, rtol=1e-6)

    def test_different_actions_per_cell(self):
        """Different action values for different cells produce different goals."""
        # Two cells: cell 0 drives springs (0, 1), cell 1 drives springs (2, 3)
        state, topo, act_info = _make_actuator_system(
            cell_spring_pairs=((0, 1), (2, 3))
        )
        action = jnp.array([1.5, 0.7], dtype=jnp.float32)
        new_state = set_actuator_goals(state, act_info, action)

        # Cell 0: springs 0, 1 → goal = 0.1 * 1.5 = 0.15
        np.testing.assert_allclose(float(new_state.spring_rest_length_goal[0]), 0.15, rtol=1e-5)
        np.testing.assert_allclose(float(new_state.spring_rest_length_goal[1]), 0.15, rtol=1e-5)
        # Cell 1: springs 2, 3 → goal = 0.1 * 0.7 = 0.07
        np.testing.assert_allclose(float(new_state.spring_rest_length_goal[2]), 0.07, rtol=1e-5)
        np.testing.assert_allclose(float(new_state.spring_rest_length_goal[3]), 0.07, rtol=1e-5)

    def test_shared_edge_averaging(self):
        """Two cells sharing a spring get averaged action on that spring.

        Cell 0 drives springs (0, 1), cell 1 drives springs (1, 2).
        Spring 1 is shared → count=2, gets average of both actions.
        """
        state, topo, act_info = _make_actuator_system(
            n_springs=4, cell_spring_pairs=((0, 1), (1, 2))
        )
        action = jnp.array([1.4, 0.8], dtype=jnp.float32)
        new_state = set_actuator_goals(state, act_info, action)

        # Spring 0: only cell 0 → goal = 0.1 * 1.4 = 0.14
        np.testing.assert_allclose(float(new_state.spring_rest_length_goal[0]), 0.14, rtol=1e-5)
        # Spring 1: shared, average of 1.4 and 0.8 = 1.1 → goal = 0.1 * 1.1 = 0.11
        np.testing.assert_allclose(float(new_state.spring_rest_length_goal[1]), 0.11, rtol=1e-5)
        # Spring 2: only cell 1 → goal = 0.1 * 0.8 = 0.08
        np.testing.assert_allclose(float(new_state.spring_rest_length_goal[2]), 0.08, rtol=1e-5)
        # Spring 3: not actuated → unchanged at 0.1
        np.testing.assert_allclose(float(new_state.spring_rest_length_goal[3]), 0.1, rtol=1e-6)


# ---------------------------------------------------------------------------
# Convergence tests
# ---------------------------------------------------------------------------

class TestUpdateActuators:
    """Test update_actuators convergence."""

    def test_convergence_factor(self):
        """Verify exact formula: current_dist + (goal - current_dist) * 0.006."""
        state, topo, act_info = _make_actuator_system()
        constants = default_physics_constants()
        # Set a goal different from current distance
        new_goal = state.spring_rest_length_goal.at[1].set(0.2)
        state = state._replace(spring_rest_length_goal=new_goal)

        # Current distance for spring 1: points 1 and 2 are at (0.1,0) and (0.2,0) → dist=0.1
        current_dist = 0.1
        expected_new = current_dist + (0.2 - current_dist) * C.ACTUATOR_CONVERGENCE

        new_state = update_actuators(state, topo, constants)
        np.testing.assert_allclose(
            float(new_state.spring_rest_length[1]), expected_new, rtol=1e-5
        )

    def test_goal_greater_than_dist_increases_rest_length(self):
        """goal > current_dist → rest_length increases but doesn't overshoot."""
        state, topo, act_info = _make_actuator_system()
        constants = default_physics_constants()
        new_goal = state.spring_rest_length_goal.at[0].set(0.5)
        state = state._replace(spring_rest_length_goal=new_goal)

        new_state = update_actuators(state, topo, constants)
        # Rest length should increase but not jump to 0.5
        assert float(new_state.spring_rest_length[0]) > float(state.spring_rest_length[0])
        assert float(new_state.spring_rest_length[0]) < 0.5

    def test_all_springs_updated(self):
        """All springs (not just actuator springs) get updated."""
        state, topo, act_info = _make_actuator_system()
        constants = default_physics_constants()
        new_state = update_actuators(state, topo, constants)
        assert new_state.spring_rest_length.shape == state.spring_rest_length.shape


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------

class TestJIT:
    """Test JIT compatibility."""

    def test_jit_set_actuator_goals(self):
        """jax.jit(set_actuator_goals) runs without error."""
        state, topo, act_info = _make_actuator_system()
        action = jnp.array([1.0], dtype=jnp.float32)  # 1 cell
        jit_fn = jax.jit(set_actuator_goals)
        new_state = jit_fn(state, act_info, action)
        assert new_state.spring_rest_length_goal.shape == state.spring_rest_length_goal.shape

    def test_jit_update_actuators(self):
        """jax.jit(update_actuators) runs without error."""
        state, topo, act_info = _make_actuator_system()
        constants = default_physics_constants()
        jit_fn = jax.jit(update_actuators)
        new_state = jit_fn(state, topo, constants)
        assert new_state.spring_rest_length.shape == state.spring_rest_length.shape


class TestSetPerAxisGoals:
    def test_hv_isolation_and_averaging(self):
        """Horizontal and vertical actions are isolated and shared edges are averaged."""
        state, info = _make_per_axis_system()

        h_action = jnp.array([1.4, 0.8], dtype=jnp.float32)
        v_action = jnp.array([1.2, 1.6], dtype=jnp.float32)
        new_state = set_per_axis_goals(state, info, h_action, v_action)

        # Horizontal only
        np.testing.assert_allclose(float(new_state.spring_rest_length_goal[0]), 0.14, rtol=1e-5)
        np.testing.assert_allclose(float(new_state.spring_rest_length_goal[1]), 0.11, rtol=1e-5)
        np.testing.assert_allclose(float(new_state.spring_rest_length_goal[4]), 0.08, rtol=1e-5)

        # Vertical only
        np.testing.assert_allclose(float(new_state.spring_rest_length_goal[2]), 0.12, rtol=1e-5)
        np.testing.assert_allclose(float(new_state.spring_rest_length_goal[3]), 0.14, rtol=1e-5)
        np.testing.assert_allclose(float(new_state.spring_rest_length_goal[5]), 0.16, rtol=1e-5)

    def test_non_actuated_springs_unchanged(self):
        """Springs not in actuated mask should keep previous goals."""
        state, info = _make_per_axis_system()
        new_state = set_per_axis_goals(
            state,
            info,
            jnp.array([1.0, 1.0], dtype=jnp.float32),
            jnp.array([1.0, 2.0], dtype=jnp.float32),
        )
        np.testing.assert_allclose(float(new_state.spring_rest_length_goal[6]), 0.1, rtol=1e-6)

    def test_jit_set_per_axis_goals(self):
        """jax.jit(set_per_axis_goals) runs without retrace-sensitive shape issues."""
        state, info = _make_per_axis_system()
        jit_fn = jax.jit(set_per_axis_goals)
        out = jit_fn(
            state,
            info,
            jnp.array([1.2, 0.9], dtype=jnp.float32),
            jnp.array([0.8, 1.1], dtype=jnp.float32),
        )
        assert out.spring_rest_length_goal.shape == state.spring_rest_length_goal.shape

    def test_compact_matches_dense_path(self):
        state, info = _make_per_axis_system()
        h_action = jnp.array([1.2, 0.9], dtype=jnp.float32)
        v_action = jnp.array([0.8, 1.1], dtype=jnp.float32)
        dense_state = set_per_axis_goals(state, info, h_action, v_action)
        compact_state = set_per_axis_goals_compact(state, info, h_action, v_action)
        np.testing.assert_allclose(
            np.asarray(compact_state.spring_rest_length_goal),
            np.asarray(dense_state.spring_rest_length_goal),
            atol=1e-6,
        )

    def test_jit_set_per_axis_goals_compact(self):
        state, info = _make_per_axis_system()
        jit_fn = jax.jit(set_per_axis_goals_compact)
        out = jit_fn(
            state,
            info,
            jnp.array([1.2, 0.9], dtype=jnp.float32),
            jnp.array([0.8, 1.1], dtype=jnp.float32),
        )
        assert out.spring_rest_length_goal.shape == state.spring_rest_length_goal.shape


def test_dense_per_axis_ignores_inactive_cell_axes():
    """Inactive axes use dummy pairs and must not scatter into spring zero."""
    from jax_evogym import EvoWorld, H_ACT, V_ACT, SOFT, CONTRACTILE, compile_world_template, instantiate_world

    world = EvoWorld()
    world.add_from_array('robot', np.array([[SOFT, H_ACT, V_ACT, CONTRACTILE]]), 0, 1)
    built = instantiate_world(compile_world_template(world, robot_name='robot'))
    info = built.per_axis_info
    h = jnp.array([1.1, 1.2, 1.3, 1.4])
    v = jnp.array([0.9, 0.8, 0.7, 0.6])
    compact = set_per_axis_goals_compact(
        built.sim_state, info, h[info.h_compact_cell_indices], v[info.v_compact_cell_indices],
    )
    for setter in (set_per_axis_goals, jax.jit(set_per_axis_goals)):
        dense = setter(built.sim_state, info, h, v)
        np.testing.assert_allclose(dense.spring_rest_length_goal, compact.spring_rest_length_goal, atol=1e-7)
        inactive = ~np.asarray(info.actuated_spring_mask)
        np.testing.assert_array_equal(dense.spring_rest_length_goal[inactive], built.sim_state.spring_rest_length_goal[inactive])


def test_dense_scalar_padding_does_not_actuate_passive_springs():
    from jax_evogym import SOFT, H_ACT, V_ACT, precompute_grid, jax_build_sim_state, jax_build_actuator_info

    body = jnp.array([[SOFT, H_ACT, V_ACT]], dtype=jnp.int32)
    grid = precompute_grid(H=1, W=3)
    state, _ = jax_build_sim_state(body, grid)
    info, mask = jax_build_actuator_info(body, grid)
    actions = jnp.array([1.6, 1.2, 0.8])
    expected = set_actuator_goals(state, info, jnp.where(mask, actions, 0.))
    for setter in (set_actuator_goals, jax.jit(set_actuator_goals)):
        actual = setter(state, info, actions)
        np.testing.assert_allclose(actual.spring_rest_length_goal, expected.spring_rest_length_goal, atol=1e-7)
        inactive = ~np.asarray(info.actuated_spring_mask)
        np.testing.assert_array_equal(actual.spring_rest_length_goal[inactive], state.spring_rest_length_goal[inactive])
