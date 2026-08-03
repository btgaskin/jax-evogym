"""Tests for jax_evogym.types — pytree compatibility, vmap, scan."""

import jax
import jax.numpy as jnp
import pytest

from jax_evogym.types import (
    ActuatorInfo,
    PhysicsConstants,
    SimState,
    SpringTopology,
    default_physics_constants,
    stiction_physics_constants,
)
from jax_evogym import constants as C


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_state(n_points=4, n_springs=6):
    """Create a minimal SimState for testing."""
    return SimState(
        positions=jnp.zeros((n_points, 2)),
        velocities=jnp.zeros((n_points, 2)),
        positions_last=jnp.zeros((n_points, 2)),
        velocities_true=jnp.zeros((n_points, 2)),
        masses=jnp.ones((n_points, 2)),
        fixed=jnp.zeros((n_points, 2), dtype=bool),
        point_mask=jnp.ones(n_points, dtype=bool),
        spring_rest_length=jnp.ones(n_springs) * 0.1,
        spring_rest_length_goal=jnp.ones(n_springs) * 0.1,
        spring_init_rest_length=jnp.ones(n_springs) * 0.1,
        spring_const=jnp.ones(n_springs) * 1e6,
        spring_mask=jnp.ones(n_springs, dtype=bool),
        rigid_spring_mask=jnp.zeros(n_springs, dtype=bool),
        external_forces=jnp.zeros((n_points, 2)),
        tangential_deformation=jnp.zeros((n_points, 2)),
        friction_anchor=jnp.zeros((n_points, 2)),
        friction_anchor_active=jnp.zeros((n_points,), dtype=bool),
        friction_anchor_no_contact_count=jnp.zeros((n_points,), dtype=jnp.int32),
        static_manifold_cell_idx=jnp.full((n_points,), -1, dtype=jnp.int32),
        static_manifold_edge_idx=jnp.full((n_points,), -1, dtype=jnp.int32),
        static_manifold_active=jnp.zeros((n_points,), dtype=bool),
        static_manifold_no_contact_count=jnp.zeros((n_points,), dtype=jnp.int32),
        static_manifold_normal_force=jnp.zeros((n_points,), dtype=jnp.float32),
        static_manifold_normal=jnp.zeros((n_points, 2), dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPytreeCompatibility:
    """NamedTuples must be valid JAX pytrees."""

    def test_sim_state_tree_map(self):
        state = _make_dummy_state()
        doubled = jax.tree.map(lambda x: x * 2, state)
        assert jnp.allclose(doubled.positions, state.positions * 2)
        assert jnp.allclose(doubled.spring_const, state.spring_const * 2)

    def test_sim_state_tree_leaves(self):
        state = _make_dummy_state()
        leaves = jax.tree.leaves(state)
        assert len(leaves) == len(SimState._fields)

    def test_spring_topology_tree_map(self):
        topo = SpringTopology(
            a_idx=jnp.array([0, 1, 2], dtype=jnp.int32),
            b_idx=jnp.array([1, 2, 3], dtype=jnp.int32),
        )
        shifted = jax.tree.map(lambda x: x + 10, topo)
        assert jnp.array_equal(shifted.a_idx, jnp.array([10, 11, 12]))

    def test_physics_constants_tree_map(self):
        pc = default_physics_constants()
        doubled = jax.tree.map(lambda x: x * 2, pc)
        assert doubled.gravity == pytest.approx(C.GRAVITY * 2)

    def test_actuator_info_tree_map(self):
        ai = ActuatorInfo(
            cell_spring_indices=jnp.array([[0, 5], [2, 7]], dtype=jnp.int32),
            spring_act_count=jnp.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=jnp.float32),
            actuated_spring_mask=jnp.array([True, False, True, False, False, True, False, True]),
        )
        shifted = jax.tree.map(lambda x: x + 1, ai)
        assert jnp.array_equal(shifted.cell_spring_indices, jnp.array([[1, 6], [3, 8]]))


class TestVmap:
    """SimState must work with jax.vmap."""

    def test_vmap_over_sim_state(self):
        batch_size = 4
        state = _make_dummy_state(n_points=4, n_springs=6)
        # Stack into batch
        batched = jax.tree.map(
            lambda x: jnp.stack([x] * batch_size), state
        )

        def compute_kinetic_energy(s: SimState):
            ke = 0.5 * jnp.sum(s.masses * s.velocities ** 2)
            return ke

        energies = jax.vmap(compute_kinetic_energy)(batched)
        assert energies.shape == (batch_size,)
        assert jnp.allclose(energies, 0.0)

    def test_vmap_modifies_positions(self):
        batch_size = 3
        state = _make_dummy_state(n_points=4, n_springs=6)
        batched = jax.tree.map(
            lambda x: jnp.stack([x] * batch_size), state
        )

        def shift_positions(s: SimState, offset: jnp.ndarray):
            return s._replace(positions=s.positions + offset)

        offsets = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        shifted = jax.vmap(shift_positions)(batched, offsets[:, None, :])
        assert shifted.positions.shape == (3, 4, 2)
        assert jnp.allclose(shifted.positions[0, :, 0], 1.0)
        assert jnp.allclose(shifted.positions[1, :, 1], 1.0)


class TestScan:
    """SimState must work with jax.lax.scan as carry."""

    def test_scan_with_sim_state(self):
        state = _make_dummy_state(n_points=4, n_springs=6)
        # Give some velocity
        state = state._replace(
            velocities=jnp.ones((4, 2)) * 0.01,
        )

        def step_fn(carry: SimState, _):
            new_pos = carry.positions + carry.velocities * 0.0001
            new_carry = carry._replace(positions=new_pos)
            return new_carry, new_pos

        n_steps = 10
        final_state, trajectory = jax.lax.scan(
            step_fn, state, None, length=n_steps
        )
        assert trajectory.shape == (n_steps, 4, 2)
        assert final_state.positions.shape == (4, 2)
        # After 10 steps of dt=0.0001, v=0.01: displacement = 10 * 0.01 * 0.0001 = 0.00001
        expected = 10 * 0.01 * 0.0001
        assert jnp.allclose(final_state.positions, expected, atol=1e-8)


class TestDefaultPhysicsConstants:

    def test_values_match_constants(self):
        pc = default_physics_constants()
        assert pc.dt == C.DT
        assert pc.gravity == C.GRAVITY
        assert pc.viscous_drag == C.VISCOUS_DRAG
        assert pc.collision_const_ground == C.COLLISION_CONST_GROUND
        assert pc.collision_const_obj == C.COLLISION_CONST_OBJ
        assert pc.collision_vel_damping == C.COLLISION_VEL_DAMPING
        assert pc.collision_base_dist == C.COLLISION_BASE_DIST
        assert pc.dynamic_friction_const == C.DYNAMIC_FRICTION_CONST
        assert pc.friction_const == C.FRICTION_CONST
        assert pc.ground_friction_const == C.GROUND_FRICTION_CONST
        assert pc.actuator_convergence == C.ACTUATOR_CONVERGENCE
        assert pc.integrator == C.INTEGRATOR_RK4
        assert pc.static_friction_const == 0.0
        assert pc.terrain_stiction_stiffness == 0.0
        assert pc.ground_stiction_stiffness == 0.0
        assert pc.terrain_stiction_switch_speed == 0.0
        assert pc.ground_stiction_switch_speed == 0.0
        assert pc.terrain_stiction_scope == C.STICTION_SCOPE_FLOOR_SLOPE_ONLY
        assert pc.friction_model == C.FRICTION_MODEL_NONE
        assert pc.anchor_stiffness == C.ANCHOR_STIFFNESS
        assert pc.deformation_stiffness == C.DEFORMATION_STIFFNESS
        assert pc.deformation_damping == C.DEFORMATION_DAMPING
        assert pc.deformation_decay == C.DEFORMATION_DECAY
        assert pc.deformation_normal_bleed == C.DEFORMATION_NORMAL_BLEED
        assert pc.breakaway_threshold == C.BREAKAWAY_THRESHOLD

    def test_gravity_not_1100(self):
        """Ensure we're using the correct gravity (110, not MLX's 1100)."""
        pc = default_physics_constants()
        assert pc.gravity == 110.0

    def test_viscous_drag_not_1(self):
        """Ensure we're using the correct drag (0.1, not MLX's 1.0)."""
        pc = default_physics_constants()
        assert pc.viscous_drag == 0.1

    def test_stiction_physics_constants_defaults(self):
        pc = stiction_physics_constants()
        assert pc.static_friction_const == 0.0
        assert pc.terrain_stiction_stiffness == 0.0
        assert pc.ground_stiction_stiffness == 0.0
        assert pc.terrain_stiction_switch_speed == 0.0
        assert pc.ground_stiction_switch_speed == 0.0
        assert pc.terrain_stiction_scope == C.STICTION_SCOPE_FLOOR_SLOPE_ONLY
        assert pc.friction_model == C.FRICTION_MODEL_NONE

    def test_stiction_physics_constants_surface_toggles(self):
        terrain_only = stiction_physics_constants(terrain=True, ground=False)
        assert terrain_only.static_friction_const > 0.0
        assert terrain_only.terrain_stiction_stiffness > 0.0
        assert terrain_only.ground_stiction_stiffness == 0.0
        assert terrain_only.terrain_stiction_switch_speed > 0.0
        assert terrain_only.ground_stiction_switch_speed == 0.0

        ground_only = stiction_physics_constants(terrain=False, ground=True)
        assert ground_only.static_friction_const > 0.0
        assert ground_only.terrain_stiction_stiffness == 0.0
        assert ground_only.ground_stiction_stiffness > 0.0
        assert ground_only.terrain_stiction_switch_speed == 0.0
        assert ground_only.ground_stiction_switch_speed > 0.0

        none_enabled = stiction_physics_constants(terrain=False, ground=False)
        assert none_enabled.static_friction_const == 0.0
        assert none_enabled.terrain_stiction_stiffness == 0.0
        assert none_enabled.ground_stiction_stiffness == 0.0
        assert none_enabled.terrain_stiction_switch_speed == 0.0
        assert none_enabled.ground_stiction_switch_speed == 0.0
