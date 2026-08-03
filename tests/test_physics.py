"""Tests for jax_evogym.physics — force computation and RK4 integration."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_evogym.types import SimState, SpringTopology, PhysicsConstants, default_physics_constants
from jax_evogym.physics import compute_forces, rk4_step, symplectic_euler_step
from jax_evogym import constants as C
from jax_evogym.utils import voxel_to_dense_mass_spring, make_sim_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_two_point_system(rest_length=0.1, spring_const=C.SOFT_MAIN_K,
                           pos_a=(0.0, 0.0), pos_b=(0.2, 0.0)):
    """Two active points connected by one spring. Returns (state, topology, constants)."""
    positions = jnp.array([list(pos_a), list(pos_b)], dtype=jnp.float32)
    velocities = jnp.zeros((2, 2), dtype=jnp.float32)
    masses = jnp.ones((2, 2), dtype=jnp.float32)
    fixed = jnp.zeros((2, 2), dtype=bool)
    point_mask = jnp.ones(2, dtype=bool)
    spring_rest = jnp.array([rest_length], dtype=jnp.float32)
    spring_k = jnp.array([spring_const], dtype=jnp.float32)
    spring_mask = jnp.ones(1, dtype=bool)

    state = SimState(
        positions=positions,
        velocities=velocities,
        positions_last=positions,
        velocities_true=jnp.zeros((2, 2), dtype=jnp.float32),
        masses=masses,
        fixed=fixed,
        point_mask=point_mask,
        spring_rest_length=spring_rest,
        spring_rest_length_goal=spring_rest,
        spring_init_rest_length=spring_rest,
        spring_const=spring_k,
        spring_mask=spring_mask,
        rigid_spring_mask=jnp.zeros(1, dtype=bool),
        external_forces=jnp.zeros((2, 2), dtype=jnp.float32),
        tangential_deformation=jnp.zeros((2, 2), dtype=jnp.float32),
        friction_anchor=positions,
        friction_anchor_active=jnp.zeros((2,), dtype=bool),
        friction_anchor_no_contact_count=jnp.zeros((2,), dtype=jnp.int32),
        static_manifold_cell_idx=jnp.full((2,), -1, dtype=jnp.int32),
        static_manifold_edge_idx=jnp.full((2,), -1, dtype=jnp.int32),
        static_manifold_active=jnp.zeros((2,), dtype=bool),
        static_manifold_no_contact_count=jnp.zeros((2,), dtype=jnp.int32),
        static_manifold_normal_force=jnp.zeros((2,), dtype=jnp.float32),
        static_manifold_normal=jnp.zeros((2, 2), dtype=jnp.float32),
    )
    topology = SpringTopology(
        a_idx=jnp.array([0], dtype=jnp.int32),
        b_idx=jnp.array([1], dtype=jnp.int32),
    )
    constants = default_physics_constants()
    return state, topology, constants


# ---------------------------------------------------------------------------
# Spring force tests
# ---------------------------------------------------------------------------

class TestSpringForces:
    """Test spring force computation."""

    def test_stretched_spring(self):
        """2-point system stretched to 2x → Hooke's law magnitude & direction."""
        state, topo, consts = _make_two_point_system(
            rest_length=0.1, pos_a=(0.0, 0.0), pos_b=(0.2, 0.0)
        )
        # dist=0.2, rest=0.1, so (dist-rest)*k/dist = 0.1*k/0.2 = k/2
        forces = compute_forces(
            state.positions, state.velocities, state.masses, state.point_mask,
            topo, state.spring_rest_length, state.spring_const, state.spring_mask,
            state.external_forces, consts,
        )
        # Point a (left) should be pulled RIGHT (+x), point b (right) pulled LEFT (-x)
        # Force on a = -spring_force (toward b), force on b = +spring_force (toward a)
        # vec_q_to_p = pos[a] - pos[b] = (-0.2, 0), force_mag = positive
        # forces[a] += -force_mag * vec = -force_mag*(-0.2, 0) = (+..., 0) ✓
        # forces[b] += +force_mag * vec = force_mag*(-0.2, 0) = (-..., 0) ✓
        k = C.SOFT_MAIN_K
        # force_mag = (dist-rest)*k/dist = 0.1*k/0.2 = k/2
        # spring_force_vec = vec_q_to_p * force_mag = (-0.2, 0) * k/2
        # force on a = -spring_force_vec = (0.2*k/2, 0) = (k*0.1, 0)
        expected_fx = (0.2 - 0.1) * k / 0.2 * 0.2  # force_mag * |dx| = k*0.1*0.2/0.2 * 0.2...
        # More directly: F_a_x = -(pos_a_x - pos_b_x) * force_mag = 0.2 * (0.1 * k / 0.2) = 0.1 * k
        expected_fx = 0.1 * k  # = 5,000,000

        # x component is pure spring (no drag at zero vel, no gravity in x)
        spring_fx_a = float(forces[0, 0])
        assert spring_fx_a > 0, "Point a should be pulled toward point b (+x)"
        np.testing.assert_allclose(spring_fx_a, expected_fx, rtol=1e-4)

    def test_compressed_spring(self):
        """Spring compressed to 0.5x → force pushes apart."""
        state, topo, consts = _make_two_point_system(
            rest_length=0.1, pos_a=(0.0, 0.0), pos_b=(0.05, 0.0)
        )
        forces = compute_forces(
            state.positions, state.velocities, state.masses, state.point_mask,
            topo, state.spring_rest_length, state.spring_const, state.spring_mask,
            state.external_forces, consts,
        )
        # dist=0.05 < rest=0.1, so force_mag = (0.05-0.1)*k/0.05 < 0
        # Point a should be pushed LEFT (-x), point b pushed RIGHT (+x)
        assert forces[0, 0] < 0, "Point a should be pushed away (-x)"
        assert forces[1, 0] > 0, "Point b should be pushed away (+x)"

    def test_spring_internal_forces_sum_to_zero_without_body_forces(self):
        """A single spring should not create net force on its two-point system."""
        state, topo, consts = _make_two_point_system(
            rest_length=0.1,
            pos_a=(0.0, 1.0),
            pos_b=(0.18, 1.12),
        )
        consts = consts._replace(gravity=0.0, viscous_drag=0.0)

        forces = compute_forces(
            state.positions, state.velocities, state.masses, state.point_mask,
            topo, state.spring_rest_length, state.spring_const, state.spring_mask,
            state.external_forces, consts,
        )

        np.testing.assert_allclose(np.asarray(forces).sum(axis=0), [0.0, 0.0], atol=1e-3)


# ---------------------------------------------------------------------------
# Gravity tests
# ---------------------------------------------------------------------------

class TestGravity:
    """Test gravity force computation."""

    def test_gravity_force(self):
        """No springs, verify F_y = -gravity * mass."""
        positions = jnp.array([[0.5, 0.5]], dtype=jnp.float32)
        velocities = jnp.zeros((1, 2), dtype=jnp.float32)
        masses = jnp.ones((1, 2), dtype=jnp.float32)
        point_mask = jnp.ones(1, dtype=bool)
        topo = SpringTopology(
            a_idx=jnp.zeros(0, dtype=jnp.int32),
            b_idx=jnp.zeros(0, dtype=jnp.int32),
        )
        consts = default_physics_constants()

        forces = compute_forces(
            positions, velocities, masses, point_mask,
            topo, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0, dtype=bool),
            jnp.zeros((1, 2)), consts,
        )
        # Only gravity acts (no springs, no drag, no collision)
        np.testing.assert_allclose(float(forces[0, 1]), -C.GRAVITY * 1.0, rtol=1e-5)
        np.testing.assert_allclose(float(forces[0, 0]), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Viscous drag tests
# ---------------------------------------------------------------------------

class TestViscousDrag:
    """Test viscous drag force computation."""

    def test_drag_opposes_velocity(self):
        """Verify F = -drag * vel."""
        positions = jnp.array([[0.5, 0.5]], dtype=jnp.float32)
        velocities = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
        masses = jnp.ones((1, 2), dtype=jnp.float32)
        point_mask = jnp.ones(1, dtype=bool)
        topo = SpringTopology(
            a_idx=jnp.zeros(0, dtype=jnp.int32),
            b_idx=jnp.zeros(0, dtype=jnp.int32),
        )
        consts = default_physics_constants()

        forces = compute_forces(
            positions, velocities, masses, point_mask,
            topo, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0, dtype=bool),
            jnp.zeros((1, 2)), consts,
        )
        # x: drag only = -0.1 * 1.0
        np.testing.assert_allclose(float(forces[0, 0]), -C.VISCOUS_DRAG * 1.0, rtol=1e-5)
        # y: drag + gravity = -0.1*2.0 + (-110*1.0)
        expected_y = -C.VISCOUS_DRAG * 2.0 - C.GRAVITY * 1.0
        np.testing.assert_allclose(float(forces[0, 1]), expected_y, rtol=1e-5)


# ---------------------------------------------------------------------------
# Ground collision tests
# ---------------------------------------------------------------------------

class TestGroundCollision:
    """Test ground collision and friction forces."""

    def test_below_ground_normal_force(self):
        """Point at y=-0.01 → upward normal force."""
        positions = jnp.array([[0.5, -0.01]], dtype=jnp.float32)
        velocities = jnp.zeros((1, 2), dtype=jnp.float32)
        masses = jnp.ones((1, 2), dtype=jnp.float32)
        point_mask = jnp.ones(1, dtype=bool)
        topo = SpringTopology(
            a_idx=jnp.zeros(0, dtype=jnp.int32),
            b_idx=jnp.zeros(0, dtype=jnp.int32),
        )
        consts = default_physics_constants()

        forces = compute_forces(
            positions, velocities, masses, point_mask,
            topo, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0, dtype=bool),
            jnp.zeros((1, 2)), consts,
        )
        # Normal force = -(-0.01) * 14M = +140000
        expected_normal = 0.01 * C.COLLISION_CONST_GROUND
        # Total y = gravity + normal
        expected_y = -C.GRAVITY * 1.0 + expected_normal
        np.testing.assert_allclose(float(forces[0, 1]), expected_y, rtol=1e-4)

    def test_above_ground_no_collision(self):
        """Point at y=0.01 → zero collision force."""
        positions = jnp.array([[0.5, 0.01]], dtype=jnp.float32)
        velocities = jnp.zeros((1, 2), dtype=jnp.float32)
        masses = jnp.ones((1, 2), dtype=jnp.float32)
        point_mask = jnp.ones(1, dtype=bool)
        topo = SpringTopology(
            a_idx=jnp.zeros(0, dtype=jnp.int32),
            b_idx=jnp.zeros(0, dtype=jnp.int32),
        )
        consts = default_physics_constants()

        forces = compute_forces(
            positions, velocities, masses, point_mask,
            topo, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0, dtype=bool),
            jnp.zeros((1, 2)), consts,
        )
        # Only gravity, no collision
        np.testing.assert_allclose(float(forces[0, 1]), -C.GRAVITY * 1.0, rtol=1e-5)

    def test_ground_friction_opposes_horizontal(self):
        """Friction opposes horizontal velocity when colliding with ground."""
        positions = jnp.array([[0.5, -0.01]], dtype=jnp.float32)
        velocities = jnp.array([[1.0, 0.0]], dtype=jnp.float32)
        masses = jnp.ones((1, 2), dtype=jnp.float32)
        point_mask = jnp.ones(1, dtype=bool)
        topo = SpringTopology(
            a_idx=jnp.zeros(0, dtype=jnp.int32),
            b_idx=jnp.zeros(0, dtype=jnp.int32),
        )
        consts = default_physics_constants()

        forces = compute_forces(
            positions, velocities, masses, point_mask,
            topo, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0, dtype=bool),
            jnp.zeros((1, 2)), consts,
        )
        # Moving right with positive vel_x → friction should push left
        # Also has viscous drag opposing velocity
        # Total x force should be negative (both drag and friction oppose +x motion)
        assert forces[0, 0] < 0, "Friction + drag should oppose rightward velocity"

    def test_ground_stiction_disabled_matches_default(self):
        positions = jnp.array([[0.5, -0.01]], dtype=jnp.float32)
        velocities = jnp.array([[0.05, 0.0]], dtype=jnp.float32)
        masses = jnp.ones((1, 2), dtype=jnp.float32)
        point_mask = jnp.ones(1, dtype=bool)
        topo = SpringTopology(
            a_idx=jnp.zeros(0, dtype=jnp.int32),
            b_idx=jnp.zeros(0, dtype=jnp.int32),
        )
        consts = default_physics_constants()

        forces_default = compute_forces(
            positions, velocities, masses, point_mask,
            topo, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0, dtype=bool),
            jnp.zeros((1, 2)), consts,
        )
        forces_stiction_off = compute_forces(
            positions,
            velocities,
            masses,
            point_mask,
            topo,
            jnp.zeros(0),
            jnp.zeros(0),
            jnp.zeros(0, dtype=bool),
            jnp.zeros((1, 2)),
            consts._replace(
                ground_static_friction_const=0.0,
                ground_stiction_stiffness=20_000.0,
            ),
        )
        np.testing.assert_allclose(
            np.asarray(forces_default),
            np.asarray(forces_stiction_off),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_ground_stiction_increases_low_speed_tangential_resistance(self):
        positions = jnp.array([[0.5, -1e-6]], dtype=jnp.float32)
        velocities = jnp.array([[1e-3, 0.0]], dtype=jnp.float32)
        masses = jnp.ones((1, 2), dtype=jnp.float32)
        point_mask = jnp.ones(1, dtype=bool)
        topo = SpringTopology(
            a_idx=jnp.zeros(0, dtype=jnp.int32),
            b_idx=jnp.zeros(0, dtype=jnp.int32),
        )
        consts = default_physics_constants()
        forces_dynamic = compute_forces(
            positions, velocities, masses, point_mask,
            topo, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0, dtype=bool),
            jnp.zeros((1, 2)), consts,
        )
        forces_stiction = compute_forces(
            positions,
            velocities,
            masses,
            point_mask,
            topo,
            jnp.zeros(0),
            jnp.zeros(0),
            jnp.zeros(0, dtype=bool),
            jnp.zeros((1, 2)),
            consts._replace(
                ground_static_friction_const=0.5,
                ground_stiction_stiffness=10_000.0,
                ground_stiction_switch_speed=0.12,
            ),
        )
        assert abs(float(forces_stiction[0, 0])) > abs(float(forces_dynamic[0, 0])) * 5.0

    def test_ground_stiction_static_friction_is_decoupled_from_terrain_mu(self):
        positions = jnp.array([[0.5, -1e-6]], dtype=jnp.float32)
        velocities = jnp.array([[1e-3, 0.0]], dtype=jnp.float32)
        masses = jnp.ones((1, 2), dtype=jnp.float32)
        point_mask = jnp.ones(1, dtype=bool)
        topo = SpringTopology(
            a_idx=jnp.zeros(0, dtype=jnp.int32),
            b_idx=jnp.zeros(0, dtype=jnp.int32),
        )
        consts = default_physics_constants()

        forces_dynamic = compute_forces(
            positions, velocities, masses, point_mask,
            topo, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0, dtype=bool),
            jnp.zeros((1, 2)), consts,
        )
        forces_terrain_mu_only = compute_forces(
            positions,
            velocities,
            masses,
            point_mask,
            topo,
            jnp.zeros(0),
            jnp.zeros(0),
            jnp.zeros(0, dtype=bool),
            jnp.zeros((1, 2)),
            consts._replace(
                static_friction_const=0.5,
                ground_stiction_stiffness=10_000.0,
                ground_stiction_switch_speed=0.12,
            ),
        )
        forces_ground_mu = compute_forces(
            positions,
            velocities,
            masses,
            point_mask,
            topo,
            jnp.zeros(0),
            jnp.zeros(0),
            jnp.zeros(0, dtype=bool),
            jnp.zeros((1, 2)),
            consts._replace(
                ground_static_friction_const=0.5,
                ground_stiction_stiffness=10_000.0,
                ground_stiction_switch_speed=0.12,
            ),
        )

        np.testing.assert_allclose(
            np.asarray(forces_terrain_mu_only),
            np.asarray(forces_dynamic),
            rtol=1e-6,
            atol=1e-7,
        )
        assert abs(float(forces_ground_mu[0, 0])) > abs(float(forces_dynamic[0, 0])) * 5.0


# ---------------------------------------------------------------------------
# Ghost / mask tests
# ---------------------------------------------------------------------------

class TestGhostMasking:
    """Test ghost point masking."""

    def test_ghost_points_zero_force(self):
        """Zero force when point_mask=False."""
        positions = jnp.array([[0.5, 0.5], [0.5, -0.01]], dtype=jnp.float32)
        velocities = jnp.array([[1.0, 2.0], [1.0, 2.0]], dtype=jnp.float32)
        masses = jnp.ones((2, 2), dtype=jnp.float32)
        point_mask = jnp.array([True, False], dtype=bool)
        topo = SpringTopology(
            a_idx=jnp.zeros(0, dtype=jnp.int32),
            b_idx=jnp.zeros(0, dtype=jnp.int32),
        )
        consts = default_physics_constants()

        forces = compute_forces(
            positions, velocities, masses, point_mask,
            topo, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0, dtype=bool),
            jnp.zeros((2, 2)), consts,
        )
        # Ghost point (index 1) should have zero force
        np.testing.assert_allclose(forces[1], [0.0, 0.0], atol=1e-10)
        # Active point should have nonzero force
        assert jnp.any(forces[0] != 0)


# ---------------------------------------------------------------------------
# RK4 tests
# ---------------------------------------------------------------------------

class TestRK4:
    """Test RK4 integration step."""

    def test_fixed_points_no_change(self):
        """Fixed points have zero position/velocity change after rk4_step."""
        state, topo, consts = _make_two_point_system(
            rest_length=0.1, pos_a=(0.0, 0.5), pos_b=(0.1, 0.5)
        )
        # Make point a fixed
        fixed = jnp.array([[True, True], [False, False]], dtype=bool)
        state = state._replace(fixed=fixed)

        new_state = rk4_step(state, topo, consts)
        np.testing.assert_allclose(new_state.positions[0], state.positions[0], atol=1e-10)
        np.testing.assert_allclose(new_state.velocities[0], state.velocities[0], atol=1e-10)

    def test_freefall_soft_voxel(self):
        """Elevated SOFT voxel, one RK4 step → positions decrease in y."""
        morph = np.array([[C.SOFT]])
        dense = voxel_to_dense_mass_spring(morph, H=3, W=3)
        state, topo, _ = make_sim_state(dense)
        consts = default_physics_constants()

        initial_y = float(state.positions[state.point_mask][:, 1].mean())
        new_state = rk4_step(state, topo, consts)
        new_y = float(new_state.positions[state.point_mask][:, 1].mean())

        assert new_y < initial_y, "COM should fall under gravity"

    def test_rk4_updates_velocity(self):
        """RK4 step should change velocity for a free-falling point."""
        state, topo, consts = _make_two_point_system(
            rest_length=0.1, pos_a=(0.0, 0.5), pos_b=(0.1, 0.5)
        )
        new_state = rk4_step(state, topo, consts)
        # Velocities should have changed (gravity pulls down)
        assert jnp.any(new_state.velocities != state.velocities)

    def test_rk4_force_free_motion_is_linear(self):
        """With no net force, RK4 should preserve velocity and advect positions."""
        state, topo, consts = _make_two_point_system(
            rest_length=0.1,
            pos_a=(0.0, 1.0),
            pos_b=(0.1, 1.0),
        )
        velocities = jnp.array([[0.3, 0.2], [0.3, 0.2]], dtype=jnp.float32)
        state = state._replace(velocities=velocities)
        consts = consts._replace(dt=0.05, gravity=0.0, viscous_drag=0.0)

        new_state = rk4_step(state, topo, consts)

        np.testing.assert_allclose(
            np.asarray(new_state.positions),
            np.asarray(state.positions + velocities * consts.dt),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            np.asarray(new_state.velocities),
            np.asarray(velocities),
            atol=1e-7,
        )


# ---------------------------------------------------------------------------
# Symplectic Euler tests
# ---------------------------------------------------------------------------

class TestSymplecticEuler:
    """Test velocity-first (kick-drift) symplectic Euler integration."""

    def test_fixed_points_no_change(self):
        state, topo, consts = _make_two_point_system(
            rest_length=0.1, pos_a=(0.0, 0.5), pos_b=(0.1, 0.5)
        )
        fixed = jnp.array([[True, True], [False, False]], dtype=bool)
        state = state._replace(fixed=fixed)

        new_state = symplectic_euler_step(state, topo, consts)
        np.testing.assert_allclose(new_state.positions[0], state.positions[0], atol=1e-10)
        np.testing.assert_allclose(new_state.velocities[0], state.velocities[0], atol=1e-10)

    def test_velocity_first_update_order(self):
        positions = jnp.array([[0.0, 1.0]], dtype=jnp.float32)
        velocities = jnp.array([[1.0, 0.0]], dtype=jnp.float32)
        masses = jnp.ones((1, 2), dtype=jnp.float32)
        point_mask = jnp.ones((1,), dtype=bool)
        state = SimState(
            positions=positions,
            velocities=velocities,
            positions_last=positions,
            velocities_true=jnp.zeros((1, 2), dtype=jnp.float32),
            masses=masses,
            fixed=jnp.zeros((1, 2), dtype=bool),
            point_mask=point_mask,
            spring_rest_length=jnp.zeros((0,), dtype=jnp.float32),
            spring_rest_length_goal=jnp.zeros((0,), dtype=jnp.float32),
            spring_init_rest_length=jnp.zeros((0,), dtype=jnp.float32),
            spring_const=jnp.zeros((0,), dtype=jnp.float32),
            spring_mask=jnp.zeros((0,), dtype=bool),
            rigid_spring_mask=jnp.zeros((0,), dtype=bool),
            external_forces=jnp.array([[2.0, 0.0]], dtype=jnp.float32),
            tangential_deformation=jnp.zeros((1, 2), dtype=jnp.float32),
            friction_anchor=positions,
            friction_anchor_active=jnp.zeros((1,), dtype=bool),
            friction_anchor_no_contact_count=jnp.zeros((1,), dtype=jnp.int32),
            static_manifold_cell_idx=jnp.full((1,), -1, dtype=jnp.int32),
            static_manifold_edge_idx=jnp.full((1,), -1, dtype=jnp.int32),
            static_manifold_active=jnp.zeros((1,), dtype=bool),
            static_manifold_no_contact_count=jnp.zeros((1,), dtype=jnp.int32),
            static_manifold_normal_force=jnp.zeros((1,), dtype=jnp.float32),
            static_manifold_normal=jnp.zeros((1, 2), dtype=jnp.float32),
        )
        topo = SpringTopology(
            a_idx=jnp.zeros((0,), dtype=jnp.int32),
            b_idx=jnp.zeros((0,), dtype=jnp.int32),
        )
        consts = default_physics_constants()._replace(
            dt=0.1,
            gravity=0.0,
            viscous_drag=0.0,
        )

        new_state = symplectic_euler_step(state, topo, consts)
        np.testing.assert_allclose(float(new_state.velocities[0, 0]), 1.2, atol=1e-6)
        np.testing.assert_allclose(float(new_state.positions[0, 0]), 0.12, atol=1e-6)


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------

class TestJIT:
    """Test JIT compatibility."""

    def test_jit_compute_forces(self):
        """jax.jit(compute_forces) runs without error."""
        state, topo, consts = _make_two_point_system()
        jit_forces = jax.jit(compute_forces, static_argnums=())
        forces = jit_forces(
            state.positions, state.velocities, state.masses, state.point_mask,
            topo, state.spring_rest_length, state.spring_const, state.spring_mask,
            state.external_forces, consts,
        )
        assert forces.shape == (2, 2)

    def test_jit_rk4_step(self):
        """jax.jit(rk4_step) runs without error."""
        state, topo, consts = _make_two_point_system()
        jit_rk4 = jax.jit(rk4_step)
        new_state = jit_rk4(state, topo, consts)
        assert new_state.positions.shape == state.positions.shape

    def test_jit_symplectic_euler_step(self):
        """jax.jit(symplectic_euler_step) runs without error."""
        state, topo, consts = _make_two_point_system()
        jit_step = jax.jit(symplectic_euler_step)
        new_state = jit_step(state, topo, consts)
        assert new_state.positions.shape == state.positions.shape
