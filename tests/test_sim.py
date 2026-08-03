"""Tests for jax_evogym.sim — substep, env_step, and episode step."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_evogym.types import SimState, SpringTopology, PhysicsConstants, ActuatorInfo, default_physics_constants
from jax_evogym.sim import (
    env_step,
    make_episode_step,
    physics_substep,
    physics_substep_dynamic_broadphase,
    physics_substep_dynamic_broadphase_with_probe,
)
from jax_evogym.utils import voxel_to_dense_mass_spring, make_sim_state
from jax_evogym import constants as C


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_soft_voxel_state(spawn_y=2):
    """Create a single SOFT voxel at given height. Returns (state, topology, actuator_info, constants)."""
    morph = np.array([[C.SOFT]])
    dense = voxel_to_dense_mass_spring(morph, H=5, W=5)
    state, topology, actuator_info = make_sim_state(dense, spawn_x=0, spawn_y=spawn_y)
    constants = default_physics_constants()
    return state, topology, actuator_info, constants


def _make_actuated_voxel_state(spawn_y=2):
    """Create a single H_ACT voxel. Returns (state, topology, actuator_info, constants)."""
    morph = np.array([[C.H_ACT]])
    dense = voxel_to_dense_mass_spring(morph, H=5, W=5)
    state, topology, actuator_info = make_sim_state(dense, spawn_x=0, spawn_y=spawn_y)
    constants = default_physics_constants()
    return state, topology, actuator_info, constants


# ---------------------------------------------------------------------------
# Substep tests
# ---------------------------------------------------------------------------

class TestPhysicsSubstep:
    """Test physics_substep."""

    def test_substep_changes_state(self):
        """Positions/velocities should differ after one substep."""
        state, topo, _, consts = _make_soft_voxel_state()
        new_state = physics_substep(state, topo, consts)

        active = state.point_mask
        pos_diff = jnp.sum(jnp.abs(new_state.positions[active] - state.positions[active]))
        assert float(pos_diff) > 0, "Substep should change positions"

    def test_external_forces_cleared(self):
        """External forces should be zeroed after a substep."""
        state, topo, _, consts = _make_soft_voxel_state()
        # Set some external forces
        ext = state.external_forces.at[0].set(jnp.array([100.0, 0.0]))
        state = state._replace(external_forces=ext)

        new_state = physics_substep(state, topo, consts)
        np.testing.assert_allclose(new_state.external_forces, 0.0, atol=1e-10)

    def test_positions_last_updated(self):
        """positions_last should equal current positions after substep."""
        state, topo, _, consts = _make_soft_voxel_state()
        new_state = physics_substep(state, topo, consts)
        np.testing.assert_allclose(new_state.positions_last, new_state.positions, atol=1e-10)

    def test_substep_matches_with_noncanonical_point_mask_storage(self):
        """Non-canonical bool bytes in point_mask should preserve physics semantics."""
        state, topo, _, consts = _make_soft_voxel_state()
        canonical_mask = np.asarray(state.point_mask, dtype=np.bool_)
        raw_mask = np.where(canonical_mask, np.uint8(255), np.uint8(0))
        noncanonical_mask = np.frombuffer(raw_mask.tobytes(), dtype=np.bool_)
        assert np.array_equal(noncanonical_mask, canonical_mask)

        state_noncanonical = state._replace(point_mask=jnp.asarray(noncanonical_mask, dtype=jnp.bool_))

        canonical_next = physics_substep(state, topo, consts)
        noncanonical_next = physics_substep(state_noncanonical, topo, consts)

        np.testing.assert_allclose(
            np.asarray(noncanonical_next.positions),
            np.asarray(canonical_next.positions),
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(noncanonical_next.tangential_deformation),
            np.asarray(canonical_next.tangential_deformation),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_dynamic_broadphase_probe_matches_plain_substep(self):
        """Probe wrapper should preserve dynamic-broadphase substep state updates."""
        state, topo, _, consts = _make_soft_voxel_state()

        plain_next = physics_substep_dynamic_broadphase(state, topo, consts)
        probed_next, probe = physics_substep_dynamic_broadphase_with_probe(
            state,
            topo,
            consts,
        )

        np.testing.assert_allclose(
            np.asarray(probed_next.positions),
            np.asarray(plain_next.positions),
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(probed_next.velocities_true),
            np.asarray(plain_next.velocities_true),
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(probe.final_positions),
            np.asarray(probed_next.positions),
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(probe.final_velocities),
            np.asarray(probed_next.velocities),
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(probe.velocity_true_positions_last),
            np.asarray(state.positions_last),
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(probe.velocity_true_displacement),
            np.asarray(probe.final_positions) - np.asarray(probe.velocity_true_positions_last),
            atol=1e-6,
            rtol=1e-6,
        )


# ---------------------------------------------------------------------------
# Freefall tests
# ---------------------------------------------------------------------------

class TestFreefall:
    """Test freefall behavior."""

    def test_com_falls_monotonically(self):
        """10 env_steps, COM y decreases monotonically."""
        state, topo, act_info, consts = _make_soft_voxel_state(spawn_y=5)
        active = state.point_mask

        # No actuator springs → use empty action
        action = jnp.zeros(act_info.cell_spring_indices.shape[0], dtype=jnp.float32)

        # Need action=1.0 so goals = init_rest * 1.0 (no actuation)
        # But for SOFT voxel there are no actuator springs, so action is empty
        prev_y = float(state.positions[active][:, 1].mean())
        for _ in range(10):
            state = env_step(state, topo, consts, action, act_info)
            curr_y = float(state.positions[active][:, 1].mean())
            assert curr_y < prev_y, "COM should fall monotonically"
            prev_y = curr_y


# ---------------------------------------------------------------------------
# env_step vs manual substeps
# ---------------------------------------------------------------------------

class TestEnvStepConsistency:
    """Test env_step matches 30 manual substeps."""

    def test_env_step_matches_manual(self):
        """env_step output identical to 30 manual physics_substeps."""
        state, topo, act_info, consts = _make_actuated_voxel_state()
        n_act = act_info.cell_spring_indices.shape[0]
        action = jnp.ones(n_act, dtype=jnp.float32)  # neutral action

        # env_step
        from jax_evogym.actuators import set_actuator_goals
        state_env = env_step(state, topo, consts, action, act_info)

        # Manual: set goals once, then 30 substeps
        state_manual = set_actuator_goals(state, act_info, action)
        for _ in range(30):
            state_manual = physics_substep(state_manual, topo, consts)

        np.testing.assert_allclose(
            state_env.positions, state_manual.positions, atol=1e-5
        )
        # lax.scan vs unrolled loop produce different XLA programs with
        # different scatter-add ordering. Position diff is ~1e-8 (float32
        # epsilon). Velocity diff is ~1e-5, concentrated in near-zero
        # x-velocities where symmetric spring forces nearly cancel.
        # Use rtol+atol: large y-velocities (~0.33) checked relatively,
        # near-zero x-velocities (~1e-5) checked absolutely.
        np.testing.assert_allclose(
            state_env.velocities, state_manual.velocities, rtol=1e-4, atol=1e-4
        )


# ---------------------------------------------------------------------------
# Actuated step tests
# ---------------------------------------------------------------------------

class TestActuatedStep:
    """Test actuated env_step."""

    def test_actuator_rest_lengths_changed(self):
        """After env_step with action != 1, actuator spring rest lengths change."""
        state, topo, act_info, consts = _make_actuated_voxel_state()
        n_act = act_info.cell_spring_indices.shape[0]
        action = jnp.ones(n_act, dtype=jnp.float32) * 1.5  # expand

        initial_rest = state.spring_rest_length.copy()
        new_state = env_step(state, topo, consts, action, act_info)

        # Actuator springs should have different rest lengths
        actuated_mask = act_info.actuated_spring_mask
        actuated_init = initial_rest[actuated_mask]
        actuated_new = new_state.spring_rest_length[actuated_mask]
        assert not jnp.allclose(actuated_new, actuated_init), (
            "Actuator spring rest lengths should change"
        )


# ---------------------------------------------------------------------------
# Ground contact tests
# ---------------------------------------------------------------------------

class TestGroundContact:
    """Test ground contact behavior."""

    def test_robot_at_ground_doesnt_fall_through(self):
        """Robot at y=0 doesn't fall through ground."""
        morph = np.array([[C.SOFT]])
        dense = voxel_to_dense_mass_spring(morph, H=5, W=5)
        state, topo, act_info = make_sim_state(dense, spawn_x=0, spawn_y=0)
        consts = default_physics_constants()
        action = jnp.zeros(act_info.cell_spring_indices.shape[0], dtype=jnp.float32)

        # Run several env_steps
        for _ in range(10):
            state = env_step(state, topo, consts, action, act_info)

        # Active points should not be far below ground
        active_y = state.positions[state.point_mask][:, 1]
        min_y = float(jnp.min(active_y))
        assert min_y > -0.1, f"Robot fell through ground: min_y={min_y}"


# ---------------------------------------------------------------------------
# make_episode_step tests
# ---------------------------------------------------------------------------

class TestMakeEpisodeStep:
    """Test make_episode_step with jax.lax.scan."""

    def test_episode_step_scan(self):
        """jax.lax.scan with episode_step produces correct trajectory shape."""
        state, topo, act_info, consts = _make_soft_voxel_state()
        n_act = act_info.cell_spring_indices.shape[0]

        episode_step = make_episode_step(topo, consts, act_info)

        # 5 steps with action sequence
        actions = jnp.zeros((5, n_act), dtype=jnp.float32)
        final_state, trajectory = jax.lax.scan(episode_step, state, actions)

        assert trajectory.shape == (5, state.positions.shape[0], 2)
        assert final_state.positions.shape == state.positions.shape

    def test_episode_step_trajectory_evolves(self):
        """Trajectory positions should change over time."""
        state, topo, act_info, consts = _make_soft_voxel_state(spawn_y=5)
        n_act = act_info.cell_spring_indices.shape[0]

        episode_step = make_episode_step(topo, consts, act_info)
        actions = jnp.zeros((5, n_act), dtype=jnp.float32)
        _, trajectory = jax.lax.scan(episode_step, state, actions)

        # First and last timesteps should differ
        diff = jnp.sum(jnp.abs(trajectory[0] - trajectory[-1]))
        assert float(diff) > 0, "Trajectory should evolve over time"


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------

class TestJIT:
    """Test JIT compatibility."""

    def test_jit_env_step(self):
        """jax.jit(env_step) runs without error."""
        state, topo, act_info, consts = _make_actuated_voxel_state()
        n_act = act_info.cell_spring_indices.shape[0]
        action = jnp.ones(n_act, dtype=jnp.float32)

        jit_fn = jax.jit(env_step, static_argnums=())
        new_state = jit_fn(state, topo, consts, action, act_info)
        assert new_state.positions.shape == state.positions.shape

    def test_jit_physics_substep(self):
        """jax.jit(physics_substep) runs without error."""
        state, topo, _, consts = _make_soft_voxel_state()
        jit_fn = jax.jit(physics_substep)
        new_state = jit_fn(state, topo, consts)
        assert new_state.positions.shape == state.positions.shape
