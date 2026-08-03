"""Tests for deformation observation system.

Tests:
- Rest state → all deformations = 1.0
- Known displacement → correct strain ratio
- Sentinel (-1) spring index handling
- JIT compatibility
- vmap compatibility
- Integration: env obs includes deformation, obs_dim updated
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_evogym import constants as C
from jax_evogym.deformation import compute_deformation, compute_dual_deformation
from jax_evogym.types import DeformationInfo, SpringTopology
from jax_evogym.environments.walker import WalkerV0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_env():
    body = np.array([[C.H_ACT, C.SOFT, C.V_ACT]])
    return WalkerV0(body)


@pytest.fixture
def simple_env_no_deform():
    body = np.array([[C.H_ACT, C.SOFT, C.V_ACT]])
    return WalkerV0(body, include_deformation=False)


# ---------------------------------------------------------------------------
# Unit tests: compute_deformation
# ---------------------------------------------------------------------------

class TestRestState:
    """At rest, all springs are at their initial rest length → deformation = 1.0."""

    def test_rest_deformation_is_one(self, simple_env):
        obs, state = simple_env.reset()
        deform = compute_deformation(
            state.sim_state.positions,
            simple_env.topology,
            state.sim_state.spring_init_rest_length,
            simple_env.deformation_info,
        )
        assert deform.shape == (simple_env.n_robot_voxels, 2)
        np.testing.assert_allclose(np.array(deform), 1.0, atol=1e-4)

    def test_rest_deformation_obs_is_one(self, simple_env):
        obs, _ = simple_env.reset()
        n_deform = 2 * simple_env.n_robot_voxels
        deform_obs = np.array(obs[-n_deform:])
        np.testing.assert_allclose(deform_obs, 1.0, atol=1e-4)


class TestKnownDisplacement:
    """Manually create a scenario with known strain."""

    def test_extension_produces_ratio_gt_1(self, simple_env):
        """Moving spring endpoints apart → deformation > 1."""
        obs, state = simple_env.reset()

        # Pull robot points apart by scaling positions
        robot_pos = state.sim_state.positions[simple_env.robot_point_indices]
        com = jnp.mean(robot_pos, axis=0)
        scale = 1.5  # 50% extension
        new_positions = state.sim_state.positions.at[simple_env.robot_point_indices].set(
            com + (robot_pos - com) * scale
        )
        new_sim = state.sim_state._replace(positions=new_positions)

        deform = compute_deformation(
            new_sim.positions,
            simple_env.topology,
            new_sim.spring_init_rest_length,
            simple_env.deformation_info,
        )
        # Extended → ratios should be > 1
        assert jnp.all(deform > 1.0), f"Extension should produce deform > 1, got {deform}"

    def test_compression_produces_ratio_lt_1(self, simple_env):
        """Moving spring endpoints together → deformation < 1."""
        obs, state = simple_env.reset()

        robot_pos = state.sim_state.positions[simple_env.robot_point_indices]
        com = jnp.mean(robot_pos, axis=0)
        scale = 0.5  # 50% compression
        new_positions = state.sim_state.positions.at[simple_env.robot_point_indices].set(
            com + (robot_pos - com) * scale
        )
        new_sim = state.sim_state._replace(positions=new_positions)

        deform = compute_deformation(
            new_sim.positions,
            simple_env.topology,
            new_sim.spring_init_rest_length,
            simple_env.deformation_info,
        )
        # Compressed → ratios should be < 1
        assert jnp.all(deform < 1.0), f"Compression should produce deform < 1, got {deform}"


class TestComplianceMode:
    """Compliance deformation uses current rest length as reference."""

    def test_compliance_rest_state_matches_absolute(self, simple_env):
        _, state = simple_env.reset()
        absolute = compute_deformation(
            state.sim_state.positions,
            simple_env.topology,
            state.sim_state.spring_init_rest_length,
            simple_env.deformation_info,
        )
        compliance = compute_deformation(
            state.sim_state.positions,
            simple_env.topology,
            state.sim_state.spring_init_rest_length,
            simple_env.deformation_info,
            spring_rest_length=state.sim_state.spring_rest_length,
            reference="current",
        )
        np.testing.assert_allclose(np.array(absolute), 1.0, atol=1e-4)
        np.testing.assert_allclose(np.array(compliance), 1.0, atol=1e-4)

    def test_compliance_during_actuation(self, simple_env):
        """Actuated equilibrium should be near 1.0 in compliance mode."""
        _, state = simple_env.reset()
        scale = 1.4
        robot_pos = state.sim_state.positions[simple_env.robot_point_indices]
        com = jnp.mean(robot_pos, axis=0)
        new_positions = state.sim_state.positions.at[simple_env.robot_point_indices].set(
            com + (robot_pos - com) * scale
        )
        new_sim = state.sim_state._replace(
            positions=new_positions,
            spring_rest_length=state.sim_state.spring_init_rest_length * scale,
        )

        absolute = compute_deformation(
            new_sim.positions,
            simple_env.topology,
            new_sim.spring_init_rest_length,
            simple_env.deformation_info,
        )
        compliance = compute_deformation(
            new_sim.positions,
            simple_env.topology,
            new_sim.spring_init_rest_length,
            simple_env.deformation_info,
            spring_rest_length=new_sim.spring_rest_length,
            reference="current",
        )

        max_abs_deviation = np.max(np.abs(np.array(absolute) - 1.0))
        assert max_abs_deviation > 1e-3
        np.testing.assert_allclose(np.array(compliance), 1.0, atol=1e-3)

    def test_compliance_with_external_force(self, simple_env):
        """External displacement should move compliance deformation away from 1.0."""
        _, state = simple_env.reset()
        scale = 1.4
        robot_pos = state.sim_state.positions[simple_env.robot_point_indices]
        com = jnp.mean(robot_pos, axis=0)
        actuated_positions = state.sim_state.positions.at[simple_env.robot_point_indices].set(
            com + (robot_pos - com) * scale
        )
        point_idx = int(simple_env.robot_point_indices[0])
        perturbed_positions = actuated_positions.at[point_idx, 0].add(0.02)
        new_sim = state.sim_state._replace(
            positions=perturbed_positions,
            spring_rest_length=state.sim_state.spring_init_rest_length * scale,
        )

        compliance = compute_deformation(
            new_sim.positions,
            simple_env.topology,
            new_sim.spring_init_rest_length,
            simple_env.deformation_info,
            spring_rest_length=new_sim.spring_rest_length,
            reference="current",
        )
        max_deviation = np.max(np.abs(np.array(compliance) - 1.0))
        assert max_deviation > 1e-3

    def test_dual_deformation_matches_separate_passes(self, simple_env):
        _, state = simple_env.reset()
        dual = compute_dual_deformation(
            state.sim_state.positions,
            simple_env.topology,
            state.sim_state.spring_init_rest_length,
            simple_env.deformation_info,
            state.sim_state.spring_rest_length,
        )
        absolute = compute_deformation(
            state.sim_state.positions,
            simple_env.topology,
            state.sim_state.spring_init_rest_length,
            simple_env.deformation_info,
        )
        compliance = compute_deformation(
            state.sim_state.positions,
            simple_env.topology,
            state.sim_state.spring_init_rest_length,
            simple_env.deformation_info,
            spring_rest_length=state.sim_state.spring_rest_length,
            reference="current",
        )
        np.testing.assert_allclose(np.array(dual[:, :2]), np.array(absolute), atol=1e-6)
        np.testing.assert_allclose(np.array(dual[:, 2:]), np.array(compliance), atol=1e-6)

    def test_reference_validation(self, simple_env):
        _, state = simple_env.reset()
        with pytest.raises(ValueError, match="spring_rest_length"):
            compute_deformation(
                state.sim_state.positions,
                simple_env.topology,
                state.sim_state.spring_init_rest_length,
                simple_env.deformation_info,
                reference="current",
            )


class TestSentinelHandling:
    """Springs indexed as -1 should contribute ratio=1.0 (no deformation)."""

    def test_all_sentinel_gives_ones(self):
        """DeformationInfo with all -1 indices → all 1.0."""
        n_voxels = 2
        deform_info = DeformationInfo(
            voxel_horiz_springs=jnp.full((n_voxels, 2), -1, dtype=jnp.int32),
            voxel_vert_springs=jnp.full((n_voxels, 2), -1, dtype=jnp.int32),
            n_robot_voxels=n_voxels,
        )
        # Minimal topology and positions
        n_springs = 4
        topology = SpringTopology(
            a_idx=jnp.array([0, 1, 0, 1], dtype=jnp.int32),
            b_idx=jnp.array([1, 2, 2, 3], dtype=jnp.int32),
        )
        positions = jnp.array([
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]
        ])
        rest_lengths = jnp.array([0.1, 0.1, 0.1, 0.1])

        deform = compute_deformation(positions, topology, rest_lengths, deform_info)
        np.testing.assert_allclose(np.array(deform), 1.0, atol=1e-6)


class TestDeformationInfoShape:
    """Verify DeformationInfo has correct shapes."""

    def test_shape_matches_n_robot_voxels(self, simple_env):
        info = simple_env.deformation_info
        assert info.voxel_horiz_springs.shape == (info.n_robot_voxels, 2)
        assert info.voxel_vert_springs.shape == (info.n_robot_voxels, 2)

    def test_n_robot_voxels_correct(self, simple_env):
        """3x1 body has 3 non-empty voxels."""
        assert simple_env.deformation_info.n_robot_voxels == 3

    def test_spring_indices_valid(self, simple_env):
        """All spring indices should be >= -1 and < n_springs."""
        info = simple_env.deformation_info
        n_springs = simple_env.topology.a_idx.shape[0]
        h_valid = jnp.all((info.voxel_horiz_springs >= -1) &
                          (info.voxel_horiz_springs < n_springs))
        v_valid = jnp.all((info.voxel_vert_springs >= -1) &
                          (info.voxel_vert_springs < n_springs))
        assert h_valid, "Invalid horizontal spring indices"
        assert v_valid, "Invalid vertical spring indices"


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------

class TestJIT:
    def test_jit_compute_deformation(self, simple_env):
        obs, state = simple_env.reset()
        jit_fn = jax.jit(lambda pos: compute_deformation(
            pos, simple_env.topology,
            state.sim_state.spring_init_rest_length,
            simple_env.deformation_info,
        ))
        deform = jit_fn(state.sim_state.positions)
        assert deform.shape == (simple_env.n_robot_voxels, 2)
        assert jnp.all(jnp.isfinite(deform))

    def test_jit_compute_deformation_compliance(self, simple_env):
        _, state = simple_env.reset()
        jit_fn = jax.jit(
            lambda pos, spring_rest: compute_deformation(
                pos,
                simple_env.topology,
                state.sim_state.spring_init_rest_length,
                simple_env.deformation_info,
                spring_rest_length=spring_rest,
                reference="current",
            )
        )
        deform = jit_fn(
            state.sim_state.positions,
            state.sim_state.spring_rest_length,
        )
        assert deform.shape == (simple_env.n_robot_voxels, 2)
        assert jnp.all(jnp.isfinite(deform))

    def test_jit_env_step_with_deformation(self, simple_env):
        obs, state = simple_env.reset()
        action = jnp.ones(simple_env.n_actuators)
        obs2, state2, reward, done = simple_env.step(state, action)
        assert obs2.shape == (simple_env.obs_dim,)
        assert jnp.all(jnp.isfinite(obs2))


# ---------------------------------------------------------------------------
# vmap compatibility
# ---------------------------------------------------------------------------

class TestVmap:
    def test_vmap_compute_deformation(self, simple_env):
        """Deformation should work under vmap over batch of positions."""
        obs, state = simple_env.reset()
        batch_size = 4

        # Create batch of positions
        batch_pos = jnp.stack([state.sim_state.positions] * batch_size)

        vmap_fn = jax.vmap(lambda pos: compute_deformation(
            pos, simple_env.topology,
            state.sim_state.spring_init_rest_length,
            simple_env.deformation_info,
        ))
        batch_deform = vmap_fn(batch_pos)
        assert batch_deform.shape == (batch_size, simple_env.n_robot_voxels, 2)
        np.testing.assert_allclose(np.array(batch_deform), 1.0, atol=1e-4)

    def test_vmap_compute_deformation_compliance(self, simple_env):
        _, state = simple_env.reset()
        batch_size = 4
        batch_pos = jnp.stack([state.sim_state.positions] * batch_size)
        batch_rest = jnp.stack([state.sim_state.spring_rest_length] * batch_size)

        vmap_fn = jax.vmap(
            lambda pos, spring_rest: compute_deformation(
                pos,
                simple_env.topology,
                state.sim_state.spring_init_rest_length,
                simple_env.deformation_info,
                spring_rest_length=spring_rest,
                reference="current",
            )
        )
        batch_deform = vmap_fn(batch_pos, batch_rest)
        assert batch_deform.shape == (batch_size, simple_env.n_robot_voxels, 2)
        np.testing.assert_allclose(np.array(batch_deform), 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Integration: environment observations
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_obs_includes_deformation(self, simple_env):
        """Obs vector should end with deformation values."""
        obs, state = simple_env.reset()
        n_deform = 2 * simple_env.n_robot_voxels
        assert obs.shape[0] == simple_env.obs_dim
        # Deformation at rest should be ~1.0
        deform_portion = obs[-n_deform:]
        np.testing.assert_allclose(np.array(deform_portion), 1.0, atol=1e-4)

    def test_obs_without_deformation(self, simple_env_no_deform):
        """include_deformation=False should give original obs."""
        env = simple_env_no_deform
        obs, _ = env.reset()
        expected_dim = 2 + 2 * env.n_robot_points
        assert obs.shape[0] == expected_dim
        assert env.obs_dim == expected_dim

    def test_deformation_changes_after_step(self, simple_env):
        """After physics step, deformation should deviate from 1.0."""
        obs, state = simple_env.reset()
        # Take an extreme action
        action = jnp.full(simple_env.n_actuators, 1.6)
        obs2, state2, _, _ = simple_env.step(state, action)

        n_deform = 2 * simple_env.n_robot_voxels
        deform_after = np.array(obs2[-n_deform:])
        # At least some deformation values should differ from 1.0
        max_deviation = np.max(np.abs(deform_after - 1.0))
        assert max_deviation > 1e-4, (
            f"Deformation didn't change after step: max deviation={max_deviation}"
        )

    def test_multi_step_deformation_finite(self, simple_env):
        """Deformation should stay finite over 20 steps."""
        obs, state = simple_env.reset()
        for step in range(20):
            t = step / 20
            action = jnp.array([1.0 + 0.5 * jnp.sin(2 * jnp.pi * t + i * 0.5)
                                for i in range(simple_env.n_actuators)])
            action = jnp.clip(action, 0.6, 1.6)
            obs, state, _, _ = simple_env.step(state, action)

        n_deform = 2 * simple_env.n_robot_voxels
        deform = np.array(obs[-n_deform:])
        assert np.all(np.isfinite(deform)), f"Non-finite deformation: {deform}"
        assert np.all(deform > 0), f"Deformation should be positive: {deform}"
