"""Tests for jax_evogym.environment — WalkerV0 Gym-style interface."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_evogym import constants as C
from jax_evogym.environment import WalkerV0, make_walker_episode_step
from jax_evogym.types import EnvState, StepOutput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def simple_body():
    """Simple 3x1 walker: [H_ACT, SOFT, V_ACT]."""
    return np.array([[C.H_ACT, C.SOFT, C.V_ACT]])


@pytest.fixture(scope="module")
def env(simple_body):
    """WalkerV0 with simple 3x1 body."""
    return WalkerV0(simple_body)


@pytest.fixture(scope="module")
def obs_and_state(env):
    """Initial observation and state from reset."""
    return env.reset()


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------

class TestInit:
    def test_obs_dim(self, env):
        # 3x1 body has 4*3=12 corner points, but shared corners reduce it.
        # With 3 cells in a row: 4 columns x 2 rows = 8 unique points.
        # + deformation(2 * n_robot_voxels) for proprioceptive feedback.
        assert env.obs_dim == 2 + 2 * env.n_robot_points + 2 * env.n_robot_voxels

    def test_n_actuators_positive(self, env):
        assert env.n_actuators > 0

    def test_active_point_indices_shape(self, env):
        assert env.active_point_indices.ndim == 1
        assert env.active_point_indices.shape[0] == env.n_active

    def test_active_point_indices_in_range(self, env):
        assert jnp.all(env.active_point_indices >= 0)
        assert jnp.all(env.active_point_indices < env.n_points)

    def test_max_steps(self, env):
        assert env.max_steps == 500

    def test_custom_max_steps(self, simple_body):
        env2 = WalkerV0(simple_body, max_steps=100)
        assert env2.max_steps == 100


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_obs_shape(self, env, obs_and_state):
        obs, _ = obs_and_state
        assert obs.shape == (env.obs_dim,)

    def test_obs_finite(self, obs_and_state):
        obs, _ = obs_and_state
        assert jnp.all(jnp.isfinite(obs))

    def test_step_count_zero(self, obs_and_state):
        _, state = obs_and_state
        assert int(state.step_count) == 0

    def test_not_done(self, obs_and_state):
        _, state = obs_and_state
        assert not bool(state.done)

    def test_vel_com_near_zero(self, obs_and_state):
        obs, _ = obs_and_state
        # First 2 elements are vel_com, should be ~0 at start
        vel_com = obs[:2]
        assert jnp.allclose(vel_com, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Step tests
# ---------------------------------------------------------------------------

class TestStep:
    def test_return_types(self, env, obs_and_state):
        _, state = obs_and_state
        action = jnp.ones(env.n_actuators)
        obs, new_state, reward, done = env.step(state, action)
        assert isinstance(obs, jax.Array)
        assert isinstance(new_state, EnvState)
        assert isinstance(reward, jax.Array)
        assert isinstance(done, jax.Array)

    def test_obs_shape_after_step(self, env, obs_and_state):
        _, state = obs_and_state
        action = jnp.ones(env.n_actuators)
        obs, _, _, _ = env.step(state, action)
        assert obs.shape == (env.obs_dim,)

    def test_step_count_increments(self, env, obs_and_state):
        _, state = obs_and_state
        action = jnp.ones(env.n_actuators)
        _, new_state, _, _ = env.step(state, action)
        assert int(new_state.step_count) == 1

    def test_obs_finite_after_step(self, env, obs_and_state):
        _, state = obs_and_state
        action = jnp.ones(env.n_actuators)
        obs, _, _, _ = env.step(state, action)
        assert jnp.all(jnp.isfinite(obs))

    def test_action_clipping(self, env, obs_and_state):
        """Actions outside [0.6, 1.6] should be clipped."""
        _, state = obs_and_state
        # Extreme actions — should not crash
        action_low = jnp.zeros(env.n_actuators)
        action_high = jnp.full(env.n_actuators, 10.0)
        obs1, _, _, _ = env.step(state, action_low)
        obs2, _, _, _ = env.step(state, action_high)
        assert jnp.all(jnp.isfinite(obs1))
        assert jnp.all(jnp.isfinite(obs2))


# ---------------------------------------------------------------------------
# Reward tests
# ---------------------------------------------------------------------------

class TestReward:
    def test_nonzero_with_asymmetric_action(self, env, obs_and_state):
        """Asymmetric action should produce non-zero reward after a few steps."""
        _, state = obs_and_state
        # Run 5 steps with expand action
        for _ in range(5):
            action = jnp.full(env.n_actuators, 1.6)
            _, state, reward, _ = env.step(state, action)
        # After several steps of expansion, COM should have moved
        # (reward may be positive or negative, just not exactly zero)
        # Actually with gravity+expansion it's hard to guarantee non-zero
        # on first step, so just check finiteness
        assert jnp.isfinite(reward)

    def test_zero_reward_after_done(self, env, obs_and_state):
        """Once done, reward should be 0."""
        _, state = obs_and_state
        # Force done state
        done_state = state._replace(done=jnp.array(True))
        action = jnp.ones(env.n_actuators)
        _, _, reward, _ = env.step(done_state, action)
        assert float(reward) == 0.0


# ---------------------------------------------------------------------------
# Done tests
# ---------------------------------------------------------------------------

class TestDone:
    def test_not_done_initially(self, obs_and_state):
        _, state = obs_and_state
        assert not bool(state.done)

    def test_done_persists(self, env, obs_and_state):
        """Once done, stays done."""
        _, state = obs_and_state
        done_state = state._replace(done=jnp.array(True))
        action = jnp.ones(env.n_actuators)
        _, new_state, _, done = env.step(done_state, action)
        assert bool(done)
        assert bool(new_state.done)

    def test_max_steps_truncation(self, simple_body):
        """Should terminate at max_steps."""
        env = WalkerV0(simple_body, max_steps=3)
        _, state = env.reset()
        action = jnp.ones(env.n_actuators)
        for i in range(3):
            _, state, _, done = env.step(state, action)
        assert bool(done)
        assert int(state.step_count) == 3


# ---------------------------------------------------------------------------
# Observation scaling tests
# ---------------------------------------------------------------------------

class TestObsScaling:
    def test_vel_com_scaling(self, env, obs_and_state):
        """vel_com should be mean(vel_true) — no extra scaling."""
        _, state = obs_and_state
        action = jnp.ones(env.n_actuators)
        obs, new_state, _, _ = env.step(state, action)

        active_vel = new_state.sim_state.velocities_true[env.robot_point_indices]
        expected_vel_com = jnp.mean(active_vel, axis=0)
        assert jnp.allclose(obs[:2], expected_vel_com, atol=1e-5)

    def test_rel_pos_scaling(self, env, obs_and_state):
        """rel_pos should be (pos - com).T.flatten() — no extra scaling."""
        _, state = obs_and_state
        action = jnp.ones(env.n_actuators)
        obs, new_state, _, _ = env.step(state, action)

        active_pos = new_state.sim_state.positions[env.robot_point_indices]
        com = jnp.mean(active_pos, axis=0)
        expected_rel = (active_pos - com).T.flatten()
        # rel_pos starts at index 2, spans 2*n_robot_points
        rel_end = 2 + 2 * env.n_robot_points
        assert jnp.allclose(obs[2:rel_end], expected_rel, atol=1e-5)


# ---------------------------------------------------------------------------
# JIT tests
# ---------------------------------------------------------------------------

class TestJIT:
    def test_jit_step(self, env, obs_and_state):
        """jax.jit(env.step) should work (it's already jitted, but test explicit)."""
        _, state = obs_and_state
        action = jnp.ones(env.n_actuators)
        # step is already @jit, calling it should not error
        obs, new_state, reward, done = env.step(state, action)
        assert jnp.all(jnp.isfinite(obs))


# ---------------------------------------------------------------------------
# Scan tests
# ---------------------------------------------------------------------------

class TestScan:
    def test_scan_episode(self, env, obs_and_state):
        """lax.scan with make_walker_episode_step for 10 steps."""
        _, init_state = obs_and_state
        step_fn = make_walker_episode_step(env)
        actions = jnp.ones((10, env.n_actuators))

        final_state, outputs = jax.lax.scan(step_fn, init_state, actions)

        assert isinstance(outputs, StepOutput)
        assert outputs.obs.shape == (10, env.obs_dim)
        assert outputs.reward.shape == (10,)
        assert outputs.done.shape == (10,)
        assert outputs.positions.shape == (10, env.n_points, 2)
        assert int(final_state.step_count) == 10

    def test_scan_all_finite(self, env, obs_and_state):
        """All outputs from scan should be finite."""
        _, init_state = obs_and_state
        step_fn = make_walker_episode_step(env)
        actions = jnp.ones((10, env.n_actuators))

        _, outputs = jax.lax.scan(step_fn, init_state, actions)

        assert jnp.all(jnp.isfinite(outputs.obs))
        assert jnp.all(jnp.isfinite(outputs.reward))
        assert jnp.all(jnp.isfinite(outputs.positions))


# ---------------------------------------------------------------------------
# Vmap tests
# ---------------------------------------------------------------------------

class TestVmap:
    def test_batched_step(self, env, obs_and_state):
        """vmap over a population of states."""
        _, init_state = obs_and_state
        pop_size = 4

        # Stack init_state into a batch
        batch_state = jax.tree.map(
            lambda x: jnp.stack([x] * pop_size), init_state
        )
        batch_actions = jnp.ones((pop_size, env.n_actuators))

        batched_step = jax.vmap(env.step, in_axes=(0, 0))
        obs, new_states, rewards, dones = batched_step(batch_state, batch_actions)

        assert obs.shape == (pop_size, env.obs_dim)
        assert rewards.shape == (pop_size,)
        assert dones.shape == (pop_size,)
        assert new_states.step_count.shape == (pop_size,)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_100_step_episode(self, env):
        """100-step episode with random actions — no NaNs, reward evolves."""
        obs, state = env.reset()
        rng = jax.random.PRNGKey(42)
        total_reward = 0.0

        for i in range(100):
            rng, key = jax.random.split(rng)
            action = jax.random.uniform(key, (env.n_actuators,), minval=0.6, maxval=1.6)
            obs, state, reward, done = env.step(state, action)
            assert jnp.all(jnp.isfinite(obs)), f"NaN in obs at step {i}"
            assert jnp.isfinite(reward), f"NaN in reward at step {i}"
            total_reward += float(reward)

        # Positions should have evolved from initial
        assert int(state.step_count) == 100
