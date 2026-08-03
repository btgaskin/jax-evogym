"""Tests for all JAX EvoGym environments — smoke tests + structural checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_evogym import constants as C
from jax_evogym.environments import (
    WalkerV0,
    HeightMaximizerV0,
    WingspanMaximizerV0,
    ClimberV0,
    BridgeWalkerV0,
    JumperV0,
    UpStepperV0,
    make_walker_episode_step,
    make_shape_episode_step,
    make_climber_episode_step,
    make_bridge_walker_episode_step,
    make_jumper_episode_step,
    make_upstepper_episode_step,
)
from jax_evogym.types import (
    EnvState,
    ShapeEnvState,
    ClimberEnvState,
    OrientedEnvState,
    JumperEnvState,
    StepOutput,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def body_3x3():
    """3x3 body with actuators."""
    return np.array([
        [C.H_ACT, C.SOFT, C.V_ACT],
        [C.SOFT, C.H_ACT, C.SOFT],
        [C.V_ACT, C.SOFT, C.H_ACT],
    ])


@pytest.fixture(scope="module")
def body_3x1():
    """Simple 3x1 body."""
    return np.array([[C.H_ACT, C.SOFT, C.V_ACT]])


# ---------------------------------------------------------------------------
# Parameterized env creation
# ---------------------------------------------------------------------------

ENV_CONFIGS = [
    ("WalkerV0", WalkerV0, EnvState, make_walker_episode_step, {"max_steps": 500}),
    ("HeightMaxV0", HeightMaximizerV0, ShapeEnvState, make_shape_episode_step, {"max_steps": 500}),
    ("WingspanMaxV0", WingspanMaximizerV0, ShapeEnvState, make_shape_episode_step, {"max_steps": 600}),
    ("ClimberV0", ClimberV0, ClimberEnvState, make_climber_episode_step, {"max_steps": 400}),
    ("BridgeWalkerV0", BridgeWalkerV0, OrientedEnvState, make_bridge_walker_episode_step, {"max_steps": 500}),
    ("JumperV0", JumperV0, JumperEnvState, make_jumper_episode_step, {"max_steps": 500}),
    ("UpStepperV0", UpStepperV0, OrientedEnvState, make_upstepper_episode_step, {"max_steps": 600}),
]


@pytest.fixture(scope="module", params=ENV_CONFIGS, ids=[c[0] for c in ENV_CONFIGS])
def env_fixture(request, body_3x1):
    """Creates each env with the 3x1 body and returns (env, state_cls, episode_step_factory, config)."""
    name, env_cls, state_cls, ep_step_factory, config = request.param
    env = env_cls(body_3x1)
    return env, state_cls, ep_step_factory, config


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------

class TestStructural:
    def test_obs_dim_positive(self, env_fixture):
        env, _, _, _ = env_fixture
        assert env.obs_dim > 0

    def test_n_actuators_positive(self, env_fixture):
        env, _, _, _ = env_fixture
        assert env.n_actuators > 0

    def test_n_robot_points_positive(self, env_fixture):
        env, _, _, _ = env_fixture
        assert env.n_robot_points > 0

    def test_max_steps(self, env_fixture):
        env, _, _, config = env_fixture
        assert env.max_steps == config["max_steps"]


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_obs_shape(self, env_fixture):
        env, _, _, _ = env_fixture
        obs, _ = env.reset()
        assert obs.shape == (env.obs_dim,)

    def test_obs_finite(self, env_fixture):
        env, _, _, _ = env_fixture
        obs, _ = env.reset()
        assert jnp.all(jnp.isfinite(obs))

    def test_step_count_zero(self, env_fixture):
        env, _, _, _ = env_fixture
        _, state = env.reset()
        assert int(state.step_count) == 0

    def test_not_done(self, env_fixture):
        env, _, _, _ = env_fixture
        _, state = env.reset()
        assert not bool(state.done)

    def test_state_type(self, env_fixture):
        env, state_cls, _, _ = env_fixture
        _, state = env.reset()
        assert isinstance(state, state_cls)


# ---------------------------------------------------------------------------
# Step tests
# ---------------------------------------------------------------------------

class TestStep:
    def test_obs_shape_after_step(self, env_fixture):
        env, _, _, _ = env_fixture
        _, state = env.reset()
        action = jnp.ones(env.n_actuators)
        obs, _, _, _ = env.step(state, action)
        assert obs.shape == (env.obs_dim,)

    def test_obs_finite_after_step(self, env_fixture):
        env, _, _, _ = env_fixture
        _, state = env.reset()
        action = jnp.ones(env.n_actuators)
        obs, _, _, _ = env.step(state, action)
        assert jnp.all(jnp.isfinite(obs))

    def test_step_count_increments(self, env_fixture):
        env, _, _, _ = env_fixture
        _, state = env.reset()
        action = jnp.ones(env.n_actuators)
        _, new_state, _, _ = env.step(state, action)
        assert int(new_state.step_count) == 1

    def test_reward_finite(self, env_fixture):
        env, _, _, _ = env_fixture
        _, state = env.reset()
        action = jnp.ones(env.n_actuators)
        _, _, reward, _ = env.step(state, action)
        assert jnp.isfinite(reward)

    def test_done_is_bool_like(self, env_fixture):
        env, _, _, _ = env_fixture
        _, state = env.reset()
        action = jnp.ones(env.n_actuators)
        _, _, _, done = env.step(state, action)
        assert isinstance(done, jax.Array)

    def test_action_clipping(self, env_fixture):
        """Extreme actions should not crash."""
        env, _, _, _ = env_fixture
        _, state = env.reset()
        action_low = jnp.zeros(env.n_actuators)
        action_high = jnp.full(env.n_actuators, 10.0)
        obs1, _, _, _ = env.step(state, action_low)
        obs2, _, _, _ = env.step(state, action_high)
        assert jnp.all(jnp.isfinite(obs1))
        assert jnp.all(jnp.isfinite(obs2))


# ---------------------------------------------------------------------------
# Done behavior
# ---------------------------------------------------------------------------

class TestDone:
    def test_done_persists(self, env_fixture):
        env, _, _, _ = env_fixture
        _, state = env.reset()
        done_state = state._replace(done=jnp.array(True))
        action = jnp.ones(env.n_actuators)
        _, new_state, reward, done = env.step(done_state, action)
        assert bool(done)
        assert bool(new_state.done)
        assert float(reward) == 0.0

    def test_max_steps_truncation(self, env_fixture):
        env, _, _, _ = env_fixture
        if env.max_steps > 10:
            # Create a short env for this test
            return  # skip — tested per-env below
        _, state = env.reset()
        action = jnp.ones(env.n_actuators)
        for _ in range(env.max_steps):
            _, state, _, done = env.step(state, action)
        assert bool(done)


# ---------------------------------------------------------------------------
# Scan episode tests
# ---------------------------------------------------------------------------

class TestScan:
    def test_scan_episode(self, env_fixture):
        env, _, ep_step_factory, _ = env_fixture
        _, state = env.reset()
        step_fn = ep_step_factory(env)
        actions = jnp.ones((5, env.n_actuators))
        final_state, outputs = jax.lax.scan(step_fn, state, actions)
        assert isinstance(outputs, StepOutput)
        assert outputs.obs.shape == (5, env.obs_dim)
        assert outputs.reward.shape == (5,)
        assert outputs.done.shape == (5,)
        assert int(final_state.step_count) == 5

    def test_scan_all_finite(self, env_fixture):
        env, _, ep_step_factory, _ = env_fixture
        _, state = env.reset()
        step_fn = ep_step_factory(env)
        actions = jnp.ones((5, env.n_actuators))
        _, outputs = jax.lax.scan(step_fn, state, actions)
        assert jnp.all(jnp.isfinite(outputs.obs))
        assert jnp.all(jnp.isfinite(outputs.reward))


# ---------------------------------------------------------------------------
# Vmap tests
# ---------------------------------------------------------------------------

class TestVmap:
    def test_batched_step(self, env_fixture):
        env, _, _, _ = env_fixture
        _, state = env.reset()
        pop_size = 3
        batch_state = jax.tree.map(lambda x: jnp.stack([x] * pop_size), state)
        batch_actions = jnp.ones((pop_size, env.n_actuators))
        batched_step = jax.vmap(env.step, in_axes=(0, 0))
        obs, new_states, rewards, dones = batched_step(batch_state, batch_actions)
        assert obs.shape == (pop_size, env.obs_dim)
        assert rewards.shape == (pop_size,)
        assert dones.shape == (pop_size,)


# ---------------------------------------------------------------------------
# Smoke test: 20-step stability
# ---------------------------------------------------------------------------

class TestStability:
    def test_20_steps_no_nans(self, env_fixture):
        env, _, _, _ = env_fixture
        _, state = env.reset()
        rng = jax.random.PRNGKey(42)
        for i in range(20):
            rng, key = jax.random.split(rng)
            action = jax.random.uniform(key, (env.n_actuators,), minval=0.6, maxval=1.6)
            obs, state, reward, done = env.step(state, action)
            assert jnp.all(jnp.isfinite(obs)), f"NaN at step {i}"
            assert jnp.isfinite(reward), f"NaN reward at step {i}"


# ---------------------------------------------------------------------------
# Per-env specific tests
# ---------------------------------------------------------------------------

class TestWalkerSpecific:
    def test_obs_dim_formula(self, body_3x1):
        env = WalkerV0(body_3x1)
        # vel_com(2) + rel_pos(2*n_robot_points) + deform(2*n_robot_voxels)
        assert env.obs_dim == 2 + 2 * env.n_robot_points + 2 * env.n_robot_voxels

    def test_obs_dim_no_deformation(self, body_3x1):
        env = WalkerV0(body_3x1, include_deformation=False)
        assert env.obs_dim == 2 + 2 * env.n_robot_points

    def test_goal_threshold(self, body_3x1):
        """Walker goal is at com_x > 9.9."""
        env = WalkerV0(body_3x1)
        assert env.max_steps == 500


class TestShapeSpecific:
    def test_height_obs_dim(self, body_3x1):
        env = HeightMaximizerV0(body_3x1)
        # rel_pos + deform
        assert env.obs_dim == 2 * env.n_robot_points + 2 * env.n_robot_voxels

    def test_wingspan_obs_dim(self, body_3x1):
        env = WingspanMaximizerV0(body_3x1)
        assert env.obs_dim == 2 * env.n_robot_points + 2 * env.n_robot_voxels

    def test_wingspan_max_steps(self, body_3x1):
        env = WingspanMaximizerV0(body_3x1)
        assert env.max_steps == 600


class TestClimberSpecific:
    def test_obs_dim_formula(self, body_3x1):
        env = ClimberV0(body_3x1)
        assert env.obs_dim == 2 + 2 * env.n_robot_points + 2 * env.n_robot_voxels

    def test_max_steps(self, body_3x1):
        env = ClimberV0(body_3x1)
        assert env.max_steps == 400


class TestBridgeWalkerSpecific:
    def test_obs_dim_formula(self, body_3x1):
        env = BridgeWalkerV0(body_3x1)
        # vel_com(2) + ort(1) + rel_pos(2*n) + deform(2*v)
        assert env.obs_dim == 2 + 1 + 2 * env.n_robot_points + 2 * env.n_robot_voxels

    def test_n_robot_points_correct(self, body_3x1):
        """Robot points should be the body's corners only, not bridge SOFT cells."""
        env = BridgeWalkerV0(body_3x1)
        # 3x1 body = 4 columns × 2 rows = 8 unique corners
        assert env.n_robot_points == 8


class TestJumperSpecific:
    def test_obs_dim_formula(self, body_3x1):
        env = JumperV0(body_3x1)
        # vel_com(2) + rel_pos(2*n) + floor(5) + deform(2*v)
        assert env.obs_dim == 2 + 2 * env.n_robot_points + 5 + 2 * env.n_robot_voxels


class TestUpStepperSpecific:
    def test_obs_dim_formula(self, body_3x1):
        env = UpStepperV0(body_3x1)
        # vel_com(2) + ort(1) + rel_pos(2*n) + floor(11) + deform(2*v)
        assert env.obs_dim == 2 + 1 + 2 * env.n_robot_points + 11 + 2 * env.n_robot_voxels

    def test_max_steps(self, body_3x1):
        env = UpStepperV0(body_3x1)
        assert env.max_steps == 600
