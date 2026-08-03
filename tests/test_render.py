"""Tests for jax_evogym rendering system."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_evogym import constants as C
from jax_evogym.types import RenderInfo, RenderStepOutput

PIL = pytest.importorskip("PIL", reason="Pillow required for render tests")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def simple_body():
    """Simple 3x1 walker: [H_ACT, SOFT, V_ACT]."""
    return np.array([[C.H_ACT, C.SOFT, C.V_ACT]])


@pytest.fixture(scope="module")
def env(simple_body):
    from jax_evogym.environments.walker import WalkerV0
    return WalkerV0(simple_body)


@pytest.fixture(scope="module")
def obs_and_state(env):
    return env.reset()


# ---------------------------------------------------------------------------
# RenderInfo construction
# ---------------------------------------------------------------------------

class TestRenderInfo:
    def test_render_info_exists(self, env):
        assert hasattr(env, "render_info")
        assert isinstance(env.render_info, RenderInfo)

    def test_cell_vertices_shape(self, env):
        ri = env.render_info
        n_cells = ri.cell_mask.shape[0]
        assert ri.cell_vertices.shape == (n_cells, 4)
        assert ri.cell_vertices.dtype == np.int32

    def test_cell_vertices_reference_dynamic_and_static_points(self, env):
        ri = env.render_info
        n_total_points = env.n_points + ri.static_point_positions.shape[0]
        active_corners = ri.cell_vertices[ri.cell_mask]
        assert np.all(active_corners >= 0)
        assert np.all(active_corners < n_total_points)

    def test_cell_types_shape(self, env):
        ri = env.render_info
        assert ri.cell_types.shape == (ri.cell_mask.shape[0],)

    def test_cell_mask_shape(self, env):
        ri = env.render_info
        assert ri.cell_mask.shape == (ri.cell_vertices.shape[0],)
        assert ri.cell_mask.dtype == bool

    def test_surface_edge_mask_shape(self, env):
        ri = env.render_info
        assert ri.surface_edge_mask.shape == (ri.cell_vertices.shape[0], 4)

    def test_cell_is_robot_shape(self, env):
        ri = env.render_info
        assert ri.cell_is_robot.shape == (ri.cell_vertices.shape[0],)
        assert ri.cell_is_robot.dtype == bool
        assert np.any(ri.cell_is_robot)

    def test_actuator_cell_indices_length(self, env):
        """Should match n_actuators."""
        ri = env.render_info
        assert len(ri.actuator_cell_indices) == env.n_actuators

    def test_render_cell_spring_pair_shapes(self, env):
        ri = env.render_info
        n_cells = ri.cell_mask.shape[0]
        assert ri.cell_h_spring_pairs.shape == (n_cells, 2)
        assert ri.cell_v_spring_pairs.shape == (n_cells, 2)

    def test_spring_init_rest_length_cached(self, env):
        """Should be numpy, matching sim state."""
        ri = env.render_info
        assert isinstance(ri.spring_init_rest_length, np.ndarray)
        expected = np.array(env._init_sim_state.spring_init_rest_length)
        np.testing.assert_array_equal(ri.spring_init_rest_length, expected)

    def test_robot_point_indices(self, env):
        ri = env.render_info
        assert isinstance(ri.robot_point_indices, np.ndarray)
        assert len(ri.robot_point_indices) == env.n_robot_points


class TestRenderActuationMapping:
    def test_nonzero_spawn_mapping_is_render_cell_indexed(self):
        from jax_evogym.build import compile_world_template, instantiate_world
        from jax_evogym.world import EvoWorld

        world = EvoWorld()
        world.add_from_array(
            "robot",
            np.array([[C.H_ACT, C.CONTRACTILE, C.V_ACT]], dtype=np.int32),
            5,
            3,
        )
        template = compile_world_template(world, robot_name="robot")
        built = instantiate_world(template)
        ri = built.render_info

        expected_indices = np.flatnonzero(
            ri.cell_mask
            & ri.cell_is_robot
            & np.isin(ri.cell_types, np.array([C.H_ACT, C.V_ACT, C.CONTRACTILE], dtype=np.int32))
        ).astype(np.int32)
        np.testing.assert_array_equal(ri.actuator_cell_indices, expected_indices)

        n_springs = int(np.asarray(built.sim_state.spring_rest_length).shape[0])
        for cell_idx in ri.actuator_cell_indices:
            vtype = int(ri.cell_types[cell_idx])
            h_pair = np.asarray(ri.cell_h_spring_pairs[cell_idx], dtype=np.int32)
            v_pair = np.asarray(ri.cell_v_spring_pairs[cell_idx], dtype=np.int32)

            if vtype == C.H_ACT:
                assert np.all(h_pair >= 0)
                assert np.all(h_pair < n_springs)
                np.testing.assert_array_equal(v_pair, np.array([-1, -1], dtype=np.int32))
            elif vtype == C.V_ACT:
                assert np.all(v_pair >= 0)
                assert np.all(v_pair < n_springs)
                np.testing.assert_array_equal(h_pair, np.array([-1, -1], dtype=np.int32))
            elif vtype == C.CONTRACTILE:
                assert np.all(h_pair >= 0)
                assert np.all(v_pair >= 0)
                assert np.all(h_pair < n_springs)
                assert np.all(v_pair < n_springs)

    def test_contractile_cell_actuation_uses_both_axes(self):
        from jax_evogym.build import compile_world_template, instantiate_world
        from jax_evogym.render import _compute_cell_actuation
        from jax_evogym.world import EvoWorld

        world = EvoWorld()
        world.add_from_array("robot", np.array([[C.CONTRACTILE]], dtype=np.int32), 4, 2)
        template = compile_world_template(world, robot_name="robot")
        built = instantiate_world(template)
        ri = built.render_info

        contractile_cells = np.flatnonzero(
            ri.cell_mask & ri.cell_is_robot & (ri.cell_types == C.CONTRACTILE)
        )
        assert contractile_cells.size == 1
        cell_idx = int(contractile_cells[0])

        h_pair = np.asarray(ri.cell_h_spring_pairs[cell_idx], dtype=np.int32)
        v_pair = np.asarray(ri.cell_v_spring_pairs[cell_idx], dtype=np.int32)
        assert np.all(h_pair >= 0)
        assert np.all(v_pair >= 0)

        spring_rl = np.array(ri.spring_init_rest_length, dtype=np.float32, copy=True)
        target_ratios = np.array([0.8, 0.9, 1.3, 1.4], dtype=np.float32)
        spring_rl[h_pair[0]] *= target_ratios[0]
        spring_rl[h_pair[1]] *= target_ratios[1]
        spring_rl[v_pair[0]] *= target_ratios[2]
        spring_rl[v_pair[1]] *= target_ratios[3]

        act = _compute_cell_actuation(ri, spring_rl)
        np.testing.assert_allclose(act[cell_idx], np.mean(target_ratios), atol=1e-6)


# ---------------------------------------------------------------------------
# RenderStepOutput via make_render_episode_step
# ---------------------------------------------------------------------------

class TestRenderStepOutput:
    def test_scan_produces_correct_shapes(self, env, obs_and_state):
        from jax_evogym.render import make_render_episode_step

        _, init_state = obs_and_state
        render_step = make_render_episode_step(env)
        actions = jnp.ones((10, env.n_actuators))

        final_state, outputs = jax.lax.scan(render_step, init_state, actions)

        assert isinstance(outputs, RenderStepOutput)
        assert outputs.positions.shape == (10, env.n_points, 2)
        assert outputs.spring_rest_length.shape[0] == 10
        assert outputs.reward.shape == (10,)
        assert outputs.done.shape == (10,)

    def test_scan_all_finite(self, env, obs_and_state):
        from jax_evogym.render import make_render_episode_step

        _, init_state = obs_and_state
        render_step = make_render_episode_step(env)
        actions = jnp.ones((10, env.n_actuators))

        _, outputs = jax.lax.scan(render_step, init_state, actions)

        assert jnp.all(jnp.isfinite(outputs.positions))
        assert jnp.all(jnp.isfinite(outputs.spring_rest_length))
        assert jnp.all(jnp.isfinite(outputs.reward))


# ---------------------------------------------------------------------------
# RenderConfig timing
# ---------------------------------------------------------------------------

class TestRenderConfig:
    def test_real_time_frame_skip(self):
        from jax_evogym.render import RenderConfig
        config = RenderConfig(real_time=True, fps=50)
        assert config.effective_frame_skip == 7

    def test_manual_frame_skip(self):
        from jax_evogym.render import RenderConfig
        config = RenderConfig(real_time=False, frame_skip=3)
        assert config.effective_frame_skip == 3

    def test_real_time_frame_skip_uses_runtime_step_timing(self):
        from jax_evogym.render import RenderConfig

        config = RenderConfig(real_time=True, fps=50, sim_time_per_step=12 * C.DT)
        assert config.effective_frame_skip == 17

    def test_frame_duration_ms(self):
        from jax_evogym.render import RenderConfig
        config = RenderConfig(fps=50)
        assert config.frame_duration_ms == 20

    def test_frame_duration_min_clamped(self):
        from jax_evogym.render import RenderConfig
        config = RenderConfig(fps=200)  # would be 5ms, clamped to 20
        assert config.frame_duration_ms >= 20

    def test_frame_skip_respects_target_frame_budget(self):
        from jax_evogym.render import RenderConfig

        config = RenderConfig(real_time=True, fps=50, target_frames=10)
        # real-time skip @50fps is 7; budget for 100 steps is 10.
        assert config.frame_skip_for_steps(100) == 10

    def test_new_style_defaults(self):
        from jax_evogym.render import RenderConfig

        config = RenderConfig()
        assert config.camera_mode == "fit_robot"
        assert config.supersample == 2
        assert config.grid_major_every == 5
        assert config.edge_width_px == 2


# ---------------------------------------------------------------------------
# render_frame smoke test
# ---------------------------------------------------------------------------

class TestRenderFrame:
    def test_produces_image(self, env, obs_and_state):
        from PIL import Image
        from jax_evogym.render import RenderConfig, render_frame

        _, state = obs_and_state
        positions_np = np.array(state.sim_state.positions)
        config = RenderConfig(enabled=True, width=600, height=300)

        img, cam = render_frame(positions_np, env.render_info, config)

        assert isinstance(img, Image.Image)
        assert img.size == (600, 300)
        assert cam is not None
        assert len(cam) == 2

    def test_with_spring_rest_length(self, env, obs_and_state):
        from PIL import Image
        from jax_evogym.render import RenderConfig, render_frame

        _, state = obs_and_state
        positions_np = np.array(state.sim_state.positions)
        srl_np = np.array(state.sim_state.spring_rest_length)
        config = RenderConfig(enabled=True, dynamic_actuator_colors=True)

        img, cam = render_frame(
            positions_np, env.render_info, config,
            spring_rest_length_np=srl_np,
        )

        assert isinstance(img, Image.Image)

    def test_with_grid(self, env, obs_and_state):
        from PIL import Image
        from jax_evogym.render import RenderConfig, render_frame

        _, state = obs_and_state
        positions_np = np.array(state.sim_state.positions)
        config = RenderConfig(enabled=True, show_grid=True)

        img, _ = render_frame(positions_np, env.render_info, config)
        assert isinstance(img, Image.Image)

    def test_contractile_palette_is_distinct(self):
        from jax_evogym.render import _actuation_color

        assert _actuation_color(C.CONTRACTILE, 1.0) != _actuation_color(C.H_ACT, 1.0)


# ---------------------------------------------------------------------------
# render_episode smoke test
# ---------------------------------------------------------------------------

class TestRenderEpisode:
    def test_correct_frame_count(self, env, obs_and_state):
        from jax_evogym.render import RenderConfig, render_episode, make_render_episode_step

        _, init_state = obs_and_state
        render_step = make_render_episode_step(env)
        actions = jnp.ones((50, env.n_actuators))
        _, outputs = jax.lax.scan(render_step, init_state, actions)

        config = RenderConfig(enabled=True, real_time=True, fps=50)
        frames = render_episode(outputs, env.render_info, config)

        # 50 steps / frame_skip=7 -> ceil(50/7) frames = 8 (indices 0,7,14,21,28,35,42,49)
        expected = len(range(0, 50, 7))
        assert len(frames) == expected

    def test_frame_skip_1(self, env, obs_and_state):
        from jax_evogym.render import RenderConfig, render_episode, make_render_episode_step

        _, init_state = obs_and_state
        render_step = make_render_episode_step(env)
        actions = jnp.ones((10, env.n_actuators))
        _, outputs = jax.lax.scan(render_step, init_state, actions)

        config = RenderConfig(enabled=True, real_time=False, frame_skip=1)
        frames = render_episode(outputs, env.render_info, config)

        assert len(frames) == 10

    def test_target_frames_caps_frame_count(self, env, obs_and_state):
        from jax_evogym.render import RenderConfig, render_episode, make_render_episode_step

        _, init_state = obs_and_state
        render_step = make_render_episode_step(env)
        actions = jnp.ones((120, env.n_actuators))
        _, outputs = jax.lax.scan(render_step, init_state, actions)

        config = RenderConfig(
            enabled=True,
            real_time=False,
            frame_skip=1,
            target_frames=12,
        )
        frames = render_episode(outputs, env.render_info, config)
        assert len(frames) <= 12


# ---------------------------------------------------------------------------
# save_gif smoke test
# ---------------------------------------------------------------------------

class TestSaveGif:
    def test_writes_file(self, env, obs_and_state, tmp_path):
        from jax_evogym.render import RenderConfig, render_episode, save_gif, make_render_episode_step

        _, init_state = obs_and_state
        render_step = make_render_episode_step(env)
        actions = jnp.ones((14, env.n_actuators))
        _, outputs = jax.lax.scan(render_step, init_state, actions)

        config = RenderConfig(enabled=True, real_time=True, fps=50)
        frames = render_episode(outputs, env.render_info, config)

        gif_path = tmp_path / "test.gif"
        result = save_gif(frames, gif_path, config)

        assert gif_path.exists()
        assert gif_path.stat().st_size > 0
        assert result == str(gif_path)

    def test_rejects_empty_frame_list(self, tmp_path):
        from jax_evogym.render import RenderConfig, save_gif

        with pytest.raises(ValueError, match="at least one frame"):
            save_gif([], tmp_path / "empty.gif", RenderConfig(enabled=True))


# ---------------------------------------------------------------------------
# Top-K extraction from time-major traces
# ---------------------------------------------------------------------------

class TestTopKExtraction:
    def test_time_major_slicing(self):
        """Validate (T, P, ...) -> (T, ...) slicing picks correct individual."""
        T, P, D = 10, 5, 3
        data = jnp.arange(T * P * D).reshape(T, P, D)
        idx = 2

        sliced = data[:, idx, :]
        assert sliced.shape == (T, D)

        # Verify values
        for t in range(T):
            expected = jnp.arange(t * P * D + idx * D, t * P * D + (idx + 1) * D)
            np.testing.assert_array_equal(sliced[t], expected)

    def test_tree_map_slicing(self):
        """jax.tree.map slicing on RenderStepOutput-like structure."""
        T, P = 10, 5
        positions = jnp.ones((T, P, 20, 2))
        spring_rl = jnp.ones((T, P, 30))
        reward = jnp.ones((T, P))
        done = jnp.zeros((T, P), dtype=jnp.bool_)

        batch = RenderStepOutput(
            positions=positions,
            spring_rest_length=spring_rl,
            reward=reward,
            done=done,
        )

        idx = 3
        individual = jax.tree.map(lambda x: x[:, idx], batch)

        assert individual.positions.shape == (T, 20, 2)
        assert individual.spring_rest_length.shape == (T, 30)
        assert individual.reward.shape == (T,)
        assert individual.done.shape == (T,)
