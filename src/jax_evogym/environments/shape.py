"""Shape-changing environments: robot in walled arena, maximize span.

HeightMaximizerV0: maximize vertical span (y_max - y_min).
WingspanMaximizerV0: maximize horizontal span (x_max - x_min).

Terrain: ShapeChange.json — walled enclosure, robot at (7, 1).
Obs: rel_pos only (2 * n_robot_points).
Reward: delta(span) per step.
Done: self-collision (-3.0 penalty) or max_steps.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..types import ShapeEnvState, StepOutput
from ..collision import is_self_colliding
from ..sim import env_step
from ._base import EvoGymBaseEnv
from ._obs import get_relative_pos_obs, get_deformation_obs


def _shape_span(sim_state, robot_point_indices, span_axis):
    pos = sim_state.positions[robot_point_indices]
    return jnp.max(pos[:, span_axis]) - jnp.min(pos[:, span_axis])


def _shape_obs(
    sim_state,
    robot_point_indices,
    topology,
    deformation_info,
    include_deformation,
):
    parts = [get_relative_pos_obs(sim_state.positions, robot_point_indices)]
    if include_deformation:
        deform = get_deformation_obs(
            sim_state.positions,
            topology,
            sim_state.spring_init_rest_length,
            deformation_info,
        )
        parts.append(deform)
    return jnp.concatenate(parts)


@partial(jax.jit, static_argnames=("max_steps", "include_deformation", "span_axis"))
def _shape_step_impl(
    env_state: ShapeEnvState,
    action: jnp.ndarray,
    *,
    topology,
    constants,
    actuator_info,
    collision_data,
    static_collider_data,
    max_steps: int,
    robot_point_indices,
    deformation_info,
    include_deformation: bool,
    span_axis: int,
):
    sim_state = env_state.sim_state
    action = jnp.clip(action, 0.6, 1.6)
    action = jnp.where(jnp.abs(action) < 1e-8, 0.0, action)

    new_sim_state = env_step(
        sim_state,
        topology,
        constants,
        action,
        actuator_info,
        collision_data=collision_data,
        static_collider_data=static_collider_data,
    )

    new_span = _shape_span(new_sim_state, robot_point_indices, span_axis)
    reward = new_span - env_state.prev_span

    new_step_count = env_state.step_count + 1
    self_colliding = is_self_colliding(new_sim_state.positions, collision_data)
    truncated = new_step_count >= max_steps

    reward = jnp.where(self_colliding, reward - 3.0, reward)
    done = self_colliding | truncated
    reward = jnp.where(env_state.done, 0.0, reward)
    done = env_state.done | done

    obs = _shape_obs(
        new_sim_state,
        robot_point_indices,
        topology,
        deformation_info,
        include_deformation,
    )
    new_env_state = ShapeEnvState(
        sim_state=new_sim_state,
        step_count=new_step_count,
        done=done,
        prev_span=new_span,
    )
    return obs, new_env_state, reward, done


class _ShapeBaseEnv(EvoGymBaseEnv):
    """Shared logic for shape-changing environments."""

    # span_axis: 0 = x (wingspan), 1 = y (height)
    span_axis: int

    def __init__(
        self,
        body: np.ndarray,
        connections: np.ndarray | None = None,
        max_steps: int = 500,
        include_deformation: bool = True,
    ):
        super().__init__(
            "ShapeChange.json", body, spawn_x=7, spawn_y=1,
            connections=connections, max_steps=max_steps,
        )
        self.include_deformation = include_deformation
        self.obs_dim = 2 * self.n_robot_points
        if include_deformation:
            self.obs_dim += 2 * self.n_robot_voxels

        # Build initial ShapeEnvState
        init_span = self._compute_span(self._init_sim_state)
        self._init_env_state = ShapeEnvState(
            sim_state=self._init_sim_state,
            step_count=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False),
            prev_span=init_span,
        )

    def reset(self):
        """Return (obs, ShapeEnvState) for episode start."""
        obs = self._compute_obs(self._init_sim_state)
        return obs, self._init_env_state

    def step(self, env_state: ShapeEnvState, action: jnp.ndarray):
        """One environment step. Returns (obs, new_env_state, reward, done)."""
        return _shape_step_impl(
            env_state,
            action,
            topology=self.topology,
            constants=self.constants,
            actuator_info=self.actuator_info,
            collision_data=self.collision_data,
            static_collider_data=self.static_collider_data,
            max_steps=self.max_steps,
            robot_point_indices=self.robot_point_indices,
            deformation_info=self.deformation_info,
            include_deformation=self.include_deformation,
            span_axis=self.span_axis,
        )

    def _compute_span(self, sim_state):
        return _shape_span(sim_state, self.robot_point_indices, self.span_axis)

    def _compute_obs(self, sim_state):
        return _shape_obs(
            sim_state,
            self.robot_point_indices,
            self.topology,
            self.deformation_info,
            self.include_deformation,
        )


class HeightMaximizerV0(_ShapeBaseEnv):
    """Maximize vertical span (y_max - y_min) of robot body."""

    span_axis = 1

    def __init__(
        self,
        body: np.ndarray,
        connections: np.ndarray | None = None,
        max_steps: int = 500,
        include_deformation: bool = True,
    ):
        super().__init__(body, connections=connections, max_steps=max_steps,
                         include_deformation=include_deformation)


class WingspanMaximizerV0(_ShapeBaseEnv):
    """Maximize horizontal span (x_max - x_min) of robot body."""

    span_axis = 0

    def __init__(
        self,
        body: np.ndarray,
        connections: np.ndarray | None = None,
        max_steps: int = 600,
        include_deformation: bool = True,
    ):
        super().__init__(body, connections=connections, max_steps=max_steps,
                         include_deformation=include_deformation)


def make_shape_episode_step(env: _ShapeBaseEnv):
    """Factory returning a lax.scan-compatible step function for shape envs."""

    def episode_step(env_state: ShapeEnvState, action: jnp.ndarray):
        obs, new_env_state, reward, done = env.step(env_state, action)
        output = StepOutput(
            obs=obs, reward=reward, done=done,
            positions=new_env_state.sim_state.positions,
        )
        return new_env_state, output

    return episode_step
