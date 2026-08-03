"""Climber-v0: robot climbs a vertical tube.

Terrain: Climber-v0.json — tall vertical tube with walls.
Obs: vel_com (2) + rel_pos (2*n_robot_points).
Reward: delta COM y per step, +1.0 at goal (com_y > 8.6).
Done: self-collision (-3.0) / goal (+1.0) / max_steps.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..types import ClimberEnvState, StepOutput
from ..collision import is_self_colliding
from ..sim import env_step
from ._base import EvoGymBaseEnv
from ._obs import get_vel_com_obs, get_relative_pos_obs, get_deformation_obs


def _climber_com_y(sim_state, robot_point_indices):
    active_pos = sim_state.positions[robot_point_indices]
    return jnp.mean(active_pos[:, 1])


def _climber_obs(
    sim_state,
    robot_point_indices,
    topology,
    deformation_info,
    include_deformation,
):
    vel_com = get_vel_com_obs(sim_state.velocities_true, robot_point_indices)
    rel_pos = get_relative_pos_obs(sim_state.positions, robot_point_indices)
    parts = [vel_com, rel_pos]
    if include_deformation:
        deform = get_deformation_obs(
            sim_state.positions,
            topology,
            sim_state.spring_init_rest_length,
            deformation_info,
        )
        parts.append(deform)
    return jnp.concatenate(parts)


@partial(jax.jit, static_argnames=("max_steps", "include_deformation"))
def _climber_step_impl(
    env_state: ClimberEnvState,
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

    new_com_y = _climber_com_y(new_sim_state, robot_point_indices)
    reward = new_com_y - env_state.prev_com_y

    new_step_count = env_state.step_count + 1
    self_colliding = is_self_colliding(new_sim_state.positions, collision_data)
    reached_goal = new_com_y > 8.6
    truncated = new_step_count >= max_steps

    reward = jnp.where(self_colliding, reward - 3.0, reward)
    reward = jnp.where(reached_goal, reward + 1.0, reward)

    done = self_colliding | reached_goal | truncated
    reward = jnp.where(env_state.done, 0.0, reward)
    done = env_state.done | done

    obs = _climber_obs(
        new_sim_state,
        robot_point_indices,
        topology,
        deformation_info,
        include_deformation,
    )
    new_env_state = ClimberEnvState(
        sim_state=new_sim_state,
        step_count=new_step_count,
        done=done,
        prev_com_y=new_com_y,
    )
    return obs, new_env_state, reward, done


class ClimberV0(EvoGymBaseEnv):
    """Climber-v0: soft robot climbs upward in a vertical tube."""

    def __init__(
        self,
        body: np.ndarray,
        connections: np.ndarray | None = None,
        max_steps: int = 400,
        include_deformation: bool = True,
    ):
        super().__init__(
            "Climber-v0.json", body, spawn_x=1, spawn_y=1,
            connections=connections, max_steps=max_steps,
        )
        self.include_deformation = include_deformation
        self.obs_dim = 2 + 2 * self.n_robot_points
        if include_deformation:
            self.obs_dim += 2 * self.n_robot_voxels

        init_com_y = self._compute_com_y(self._init_sim_state)
        self._init_env_state = ClimberEnvState(
            sim_state=self._init_sim_state,
            step_count=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False),
            prev_com_y=init_com_y,
        )

    def reset(self):
        """Return (obs, ClimberEnvState) for episode start."""
        obs = self._compute_obs(self._init_sim_state)
        return obs, self._init_env_state

    def step(self, env_state: ClimberEnvState, action: jnp.ndarray):
        return _climber_step_impl(
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
        )

    def _compute_com_y(self, sim_state):
        return _climber_com_y(sim_state, self.robot_point_indices)

    def _compute_obs(self, sim_state):
        return _climber_obs(
            sim_state,
            self.robot_point_indices,
            self.topology,
            self.deformation_info,
            self.include_deformation,
        )


def make_climber_episode_step(env: ClimberV0):
    """Factory returning a lax.scan-compatible step function."""

    def episode_step(env_state: ClimberEnvState, action: jnp.ndarray):
        obs, new_env_state, reward, done = env.step(env_state, action)
        output = StepOutput(
            obs=obs, reward=reward, done=done,
            positions=new_env_state.sim_state.positions,
        )
        return new_env_state, output

    return episode_step
