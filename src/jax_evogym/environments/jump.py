"""Jumper-v0: robot jumps vertically with minimal lateral drift.

Terrain: Jumper-v0.json — flat ground, robot at (32, 1).
Obs: vel_com (2) + rel_pos (2*n_robot_points) + floor_obs(sight_dist=2, 5).
Reward: 10 * delta_y - 5 * |delta_x|.
Done: self-collision (-3.0) / max_steps.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..types import JumperEnvState, StepOutput
from ..collision import is_self_colliding
from ..sim import env_step
from ._base import EvoGymBaseEnv
from ._obs import get_vel_com_obs, get_relative_pos_obs, get_floor_obs, get_deformation_obs


def _jumper_com(sim_state, robot_point_indices):
    active_pos = sim_state.positions[robot_point_indices]
    return jnp.mean(active_pos, axis=0)


def _jumper_obs(
    sim_state,
    robot_point_indices,
    topology,
    terrain_positions,
    terrain_mask,
    deformation_info,
    include_deformation,
    sight_dist,
):
    vel_com = get_vel_com_obs(sim_state.velocities_true, robot_point_indices)
    rel_pos = get_relative_pos_obs(sim_state.positions, robot_point_indices)
    com = _jumper_com(sim_state, robot_point_indices)
    floor = get_floor_obs(com, terrain_positions, terrain_mask, sight_dist)
    parts = [vel_com, rel_pos, floor]
    if include_deformation:
        deform = get_deformation_obs(
            sim_state.positions,
            topology,
            sim_state.spring_init_rest_length,
            deformation_info,
        )
        parts.append(deform)
    return jnp.concatenate(parts)


@partial(jax.jit, static_argnames=("max_steps", "include_deformation", "sight_dist"))
def _jumper_step_impl(
    env_state: JumperEnvState,
    action: jnp.ndarray,
    *,
    topology,
    constants,
    actuator_info,
    collision_data,
    static_collider_data,
    max_steps: int,
    robot_point_indices,
    terrain_positions,
    terrain_mask,
    deformation_info,
    include_deformation: bool,
    sight_dist: int,
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

    new_com = _jumper_com(new_sim_state, robot_point_indices)
    delta_y = new_com[1] - env_state.prev_com_y
    delta_x = new_com[0] - env_state.prev_com_x
    reward = 10.0 * delta_y - 5.0 * jnp.abs(delta_x)

    new_step_count = env_state.step_count + 1
    self_colliding = is_self_colliding(new_sim_state.positions, collision_data)
    truncated = new_step_count >= max_steps

    reward = jnp.where(self_colliding, reward - 3.0, reward)
    done = self_colliding | truncated
    reward = jnp.where(env_state.done, 0.0, reward)
    done = env_state.done | done

    obs = _jumper_obs(
        new_sim_state,
        robot_point_indices,
        topology,
        terrain_positions,
        terrain_mask,
        deformation_info,
        include_deformation,
        sight_dist,
    )
    new_env_state = JumperEnvState(
        sim_state=new_sim_state,
        step_count=new_step_count,
        done=done,
        prev_com_x=new_com[0],
        prev_com_y=new_com[1],
    )
    return obs, new_env_state, reward, done


class JumperV0(EvoGymBaseEnv):
    """Jumper-v0: soft robot jumps vertically."""

    SIGHT_DIST = 2

    def __init__(
        self,
        body: np.ndarray,
        connections: np.ndarray | None = None,
        max_steps: int = 500,
        include_deformation: bool = True,
    ):
        super().__init__(
            "Jumper-v0.json", body, spawn_x=32, spawn_y=1,
            connections=connections, max_steps=max_steps,
        )
        self.include_deformation = include_deformation
        # vel_com(2) + rel_pos(2*n) + floor_obs(2*sd+1)
        self.obs_dim = 2 + 2 * self.n_robot_points + (2 * self.SIGHT_DIST + 1)
        if include_deformation:
            self.obs_dim += 2 * self.n_robot_voxels

        com = self._compute_com(self._init_sim_state)
        self._init_env_state = JumperEnvState(
            sim_state=self._init_sim_state,
            step_count=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False),
            prev_com_x=com[0],
            prev_com_y=com[1],
        )

    def reset(self):
        """Return (obs, JumperEnvState) for episode start."""
        obs = self._compute_obs(self._init_sim_state)
        return obs, self._init_env_state

    def step(self, env_state: JumperEnvState, action: jnp.ndarray):
        return _jumper_step_impl(
            env_state,
            action,
            topology=self.topology,
            constants=self.constants,
            actuator_info=self.actuator_info,
            collision_data=self.collision_data,
            static_collider_data=self.static_collider_data,
            max_steps=self.max_steps,
            robot_point_indices=self.robot_point_indices,
            terrain_positions=self.terrain_positions,
            terrain_mask=self.terrain_mask,
            deformation_info=self.deformation_info,
            include_deformation=self.include_deformation,
            sight_dist=self.SIGHT_DIST,
        )

    def _compute_com(self, sim_state):
        return _jumper_com(sim_state, self.robot_point_indices)

    def _compute_obs(self, sim_state):
        return _jumper_obs(
            sim_state,
            self.robot_point_indices,
            self.topology,
            self.terrain_positions,
            self.terrain_mask,
            self.deformation_info,
            self.include_deformation,
            self.SIGHT_DIST,
        )


def make_jumper_episode_step(env: JumperV0):
    """Factory returning a lax.scan-compatible step function."""

    def episode_step(env_state: JumperEnvState, action: jnp.ndarray):
        obs, new_env_state, reward, done = env.step(env_state, action)
        output = StepOutput(
            obs=obs, reward=reward, done=done,
            positions=new_env_state.sim_state.positions,
        )
        return new_env_state, output

    return episode_step
