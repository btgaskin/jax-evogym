"""Walker-v0 environment: soft robot walks right on flat ground.

Terrain: 100×1 flat ground from Walker-v0.json.
Obs: vel_com (2) + rel_pos (2*n_robot_points).
Reward: delta COM x per step.
Done: self-collision (-3), goal x>9.9 (+1), or max_steps.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..types import EnvState, StepOutput
from ..collision import is_self_colliding
from ..sim import env_step
from ._base import EvoGymBaseEnv
from ._obs import get_vel_com_obs, get_relative_pos_obs, get_deformation_obs


def _walker_com_x(sim_state, robot_point_indices):
    active_pos = sim_state.positions[robot_point_indices]
    return jnp.mean(active_pos[:, 0])


def _walker_obs(
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
def _walker_step_impl(
    env_state: EnvState,
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

    new_com_x = _walker_com_x(new_sim_state, robot_point_indices)
    reward = new_com_x - env_state.prev_com_x

    new_step_count = env_state.step_count + 1
    self_colliding = is_self_colliding(new_sim_state.positions, collision_data)
    reached_goal = new_com_x > 9.9
    truncated = new_step_count >= max_steps

    reward = jnp.where(self_colliding, reward - 3.0, reward)
    reward = jnp.where(reached_goal, reward + 1.0, reward)

    done = self_colliding | reached_goal | truncated
    reward = jnp.where(env_state.done, 0.0, reward)
    done = env_state.done | done

    obs = _walker_obs(
        new_sim_state,
        robot_point_indices,
        topology,
        deformation_info,
        include_deformation,
    )
    new_env_state = EnvState(
        sim_state=new_sim_state,
        step_count=new_step_count,
        done=done,
        prev_com_x=new_com_x,
    )
    return obs, new_env_state, reward, done


class WalkerV0(EvoGymBaseEnv):
    """Walker-v0: soft robot walks right on flat ground.

    Uses real terrain voxels from Walker-v0.json (100 FIXED ground cells)
    for collision detection, replacing the old implicit y<0 penalty.
    """

    def __init__(
        self,
        body: np.ndarray,
        connections: np.ndarray | None = None,
        max_steps: int = 500,
        include_deformation: bool = True,
    ):
        super().__init__(
            "Walker-v0.json", body, spawn_x=1, spawn_y=1,
            connections=connections, max_steps=max_steps,
        )
        self.include_deformation = include_deformation
        self.obs_dim = 2 + 2 * self.n_robot_points
        if include_deformation:
            self.obs_dim += 2 * self.n_robot_voxels

        # Build initial EnvState
        init_com_x = self._compute_com_x(self._init_sim_state)
        self._init_env_state = EnvState(
            sim_state=self._init_sim_state,
            step_count=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False),
            prev_com_x=init_com_x,
        )

    def reset(self):
        """Return (obs, EnvState) for episode start."""
        obs = self._compute_obs(self._init_sim_state)
        return obs, self._init_env_state

    def step(self, env_state: EnvState, action: jnp.ndarray):
        """One environment step. Returns (obs, new_env_state, reward, done)."""
        return _walker_step_impl(
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

    def _compute_com_x(self, sim_state):
        return _walker_com_x(sim_state, self.robot_point_indices)

    def _compute_obs(self, sim_state):
        return _walker_obs(
            sim_state,
            self.robot_point_indices,
            self.topology,
            self.deformation_info,
            self.include_deformation,
        )


def make_walker_episode_step(env: WalkerV0):
    """Factory returning a lax.scan-compatible step function."""

    def episode_step(env_state: EnvState, action: jnp.ndarray):
        obs, new_env_state, reward, done = env.step(env_state, action)
        output = StepOutput(
            obs=obs, reward=reward, done=done,
            positions=new_env_state.sim_state.positions,
        )
        return new_env_state, output

    return episode_step
