---
title: Custom Environments
description: Build your own environments using the base class and observation helpers.
---

## The base class

All built-in environments extend `EvoGymBaseEnv`, which handles the full build pipeline:

```python
class EvoGymBaseEnv:
    def __init__(self, json_name, body, spawn_x, spawn_y, connections=None, max_steps=500):
        # 1. Load terrain from bundled JSON
        world = EvoWorld.from_json(data_path(json_name))
        # 2. Add the robot
        world.add_from_array("robot", body, spawn_x, spawn_y, connections=connections)
        # 3. Compile and instantiate
        template_set = compile_world_templates(world, robot_name="robot")
        built = instantiate_world(template_set.primary)
        # 4. Store all simulation data as instance attributes
        self.topology = built.topology
        self.constants = built.constants
        self.actuator_info = built.actuator_info
        self.collision_data = built.collision_data
        # ... etc
```

After construction, the base class exposes:

| Attribute | Description |
|-----------|-------------|
| `topology` | `SpringTopology` — spring connectivity |
| `constants` | `PhysicsConstants` — simulation parameters |
| `actuator_info` | `ActuatorInfo` — per-cell actuator mappings |
| `per_axis_info` | `PerAxisActuatorInfo` — for CONTRACTILE cells |
| `collision_data` | `CollisionData` — object-object collision geometry |
| `static_collider_data` | `StaticColliderData` — static terrain geometry |
| `deformation_info` | `DeformationInfo` — voxel-to-spring mapping |
| `robot_point_indices` | `(n_robot_points,)` int32 — robot point indices |
| `terrain_positions` | `(n, 2)` float32 — terrain sample positions for floor obs |
| `terrain_mask` | `(n,)` bool — valid terrain points |
| `n_robot_points` | Number of robot points |
| `n_robot_voxels` | Number of robot voxels |
| `n_actuators` | Number of actuator cells |
| `n_points` | Total dynamic point count |
| `render_info` | `RenderInfo` — static rendering metadata |

## Environment pattern

Every environment follows the same structure:

1. Call `super().__init__()` with terrain JSON and spawn position
2. Compute `obs_dim` from observation components
3. Build the initial env state in `__init__`
4. Implement `reset()` returning `(obs, init_state)`
5. Implement `step(state, action)` with `@jax.jit`

```python
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

from jax_evogym.types import EnvState, StepOutput
from jax_evogym.collision import is_self_colliding
from jax_evogym.sim import env_step
from jax_evogym.environments._base import EvoGymBaseEnv
from jax_evogym.environments._obs import (
    get_vel_com_obs, get_relative_pos_obs, get_deformation_obs,
)


class MyEnvV0(EvoGymBaseEnv):
    def __init__(self, body, connections=None, max_steps=500, include_deformation=True):
        super().__init__(
            "Walker-v0.json",  # terrain JSON
            body,
            spawn_x=1, spawn_y=1,
            connections=connections,
            max_steps=max_steps,
        )
        self.include_deformation = include_deformation
        self.obs_dim = 2 + 2 * self.n_robot_points
        if include_deformation:
            self.obs_dim += 2 * self.n_robot_voxels

        init_com_x = self._compute_com_x(self._init_sim_state)
        self._init_env_state = EnvState(
            sim_state=self._init_sim_state,
            step_count=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False),
            prev_com_x=init_com_x,
        )

    def reset(self):
        obs = self._compute_obs(self._init_sim_state)
        return obs, self._init_env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, env_state, action):
        sim_state = env_state.sim_state

        # Action clipping
        action = jnp.clip(action, 0.6, 1.6)
        action = jnp.where(jnp.abs(action) < 1e-8, 0.0, action)

        # Physics step
        new_sim_state = env_step(
            sim_state, self.topology, self.constants,
            action, self.actuator_info,
            collision_data=self.collision_data,
            static_collider_data=self.static_collider_data,
        )

        # Reward and done (customize these)
        new_com_x = self._compute_com_x(new_sim_state)
        reward = new_com_x - env_state.prev_com_x

        new_step_count = env_state.step_count + 1
        self_colliding = is_self_colliding(
            new_sim_state.positions, self.collision_data
        )
        done = self_colliding | (new_step_count >= self.max_steps)
        reward = jnp.where(self_colliding, reward - 3.0, reward)
        reward = jnp.where(env_state.done, 0.0, reward)
        done = env_state.done | done

        obs = self._compute_obs(new_sim_state)
        new_env_state = EnvState(
            sim_state=new_sim_state,
            step_count=new_step_count,
            done=done,
            prev_com_x=new_com_x,
        )
        return obs, new_env_state, reward, done

    def _compute_com_x(self, sim_state):
        return jnp.mean(sim_state.positions[self.robot_point_indices][:, 0])

    def _compute_obs(self, sim_state):
        parts = [
            get_vel_com_obs(sim_state.velocities_true, self.robot_point_indices),
            get_relative_pos_obs(sim_state.positions, self.robot_point_indices),
        ]
        if self.include_deformation:
            parts.append(get_deformation_obs(
                sim_state.positions, self.topology,
                sim_state.spring_init_rest_length, self.deformation_info,
            ))
        return jnp.concatenate(parts)
```

## Available observation helpers

See [Observation Helpers](../../reference/observation-helpers/) for full signatures.

| Function | Output shape | Description |
|----------|-------------|-------------|
| `get_vel_com_obs` | `(2,)` | Mean robot velocity |
| `get_relative_pos_obs` | `(2n,)` | Point positions relative to COM |
| `compute_orientation` | scalar | Angular displacement from initial config |
| `get_floor_obs` | `(2d+1,)` | Terrain elevations below COM |
| `get_deformation_obs` | `(2v,)` | Per-voxel spring strain |

## Bundled terrain JSONs

| File | Description |
|------|-------------|
| `Walker-v0.json` | 100-cell flat ground |
| `BridgeWalker-v0.json` | Soft bridge terrain |
| `Climber-v0.json` | Vertical tube with walls |
| `ShapeChange.json` | Walled enclosure |
| `Jumper-v0.json` | Flat ground (wider) |
| `UpStepper-v0.json` | Ascending staircase |

To create custom terrain, use the [Designer](../../designer/) to build and export EvoGym JSON, or construct an `EvoWorld` programmatically:

```python
from jax_evogym import EvoWorld, FIXED

world = EvoWorld()
world.add_from_array("ground", np.array([[FIXED, FIXED, FIXED, FIXED]]), 0, 0)
world.add_from_array("robot", body, 1, 2)
```

## Episode step factory

Create a `lax.scan`-compatible function for your environment:

```python
def make_my_episode_step(env):
    def episode_step(env_state, action):
        obs, new_env_state, reward, done = env.step(env_state, action)
        output = StepOutput(obs=obs, reward=reward, done=done,
                           positions=new_env_state.sim_state.positions)
        return new_env_state, output
    return episode_step
```

## Environment state types

Choose the state type that matches your reward computation:

| Type | Fields | Use when |
|------|--------|----------|
| `EnvState` | `prev_com_x` | Reward based on horizontal progress |
| `ShapeEnvState` | `prev_span` | Reward based on body span change |
| `OrientedEnvState` | `prev_com_x` + `initial_robot_positions` | Need orientation observation |
| `JumperEnvState` | `prev_com_x` + `prev_com_y` | Track both axes |
| `ClimberEnvState` | `prev_com_y` | Reward based on vertical progress |

Or define your own `NamedTuple` with a `sim_state` field — JAX pytree registration is automatic.
