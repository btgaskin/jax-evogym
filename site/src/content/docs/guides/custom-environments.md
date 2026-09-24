---
title: Custom Environments
description: Build your own environments using the base class and observation helpers.
---

Use this guide when the [built-in environments](../environments/) do not express your task. It assumes you understand the [controller and action interface](../controllers/). Start by changing one reward before defining a new state or physics workflow.

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

## Start by changing one part of an existing task

If you only need a different reward, subclass an existing environment. Keep
its observations, state, and physics so the change is easy to inspect.
The runnable implementation is
[`examples/custom_environment.py`](https://github.com/btgaskin/jax-evogym/blob/main/examples/custom_environment.py).

```python
import jax
import jax.numpy as jnp
from jax_evogym import WalkerV0

@jax.jit
def penalize_effort(reward, action, already_done, coefficient):
    clipped = jnp.clip(action, 0.6, 1.6)
    effort = jnp.sum(jnp.square(clipped - 1.0))
    return jnp.where(already_done, 0.0, reward - coefficient * effort)

class EffortWalker(WalkerV0):
    def __init__(self, body, effort_coefficient=0.001, **kwargs):
        if effort_coefficient < 0:
            raise ValueError("effort_coefficient must be non-negative")
        super().__init__(body, **kwargs)
        self.effort_coefficient = effort_coefficient

    def step(self, state, action):
        obs, next_state, reward, done = super().step(state, action)
        reward = penalize_effort(reward, action, state.done, self.effort_coefficient)
        return obs, next_state, reward, done
```

The coefficient is a task-design choice, not a simulator default. The terminal
transition still receives a reward; calls after termination receive zero.

JIT helpers take runtime arrays as arguments. Avoid making the environment
instance a static JIT argument: it holds substantial array data and changing
its attributes can interact poorly with compilation caches. Keep only actual
compile-time choices, such as observation structure, static.

## A new task or custom world

For a different state or observation contract:

1. Build the world once outside the rollout.
2. Define a JAX pytree holding simulation state and reward/termination bookkeeping.
3. Implement `reset()` returning `(obs, state)`.
4. Pass state, actions, and simulation arrays into a module-level JIT step helper.
5. Return `(obs, next_state, reward, done)` and specify what happens after `done`.

`EvoGymBaseEnv` loads terrain through the package's `data_path(json_name)`.
For an exported designer world, use the explicit builder workflow in
[Your designed world](../designer-world/) instead of copying a JSON file into
the installed package. The low-level builder supplies physics; your task still
needs to define observations, reward, and termination.

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
import numpy as np
from jax_evogym import EvoWorld, FIXED, H_ACT, V_ACT

body = np.array([[H_ACT, V_ACT]])

world = EvoWorld()
world.add_from_array("ground", np.array([[FIXED, FIXED, FIXED, FIXED]]), 0, 0)
world.add_from_array("robot", body, 1, 2)
```

## Episode step factory

Create a `lax.scan`-compatible function for your environment:

```python
from jax_evogym import StepOutput

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
