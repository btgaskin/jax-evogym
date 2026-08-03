---
title: Environments
description: All 7 built-in environments with observations, rewards, and done conditions.
---

All environments share a common interface:

```python
env = WalkerV0(body)
obs, state = env.reset()
obs, state, reward, done = env.step(state, action)
```

Actions are clipped to `[0.6, 1.6]` (60%-160% of rest length), so smaller values become `0.6` and larger values become `1.6`. Each `step` runs 30 physics substeps via `env_step`.

## Summary

| Environment | Task | Terrain | Reward | State Type | Default Steps |
|-------------|------|---------|--------|------------|---------------|
| `WalkerV0` | Walk right | Flat ground | delta COM x | `EnvState` | 500 |
| `BridgeWalkerV0` | Walk across bridge | Soft bridge | delta COM x | `OrientedEnvState` | 500 |
| `ClimberV0` | Climb upward | Vertical tube | delta COM y | `ClimberEnvState` | 400 |
| `HeightMaximizerV0` | Maximize height span | Walled arena | delta height | `ShapeEnvState` | 500 |
| `WingspanMaximizerV0` | Maximize width span | Walled arena | delta width | `ShapeEnvState` | 600 |
| `JumperV0` | Jump vertically | Flat ground | 10dy - 5\|dx\| | `JumperEnvState` | 500 |
| `UpStepperV0` | Traverse stairs | Ascending stairs | delta COM x | `OrientedEnvState` | 600 |

All environments accept `include_deformation=True` (default), which appends `2 * n_robot_voxels` deformation observations to the observation vector.

## WalkerV0

Walk as far right as possible on flat ground.

- **Terrain**: `Walker-v0.json` — 100-cell flat ground
- **Spawn**: (1, 1)
- **Observation**: `vel_com(2) + rel_pos(2n) [+ deform(2v)]`
- **Reward**: `new_com_x - prev_com_x`
- **Done conditions**:
  - Self-collision: `-3.0` penalty, episode ends
  - Goal reached (`com_x > 9.9`): `+1.0` bonus, episode ends
  - Max steps (500)
- **Factory**: `make_walker_episode_step(env)`

## BridgeWalkerV0

Walk across a bridge made of soft terrain.

- **Terrain**: `BridgeWalker-v0.json` — bridge of SOFT voxels
- **Spawn**: (2, 5)
- **Observation**: `vel_com(2) + orientation(1) + rel_pos(2n) [+ deform(2v)]`
- **Reward**: `new_com_x - prev_com_x`
- **Done conditions**:
  - Self-collision: `-3.0` penalty
  - Goal reached (`com_x > 6.0`): `+1.0` bonus
  - Max steps (500)
- **State**: `OrientedEnvState` — stores `initial_robot_positions` for orientation computation
- **Factory**: `make_bridge_walker_episode_step(env)`

Orientation is a weighted angular displacement from the initial configuration, computed via `compute_orientation`. See [Observation Helpers](../../reference/observation-helpers/).

## ClimberV0

Climb upward inside a vertical tube.

- **Terrain**: `Climber-v0.json` — tall vertical tube with walls
- **Spawn**: (1, 1)
- **Observation**: `vel_com(2) + rel_pos(2n) [+ deform(2v)]`
- **Reward**: `new_com_y - prev_com_y`
- **Done conditions**:
  - Self-collision: `-3.0` penalty
  - Goal reached (`com_y > 8.6`): `+1.0` bonus
  - Max steps (400)
- **Factory**: `make_climber_episode_step(env)`

## HeightMaximizerV0

Maximize the vertical span (y_max - y_min) of the robot body.

- **Terrain**: `ShapeChange.json` — walled enclosure
- **Spawn**: (7, 1)
- **Observation**: `rel_pos(2n) [+ deform(2v)]`
- **Reward**: `new_span - prev_span` where `span = max(y) - min(y)` of robot points
- **Done conditions**:
  - Self-collision: `-3.0` penalty
  - Max steps (500)
- **Factory**: `make_shape_episode_step(env)`

No velocity or orientation observations — the task depends only on body shape.

## WingspanMaximizerV0

Maximize the horizontal span (x_max - x_min) of the robot body.

- **Terrain**: `ShapeChange.json` — walled enclosure
- **Spawn**: (7, 1)
- **Observation**: `rel_pos(2n) [+ deform(2v)]`
- **Reward**: `new_span - prev_span` where `span = max(x) - min(x)` of robot points
- **Done conditions**:
  - Self-collision: `-3.0` penalty
  - Max steps (600)
- **Factory**: `make_shape_episode_step(env)`

## JumperV0

Jump as high as possible with minimal lateral drift.

- **Terrain**: `Jumper-v0.json` — flat ground
- **Spawn**: (32, 1)
- **Observation**: `vel_com(2) + rel_pos(2n) + floor_obs(5) [+ deform(2v)]`
- **Reward**: `10 * delta_y - 5 * |delta_x|`
- **Done conditions**:
  - Self-collision: `-3.0` penalty
  - Max steps (500)
- **Floor observation**: `sight_dist=2`, producing 5 columns of terrain elevation relative to COM
- **Factory**: `make_jumper_episode_step(env)`

The reward strongly favours vertical displacement and penalises lateral drift.

## UpStepperV0

Traverse ascending stairs, staying upright.

- **Terrain**: `UpStepper-v0.json` — ascending staircase
- **Spawn**: (1, 1)
- **Observation**: `vel_com(2) + orientation(1) + rel_pos(2n) + floor_obs(11) [+ deform(2v)]`
- **Reward**: `new_com_x - prev_com_x`
- **Done conditions**:
  - Self-collision: `-3.0` penalty
  - Goal reached (`com_x > 6.9`): `+2.0` bonus
  - Flipped (orientation in [~1.31, ~4.97] radians): `-3.0` penalty
  - Max steps (600)
- **Floor observation**: `sight_dist=5`, producing 11 columns of terrain elevation
- **State**: `OrientedEnvState` — stores initial positions for orientation and flip detection
- **Factory**: `make_upstepper_episode_step(env)`

The flip detection uses orientation thresholds of `pi/2 - pi/12` (~1.309) and `3*pi/2 + pi/12` (~4.974).

## Episode rollouts with lax.scan

Each environment has a companion factory function that returns a `lax.scan`-compatible step function:

```python
from jax_evogym import WalkerV0, make_walker_episode_step
import jax
import jax.numpy as jnp

env = WalkerV0(body)
obs, init_state = env.reset()

episode_step = make_walker_episode_step(env)
actions = jnp.ones((500, env.n_actuators))  # 500 steps of constant action

final_state, outputs = jax.lax.scan(episode_step, init_state, actions)
# outputs.reward: (500,), outputs.positions: (500, n_points, 2)
```

The `StepOutput` contains `obs`, `reward`, `done`, and `positions` at each step.

## Constructor parameters

All environments accept:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `body` | `np.ndarray` | required | 2D morphology array using voxel type constants |
| `connections` | `np.ndarray \| None` | `None` | Optional connectivity override |
| `max_steps` | `int` | varies | Episode length before truncation |
| `include_deformation` | `bool` | `True` | Append deformation observations |
