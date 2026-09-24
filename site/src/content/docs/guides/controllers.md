---
title: Controllers and Actions
description: Map observations or controller state to simulator actions.
---

This guide assumes you can run [First Simulation](../../getting-started/first-simulation/) and have chosen an [environment](../environments/) or custom world.

A controller chooses actions. The simulator applies those actions and returns the next state. Evolution, reinforcement learning, and hand-written policies can use the same interface; they supply their own parameter updates.

## Scalar actions in built-in environments

```python
import jax.numpy as jnp
from jax_evogym import WalkerV0
from examples.common import BODY

env = WalkerV0(BODY)
obs, state = env.reset()
action = jnp.ones((env.n_actuators,), dtype=jnp.float32)
obs, state, reward, done = env.step(state, action)
```

`reset()` returns `(observation, state)` and `step()` returns `(observation, next_state, reward, done)`. This is an explicit-state JAX interface, not the Gymnasium tuple API. Reset the controller's hidden state separately when starting a new episode.

| Quantity | Meaning |
|---|---|
| Observation | One environment-specific vector; inspect `obs.shape` and the [environment guide](../environments/). |
| Action | Shape `(env.n_actuators,)`, one value per `H_ACT` or `V_ACT` cell. |
| Value `1.0` | Rest-length target at the original size; it does not freeze the body. |
| Values below/above `1.0` | Contraction/extension along that cell's active axis. |
| Range | Built-in scalar stepping clips to `[0.6, 1.6]`. |
| Reward | Scalar task reward for the transition; sum over a declared evaluation horizon. |
| `done` | Sticky completion flag, including the environment time limit. No automatic reset. |

Actions follow occupied actuator cells in row-major order: increasing x within each y row. Array row zero is the bottom of the body, with y increasing upward. Passive and empty cells occupy no scalar action slot. Shared actuated edges average their contributing targets.

A feed-forward controller can map `obs` to logits, then use `0.6 + jax.nn.sigmoid(logits)` for bounded actions. Its output width must equal `env.n_actuators`. For a recurrent controller, carry `(environment_state, hidden_state, random_key)` through `lax.scan`. Split random keys explicitly when using stochastic actions. Freeze or reset all three consistently at episode boundaries.

## Handle completed episodes

Built-in environments retain `done` and return zero reward on subsequent steps, but physics continues. A fixed-length rollout should mask those steps or reset them deliberately. [`examples/common.py`](https://github.com/btgaskin/jax-evogym/blob/main/examples/common.py) freezes the complete state after completion and retains the terminal transition's reward. Do not mask that reward using the new `done` flag.

## Contractile and per-axis control

`CONTRACTILE` cells have independent horizontal and vertical targets. They are not included in the built-in scalar action vector. Use the object builder's `built.per_axis_info` with `set_per_axis_goals`, then advance `physics_substep`. Calling scalar `env_step` afterward would overwrite overlapping actuator goals.

```python
import jax
import jax.numpy as jnp
from jax_evogym import set_per_axis_goals, physics_substep
from jax_evogym.constants import PHYSICS_UPDATES_PER_STEP

# built is an instantiated world, as in the Designer to Simulation guide.
def per_axis_step(state, h_action, v_action):
    state = set_per_axis_goals(
        state, built.per_axis_info,
        jnp.clip(h_action, 0.6, 1.6), jnp.clip(v_action, 0.6, 1.6),
    )
    return jax.lax.fori_loop(
        0, PHYSICS_UPDATES_PER_STEP,
        lambda _, s: physics_substep(
            s, built.topology, built.constants,
            collision_data=built.collision_data,
            static_collider_data=built.static_collider_data,
        ), state,
    )
```

Each action array has one entry per cell in the named robot's bounding grid, including empty cells. Disabled axes are ignored. This low-level setter does not clip for you; the wrapper above does. Observations, rewards, controller state, and episode limits remain the caller's responsibility. The compact variant in `jax_evogym.actuators` uses active-axis ordering with padded arrays; use its index maps and counts rather than assuming scalar ordering.

Start with the [periodic example](../../getting-started/first-simulation/), then [compare fixed controllers](../batched-evaluation/). [Sensing at the Edges](../../research/) illustrates a more complex distributed recurrent controller.
