---
title: Quick Start
description: A short introduction to world construction, environment stepping, and the builder API.
---

After [installation](../installation/), use these three small examples to understand the Python API. Each code block is self-contained. For the relationship between worlds, state, and environments, read [Concepts](../concepts/).

## Create a world

```python
import numpy as np
from jax_evogym import EvoWorld, H_ACT, SOFT, V_ACT

world = EvoWorld()
world.add_from_array("robot", np.array([[H_ACT, SOFT, V_ACT]]), 0, 0)
world.to_json("robot.json")
```

`EvoWorld` stores the world definition in NumPy. This example does not run a simulation; the package installation still includes JAX.

## Run an environment

```python
import jax.numpy as jnp
import numpy as np
from jax_evogym import H_ACT, SOFT, V_ACT, WalkerV0

body = np.array([[H_ACT, SOFT, V_ACT]])
env = WalkerV0(body)

obs, state = env.reset()
action = jnp.ones(env.n_actuators)
obs, state, reward, done = env.step(state, action)
```

Environments handle the full build pipeline internally: compile the world template, instantiate the built world, and expose a JIT-compiled `step` function.

## Use the builder API directly

```python
import numpy as np
from jax_evogym import EvoWorld, FIXED, H_ACT, SOFT, V_ACT
from jax_evogym import compile_world_template, instantiate_world

world = EvoWorld()
world.add_from_array("ground", np.array([[FIXED, FIXED, FIXED, FIXED]]), 0, 0)
world.add_from_array("robot", np.array([[H_ACT, SOFT, V_ACT]]), 1, 2)

template = compile_world_template(world, robot_name="robot")
built = instantiate_world(template)

print(built.sim_state.positions.shape)
print(built.static_collider_data.point_positions.shape)
```

The builder API gives you direct access to `SimState`, `CollisionData`, `ActuatorInfo`, and all other simulation primitives. Use this when you need control beyond what the built-in environments provide.

## Where to go next

Continue to [First Simulation](../first-simulation/) to run a multi-step simulation and inspect an animation. Then choose a task:

- [Designer to Simulation](../../guides/designer-world/) — load an exported world.
- [Environments](../../guides/environments/) — choose a built-in task.
- [Controllers and Actions](../../guides/controllers/) — supply a policy and handle episode completion.

Keep [Voxel Types](../../reference/voxel-types/) nearby as a reference. The [research example](../../research/) shows how these pieces support a larger experiment.
