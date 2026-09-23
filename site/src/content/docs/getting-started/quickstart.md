---
title: Quick Start
description: Three progressive examples to get running with jax-evogym.
---

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

- [Concepts](../../getting-started/concepts/) — how the pieces fit together
- [Environments](../../guides/environments/) — all 7 built-in environments
- [Rendering](../../guides/rendering/) — generate GIFs from rollouts
- [Voxel Types](../../reference/voxel-types/) — the full cell type reference
