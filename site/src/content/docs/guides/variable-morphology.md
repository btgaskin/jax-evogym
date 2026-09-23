---
title: Variable Morphology
description: Vary robot design at runtime using robot_override and the dense grid pipeline.
---

jax-evogym supports two mechanisms for varying robot morphology:

1. **`robot_override`** — swap the robot's body within a fixed template frame
2. **Dense grid pipeline** — build simulation data from scratch in pure JAX, vmap-able across populations

## robot_override

Compile a template once, then instantiate it with different morphologies:

```python
import numpy as np
from jax_evogym import (
    EvoWorld, FIXED, H_ACT, SOFT, V_ACT, CONTRACTILE, EMPTY, RIGID,
    compile_world_template, instantiate_world,
)

# Compile once
world = EvoWorld()
world.add_from_array("ground", np.array([[FIXED, FIXED, FIXED, FIXED]]), 0, 0)
world.add_from_array("robot", np.array([[H_ACT, SOFT, V_ACT]]), 1, 2)
template = compile_world_template(world, robot_name="robot")

# Instantiate with different bodies
dense = np.array([[H_ACT, SOFT, V_ACT]])
sparse = np.array([[H_ACT, EMPTY, V_ACT]])

dense_built = instantiate_world(template, robot_override=dense)
sparse_built = instantiate_world(template, robot_override=sparse)
```

### Rules

- Override shape must match the robot template grid exactly
- Same array orientation as `add_from_array`
- `EMPTY` disables cells within the frame
- Any dynamic voxel type is allowed (`RIGID`, `SOFT`, `H_ACT`, `V_ACT`, `CONTRACTILE`)
- Slope values are rejected
- Terrain objects are never affected

### With mirroring

`instantiate_world` does **not** auto-mirror `robot_override`. Pass the override explicitly to each template:

```python
from jax_evogym import compile_world_templates

template_set = compile_world_templates(world, robot_name="robot", mirror_mode="paired")

built_primary = instantiate_world(template_set.primary, robot_override=dense)
built_mirror = instantiate_world(template_set.mirror, robot_override=dense)
```

If you want a truly mirrored body, construct the mirrored array yourself.

## Dense grid pipeline

For evolutionary search where you need to evaluate thousands of morphologies in parallel, the dense grid pipeline provides pure JAX, vmap-able construction:

```python
import jax.numpy as jnp
from jax_evogym import EMPTY, H_ACT, SOFT, V_ACT
from jax_evogym import precompute_grid, jax_build_sim_state, jax_build_collision_data

# Dense grids use y-up rows. This example contains only a robot.
body_array = jnp.array([[H_ACT, SOFT, V_ACT]], dtype=jnp.int32)
grid_data = precompute_grid(H=1, W=3)
robot_mask = body_array != EMPTY

sim_state, topology = jax_build_sim_state(body_array, grid_data)
collision_data = jax_build_collision_data(body_array, robot_mask, grid_data)
```

See [Dense Grid Pipeline](../../guides/dense-grid/) for the full guide.

## When to use which

| Scenario | Use |
|----------|-----|
| Standard environment usage | Built-in environments (handle everything internally) |
| Fixed terrain, varying robot body | `robot_override` on a compiled template |
| Terrain with multiple objects | Object-separated build (`compile_world_template`) |
| Population-level morphology search with vmap | Dense grid pipeline |
| Custom terrain + custom morphology | Build the `EvoWorld` programmatically, compile, instantiate |
