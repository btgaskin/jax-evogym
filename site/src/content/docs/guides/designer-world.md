---
title: From Designer to Simulation
description: Export a world, choose its robot, and render a Python rollout.
---

The [browser designer](/designer/) creates geometry and connectivity. Python runs the physics. The browser does not train controllers or preview a physical rollout.

## Prepare and export

1. Create an object named **robot**. Draw a connected body with at least one **H-Act** or **V-Act** cell.
2. Create a separate terrain object and draw **Fixed** ground below the body. Leave space for the robot to move.
3. Validate the world, resolve errors, and export `evogym-world.json`.

This introductory script uses scalar actions. It rejects **Contractile** cells because they need the [per-axis control path](../controllers/#contractile-and-per-axis-control).

Spawn, target, and direction markers are **design annotations**. They do not move the robot, define a reward, or configure the simulation. Initial position comes from the cells' world coordinates. Keep your original export: the Python loader ignores metadata, and a Python JSON round trip drops it.

## Run the exported world

From an installed source checkout with the visualization extra:

```bash
python -m examples.designer_world evogym-world.json --robot robot --output /tmp/world.gif
```

Use the exact object name from the designer. The script lists available names if it cannot find the robot. It loads the complete world, compiles a template, instantiates state, applies periodic actions, and saves a GIF. Static and dynamic collision data are passed to every simulation step.

This is a physics-only rollout: its recorded reward is zero and it defines no task completion condition. To evaluate a controller, define observations, a reward, and termination in a [custom environment](../custom-environments/).

A checked-in fixture is available for testing the path without drawing first:

```bash
python -m examples.designer_world examples/world.json --output /tmp/world.gif
```

## Use only the body in a benchmark

```python
from jax_evogym import EvoWorld, WalkerV0

world = EvoWorld.from_json("evogym-world.json")
robot = world.objects["robot"]
env = WalkerV0(robot.get_structure(), connections=robot.get_connections())
obs, state = env.reset()
```

This uses the robot's local structure and connectivity. `WalkerV0` supplies its own terrain and placement; your exported terrain and markers do not configure that benchmark.

See [`examples/designer_world.py`](https://github.com/btgaskin/jax-evogym/blob/main/examples/designer_world.py), the [JSON reference](../../reference/world-json/), and the [designer guide](../../designer/designer-guide/) for details.
