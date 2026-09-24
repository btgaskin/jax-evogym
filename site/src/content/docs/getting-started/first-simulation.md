---
title: First Simulation
description: Run a multi-step simulation, inspect its result, and save an animation.
---

The [Quick Start](../quickstart/) introduces world construction and one environment step. This walkthrough connects those pieces into a complete rollout workflow: choose actions, advance the simulator, check the result, and save an animation.

Use the [source checkout](../installation/) with the visualization extra installed. Run commands from the repository root:

```bash
python -m pip install -e ".[viz]"
python -m examples.first_motion --steps 300 --output /tmp/first-motion.gif
```

Open `/tmp/first-motion.gif`. You should see a three-cell body deform above flat ground. The command prints the frame count and total reward. It checks that positions are finite before saving. Choose a different output path on Windows.

This example demonstrates actuation and rendering. Its sinusoidal controller is not a trained walking policy, and forward travel is not guaranteed.

## What runs

1. `WalkerV0` builds a world around a body containing horizontal and vertical actuators.
2. `reset()` returns the initial observation and explicit environment state.
3. A periodic controller produces one multiplier per actuator, centred on `1.0`.
4. `jax.lax.scan` applies the actions in sequence and records positions and spring lengths.
5. `render_episode` draws the saved trajectory; `save_gif` writes the animation.

The default 300 environment steps represent 0.9 seconds of simulated time: each step contains 30 physics updates of 0.0001 seconds. GIF playback settings can change the apparent speed. The first run includes JAX compilation, so it is not a useful throughput measurement.

The shared rollout helper keeps the terminal transition and freezes the state on later steps. Built-in environments otherwise keep advancing physics after `done` becomes true.

Read [`examples/first_motion.py`](https://github.com/btgaskin/jax-evogym/blob/main/examples/first_motion.py) and [`examples/common.py`](https://github.com/btgaskin/jax-evogym/blob/main/examples/common.py) for the complete implementation. Next, [load your own design](../../guides/designer-world/) or [write a controller](../../guides/controllers/).
