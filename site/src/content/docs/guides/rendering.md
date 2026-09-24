---
title: Rendering
description: Generate GIFs from simulation rollouts using the PIL-based renderer.
---

For a complete runnable example, start with [First Simulation](../../getting-started/first-simulation/).
Rendering happens after simulation: record positions and spring lengths, convert them to frames, then save a GIF. It runs outside the JIT-compiled physics loop.

When evaluating a population, compute fitness first and replay only selected candidates for rendering. This avoids storing full trajectories for every candidate.

## RenderConfig

All rendering options are controlled by `RenderConfig`:

```python
from jax_evogym import RenderConfig

config = RenderConfig(
    enabled=True,
    width=600,
    height=300,
    fps=50,
    camera_mode="fit_robot",
)
```

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `False` | Caller-side preference; rendering functions do not check this flag |
| `width` | `600` | Output pixel width |
| `height` | `300` | Output pixel height |
| `fps` | `50` | Target frame rate |
| `real_time` | `True` | Match physics timing (auto-computes frame skip) |
| `frame_skip` | `7` | Manual frame skip (used when `real_time=False`) |
| `show_grid` | `False` | Draw grid lines |
| `show_edges` | `True` | Draw surface edges |
| `dynamic_actuator_colors` | `True` | Color actuators by current actuation level |
| `camera_padding` | `0.3` | World units of padding around robot |
| `viewport_width` | `4.0` | World units visible horizontally |
| `camera_mode` | `"fit_robot"` | Camera tracking mode |
| `camera_smoothing` | `0.18` | Camera lerp factor (0=static after initial placement, 1=instant) |
| `supersample` | `2` | Supersampling factor for antialiasing |
| `output_dir` | `"renders"` | Directory for saved GIFs |
| `target_frames` | `None` | Optional frame budget cap |
| `max_colors` | `256` | GIF palette size |
| `optimize` | `False` | GIF frame optimization |
| `dither` | `True` | Floyd-Steinberg dithering |

### Camera modes

- **`"fit_robot"`**: viewport expands to fit the robot with padding, never smaller than `viewport_width`
- **Fixed**: any other value uses `viewport_width` as-is, centered on robot COM

## End-to-end example

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax_evogym import (
    H_ACT, SOFT, V_ACT, WalkerV0, RenderConfig,
    make_render_episode_step, render_episode, save_gif,
)

# 1. Create environment
body = np.array([[H_ACT, SOFT, V_ACT]])
env = WalkerV0(body)
obs, init_state = env.reset()

# 2. Run render pass (captures positions + spring state)
render_step = make_render_episode_step(env)
actions = jnp.ones((500, env.n_actuators))  # neutral targets, not a walking policy
final_state, outputs = jax.lax.scan(render_step, init_state, actions)

# 3. Render and save
config = RenderConfig(enabled=True)
frames = render_episode(outputs, env.render_info, config)
save_gif(frames, "walker.gif", config)
```

## Core functions

### make_render_episode_step

```python
render_step = make_render_episode_step(env)
```

Returns a `lax.scan`-compatible step function that captures `RenderStepOutput` (positions, spring rest lengths, reward, done) at each step. Works with built-in environments using the standard state fields. It records `done` but does not freeze finished states; use the helper in [First Simulation](../../getting-started/first-simulation/) when that behavior is required.

### render_episode

```python
frames = render_episode(render_outputs, render_info, config)
```

Converts a batch of `RenderStepOutput` from `lax.scan` into a list of PIL Image frames. Handles frame skipping, camera tracking, viewport culling, and dynamic actuator coloring.

### render_frame

```python
img, camera_pos = render_frame(positions_np, render_info, config, ...)
```

Renders a single frame. Lower-level than `render_episode` — you handle frame iteration yourself. Returns the PIL Image and updated camera position for smooth tracking.

### save_gif

```python
path = save_gif(frames, "output.gif", config)
```

Quantizes frames to a shared palette and saves as animated GIF. Uses `config.frame_duration_ms`, `config.max_colors`, `config.dither`, and `config.optimize`.

## Batch rendering

### render_top_k

```python
paths = render_top_k(
    batched_render_outputs,  # (n_steps, pop_size, ...)
    rewards,                 # (pop_size,)
    render_info, config,
    k=3, generation=0,
)
```

Renders GIFs for the top-K individuals from a batched evaluation. Expects time-major batched outputs from a vmapped render pass.

### rerun_top_k

```python
paths = rerun_top_k(
    env, init_state, all_actions, rewards,
    k=3, config=config, generation=0,
)
```

Higher-level: selects top-K by reward, reruns each individually with `make_render_episode_step`, renders, and saves. Use this when you only have the fitness pass outputs and need to rerun for render data.

## Dynamic actuator coloring

When `dynamic_actuator_colors=True` (default), actuator cells are colored by their current spring rest-length ratio relative to the initial rest length:

- Ratio near `0.6` (compressed): one end of the gradient
- Ratio near `1.6` (extended): other end of the gradient
- `H_ACT`: cream to burnt orange gradient
- `V_ACT`: light blue to dark blue gradient
- `CONTRACTILE`: light green to dark green gradient

Non-actuator cells use static colors (`RIGID`: dark charcoal, `SOFT`: light gray, `FIXED`: medium gray).

## Static terrain rendering

Static terrain (`FIXED` cells) is rendered from `RenderInfo.static_point_positions` — true static geometry, not dynamically simulated points. This matches the object-separated architecture where static terrain is excluded from `SimState`.
