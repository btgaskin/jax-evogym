# `jax-evogym`

## What This Library Is

`jax-evogym` is a JAX-native EvoGym core library for soft-body robot simulation, world construction, rendering, and parity validation. The runtime is now built around **object-separated simulation** rather than a single composited robot-plus-terrain grid.

Core guarantees:

- robot and terrain are compiled as separate objects
- pure fixed terrain is static collider/render geometry, not dynamic simulation state
- `FIXED` quads are the supported static terrain primitive
- soft or mixed soft+fixed terrain remains dynamic
- there are no shared robot-terrain simulation points
- canonical corrected validation uses `30` physics substeps per env step

Slope terrain code is retained in the repository as an unsupported experimental
geometry path, but it is not part of the stable public API or release contract.

## Project Status

This repository is a public-alpha community research library. Its stable
contract is documented in the
[API boundary](https://jax-evogym.pages.dev/reference/api-boundary/),
with release priorities and known limitations tracked in
[ROADMAP.md](ROADMAP.md).

Community-facing process files are intentionally lightweight:

- [CONTRIBUTING.md](CONTRIBUTING.md): local checks, CI gates, and physics-change expectations
- [CHANGELOG.md](CHANGELOG.md): user-facing release notes
- [SECURITY.md](SECURITY.md): vulnerability reporting scope
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md): contributor expectations

## Core Simulation Model

Worlds are compiled object-by-object into reusable templates. Dynamic robot state, dynamic terrain state, static fixed terrain collider geometry, collision metadata, and render metadata are all built from those templates rather than reconstructed from a merged occupancy grid.

The important rules are:

- **Robot and terrain are separate objects.**
- **Pure static-terrain non-robot objects are static-only.**
  They contribute collider geometry, render geometry, and terrain observation samples, but they do not contribute dynamic points, masses, velocities, or springs to `SimState`.
- **Stable static terrain uses `FIXED` cells only.**
  Experimental slope cells still exist behind `allow_experimental_slopes=True`, but they are deliberately excluded from the stable builder contract. Sloped edges are geometry-only: normal contact may be inspected, while slope friction, support, grip, and stiction are disabled.
- **Mixed or soft non-robot objects remain dynamic.**
  This preserves anchored-soft tasks such as bridge-like terrain.
- **Robot ownership is explicit.**
  Robot points, actuators, deformation metadata, and render ownership come from object identity, not composited point reuse.

## Install

Base package:

```bash
pip install .
```

With visualization support:

```bash
pip install ".[viz]"
```

For local development:

```bash
pip install -e ".[viz,dev]"
```

If you use `uv`, the equivalent editable install is:

```bash
uv pip install -e ".[viz,dev]"
```

`Pillow` is required for renderer output. It is included in `.[viz]`.

## JAX Version Policy

This repository is pinned to:

- `jax==0.9.0.1`
- `jaxlib==0.9.0.1`

Reasoning:

- `0.9.1` is currently avoided because of an open upstream compile-time regression: [`jax-ml/jax#35958`](https://github.com/jax-ml/jax/issues/35958)
- the CUDA runtime still has relevant open upstream issues on the pinned line, so we optimize for the version we have already validated rather than chasing latest

Upstream JAX issues we currently track as relevant to this simulator:

- [`jax-ml/jax#35667`](https://github.com/jax-ml/jax/issues/35667): GPU runtime failures with cached HLO configs. This is the main remaining operational risk on the pinned version.
- [`jax-ml/jax#35762`](https://github.com/jax-ml/jax/issues/35762): incorrect GPU reduction of closure-captured all-`True` boolean constants under `jit`.
- [`jax-ml/jax#17844`](https://github.com/jax-ml/jax/issues/17844): severe `scatter_add` slowdown on GPU when deterministic ops are enabled.
- [`jax-ml/jax#35994`](https://github.com/jax-ml/jax/issues/35994): GPU OOMs are not always propagated cleanly to Python.

Mitigations implemented in this repo:

- environment `step()` methods now route runtime arrays through module-level JIT helpers instead of capturing large runtime structs through `@jax.jit(static_argnums=(0,))`
- collision metadata stores exact active-triple and robot-surface counts, and hot collision paths use those counts instead of reducing closure-captured boolean masks when an exact count is already known

If you change JAX versions, rerun the parity suite in [`tests`](tests) before trusting any parity or stability result.

## Quick Start

### World creation

```python
import numpy as np

from jax_evogym import EvoWorld, H_ACT, SOFT, V_ACT

world = EvoWorld()
world.add_from_array("robot", np.array([[H_ACT, SOFT, V_ACT]]), 0, 0)
world.to_json("robot.json")
```

### Standard environment usage

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

### Builder API

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

## Templates, Mirroring, and Variable Morphology

The core builder exposes both single-template and paired-template workflows:

- `compile_world_template(world, robot_name="robot")`
- `compile_world_templates(world, robot_name="robot", mirror_mode="paired")`
- `instantiate_world(template, robot_override=None)`

Paired mirroring:

```python
import numpy as np

from jax_evogym import EvoWorld, FIXED, H_ACT, SOFT, V_ACT
from jax_evogym import compile_world_templates, instantiate_world

world = EvoWorld()
world.add_from_array("floor", np.array([[FIXED, FIXED, FIXED, FIXED, FIXED]]), 0, 0)
world.add_from_array("robot", np.array([[H_ACT, SOFT, V_ACT]]), 1, 2)

template_set = compile_world_templates(world, robot_name="robot", mirror_mode="paired")

primary_built = instantiate_world(template_set.primary)
mirror_built = instantiate_world(template_set.mirror)
```

`WorldTemplateSet` contains:

- `primary`
- `mirror`
- `mirror_mode`

Mirrored template sets are intended to be evaluated **within the same genome** when you want paired primary/mirrored terrain semantics.

### Variable morphology with `robot_override`

`robot_override` lets you vary the robot’s morphology inside a fixed robot frame:

```python
import numpy as np

from jax_evogym import CONTRACTILE, EMPTY, H_ACT, RIGID, SOFT, V_ACT
from jax_evogym import compile_world_template, instantiate_world

template = compile_world_template(world, robot_name="robot")

dense = np.array([
    [H_ACT, SOFT, V_ACT],
    [SOFT, CONTRACTILE, RIGID],
])

sparse = np.array([
    [H_ACT, EMPTY, V_ACT],
    [EMPTY, CONTRACTILE, EMPTY],
])

dense_built = instantiate_world(template, robot_override=dense)
sparse_built = instantiate_world(template, robot_override=sparse)
```

Important caveat:

- `robot_override` varies morphology **inside a fixed robot frame only**
- `instantiate_world` does **not** auto-mirror `robot_override`
- if you want mirrored genome evaluation, you instantiate both templates explicitly

## Terrain Semantics

The runtime distinguishes terrain by object classification, not by a composited grid pass:

- **Pure `FIXED` non-robot object**: static collider/render geometry only
- **Mixed or soft non-robot object**: dynamic object

This preserves bridge-like and anchored-soft tasks without forcing flat ground, walls, and fixed stairs into the dynamic simulation state.

Unsupported slope note:

- slope constants remain in `jax_evogym.constants`
- legacy slope fixtures live under `tests/experimental_slope_fixtures`
- public top-level imports intentionally do not export slope constants
- `compile_world_template(...)` and `compile_world_templates(...)` reject slope cells by default
- pass `allow_experimental_slopes=True` only for internal slope investigations and legacy fixtures
- slope friction/support controls are compatibility no-ops; flat ground and flat static-terrain friction remain supported

## Rendering

Rendering now consumes the same ownership-aware geometry that the simulation uses:

- dynamic positions come from `SimState`
- static terrain is appended from `RenderInfo.static_point_positions`
- cells are labeled by object ownership and dynamic/static status

This means fixed terrain is rendered from true static geometry instead of from dynamically simulated terrain points.

The renderer still reads its visual token document from [`src/jax_evogym/design_tool`](src/jax_evogym/design_tool).

## Docs Site + Designer

The browser designer now lives inside the Astro/Starlight docs site under [`site`](site).

```bash
bun --cwd site install
bun --cwd site dev
```

Build the static site:

```bash
bun --cwd site build
```

The designer route is served at `/designer` during local development and in the static build. It imports and exports canonical EvoGym world JSON directly in the browser.

Designer note: the public designer palette is limited to the stable voxel types (`RIGID`, `SOFT`, `FIXED`, `H_ACT`, `V_ACT`, `CONTRACTILE`). Slopes are omitted because they remain experimental and still need a defensible friction/support model before becoming stable terrain.

Core docs migrated into the site currently include:

- Core Architecture
- Build API
- Mirroring
- Parity

## Parity Validation

Reference parity consumes the checked-in `.npz` traces under [`tests/reference_data`](tests/reference_data), driven by [`tests/parity_harness.py`](tests/parity_harness.py). The full suite covers Walker + multi-env reference parity, terrain stability, mirroring, structural terrain semantics, and variable morphology checks:

```bash
uv run --extra dev pytest tests/
```

## Tests

Repository tests live in [`tests`](tests).

Lightweight local suggestions:

```bash
pytest tests/test_world.py tests/test_render.py
pytest tests/test_cpp_parity.py tests/test_multi_env_parity.py tests/test_terrain_stability.py
```

## Repository Layout

- [`src/jax_evogym`](src/jax_evogym): core package
- [`src/jax_evogym/build.py`](src/jax_evogym/build.py): object-separated world compilation and instantiation
- [`src/jax_evogym/types.py`](src/jax_evogym/types.py): core runtime/build types
- [`src/jax_evogym/render.py`](src/jax_evogym/render.py): renderer
- [`site`](site): Astro/Starlight docs site and embedded browser designer
- [`src/jax_evogym/design_tool`](src/jax_evogym/design_tool): shared render token helpers
- [`tests`](tests): parity, stability, renderer, and world tests

## Current Scope

This README documents the standalone **core library**. The research code built on top of it lives in the companion repository, [sensing-at-the-edges](https://github.com/btgaskin/sensing-at-the-edges).

The current stable scope is square-cell soft-body simulation with fixed static terrain, flat terrain friction, variable robot morphology, rendering, and parity validation. Slopes, friction-on-slope behavior, general freeform terrain, and dynamic polygon bodies remain future work or unsupported experimental code.

## Acknowledgements

`jax-evogym` is a JAX-native reimplementation of [Evolution Gym](https://github.com/EvolutionGym/evogym) (EvoGym), the soft-body robot benchmark by Jagdeep Bhatia, Holly Jackson, Yunsheng Tian, Jie Xu, and Wojciech Matusik (NeurIPS 2021). Evolution Gym is distributed under the MIT License (Copyright © 2022 jagdeepsb). The physics model and the reference traces used for [parity validation](#parity-validation) are derived from it. See [`NOTICE`](NOTICE) for the full attribution and license text.

If you build on this work, please also cite the original benchmark:

> Bhatia, J., Jackson, H., Tian, Y., Xu, J., & Matusik, W. (2021). *Evolution Gym: A Large-Scale Benchmark for Evolving Soft Robots.* Advances in Neural Information Processing Systems (NeurIPS).

## License

`jax-evogym` is licensed under [Apache-2.0](LICENSE). EvoGym-derived material,
including reference traces and physics parameterization, is attributed under
the MIT License in [NOTICE](NOTICE).
