---
title: Concepts
description: How the pieces of jax-evogym fit together — the mental model you need before diving deeper.
---

## The three-phase pipeline

Every jax-evogym simulation passes through three phases:

1. **Define** — describe the world in numpy
2. **Compile** — convert it to a reusable template
3. **Instantiate** — produce JAX arrays ready for simulation

```
EvoWorld (numpy)
    │
    ▼
compile_world_template()
    │
    ▼
WorldTemplate (numpy, reusable)
    │
    ▼
instantiate_world()
    │
    ▼
BuiltWorld (JAX arrays, ready to simulate)
```

The built-in environments handle all three phases internally. The builder API is for when you need direct control.

## Objects, not grids

The original EvoGym merges everything into a single occupancy grid. jax-evogym compiles each `WorldObject` independently, then assembles them into a world. This means:

- Robot points, springs, and actuators are tracked by object identity
- Terrain objects are separate from the robot
- Fixed terrain can be excluded from the dynamic simulation entirely

A typical world has 2-3 objects: a robot, ground/walls, and sometimes dynamic terrain (like the soft bridge in BridgeWalkerV0).

## Dynamic vs static

Objects are classified at compile time:

- **Dynamic objects**: the robot, and any terrain containing `SOFT`, `RIGID`, or actuator cells. These get points, masses, velocities, and springs in `SimState`.
- **Static objects**: terrain composed entirely of `FIXED` cells. These contribute collider geometry and render geometry, but no dynamic simulation state.

This classification is automatic. Flat ground, walls, and stairs made from
`FIXED` cells are static. A soft rope-bridge is dynamic. Legacy slope ids are
experimental geometry only and are rejected by stable compilation.

## SimState

`SimState` is the core simulation state, vmapped across a population in evolutionary search:

- **Point state** (changes every substep): positions, velocities, positions_last, velocities_true
- **Point properties** (fixed per episode): masses, fixed flags, point_mask
- **Spring state** (changes with actuation): rest_length, rest_length_goal, init_rest_length, spring_const, spring_mask, rigid_spring_mask
- **External forces**: collision forces accumulated each substep, cleared after integration
- **Friction memory**: optional non-slope terrain friction fields used by opt-in stiction models

Everything else — spring topology, physics constants, actuator info, collision geometry — is shared across the batch and does not appear in `SimState`.

## The simulation loop

Each environment `step` calls `env_step`, which runs 30 physics substeps:

```
env_step(sim_state, topology, constants, action, actuator_info, ...)
  │
  ├─ set_actuator_goals(...)     # once: map actions to spring rest-length goals
  │
  └─ lax.scan(physics_substep, state, length=30)
       │
       ├─ resolve_collisions()         # object-object forces
       ├─ resolve_static_collisions()  # dynamic objects against static terrain
       ├─ update_actuators()           # converge rest lengths toward goals
       ├─ rk4_step()                   # force computation + integration
       └─ resolve_edge_constraints()   # PBD position correction
```

The integrator (RK4 by default) computes spring forces, viscous drag, gravity,
ground-plane contact/friction, and external forces from collision, then advances
positions and velocities.

## Environments wrap the pipeline

Each built-in environment:

1. Loads terrain from a bundled JSON file
2. Compiles and instantiates the world with the user's morphology
3. Exposes `reset()` and `step(state, action)` as a JIT-compiled functional interface
4. Computes observations, rewards, and done conditions specific to the task

The `step` function is pure — given the same state and action, it always returns the same result. This makes environments compatible with `jax.vmap` and `jax.lax.scan`.

## lax.scan episode rollouts

Each environment has a companion factory (`make_*_episode_step`) that returns a function compatible with `jax.lax.scan`:

```python
episode_step = make_walker_episode_step(env)
final_state, outputs = jax.lax.scan(episode_step, init_state, actions)
```

This is the standard way to run full episodes efficiently in JAX — the entire rollout compiles to a single XLA program.

## Two build pipelines

jax-evogym has two ways to build simulation data:

| | Object-separated (`build.py`) | Dense grid (`jax_utils.py`) |
|---|---|---|
| **Entry point** | `compile_world_template()` | `precompute_grid()` + `jax_build_sim_state()` |
| **Supports terrain** | Yes — multi-object worlds and fixed static terrain | No — single robot object only |
| **JAX-native** | No — uses numpy, produces JAX arrays at the end | Yes — pure JAX, JIT-able, vmap-able |
| **Use case** | Standard environments, custom worlds | Evolutionary search over morphology populations |

The built-in environments use the object-separated pipeline. The dense grid pipeline is for research scenarios where you need to `jax.vmap` across thousands of different morphologies. See [Dense Grid Pipeline](../../guides/dense-grid/) for details.

## NamedTuples as pytrees

All state types (`SimState`, `EnvState`, `CollisionData`, etc.) are Python `NamedTuple` subclasses. JAX automatically registers NamedTuples as pytrees, so they work with `jax.jit`, `jax.vmap`, and `jax.lax.scan` without custom registration.
