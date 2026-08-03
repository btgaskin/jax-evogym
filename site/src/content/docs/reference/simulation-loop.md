---
title: Simulation Loop
description: The physics substep pipeline, integrators, and episode step factories.
---

## env_step

The main simulation entry point. Runs one environment step (30 physics substeps by default).

```python
new_sim_state = env_step(
    state,              # SimState
    topology,           # SpringTopology
    constants,          # PhysicsConstants
    action,             # (n_actuators,) float32
    actuator_info,      # ActuatorInfo
    collision_data=..., # CollisionData (optional)
    static_collider_data=...,  # StaticColliderData (optional)
)
```

Internally:

1. `set_actuator_goals(state, actuator_info, action)` — maps actions to spring rest-length goals once per step
2. `jax.lax.scan(physics_substep, state, length=30)` — runs 30 substeps

### Collision strategy

The `collision_strategy` parameter selects between two collision detection paths:

- **`0` (static indexed)**: uses precomputed collision triples. Fast when the world geometry doesn't change dramatically. Default for built-in environments.
- **`1` (dynamic broadphase)**: bin-based spatial filtering, recomputes candidate pairs each substep. Required when collision geometry changes between steps (e.g., grown morphologies).

There is also `env_step_dynamic_broadphase`, which always uses the dynamic broadphase path without the strategy switch.

## physics_substep

One physics substep. Called 30 times per `env_step`.

```
physics_substep(state, topology, constants, collision_data, static_collider_data)
```

Pipeline:

1. **resolve_collisions** — object-object collision forces (robot vs dynamic terrain, self-collision)
2. **resolve_static_collisions** — dynamic objects against static `FIXED` terrain geometry
3. **update_actuators** — converge spring rest lengths toward goals (`convergence = 0.006`)
4. **Integration** — RK4 or symplectic Euler
5. **resolve_edge_constraints** — Position-Based Dynamics correction (two phases)
6. **Velocity update** — `velocities_true = (new_pos - positions_last) / dt`
7. **Clear external forces**

## Integrators

Selected via `PhysicsConstants.integrator`:

### RK4 (default, `integrator=0`)

Fourth-order Runge-Kutta. Computes forces at 4 intermediate points, combines with standard RK4 weights. More accurate, higher compute cost.

`compute_forces` evaluates: spring forces + viscous drag + gravity + ground-plane contact + ground-plane friction + external forces from collision detection.

Static terrain contact is separate from the ground plane. Flat `FIXED` static
terrain can contribute tangential collision friction and opt-in floor stiction
through `resolve_static_collisions`. Experimental sloped static edges are
normal-only geometry: slope friction, support, grip, and stiction controls are
ignored.

### Symplectic Euler (`integrator=1`)

Kick-drift symplectic integrator. Updates velocity first (kick), then position (drift). Less accurate but faster and energy-stable.

## Actuator mechanics

### set_actuator_goals

Called once per `env_step`. Maps the action vector to spring rest-length goals:

1. Scatter-add each action value to the two springs owned by that actuator cell
2. Average shared springs (edges between two actuator cells get the mean)
3. Goal = `(sum / count) * spring_init_rest_length`

Actions are expected in `[0.6, 1.6]` — the ratio of target rest length to initial rest length.

### update_actuators

Called every substep. Converges rest lengths toward goals:

```
new_rest = current_dist + (goal - current_dist) * 0.006
```

This gradual convergence prevents instantaneous spring length changes that would create instability.

## Edge constraints (PBD)

Position-Based Dynamics correction applied after integration:

- **Phase 1**: all active springs, `+/-25%` length tolerance, correction factor `0.8`
- **Phase 2**: springs where `rigid_spring_mask=True`, `+/-3%` length tolerance, correction factor `0.5`

The Phase 2 constraint pass itself does not branch on `fixed`; `physics_substep` restores anchored points immediately afterward. In the current builders, `rigid_spring_mask` includes `RIGID` springs and `FIXED` springs inside dynamic mixed objects, which is slightly broader than the original C++ rigid-boxel traversal.

## make_episode_step

Lower-level factory that creates a `lax.scan`-compatible step operating on `SimState` directly (no env-level reward/done logic):

```python
from jax_evogym import make_episode_step

step_fn = make_episode_step(
    topology, constants, actuator_info,
    collision_data=...,
    static_collider_data=...,
)

final_state, all_positions = jax.lax.scan(step_fn, init_sim_state, actions)
```

Returns `(new_sim_state, positions)` at each step.

## Key constants

| Constant | Value | Description |
|----------|-------|-------------|
| `PHYSICS_UPDATES_PER_STEP` | 30 | Substeps per env step |
| `DT` | 0.0001 | Time step (seconds) |
| `ACTUATOR_CONVERGENCE` | 0.006 | Rest-length convergence rate |
| `GRAVITY` | 110.0 | Gravitational constant |
| `VISCOUS_DRAG` | 0.1 | Velocity damping coefficient |

Each env step advances `30 * 0.0001 = 0.003` seconds of simulated time.
