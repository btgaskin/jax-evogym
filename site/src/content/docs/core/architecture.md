---
title: Core Architecture
description: How jax-evogym compiles object-separated worlds and classifies dynamic versus static terrain.
---

`jax-evogym` now compiles worlds as **separate objects**, not as a single merged robot-plus-terrain point/spring graph.

## Object compilation

Each `WorldObject` is compiled independently into an `ObjectTemplate`:

- local point positions
- local spring topology
- dynamic boxel geometry
- generalized cell geometry for rendering and static collision
- actuator metadata
- deformation metadata
- ownership flags

The object templates are then assembled into a `WorldTemplate`.

## Dynamic vs static classification

The runtime classifies objects by object composition:

- **robot object**: always dynamic
- **pure `FIXED` non-robot object**: static-only
- **mixed or soft non-robot object**: dynamic

This means:

- flat ground, walls, and fixed stairs do not consume dynamic simulation state
- bridge-like soft terrain remains dynamic
- fixed anchors inside a dynamic terrain object remain fixed within that dynamic object
- unsupported experimental slope terrain is retained behind an explicit compiler opt-in as geometry-only collider/render data; slope friction/support is not part of the stable architecture contract

## Collision model

Collision is split into two direct paths:

- **dynamic-dynamic**
  - robot vs dynamic terrain
  - robot self-collision
  - dynamic terrain cross-object collision where applicable
- **dynamic-static**
  - dynamic objects against precomputed fixed terrain collider geometry
  - stable static cells are fixed quads (`FIXED`)
  - experimental sloped edges are normal-only geometry when explicitly enabled

The active collision path uses **real global point indices** only. There is no terrain-point duplication or offset terrain point space.

## Render model

Rendering uses the same ownership-aware geometry as the simulator:

- dynamic points come from `SimState.positions`
- static terrain comes from `RenderInfo.static_point_positions`
- render cells are marked by object ownership and dynamic/static status

This is why static terrain is no longer visible as “moving terrain points” in the renderer.

## Why fixed terrain is not in `SimState`

Pure fixed terrain does not need:

- masses
- velocities
- springs
- PBD corrections
- actuator state

Removing it from `SimState` gives both:

- cleaner mechanics
- better simulation efficiency, especially on sparse worlds
