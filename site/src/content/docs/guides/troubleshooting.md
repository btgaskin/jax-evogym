---
title: Troubleshooting
description: Diagnose installation, action, rendering, and designer workflow problems.
---

Use this page when a workflow fails. If you are choosing where to begin, return to [Installation](../../getting-started/installation/) or [First Simulation](../../getting-started/first-simulation/).

## Import or rendering fails

Run commands from the repository root with the intended virtual environment active. Install `python -m pip install -e ".[viz,dev]"` for examples and tests. The visualization extra supplies Pillow. The examples are source-checkout modules, not installed command-line programs in the library wheel.

## The first step is slow

JAX compiles on first use. Separate compilation from steady-state execution and synchronize device results before timing. Inspect `jax.devices()` to see the selected backend. A GPU is not required for a small introductory rollout; installing base JAX does not guarantee accelerator support.

## The robot does not walk

Neutral actions preserve target lengths; they do not create a gait. Periodic actions show deformation but are not trained controllers. Check actuator types, object placement, action ordering, and the episode duration. Watch a single rollout before interpreting its score.

## Action shape is wrong, or contractile cells do not respond

Built-in scalar actions need exactly `env.n_actuators` entries for H/V cells. Contractile cells require the [per-axis path](../controllers/#contractile-and-per-axis-control). A body change can alter both observation and action dimensions.

## The world differs from the designer

Choose the exact robot object name. Use the complete-world example to preserve terrain and placement. Passing only a body into `WalkerV0` uses the benchmark's own world. Spawn, target, and direction markers are annotations, not simulation settings. Python serialization does not preserve those annotations.

## A finished episode keeps moving

`done` remains true and subsequent reward is zero, but built-in stepping continues physics. Freeze the state or reset the episode explicitly; see the [controller guide](../controllers/).

## An evaluation produces NaN or infinite values

Keep the failed body, actions, and configuration. Reproduce the failure with one specimen, verify bounded actions and non-overlapping initial geometry, and inspect positions before rendering or selecting winners. Do not silently treat a failed rollout as a valid high-scoring candidate.

## A browser draft disappears

Designer drafts are stored in that browser's local storage. Different browsers, site origins, private sessions, or cleared storage do not share them. Export JSON files for durable copies and keep the original files if you need annotations.

For a reproducible bug, [open an issue](https://github.com/btgaskin/jax-evogym/issues) with the revision, platform, dependency versions, smallest world/body, actions, and full error output. Exclude credentials and private data.
