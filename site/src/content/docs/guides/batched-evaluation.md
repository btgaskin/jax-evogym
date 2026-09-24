---
title: Evaluate Controller Candidates
description: Compare fixed controllers with scan over time and vmap over candidates.
---

This guide builds on [Controllers and Actions](../controllers/). Keep the body and task fixed while comparing controller parameters; introduce body variation later.

Begin with a single finite rollout and inspect its animation. Then batch controllers that share the same body and task. The example below compares four fixed amplitude/phase pairs; it does not train a policy or establish learning performance.

Run population evaluation on a suitable compute host. This repository's local execution policy keeps population workloads and training off the development machine unless explicitly approved.

```bash
python -m examples.controller_evaluation --steps 100 --output /tmp/best.gif
```

The script prints four scores and renders only the best candidate when `--output` is supplied. The ranking applies to this body, task, and short horizon. It is not evidence of robust locomotion.

## Array layout and episode handling

[`examples/controller_evaluation.py`](https://github.com/btgaskin/jax-evogym/blob/main/examples/controller_evaluation.py) separates three operations:

- `evaluate_one` scans over time and sums rewards for one parameter vector.
- `evaluate_batch` maps that evaluator over candidates and compiles the result.
- `run` selects a result on the host and optionally replays it for rendering.

The input has shape `(candidates, 2)` for amplitude and phase. Scores have shape `(candidates,)`. Final-state arrays have a leading candidate axis. An outer `vmap` around a scan that records trajectories would produce `(candidates, time, ...)`; a scan around a batched step would produce `(time, candidates, ...)`.

All candidates start from the same reset state here. The helper freezes completed states, keeps terminal rewards, and rejects non-finite scores before selection. For stochastic policies or randomized tasks, use explicit independent keys and define which initial conditions candidates share.

## Cost and validity

Batching trades memory for throughput. Avoid saving every frame of every candidate; return fitness and only the summaries you need. JAX's first call includes compilation. Synchronize results before timing, for example with `scores.block_until_ready()`.

Different array shapes can require recompilation. If bodies vary, first read [variable morphology](../variable-morphology/) for masks and fixed-shape construction. Do not assume a controller's action slots retain their meaning after changing the body.

CI compares a tiny batched evaluation against separate scalar evaluations. That check verifies implementation consistency, not accelerator performance or scientific results. Next, connect the evaluator to an [evolutionary algorithm](../evolutionary-search/).
