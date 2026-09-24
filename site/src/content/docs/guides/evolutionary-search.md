---
title: Connect an Evolutionary Algorithm
description: Separate candidate generation, simulation, fitness, and selection.
---

JAX EvoGym supplies simulation and benchmark environments. It does not provide a general training loop or a built-in evolutionary optimizer. Start with [controller evaluation](../batched-evaluation/), then connect its scores to an optimizer of your choice.

## Define the experiment

Choose the genotype: controller parameters, body cells, or both. Fix the task, episode horizon, initial conditions, action mapping, and fitness rule before comparing candidates. A controller-only experiment keeps body geometry fixed. Body evolution must also validate connectivity, actuator presence, allowed materials, and controller input/output dimensions.

A generation has the following structure. This is integration pseudocode; `optimizer` and `evaluate` are supplied by your experiment:

```text
candidates = optimizer.ask()
fitness, diagnostics = evaluate(candidates, evaluation_conditions)
reject_or_penalize_invalid_candidates(fitness, diagnostics)
optimizer.tell(candidates, fitness)
save_checkpoint(optimizer, candidates, fitness, configuration)
```

Check whether your optimizer minimizes or maximizes. Built-in environment reward is not automatically the fitness you want: decide how to combine episodes, penalize invalid simulations, and measure robustness. Quality-diversity methods also need behavior descriptors and an archive update rule.

## Preserve what a score means

Save the code revision, dependency versions, world/body definition, controller parameters, random seeds, evaluation horizon, and fitness definition. Keep raw episode scores and failures. Checkpoint optimizer state as well as the current best controller so training can resume.

Use held-out conditions for final evaluation. Replaying a high-scoring candidate demonstrates its behavior under those replay conditions; it does not establish generalization. Compare against simple baselines, including neutral and periodic actions.

Run training and population sweeps on remote compute in accordance with the repository execution policy. Validate the simulator and one deterministic specimen before allocating a large run.

## A research example

[Sensing at the Edges](../../research/) combines a fixed body, a distributed NCA-CTRNN controller, and CMA-MAE search. Its public code is a useful advanced reference for controller state, terrain evaluation, and experiment configuration. It is not a minimal dependency or a drop-in training API for this package. Use archived configurations and the paper's stated protocol when investigating its results.
