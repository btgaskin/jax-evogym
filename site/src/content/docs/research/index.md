---
title: Research & Data
description: Sensing at the Edges as a worked research application, with links to the paper, code, and specimen records.
---

## Sensing at the Edges

[Sensing at the Edges](https://www.bengaskin.com/sensing-edges/) shows one way to
combine soft-body simulation, distributed control, and evolutionary search.
The experiment studies coordination in a body of identical contractile cells
whose controllers receive local deformation feedback and neighbouring signals.

- [Project overview and animations](https://www.bengaskin.com/sensing-edges/) — start here to see the system and its behaviour.
- [Paper PDF](https://github.com/btgaskin/sensing-at-the-edges/blob/main/gaskin_alife.pdf) — *Sensing at the edges: synthetic specimens for early contractile coordination*, Ben Gaskin.
- [Research code and reproduction notes](https://github.com/btgaskin/sensing-at-the-edges) — the implementation and its documented limitations.

## How this uses the simulator

Read the example in three layers:

1. **Simulation:** JAX EvoGym supplies the soft-body physics, deformation
   measurements, and actuator primitives. See [Simulation Loop](../reference/simulation-loop/)
   and [Observation Helpers](../reference/observation-helpers/).
2. **Control:** the research project supplies an NCA-CTRNN controller: recurrent
   dynamics within each cell, combined with local communication between cells.
   It converts feedback into horizontal and vertical actuation and signalling.
   Contractile cells use per-axis control; the built-in environments' scalar
   action interface is not a drop-in replacement.
3. **Search and analysis:** the research project supplies CMA-MAE
   quality-diversity search, task definitions, fitness, and post-run assays.
   These are choices made for the experiment, rather than requirements imposed
   by the simulator. In this experiment, evolution changes the controller
   parameters while the body stays fixed.

This is an advanced worked application. For a first simulation, begin with
[First Simulation](../getting-started/first-simulation/) and [Controllers and Actions](../guides/controllers/).
Then use the paper and code to follow how sensing, controller state, actuation,
and fitness fit together. You can bring a different controller or search
algorithm to the same simulator.

## Where to look in the code

The [research repository](https://github.com/btgaskin/sensing-at-the-edges) contains:

- `sensing_edges/` — controllers, training, evaluation, task configuration, and terrain suites.
- `preprint/` — analysis and figure scripts.
- `s4/` — specimen bundles, replay/probe tools, and the validity audit.

Start with a saved specimen and its configuration before attempting a training
run. Training uses Modal GPU infrastructure; consult the repository's
reproduction instructions for setup and compute requirements.

## Reproduction and evidence limits

The repository documents differences between its current training preset and
the settings that produced archived specimens. Use each specimen's
`config_snapshot.yaml` and the audit notes when interpreting a replay. Some
resumed runs lack the parent checkpoints needed to regenerate their evolution
from scratch. Specimen replay and training reproduction are distinct tasks.

The paper's results concern its chosen bodies, controllers, tasks, and selected
specimens. They do not establish that every robot or controller built with
JAX EvoGym will exhibit the same behaviour. Read the project's
[discussion of evidence and limits](https://www.bengaskin.com/sensing-edges/)
alongside the simulator's [parity scope](../core/parity/).

## Full datasets

Curated specimen bundles are included in the research repository. Complete run
outputs are available on request: [open an issue there](https://github.com/btgaskin/sensing-at-the-edges/issues)
and identify the figure, specimen, or dataset you need.
