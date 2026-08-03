---
title: Research & Data
description: The research built on jax-evogym, and where to find the code and data behind it.
---

`jax-evogym` was built to support research on deformation sensing in evolved
soft-body robots. The research code will be published in a companion repository
alongside the paper, so the link may not resolve until then:

- [sensing-at-the-edges](https://github.com/btgaskin/sensing-at-the-edges) —
  the full paper pipeline: QD training (CMA-MAE with an NCA-CTRNN controller),
  post-run assays, analysis scripts, and the specimen audit.

## What ships where

The companion repository includes:

- the complete training and evaluation code (`sensing_edges/`), including the
  Hydra presets used for the paper runs
- the terrain suites the runs were trained and evaluated on
- the preprint analysis and figure scripts (`preprint/`)
- the specimen audit and probe tooling (`s4/`), with curated specimen bundles —
  genome, controller parameters, resolved config, and representative renders
  for each featured robot

## Full datasets

The complete run outputs behind the paper figures (assay trees, trace bundles,
and derived analyses, tens of gigabytes) are not hosted in either repository.
They are available on request.
