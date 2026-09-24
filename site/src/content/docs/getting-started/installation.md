---
title: Installation
description: Install jax-evogym and its optional dependencies.
---

## Base package

`jax-evogym` is not yet published on PyPI. Install it from a source checkout.

Requires Python 3.11 or later. Clone the repository, then install from its root:

```bash
git clone https://github.com/btgaskin/jax-evogym.git
cd jax-evogym
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install .
```

With visualization support (includes Pillow):

```bash
python -m pip install ".[viz]"
```

For local development:

```bash
python -m pip install -e ".[viz,dev]"
```

If you use `uv`:

```bash
uv pip install -e ".[viz,dev]"
```

## JAX backend

The base install includes the pinned JAX version for CPU use. For CUDA 12 on Linux, install the project CUDA extra from the source checkout:

```bash
python -m pip install ".[cuda]"
```

This keeps the exact JAX version that the test suite validates while installing the CUDA backend.

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for full platform instructions.

## Docs site and designer

The documentation site and browser-based designer are a separate Astro/Starlight frontend. They do not require JAX.

```bash
cd site
bun install --frozen-lockfile
bun run dev --host 127.0.0.1
```

Build the static site:

```bash
bun run build  # from site/
```

The designer is served at `/designer` during local development and in the static build.

## Choose your next step

Read [Concepts](../concepts/) for the system's structure, then [Quick Start](../quickstart/) for a short API introduction and [First Simulation](../first-simulation/) for a complete rollout.

If you only want to draw a world, the [Designer Guide](../../designer/designer-guide/) needs no Python installation. Later, [Designer to Simulation](../../guides/designer-world/) explains how to run that export in Python.
