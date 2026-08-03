---
title: Installation
description: Install jax-evogym and its optional dependencies.
---

## Base package

`jax-evogym` is not yet published on PyPI. Install it from a source checkout.

```bash
pip install .
```

With visualization support (includes Pillow):

```bash
pip install ".[viz]"
```

For local development:

```bash
pip install -e ".[viz,dev]"
```

If you use `uv`:

```bash
uv pip install -e ".[viz,dev]"
```

## JAX backend

The base install includes the pinned JAX version for CPU use. For CUDA 12 on Linux, install the project CUDA extra from the source checkout:

```bash
pip install ".[cuda]"
```

This keeps the exact JAX version that the test suite validates while installing the CUDA backend.

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for full platform instructions.

## Docs site and designer

The documentation site and browser-based designer are a separate Astro/Starlight frontend. They do not require JAX.

```bash
bun --cwd site install
bun --cwd site dev
```

Build the static site:

```bash
bun --cwd site build
```

The designer is served at `/designer` during local development and in the static build.
