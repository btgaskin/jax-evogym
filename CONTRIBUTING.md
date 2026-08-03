# Contributing

`jax-evogym` is a public-alpha community research library. Contributions should keep the public core small, deterministic, and easy to validate.

## Scope

- Keep changes to the core library under `src/jax_evogym` focused and tested.
- Keep experiment and research workflows in the companion research repository, not in this library.
- Treat slope terrain as unsupported experimental code unless a change explicitly opts into that path with `allow_experimental_slopes=True`.

## Local Checks

Use deterministic, specimen-level checks locally. Do not run local training or population sweeps.

Recommended core checks:

```bash
uv run --extra dev pytest -q tests
uv build --wheel --sdist
bun --cwd site test
ASTRO_TELEMETRY_DISABLED=1 bun --cwd site build
```

CI mirrors these checks for pull requests and pushes to `main`. Package builds
also scan wheel and sdist contents so experiment outputs, paper artifacts, and
other local/generated files do not leak into public artifacts.

Large-scale training and evaluation protocols live in the companion research repository, [sensing-at-the-edges](https://github.com/btgaskin/sensing-at-the-edges), not in this library.

## Physics Changes

Physics and math changes need a tighter standard than ordinary refactors:

- State whether the behavior is EvoGym-derived, a JAX implementation detail, or an intentional extension.
- Add focused invariant tests for force direction, masking, fixed-point behavior, finiteness, and shape contracts.
- Update parity/probe documentation when a behavior changes the published simulation contract.
