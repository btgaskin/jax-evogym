# Roadmap

`jax-evogym` is a public-alpha community research library.

## Near-Term Goals

- Keep the stable public API focused on square-cell EvoGym simulation.
- Preserve deterministic local tests, package builds, docs builds, and artifact
  boundary checks as release gates.
- Keep experiment code secondary to the core package and out of PyPI artifacts.
- Document every physics change as EvoGym-derived, JAX-specific, or an
  intentional extension.

## Known Limitations

- Slopes are experimental geometry only. They can be loaded and compiled through
  the explicit experimental path, but slope friction/support is disabled.
- Parity validation is CPU-based; GPU backends are not yet covered by a
  dedicated validation gate.
- Performance benchmarks are not yet a release gate.

## Versioning Policy

During `0.x`, minor versions may tighten the public API boundary or move
experimental behavior. Patch versions should be bug fixes, docs corrections, and
test/build maintenance.

Stable API commitments apply to the top-level public symbols and documented
square-cell behavior. Compatibility-only symbols, including slope constants and
slope contact/grip fields, may change before `1.0`.

## Deferred Work

- A defensible slope friction/support model.
- Performance comparison harnesses against original EvoGym and other JAX soft
  body simulators.
- A broader accelerator compatibility matrix beyond the currently validated JAX
  pin.
