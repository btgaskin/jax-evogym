---
title: Parity
description: Reference data, the parity test suite, and the thresholds used to decide whether parity still holds.
---

The core parity workflow is built around two pieces:

- checked-in reference traces under [`tests/reference_data`](https://github.com/btgaskin/jax-evogym/tree/main/tests/reference_data)
- the shared parity harness in [`tests/parity_harness.py`](https://github.com/btgaskin/jax-evogym/blob/main/tests/parity_harness.py)

## Reference data

Current checked-in reference traces include:

- `walker_v0_simple_3x1.npz`
- `walker_v0_biped_3x4.npz`
- `bridge_walker_simple_3x1.npz`
- `climber_simple_3x1.npz`
- `height_max_simple_3x1.npz`
- `jumper_simple_3x1.npz`
- `upstepper_simple_3x1.npz`
- `wingspan_max_simple_3x1.npz`

These are consumed directly by the parity harness. Refreshing them is a separate maintenance task and is not part of normal parity execution.

## Running the parity suite

The full suite runs through pytest:

```bash
uv run --extra dev pytest tests/
```

A focused parity pass covers the same ground faster:

```bash
pytest tests/test_cpp_parity.py tests/test_multi_env_parity.py tests/test_terrain_stability.py
```

The parity-relevant test modules are:

- `tests/test_cpp_parity.py` — Walker reference parity against the checked-in traces
- `tests/test_multi_env_parity.py` — multi-env reference parity
- `tests/test_terrain_stability.py` — terrain stability invariants
- `tests/test_core_parity_harness.py` — harness-level checks: structural semantics, mirroring, and variable morphology / `robot_override`

## Thresholds

### Walker reference

- initial COM x error `< 0.15`
- COM x max error over 50 steps `< 0.5`
- COM y max error over 50 steps `< 0.5`
- observation max diff:
  - step `0` `< 0.25`
  - step `10` `< 1.0`
  - step `25` `< 2.0`
  - step `50` `< 5.0`
- cumulative reward within the existing diagnostic ratio / near-zero thresholds

### Multi-env reference

- free-body envs: COM x/y max error `< 1.0`
- terrain-contact envs: COM x/y max error `< 1.5`
- cumulative reward within the existing diagnostic ratio / near-zero thresholds

### Terrain stability

- dynamic fixed points must remain unchanged
- dynamic springs with both endpoints fixed must retain their rest length within the current tolerance
- static terrain geometry must remain invariant

## What counts as a parity failure

A parity failure is any of:

- threshold violation
- NaN / non-finite rollout state
- incorrect structural semantics
- broken mirror/template invariants
- inconsistent `robot_override` bookkeeping

## Experimental slope checks

The parity harness still contains slope-specific checks for internal review and
legacy fixtures, but those checks require the explicit experimental compiler
path and are not part of the stable release gate. These checks cover legacy
geometry and validation only; slope friction/support is intentionally disabled.

Because parity status is time-sensitive, update any recorded pass/fail notes only after rerunning the relevant parity tests.
