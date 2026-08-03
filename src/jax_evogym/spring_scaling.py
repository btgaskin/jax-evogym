"""Helpers for contractile spring stiffness scaling laws."""

from __future__ import annotations

import jax.numpy as jnp

from . import constants as C

_SOFT_ANCHOR_SCALE = C.SOFT_MAIN_K / C.ACTUATOR_MAIN_K
_SOFT_RATIO = C.SOFT_DIAG_K / C.SOFT_MAIN_K
_ACT_RATIO = C.ACTUATOR_DIAG_K / C.ACTUATOR_MAIN_K
_BLEND_DENOM = 1.0 - _SOFT_ANCHOR_SCALE


def contractile_diag_ratio(scale: float) -> float:
    """Bounded main->diag ratio for CONTRACTILE springs at stiffness scale ``scale``.

    Anchors:
      - k <= SOFT_MAIN_K / ACTUATOR_MAIN_K: ratio = SOFT_DIAG_K / SOFT_MAIN_K
      - k >= 1.0: ratio = ACTUATOR_DIAG_K / ACTUATOR_MAIN_K
      - k in (soft_anchor, 1): linear blend between the two ratios
    """
    k = float(scale)
    if k <= _SOFT_ANCHOR_SCALE:
        return _SOFT_RATIO
    if k >= 1.0:
        return _ACT_RATIO
    blend = (k - _SOFT_ANCHOR_SCALE) / _BLEND_DENOM
    return _SOFT_RATIO + blend * (_ACT_RATIO - _SOFT_RATIO)


def contractile_diag_ratio_jax(scale: jnp.ndarray | float) -> jnp.ndarray:
    """JAX-compatible version of :func:`contractile_diag_ratio`."""
    k = jnp.asarray(scale, dtype=jnp.float32)
    blend = jnp.clip(
        (k - jnp.float32(_SOFT_ANCHOR_SCALE)) / jnp.float32(_BLEND_DENOM),
        jnp.float32(0.0),
        jnp.float32(1.0),
    )
    return jnp.float32(_SOFT_RATIO) + blend * jnp.float32(_ACT_RATIO - _SOFT_RATIO)
