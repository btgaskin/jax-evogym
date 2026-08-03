"""Shared tangential friction helpers.

Ground-plane contact and static-terrain contact use the same stick/slip blend,
but the dynamic branch currently has two distinct saturation laws:

- legacy ground-plane: tanh(v * N * mu_d)
- load-normalized static terrain: tanh(v / (N * mu_d))

Keeping both in one place makes that choice explicit and testable.
"""

from __future__ import annotations

import jax.numpy as jnp


def dynamic_friction_scalar(
    tangential_velocity: jnp.ndarray,
    normal_force_mag: jnp.ndarray,
    *,
    friction_const: float | jnp.ndarray,
    mu_d: float | jnp.ndarray,
    normalize_by_load: bool = True,
) -> jnp.ndarray:
    """Return the dynamic tangential friction scalar.

    When ``normalize_by_load`` is true, low-speed slope is approximately
    ``-friction_const * tangential_velocity`` and is independent of normal load.
    When false, the helper preserves the legacy ground-plane saturation law.
    """
    v_tang = jnp.asarray(tangential_velocity, dtype=jnp.float32)
    normal_force_mag = jnp.asarray(normal_force_mag, dtype=jnp.float32)
    friction_const_f32 = jnp.asarray(friction_const, dtype=jnp.float32)
    mu_d_f32 = jnp.asarray(mu_d, dtype=jnp.float32)

    if normalize_by_load:
        dyn_arg = v_tang / jnp.maximum(mu_d_f32 * normal_force_mag, 1e-10)
    else:
        dyn_arg = v_tang * normal_force_mag * mu_d_f32

    return (
        -friction_const_f32
        * mu_d_f32
        * normal_force_mag
        * jnp.tanh(dyn_arg)
    )


def stiction_force_scalar(
    tangential_velocity: jnp.ndarray,
    normal_force_mag: jnp.ndarray,
    *,
    mu_s: float | jnp.ndarray,
    k_stick: float | jnp.ndarray,
    tangential_load: float | jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return the quasi-static tangential hold force with Coulomb clipping."""
    v_tang = jnp.asarray(tangential_velocity, dtype=jnp.float32)
    normal_force_mag = jnp.asarray(normal_force_mag, dtype=jnp.float32)
    mu_s_f32 = jnp.asarray(mu_s, dtype=jnp.float32)
    k_stick_f32 = jnp.asarray(k_stick, dtype=jnp.float32)
    tangential_load_f32 = (
        jnp.zeros_like(v_tang, dtype=jnp.float32)
        if tangential_load is None
        else jnp.asarray(tangential_load, dtype=jnp.float32)
    )
    stick_cap = mu_s_f32 * normal_force_mag
    return jnp.clip(
        tangential_load_f32 - k_stick_f32 * v_tang,
        -stick_cap,
        stick_cap,
    )


def stiction_blend_alpha(
    tangential_velocity: jnp.ndarray,
    switch_speed: float | jnp.ndarray,
) -> jnp.ndarray:
    """Return the speed-gated blend weight between stick and slip models."""
    v_tang = jnp.asarray(tangential_velocity, dtype=jnp.float32)
    switch_speed_f32 = jnp.asarray(switch_speed, dtype=jnp.float32)
    return jnp.tanh(jnp.abs(v_tang) / jnp.maximum(switch_speed_f32, 1e-10))


def blended_friction_scalar(
    tangential_velocity: jnp.ndarray,
    normal_force_mag: jnp.ndarray,
    *,
    friction_const: float | jnp.ndarray,
    mu_d: float | jnp.ndarray,
    mu_s: float | jnp.ndarray,
    k_stick: float | jnp.ndarray,
    switch_speed: float | jnp.ndarray,
    tangential_load: float | jnp.ndarray | None = None,
    normalize_dynamic_by_load: bool = True,
) -> jnp.ndarray:
    """Blend quasi-static stiction with dynamic slip friction."""
    mu_s_f32 = jnp.asarray(mu_s, dtype=jnp.float32)
    k_stick_f32 = jnp.asarray(k_stick, dtype=jnp.float32)
    switch_speed_f32 = jnp.asarray(switch_speed, dtype=jnp.float32)

    f_dyn = dynamic_friction_scalar(
        tangential_velocity,
        normal_force_mag,
        friction_const=friction_const,
        mu_d=mu_d,
        normalize_by_load=normalize_dynamic_by_load,
    )
    f_stick = stiction_force_scalar(
        tangential_velocity,
        normal_force_mag,
        mu_s=mu_s_f32,
        k_stick=k_stick_f32,
        tangential_load=tangential_load,
    )
    alpha = stiction_blend_alpha(tangential_velocity, switch_speed_f32)
    f_blended = (1.0 - alpha) * f_stick + alpha * f_dyn
    use_stiction = (mu_s_f32 > 0.0) & (k_stick_f32 > 0.0) & (switch_speed_f32 > 0.0)
    return jnp.where(use_stiction, f_blended, f_dyn)
