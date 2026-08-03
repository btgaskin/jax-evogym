"""Deformation observation computation — pure JAX, JIT-compatible.

Computes per-voxel spring strain as proprioceptive deformation sensing.
This is the key enhancement over original C++ EvoGym: robots get feedback
about their body configuration changes.

Deformation ratio = current_length / reference_rest_length:
  1.0 = no deformation
  >1  = extension
  <1  = compression

Reference rest length can be either:
  - initial rest length (absolute deformation), or
  - current actuated rest length (compliance deformation).
"""

import jax.numpy as jnp

from .types import DeformationInfo, SpringTopology


def _spring_lengths(
    positions: jnp.ndarray,
    topology: SpringTopology,
) -> jnp.ndarray:
    """Compute current physical spring lengths."""
    pos_a = positions[topology.a_idx]
    pos_b = positions[topology.b_idx]
    return jnp.sqrt(jnp.sum((pos_a - pos_b) ** 2, axis=-1) + 1e-12)


def _deformation_from_lengths(
    lengths: jnp.ndarray,
    reference_rest_length: jnp.ndarray,
    deformation_info: DeformationInfo,
) -> jnp.ndarray:
    """Project spring-length ratios onto per-voxel horizontal/vertical channels."""
    ratios = lengths / jnp.maximum(reference_rest_length, 1e-12)

    # Pad ratios with a 1.0 at index -1 for sentinel handling.
    ratios_padded = jnp.concatenate([ratios, jnp.ones(1)])

    h_idx = deformation_info.voxel_horiz_springs
    h_ratios = ratios_padded[h_idx]
    h_mask = h_idx >= 0
    deform_x = jnp.where(h_mask, h_ratios, 1.0).mean(axis=-1)

    v_idx = deformation_info.voxel_vert_springs
    v_ratios = ratios_padded[v_idx]
    v_mask = v_idx >= 0
    deform_y = jnp.where(v_mask, v_ratios, 1.0).mean(axis=-1)

    return jnp.stack([deform_x, deform_y], axis=-1)


def compute_deformation(
    positions: jnp.ndarray,
    topology: SpringTopology,
    spring_init_rest_length: jnp.ndarray,
    deformation_info: DeformationInfo,
    spring_rest_length: jnp.ndarray | None = None,
    reference: str = "init",
) -> jnp.ndarray:
    """Compute per-voxel deformation → (n_robot_voxels, 2) [deform_x, deform_y].

    Deformation ratio per spring is current physical length divided by either:
      - initial rest length (`reference="init"`), or
      - current actuated rest length (`reference="current"`).

    Args:
        positions: (n_points, 2) current point positions.
        topology: Spring connectivity (a_idx, b_idx).
        spring_init_rest_length: (n_springs,) initial rest lengths.
        deformation_info: Precomputed voxel→spring mapping.
        spring_rest_length: (n_springs,) current rest lengths. Required when
            reference="current".
        reference: Length reference mode. One of {"init", "current"}.

    Returns:
        (n_robot_voxels, 2) array: [deform_x, deform_y] per voxel.
    """
    if reference == "init":
        ref_rest_length = spring_init_rest_length
    elif reference == "current":
        if spring_rest_length is None:
            raise ValueError(
                "spring_rest_length is required when reference='current'."
            )
        ref_rest_length = spring_rest_length
    else:
        raise ValueError(
            f"Invalid reference={reference!r}. Expected 'init' or 'current'."
        )

    lengths = _spring_lengths(positions, topology)
    return _deformation_from_lengths(lengths, ref_rest_length, deformation_info)


def compute_dual_deformation(
    positions: jnp.ndarray,
    topology: SpringTopology,
    spring_init_rest_length: jnp.ndarray,
    deformation_info: DeformationInfo,
    spring_rest_length: jnp.ndarray,
) -> jnp.ndarray:
    """Compute absolute and compliance deformation from one spring-length pass."""
    lengths = _spring_lengths(positions, topology)
    abs_def = _deformation_from_lengths(lengths, spring_init_rest_length, deformation_info)
    comp_def = _deformation_from_lengths(lengths, spring_rest_length, deformation_info)
    return jnp.concatenate([abs_def, comp_def], axis=-1)
