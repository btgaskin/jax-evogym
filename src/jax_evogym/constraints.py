"""Position-Based Dynamics (PBD) edge constraints.

Matches C++ PhysicsEngine.cpp:608-713 in solver constants and phase order.
Phase 1: all springs, ±25% tolerance, factor 0.8
Phase 2: springs selected by ``rigid_spring_mask``, ±3% tolerance, factor 0.5
"""

import jax
import jax.numpy as jnp


def resolve_edge_constraints(
    positions: jnp.ndarray,
    fixed: jnp.ndarray,
    topology,
    init_rest_length: jnp.ndarray,
    spring_mask: jnp.ndarray,
    point_mask: jnp.ndarray,
    rigid_spring_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Apply PBD Phase 1 + Phase 2 edge constraints. Returns corrected positions.

    Phase 1 (C++ lines 610-671): all springs, ±25%, correction factor 0.8.
    Phase 2 (C++ lines 675-710): springs in ``rigid_spring_mask``, ±3%,
    correction factor 0.5.
    Phase 2 does not check fixed points, matching the C++ solver branch.
    Which springs enter Phase 2 is determined by the builders; current JAX
    builders include springs from ``RIGID`` cells and from ``FIXED`` cells that
    remain in dynamic mixed objects.
    """
    a_idx = topology.a_idx
    b_idx = topology.b_idx

    # --- Phase 1: all springs, ±25%, factor 0.8 ---
    positions = _pbd_phase(
        positions, fixed, a_idx, b_idx, init_rest_length, spring_mask,
        thresh=0.25, factor=0.8, check_fixed=True,
    )

    # --- Phase 2: rigid springs only, ±3%, factor 0.5 ---
    phase2_mask = rigid_spring_mask & spring_mask
    positions = _pbd_phase(
        positions, fixed, a_idx, b_idx, init_rest_length, phase2_mask,
        thresh=0.03, factor=0.5, check_fixed=False,
    )

    return positions


def apply_friction_position_constraint(
    positions: jnp.ndarray,
    *,
    fixed: jnp.ndarray,
    point_mask: jnp.ndarray,
    friction_anchor: jnp.ndarray,
    friction_anchor_active: jnp.ndarray,
    friction_anchor_no_contact_count: jnp.ndarray,
    contact_mask: jnp.ndarray,
    contact_normal: jnp.ndarray,
    normal_force_mag: jnp.ndarray,
    gravity: float,
    point_mass: float,
    static_friction_const: float,
    anchor_stiffness: float,
    correction: float,
    persist_substeps: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply a terrain-contact tangential anchor constraint in position space.

    This constraint runs after spring PBD and before finite-difference velocity
    recomputation. It only applies tangential correction; normal penetration
    remains handled by existing collision/PBD stages.
    """
    pos = jnp.asarray(positions, dtype=jnp.float32)
    point_mask_b = jnp.asarray(point_mask, dtype=jnp.bool_)
    fixed_mask = jnp.asarray(fixed, dtype=jnp.bool_)

    anchor = jnp.asarray(friction_anchor, dtype=jnp.float32)
    anchor_active_prev = jnp.asarray(friction_anchor_active, dtype=jnp.bool_) & point_mask_b
    no_contact_count_prev = (
        jnp.asarray(friction_anchor_no_contact_count, dtype=jnp.int32)
        * point_mask_b.astype(jnp.int32)
    )
    in_contact = jnp.asarray(contact_mask, dtype=jnp.bool_) & point_mask_b

    persist = jnp.maximum(jnp.asarray(persist_substeps, dtype=jnp.int32), jnp.int32(0))
    no_contact_count_next = jnp.where(
        in_contact,
        jnp.int32(0),
        jnp.where(
            anchor_active_prev,
            no_contact_count_prev + jnp.int32(1),
            no_contact_count_prev,
        ),
    )
    alive_prev = anchor_active_prev & (no_contact_count_prev < persist)
    anchor_alive = in_contact | (anchor_active_prev & (no_contact_count_next < persist))
    entering = in_contact & ~alive_prev
    anchor_curr = jnp.where(entering[:, None], pos, anchor)

    normal = jnp.asarray(contact_normal, dtype=jnp.float32)
    normal_norm = jnp.linalg.norm(normal, axis=-1, keepdims=True)
    normal_safe = normal / jnp.maximum(normal_norm, 1e-10)

    delta = pos - anchor_curr
    delta_n = jnp.sum(delta * normal_safe, axis=-1, keepdims=True)
    delta_t = delta - delta_n * normal_safe
    tang_mag = jnp.linalg.norm(delta_t, axis=-1)

    mu_s = jnp.asarray(static_friction_const, dtype=jnp.float32)
    k_anchor = jnp.maximum(jnp.asarray(anchor_stiffness, dtype=jnp.float32), 1e-10)
    normal_force_contact = jnp.asarray(normal_force_mag, dtype=jnp.float32)
    # Penalty normals collapse near zero after settling; keep a quasistatic support term.
    support_normal = (
        jnp.asarray(point_mass, dtype=jnp.float32)
        * jnp.abs(jnp.asarray(gravity, dtype=jnp.float32))
        * jnp.abs(normal_safe[:, 1])
    )
    normal_force_eff = jnp.where(
        in_contact,
        jnp.maximum(normal_force_contact, support_normal),
        normal_force_contact,
    )
    cap_disp = mu_s * normal_force_eff / k_anchor

    # Contacts with near-zero normals cannot define a tangent robustly; treat as slip.
    normal_valid = jnp.squeeze(normal_norm, axis=-1) > 1e-6
    sticking = in_contact & anchor_alive & normal_valid & (tang_mag <= cap_disp)
    apply_tangential = sticking[:, None].astype(jnp.float32)
    apply_normal = (in_contact & anchor_alive & normal_valid)[:, None].astype(jnp.float32)
    corr_gain = jnp.asarray(correction, dtype=jnp.float32)
    normal_hold_gain = jnp.float32(0.35)
    delta_n_away = jnp.maximum(delta_n, 0.0)
    normal_hold = -normal_safe * delta_n_away * normal_hold_gain
    pos_correction = (-delta_t * apply_tangential + normal_hold * apply_normal) * corr_gain
    pos_correction = jnp.where(fixed_mask, 0.0, pos_correction)
    pos_next = pos + pos_correction

    breaking = in_contact & ~sticking
    reset_anchor = entering | breaking
    anchor_next = jnp.where(reset_anchor[:, None], pos_next, anchor_curr)
    anchor_next = jnp.where(point_mask_b[:, None], anchor_next, anchor)
    active_next = anchor_alive & point_mask_b
    no_contact_count_next = jnp.where(anchor_alive, no_contact_count_next, jnp.int32(0))
    no_contact_count_next = jnp.where(
        point_mask_b,
        no_contact_count_next,
        jnp.int32(0),
    )
    return pos_next, anchor_next, active_next, no_contact_count_next


def _pbd_phase(
    positions, fixed, a_idx, b_idx, init_rest_length, active_mask,
    thresh, factor, check_fixed,
):
    """Single PBD correction phase."""
    vec_q_to_p = positions[a_idx] - positions[b_idx]  # (n_springs, 2)
    dist = jnp.sqrt(jnp.sum(vec_q_to_p ** 2, axis=1))  # (n_springs,)
    dist_safe = jnp.maximum(dist, 1e-10)

    stress = dist / jnp.maximum(init_rest_length, 1e-10)

    # Overshoot: stress > 1 + thresh
    overshoot_mag = (dist - init_rest_length * (1 + thresh)) / (dist_safe * 2)
    is_overshoot = stress > (1 + thresh)
    overshoot_mag = jnp.where(is_overshoot, overshoot_mag, 0.0)

    # Undershoot: stress < 1 - thresh
    undershoot_mag = (dist - init_rest_length * (1 - thresh)) / (dist_safe * 2)
    is_undershoot = stress < (1 - thresh)
    undershoot_mag = jnp.where(is_undershoot, undershoot_mag, 0.0)

    # Combined correction per spring
    total_mag = (overshoot_mag + undershoot_mag)  # (n_springs,)
    correction = vec_q_to_p * total_mag[:, None] * factor  # (n_springs, 2)

    # Mask inactive springs
    correction = correction * active_mask[:, None]

    if check_fixed:
        # Fixed points don't move (Phase 1 only)
        a_fixed = fixed[a_idx]  # (n_springs, 2) bool
        b_fixed = fixed[b_idx]
        corr_a = jnp.where(a_fixed, 0.0, correction)
        corr_b = jnp.where(b_fixed, 0.0, correction)
    else:
        # Phase 2: no per-endpoint fixed-point gating.
        # physics_substep() re-snaps anchored points after the PBD pass.
        corr_a = correction
        corr_b = correction

    # Deterministic endpoint accumulation (avoids repeated-index scatter-add on GPU).
    n_points = int(positions.shape[0])
    a_weights = jax.nn.one_hot(a_idx, n_points, dtype=positions.dtype)
    b_weights = jax.nn.one_hot(b_idx, n_points, dtype=positions.dtype)
    positions = positions + a_weights.T @ (-corr_a) + b_weights.T @ corr_b

    return positions
