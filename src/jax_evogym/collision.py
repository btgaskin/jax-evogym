"""Direct object-separated collision detection and response."""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import constants as C
from .friction import blended_friction_scalar, dynamic_friction_scalar
from .types import CollisionData, PhysicsConstants, StaticColliderData


class StaticCollisionResult(NamedTuple):
    forces: jnp.ndarray               # (n_points, 2)
    contact_mask: jnp.ndarray         # (n_points,) bool, true colliding contact only
    contact_normal: jnp.ndarray       # (n_points, 2) unit normal (dominant colliding contact)
    contact_tangent: jnp.ndarray      # (n_points, 2) unit tangent (dominant colliding contact)
    contact_normal_force: jnp.ndarray # (n_points,) max |N| on colliding contact
    memory_contact_mask: jnp.ndarray  # (n_points,) bool, colliding contacts eligible for memory-state updates
    memory_contact_normal: jnp.ndarray  # (n_points, 2) normal for dominant memory-eligible contact
    memory_contact_tangent: jnp.ndarray  # (n_points, 2) tangent for dominant memory-eligible contact
    memory_contact_normal_force: jnp.ndarray  # (n_points,) max |N| on dominant memory-eligible contact
    support_mask: jnp.ndarray         # (n_points,) bool, non-penetrating support only
    support_normal: jnp.ndarray       # (n_points, 2) unit normal (dominant support contact)
    support_tangent: jnp.ndarray      # (n_points, 2) unit tangent (dominant support contact)
    support_normal_force: jnp.ndarray # (n_points,) support load used for grip cap
    static_manifold_cell_idx: jnp.ndarray  # (n_points,) int32
    static_manifold_edge_idx: jnp.ndarray  # (n_points,) int32
    static_manifold_active: jnp.ndarray  # (n_points,) bool
    static_manifold_no_contact_count: jnp.ndarray  # (n_points,) int32
    static_manifold_normal_force: jnp.ndarray  # (n_points,) float32
    static_manifold_normal: jnp.ndarray  # (n_points, 2) float32
    diagnostics: "StaticCollisionDiagnostics"


class StaticCollisionDiagnostics(NamedTuple):
    allocated_surface_slot_count: jnp.ndarray
    active_surface_boxel_count: jnp.ndarray
    allocated_static_slot_count: jnp.ndarray
    active_static_cell_count: jnp.ndarray
    allocated_worklist_count: jnp.ndarray
    active_worklist_count_hint: jnp.ndarray
    active_triple_count: jnp.ndarray
    inside_count: jnp.ndarray
    viable_count: jnp.ndarray
    colliding_count: jnp.ndarray
    stiction_active_count: jnp.ndarray
    floor_stiction_active_count: jnp.ndarray
    slope_contact_count: jnp.ndarray
    floor_contact_count: jnp.ndarray
    wall_contact_count: jnp.ndarray
    contact_point_count: jnp.ndarray
    tangent_degenerate_point_count: jnp.ndarray
    sum_normal_force: jnp.ndarray
    sum_abs_v_tang: jnp.ndarray
    sum_abs_friction: jnp.ndarray
    sum_stick_cap: jnp.ndarray
    friction_cap_count: jnp.ndarray
    sum_cap_disp: jnp.ndarray
    sum_step_slip_disp: jnp.ndarray
    memory_clamped_count: jnp.ndarray
    persistent_contact_count: jnp.ndarray
    persistent_point_count: jnp.ndarray
    persistent_without_collision_count: jnp.ndarray
    persistent_candidate_count: jnp.ndarray
    persistent_block_no_prev_manifold_count: jnp.ndarray
    persistent_block_manifold_mismatch_count: jnp.ndarray
    persistent_block_persist_window_count: jnp.ndarray
    persistent_block_gap_count: jnp.ndarray
    persistent_block_release_speed_count: jnp.ndarray
    sum_persistent_abs_friction: jnp.ndarray
    support_point_count: jnp.ndarray
    support_without_collision_count: jnp.ndarray
    support_slope_count: jnp.ndarray
    support_floor_count: jnp.ndarray
    support_wall_count: jnp.ndarray
    sum_support_normal_force: jnp.ndarray
    sum_support_abs_friction: jnp.ndarray


class StaticCollisionWorklist(NamedTuple):
    triple_i: jnp.ndarray
    triple_j: jnp.ndarray
    triple_k: jnp.ndarray
    triple_active: jnp.ndarray
    allocated_surface_slot_count: jnp.ndarray
    active_surface_boxel_count: jnp.ndarray
    allocated_static_slot_count: jnp.ndarray
    active_static_cell_count: jnp.ndarray
    allocated_worklist_count: jnp.ndarray
    active_worklist_count_hint: jnp.ndarray


class DynamicBroadphaseCollisionProbe(NamedTuple):
    """Probe-only dynamic broadphase diagnostics for one collision solve."""

    selected_pair_mask: jnp.ndarray
    overflow: jnp.ndarray
    selected_triple_active: jnp.ndarray
    triple_active: jnp.ndarray


def _deterministic_sum_by_index(
    indices: jnp.ndarray,
    values: jnp.ndarray,
    *,
    size: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Deterministic index accumulation via dense one-hot reduction."""
    weights = jax.nn.one_hot(indices, size, dtype=dtype)
    return weights.T @ values


def _zero_static_collision_diagnostics() -> StaticCollisionDiagnostics:
    zf = jnp.float32(0.0)
    zi = jnp.int32(0)
    return StaticCollisionDiagnostics(
        allocated_surface_slot_count=zi,
        active_surface_boxel_count=zi,
        allocated_static_slot_count=zi,
        active_static_cell_count=zi,
        allocated_worklist_count=zi,
        active_worklist_count_hint=zi,
        active_triple_count=zi,
        inside_count=zi,
        viable_count=zi,
        colliding_count=zi,
        stiction_active_count=zi,
        floor_stiction_active_count=zi,
        slope_contact_count=zi,
        floor_contact_count=zi,
        wall_contact_count=zi,
        contact_point_count=zi,
        tangent_degenerate_point_count=zi,
        sum_normal_force=zf,
        sum_abs_v_tang=zf,
        sum_abs_friction=zf,
        sum_stick_cap=zf,
        friction_cap_count=zi,
        sum_cap_disp=zf,
        sum_step_slip_disp=zf,
        memory_clamped_count=zi,
        persistent_contact_count=zi,
        persistent_point_count=zi,
        persistent_without_collision_count=zi,
        persistent_candidate_count=zi,
        persistent_block_no_prev_manifold_count=zi,
        persistent_block_manifold_mismatch_count=zi,
        persistent_block_persist_window_count=zi,
        persistent_block_gap_count=zi,
        persistent_block_release_speed_count=zi,
        sum_persistent_abs_friction=zf,
        support_point_count=zi,
        support_without_collision_count=zi,
        support_slope_count=zi,
        support_floor_count=zi,
        support_wall_count=zi,
        sum_support_normal_force=zf,
        sum_support_abs_friction=zf,
    )


@lru_cache(maxsize=None)
def _cached_grid_data(grid_h: int, grid_w: int):
    from .jax_utils import precompute_grid

    return precompute_grid(int(grid_h), int(grid_w))


def _default_robot_cell_mask(morphology: np.ndarray) -> np.ndarray:
    """Legacy classification: dynamic voxels are robot, fixed/slope voxels are terrain."""
    return (morphology != C.EMPTY) & ~np.isin(
        morphology,
        np.array(sorted(C.STATIC_TERRAIN_TYPES), dtype=np.int32),
    )


def make_collision_data(
    morphology: np.ndarray | jnp.ndarray,
    grid_h: int,
    grid_w: int,
    robot_cell_mask: np.ndarray | jnp.ndarray | None = None,
    *,
    max_triples: int = C.MAX_TRIPLES,
    max_self_triples: int = C.MAX_SELF_TRIPLES,
    max_active_boxels: int | None = None,
) -> CollisionData:
    """Compatibility constructor for grid-based collision metadata.

    This keeps the legacy experiment/tests working while delegating the actual
    indexed-triple construction to the JAX-native builder.
    """
    morphology_np = np.asarray(morphology, dtype=np.int32)
    expected_shape = (int(grid_h), int(grid_w))
    if morphology_np.shape != expected_shape:
        raise ValueError(
            f"morphology shape {morphology_np.shape} does not match "
            f"(grid_h, grid_w)={expected_shape}"
        )

    if robot_cell_mask is None:
        robot_mask_np = _default_robot_cell_mask(morphology_np)
    else:
        robot_mask_np = np.asarray(robot_cell_mask, dtype=bool)
        if robot_mask_np.shape != morphology_np.shape:
            raise ValueError(
                f"robot_cell_mask shape {robot_mask_np.shape} does not match "
                f"morphology shape {morphology_np.shape}"
            )

    from .jax_utils import jax_build_collision_data

    collision_data = jax_build_collision_data(
        jnp.asarray(morphology_np, dtype=jnp.int32),
        jnp.asarray(robot_mask_np, dtype=jnp.bool_),
        _cached_grid_data(int(grid_h), int(grid_w)),
        max_triples=int(max_triples),
        max_self_triples=int(max_self_triples),
        max_active_boxels=(
            None if max_active_boxels is None else int(max_active_boxels)
        ),
    )
    n_triples = int(np.asarray(collision_data.n_triples).item())
    if n_triples > int(max_triples):
        raise ValueError(
            f"Dynamic collision triple count {n_triples} exceeds max_triples={max_triples}"
        )
    n_self_triples = int(np.asarray(collision_data.n_self_triples).item())
    if n_self_triples > int(max_self_triples):
        raise ValueError(
            f"Self collision triple count {n_self_triples} exceeds "
            f"max_self_triples={max_self_triples}"
        )
    return collision_data


def check_triple_capacity(
    grid_h: int,
    grid_w: int,
    canvas_size: int | None = None,
    canvas_h: int | None = None,
    canvas_w: int | None = None,
    max_active: int | None = None,
    max_triples: int = C.MAX_TRIPLES,
    max_self_triples: int = C.MAX_SELF_TRIPLES,
) -> tuple[int, int]:
    """Return worst-case dynamic triple bounds for a fixed robot frame."""
    if canvas_h is not None or canvas_w is not None:
        if canvas_h is None or canvas_w is None:
            raise ValueError("canvas_h and canvas_w must both be provided")
        n_canvas = int(canvas_h) * int(canvas_w)
    else:
        if canvas_size is None:
            canvas_size = max(grid_h, grid_w)
        n_canvas = int(canvas_size) * int(canvas_size)
    n_max = int(max_active if max_active is not None else grid_h * grid_w)
    triple_bound = n_max * max(0, n_max - 9) * 4
    self_bound = n_canvas * max(0, n_canvas - 9) * 4
    if triple_bound > max_triples:
        raise ValueError(
            f"Up to {n_max} active boxels may produce {triple_bound} triples, "
            f"exceeding max_triples={max_triples}."
        )
    if self_bound > max_self_triples:
        raise ValueError(
            f"Canvas bound {self_bound} exceeds max_self_triples={max_self_triples}."
        )
    return triple_bound, self_bound


def _cross2d(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def point_in_quad(point, quad_corners_pos):
    """Test if a point lies inside a quad."""
    bl = quad_corners_pos[..., 0, :]
    br = quad_corners_pos[..., 1, :]
    tr = quad_corners_pos[..., 2, :]
    tl = quad_corners_pos[..., 3, :]

    tl_r = tl - point
    tr_r = tr - point
    bl_r = bl - point
    br_r = br - point

    a = _cross2d(tl_r, tr_r)
    b = _cross2d(tr_r, br_r)
    c = _cross2d(br_r, tl_r)
    in_tri1 = (c * a >= 0.0) & (c * b >= 0.0)

    d = _cross2d(tl_r, bl_r)
    e = _cross2d(bl_r, br_r)
    in_tri2 = (c * d >= 0.0) & (c * e >= 0.0)
    return in_tri1 | in_tri2


def point_in_convex_cell(point, edge_a_pos, edge_b_pos, edge_mask):
    """Test if a point lies inside a convex cell using outward edge normals."""
    slope = edge_b_pos - edge_a_pos
    outward_normals = jnp.stack([-slope[..., 1], slope[..., 0]], axis=-1)
    signed_dist = jnp.sum((point[:, None, :] - edge_a_pos) * outward_normals, axis=-1)
    return jnp.all((signed_dist <= 1e-6) | ~edge_mask, axis=-1)


def edge_intersection(a1, a2, b1, b2):
    """Test if two segments intersect."""
    diff1 = a2 - a1
    diff2 = b2 - b1
    det = -diff1[..., 0] * diff2[..., 1] + diff2[..., 0] * diff1[..., 1]
    parallel = jnp.abs(det) < 1e-4

    x_diff = b1[..., 0] - a1[..., 0]
    y_diff = b1[..., 1] - a1[..., 1]
    det_safe = jnp.where(parallel, 1.0, det)
    t = (-diff2[..., 1] * x_diff + diff2[..., 0] * y_diff) / det_safe
    s = (-diff1[..., 1] * x_diff + diff1[..., 0] * y_diff) / det_safe

    tol = 1e-6
    in_range = (t >= -tol) & (t <= 1.0 + tol) & (s >= -tol) & (s <= 1.0 + tol)
    return in_range & ~parallel


def dist_point_to_edge(point, edge_a, edge_b):
    vec_base = edge_b - edge_a
    base_sq = jnp.sum(vec_base ** 2, axis=-1)
    vec_slant = point - edge_a
    t = jnp.sum(vec_slant * vec_base, axis=-1) / jnp.maximum(base_sq, 1e-10)
    t = jnp.clip(t, 0.0, 1.0)
    closest = edge_a + t[..., None] * vec_base
    delta = point - closest
    dist = jnp.sqrt(jnp.sum(delta ** 2, axis=-1))
    return jnp.where(base_sq < 1e-12, 1000.0, dist)


def _surface_scope_active(
    scope: int | jnp.ndarray,
    upward_contact: jnp.ndarray,
    is_sloped_edge: jnp.ndarray,
) -> jnp.ndarray:
    """Return whether a surface is included by a terrain/slope scope selector."""
    scope_i32 = jnp.asarray(scope, dtype=jnp.int32)
    return (
        (scope_i32 == jnp.int32(C.STICTION_SCOPE_ALL_STATIC))
        | (
            (scope_i32 == jnp.int32(C.STICTION_SCOPE_FLOOR_SLOPE_ONLY))
            & upward_contact
        )
        | (
            (scope_i32 == jnp.int32(C.STICTION_SCOPE_SLOPE_ONLY))
            & upward_contact
            & is_sloped_edge
        )
    )


def _compute_friction_with_stiction(
    v_tang: jnp.ndarray,
    normal_force_mag: jnp.ndarray,
    unit_tangent: jnp.ndarray,
    *,
    friction_const: float,
    mu_d: float | jnp.ndarray,
    mu_s: float,
    k_stick: jnp.ndarray | float,
    switch_speed: jnp.ndarray | float,
    tangential_load: jnp.ndarray | float | None = None,
) -> jnp.ndarray:
    """Tangential friction with optional stick-slip blending.

    Setting ``mu_s <= 0`` or ``k_stick <= 0`` falls back to pure dynamic friction.
    Static terrain uses the load-normalized dynamic branch.
    """
    v_tang = jnp.asarray(v_tang, dtype=jnp.float32)
    f_total = blended_friction_scalar(
        v_tang,
        normal_force_mag,
        friction_const=friction_const,
        mu_d=mu_d,
        mu_s=mu_s,
        k_stick=k_stick,
        switch_speed=switch_speed,
        tangential_load=tangential_load,
        normalize_dynamic_by_load=True,
    )
    return unit_tangent * f_total[:, None]


def _resolve_dynamic_triples(
    positions: jnp.ndarray,
    velocities_true: jnp.ndarray,
    collision_data: CollisionData,
    constants: PhysicsConstants,
    triple_active: jnp.ndarray,
    max_narrow: int | None = None,
    n_active_hint: jnp.ndarray | int | None = None,
) -> jnp.ndarray:
    # Avoid bool->numeric casts in runtime force gating; use predicate selects.
    cd = collision_data
    n_points = positions.shape[0]
    expanded_positions = jnp.concatenate([positions, positions], axis=0)
    expanded_velocities = jnp.concatenate([velocities_true, velocities_true], axis=0)
    corner_pos = expanded_positions[cd.boxel_corners]

    def _solve_triples(ti, tj, tk, active_mask):
        point_idx = cd.boxel_corners[ti, tk]
        point_pos = expanded_positions[point_idx]
        quad_pos = corner_pos[tj]

        inside = point_in_quad(point_pos, quad_pos) & active_mask
        cei = cd.corner_edge_indices
        main_ea_idx = cd.boxel_edge_a[ti[:, None], cei[tk]]
        main_eb_idx = cd.boxel_edge_b[ti[:, None], cei[tk]]
        main_ea_pos = expanded_positions[main_ea_idx]
        main_eb_pos = expanded_positions[main_eb_idx]
        ref_ea_pos = expanded_positions[cd.boxel_edge_a[tj]]
        ref_eb_pos = expanded_positions[cd.boxel_edge_b[tj]]

        intersects = edge_intersection(
            main_ea_pos[:, :, None, :],
            main_eb_pos[:, :, None, :],
            ref_ea_pos[:, None, :, :],
            ref_eb_pos[:, None, :, :],
        )
        intersects_any = jnp.any(intersects, axis=1)
        surface = cd.surface_edge_mask[tj]
        viable = surface & intersects_any & inside[:, None]

        dists = dist_point_to_edge(point_pos[:, None, :], ref_ea_pos, ref_eb_pos)
        dists_masked = jnp.where(viable, dists, 1e10)
        min_edge = jnp.argmin(dists_masked, axis=-1)
        min_dist = jnp.min(dists_masked, axis=-1)
        has_viable = jnp.any(viable, axis=-1)
        colliding = inside & has_viable

        sel_re_a_idx = cd.boxel_edge_a[tj, min_edge]
        sel_re_b_idx = cd.boxel_edge_b[tj, min_edge]
        sel_re_a_pos = expanded_positions[sel_re_a_idx]
        sel_re_b_pos = expanded_positions[sel_re_b_idx]
        slope = sel_re_b_pos - sel_re_a_pos
        raw_normal = jnp.stack([-slope[..., 1], slope[..., 0]], axis=-1)
        normal_len = jnp.sqrt(jnp.sum(raw_normal ** 2, axis=-1, keepdims=True))
        unit_normal = raw_normal / jnp.maximum(normal_len, 1e-10)

        point_vel = expanded_velocities[point_idx]
        force_mag = constants.collision_const_obj * (constants.collision_base_dist + min_dist)
        normal_force = unit_normal * force_mag[:, None]

        vel_normal = jnp.sum(point_vel * unit_normal, axis=-1, keepdims=True)
        damping = -unit_normal * vel_normal * constants.collision_vel_damping

        ea_vel = expanded_velocities[sel_re_a_idx]
        eb_vel = expanded_velocities[sel_re_b_idx]
        ea_vel_normal = jnp.sum(ea_vel * unit_normal, axis=-1, keepdims=True)
        eb_vel_normal = jnp.sum(eb_vel * unit_normal, axis=-1, keepdims=True)
        ea_damping = unit_normal * ea_vel_normal * constants.collision_vel_damping
        eb_damping = unit_normal * eb_vel_normal * constants.collision_vel_damping

        unit_tangent = jnp.stack([-unit_normal[..., 1], unit_normal[..., 0]], axis=-1)
        v_tang = jnp.sum(point_vel * unit_tangent, axis=-1)
        flip = jnp.where(v_tang < 0, -1.0, 1.0)
        unit_tangent_f = unit_tangent * flip[:, None]
        v_tang_abs = jnp.abs(v_tang)
        normal_force_mag = jnp.sqrt(jnp.sum(normal_force ** 2, axis=-1))
        denom = jnp.maximum(constants.dynamic_friction_const * normal_force_mag, 1e-10)
        friction = (
            -constants.friction_const
            * constants.dynamic_friction_const
            * normal_force_mag
            * jnp.tanh(v_tang_abs / denom)
        )
        friction_force = unit_tangent_f * friction[:, None]

        point_force = normal_force + damping + friction_force
        ea_force = -(normal_force / 2 + ea_damping / 2 + friction_force / 2)
        eb_force = -(normal_force / 2 + eb_damping / 2 + friction_force / 2)

        colliding_mask = colliding[:, None]
        point_force = jnp.where(colliding_mask, point_force, 0.0)
        ea_force = jnp.where(colliding_mask, ea_force, 0.0)
        eb_force = jnp.where(colliding_mask, eb_force, 0.0)

        expanded_point_count = int(n_points * 2)
        collision_forces = _deterministic_sum_by_index(
            point_idx,
            point_force,
            size=expanded_point_count,
            dtype=point_force.dtype,
        )
        collision_forces = collision_forces + _deterministic_sum_by_index(
            sel_re_a_idx,
            ea_force,
            size=expanded_point_count,
            dtype=ea_force.dtype,
        )
        collision_forces = collision_forces + _deterministic_sum_by_index(
            sel_re_b_idx,
            eb_force,
            size=expanded_point_count,
            dtype=eb_force.dtype,
        )
        return collision_forces.reshape(2, n_points, 2).sum(axis=0)

    n_total = int(triple_active.shape[0])
    if n_active_hint is None:
        n_active = jnp.sum(jnp.where(triple_active, jnp.int32(1), jnp.int32(0)))
    else:
        n_active = jnp.asarray(n_active_hint, dtype=jnp.int32)
    if max_narrow is None or max_narrow <= 0:
        active_idx = jnp.nonzero(triple_active, size=n_total, fill_value=0)[0]
        compact_active = jnp.arange(n_total, dtype=jnp.int32) < n_active
        return _solve_triples(
            cd.triple_i[active_idx],
            cd.triple_j[active_idx],
            cd.triple_k[active_idx],
            compact_active,
        )
    if max_narrow >= n_total:
        active_idx = jnp.nonzero(triple_active, size=n_total, fill_value=0)[0]
        compact_active = jnp.arange(n_total, dtype=jnp.int32) < n_active
        return jax.lax.cond(
            n_active < n_total,
            lambda _: _solve_triples(
                cd.triple_i[active_idx],
                cd.triple_j[active_idx],
                cd.triple_k[active_idx],
                compact_active,
            ),
            lambda _: _solve_triples(cd.triple_i, cd.triple_j, cd.triple_k, triple_active),
            operand=None,
        )

    narrow_cap = jnp.asarray(int(max_narrow), dtype=jnp.int32)
    overflow = n_active > narrow_cap
    active_idx = jnp.nonzero(triple_active, size=int(max_narrow), fill_value=0)[0]
    compact_active = jnp.arange(int(max_narrow), dtype=jnp.int32) < jnp.minimum(
        n_active, narrow_cap
    )

    return jax.lax.cond(
        overflow,
        lambda _: _solve_triples(cd.triple_i, cd.triple_j, cd.triple_k, triple_active),
        lambda _: _solve_triples(
            cd.triple_i[active_idx],
            cd.triple_j[active_idx],
            cd.triple_k[active_idx],
            compact_active,
        ),
        operand=None,
    )


def resolve_collisions_static_indexed(
    positions: jnp.ndarray,
    velocities_true: jnp.ndarray,
    collision_data: CollisionData,
    constants: PhysicsConstants,
    max_narrow: int | None = None,
) -> jnp.ndarray:
    return _resolve_dynamic_triples(
        positions,
        velocities_true,
        collision_data,
        constants,
        collision_data.triple_active,
        max_narrow=max_narrow,
        n_active_hint=collision_data.n_triples,
    )


def build_static_collision_worklist(
    collision_data: CollisionData,
    static_collider_data: StaticColliderData,
    *,
    max_surface_boxels: int | None = None,
    max_surface_columns: int | None = None,
) -> StaticCollisionWorklist:
    """Build the dynamic/static/corner worklist for one env's fixed geometry."""
    del max_surface_columns
    n_dynamic_slots = int(collision_data.boxel_mask.shape[0])
    n_static_cells = int(static_collider_data.cell_vertices.shape[0])
    if n_dynamic_slots == 0 or n_static_cells == 0:
        zeros = jnp.zeros((0,), dtype=jnp.int32)
        return StaticCollisionWorklist(
            triple_i=zeros,
            triple_j=zeros,
            triple_k=zeros,
            triple_active=jnp.zeros((0,), dtype=jnp.bool_),
            allocated_surface_slot_count=jnp.int32(0),
            active_surface_boxel_count=jnp.int32(0),
            allocated_static_slot_count=jnp.int32(0),
            active_static_cell_count=jnp.int32(0),
            allocated_worklist_count=jnp.int32(0),
            active_worklist_count_hint=jnp.int32(0),
        )

    dynamic_slot_capacity = n_dynamic_slots
    if max_surface_boxels is not None:
        dynamic_slot_capacity = max(1, min(int(max_surface_boxels), n_dynamic_slots))

    surface_boxel_mask = collision_data.boxel_mask & jnp.any(
        collision_data.surface_edge_mask,
        axis=1,
    )
    dynamic_idx = jnp.nonzero(
        surface_boxel_mask,
        size=dynamic_slot_capacity,
        fill_value=0,
    )[0]
    n_dynamic = jnp.minimum(
        jnp.sum(surface_boxel_mask.astype(jnp.int32)),
        jnp.int32(dynamic_slot_capacity),
    )
    dynamic_active = jnp.arange(dynamic_slot_capacity, dtype=jnp.int32) < n_dynamic

    # Cached worklists must remain valid across the entire rollout, so the
    # static side uses all active terrain cells instead of an initial-position
    # column window.
    static_idx = jnp.arange(n_static_cells, dtype=jnp.int32)
    static_active = static_collider_data.cell_vertex_count > 0
    static_slot_capacity = n_static_cells

    ti = jnp.broadcast_to(
        dynamic_idx[:, None, None],
        (dynamic_slot_capacity, static_slot_capacity, 4),
    ).reshape(-1)
    tj = jnp.broadcast_to(
        static_idx[None, :, None],
        (dynamic_slot_capacity, static_slot_capacity, 4),
    ).reshape(-1)
    tk = jnp.broadcast_to(
        jnp.arange(4, dtype=jnp.int32)[None, None, :],
        (dynamic_slot_capacity, static_slot_capacity, 4),
    ).reshape(-1)
    active_mask = jnp.broadcast_to(
        dynamic_active[:, None, None] & static_active[None, :, None],
        (dynamic_slot_capacity, static_slot_capacity, 4),
    ).reshape(-1)
    active_static_cell_count = jnp.sum(static_active.astype(jnp.int32))
    return StaticCollisionWorklist(
        triple_i=ti,
        triple_j=tj,
        triple_k=tk,
        triple_active=active_mask,
        allocated_surface_slot_count=jnp.int32(dynamic_slot_capacity),
        active_surface_boxel_count=n_dynamic,
        allocated_static_slot_count=jnp.int32(static_slot_capacity),
        active_static_cell_count=active_static_cell_count,
        allocated_worklist_count=jnp.int32(dynamic_slot_capacity * static_slot_capacity * 4),
        active_worklist_count_hint=n_dynamic * active_static_cell_count * jnp.int32(4),
    )


def resolve_static_collisions_with_contacts(
    positions: jnp.ndarray,
    velocities_true: jnp.ndarray,
    collision_data: CollisionData,
    static_collider_data: StaticColliderData | None,
    constants: PhysicsConstants,
    static_collision_worklist: StaticCollisionWorklist | None = None,
    tangential_deformation: jnp.ndarray | None = None,
    static_manifold_cell_idx: jnp.ndarray | None = None,
    static_manifold_edge_idx: jnp.ndarray | None = None,
    static_manifold_active: jnp.ndarray | None = None,
    static_manifold_no_contact_count: jnp.ndarray | None = None,
    static_manifold_normal_force: jnp.ndarray | None = None,
    static_manifold_normal: jnp.ndarray | None = None,
) -> StaticCollisionResult:
    if static_collider_data is None or static_collider_data.cell_vertices.shape[0] == 0:
        zeros_forces = jnp.zeros_like(positions)
        n_points = positions.shape[0]
        return StaticCollisionResult(
            forces=zeros_forces,
            contact_mask=jnp.zeros((n_points,), dtype=jnp.bool_),
            contact_normal=jnp.zeros((n_points, 2), dtype=jnp.float32),
            contact_tangent=jnp.zeros((n_points, 2), dtype=jnp.float32),
            contact_normal_force=jnp.zeros((n_points,), dtype=jnp.float32),
            memory_contact_mask=jnp.zeros((n_points,), dtype=jnp.bool_),
            memory_contact_normal=jnp.zeros((n_points, 2), dtype=jnp.float32),
            memory_contact_tangent=jnp.zeros((n_points, 2), dtype=jnp.float32),
            memory_contact_normal_force=jnp.zeros((n_points,), dtype=jnp.float32),
            support_mask=jnp.zeros((n_points,), dtype=jnp.bool_),
            support_normal=jnp.zeros((n_points, 2), dtype=jnp.float32),
            support_tangent=jnp.zeros((n_points, 2), dtype=jnp.float32),
            support_normal_force=jnp.zeros((n_points,), dtype=jnp.float32),
            static_manifold_cell_idx=jnp.full((n_points,), -1, dtype=jnp.int32),
            static_manifold_edge_idx=jnp.full((n_points,), -1, dtype=jnp.int32),
            static_manifold_active=jnp.zeros((n_points,), dtype=jnp.bool_),
            static_manifold_no_contact_count=jnp.zeros((n_points,), dtype=jnp.int32),
            static_manifold_normal_force=jnp.zeros((n_points,), dtype=jnp.float32),
            static_manifold_normal=jnp.zeros((n_points, 2), dtype=jnp.float32),
            diagnostics=_zero_static_collision_diagnostics(),
        )

    scd = static_collider_data
    n_points = positions.shape[0]
    if tangential_deformation is None:
        tangential_deformation = jnp.zeros((n_points, 2), dtype=jnp.float32)
    if static_manifold_cell_idx is None:
        static_manifold_cell_idx = jnp.full((n_points,), -1, dtype=jnp.int32)
    if static_manifold_edge_idx is None:
        static_manifold_edge_idx = jnp.full((n_points,), -1, dtype=jnp.int32)
    if static_manifold_active is None:
        static_manifold_active = jnp.zeros((n_points,), dtype=jnp.bool_)
    if static_manifold_no_contact_count is None:
        static_manifold_no_contact_count = jnp.zeros((n_points,), dtype=jnp.int32)
    if static_manifold_normal_force is None:
        static_manifold_normal_force = jnp.zeros((n_points,), dtype=jnp.float32)
    if static_manifold_normal is None:
        static_manifold_normal = jnp.zeros((n_points, 2), dtype=jnp.float32)

    if static_collision_worklist is None:
        static_collision_worklist = build_static_collision_worklist(collision_data, scd)

    ti = static_collision_worklist.triple_i
    tj = static_collision_worklist.triple_j
    tk = static_collision_worklist.triple_k
    active_mask = static_collision_worklist.triple_active

    point_idx = collision_data.boxel_corners[ti, tk]
    point_pos = positions[point_idx]
    dynamic_corner_edge_indices = collision_data.corner_edge_indices[tk]
    main_ea_idx = collision_data.boxel_edge_a[ti[:, None], dynamic_corner_edge_indices]
    main_eb_idx = collision_data.boxel_edge_b[ti[:, None], dynamic_corner_edge_indices]
    main_ea_pos = positions[main_ea_idx]
    main_eb_pos = positions[main_eb_idx]
    ref_ea_pos = scd.point_positions[scd.cell_edge_a[tj]]
    ref_eb_pos = scd.point_positions[scd.cell_edge_b[tj]]
    ref_edge_mask = jnp.arange(4, dtype=jnp.int32)[None, :] < scd.cell_vertex_count[tj, None]
    inside = point_in_convex_cell(point_pos, ref_ea_pos, ref_eb_pos, ref_edge_mask) & active_mask

    intersects = edge_intersection(
        main_ea_pos[:, :, None, :],
        main_eb_pos[:, :, None, :],
        ref_ea_pos[:, None, :, :],
        ref_eb_pos[:, None, :, :],
    )
    intersects_any = jnp.any(intersects, axis=1)
    surface = scd.surface_edge_mask[tj] & ref_edge_mask
    viable = surface & intersects_any & inside[:, None]

    dists = dist_point_to_edge(point_pos[:, None, :], ref_ea_pos, ref_eb_pos)
    dists_masked = jnp.where(viable, dists, 1e10)
    min_edge_viable = jnp.argmin(dists_masked, axis=-1)
    min_dist_viable = jnp.min(dists_masked, axis=-1)
    # For slope cells, prefer the hypotenuse (sloped) edge for geometry-only
    # near-contact selection.
    # dist_point_to_edge uses infinite-line distance which can select horizontal/vertical
    # edges over the hypotenuse for points outside slope triangles, giving wrong
    # tangent/normal vectors for sloped-edge normal response.
    edge_vec = ref_eb_pos - ref_ea_pos
    edge_is_sloped = (jnp.abs(edge_vec[..., 0]) > 1e-6) & (jnp.abs(edge_vec[..., 1]) > 1e-6)
    sloped_surface = surface & edge_is_sloped
    has_sloped_surface = jnp.any(sloped_surface, axis=-1)
    surface_dists = jnp.where(
        jnp.where(has_sloped_surface[:, None], sloped_surface, surface),
        dists,
        1e10,
    )
    min_edge_surface = jnp.argmin(surface_dists, axis=-1)
    min_dist_surface = jnp.min(surface_dists, axis=-1)
    has_surface_edge = jnp.any(surface, axis=-1)
    has_viable = jnp.any(viable, axis=-1)
    colliding = inside & has_viable

    min_edge = jnp.where(colliding, min_edge_viable, min_edge_surface)
    min_dist = jnp.where(colliding, min_dist_viable, min_dist_surface)
    sel_re_a_pos = scd.point_positions[scd.cell_edge_a[tj, min_edge]]
    sel_re_b_pos = scd.point_positions[scd.cell_edge_b[tj, min_edge]]
    slope = sel_re_b_pos - sel_re_a_pos
    raw_normal = jnp.stack([-slope[..., 1], slope[..., 0]], axis=-1)
    normal_len = jnp.sqrt(jnp.sum(raw_normal ** 2, axis=-1, keepdims=True))
    unit_normal = raw_normal / jnp.maximum(normal_len, 1e-10)

    point_vel = velocities_true[point_idx]
    force_mag = constants.collision_const_obj * (constants.collision_base_dist + min_dist)
    normal_force = unit_normal * force_mag[:, None]
    vel_normal = jnp.sum(point_vel * unit_normal, axis=-1, keepdims=True)
    damping = -unit_normal * vel_normal * constants.collision_vel_damping

    unit_tangent = jnp.stack([-unit_normal[..., 1], unit_normal[..., 0]], axis=-1)
    v_tang = jnp.sum(point_vel * unit_tangent, axis=-1)
    normal_force_mag = jnp.sqrt(jnp.sum(normal_force ** 2, axis=-1))
    upward_contact = unit_normal[:, 1] > jnp.float32(1e-3)
    slope_eps = jnp.float32(1e-6)
    is_sloped_edge = (jnp.abs(slope[:, 0]) > slope_eps) & (jnp.abs(slope[:, 1]) > slope_eps)
    # Sloped-edge terrain is retained as experimental geometry only. Its normal
    # contact response is still useful for fixture inspection, but the
    # tangential friction/support model is deliberately disabled until a
    # defensible slope friction law is implemented.
    stiction_master = jnp.bool_(constants.stiction_enabled)
    stiction_active = (
        stiction_master
        & _surface_scope_active(
            constants.terrain_stiction_scope,
            upward_contact,
            is_sloped_edge,
        )
        & ~is_sloped_edge
    )
    slope_contact_mode = jnp.asarray(constants.slope_contact_mode, dtype=jnp.int32)
    use_gap_support = slope_contact_mode == jnp.int32(C.SLOPE_CONTACT_MODE_GAP_SUPPORT)
    use_continuity_assist = slope_contact_mode == jnp.int32(
        C.SLOPE_CONTACT_MODE_CONTINUITY_ASSIST
    )
    use_gravity_preload_support = slope_contact_mode == jnp.int32(
        C.SLOPE_CONTACT_MODE_GRAVITY_PRELOAD_EXPERIMENTAL
    )
    slope_contact_active = jnp.zeros_like(is_sloped_edge, dtype=jnp.bool_)
    gap_support_grip_active = jnp.zeros_like(is_sloped_edge, dtype=jnp.bool_)
    vel_normal_scalar = jnp.sum(point_vel * unit_normal, axis=-1)
    legacy_near_contact_dist = jnp.maximum(
        jnp.asarray(constants.collision_base_dist, dtype=jnp.float32),
        jnp.asarray(constants.slope_grip_near_contact_dist, dtype=jnp.float32),
    )
    legacy_release_speed = jnp.asarray(constants.slope_grip_near_release_speed, dtype=jnp.float32)
    slope_contact_near_dist = jnp.maximum(
        jnp.asarray(constants.collision_base_dist, dtype=jnp.float32),
        jnp.asarray(constants.slope_contact_near_dist, dtype=jnp.float32),
    )
    slope_contact_release_speed = jnp.asarray(
        constants.slope_contact_release_speed, dtype=jnp.float32
    )
    support_near_contact_dist = jnp.where(
        use_gap_support,
        legacy_near_contact_dist,
        slope_contact_near_dist,
    )
    support_release_speed = jnp.where(
        use_gap_support,
        legacy_release_speed,
        slope_contact_release_speed,
    )
    support_mode_active = gap_support_grip_active | slope_contact_active
    support_contact = (
        (~colliding)
        & active_mask
        & has_surface_edge
        & support_mode_active
        & (min_dist_surface >= 0.0)
        & (min_dist_surface <= support_near_contact_dist)
        & (vel_normal_scalar <= support_release_speed)
    )
    persist_substeps = jnp.maximum(
        jnp.asarray(constants.friction_anchor_persist_substeps, dtype=jnp.int32),
        jnp.int32(0),
    )
    point_prev_manifold_cell = jnp.asarray(static_manifold_cell_idx, dtype=jnp.int32)[point_idx]
    point_prev_manifold_edge = jnp.asarray(static_manifold_edge_idx, dtype=jnp.int32)[point_idx]
    point_prev_manifold_active = jnp.asarray(static_manifold_active, dtype=jnp.bool_)[point_idx]
    point_prev_manifold_count = jnp.asarray(
        static_manifold_no_contact_count, dtype=jnp.int32
    )[point_idx]
    point_prev_manifold_load = jnp.asarray(
        static_manifold_normal_force, dtype=jnp.float32
    )[point_idx]
    point_prev_manifold_normal_vec = jnp.asarray(
        static_manifold_normal, dtype=jnp.float32
    )[point_idx]
    normal_dot = jnp.sum(point_prev_manifold_normal_vec * unit_normal, axis=-1)
    cos_threshold = jnp.float32(C.MANIFOLD_NORMAL_COS_THRESHOLD)
    same_prev_manifold = (
        point_prev_manifold_active
        & (normal_dot > cos_threshold)
    )
    manifold_persist_weight = jnp.where(
        persist_substeps > 0,
        jnp.clip(
            1.0
            - (
                point_prev_manifold_count.astype(jnp.float32)
                / jnp.maximum(persist_substeps.astype(jnp.float32), 1.0)
            ),
            0.0,
            1.0,
        ),
        0.0,
    )
    persistent_candidate = (
        (~colliding)
        & active_mask
        & has_surface_edge
        & stiction_active
        & upward_contact
        & is_sloped_edge
        & (persist_substeps > 0)
    )
    persistent_no_prev_manifold = persistent_candidate & (~point_prev_manifold_active)
    persistent_manifold_mismatch = (
        persistent_candidate & point_prev_manifold_active & (~same_prev_manifold)
    )
    persistent_same_manifold = persistent_candidate & same_prev_manifold
    persistent_in_window = persistent_same_manifold & (point_prev_manifold_count < persist_substeps)
    persistent_persist_window_block = persistent_same_manifold & (~persistent_in_window)
    persistent_gap_ok = (
        (min_dist_surface >= 0.0)
        & (
            min_dist_surface
            <= jnp.maximum(
                jnp.asarray(constants.collision_base_dist, dtype=jnp.float32),
                jnp.asarray(constants.slope_contact_near_dist, dtype=jnp.float32),
            )
        )
    )
    persistent_gap_block = persistent_in_window & (~persistent_gap_ok)
    persistent_release_ok = vel_normal_scalar <= jnp.asarray(
        constants.slope_contact_release_speed, dtype=jnp.float32
    )
    persistent_release_block = persistent_in_window & persistent_gap_ok & (~persistent_release_ok)
    persistent_contact = (
        persistent_in_window
        & persistent_gap_ok
        & persistent_release_ok
    )
    terrain_k = jnp.where(
        stiction_active,
        jnp.asarray(constants.terrain_stiction_stiffness, dtype=jnp.float32),
        jnp.float32(0.0),
    )
    terrain_switch_speed = jnp.where(
        stiction_active,
        jnp.asarray(constants.terrain_stiction_switch_speed, dtype=jnp.float32),
        jnp.float32(0.0),
    )
    gravity_force = jnp.stack(
        [
            jnp.float32(0.0),
            -jnp.asarray(constants.gravity * C.POINT_MASS, dtype=jnp.float32),
        ]
    )
    tangential_gravity_load = -jnp.sum(unit_tangent * gravity_force[None, :], axis=-1)
    mu_d_scalar = jnp.asarray(constants.dynamic_friction_const, dtype=jnp.float32)
    gap_support_mu_s = jnp.where(
        gap_support_grip_active,
        jnp.asarray(constants.slope_grip_static_friction_const, dtype=jnp.float32),
        jnp.float32(0.0),
    )
    gap_support_k = jnp.where(
        gap_support_grip_active,
        jnp.asarray(constants.slope_grip_support_stiffness, dtype=jnp.float32),
        jnp.float32(0.0),
    )
    gap_support_switch_speed = jnp.where(
        gap_support_grip_active,
        jnp.asarray(constants.slope_grip_switch_speed, dtype=jnp.float32),
        jnp.float32(0.0),
    )
    floor_k_base = jnp.asarray(constants.floor_stiction_stiffness, dtype=jnp.float32)
    floor_stiction_active = (
        colliding
        & upward_contact
        & ~is_sloped_edge
        & (floor_k_base > jnp.float32(0.0))
        & ~stiction_active
    )
    floor_mu_s = jnp.where(
        floor_stiction_active,
        jnp.asarray(constants.floor_static_friction_const, dtype=jnp.float32),
        gap_support_mu_s,
    )
    floor_k = jnp.where(floor_stiction_active, floor_k_base, gap_support_k)
    floor_switch = jnp.where(
        floor_stiction_active,
        jnp.asarray(constants.floor_stiction_switch_speed, dtype=jnp.float32),
        gap_support_switch_speed,
    )
    colliding_mu_s = jnp.where(
        stiction_active,
        jnp.asarray(constants.static_friction_const, dtype=jnp.float32),
        floor_mu_s,
    )
    colliding_k = jnp.where(stiction_active, terrain_k, floor_k)
    colliding_switch_speed = jnp.where(
        stiction_active, terrain_switch_speed, floor_switch
    )

    friction_force_stateless = _compute_friction_with_stiction(
        v_tang,
        normal_force_mag,
        unit_tangent,
        friction_const=constants.friction_const,
        mu_d=mu_d_scalar,
        mu_s=colliding_mu_s,
        k_stick=colliding_k,
        switch_speed=colliding_switch_speed,
        tangential_load=tangential_gravity_load,
    )
    model = jnp.asarray(constants.friction_model, dtype=jnp.int32)
    use_anchor_model = model == jnp.int32(C.FRICTION_MODEL_PROJECTED_ANCHOR)
    use_deformation_model = model == jnp.int32(C.FRICTION_MODEL_PROJECTED_DEFORMATION)
    use_memory_model = use_anchor_model | use_deformation_model

    m_contact = jnp.asarray(tangential_deformation, dtype=jnp.float32)[point_idx]
    m_tang = jnp.sum(m_contact * unit_tangent, axis=-1)
    mu_s = jnp.asarray(constants.static_friction_const, dtype=jnp.float32)
    stick_cap = mu_s * normal_force_mag
    mu_d_f32 = mu_d_scalar
    friction_const_f32 = jnp.asarray(constants.friction_const, dtype=jnp.float32)
    f_dyn = dynamic_friction_scalar(
        v_tang,
        normal_force_mag,
        friction_const=friction_const_f32,
        mu_d=mu_d_f32,
        normalize_by_load=True,
    )

    k_anchor = jnp.asarray(constants.anchor_stiffness, dtype=jnp.float32)
    f_anchor_spring = -k_anchor * m_tang
    f_anchor = stick_cap * jnp.tanh(f_anchor_spring / jnp.maximum(stick_cap, 1e-10))

    k_def = jnp.asarray(constants.deformation_stiffness, dtype=jnp.float32)
    d_def = jnp.asarray(constants.deformation_damping, dtype=jnp.float32)
    f_def_stick = -k_def * m_tang - d_def * v_tang
    breakaway = jnp.asarray(constants.breakaway_threshold, dtype=jnp.float32)
    f_def_stick = jnp.where(jnp.abs(m_tang) < breakaway, 0.0, f_def_stick)
    sticking = jnp.abs(f_def_stick) <= stick_cap
    f_deformation = jnp.where(sticking, f_def_stick, f_dyn)

    f_memory_model = jnp.where(use_anchor_model, f_anchor, f_deformation)
    f_memory_total = jnp.where(stiction_active, f_memory_model, f_dyn)
    friction_force_memory = unit_tangent * f_memory_total[:, None]
    friction_force_force_model = jnp.where(
        use_memory_model, friction_force_memory, friction_force_stateless
    )
    constraint_mode = jnp.asarray(constants.friction_constraint_mode, dtype=jnp.int32)
    use_pbd_anchor = constraint_mode == jnp.int32(C.FRICTION_CONSTRAINT_MODE_TERRAIN_PBD_ANCHOR)
    friction_force_dyn_only = unit_tangent * f_dyn[:, None]
    friction_force = jnp.where(use_pbd_anchor, friction_force_dyn_only, friction_force_force_model)
    friction_force = jnp.where(is_sloped_edge[:, None], 0.0, friction_force)
    colliding_force = normal_force + damping + friction_force

    support_gap = jnp.clip(
        support_near_contact_dist - min_dist_surface, 0.0, support_near_contact_dist
    )
    support_weight_scalar = support_gap / jnp.maximum(support_near_contact_dist, 1e-10)
    support_weight_scalar = support_weight_scalar * support_weight_scalar * (
        3.0 - 2.0 * support_weight_scalar
    )
    support_weight = support_weight_scalar[:, None]

    gap_support_normal_force_mag = gap_support_k * support_gap
    gap_support_friction_force = _compute_friction_with_stiction(
        v_tang,
        gap_support_normal_force_mag,
        unit_tangent,
        friction_const=constants.friction_const,
        mu_d=mu_d_scalar,
        mu_s=gap_support_mu_s,
        k_stick=gap_support_k,
        switch_speed=gap_support_switch_speed,
        tangential_load=tangential_gravity_load,
    )
    approach_vel_normal = jnp.minimum(vel_normal, 0.0)
    gap_support_damping = -unit_normal * approach_vel_normal * (
        constants.collision_vel_damping * 0.35
    ) * support_weight
    gap_support_force = gap_support_damping + gap_support_friction_force

    continuity_normal_damping = jnp.asarray(
        constants.slope_contact_normal_damping, dtype=jnp.float32
    )
    continuity_tangent_damping = jnp.asarray(
        constants.slope_contact_tangent_damping, dtype=jnp.float32
    )
    continuity_mu = jnp.asarray(constants.slope_contact_assist_mu, dtype=jnp.float32)
    continuity_approach_speed = jnp.maximum(-vel_normal_scalar, 0.0)
    continuity_support_normal_force_mag = (
        continuity_normal_damping * continuity_approach_speed * support_weight_scalar
    )
    continuity_support_cap = continuity_mu * continuity_support_normal_force_mag
    continuity_support_tangent_force_mag = jnp.clip(
        -continuity_tangent_damping * v_tang,
        -continuity_support_cap,
        continuity_support_cap,
    )
    continuity_support_force = (
        unit_normal * continuity_support_normal_force_mag[:, None]
        + unit_tangent * continuity_support_tangent_force_mag[:, None]
    )

    gravity_normal_load = jnp.maximum(
        jnp.sum(unit_normal * (-gravity_force)[None, :], axis=-1),
        0.0,
    ) * support_weight_scalar
    gravity_support_cap = continuity_mu * gravity_normal_load
    gravity_support_tangent_force_mag = jnp.clip(
        tangential_gravity_load - continuity_tangent_damping * v_tang,
        -gravity_support_cap,
        gravity_support_cap,
    )
    gravity_support_force = unit_tangent * gravity_support_tangent_force_mag[:, None]

    support_normal_force_mag = jnp.where(
        use_gap_support,
        gap_support_normal_force_mag,
        jnp.where(
            use_continuity_assist,
            continuity_support_normal_force_mag,
            jnp.zeros_like(gap_support_normal_force_mag),
        ),
    )
    support_force = jnp.where(
        use_gap_support,
        gap_support_force,
        jnp.where(
            use_continuity_assist,
            continuity_support_force,
            gravity_support_force,
        ),
    )
    support_f_tang_total = jnp.sum(support_force * unit_tangent, axis=-1)
    persistent_normal_force_mag = point_prev_manifold_load * manifold_persist_weight
    persistent_f_dyn = dynamic_friction_scalar(
        v_tang,
        persistent_normal_force_mag,
        friction_const=friction_const_f32,
        mu_d=mu_d_f32,
        normalize_by_load=True,
    )
    persistent_stick_cap = mu_s * persistent_normal_force_mag
    persistent_f_anchor_spring = -k_anchor * m_tang
    persistent_f_anchor = persistent_stick_cap * jnp.tanh(
        persistent_f_anchor_spring / jnp.maximum(persistent_stick_cap, 1e-10)
    )
    persistent_f_def_stick = -k_def * m_tang - d_def * v_tang
    persistent_f_def_stick = jnp.where(
        jnp.abs(m_tang) < breakaway,
        0.0,
        persistent_f_def_stick,
    )
    persistent_sticking = jnp.abs(persistent_f_def_stick) <= persistent_stick_cap
    persistent_f_deformation = jnp.where(
        persistent_sticking,
        persistent_f_def_stick,
        persistent_f_dyn,
    )
    persistent_f_memory_model = jnp.where(
        use_anchor_model,
        persistent_f_anchor,
        persistent_f_deformation,
    )
    persistent_force_stateless = _compute_friction_with_stiction(
        v_tang,
        persistent_normal_force_mag,
        unit_tangent,
        friction_const=constants.friction_const,
        mu_d=mu_d_scalar,
        mu_s=colliding_mu_s,
        k_stick=colliding_k,
        switch_speed=colliding_switch_speed,
        tangential_load=tangential_gravity_load,
    )
    persistent_f_total = jnp.where(
        use_memory_model,
        persistent_f_memory_model,
        jnp.sum(persistent_force_stateless * unit_tangent, axis=-1),
    )
    persistent_force = jnp.where(
        use_pbd_anchor,
        jnp.zeros_like(unit_tangent),
        unit_tangent * persistent_f_total[:, None],
    )
    persistent_f_tang_total = jnp.sum(persistent_force * unit_tangent, axis=-1)

    # Avoid bool->numeric casts in runtime force composition; gate via predicates.
    point_force = (
        jnp.where(colliding[:, None], colliding_force, 0.0)
        + jnp.where(persistent_contact[:, None], persistent_force, 0.0)
        + jnp.where(support_contact[:, None], support_force, 0.0)
    )

    # Deterministic per-point accumulation (avoids repeated-index scatter-add on GPU).
    point_weights = jax.nn.one_hot(point_idx, n_points, dtype=point_force.dtype)
    collision_forces = point_weights.T @ point_force

    contact_mask = jnp.zeros((n_points,), dtype=jnp.bool_)
    contact_mask = contact_mask.at[point_idx].max(colliding)
    contact_normal_force = jnp.zeros((n_points,), dtype=jnp.float32)
    contact_normal_force = contact_normal_force.at[point_idx].max(
        jnp.where(colliding, normal_force_mag, 0.0)
    )
    dominant_contact = (
        colliding
        & (normal_force_mag >= (contact_normal_force[point_idx] - 1e-6))
        & (normal_force_mag > 0.0)
    )
    tangent_sum = point_weights.T @ jnp.where(dominant_contact[:, None], unit_tangent, 0.0)
    tangent_norm = jnp.linalg.norm(tangent_sum, axis=-1, keepdims=True)
    contact_tangent = tangent_sum / jnp.maximum(tangent_norm, 1e-10)
    normal_sum = point_weights.T @ jnp.where(dominant_contact[:, None], unit_normal, 0.0)
    normal_norm = jnp.linalg.norm(normal_sum, axis=-1, keepdims=True)
    contact_normal = normal_sum / jnp.maximum(normal_norm, 1e-10)

    colliding_slope_manifold = colliding & stiction_active & upward_contact & is_sloped_edge
    stiction_contact = colliding & stiction_active
    memory_eligible_contact = stiction_contact | persistent_contact
    memory_contact_mask = jnp.zeros((n_points,), dtype=jnp.bool_)
    memory_contact_mask = memory_contact_mask.at[point_idx].max(memory_eligible_contact)
    persistent_contact_mask = jnp.zeros((n_points,), dtype=jnp.bool_)
    persistent_contact_mask = persistent_contact_mask.at[point_idx].max(persistent_contact)
    memory_contact_normal_force = jnp.zeros((n_points,), dtype=jnp.float32)
    memory_contact_normal_force = memory_contact_normal_force.at[point_idx].max(
        jnp.where(
            memory_eligible_contact,
            jnp.where(colliding, normal_force_mag, persistent_normal_force_mag),
            0.0,
        )
    )
    dominant_memory_contact = (
        memory_eligible_contact
        & (
            jnp.where(colliding, normal_force_mag, persistent_normal_force_mag)
            >= (memory_contact_normal_force[point_idx] - 1e-6)
        )
        & (jnp.where(colliding, normal_force_mag, persistent_normal_force_mag) > 0.0)
    )
    memory_tangent_sum = point_weights.T @ jnp.where(
        dominant_memory_contact[:, None], unit_tangent, 0.0
    )
    memory_tangent_norm = jnp.linalg.norm(memory_tangent_sum, axis=-1, keepdims=True)
    memory_contact_tangent = memory_tangent_sum / jnp.maximum(memory_tangent_norm, 1e-10)
    memory_normal_sum = point_weights.T @ jnp.where(
        dominant_memory_contact[:, None], unit_normal, 0.0
    )
    memory_normal_norm = jnp.linalg.norm(memory_normal_sum, axis=-1, keepdims=True)
    memory_contact_normal = memory_normal_sum / jnp.maximum(memory_normal_norm, 1e-10)

    slope_manifold_force = jnp.zeros((n_points,), dtype=jnp.float32)
    slope_manifold_force = slope_manifold_force.at[point_idx].max(
        jnp.where(colliding_slope_manifold, normal_force_mag, 0.0)
    )
    dominant_slope_manifold = (
        colliding_slope_manifold
        & (normal_force_mag >= (slope_manifold_force[point_idx] - 1e-6))
        & (normal_force_mag > 0.0)
    )
    refreshed_manifold_cell_idx = jnp.full((n_points,), -1, dtype=jnp.int32)
    refreshed_manifold_cell_idx = refreshed_manifold_cell_idx.at[point_idx].max(
        jnp.where(dominant_slope_manifold, tj, -1)
    )
    refreshed_manifold_edge_idx = jnp.full((n_points,), -1, dtype=jnp.int32)
    refreshed_manifold_edge_idx = refreshed_manifold_edge_idx.at[point_idx].max(
        jnp.where(dominant_slope_manifold, min_edge, -1)
    )
    refreshed_manifold_load = jnp.zeros((n_points,), dtype=jnp.float32)
    refreshed_manifold_load = refreshed_manifold_load.at[point_idx].max(
        jnp.where(dominant_slope_manifold, normal_force_mag, 0.0)
    )
    refreshed_manifold_normal_sum = point_weights.T @ jnp.where(
        dominant_slope_manifold[:, None], unit_normal, 0.0
    )
    refreshed_manifold_normal_norm = jnp.linalg.norm(
        refreshed_manifold_normal_sum, axis=-1, keepdims=True
    )
    refreshed_manifold_normal = refreshed_manifold_normal_sum / jnp.maximum(
        refreshed_manifold_normal_norm, 1e-10
    )
    refreshed_manifold_active = refreshed_manifold_load > 0.0
    persisting_by_timeout = (
        jnp.asarray(static_manifold_active, dtype=jnp.bool_)
        & (jnp.asarray(static_manifold_no_contact_count, dtype=jnp.int32) < persist_substeps)
        & ~refreshed_manifold_active
    )
    kept_manifold_active = persisting_by_timeout | (memory_contact_mask & ~refreshed_manifold_active)
    next_static_manifold_cell_idx = jnp.where(
        refreshed_manifold_active,
        refreshed_manifold_cell_idx,
        jnp.where(kept_manifold_active, static_manifold_cell_idx, jnp.full((n_points,), -1, dtype=jnp.int32)),
    )
    next_static_manifold_edge_idx = jnp.where(
        refreshed_manifold_active,
        refreshed_manifold_edge_idx,
        jnp.where(kept_manifold_active, static_manifold_edge_idx, jnp.full((n_points,), -1, dtype=jnp.int32)),
    )
    next_static_manifold_active = refreshed_manifold_active | kept_manifold_active
    next_static_manifold_no_contact_count = jnp.where(
        refreshed_manifold_active,
        jnp.int32(0),
        jnp.where(
            kept_manifold_active,
            jnp.asarray(static_manifold_no_contact_count, dtype=jnp.int32) + jnp.int32(1),
            jnp.int32(0),
        ),
    )
    next_static_manifold_normal_force = jnp.where(
        refreshed_manifold_active,
        refreshed_manifold_load,
        jnp.where(
            kept_manifold_active,
            jnp.asarray(static_manifold_normal_force, dtype=jnp.float32)
            * jnp.where(
                persist_substeps > 0,
                jnp.clip(
                    1.0
                    - (
                        next_static_manifold_no_contact_count.astype(jnp.float32)
                        / jnp.maximum(persist_substeps.astype(jnp.float32), 1.0)
                    ),
                    0.0,
                    1.0,
                ),
                0.0,
            ),
            0.0,
        ),
    )
    next_static_manifold_normal = jnp.where(
        refreshed_manifold_active[:, None],
        refreshed_manifold_normal,
        jnp.where(
            kept_manifold_active[:, None],
            static_manifold_normal,
            jnp.zeros((n_points, 2), dtype=jnp.float32),
        ),
    )

    support_mask = jnp.zeros((n_points,), dtype=jnp.bool_)
    support_mask = support_mask.at[point_idx].max(support_contact)
    support_normal_force = jnp.zeros((n_points,), dtype=jnp.float32)
    support_normal_force = support_normal_force.at[point_idx].max(
        jnp.where(support_contact, support_normal_force_mag, 0.0)
    )
    dominant_support = (
        support_contact
        & (support_normal_force_mag >= (support_normal_force[point_idx] - 1e-6))
        & (support_normal_force_mag > 0.0)
    )
    support_tangent_sum = point_weights.T @ jnp.where(
        dominant_support[:, None], unit_tangent, 0.0
    )
    support_tangent_norm = jnp.linalg.norm(support_tangent_sum, axis=-1, keepdims=True)
    support_tangent = support_tangent_sum / jnp.maximum(support_tangent_norm, 1e-10)
    support_normal_sum = point_weights.T @ jnp.where(
        dominant_support[:, None], unit_normal, 0.0
    )
    support_normal_norm = jnp.linalg.norm(support_normal_sum, axis=-1, keepdims=True)
    support_normal = support_normal_sum / jnp.maximum(support_normal_norm, 1e-10)

    def _build_diagnostics(_):
        colliding_f1 = colliding.astype(jnp.float32)
        stiction_contact_f1 = stiction_contact.astype(jnp.float32)
        colliding_stiction_contact = (
            colliding
            & (colliding_mu_s > 0.0)
            & (colliding_k > 0.0)
            & (colliding_switch_speed > 0.0)
        )
        colliding_stiction_contact_f1 = colliding_stiction_contact.astype(jnp.float32)
        support_contact_f1 = support_contact.astype(jnp.float32)
        f_tang_total = jnp.sum(friction_force * unit_tangent, axis=-1)
        sum_normal_force = jnp.sum(normal_force_mag * colliding_f1)
        sum_abs_v_tang = jnp.sum(jnp.abs(v_tang) * colliding_f1)
        sum_abs_friction = jnp.sum(jnp.abs(f_tang_total) * colliding_f1)
        sum_persistent_abs_friction = jnp.sum(
            jnp.abs(persistent_f_tang_total) * persistent_contact.astype(jnp.float32)
        )
        sum_support_normal_force = jnp.sum(support_normal_force_mag * support_contact_f1)
        sum_support_abs_friction = jnp.sum(jnp.abs(support_f_tang_total) * support_contact_f1)

        effective_stick_cap = colliding_mu_s * normal_force_mag
        stick_cap_pos = jnp.maximum(effective_stick_cap, 0.0)
        friction_cap_mask = colliding_stiction_contact & (stick_cap_pos > 0.0)
        friction_cap_count = jnp.sum(friction_cap_mask.astype(jnp.int32))
        sum_stick_cap = jnp.sum(stick_cap_pos * colliding_stiction_contact_f1)

        k_stateless = jnp.maximum(colliding_k, 1e-10)
        k_memory = jnp.where(
            use_anchor_model,
            jnp.maximum(k_anchor, 1e-10),
            jnp.where(
                use_deformation_model,
                jnp.maximum(k_def, 1e-10),
                k_stateless,
            ),
        )
        k_model = jnp.where(
            stiction_contact,
            k_memory,
            k_stateless,
        )
        cap_disp = jnp.where(colliding_stiction_contact, stick_cap_pos / k_model, 0.0)
        sum_cap_disp = jnp.sum(cap_disp)
        sum_step_slip_disp = jnp.sum(jnp.abs(v_tang) * jnp.float32(constants.dt) * colliding_f1)
        memory_clamped = (
            stiction_contact
            & use_memory_model
            & (jnp.abs(m_tang) >= jnp.maximum(cap_disp - 1e-8, 0.0))
        )

        return StaticCollisionDiagnostics(
            allocated_surface_slot_count=static_collision_worklist.allocated_surface_slot_count,
            active_surface_boxel_count=static_collision_worklist.active_surface_boxel_count,
            allocated_static_slot_count=static_collision_worklist.allocated_static_slot_count,
            active_static_cell_count=static_collision_worklist.active_static_cell_count,
            allocated_worklist_count=static_collision_worklist.allocated_worklist_count,
            active_worklist_count_hint=static_collision_worklist.active_worklist_count_hint,
            active_triple_count=static_collision_worklist.active_worklist_count_hint,
            inside_count=jnp.sum(inside.astype(jnp.int32)),
            viable_count=jnp.sum((inside & has_viable).astype(jnp.int32)),
            colliding_count=jnp.sum(colliding.astype(jnp.int32)),
            stiction_active_count=jnp.sum((colliding & stiction_active).astype(jnp.int32)),
            floor_stiction_active_count=jnp.sum(floor_stiction_active.astype(jnp.int32)),
            slope_contact_count=jnp.sum((colliding & upward_contact & is_sloped_edge).astype(jnp.int32)),
            floor_contact_count=jnp.sum((colliding & upward_contact & ~is_sloped_edge).astype(jnp.int32)),
            wall_contact_count=jnp.sum((colliding & ~upward_contact).astype(jnp.int32)),
            contact_point_count=jnp.sum(contact_mask.astype(jnp.int32)),
            tangent_degenerate_point_count=jnp.sum(
                (contact_mask & (jnp.squeeze(tangent_norm, axis=-1) <= 1e-6)).astype(jnp.int32)
            ),
            sum_normal_force=sum_normal_force,
            sum_abs_v_tang=sum_abs_v_tang,
            sum_abs_friction=sum_abs_friction,
            sum_stick_cap=sum_stick_cap,
            friction_cap_count=friction_cap_count,
            sum_cap_disp=sum_cap_disp,
            sum_step_slip_disp=sum_step_slip_disp,
            memory_clamped_count=jnp.sum(memory_clamped.astype(jnp.int32)),
            persistent_contact_count=jnp.sum(persistent_contact.astype(jnp.int32)),
            persistent_point_count=jnp.sum(persistent_contact_mask.astype(jnp.int32)),
            persistent_without_collision_count=jnp.sum(
                (persistent_contact_mask & ~contact_mask).astype(jnp.int32)
            ),
            persistent_candidate_count=jnp.sum(persistent_candidate.astype(jnp.int32)),
            persistent_block_no_prev_manifold_count=jnp.sum(
                persistent_no_prev_manifold.astype(jnp.int32)
            ),
            persistent_block_manifold_mismatch_count=jnp.sum(
                persistent_manifold_mismatch.astype(jnp.int32)
            ),
            persistent_block_persist_window_count=jnp.sum(
                persistent_persist_window_block.astype(jnp.int32)
            ),
            persistent_block_gap_count=jnp.sum(persistent_gap_block.astype(jnp.int32)),
            persistent_block_release_speed_count=jnp.sum(
                persistent_release_block.astype(jnp.int32)
            ),
            sum_persistent_abs_friction=sum_persistent_abs_friction,
            support_point_count=jnp.sum(support_mask.astype(jnp.int32)),
            support_without_collision_count=jnp.sum(
                (support_mask & ~contact_mask).astype(jnp.int32)
            ),
            support_slope_count=jnp.sum(
                (support_contact & upward_contact & is_sloped_edge).astype(jnp.int32)
            ),
            support_floor_count=jnp.sum(
                (support_contact & upward_contact & ~is_sloped_edge).astype(jnp.int32)
            ),
            support_wall_count=jnp.sum((support_contact & ~upward_contact).astype(jnp.int32)),
            sum_support_normal_force=sum_support_normal_force,
            sum_support_abs_friction=sum_support_abs_friction,
        )

    diagnostics = jax.lax.cond(
        jnp.bool_(constants.diagnostics_enabled),
        _build_diagnostics,
        lambda _: _zero_static_collision_diagnostics(),
        operand=None,
    )
    return StaticCollisionResult(
        forces=collision_forces,
        contact_mask=contact_mask,
        contact_normal=contact_normal,
        contact_tangent=contact_tangent,
        contact_normal_force=contact_normal_force,
        memory_contact_mask=memory_contact_mask,
        memory_contact_normal=memory_contact_normal,
        memory_contact_tangent=memory_contact_tangent,
        memory_contact_normal_force=memory_contact_normal_force,
        support_mask=support_mask,
        support_normal=support_normal,
        support_tangent=support_tangent,
        support_normal_force=support_normal_force,
        static_manifold_cell_idx=next_static_manifold_cell_idx,
        static_manifold_edge_idx=next_static_manifold_edge_idx,
        static_manifold_active=next_static_manifold_active,
        static_manifold_no_contact_count=next_static_manifold_no_contact_count,
        static_manifold_normal_force=next_static_manifold_normal_force,
        static_manifold_normal=next_static_manifold_normal,
        diagnostics=diagnostics,
    )


def resolve_static_collisions(
    positions: jnp.ndarray,
    velocities_true: jnp.ndarray,
    collision_data: CollisionData,
    static_collider_data: StaticColliderData | None,
    constants: PhysicsConstants,
    static_collision_worklist: StaticCollisionWorklist | None = None,
) -> jnp.ndarray:
    return resolve_static_collisions_with_contacts(
        positions,
        velocities_true,
        collision_data,
        static_collider_data,
        constants,
        static_collision_worklist=static_collision_worklist,
        tangential_deformation=None,
    ).forces


def _select_dynamic_pair_mask(
    positions: jnp.ndarray,
    collision_data: CollisionData,
    *,
    max_candidates: jnp.ndarray | int,
    bin_size: jnp.ndarray | float,
    impl: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Select candidate boxel pairs for dynamic broadphase."""
    cd = collision_data
    pair_i = cd.candidate_pair_i
    pair_j = cd.candidate_pair_j
    pair_active = cd.candidate_pair_active & cd.boxel_mask[pair_i] & cd.boxel_mask[pair_j]

    safe_bin = jnp.maximum(jnp.asarray(bin_size, dtype=jnp.float32), jnp.float32(1e-4))
    max_candidates_i32 = jnp.maximum(jnp.int32(1), jnp.asarray(max_candidates, dtype=jnp.int32))

    expanded_positions = jnp.concatenate([positions, positions], axis=0)
    corner_pos = expanded_positions[cd.boxel_corners]
    boxel_min = jnp.min(corner_pos, axis=1)
    boxel_max = jnp.max(corner_pos, axis=1)
    centers = 0.5 * (boxel_min + boxel_max)
    bin_xy = jnp.floor(centers / safe_bin).astype(jnp.int32)

    bin_i = bin_xy[pair_i]
    bin_j = bin_xy[pair_j]
    near_bins = (
        (jnp.abs(bin_i[:, 0] - bin_j[:, 0]) <= 1)
        & (jnp.abs(bin_i[:, 1] - bin_j[:, 1]) <= 1)
    )
    min_i = boxel_min[pair_i]
    max_i = boxel_max[pair_i]
    min_j = boxel_min[pair_j]
    max_j = boxel_max[pair_j]
    overlap_x = (min_i[:, 0] <= max_j[:, 0]) & (max_i[:, 0] >= min_j[:, 0])
    overlap_y = (min_i[:, 1] <= max_j[:, 1]) & (max_i[:, 1] >= min_j[:, 1])
    pair_mask = pair_active & near_bins & overlap_x & overlap_y

    n_pairs = jnp.sum(pair_mask.astype(jnp.int32))
    overflow = n_pairs > max_candidates_i32
    if impl in ("direct", "dense_mask"):
        selected_mask = pair_mask & (
            jnp.cumsum(pair_mask.astype(jnp.int32)) <= max_candidates_i32
        )
    elif impl == "sparse_candidate_list":
        active_idx = jnp.nonzero(pair_mask, size=pair_mask.shape[0], fill_value=0)[0]
        keep = jnp.arange(pair_mask.shape[0], dtype=jnp.int32) < jnp.minimum(
            n_pairs, max_candidates_i32
        )
        selected_mask = jnp.zeros_like(pair_mask).at[active_idx].max(keep)
    else:  # pragma: no cover - defensive boundary
        raise ValueError(
            f"Unsupported dynamic broadphase impl={impl!r}. "
            "Expected 'dense_mask' or 'sparse_candidate_list'."
        )
    return selected_mask, overflow


def resolve_collisions(
    positions: jnp.ndarray,
    velocities_true: jnp.ndarray,
    collision_data: CollisionData,
    constants: PhysicsConstants,
    collision_strategy: jnp.ndarray | int = 0,
    dynamic_max_candidates: jnp.ndarray | int = 4096,
    dynamic_bin_size: jnp.ndarray | float = 0.2,
    dynamic_impl: str = "direct",
    dynamic_max_narrow: int | None = None,
) -> jnp.ndarray:
    strategy = jnp.asarray(collision_strategy, dtype=jnp.int32)

    def _dynamic(_):
        return resolve_collisions_dynamic_broadphase(
            positions,
            velocities_true,
            collision_data,
            constants,
            max_candidates=dynamic_max_candidates,
            bin_size=dynamic_bin_size,
            impl=dynamic_impl,
            max_narrow=dynamic_max_narrow,
        )

    def _static(_):
        return resolve_collisions_static_indexed(
            positions,
            velocities_true,
            collision_data,
            constants,
            max_narrow=dynamic_max_narrow,
        )

    return jax.lax.cond(strategy == 1, _dynamic, _static, operand=None)


def resolve_collisions_dynamic_broadphase(
    positions: jnp.ndarray,
    velocities_true: jnp.ndarray,
    collision_data: CollisionData,
    constants: PhysicsConstants,
    max_candidates: jnp.ndarray | int = 4096,
    bin_size: jnp.ndarray | float = 0.2,
    impl: str = "direct",
    max_narrow: int | None = None,
) -> jnp.ndarray:
    forces, _ = resolve_collisions_dynamic_broadphase_with_probe(
        positions,
        velocities_true,
        collision_data,
        constants,
        max_candidates=max_candidates,
        bin_size=bin_size,
        impl=impl,
        max_narrow=max_narrow,
    )
    return forces


def resolve_collisions_dynamic_broadphase_with_probe(
    positions: jnp.ndarray,
    velocities_true: jnp.ndarray,
    collision_data: CollisionData,
    constants: PhysicsConstants,
    max_candidates: jnp.ndarray | int = 4096,
    bin_size: jnp.ndarray | float = 0.2,
    impl: str = "direct",
    max_narrow: int | None = None,
) -> tuple[jnp.ndarray, DynamicBroadphaseCollisionProbe]:
    selected_pair_mask, overflow = _select_dynamic_pair_mask(
        positions,
        collision_data,
        max_candidates=max_candidates,
        bin_size=bin_size,
        impl=impl,
    )
    selected_triple_active = (
        collision_data.triple_active
        & selected_pair_mask[collision_data.triple_pair_slot]
    )
    selected_triple_count = jnp.sum(
        jnp.where(selected_triple_active, jnp.int32(1), jnp.int32(0))
    )

    triple_active = jax.lax.cond(
        overflow,
        lambda _: collision_data.triple_active,
        lambda _: selected_triple_active,
        operand=None,
    )
    n_active_hint = jax.lax.cond(
        overflow,
        lambda _: jnp.asarray(collision_data.n_triples, dtype=jnp.int32),
        lambda _: selected_triple_count,
        operand=None,
    )
    forces = _resolve_dynamic_triples(
        positions,
        velocities_true,
        collision_data,
        constants,
        triple_active,
        max_narrow=max_narrow,
        n_active_hint=n_active_hint,
    )
    return forces, DynamicBroadphaseCollisionProbe(
        selected_pair_mask=selected_pair_mask,
        overflow=overflow,
        selected_triple_active=selected_triple_active,
        triple_active=triple_active,
    )


def is_self_colliding(
    positions: jnp.ndarray,
    collision_data: CollisionData,
) -> jnp.ndarray:
    cd = collision_data
    if cd.self_triple_active.shape[0] == 0:
        return jnp.bool_(False)

    n_boxels = jnp.asarray(cd.boxel_mask.shape[0], dtype=jnp.int32)
    pair_i = jnp.minimum(cd.self_triple_i, cd.self_triple_j)
    pair_j = jnp.maximum(cd.self_triple_i, cd.self_triple_j)
    pair_id = pair_i * n_boxels + pair_j
    sentinel = n_boxels * n_boxels
    pair_ids = jnp.where(cd.self_triple_active, pair_id, sentinel)
    pair_ids_sorted = jnp.sort(pair_ids)
    valid = pair_ids_sorted < sentinel
    prev_ids = jnp.concatenate(
        [jnp.asarray([-1], dtype=jnp.int32), pair_ids_sorted[:-1]],
        axis=0,
    )
    unique_pairs = valid & (pair_ids_sorted != prev_ids)

    # Match EvoGym C++ self-collision semantics: count non-adjacent boxel bbox
    # overlaps, not corner-in-quad hits. Do not reintroduce point_in_quad here.
    safe_pair_ids = jnp.minimum(pair_ids_sorted, sentinel - 1)
    unique_pair_i = safe_pair_ids // n_boxels
    unique_pair_j = safe_pair_ids % n_boxels

    expanded_positions = jnp.concatenate([positions, positions], axis=0)
    corner_pos = expanded_positions[cd.boxel_corners]
    boxel_min = jnp.min(corner_pos, axis=1)
    boxel_max = jnp.max(corner_pos, axis=1)

    min_i = boxel_min[unique_pair_i]
    max_i = boxel_max[unique_pair_i]
    min_j = boxel_min[unique_pair_j]
    max_j = boxel_max[unique_pair_j]
    overlap_x = (min_i[:, 0] <= max_j[:, 0]) & (max_i[:, 0] >= min_j[:, 0])
    overlap_y = (min_i[:, 1] <= max_j[:, 1]) & (max_i[:, 1] >= min_j[:, 1])
    pair_collision_count = jnp.sum((unique_pairs & overlap_x & overlap_y).astype(jnp.int32))

    # C++ parity: self-collision triggers only when non-adjacent colliding
    # pair count exceeds the robot's number of surface boxels.
    robot_surface_boxel_count = jnp.asarray(
        cd.n_robot_surface_boxels,
        dtype=jnp.int32,
    )
    return pair_collision_count > robot_surface_boxel_count
