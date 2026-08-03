"""JAX-native morphology construction for vmap-able env building.

All functions operate on a pre-computed GridData (static for a given H, W).
The morphology array determines VALUES (which springs are active, constants)
but not SHAPES — enabling jax.vmap across a batch of morphologies.

precompute_grid: numpy, called once per grid size.
All other functions: pure JAX, JIT-able, vmap-able over morphology arrays.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.errors import TracerArrayConversionError

from . import constants as C
from .spring_scaling import contractile_diag_ratio_jax
from .types import (
    ActuatorInfo,
    CollisionData,
    DeformationInfo,
    GridData,
    PerAxisActuatorInfo,
    SimState,
    SpringTopology,
)

_SLOPE_TYPE_ARRAY = np.array(sorted(C.SLOPE_TYPES), dtype=np.int32)


# ---------------------------------------------------------------------------
# Grid pre-computation (numpy, called once)
# ---------------------------------------------------------------------------


def _is_actuator_type(vtype: jnp.ndarray) -> jnp.ndarray:
    return (vtype == C.H_ACT) | (vtype == C.V_ACT) | (vtype == C.CONTRACTILE)


def _ensure_no_slope_cells(morphology: object, *, context: str) -> None:
    try:
        morph_np = np.asarray(morphology)
    except (TracerArrayConversionError, TypeError):
        return
    if np.any(np.isin(morph_np, _SLOPE_TYPE_ARRAY)):
        raise ValueError(
            f"{context} only supports square cells; slope terrain must use the "
            "object-separated build pipeline"
        )


def precompute_grid(H: int, W: int) -> GridData:
    """Build all static grid infrastructure for a (H, W) grid.

    Pure numpy, called once. Returns GridData with JAX arrays.
    """
    W1 = W + 1
    n_points = (H + 1) * W1
    n_horiz = (H + 1) * W
    n_vert = H * W1
    n_diag = 2 * H * W
    n_springs = n_horiz + n_vert + n_diag
    vert_offset = n_horiz
    diag_offset = n_horiz + n_vert

    # --- Point positions ---
    gx, gy = np.meshgrid(np.arange(W + 1), np.arange(H + 1))
    positions = np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32) * C.CELL_SIZE

    # --- Spring arrays ---
    spring_a_idx = np.zeros(n_springs, dtype=np.int32)
    spring_b_idx = np.zeros(n_springs, dtype=np.int32)
    spring_types = np.zeros(n_springs, dtype=np.int32)
    spring_cell_a = np.full(n_springs, -1, dtype=np.int32)
    spring_cell_b = np.full(n_springs, -1, dtype=np.int32)

    # --- Horizontal springs: (H+1)*W ---
    # Index within horiz section: grid_y * W + grid_x
    h_gy = np.repeat(np.arange(H + 1), W)
    h_gx = np.tile(np.arange(W), H + 1)
    h_p1 = h_gy * W1 + h_gx
    h_p2 = h_gy * W1 + (h_gx + 1)

    spring_a_idx[:n_horiz] = h_p1
    spring_b_idx[:n_horiz] = h_p2
    spring_types[:n_horiz] = 0

    # Adjacent cells for horizontal springs
    h_vy_above = h_gy      # cell whose bottom edge this is
    h_vy_below = h_gy - 1  # cell whose top edge this is
    h_cell_above = np.where((h_vy_above >= 0) & (h_vy_above < H), h_vy_above * W + h_gx, -1)
    h_cell_below = np.where((h_vy_below >= 0) & (h_vy_below < H), h_vy_below * W + h_gx, -1)
    # cell_a = later cell (higher vy), or the only cell for boundary
    spring_cell_a[:n_horiz] = np.where(h_cell_above >= 0, h_cell_above, h_cell_below)
    spring_cell_b[:n_horiz] = np.where(
        (h_cell_above >= 0) & (h_cell_below >= 0), h_cell_below, -1
    )

    # --- Vertical springs: H*(W+1) ---
    # Index within vert section: grid_y * (W+1) + grid_x
    v_gy = np.repeat(np.arange(H), W1)
    v_gx = np.tile(np.arange(W1), H)
    v_p1 = v_gy * W1 + v_gx
    v_p2 = (v_gy + 1) * W1 + v_gx

    spring_a_idx[vert_offset:vert_offset + n_vert] = v_p1
    spring_b_idx[vert_offset:vert_offset + n_vert] = v_p2
    spring_types[vert_offset:vert_offset + n_vert] = 1

    # Adjacent cells for vertical springs
    v_vx_left = v_gx       # cell whose left edge this is
    v_vx_right = v_gx - 1  # cell whose right edge this is
    v_cell_left = np.where((v_vx_left >= 0) & (v_vx_left < W), v_gy * W + v_vx_left, -1)
    v_cell_right = np.where((v_vx_right >= 0) & (v_vx_right < W), v_gy * W + v_vx_right, -1)
    # cell_a = later cell (higher vx), or the only cell for boundary
    spring_cell_a[vert_offset:vert_offset + n_vert] = np.where(
        v_cell_left >= 0, v_cell_left, v_cell_right
    )
    spring_cell_b[vert_offset:vert_offset + n_vert] = np.where(
        (v_cell_left >= 0) & (v_cell_right >= 0), v_cell_right, -1
    )

    # --- Diagonal springs: 2*H*W (interleaved diag1, diag2 per cell) ---
    d_vy = np.repeat(np.arange(H), W)
    d_vx = np.tile(np.arange(W), H)
    # diag1: TL-BR, diag2: TR-BL
    d_tl = (d_vy + 1) * W1 + d_vx
    d_tr = (d_vy + 1) * W1 + (d_vx + 1)
    d_bl = d_vy * W1 + d_vx
    d_br = d_vy * W1 + (d_vx + 1)

    # Interleave: diag1 at even indices, diag2 at odd
    d_indices = np.arange(n_diag)
    d_cell_idx = d_indices // 2  # which cell this diagonal belongs to
    d_is_diag2 = (d_indices % 2 == 1)

    # Existing convention: diag1 = BL→TR, diag2 = BR→TL
    d_a = np.where(d_is_diag2, d_br[d_cell_idx], d_bl[d_cell_idx])
    d_b = np.where(d_is_diag2, d_tl[d_cell_idx], d_tr[d_cell_idx])

    spring_a_idx[diag_offset:] = d_a
    spring_b_idx[diag_offset:] = d_b
    spring_types[diag_offset:] = 2
    # Diagonals belong to exactly one cell
    spring_cell_a[diag_offset:] = d_vy[d_cell_idx] * W + d_vx[d_cell_idx]
    spring_cell_b[diag_offset:] = -1

    # --- Spring rest lengths ---
    dx = positions[spring_b_idx, 0] - positions[spring_a_idx, 0]
    dy = positions[spring_b_idx, 1] - positions[spring_a_idx, 1]
    spring_rest_length = np.sqrt(dx * dx + dy * dy).astype(np.float32)

    # --- Per-cell data ---
    n_cells = H * W
    c_vy = np.repeat(np.arange(H), W)
    c_vx = np.tile(np.arange(W), H)

    # Corner indices: [BL, BR, TR, TL]
    cell_bl = c_vy * W1 + c_vx
    cell_br = c_vy * W1 + (c_vx + 1)
    cell_tr = (c_vy + 1) * W1 + (c_vx + 1)
    cell_tl = (c_vy + 1) * W1 + c_vx
    cell_corner_indices = np.stack([cell_bl, cell_br, cell_tr, cell_tl], axis=-1).astype(np.int32)

    # Per-cell spring indices
    cell_horiz_bottom = (c_vy * W + c_vx).astype(np.int32)
    cell_horiz_top = ((c_vy + 1) * W + c_vx).astype(np.int32)
    cell_vert_left = (vert_offset + c_vy * W1 + c_vx).astype(np.int32)
    cell_vert_right = (vert_offset + c_vy * W1 + (c_vx + 1)).astype(np.int32)
    cell_diag1 = (diag_offset + 2 * (c_vy * W + c_vx)).astype(np.int32)
    cell_diag2 = (diag_offset + 2 * (c_vy * W + c_vx) + 1).astype(np.int32)

    return GridData(
        positions=jnp.array(positions),
        spring_a_idx=jnp.array(spring_a_idx),
        spring_b_idx=jnp.array(spring_b_idx),
        spring_rest_length=jnp.array(spring_rest_length),
        spring_types=jnp.array(spring_types),
        spring_cell_a=jnp.array(spring_cell_a),
        spring_cell_b=jnp.array(spring_cell_b),
        cell_corner_indices=jnp.array(cell_corner_indices),
        cell_grid_vy=jnp.array(c_vy, dtype=jnp.int32),
        cell_grid_vx=jnp.array(c_vx, dtype=jnp.int32),
        cell_horiz_bottom=jnp.array(cell_horiz_bottom),
        cell_horiz_top=jnp.array(cell_horiz_top),
        cell_vert_left=jnp.array(cell_vert_left),
        cell_vert_right=jnp.array(cell_vert_right),
        cell_diag1=jnp.array(cell_diag1),
        cell_diag2=jnp.array(cell_diag2),
        H=H,
        W=W,
    )


# ---------------------------------------------------------------------------
# Grid composition (JAX, JIT-able)
# ---------------------------------------------------------------------------

def jax_compose_grid(
    terrain: jnp.ndarray,
    robot_morph: jnp.ndarray,
    spawn_y: int,
    spawn_x: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compose robot onto terrain grid.

    Args:
        terrain: (H, W) int32 — terrain grid (y-up, FIXED voxels)
        robot_morph: (rH, rW) int32 — robot morphology (y-up, already flipped)
        spawn_y, spawn_x: int — spawn position in grid coordinates

    Returns:
        composited: (H, W) int32 — terrain + robot
        robot_mask: (H, W) bool — True where robot cells are placed
    """
    _ensure_no_slope_cells(terrain, context="jax_compose_grid")
    _ensure_no_slope_cells(robot_morph, context="jax_compose_grid")
    robot_padded = jnp.zeros_like(terrain)
    robot_padded = jax.lax.dynamic_update_slice(
        robot_padded, robot_morph, (spawn_y, spawn_x)
    )
    robot_mask = robot_padded != C.EMPTY
    composited = jnp.where(robot_mask, robot_padded, terrain)
    return composited, robot_mask


# ---------------------------------------------------------------------------
# SimState construction (JAX, JIT-able, vmap-able)
# ---------------------------------------------------------------------------

def jax_build_sim_state(
    morphology: jnp.ndarray,
    grid_data: GridData,
    spawn_x: float = 0.0,
    spawn_y: float = 0.0,
    spring_stiffness_scale: float = 1.0,
) -> tuple[SimState, SpringTopology]:
    """Build SimState from composited morphology grid. Pure JAX, JIT-able.

    Implements the two-pass spring constant assignment matching
    C++ ObjectCreator's behavior. Assumes composited_grid=True semantics.

    Args:
        morphology: (H, W) int32 — composited grid (y-up)
        grid_data: GridData — pre-computed grid infrastructure
        spawn_x, spawn_y: float — spawn offset in grid units
        spring_stiffness_scale: scales CONTRACTILE main spring stiffness; CONTRACTILE
            diagonal stiffness follows an anchor-based bounded ratio law.
    """
    _ensure_no_slope_cells(morphology, context="jax_build_sim_state")
    gd = grid_data
    n_points = (gd.H + 1) * (gd.W + 1)
    n_springs = gd.spring_a_idx.shape[0]
    morph_flat = morphology.flatten()
    corners = gd.cell_corner_indices  # (H*W, 4)

    # --- Point properties ---
    cell_active = morph_flat != C.EMPTY
    cell_is_fixed = morph_flat == C.FIXED
    cell_is_nonfixed = cell_active & ~cell_is_fixed

    # Scatter cell_active to corner points (OR across adjacent cells)
    point_mask = jnp.zeros(n_points, dtype=jnp.bool_)
    for c in range(4):
        point_mask = point_mask.at[corners[:, c]].max(cell_active)

    masses = jnp.where(
        jnp.broadcast_to(point_mask[:, jnp.newaxis], (n_points, 2)),
        C.POINT_MASS, 0.0,
    ).astype(jnp.float32)

    # Fixed points: composited grid logic
    # A point is fixed if it is NOT touched by any non-FIXED non-EMPTY cell.
    # This covers ghosts (untouched) and exclusively-terrain points.
    point_is_nonfixed_touched = jnp.zeros(n_points, dtype=jnp.bool_)
    for c in range(4):
        point_is_nonfixed_touched = point_is_nonfixed_touched.at[corners[:, c]].max(
            cell_is_nonfixed
        )
    fixed = jnp.broadcast_to(
        (~point_is_nonfixed_touched)[:, jnp.newaxis], (n_points, 2)
    )

    # --- Spring properties: two-pass algorithm ---
    is_main = gd.spring_types < 2  # horiz or vert
    is_diag = gd.spring_types == 2

    # Safe cell type lookup (clamp -1 to 0, then mask)
    safe_a = jnp.maximum(gd.spring_cell_a, 0)
    safe_b = jnp.maximum(gd.spring_cell_b, 0)
    type_a = jnp.where(gd.spring_cell_a >= 0, morph_flat[safe_a], C.EMPTY)
    type_b = jnp.where(gd.spring_cell_b >= 0, morph_flat[safe_b], C.EMPTY)

    # Spring activation
    a_active = type_a != C.EMPTY
    b_active = type_b != C.EMPTY
    spring_active_main = (a_active | b_active) & is_main
    spring_active_diag = a_active & is_diag  # diags: cell_a = owning cell
    spring_mask = spring_active_main | spring_active_diag

    # Two-pass main spring constants: "last writer wins" in row-major
    # Pass 1: all main springs start at RIGID_MAIN_K
    # Pass 2: SOFT/ACT cells overwrite; cell_b (earlier) then cell_a (later)
    def k_for_type(vtype):
        return jnp.where(
            _is_actuator_type(vtype), C.ACTUATOR_MAIN_K,
            jnp.where(vtype == C.SOFT, C.SOFT_MAIN_K, C.RIGID_MAIN_K),
        )

    a_overwrites = (type_a == C.SOFT) | _is_actuator_type(type_a)
    b_overwrites = (type_b == C.SOFT) | _is_actuator_type(type_b)

    main_k = jnp.full(n_springs, C.RIGID_MAIN_K, dtype=jnp.float32)
    main_k = jnp.where(b_overwrites & is_main, k_for_type(type_b), main_k)
    main_k = jnp.where(a_overwrites & is_main, k_for_type(type_a), main_k)

    # Diagonal spring constants (from owning cell)
    diag_owner = type_a  # diags: cell_a = owning cell
    diag_k = jnp.where(
        _is_actuator_type(diag_owner), C.ACTUATOR_DIAG_K,
        jnp.where(diag_owner == C.SOFT, C.SOFT_DIAG_K, C.RIGID_DIAG_K),
    )

    spring_const = jnp.where(is_main, main_k, diag_k) * spring_mask
    scale = jnp.asarray(spring_stiffness_scale, dtype=jnp.float32)
    contractile_diag_ratio_value = contractile_diag_ratio_jax(scale)
    main_writer_contractile = jnp.where(
        a_overwrites,
        type_a == C.CONTRACTILE,
        jnp.where(b_overwrites, type_b == C.CONTRACTILE, False),
    )
    diag_writer_contractile = diag_owner == C.CONTRACTILE
    contractile_main_mask = (is_main & main_writer_contractile) & spring_mask
    contractile_diag_mask = (is_diag & diag_writer_contractile) & spring_mask
    contractile_main_k = jnp.float32(C.ACTUATOR_MAIN_K) * scale
    contractile_diag_k = contractile_main_k * contractile_diag_ratio_value
    spring_const = jnp.where(contractile_main_mask, contractile_main_k, spring_const)
    spring_const = jnp.where(contractile_diag_mask, contractile_diag_k, spring_const)
    rest_len = gd.spring_rest_length * spring_mask

    # Current dense-grid builder semantics: mark springs touched by RIGID cells
    # and by FIXED cells that remain in a dynamic mixed object. This is broader
    # than the upstream C++ rigid-boxel Phase 2 traversal.
    cell_rigid_or_fixed = (morph_flat == C.RIGID) | (morph_flat == C.FIXED)
    rigid_a = jnp.where(gd.spring_cell_a >= 0, cell_rigid_or_fixed[safe_a], False)
    rigid_b = jnp.where(gd.spring_cell_b >= 0, cell_rigid_or_fixed[safe_b], False)
    main_rigid = (rigid_a | rigid_b) & is_main
    diag_rigid = rigid_a & is_diag  # diags: cell_a = owning cell
    rigid_spring_mask = (main_rigid | diag_rigid) & spring_mask

    # --- Apply spawn offset ---
    offset = jnp.array([spawn_x * C.CELL_SIZE, spawn_y * C.CELL_SIZE])
    positions = gd.positions + jnp.where(point_mask[:, jnp.newaxis], offset, 0.0)

    zeros_pts = jnp.zeros((n_points, 2), dtype=jnp.float32)

    state = SimState(
        positions=positions,
        velocities=zeros_pts,
        positions_last=positions,
        velocities_true=zeros_pts,
        masses=masses,
        fixed=fixed,
        point_mask=point_mask,
        spring_rest_length=rest_len,
        spring_rest_length_goal=rest_len,
        spring_init_rest_length=rest_len,
        spring_const=spring_const,
        spring_mask=spring_mask,
        rigid_spring_mask=rigid_spring_mask,
        external_forces=zeros_pts,
        tangential_deformation=zeros_pts,
        friction_anchor=positions,
        friction_anchor_active=jnp.zeros((n_points,), dtype=jnp.bool_),
        friction_anchor_no_contact_count=jnp.zeros((n_points,), dtype=jnp.int32),
        static_manifold_cell_idx=jnp.full((n_points,), -1, dtype=jnp.int32),
        static_manifold_edge_idx=jnp.full((n_points,), -1, dtype=jnp.int32),
        static_manifold_active=jnp.zeros((n_points,), dtype=jnp.bool_),
        static_manifold_no_contact_count=jnp.zeros((n_points,), dtype=jnp.int32),
        static_manifold_normal_force=jnp.zeros((n_points,), dtype=jnp.float32),
        static_manifold_normal=jnp.zeros((n_points, 2), dtype=jnp.float32),
    )

    topology = SpringTopology(a_idx=gd.spring_a_idx, b_idx=gd.spring_b_idx)

    return state, topology


# ---------------------------------------------------------------------------
# Actuator info (JAX, JIT-able, vmap-able)
# ---------------------------------------------------------------------------

def jax_build_actuator_info(
    morphology: jnp.ndarray,
    grid_data: GridData,
) -> tuple[ActuatorInfo, jnp.ndarray]:
    """Build padded scalar ActuatorInfo from morphology grid.

    This helper exists for the older scalar actuation path, where actions are
    world-aligned. CONTRACTILE bodies must use PerAxisActuatorInfo and the
    per-axis rollout path instead.

    Returns:
        (ActuatorInfo, actuator_mask) where actuator_mask is (H*W,) bool.
    """
    _ensure_no_slope_cells(morphology, context="jax_build_actuator_info")
    gd = grid_data
    morph_flat = morphology.flatten()
    n_springs = gd.spring_a_idx.shape[0]

    is_h_act = morph_flat == C.H_ACT
    is_v_act = morph_flat == C.V_ACT
    is_actuator = is_h_act | is_v_act

    # H_ACT: (bottom_horiz, top_horiz), V_ACT: (left_vert, right_vert)
    spring_a = jnp.where(
        is_h_act, gd.cell_horiz_bottom,
        jnp.where(is_v_act, gd.cell_vert_left, 0),
    )
    spring_b = jnp.where(
        is_h_act, gd.cell_horiz_top,
        jnp.where(is_v_act, gd.cell_vert_right, 0),
    )
    cell_spring_indices = jnp.stack([spring_a, spring_b], axis=-1)  # (H*W, 2)

    # spring_act_count: scatter-add 1.0 for each actuator cell's springs
    act_float = is_actuator.astype(jnp.float32)
    act_count = jnp.zeros(n_springs, dtype=jnp.float32)
    act_count = act_count.at[spring_a].add(act_float)
    act_count = act_count.at[spring_b].add(act_float)

    actuated_mask = act_count > 0

    return ActuatorInfo(
        cell_spring_indices=cell_spring_indices,
        spring_act_count=act_count,
        actuated_spring_mask=actuated_mask,
    ), is_actuator


def jax_build_per_axis_actuator_info(
    morphology: jnp.ndarray,
    grid_data: GridData,
) -> tuple[PerAxisActuatorInfo, jnp.ndarray]:
    """Build padded PerAxisActuatorInfo from morphology grid.

    CONTRACTILE cells actuate both axis pairs, H_ACT only horizontal, V_ACT only
    vertical. Non-actuator rows are zero-index padded and contribute no action.
    """
    _ensure_no_slope_cells(morphology, context="jax_build_per_axis_actuator_info")
    gd = grid_data
    morph_flat = morphology.flatten()
    n_cells = morph_flat.shape[0]
    n_springs = gd.spring_a_idx.shape[0]

    is_h_act = morph_flat == C.H_ACT
    is_v_act = morph_flat == C.V_ACT
    is_contractile = morph_flat == C.CONTRACTILE

    h_enabled = is_h_act | is_contractile
    v_enabled = is_v_act | is_contractile
    is_actuator = h_enabled | v_enabled

    h_spring_a = jnp.where(h_enabled, gd.cell_horiz_bottom, 0)
    h_spring_b = jnp.where(h_enabled, gd.cell_horiz_top, 0)
    v_spring_a = jnp.where(v_enabled, gd.cell_vert_left, 0)
    v_spring_b = jnp.where(v_enabled, gd.cell_vert_right, 0)

    h_spring_pairs = jnp.stack([h_spring_a, h_spring_b], axis=-1)
    v_spring_pairs = jnp.stack([v_spring_a, v_spring_b], axis=-1)

    h_compact_indices = jnp.nonzero(h_enabled, size=n_cells, fill_value=0)[0].astype(jnp.int32)
    v_compact_indices = jnp.nonzero(v_enabled, size=n_cells, fill_value=0)[0].astype(jnp.int32)
    h_compact_pairs = h_spring_pairs[h_compact_indices]
    v_compact_pairs = v_spring_pairs[v_compact_indices]
    h_compact_count = jnp.sum(h_enabled.astype(jnp.int32))
    v_compact_count = jnp.sum(v_enabled.astype(jnp.int32))

    h_float = h_enabled.astype(jnp.float32)
    v_float = v_enabled.astype(jnp.float32)

    act_count = jnp.zeros(n_springs, dtype=jnp.float32)
    act_count = act_count.at[h_spring_a].add(h_float)
    act_count = act_count.at[h_spring_b].add(h_float)
    act_count = act_count.at[v_spring_a].add(v_float)
    act_count = act_count.at[v_spring_b].add(v_float)

    actuated_mask = act_count > 0.0

    return PerAxisActuatorInfo(
        h_spring_pairs=h_spring_pairs,
        v_spring_pairs=v_spring_pairs,
        h_compact_cell_indices=h_compact_indices,
        v_compact_cell_indices=v_compact_indices,
        h_compact_spring_pairs=h_compact_pairs,
        v_compact_spring_pairs=v_compact_pairs,
        h_compact_count=h_compact_count,
        v_compact_count=v_compact_count,
        actuated_spring_mask=actuated_mask,
        spring_act_count=act_count,
    ), is_actuator


# ---------------------------------------------------------------------------
# Deformation info (JAX, JIT-able, vmap-able)
# ---------------------------------------------------------------------------

def jax_build_deformation_info(
    morphology: jnp.ndarray,
    grid_data: GridData,
    robot_mask: jnp.ndarray,
) -> tuple[DeformationInfo, jnp.ndarray]:
    """Build padded DeformationInfo from morphology + robot mask.

    Padded to (H*W, 2). Non-robot rows get -1 sentinel (maps to
    ratio=1.0 via compute_deformation's ratios_padded).

    Returns:
        (DeformationInfo, deform_mask) where deform_mask is (H*W,) bool.
    """
    _ensure_no_slope_cells(morphology, context="jax_build_deformation_info")
    gd = grid_data
    morph_flat = morphology.flatten()
    robot_flat = robot_mask.flatten()

    is_robot_voxel = (morph_flat != C.EMPTY) & robot_flat

    horiz = jnp.where(
        is_robot_voxel[:, jnp.newaxis],
        jnp.stack([gd.cell_horiz_bottom, gd.cell_horiz_top], axis=-1),
        -1,
    )
    vert = jnp.where(
        is_robot_voxel[:, jnp.newaxis],
        jnp.stack([gd.cell_vert_left, gd.cell_vert_right], axis=-1),
        -1,
    )

    return DeformationInfo(
        voxel_horiz_springs=horiz,
        voxel_vert_springs=vert,
        n_robot_voxels=jnp.sum(is_robot_voxel),
    ), is_robot_voxel


# ---------------------------------------------------------------------------
# Collision data (JAX, JIT-able, vmap-able)
# ---------------------------------------------------------------------------

def jax_build_collision_data(
    morphology: jnp.ndarray,
    robot_mask: jnp.ndarray,
    grid_data: GridData,
    max_triples: int = C.MAX_TRIPLES,
    max_self_triples: int = C.MAX_SELF_TRIPLES,
    max_active_boxels: int | None = None,
) -> CollisionData:
    """Build CollisionData from morphology + robot mask. Pure JAX, JIT-able.

    Uses a sparse-active pair path: pairwise checks are computed over active
    boxels only (optionally capped by max_active_boxels) and then mapped into
    fixed-size indexed triple buffers.
    """
    _ensure_no_slope_cells(morphology, context="jax_build_collision_data")
    gd = grid_data
    n_boxels = gd.H * gd.W
    n_points = (gd.H + 1) * (gd.W + 1)
    morph_flat = morphology.flatten()
    robot_flat = robot_mask.flatten()

    # --- Category: 0=empty, 1=robot, 2=terrain ---
    category = jnp.where(
        morph_flat == C.EMPTY, 0,
        jnp.where(robot_flat, 1, 2),
    )
    boxel_mask = morph_flat != C.EMPTY

    # --- Boxel corners with terrain offset ---
    corners = gd.cell_corner_indices  # (n_boxels, 4) [BL, BR, TR, TL]
    is_terrain = boxel_mask & ~robot_flat
    terrain_offset = jnp.where(is_terrain[:, jnp.newaxis], n_points, 0)
    boxel_corners = corners + terrain_offset

    # --- Edge endpoints: edge e from corner (e+1)%4 to corner e ---
    roll_order = jnp.array([1, 2, 3, 0])
    boxel_edge_a = boxel_corners[:, roll_order]
    boxel_edge_b = boxel_corners

    # --- Surface edges ---
    vy = gd.cell_grid_vy
    vx = gd.cell_grid_vx
    adj_offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    surface_edge_mask = jnp.zeros((n_boxels, 4), dtype=jnp.bool_)
    for e, (dy, dx) in enumerate(adj_offsets):
        ny = vy + dy
        nx = vx + dx
        in_bounds = (ny >= 0) & (ny < gd.H) & (nx >= 0) & (nx < gd.W)
        adj_idx = jnp.clip(ny * gd.W + nx, 0, n_boxels - 1)
        adj_cat = jnp.where(in_bounds, category[adj_idx], 0)
        is_surface = (~in_bounds) | (adj_cat == 0) | (adj_cat != category)
        surface_edge_mask = surface_edge_mask.at[:, e].set(boxel_mask & is_surface)

    # --- Point-is-robot for corners ---
    n_total_points = 2 * n_points
    point_is_robot = jnp.zeros(n_total_points, dtype=jnp.bool_)
    for c in range(4):
        pidx = boxel_corners[:, c]
        point_is_robot = point_is_robot.at[pidx].max(robot_flat & boxel_mask)

    # Mark terrain points as NOT robot (safety)
    point_is_terrain = jnp.zeros(n_total_points, dtype=jnp.bool_)
    for c in range(4):
        pidx = boxel_corners[:, c]
        point_is_terrain = point_is_terrain.at[pidx].max(is_terrain)
    point_is_robot = point_is_robot & ~point_is_terrain

    corner_is_robot = point_is_robot[boxel_corners]  # (N, 4)

    # --- Sparse-active triple mask path (A, A, 4) ---
    # Compute pair checks over active boxel slots instead of full N x N.
    n_active_slots = n_boxels if max_active_boxels is None else int(max_active_boxels)
    n_active_slots = max(1, min(n_active_slots, n_boxels))
    active_indices = jnp.nonzero(boxel_mask, size=n_active_slots, fill_value=0)[0]
    n_active = jnp.minimum(jnp.sum(boxel_mask), jnp.int32(n_active_slots))
    active_valid = jnp.arange(n_active_slots, dtype=jnp.int32) < n_active

    active_category = category[active_indices]
    active_vy = vy[active_indices]
    active_vx = vx[active_indices]
    active_corners = boxel_corners[active_indices]
    active_corner_is_robot = corner_is_robot[active_indices]

    cat_i = active_category[:, jnp.newaxis]
    cat_j = active_category[jnp.newaxis, :]
    same_cat = cat_i == cat_j

    vy_i = active_vy[:, jnp.newaxis]
    vy_j = active_vy[jnp.newaxis, :]
    vx_i = active_vx[:, jnp.newaxis]
    vx_j = active_vx[jnp.newaxis, :]
    chebyshev = jnp.maximum(jnp.abs(vy_i - vy_j), jnp.abs(vx_i - vx_j))
    skip_adjacent = same_cat & (chebyshev <= 1)

    active_pair = active_valid[:, jnp.newaxis] & active_valid[jnp.newaxis, :]
    not_self = ~jnp.eye(n_active_slots, dtype=jnp.bool_)
    pair_valid = active_pair & not_self & ~skip_adjacent

    ci = active_corners[:, jnp.newaxis, :, jnp.newaxis]  # (A, 1, 4, 1)
    cj = active_corners[jnp.newaxis, :, jnp.newaxis, :]  # (1, A, 1, 4)
    shared = jnp.any(ci == cj, axis=-1)  # (A, A, 4)

    triple_check_mask = (
        pair_valid[:, :, jnp.newaxis]
        & active_corner_is_robot[:, jnp.newaxis, :]
        & ~shared
    )

    # Self-collision: both robot
    both_robot = (
        (active_category[:, jnp.newaxis] == 1)
        & (active_category[jnp.newaxis, :] == 1)
    )
    self_collision_mask = triple_check_mask & both_robot[:, :, jnp.newaxis]

    corner_edge_indices = jnp.array([[0, 3], [0, 1], [1, 2], [2, 3]], dtype=jnp.int32)

    # --- Dynamic broadphase sparse candidate pairs (derived from triple pairs) ---
    pair_has_any_triple = jnp.any(triple_check_mask, axis=-1)  # (A, A)
    flat_pair_mask = pair_has_any_triple.flatten()
    n_real_candidate_pairs = jnp.sum(flat_pair_mask)
    flat_pair_indices = jnp.nonzero(flat_pair_mask, size=max_triples, fill_value=0)[0]
    pair_slot_i = flat_pair_indices // n_active_slots
    pair_slot_j = flat_pair_indices % n_active_slots
    idx_candidate_pair_i = active_indices[pair_slot_i]
    idx_candidate_pair_j = active_indices[pair_slot_j]
    idx_candidate_pair_active = (
        jnp.arange(max_triples, dtype=jnp.int32)
        < jnp.minimum(n_real_candidate_pairs, max_triples)
    )

    # Slot lookup (active-slot pair -> candidate pair slot) for triple gating.
    pair_slot_lookup = -jnp.ones((n_active_slots, n_active_slots), dtype=jnp.int32)
    pair_slot_lookup = pair_slot_lookup.at[pair_slot_i, pair_slot_j].set(
        jnp.where(
            idx_candidate_pair_active,
            jnp.arange(max_triples, dtype=jnp.int32),
            jnp.int32(-1),
        )
    )

    # --- Indexed collision triples (fixed-size buffers) ---
    flat_mask = triple_check_mask.flatten()
    n_real_triples = jnp.sum(flat_mask)
    flat_indices = jnp.nonzero(flat_mask, size=max_triples, fill_value=0)[0]
    n4 = n_active_slots * 4
    slot_i = flat_indices // n4
    slot_j = (flat_indices % n4) // 4
    idx_triple_i = active_indices[slot_i]
    idx_triple_j = active_indices[slot_j]
    idx_triple_k = flat_indices % 4
    idx_triple_active = jnp.arange(max_triples) < jnp.minimum(n_real_triples, max_triples)
    idx_triple_pair_slot = jnp.where(
        idx_triple_active,
        jnp.maximum(pair_slot_lookup[slot_i, slot_j], 0),
        jnp.int32(0),
    )

    flat_self = self_collision_mask.flatten()
    n_real_self_triples = jnp.sum(flat_self)
    flat_self_indices = jnp.nonzero(flat_self, size=max_self_triples, fill_value=0)[0]
    self_slot_i = flat_self_indices // n4
    self_slot_j = (flat_self_indices % n4) // 4
    idx_self_i = active_indices[self_slot_i]
    idx_self_j = active_indices[self_slot_j]
    idx_self_k = flat_self_indices % 4
    idx_self_active = jnp.arange(max_self_triples) < jnp.minimum(
        n_real_self_triples, max_self_triples
    )
    n_robot_surface_boxels = jnp.sum(
        (boxel_mask & robot_flat & jnp.any(surface_edge_mask, axis=1)).astype(jnp.int32)
    )

    return CollisionData(
        boxel_corners=boxel_corners,
        boxel_edge_a=boxel_edge_a,
        boxel_edge_b=boxel_edge_b,
        boxel_mask=boxel_mask,
        surface_edge_mask=surface_edge_mask,
        boxel_object_id=jnp.where(
            boxel_mask,
            jnp.where(robot_flat, jnp.int32(0), jnp.int32(1)),
            jnp.int32(-1),
        ),
        boxel_is_robot=robot_flat & boxel_mask,
        boxel_world_vy=vy,
        boxel_world_vx=vx,
        corner_edge_indices=corner_edge_indices,
        n_real_points=n_points,
        triple_i=idx_triple_i,
        triple_j=idx_triple_j,
        triple_k=idx_triple_k,
        triple_active=idx_triple_active,
        triple_pair_slot=idx_triple_pair_slot,
        self_triple_i=idx_self_i,
        self_triple_j=idx_self_j,
        self_triple_k=idx_self_k,
        self_triple_active=idx_self_active,
        candidate_pair_i=idx_candidate_pair_i,
        candidate_pair_j=idx_candidate_pair_j,
        candidate_pair_active=idx_candidate_pair_active,
        n_triples=n_real_triples,
        n_self_triples=n_real_self_triples,
        n_candidate_pairs=n_real_candidate_pairs,
        n_robot_surface_boxels=n_robot_surface_boxels,
    )
