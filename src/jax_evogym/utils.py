"""Voxel-to-mass-spring conversion and connectivity helpers.

Uses numpy for one-time init. Output converts to JAX arrays via make_sim_state().
Implements C++ ObjectCreator's two-pass spring constant assignment.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math

import jax.numpy as jnp
import numpy as np

from . import constants as C
from .spring_scaling import contractile_diag_ratio
from .types import ActuatorInfo, DeformationInfo, SimState, SpringTopology

_SLOPE_TYPE_ARRAY = np.array(sorted(C.SLOPE_TYPES), dtype=np.int32)


# ---------------------------------------------------------------------------
# Dense grid helpers
# ---------------------------------------------------------------------------

def create_dense_point_grid(
    H: int = C.DEFAULT_GRID_H,
    W: int = C.DEFAULT_GRID_W,
) -> np.ndarray:
    """Create (H+1)*(W+1) point positions at cell_size spacing.

    Index = grid_y * (W+1) + grid_x.
    Returns (n_points, 2) numpy array in physical coordinates.
    """
    n_points = (H + 1) * (W + 1)
    positions = np.zeros((n_points, 2), dtype=np.float64)
    idx = 0
    for grid_y in range(H + 1):
        for grid_x in range(W + 1):
            positions[idx] = [grid_x * C.CELL_SIZE, grid_y * C.CELL_SIZE]
            idx += 1
    return positions


def create_dense_spring_indices(
    H: int = C.DEFAULT_GRID_H,
    W: int = C.DEFAULT_GRID_W,
) -> List[Tuple[int, int, str, int, int]]:
    """Create spring endpoint pairs in canonical order.

    Order: horizontal -> vertical -> diagonal (TL-BR, TR-BL per cell).
    Returns list of (a_idx, b_idx, type, grid_y, grid_x).
    """
    springs: List[Tuple[int, int, str, int, int]] = []
    W1 = W + 1

    # Horizontal springs: each row of points, left-to-right
    for grid_y in range(H + 1):
        for grid_x in range(W):
            p1 = grid_y * W1 + grid_x
            p2 = grid_y * W1 + (grid_x + 1)
            springs.append((p1, p2, "horiz", grid_y, grid_x))

    # Vertical springs: between rows, each column
    for grid_y in range(H):
        for grid_x in range(W + 1):
            p1 = grid_y * W1 + grid_x
            p2 = (grid_y + 1) * W1 + grid_x
            springs.append((p1, p2, "vert", grid_y, grid_x))

    # Diagonal springs: per cell, TL-BR then TR-BL
    for grid_y in range(H):
        for grid_x in range(W):
            tl = grid_y * W1 + grid_x
            tr = grid_y * W1 + (grid_x + 1)
            bl = (grid_y + 1) * W1 + grid_x
            br = (grid_y + 1) * W1 + (grid_x + 1)
            springs.append((tl, br, "diag1", grid_y, grid_x))
            springs.append((tr, bl, "diag2", grid_y, grid_x))

    return springs


# ---------------------------------------------------------------------------
# Spring-to-voxel mapping
# ---------------------------------------------------------------------------

def _build_spring_to_voxel_map(
    springs: List[Tuple[int, int, str, int, int]],
    H: int,
    W: int,
) -> Dict[Tuple[int, int], List[Tuple[int, str]]]:
    """Map each voxel (vy, vx) to its list of (spring_idx, spring_type).

    Voxel coordinate system: vy=0 is BOTTOM row of the padded grid,
    vy=H-1 is TOP row. grid_y in springs is also bottom-up.
    A cell at voxel (vy, vx) occupies grid_y=vy..vy+1, grid_x=vx..vx+1.
    """
    voxel_springs: Dict[Tuple[int, int], List[Tuple[int, str]]] = {}

    for spring_idx, (_, _, stype, grid_y, grid_x) in enumerate(springs):
        if stype == "horiz":
            # Horizontal spring at grid_y, between grid_x and grid_x+1
            # Bottom edge of voxel (vx, vy) where vy = grid_y (cell above)
            # Top edge of voxel (vx, vy) where vy = grid_y - 1 (cell below)
            vy_above = grid_y  # cell whose bottom edge this is
            vy_below = grid_y - 1  # cell whose top edge this is
            if 0 <= vy_above < H and 0 <= grid_x < W:
                voxel_springs.setdefault((vy_above, grid_x), []).append(
                    (spring_idx, "main")
                )
            if 0 <= vy_below < H and 0 <= grid_x < W:
                voxel_springs.setdefault((vy_below, grid_x), []).append(
                    (spring_idx, "main")
                )
        elif stype == "vert":
            # Vertical spring at grid_x, between grid_y and grid_y+1
            # Right edge of voxel (vx, vy) where vx = grid_x - 1
            # Left edge of voxel (vx, vy) where vx = grid_x
            vy = grid_y  # cell at this row
            vx_right = grid_x - 1
            vx_left = grid_x
            if 0 <= vy < H and 0 <= vx_right < W:
                voxel_springs.setdefault((vy, vx_right), []).append(
                    (spring_idx, "main")
                )
            if 0 <= vy < H and 0 <= vx_left < W:
                voxel_springs.setdefault((vy, vx_left), []).append(
                    (spring_idx, "main")
                )
        elif stype in ("diag1", "diag2"):
            # Diagonal belongs to exactly one cell
            vy = grid_y
            vx = grid_x
            if 0 <= vy < H and 0 <= vx < W:
                voxel_springs.setdefault((vy, vx), []).append(
                    (spring_idx, "diag")
                )

    return voxel_springs


def _identify_actuator_cells(
    springs: List[Tuple[int, int, str, int, int]],
    morph_padded: np.ndarray,
    H: int,
    W: int,
    n_springs: int,
) -> dict:
    """Identify per-cell actuator→spring mapping matching C++ PhysicsEngine.cpp:38-97.

    One action per actuator cell. Each cell drives exactly 2 springs.
    H_ACT at (vy, vx) → top + bottom horizontal edges.
    V_ACT at (vy, vx) → left + right vertical edges.
    Shared edges between adjacent same-type actuator cells get averaged.

    Returns dict with:
        actuator_cell_spring_indices: (n_cells, 2) int — spring pair per cell
        actuator_spring_act_count: (n_springs,) float — how many cells actuate each spring
        actuated_spring_mask: (n_springs,) bool
    """
    # Build lookup: (grid_y, grid_x) → spring_idx for horiz and vert
    horiz_idx: Dict[Tuple[int, int], int] = {}
    vert_idx: Dict[Tuple[int, int], int] = {}

    for spring_idx, (_, _, stype, grid_y, grid_x) in enumerate(springs):
        if stype == "horiz":
            horiz_idx[(grid_y, grid_x)] = spring_idx
        elif stype == "vert":
            vert_idx[(grid_y, grid_x)] = spring_idx

    cell_spring_pairs: List[Tuple[int, int]] = []
    spring_act_count = np.zeros(n_springs, dtype=np.float32)

    # Enumerate cells row-major (vy outer, vx inner) matching C++ Robot::init()
    for vy in range(H):
        for vx in range(W):
            vtype = morph_padded[vy, vx]
            if vtype == C.H_ACT:
                # Bottom horizontal edge at grid_y=vy, top at grid_y=vy+1
                bot = horiz_idx[(vy, vx)]
                top = horiz_idx[(vy + 1, vx)]
                cell_spring_pairs.append((bot, top))
                spring_act_count[bot] += 1.0
                spring_act_count[top] += 1.0
            elif vtype == C.V_ACT:
                # Left vertical edge at grid_x=vx, right at grid_x=vx+1
                left = vert_idx[(vy, vx)]
                right = vert_idx[(vy, vx + 1)]
                cell_spring_pairs.append((left, right))
                spring_act_count[left] += 1.0
                spring_act_count[right] += 1.0

    actuated_mask = spring_act_count > 0

    if len(cell_spring_pairs) == 0:
        cell_arr = np.zeros((0, 2), dtype=np.int32)
    else:
        cell_arr = np.array(cell_spring_pairs, dtype=np.int32)

    return {
        "actuator_cell_spring_indices": cell_arr,
        "actuator_spring_act_count": spring_act_count,
        "actuated_spring_mask": actuated_mask,
    }


def _build_deformation_info(
    springs: List[Tuple[int, int, str, int, int]],
    morph_padded: np.ndarray,
    H: int,
    W: int,
    robot_cell_positions: set | None = None,
) -> DeformationInfo:
    """Build voxel→spring mapping for deformation sensing.

    For each non-empty robot voxel, identifies its horizontal (bottom/top)
    and vertical (left/right) edge springs. Uses -1 sentinel for missing
    springs (edge-of-grid voxels).

    Args:
        springs: Spring list from create_dense_spring_indices.
        morph_padded: (H, W) padded morphology grid.
        H, W: Grid dimensions.
        robot_cell_positions: set of (world_x, world_y) for robot cells.
            If None, all non-empty non-FIXED cells are robot cells.
    """
    # Build (grid_y, grid_x) → spring_idx lookups
    horiz_idx: Dict[Tuple[int, int], int] = {}
    vert_idx: Dict[Tuple[int, int], int] = {}
    for spring_idx, (_, _, stype, grid_y, grid_x) in enumerate(springs):
        if stype == "horiz":
            horiz_idx[(grid_y, grid_x)] = spring_idx
        elif stype == "vert":
            vert_idx[(grid_y, grid_x)] = spring_idx

    # Determine which cells are robot voxels
    if robot_cell_positions is not None:
        robot_cells = robot_cell_positions
    else:
        robot_cells = set()
        for vy in range(H):
            for vx in range(W):
                if morph_padded[vy, vx] not in (C.EMPTY, C.FIXED):
                    robot_cells.add((vx, vy))

    # Enumerate robot voxels in row-major order (vy outer, vx inner)
    horiz_springs = []  # (bottom, top) per voxel
    vert_springs = []   # (left, right) per voxel

    for vy in range(H):
        for vx in range(W):
            if morph_padded[vy, vx] == C.EMPTY:
                continue
            if (vx, vy) not in robot_cells:
                continue

            # Bottom horizontal: grid_y=vy, grid_x=vx (connects BL↔BR)
            bot = horiz_idx.get((vy, vx), -1)
            # Top horizontal: grid_y=vy+1, grid_x=vx (connects TL↔TR)
            top = horiz_idx.get((vy + 1, vx), -1)
            horiz_springs.append((bot, top))

            # Left vertical: grid_y=vy, grid_x=vx (connects BL↔TL)
            left = vert_idx.get((vy, vx), -1)
            # Right vertical: grid_y=vy, grid_x=vx+1 (connects BR↔TR)
            right = vert_idx.get((vy, vx + 1), -1)
            vert_springs.append((left, right))

    n_robot_voxels = len(horiz_springs)
    if n_robot_voxels == 0:
        h_arr = np.zeros((0, 2), dtype=np.int32)
        v_arr = np.zeros((0, 2), dtype=np.int32)
    else:
        h_arr = np.array(horiz_springs, dtype=np.int32)
        v_arr = np.array(vert_springs, dtype=np.int32)

    return DeformationInfo(
        voxel_horiz_springs=jnp.array(h_arr, dtype=jnp.int32),
        voxel_vert_springs=jnp.array(v_arr, dtype=jnp.int32),
        n_robot_voxels=n_robot_voxels,
    )


# ---------------------------------------------------------------------------
# Main conversion: two-pass algorithm
# ---------------------------------------------------------------------------

def _build_mass_spring_from_padded(
    morph_padded: np.ndarray,
    H: int,
    W: int,
    composited_grid: bool = False,
    spring_stiffness_scale: float = 1.0,
) -> dict:
    """Core two-pass spring constant assignment from a padded y-up grid.

    Implements C++ ObjectCreator's two-pass algorithm:
      Pass 1: All main edges of non-empty cells activated with RIGID spring constant.
      Pass 2: SOFT/ACT cells overwrite their main edges; all cells add diagonals.
      CONTRACTILE main springs scale by spring_stiffness_scale; diagonals use an
      anchor-based bounded ratio law.

    Args:
        morph_padded: (H, W) numpy int array. Row 0 = bottom (y-up convention).
        H, W: Grid dimensions.

    Returns dict with all arrays needed to construct SimState.
    """
    if spring_stiffness_scale <= 0.0:
        raise ValueError(
            "spring_stiffness_scale must be > 0; "
            f"got {spring_stiffness_scale}."
        )
    if np.any(np.isin(morph_padded, _SLOPE_TYPE_ARRAY)):
        raise ValueError(
            "dense mass-spring conversion only supports square cells; "
            "compile slope terrain through build.compile_world_template(...) instead"
        )

    # Create grid infrastructure
    positions = create_dense_point_grid(H, W)
    springs = create_dense_spring_indices(H, W)
    n_points = (H + 1) * (W + 1)
    n_springs = len(springs)

    # Initialize arrays
    masses = np.zeros((n_points, 2), dtype=np.float64)
    fixed = np.ones((n_points, 2), dtype=bool)  # ghosts are fixed
    point_mask = np.zeros(n_points, dtype=bool)

    spring_const = np.zeros(n_springs, dtype=np.float64)
    spring_rest_length = np.zeros(n_springs, dtype=np.float64)
    spring_mask = np.zeros(n_springs, dtype=bool)
    rigid_spring_mask = np.zeros(n_springs, dtype=bool)

    diagonal_length = math.sqrt(C.CELL_SIZE**2 + C.CELL_SIZE**2)

    # Build voxel-to-spring mapping
    voxel_spring_map = _build_spring_to_voxel_map(springs, H, W)

    W1 = W + 1

    # -----------------------------------------------------------------------
    # Pass 1: Activate points and main edges with RIGID constant
    # -----------------------------------------------------------------------
    for vy in range(H):
        for vx in range(W):
            vtype = morph_padded[vy, vx]
            if vtype == C.EMPTY:
                continue

            # Activate 4 corner points
            bl = vy * W1 + vx
            br = vy * W1 + (vx + 1)
            tl = (vy + 1) * W1 + vx
            tr = (vy + 1) * W1 + (vx + 1)

            for pidx in (bl, br, tl, tr):
                masses[pidx] = [C.POINT_MASS, C.POINT_MASS]
                if vtype == C.FIXED:
                    fixed[pidx] = True
                else:
                    fixed[pidx] = False
                point_mask[pidx] = True

            # Activate main edge springs with RIGID constant
            for sidx, edge_type in voxel_spring_map.get((vy, vx), []):
                if edge_type != "main":
                    continue
                if spring_mask[sidx]:
                    continue  # already activated by adjacent cell

                a, b = springs[sidx][0], springs[sidx][1]
                rest_len = np.linalg.norm(positions[a] - positions[b])

                spring_const[sidx] = C.RIGID_MAIN_K
                spring_rest_length[sidx] = rest_len
                spring_mask[sidx] = True

    # -----------------------------------------------------------------------
    # Pass 2: Cell-type customization + diagonals
    # -----------------------------------------------------------------------
    for vy in range(H):
        for vx in range(W):
            vtype = morph_padded[vy, vx]
            if vtype == C.EMPTY:
                continue

            cell_springs = voxel_spring_map.get((vy, vx), [])

            if vtype in (C.RIGID, C.FIXED):
                # Keep main edges at RIGID (from pass 1)
                # Add diagonals with RIGID_DIAG_K
                for sidx, edge_type in cell_springs:
                    if edge_type == "diag":
                        a, b = springs[sidx][0], springs[sidx][1]
                        spring_const[sidx] = C.RIGID_DIAG_K
                        spring_rest_length[sidx] = diagonal_length
                        spring_mask[sidx] = True
                    # Current builder semantics: mark all springs of RIGID/FIXED
                    # cells for PBD Phase 2. Upstream C++ only iterates rigid
                    # boxels in its second PBD pass.
                    rigid_spring_mask[sidx] = True

            elif vtype == C.SOFT:
                # Overwrite main edges to SOFT_MAIN_K
                for sidx, edge_type in cell_springs:
                    if edge_type == "main":
                        spring_const[sidx] = C.SOFT_MAIN_K
                    elif edge_type == "diag":
                        a, b = springs[sidx][0], springs[sidx][1]
                        spring_const[sidx] = C.SOFT_DIAG_K
                        spring_rest_length[sidx] = diagonal_length
                        spring_mask[sidx] = True

            elif vtype in (C.H_ACT, C.V_ACT):
                # Overwrite main edges to ACTUATOR_MAIN_K
                for sidx, edge_type in cell_springs:
                    if edge_type == "main":
                        spring_const[sidx] = C.ACTUATOR_MAIN_K
                    elif edge_type == "diag":
                        a, b = springs[sidx][0], springs[sidx][1]
                        spring_const[sidx] = C.ACTUATOR_DIAG_K
                        spring_rest_length[sidx] = diagonal_length
                        spring_mask[sidx] = True

            elif vtype == C.CONTRACTILE:
                scaled_main_k = C.ACTUATOR_MAIN_K * spring_stiffness_scale
                scaled_diag_k = scaled_main_k * contractile_diag_ratio(
                    spring_stiffness_scale
                )
                for sidx, edge_type in cell_springs:
                    if edge_type == "main":
                        spring_const[sidx] = scaled_main_k
                    elif edge_type == "diag":
                        spring_const[sidx] = scaled_diag_k
                        spring_rest_length[sidx] = diagonal_length
                        spring_mask[sidx] = True

    # Identify robot vs terrain point sets
    robot_point_set = set()
    terrain_point_set = set()
    for vy in range(H):
        for vx in range(W):
            vtype = morph_padded[vy, vx]
            if vtype == C.EMPTY:
                continue
            bl = vy * W1 + vx
            br = vy * W1 + (vx + 1)
            tl = (vy + 1) * W1 + vx
            tr = (vy + 1) * W1 + (vx + 1)
            corners = (bl, br, tl, tr)
            if vtype == C.FIXED:
                for pidx in corners:
                    terrain_point_set.add(pidx)
            else:
                for pidx in corners:
                    robot_point_set.add(pidx)

    if composited_grid:
        # Composited grid: robot + terrain share a single grid.
        # In C++, each object has separate points, so shared boundary
        # points belong to the robot (movable). Only fix points that
        # are exclusively corners of FIXED cells.
        exclusively_terrain = terrain_point_set - robot_point_set
        for pidx in range(n_points):
            if pidx in exclusively_terrain:
                fixed[pidx] = True
            elif pidx in robot_point_set:
                fixed[pidx] = False
    else:
        # Robot-only grid: FIXED voxels within a single object fix
        # their corners, even if shared with non-FIXED voxels.
        for vy in range(H):
            for vx in range(W):
                if morph_padded[vy, vx] != C.FIXED:
                    continue
                bl = vy * W1 + vx
                br = vy * W1 + (vx + 1)
                tl = (vy + 1) * W1 + vx
                tr = (vy + 1) * W1 + (vx + 1)
                for pidx in (bl, br, tl, tr):
                    fixed[pidx] = True

    # Identify actuator cells (per-cell mapping matching C++)
    actuator_cell_data = _identify_actuator_cells(
        springs, morph_padded, H, W, n_springs
    )

    robot_point_indices = np.array(sorted(robot_point_set), dtype=np.int32)
    terrain_point_indices = np.array(sorted(terrain_point_set), dtype=np.int32)

    return {
        "positions": positions,
        "masses": masses,
        "fixed": fixed,
        "point_mask": point_mask,
        "spring_a_idx": np.array([s[0] for s in springs], dtype=np.int32),
        "spring_b_idx": np.array([s[1] for s in springs], dtype=np.int32),
        "spring_const": spring_const,
        "spring_rest_length": spring_rest_length,
        "spring_mask": spring_mask,
        "rigid_spring_mask": rigid_spring_mask,
        "actuator_cell_spring_indices": actuator_cell_data["actuator_cell_spring_indices"],
        "actuator_spring_act_count": actuator_cell_data["actuator_spring_act_count"],
        "actuated_spring_mask": actuator_cell_data["actuated_spring_mask"],
        "robot_point_indices": robot_point_indices,
        "terrain_point_indices": terrain_point_indices,
        "n_points": n_points,
        "n_springs": n_springs,
        "_springs": springs,
        "_morph_padded": morph_padded,
        "_H": H,
        "_W": W,
    }


def voxel_to_dense_mass_spring(
    morphology: np.ndarray,
    H: int = C.DEFAULT_GRID_H,
    W: int = C.DEFAULT_GRID_W,
    spring_stiffness_scale: float = 1.0,
) -> dict:
    """Convert voxel morphology to dense mass-spring representation.

    Implements C++ ObjectCreator's two-pass spring constant assignment:
      Pass 1: All main edges activated with RIGID spring constant.
      Pass 2: SOFT/ACT cells overwrite their main edges; all cells add diagonals.

    The morphology is placed at the BOTTOM of the padded grid (vy=0 = bottom row).
    Voxel (vy, vx) corners map to grid points:
      BL = vy*(W+1) + vx,  BR = vy*(W+1) + vx+1
      TL = (vy+1)*(W+1) + vx,  TR = (vy+1)*(W+1) + vx+1

    Args:
        morphology: (H_morph, W_morph) numpy array of voxel types.
            Row 0 = top of robot (user convention). Flipped internally.
        H, W: Dense grid dimensions.

    Returns dict with all arrays needed to construct SimState.
    """
    H_morph, W_morph = morphology.shape
    if H_morph > H or W_morph > W:
        raise ValueError(
            f"Morphology ({H_morph}x{W_morph}) exceeds grid ({H}x{W})"
        )

    # Flip morphology: user row 0 = top, internal row 0 = bottom
    morph_flipped = np.flipud(morphology).astype(int)

    # Place at bottom of padded grid
    morph_padded = np.zeros((H, W), dtype=int)
    morph_padded[:H_morph, :W_morph] = morph_flipped

    return _build_mass_spring_from_padded(
        morph_padded,
        H,
        W,
        spring_stiffness_scale=spring_stiffness_scale,
    )


def voxel_to_dense_mass_spring_from_grid(
    grid: np.ndarray,
    H: int,
    W: int,
    spring_stiffness_scale: float = 1.0,
) -> dict:
    """Convert a y-up world grid directly to dense mass-spring representation.

    Like voxel_to_dense_mass_spring but takes the grid directly (already
    y-up, already composited with all objects placed). No flipud, no padding.

    Args:
        grid: (H, W) numpy int array of voxel types. Row 0 = bottom (y-up).
        H, W: Grid dimensions (must match grid.shape).

    Returns dict with all arrays needed to construct SimState.
    """
    if grid.shape != (H, W):
        raise ValueError(
            f"Grid shape {grid.shape} doesn't match ({H}, {W})"
        )
    return _build_mass_spring_from_padded(
        grid.astype(int),
        H,
        W,
        composited_grid=True,
        spring_stiffness_scale=spring_stiffness_scale,
    )


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------

def make_sim_state(
    dense_data: dict,
    spawn_x: float = 0.0,
    spawn_y: float = 0.0,
) -> Tuple[SimState, SpringTopology, ActuatorInfo]:
    """Convert dense_data dict to (SimState, SpringTopology, ActuatorInfo).

    Applies spawn offset to positions, converts numpy -> jax arrays.

    Args:
        dense_data: Output of voxel_to_dense_mass_spring().
        spawn_x, spawn_y: World position in grid units (multiplied by cell_size).
    """
    n_points = dense_data["n_points"]
    n_springs = dense_data["n_springs"]

    # Apply spawn offset (in physical coordinates) to active points only
    offset = np.array([spawn_x * C.CELL_SIZE, spawn_y * C.CELL_SIZE])
    positions = dense_data["positions"].copy()
    mask = dense_data["point_mask"]
    positions[mask] += offset

    # Convert to JAX
    pos_jax = jnp.array(positions, dtype=jnp.float32)
    zeros_pts = jnp.zeros((n_points, 2), dtype=jnp.float32)
    rest_len = jnp.array(dense_data["spring_rest_length"], dtype=jnp.float32)

    state = SimState(
        positions=pos_jax,
        velocities=zeros_pts,
        positions_last=pos_jax,
        velocities_true=zeros_pts,
        masses=jnp.array(dense_data["masses"], dtype=jnp.float32),
        fixed=jnp.array(dense_data["fixed"], dtype=jnp.bool_),
        point_mask=jnp.array(dense_data["point_mask"], dtype=jnp.bool_),
        spring_rest_length=rest_len,
        spring_rest_length_goal=rest_len,
        spring_init_rest_length=rest_len,
        spring_const=jnp.array(dense_data["spring_const"], dtype=jnp.float32),
        spring_mask=jnp.array(dense_data["spring_mask"], dtype=jnp.bool_),
        rigid_spring_mask=jnp.array(
            dense_data["rigid_spring_mask"], dtype=jnp.bool_
        ),
        external_forces=zeros_pts,
        tangential_deformation=zeros_pts,
        friction_anchor=pos_jax,
        friction_anchor_active=jnp.zeros((n_points,), dtype=jnp.bool_),
        friction_anchor_no_contact_count=jnp.zeros((n_points,), dtype=jnp.int32),
        static_manifold_cell_idx=jnp.full((n_points,), -1, dtype=jnp.int32),
        static_manifold_edge_idx=jnp.full((n_points,), -1, dtype=jnp.int32),
        static_manifold_active=jnp.zeros((n_points,), dtype=jnp.bool_),
        static_manifold_no_contact_count=jnp.zeros((n_points,), dtype=jnp.int32),
        static_manifold_normal_force=jnp.zeros((n_points,), dtype=jnp.float32),
        static_manifold_normal=jnp.zeros((n_points, 2), dtype=jnp.float32),
    )

    topology = SpringTopology(
        a_idx=jnp.array(dense_data["spring_a_idx"], dtype=jnp.int32),
        b_idx=jnp.array(dense_data["spring_b_idx"], dtype=jnp.int32),
    )

    actuator_info = ActuatorInfo(
        cell_spring_indices=jnp.array(
            dense_data["actuator_cell_spring_indices"], dtype=jnp.int32
        ),
        spring_act_count=jnp.array(
            dense_data["actuator_spring_act_count"], dtype=jnp.float32
        ),
        actuated_spring_mask=jnp.array(
            dense_data["actuated_spring_mask"], dtype=jnp.bool_
        ),
    )

    return state, topology, actuator_info


# ---------------------------------------------------------------------------
# Connectivity helpers (from original evogym/utils.py)
# ---------------------------------------------------------------------------

def get_full_connectivity(structure: np.ndarray) -> np.ndarray:
    """Compute default connections from adjacency.

    Matches evogym.get_full_connectivity: returns (2, k) array of pairwise
    connections into np.flatten(structure).
    """
    out = []
    h, w = structure.shape
    for i in range(structure.size):
        x = i % w
        y = i // w
        if structure[y, x] == 0:
            continue
        # Right neighbor
        nx, ny = x + 1, y
        if nx < w and structure[ny, nx] != 0:
            out.append([i, ny * w + nx])
        # Down neighbor
        nx, ny = x, y + 1
        if ny < h and structure[ny, nx] != 0:
            out.append([i, ny * w + nx])

    if not out:
        return np.empty((2, 0), dtype=int)
    return np.array(out, dtype=int).T


def is_connected(structure: np.ndarray) -> bool:
    """Check if all non-empty voxels form a single connected component."""
    h, w = structure.shape
    non_empty = []
    for y in range(h):
        for x in range(w):
            if structure[y, x] != 0:
                non_empty.append((x, y))

    if len(non_empty) == 0:
        return False
    if len(non_empty) == 1:
        return True

    visited = set()
    stack = [non_empty[0]]
    non_empty_set = set(non_empty)

    while stack:
        cx, cy = stack.pop()
        if (cx, cy) in visited:
            continue
        visited.add((cx, cy))
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in non_empty_set and (nx, ny) not in visited:
                stack.append((nx, ny))

    return len(visited) == len(non_empty)


def has_actuator(structure: np.ndarray) -> bool:
    """Check if structure contains at least one actuator-capable voxel."""
    return bool(
        np.any(
            (structure == C.H_ACT)
            | (structure == C.V_ACT)
            | (structure == C.CONTRACTILE)
        )
    )


def sample_robot(
    shape: Tuple[int, int],
    pd: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly generate a connected robot morphology with actuators.

    Args:
        shape: (height, width) of robot.
        pd: (5,) probability distribution over voxel types
            [empty, rigid, soft, h_act, v_act]. Default: 60% empty, 10% each.
        rng: numpy random generator. Default: np.random.default_rng().

    Returns:
        (structure, connections) tuple.
    """
    if rng is None:
        rng = np.random.default_rng()

    if pd is not None:
        pd = np.asarray(pd, dtype=np.float64)
        if pd.shape != (5,):
            raise ValueError(f"pd must have shape (5,), got {pd.shape}")
        if pd[C.H_ACT] + pd[C.V_ACT] == 0:
            raise ValueError("pd must allow actuator sampling")
    else:
        pd = np.array([0.6, 0.1, 0.1, 0.1, 0.1])

    # Normalize to proper probabilities
    pd = pd / pd.sum()

    for _ in range(10_000):
        robot = rng.choice(5, size=shape, p=pd).astype(int)
        if is_connected(robot) and has_actuator(robot):
            return robot, get_full_connectivity(robot)

    raise RuntimeError(
        f"Failed to sample connected robot with actuators in shape {shape}"
    )
