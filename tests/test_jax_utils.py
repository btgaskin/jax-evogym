"""Tests for jax_evogym.jax_utils — JAX-native construction parity tests.

Validates that all JAX-native construction functions produce outputs matching
the numpy reference path (utils.py, collision.py) for diverse morphologies.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_evogym import constants as C
from jax_evogym.collision import make_collision_data
from jax_evogym.jax_utils import (
    jax_build_actuator_info,
    jax_build_collision_data,
    jax_build_deformation_info,
    jax_build_per_axis_actuator_info,
    jax_build_sim_state,
    jax_compose_grid,
    precompute_grid,
)
from jax_evogym.spring_scaling import contractile_diag_ratio
from jax_evogym.utils import (
    _build_deformation_info,
    create_dense_point_grid,
    create_dense_spring_indices,
    make_sim_state,
    sample_robot,
    voxel_to_dense_mass_spring_from_grid,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _random_composited_grid(H, W, rng, robot_h=3, robot_w=3, spawn_y=0, spawn_x=0):
    """Create a composited grid with random robot + optional terrain.

    Returns (grid, robot_cell_mask) as numpy arrays.
    """
    # Terrain: FIXED cells along bottom row
    grid = np.zeros((H, W), dtype=int)
    grid[0, :] = C.FIXED  # bottom row terrain

    # Random robot morphology
    robot = rng.choice([C.EMPTY, C.RIGID, C.SOFT, C.H_ACT, C.V_ACT], size=(robot_h, robot_w))
    # Ensure at least one non-empty cell
    if np.all(robot == C.EMPTY):
        robot[0, 0] = C.RIGID

    # Flip to y-up and place
    robot_flipped = np.flipud(robot)
    grid[spawn_y:spawn_y + robot_h, spawn_x:spawn_x + robot_w] = np.where(
        robot_flipped != C.EMPTY, robot_flipped, grid[spawn_y:spawn_y + robot_h, spawn_x:spawn_x + robot_w]
    )

    robot_cell_mask = np.zeros((H, W), dtype=bool)
    for vy in range(robot_h):
        for vx in range(robot_w):
            if robot_flipped[vy, vx] != C.EMPTY:
                robot_cell_mask[spawn_y + vy, spawn_x + vx] = True

    return grid, robot_cell_mask


def _random_robot_only_grid(H, W, rng, robot_h=3, robot_w=3):
    """Create a robot-only grid (no terrain)."""
    grid = np.zeros((H, W), dtype=int)
    robot = rng.choice([C.EMPTY, C.RIGID, C.SOFT, C.H_ACT, C.V_ACT], size=(robot_h, robot_w))
    if np.all(robot == C.EMPTY):
        robot[0, 0] = C.RIGID

    robot_flipped = np.flipud(robot)
    grid[:robot_h, :robot_w] = robot_flipped

    # For robot-only: all non-empty cells are robot
    robot_cell_mask = grid != C.EMPTY
    return grid, robot_cell_mask


def _numpy_reference(grid, H, W, robot_cell_mask):
    """Build all data via numpy path for comparison."""
    dense_data = voxel_to_dense_mass_spring_from_grid(grid, H, W)
    state, topo, act_info = make_sim_state(dense_data, spawn_x=0, spawn_y=0)
    cd = make_collision_data(grid, H, W, robot_cell_mask)

    # Deformation info
    robot_cell_positions = set()
    for vy in range(H):
        for vx in range(W):
            if robot_cell_mask[vy, vx]:
                robot_cell_positions.add((vx, vy))

    deform_info = _build_deformation_info(
        dense_data["_springs"], grid, H, W,
        robot_cell_positions=robot_cell_positions,
    )

    return state, topo, act_info, cd, deform_info, dense_data


SEEDS = list(range(10))
GRID_H, GRID_W = 5, 5


# ---------------------------------------------------------------------------
# GridData validation
# ---------------------------------------------------------------------------

class TestPrecomputeGrid:

    def test_positions_match(self):
        """GridData positions match create_dense_point_grid."""
        H, W = 5, 5
        gd = precompute_grid(H, W)
        np_pos = create_dense_point_grid(H, W)
        np.testing.assert_allclose(
            np.array(gd.positions), np_pos.astype(np.float32), atol=1e-7
        )

    def test_spring_indices_match(self):
        """GridData spring_a_idx/spring_b_idx match create_dense_spring_indices."""
        H, W = 5, 5
        gd = precompute_grid(H, W)
        np_springs = create_dense_spring_indices(H, W)
        np_a = np.array([s[0] for s in np_springs], dtype=np.int32)
        np_b = np.array([s[1] for s in np_springs], dtype=np.int32)
        np.testing.assert_array_equal(np.array(gd.spring_a_idx), np_a)
        np.testing.assert_array_equal(np.array(gd.spring_b_idx), np_b)

    def test_spring_types(self):
        """Spring types: horiz=0, vert=1, diag=2."""
        H, W = 3, 4
        gd = precompute_grid(H, W)
        n_horiz = (H + 1) * W
        n_vert = H * (W + 1)
        n_diag = 2 * H * W
        types = np.array(gd.spring_types)
        assert np.all(types[:n_horiz] == 0)
        assert np.all(types[n_horiz:n_horiz + n_vert] == 1)
        assert np.all(types[n_horiz + n_vert:] == 2)

    def test_rest_lengths(self):
        """Rest lengths are CELL_SIZE for main, CELL_SIZE*sqrt(2) for diag."""
        H, W = 3, 3
        gd = precompute_grid(H, W)
        n_horiz = (H + 1) * W
        n_vert = H * (W + 1)
        rl = np.array(gd.spring_rest_length)

        np.testing.assert_allclose(rl[:n_horiz], C.CELL_SIZE, atol=1e-6)
        np.testing.assert_allclose(rl[n_horiz:n_horiz + n_vert], C.CELL_SIZE, atol=1e-6)
        diag_len = C.CELL_SIZE * np.sqrt(2)
        np.testing.assert_allclose(rl[n_horiz + n_vert:], diag_len, atol=1e-6)

    def test_corner_indices(self):
        """Cell corner indices: [BL, BR, TR, TL] match expected."""
        H, W = 2, 3
        gd = precompute_grid(H, W)
        W1 = W + 1
        for vy in range(H):
            for vx in range(W):
                idx = vy * W + vx
                bl = vy * W1 + vx
                br = vy * W1 + (vx + 1)
                tr = (vy + 1) * W1 + (vx + 1)
                tl = (vy + 1) * W1 + vx
                expected = [bl, br, tr, tl]
                actual = np.array(gd.cell_corner_indices[idx])
                np.testing.assert_array_equal(actual, expected,
                    err_msg=f"Cell ({vy},{vx})")

    def test_cell_spring_indices(self):
        """Per-cell spring indices match manual calculation."""
        H, W = 3, 4
        gd = precompute_grid(H, W)
        vert_offset = (H + 1) * W
        diag_offset = vert_offset + H * (W + 1)

        for vy in range(H):
            for vx in range(W):
                cell = vy * W + vx
                assert int(gd.cell_horiz_bottom[cell]) == vy * W + vx
                assert int(gd.cell_horiz_top[cell]) == (vy + 1) * W + vx
                assert int(gd.cell_vert_left[cell]) == vert_offset + vy * (W + 1) + vx
                assert int(gd.cell_vert_right[cell]) == vert_offset + vy * (W + 1) + (vx + 1)
                assert int(gd.cell_diag1[cell]) == diag_offset + 2 * (vy * W + vx)
                assert int(gd.cell_diag2[cell]) == diag_offset + 2 * (vy * W + vx) + 1

    def test_spring_cell_adjacency_boundary(self):
        """Boundary springs have cell_a = the only cell, cell_b = -1."""
        H, W = 3, 3
        gd = precompute_grid(H, W)

        # Bottom horizontal spring (grid_y=0): only cell above (vy=0)
        s_idx = 0 * W + 0  # grid_y=0, grid_x=0
        assert int(gd.spring_cell_a[s_idx]) == 0 * W + 0
        assert int(gd.spring_cell_b[s_idx]) == -1

        # Top horizontal spring (grid_y=H): only cell below (vy=H-1)
        s_idx = H * W + 0  # grid_y=H, grid_x=0
        assert int(gd.spring_cell_a[s_idx]) == (H - 1) * W + 0
        assert int(gd.spring_cell_b[s_idx]) == -1

    def test_spring_cell_adjacency_interior(self):
        """Interior springs have both cell_a (later) and cell_b (earlier)."""
        H, W = 3, 3
        gd = precompute_grid(H, W)

        # Interior horizontal spring (grid_y=1, grid_x=1)
        s_idx = 1 * W + 1  # horiz section
        cell_a = int(gd.spring_cell_a[s_idx])
        cell_b = int(gd.spring_cell_b[s_idx])
        assert cell_a == 1 * W + 1  # vy_above=1 (later)
        assert cell_b == 0 * W + 1  # vy_below=0 (earlier)

    def test_n_springs(self):
        """Total spring count matches formula."""
        H, W = 5, 7
        gd = precompute_grid(H, W)
        expected = (H + 1) * W + H * (W + 1) + 2 * H * W
        assert gd.spring_a_idx.shape[0] == expected


# ---------------------------------------------------------------------------
# SimState parity
# ---------------------------------------------------------------------------

class TestSimStateParity:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_composited_grid(self, seed):
        """SimState from composited grid matches numpy path."""
        rng = np.random.default_rng(seed)
        grid, mask = _random_composited_grid(GRID_H, GRID_W, rng)
        gd = precompute_grid(GRID_H, GRID_W)

        np_state, np_topo, np_act, _, _, _ = _numpy_reference(grid, GRID_H, GRID_W, mask)
        jax_state, jax_topo = jax_build_sim_state(jnp.array(grid, dtype=jnp.int32), gd)

        # Topology
        np.testing.assert_array_equal(np.array(jax_topo.a_idx), np.array(np_topo.a_idx))
        np.testing.assert_array_equal(np.array(jax_topo.b_idx), np.array(np_topo.b_idx))

        # Bool/mask fields — exact match
        np.testing.assert_array_equal(
            np.array(jax_state.point_mask), np.array(np_state.point_mask),
            err_msg="point_mask mismatch"
        )
        np.testing.assert_array_equal(
            np.array(jax_state.spring_mask), np.array(np_state.spring_mask),
            err_msg="spring_mask mismatch"
        )
        np.testing.assert_array_equal(
            np.array(jax_state.rigid_spring_mask), np.array(np_state.rigid_spring_mask),
            err_msg="rigid_spring_mask mismatch"
        )
        np.testing.assert_array_equal(
            np.array(jax_state.fixed), np.array(np_state.fixed),
            err_msg="fixed mismatch"
        )

        # Float fields — 1e-6 tolerance
        np.testing.assert_allclose(
            np.array(jax_state.positions), np.array(np_state.positions), atol=1e-6,
            err_msg="positions mismatch"
        )
        np.testing.assert_allclose(
            np.array(jax_state.masses), np.array(np_state.masses), atol=1e-6,
            err_msg="masses mismatch"
        )
        np.testing.assert_allclose(
            np.array(jax_state.spring_const), np.array(np_state.spring_const), atol=1e-1,
            err_msg="spring_const mismatch"
        )
        np.testing.assert_allclose(
            np.array(jax_state.spring_rest_length), np.array(np_state.spring_rest_length),
            atol=1e-6, err_msg="spring_rest_length mismatch"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_robot_only_grid(self, seed):
        """SimState parity for robot-only grids (no terrain)."""
        rng = np.random.default_rng(seed)
        grid, mask = _random_robot_only_grid(GRID_H, GRID_W, rng)
        gd = precompute_grid(GRID_H, GRID_W)

        np_state, _, _, _, _, _ = _numpy_reference(grid, GRID_H, GRID_W, mask)
        jax_state, _ = jax_build_sim_state(jnp.array(grid, dtype=jnp.int32), gd)

        np.testing.assert_array_equal(
            np.array(jax_state.point_mask), np.array(np_state.point_mask)
        )
        np.testing.assert_array_equal(
            np.array(jax_state.spring_mask), np.array(np_state.spring_mask)
        )
        np.testing.assert_array_equal(
            np.array(jax_state.fixed), np.array(np_state.fixed)
        )
        np.testing.assert_allclose(
            np.array(jax_state.spring_const), np.array(np_state.spring_const), atol=1e-1
        )

    def test_spawn_offset(self):
        """Spawn offset correctly shifts active point positions."""
        grid = np.zeros((5, 5), dtype=int)
        grid[0, 0] = C.RIGID
        gd = precompute_grid(5, 5)

        s0, _ = jax_build_sim_state(jnp.array(grid, dtype=jnp.int32), gd, spawn_x=0, spawn_y=0)
        s1, _ = jax_build_sim_state(jnp.array(grid, dtype=jnp.int32), gd, spawn_x=3, spawn_y=2)

        # Active points should differ by offset
        mask = np.array(s0.point_mask)
        offset = np.array([3 * C.CELL_SIZE, 2 * C.CELL_SIZE])
        pos0 = np.array(s0.positions)[mask]
        pos1 = np.array(s1.positions)[mask]
        # Each active point should be shifted by the offset
        for i in range(pos0.shape[0]):
            np.testing.assert_allclose(pos1[i] - pos0[i], offset, atol=1e-6)

    def test_contractile_gets_actuator_stiffness(self):
        """CONTRACTILE voxel springs should use actuator main/diag constants."""
        gd = precompute_grid(1, 1)
        grid = jnp.array([[C.CONTRACTILE]], dtype=jnp.int32)
        state, _ = jax_build_sim_state(grid, gd)

        spring_const = np.asarray(state.spring_const)
        spring_mask = np.asarray(state.spring_mask)
        active_const = spring_const[spring_mask]
        # 1x1 cell -> 6 active springs: 4 main + 2 diagonal.
        assert active_const.shape[0] == 6
        np.testing.assert_allclose(active_const[:4], C.ACTUATOR_MAIN_K, atol=1e-3)
        np.testing.assert_allclose(active_const[4:], C.ACTUATOR_DIAG_K, atol=1e-3)

    def test_contractile_stiffness_scale_applies_to_contractile(self):
        gd = precompute_grid(1, 1)
        grid = jnp.array([[C.CONTRACTILE]], dtype=jnp.int32)
        k = 0.5
        state, _ = jax_build_sim_state(grid, gd, spring_stiffness_scale=k)
        active_const = np.asarray(state.spring_const)[np.asarray(state.spring_mask)]
        expected_main = C.ACTUATOR_MAIN_K * k
        expected_diag = expected_main * contractile_diag_ratio(k)
        np.testing.assert_allclose(active_const[:4], expected_main, atol=1e-3)
        np.testing.assert_allclose(active_const[4:], expected_diag, atol=1e-3)

    def test_contractile_stiffness_scale_hits_soft_anchor(self):
        gd = precompute_grid(1, 1)
        grid = jnp.array([[C.CONTRACTILE]], dtype=jnp.int32)
        k = C.SOFT_MAIN_K / C.ACTUATOR_MAIN_K
        state, _ = jax_build_sim_state(grid, gd, spring_stiffness_scale=k)
        active_const = np.asarray(state.spring_const)[np.asarray(state.spring_mask)]
        np.testing.assert_allclose(active_const[:4], C.SOFT_MAIN_K, atol=1e-3)
        np.testing.assert_allclose(active_const[4:], C.SOFT_DIAG_K, atol=1e-3)

    def test_contractile_stiffness_scale_above_one_keeps_actuator_ratio(self):
        gd = precompute_grid(1, 1)
        grid = jnp.array([[C.CONTRACTILE]], dtype=jnp.int32)
        k = 1.5
        state, _ = jax_build_sim_state(grid, gd, spring_stiffness_scale=k)
        active_const = np.asarray(state.spring_const)[np.asarray(state.spring_mask)]
        expected_main = C.ACTUATOR_MAIN_K * k
        expected_diag = expected_main * (C.ACTUATOR_DIAG_K / C.ACTUATOR_MAIN_K)
        np.testing.assert_allclose(active_const[:4], expected_main, atol=1e-3)
        np.testing.assert_allclose(active_const[4:], expected_diag, atol=1e-3)

    def test_contractile_stiffness_scale_does_not_change_h_act(self):
        gd = precompute_grid(1, 1)
        grid = jnp.array([[C.H_ACT]], dtype=jnp.int32)
        state, _ = jax_build_sim_state(grid, gd, spring_stiffness_scale=0.5)
        active_const = np.asarray(state.spring_const)[np.asarray(state.spring_mask)]
        np.testing.assert_allclose(active_const[:4], C.ACTUATOR_MAIN_K, atol=1e-3)
        np.testing.assert_allclose(active_const[4:], C.ACTUATOR_DIAG_K, atol=1e-3)


# ---------------------------------------------------------------------------
# ActuatorInfo parity
# ---------------------------------------------------------------------------

class TestActuatorInfoParity:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_parity(self, seed):
        """Padded JAX ActuatorInfo matches numpy for spring_act_count/actuated_mask."""
        rng = np.random.default_rng(seed)
        grid, mask = _random_composited_grid(GRID_H, GRID_W, rng)
        gd = precompute_grid(GRID_H, GRID_W)

        _, _, np_act, _, _, _ = _numpy_reference(grid, GRID_H, GRID_W, mask)
        jax_act, act_mask = jax_build_actuator_info(jnp.array(grid, dtype=jnp.int32), gd)

        # spring_act_count and actuated_spring_mask: exact match (same n_springs shape)
        np.testing.assert_allclose(
            np.array(jax_act.spring_act_count),
            np.array(np_act.spring_act_count),
            atol=1e-6,
            err_msg="spring_act_count mismatch"
        )
        np.testing.assert_array_equal(
            np.array(jax_act.actuated_spring_mask),
            np.array(np_act.actuated_spring_mask),
            err_msg="actuated_spring_mask mismatch"
        )

        # Cell spring indices: filter by act_mask and compare
        jax_indices = np.array(jax_act.cell_spring_indices)
        act_mask_np = np.array(act_mask)
        jax_filtered = jax_indices[act_mask_np]
        np_indices = np.array(np_act.cell_spring_indices)

        if np_indices.shape[0] > 0:
            np.testing.assert_array_equal(jax_filtered, np_indices,
                err_msg="cell_spring_indices mismatch")


class TestPerAxisActuatorInfo:
    def test_contractile_hv_mapping(self):
        """Per-axis builder maps CONTRACTILE to both axes and H/V_ACT to single axes."""
        gd = precompute_grid(2, 2)
        # Row-major flattened cells: [H_ACT, V_ACT, CONTRACTILE, SOFT]
        morph = jnp.array([[C.H_ACT, C.V_ACT], [C.CONTRACTILE, C.SOFT]], dtype=jnp.int32)
        info, actuator_mask = jax_build_per_axis_actuator_info(morph, gd)

        mask_np = np.asarray(actuator_mask)
        np.testing.assert_array_equal(mask_np, np.array([True, True, True, False]))

        h_pairs = np.asarray(info.h_spring_pairs)
        v_pairs = np.asarray(info.v_spring_pairs)

        # H_ACT cell: horizontal pair populated, vertical pair zeroed.
        np.testing.assert_array_equal(h_pairs[0], np.array([gd.cell_horiz_bottom[0], gd.cell_horiz_top[0]]))
        np.testing.assert_array_equal(v_pairs[0], np.array([0, 0]))

        # V_ACT cell: vertical pair populated, horizontal pair zeroed.
        np.testing.assert_array_equal(h_pairs[1], np.array([0, 0]))
        np.testing.assert_array_equal(v_pairs[1], np.array([gd.cell_vert_left[1], gd.cell_vert_right[1]]))

        # CONTRACTILE cell: both pairs populated.
        np.testing.assert_array_equal(h_pairs[2], np.array([gd.cell_horiz_bottom[2], gd.cell_horiz_top[2]]))
        np.testing.assert_array_equal(v_pairs[2], np.array([gd.cell_vert_left[2], gd.cell_vert_right[2]]))

        assert int(np.asarray(info.h_compact_count)) == 2
        assert int(np.asarray(info.v_compact_count)) == 2
        np.testing.assert_array_equal(
            np.asarray(info.h_compact_cell_indices)[:2],
            np.array([0, 2], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(info.v_compact_cell_indices)[:2],
            np.array([1, 2], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(info.h_compact_spring_pairs)[:2],
            h_pairs[np.array([0, 2], dtype=np.int32)],
        )
        np.testing.assert_array_equal(
            np.asarray(info.v_compact_spring_pairs)[:2],
            v_pairs[np.array([1, 2], dtype=np.int32)],
        )

    def test_jit_per_axis_builder(self):
        """jax_build_per_axis_actuator_info compiles under jit with static GridData."""
        gd = precompute_grid(3, 3)
        morph = jnp.zeros((3, 3), dtype=jnp.int32).at[1, 1].set(C.CONTRACTILE)

        @jax.jit
        def build(m):
            return jax_build_per_axis_actuator_info(m, gd)

        info, mask = build(morph)
        assert info.spring_act_count.shape[0] == gd.spring_a_idx.shape[0]
        assert mask.shape[0] == 9
        assert info.h_compact_cell_indices.shape[0] == 9
        assert info.v_compact_cell_indices.shape[0] == 9


# ---------------------------------------------------------------------------
# DeformationInfo parity
# ---------------------------------------------------------------------------

class TestDeformationInfoParity:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_parity(self, seed):
        """Padded JAX DeformationInfo matches numpy (filtered by mask)."""
        rng = np.random.default_rng(seed)
        grid, mask = _random_composited_grid(GRID_H, GRID_W, rng)
        gd = precompute_grid(GRID_H, GRID_W)

        _, _, _, _, np_deform, _ = _numpy_reference(grid, GRID_H, GRID_W, mask)
        jax_deform, deform_mask = jax_build_deformation_info(
            jnp.array(grid, dtype=jnp.int32), gd, jnp.array(mask)
        )

        dm = np.array(deform_mask)
        jax_horiz = np.array(jax_deform.voxel_horiz_springs)[dm]
        jax_vert = np.array(jax_deform.voxel_vert_springs)[dm]
        np_horiz = np.array(np_deform.voxel_horiz_springs)
        np_vert = np.array(np_deform.voxel_vert_springs)

        if np_horiz.shape[0] > 0:
            np.testing.assert_array_equal(jax_horiz, np_horiz,
                err_msg="voxel_horiz_springs mismatch")
            np.testing.assert_array_equal(jax_vert, np_vert,
                err_msg="voxel_vert_springs mismatch")

        # n_robot_voxels
        assert int(jax_deform.n_robot_voxels) == int(np_deform.n_robot_voxels)


# ---------------------------------------------------------------------------
# CollisionData parity
# ---------------------------------------------------------------------------

class TestCollisionDataParity:

    def test_custom_max_triples_parity(self):
        """JAX collision data with custom max_triples matches numpy."""
        rng = np.random.default_rng(42)
        grid, mask = _random_composited_grid(GRID_H, GRID_W, rng)
        gd = precompute_grid(GRID_H, GRID_W)

        # Build with default buffers to find real count
        np_cd_default = make_collision_data(grid, GRID_H, GRID_W, mask)
        n_real = np_cd_default.n_triples
        n_real_self = np_cd_default.n_self_triples

        # Build with tight buffers
        tight = max(64, n_real * 4)
        tight_self = max(64, n_real_self * 4)
        np_cd_tight = make_collision_data(grid, GRID_H, GRID_W, mask,
                                          max_triples=tight, max_self_triples=tight_self)
        jax_cd_tight = jax_build_collision_data(
            jnp.array(grid, dtype=jnp.int32), jnp.array(mask), gd,
            max_triples=tight, max_self_triples=tight_self,
        )

        # Buffer sizes should be tight
        assert np_cd_tight.triple_i.shape[0] == tight
        assert jax_cd_tight.triple_i.shape[0] == tight

        # Active triple counts should match
        np_active = int(np.sum(np.array(np_cd_tight.triple_active)))
        jax_active = int(np.sum(np.array(jax_cd_tight.triple_active)))
        assert np_active == n_real
        assert jax_active == n_real

    @pytest.mark.parametrize("seed", SEEDS)
    def test_sparse_active_path_parity(self, seed):
        """Sparse active-boxel path matches numpy triple reconstruction."""
        rng = np.random.default_rng(seed)
        grid, mask = _random_composited_grid(GRID_H, GRID_W, rng)
        gd = precompute_grid(GRID_H, GRID_W)

        np_cd = make_collision_data(grid, GRID_H, GRID_W, mask)
        # Active boxels include both robot and terrain voxels (non-EMPTY grid cells).
        n_active = int(np.sum(grid != C.EMPTY))

        # Ensure the sparse cap can hold all active boxels for strict parity.
        jax_cd = jax_build_collision_data(
            jnp.array(grid, dtype=jnp.int32),
            jnp.array(mask),
            gd,
            max_active_boxels=max(1, n_active),
        )

        n_boxels = GRID_H * GRID_W
        jax_ti = np.array(jax_cd.triple_i)
        jax_tj = np.array(jax_cd.triple_j)
        jax_tk = np.array(jax_cd.triple_k)
        jax_active = np.array(jax_cd.triple_active)
        np_ti = np.array(np_cd.triple_i)
        np_tj = np.array(np_cd.triple_j)
        np_tk = np.array(np_cd.triple_k)
        np_active = np.array(np_cd.triple_active)

        jax_reconstructed = np.zeros((n_boxels, n_boxels, 4), dtype=bool)
        for idx in range(len(jax_ti)):
            if jax_active[idx]:
                jax_reconstructed[jax_ti[idx], jax_tj[idx], jax_tk[idx]] = True

        np_reconstructed = np.zeros((n_boxels, n_boxels, 4), dtype=bool)
        for idx in range(len(np_ti)):
            if np_active[idx]:
                np_reconstructed[np_ti[idx], np_tj[idx], np_tk[idx]] = True

        np.testing.assert_array_equal(
            jax_reconstructed,
            np_reconstructed,
            err_msg="Sparse active triple reconstruction mismatch",
        )
        assert int(np.sum(np.array(jax_cd.triple_active))) == int(np.sum(np_active))
        assert int(np.array(jax_cd.n_triples)) == int(np_cd.n_triples)
        assert int(np.array(jax_cd.n_self_triples)) == int(np_cd.n_self_triples)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_parity(self, seed):
        """JAX CollisionData matches numpy for all masks."""
        rng = np.random.default_rng(seed)
        grid, mask = _random_composited_grid(GRID_H, GRID_W, rng)
        gd = precompute_grid(GRID_H, GRID_W)

        np_cd = make_collision_data(grid, GRID_H, GRID_W, mask)
        jax_cd = jax_build_collision_data(
            jnp.array(grid, dtype=jnp.int32), jnp.array(mask), gd
        )

        np.testing.assert_array_equal(
            np.array(jax_cd.boxel_corners), np.array(np_cd.boxel_corners),
            err_msg="boxel_corners mismatch"
        )
        np.testing.assert_array_equal(
            np.array(jax_cd.boxel_mask), np.array(np_cd.boxel_mask),
            err_msg="boxel_mask mismatch"
        )
        np.testing.assert_array_equal(
            np.array(jax_cd.surface_edge_mask), np.array(np_cd.surface_edge_mask),
            err_msg="surface_edge_mask mismatch"
        )
        np.testing.assert_array_equal(
            np.array(jax_cd.boxel_edge_a), np.array(np_cd.boxel_edge_a),
            err_msg="boxel_edge_a mismatch"
        )
        np.testing.assert_array_equal(
            np.array(jax_cd.boxel_edge_b), np.array(np_cd.boxel_edge_b),
            err_msg="boxel_edge_b mismatch"
        )
        assert int(jax_cd.n_real_points) == int(np_cd.n_real_points)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_indexed_field_parity(self, seed):
        """JAX indexed triple fields match numpy indexed fields."""
        rng = np.random.default_rng(seed)
        grid, mask = _random_composited_grid(GRID_H, GRID_W, rng)
        gd = precompute_grid(GRID_H, GRID_W)

        np_cd = make_collision_data(grid, GRID_H, GRID_W, mask)
        jax_cd = jax_build_collision_data(
            jnp.array(grid, dtype=jnp.int32), jnp.array(mask), gd
        )

        n_boxels = GRID_H * GRID_W

        # Reconstruct dense mask from indexed fields (JAX)
        jax_ti = np.array(jax_cd.triple_i)
        jax_tj = np.array(jax_cd.triple_j)
        jax_tk = np.array(jax_cd.triple_k)
        jax_active = np.array(jax_cd.triple_active)

        jax_reconstructed = np.zeros((n_boxels, n_boxels, 4), dtype=bool)
        for idx in range(len(jax_ti)):
            if jax_active[idx]:
                jax_reconstructed[jax_ti[idx], jax_tj[idx], jax_tk[idx]] = True

        # Reconstruct dense mask from indexed fields (numpy)
        np_ti = np.array(np_cd.triple_i)
        np_tj = np.array(np_cd.triple_j)
        np_tk = np.array(np_cd.triple_k)
        np_active = np.array(np_cd.triple_active)

        np_reconstructed = np.zeros((n_boxels, n_boxels, 4), dtype=bool)
        for idx in range(len(np_ti)):
            if np_active[idx]:
                np_reconstructed[np_ti[idx], np_tj[idx], np_tk[idx]] = True

        np.testing.assert_array_equal(
            jax_reconstructed,
            np_reconstructed,
            err_msg="Reconstructed triple_check_mask mismatch"
        )

        # Same for self-collision
        jax_si = np.array(jax_cd.self_triple_i)
        jax_sj = np.array(jax_cd.self_triple_j)
        jax_sk = np.array(jax_cd.self_triple_k)
        jax_s_active = np.array(jax_cd.self_triple_active)

        jax_reconstructed_self = np.zeros((n_boxels, n_boxels, 4), dtype=bool)
        for idx in range(len(jax_si)):
            if jax_s_active[idx]:
                jax_reconstructed_self[jax_si[idx], jax_sj[idx], jax_sk[idx]] = True

        np_si = np.array(np_cd.self_triple_i)
        np_sj = np.array(np_cd.self_triple_j)
        np_sk = np.array(np_cd.self_triple_k)
        np_s_active = np.array(np_cd.self_triple_active)

        np_reconstructed_self = np.zeros((n_boxels, n_boxels, 4), dtype=bool)
        for idx in range(len(np_si)):
            if np_s_active[idx]:
                np_reconstructed_self[np_si[idx], np_sj[idx], np_sk[idx]] = True

        np.testing.assert_array_equal(
            jax_reconstructed_self,
            np_reconstructed_self,
            err_msg="Reconstructed self_collision_mask mismatch"
        )


# ---------------------------------------------------------------------------
# jax_compose_grid
# ---------------------------------------------------------------------------

class TestJaxComposeGrid:

    def test_basic(self):
        """Robot cells overwrite terrain."""
        terrain = jnp.full((5, 5), C.FIXED, dtype=jnp.int32)
        robot = jnp.array([[C.RIGID, C.SOFT], [C.H_ACT, C.EMPTY]], dtype=jnp.int32)
        comp, mask = jax_compose_grid(terrain, robot, spawn_y=1, spawn_x=2)

        assert int(comp[1, 2]) == C.RIGID
        assert int(comp[1, 3]) == C.SOFT
        assert int(comp[2, 2]) == C.H_ACT
        assert int(comp[2, 3]) == C.FIXED  # EMPTY robot → terrain shows through
        assert bool(mask[1, 2]) is True
        assert bool(mask[2, 3]) is False  # EMPTY robot cell

    def test_empty_robot(self):
        """All-EMPTY robot changes nothing."""
        terrain = jnp.full((3, 3), C.FIXED, dtype=jnp.int32)
        robot = jnp.zeros((2, 2), dtype=jnp.int32)
        comp, mask = jax_compose_grid(terrain, robot, spawn_y=0, spawn_x=0)

        np.testing.assert_array_equal(np.array(comp), np.array(terrain))
        assert not np.any(np.array(mask))


# ---------------------------------------------------------------------------
# vmap + JIT tests
# ---------------------------------------------------------------------------

class TestVmapAndJIT:

    def test_jit_sim_state(self):
        """jax_build_sim_state compiles with jax.jit (GridData via closure)."""
        gd = precompute_grid(5, 5)
        grid = jnp.zeros((5, 5), dtype=jnp.int32).at[0, 0].set(C.RIGID)

        @jax.jit
        def build(morph):
            return jax_build_sim_state(morph, gd)

        state, topo = build(grid)
        assert state.positions.shape == ((5 + 1) * (5 + 1), 2)

    def test_jit_actuator_info(self):
        """jax_build_actuator_info compiles with jax.jit."""
        gd = precompute_grid(5, 5)
        grid = jnp.zeros((5, 5), dtype=jnp.int32).at[0, 0].set(C.H_ACT)

        @jax.jit
        def build(morph):
            return jax_build_actuator_info(morph, gd)

        act, mask = build(grid)
        assert act.spring_act_count.shape[0] > 0

    def test_jit_collision_data(self):
        """jax_build_collision_data compiles with jax.jit."""
        gd = precompute_grid(5, 5)
        grid = jnp.zeros((5, 5), dtype=jnp.int32).at[0, 0].set(C.RIGID)
        rmask = jnp.zeros((5, 5), dtype=jnp.bool_).at[0, 0].set(True)

        @jax.jit
        def build(morph, rmask):
            return jax_build_collision_data(morph, rmask, gd)

        cd = build(grid, rmask)
        assert cd.boxel_mask.shape == (25,)

    def test_vmap_sim_state(self):
        """vmap over batch of morphologies produces correct results."""
        H, W = 5, 5
        gd = precompute_grid(H, W)
        batch_size = 8
        rng = np.random.default_rng(42)

        grids = []
        for _ in range(batch_size):
            grid, _ = _random_robot_only_grid(H, W, rng, robot_h=3, robot_w=3)
            grids.append(grid)
        batch_grids = jnp.array(np.stack(grids), dtype=jnp.int32)

        # vmap over morphology (grid_data is static, captured in closure)
        def build_fn(morph):
            return jax_build_sim_state(morph, gd)

        batch_state, batch_topo = jax.vmap(build_fn)(batch_grids)

        # Verify each element matches unbatched
        for i in range(batch_size):
            single_state, single_topo = jax_build_sim_state(batch_grids[i], gd)
            np.testing.assert_allclose(
                np.array(batch_state.spring_const[i]),
                np.array(single_state.spring_const),
                atol=1e-1,
                err_msg=f"spring_const mismatch at batch index {i}",
            )
            np.testing.assert_array_equal(
                np.array(batch_state.spring_mask[i]),
                np.array(single_state.spring_mask),
                err_msg=f"spring_mask mismatch at batch index {i}",
            )
            np.testing.assert_array_equal(
                np.array(batch_state.point_mask[i]),
                np.array(single_state.point_mask),
                err_msg=f"point_mask mismatch at batch index {i}",
            )

    def test_vmap_actuator_info(self):
        """vmap over batch of morphologies for actuator info."""
        H, W = 5, 5
        gd = precompute_grid(H, W)
        rng = np.random.default_rng(123)

        grids = []
        for _ in range(4):
            grid, _ = _random_robot_only_grid(H, W, rng, robot_h=3, robot_w=3)
            grids.append(grid)
        batch_grids = jnp.array(np.stack(grids), dtype=jnp.int32)

        def build_fn(morph):
            return jax_build_actuator_info(morph, gd)

        batch_act, batch_mask = jax.vmap(build_fn)(batch_grids)
        assert batch_act.spring_act_count.shape[0] == 4

    def test_make_jaxpr(self):
        """All functions produce valid JAXPRs (compilable).

        GridData captured via closure (not a traced argument) since it
        contains Python ints (H, W) used for static shapes.
        """
        H, W = 3, 3
        gd = precompute_grid(H, W)
        morph = jnp.zeros((H, W), dtype=jnp.int32)
        rmask = jnp.zeros((H, W), dtype=jnp.bool_)

        # Should not raise
        jax.make_jaxpr(lambda m: jax_build_sim_state(m, gd))(morph)
        jax.make_jaxpr(lambda m: jax_build_actuator_info(m, gd))(morph)
        jax.make_jaxpr(lambda m, r: jax_build_deformation_info(m, gd, r))(morph, rmask)
        jax.make_jaxpr(lambda m, r: jax_build_collision_data(m, r, gd))(morph, rmask)

    def test_jit_collision_data_custom_max_triples(self):
        """jax_build_collision_data with custom max_triples compiles with jax.jit."""
        gd = precompute_grid(5, 5)
        grid = jnp.zeros((5, 5), dtype=jnp.int32).at[0, 0].set(C.RIGID)
        rmask = jnp.zeros((5, 5), dtype=jnp.bool_).at[0, 0].set(True)

        @jax.jit
        def build(morph, rmask):
            return jax_build_collision_data(morph, rmask, gd,
                                           max_triples=256, max_self_triples=64)

        cd = build(grid, rmask)
        assert cd.triple_i.shape[0] == 256
        assert cd.self_triple_i.shape[0] == 64


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_all_empty(self):
        """All-empty morphology: no active springs/points."""
        H, W = 5, 5
        gd = precompute_grid(H, W)
        grid = jnp.zeros((H, W), dtype=jnp.int32)

        state, _ = jax_build_sim_state(grid, gd)
        assert not np.any(np.array(state.point_mask))
        assert not np.any(np.array(state.spring_mask))

    def test_all_rigid(self):
        """All-RIGID morphology: no actuators, all springs RIGID."""
        H, W = 3, 3
        gd = precompute_grid(H, W)
        grid = jnp.full((H, W), C.RIGID, dtype=jnp.int32)

        state, _ = jax_build_sim_state(grid, gd)
        act, act_mask = jax_build_actuator_info(grid, gd)

        assert not np.any(np.array(act_mask))
        assert not np.any(np.array(act.actuated_spring_mask))

        # All active springs should be rigid
        mask = np.array(state.spring_mask)
        rigid = np.array(state.rigid_spring_mask)
        np.testing.assert_array_equal(mask, rigid)

    def test_single_cell(self):
        """Single non-empty cell at various positions."""
        H, W = 5, 5
        gd = precompute_grid(H, W)

        for vy in range(H):
            for vx in range(W):
                grid = np.zeros((H, W), dtype=int)
                grid[vy, vx] = C.SOFT
                mask = grid != C.EMPTY

                np_state, _, _, _, _, _ = _numpy_reference(grid, H, W, mask)
                jax_state, _ = jax_build_sim_state(jnp.array(grid, dtype=jnp.int32), gd)

                np.testing.assert_array_equal(
                    np.array(jax_state.point_mask), np.array(np_state.point_mask),
                    err_msg=f"Single cell at ({vy},{vx})"
                )
                np.testing.assert_allclose(
                    np.array(jax_state.spring_const), np.array(np_state.spring_const),
                    atol=1e-1, err_msg=f"spring_const at ({vy},{vx})"
                )

    def test_composited_shared_boundary(self):
        """Robot + terrain with shared boundary points."""
        H, W = 5, 5
        grid = np.zeros((H, W), dtype=int)
        grid[0, :] = C.FIXED  # bottom row = terrain
        grid[1, 1] = C.SOFT   # robot cell directly above terrain

        robot_cell_mask = np.zeros((H, W), dtype=bool)
        robot_cell_mask[1, 1] = True

        gd = precompute_grid(H, W)
        np_state, _, _, _, _, _ = _numpy_reference(grid, H, W, robot_cell_mask)
        jax_state, _ = jax_build_sim_state(jnp.array(grid, dtype=jnp.int32), gd)

        np.testing.assert_array_equal(
            np.array(jax_state.fixed), np.array(np_state.fixed),
            err_msg="fixed mismatch for shared boundary"
        )

    def test_all_fixed(self):
        """All-FIXED grid: all points fixed, no actuators."""
        H, W = 3, 3
        gd = precompute_grid(H, W)
        grid = jnp.full((H, W), C.FIXED, dtype=jnp.int32)

        state, _ = jax_build_sim_state(grid, gd)
        act, act_mask = jax_build_actuator_info(grid, gd)

        # All active points should be fixed
        mask = np.array(state.point_mask)
        fixed = np.array(state.fixed)
        for i in range(mask.shape[0]):
            if mask[i]:
                assert fixed[i, 0] and fixed[i, 1]

    def test_mixed_actuators(self):
        """Grid with both H_ACT and V_ACT cells."""
        H, W = 5, 5
        gd = precompute_grid(H, W)
        grid = np.zeros((H, W), dtype=int)
        grid[1, 1] = C.H_ACT
        grid[1, 2] = C.V_ACT
        grid[2, 1] = C.RIGID

        mask = grid != C.EMPTY
        np_state, _, np_act, _, _, _ = _numpy_reference(grid, H, W, mask)
        jax_act, act_mask = jax_build_actuator_info(jnp.array(grid, dtype=jnp.int32), gd)

        np.testing.assert_allclose(
            np.array(jax_act.spring_act_count),
            np.array(np_act.spring_act_count),
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Physics rollout parity
# ---------------------------------------------------------------------------

class TestPhysicsRolloutParity:

    def test_50_step_rollout(self):
        """50-step physics rollout matches numpy path within 1e-4."""
        from jax_evogym.sim import physics_substep
        from jax_evogym.actuators import set_actuator_goals, update_actuators
        from jax_evogym.types import default_physics_constants

        H, W = 5, 5
        rng = np.random.default_rng(42)
        grid, mask = _random_composited_grid(H, W, rng, robot_h=3, robot_w=3)

        # Ensure we have actuators
        if not np.any((grid == C.H_ACT) | (grid == C.V_ACT)):
            grid[1, 1] = C.H_ACT
            mask[1, 1] = True

        gd = precompute_grid(H, W)

        # Numpy path
        np_state, np_topo, np_act, np_cd, _, _ = _numpy_reference(grid, H, W, mask)
        constants = default_physics_constants()

        # JAX path
        jax_state, jax_topo = jax_build_sim_state(jnp.array(grid, dtype=jnp.int32), gd)
        jax_act, _ = jax_build_actuator_info(jnp.array(grid, dtype=jnp.int32), gd)
        jax_cd = jax_build_collision_data(
            jnp.array(grid, dtype=jnp.int32), jnp.array(mask), gd
        )

        # Verify starting states match
        np.testing.assert_allclose(
            np.array(jax_state.positions), np.array(np_state.positions), atol=1e-6,
            err_msg="Initial positions mismatch"
        )

        # Identify actuator cells in the flat grid (for mapping actions)
        grid_flat = grid.flatten()
        act_mask_np = (grid_flat == C.H_ACT) | (grid_flat == C.V_ACT)
        act_indices = np.where(act_mask_np)[0]
        n_act_np = np_act.cell_spring_indices.shape[0]
        n_act_jax = jax_act.cell_spring_indices.shape[0]  # H*W

        # Run 50 steps with same actions
        state_np = np_state
        state_jax = jax_state

        for step in range(50):
            # Actions for numpy path (n_actuators,)
            actions_np = jnp.array(
                rng.uniform(-1, 1, size=n_act_np), dtype=jnp.float32
            )

            # Actions for JAX path (H*W,) padded
            actions_jax = jnp.zeros(n_act_jax, dtype=jnp.float32)
            for i, idx in enumerate(act_indices):
                if i < len(actions_np):
                    actions_jax = actions_jax.at[idx].set(actions_np[i])

            # Set actuator goals (3 args: state, actuator_info, action)
            state_np = set_actuator_goals(state_np, np_act, actions_np)
            state_jax = set_actuator_goals(state_jax, jax_act, actions_jax)

            # Physics substep (state, topology, constants, collision_data)
            state_np = physics_substep(state_np, np_topo, constants, np_cd)
            state_jax = physics_substep(state_jax, jax_topo, constants, jax_cd)

            # Update actuators (state, topology, constants)
            state_np = update_actuators(state_np, np_topo, constants)
            state_jax = update_actuators(state_jax, jax_topo, constants)

        # Compare final positions
        np.testing.assert_allclose(
            np.array(state_jax.positions), np.array(state_np.positions),
            atol=1e-4,
            err_msg="Position mismatch after 50 steps"
        )
