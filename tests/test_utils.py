"""Tests for jax_evogym.utils — mass-spring conversion, connectivity."""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from jax_evogym import constants as C
from jax_evogym.spring_scaling import contractile_diag_ratio
from jax_evogym.utils import (
    create_dense_point_grid,
    create_dense_spring_indices,
    get_full_connectivity,
    has_actuator,
    is_connected,
    make_sim_state,
    sample_robot,
    voxel_to_dense_mass_spring,
)


# ---------------------------------------------------------------------------
# Dense grid helpers
# ---------------------------------------------------------------------------

class TestDensePointGrid:

    def test_default_shape(self):
        pts = create_dense_point_grid()
        assert pts.shape == (225, 2)

    def test_custom_shape(self):
        pts = create_dense_point_grid(H=3, W=5)
        assert pts.shape == (4 * 6, 2)

    def test_cell_size_spacing(self):
        pts = create_dense_point_grid(H=2, W=2)
        # Point at (1, 0) should be at x=0.1
        # Index = 0 * 3 + 1 = 1
        assert pts[1, 0] == pytest.approx(0.1)
        assert pts[1, 1] == pytest.approx(0.0)

    def test_origin(self):
        pts = create_dense_point_grid()
        assert pts[0, 0] == pytest.approx(0.0)
        assert pts[0, 1] == pytest.approx(0.0)

    def test_max_point(self):
        pts = create_dense_point_grid(H=14, W=14)
        # Last point: grid_y=14, grid_x=14
        assert pts[-1, 0] == pytest.approx(14 * 0.1)
        assert pts[-1, 1] == pytest.approx(14 * 0.1)


class TestDenseSpringIndices:

    def test_default_count(self):
        springs = create_dense_spring_indices()
        # 15*14 + 14*15 + 14*14*2 = 210 + 210 + 392 = 812
        assert len(springs) == 812

    def test_order(self):
        springs = create_dense_spring_indices()
        types = [s[2] for s in springs]
        # All horiz first, then vert, then diag
        first_vert = types.index("vert")
        first_diag = types.index("diag1")
        last_horiz = len(types) - 1 - types[::-1].index("horiz")
        last_vert = len(types) - 1 - types[::-1].index("vert")
        assert last_horiz < first_vert
        assert last_vert < first_diag

    def test_custom_grid_count(self):
        springs = create_dense_spring_indices(H=2, W=3)
        n_horiz = 3 * 3  # (H+1) * W = 3 * 3 = 9
        n_vert = 2 * 4   # H * (W+1) = 2 * 4 = 8
        n_diag = 2 * 3 * 2  # H * W * 2 = 12
        assert len(springs) == n_horiz + n_vert + n_diag


# ---------------------------------------------------------------------------
# Voxel to mass-spring conversion
# ---------------------------------------------------------------------------

class TestVoxelConversion:

    def test_empty_grid(self):
        morph = np.zeros((3, 5), dtype=int)
        data = voxel_to_dense_mass_spring(morph)
        assert data["n_points"] == 225
        assert data["n_springs"] == 812
        assert not np.any(data["point_mask"])
        assert not np.any(data["spring_mask"])

    def test_single_rigid_voxel(self):
        morph = np.array([[C.RIGID]])
        data = voxel_to_dense_mass_spring(morph)

        active_pts = np.sum(data["point_mask"])
        active_springs = np.sum(data["spring_mask"])
        assert active_pts == 4
        assert active_springs == 6  # 4 main + 2 diag

    def test_single_soft_voxel(self):
        morph = np.array([[C.SOFT]])
        data = voxel_to_dense_mass_spring(morph)

        active_mask = data["spring_mask"]
        active_const = data["spring_const"][active_mask]

        # Main edges should be SOFT_MAIN_K, diags should be SOFT_DIAG_K
        main_count = np.sum(np.isclose(active_const, C.SOFT_MAIN_K))
        diag_count = np.sum(np.isclose(active_const, C.SOFT_DIAG_K))
        assert main_count == 4
        assert diag_count == 2

    def test_single_rigid_voxel_constants(self):
        morph = np.array([[C.RIGID]])
        data = voxel_to_dense_mass_spring(morph)

        active_mask = data["spring_mask"]
        active_const = data["spring_const"][active_mask]

        main_count = np.sum(np.isclose(active_const, C.RIGID_MAIN_K))
        diag_count = np.sum(np.isclose(active_const, C.RIGID_DIAG_K))
        assert main_count == 4
        assert diag_count == 2

    def test_contractile_stiffness_scale_applies_to_contractile(self):
        morph = np.array([[C.CONTRACTILE]])
        k = 0.5
        data = voxel_to_dense_mass_spring(
            morph,
            H=1,
            W=1,
            spring_stiffness_scale=k,
        )
        active_const = data["spring_const"][data["spring_mask"]]
        expected_main = C.ACTUATOR_MAIN_K * k
        expected_diag = expected_main * contractile_diag_ratio(k)
        main_count = np.sum(np.isclose(active_const, expected_main))
        diag_count = np.sum(np.isclose(active_const, expected_diag))
        assert main_count == 4
        assert diag_count == 2

    def test_contractile_stiffness_scale_hits_soft_anchor(self):
        morph = np.array([[C.CONTRACTILE]])
        k = C.SOFT_MAIN_K / C.ACTUATOR_MAIN_K
        data = voxel_to_dense_mass_spring(
            morph,
            H=1,
            W=1,
            spring_stiffness_scale=k,
        )
        active_const = data["spring_const"][data["spring_mask"]]
        main_count = np.sum(np.isclose(active_const, C.SOFT_MAIN_K))
        diag_count = np.sum(np.isclose(active_const, C.SOFT_DIAG_K))
        assert main_count == 4
        assert diag_count == 2

    def test_contractile_stiffness_scale_above_one_keeps_actuator_ratio(self):
        morph = np.array([[C.CONTRACTILE]])
        k = 1.5
        data = voxel_to_dense_mass_spring(
            morph,
            H=1,
            W=1,
            spring_stiffness_scale=k,
        )
        active_const = data["spring_const"][data["spring_mask"]]
        expected_main = C.ACTUATOR_MAIN_K * k
        expected_diag = expected_main * (C.ACTUATOR_DIAG_K / C.ACTUATOR_MAIN_K)
        main_count = np.sum(np.isclose(active_const, expected_main))
        diag_count = np.sum(np.isclose(active_const, expected_diag))
        assert main_count == 4
        assert diag_count == 2

    def test_contractile_stiffness_scale_does_not_change_h_act(self):
        morph = np.array([[C.H_ACT]])
        data = voxel_to_dense_mass_spring(
            morph,
            H=1,
            W=1,
            spring_stiffness_scale=0.5,
        )
        active_const = data["spring_const"][data["spring_mask"]]
        main_count = np.sum(np.isclose(active_const, C.ACTUATOR_MAIN_K))
        diag_count = np.sum(np.isclose(active_const, C.ACTUATOR_DIAG_K))
        assert main_count == 4
        assert diag_count == 2

    def test_two_pass_rigid_soft_shared_edge(self):
        """Adjacent RIGID-SOFT: shared edge uses SOFT constant (C++ behavior)."""
        # RIGID on left, SOFT on right (user convention: row 0 = top)
        morph = np.array([[C.RIGID, C.SOFT]])
        data = voxel_to_dense_mass_spring(morph)

        # The shared vertical edge between the two cells
        # After two-pass, should be SOFT_MAIN_K (SOFT overwrites in pass 2)
        active_mask = data["spring_mask"]
        active_const = data["spring_const"][active_mask]

        # 2 cells: 8 main edges total, but 1 is shared = 7 unique main edges
        # Plus 4 diagonals = 11 total active springs
        # OR: each cell has 4 main + 2 diag = 6, shared edge counted once
        # Total: 4 + 4 - 1(shared) + 2 + 2 = 11
        assert np.sum(active_mask) == 11

        # Count spring constants:
        # RIGID: 3 non-shared main = RIGID_MAIN_K, 2 diag = RIGID_DIAG_K
        # SOFT overwrites shared + its 3 own = 4 SOFT_MAIN_K total
        # SOFT diag = 2 SOFT_DIAG_K
        rigid_main = np.sum(np.isclose(active_const, C.RIGID_MAIN_K))
        soft_main = np.sum(np.isclose(active_const, C.SOFT_MAIN_K))
        rigid_diag = np.sum(np.isclose(active_const, C.RIGID_DIAG_K))
        soft_diag = np.sum(np.isclose(active_const, C.SOFT_DIAG_K))

        assert rigid_main == 3  # RIGID's 3 non-shared main edges
        assert soft_main == 4   # SOFT's 4 main (including shared)
        assert rigid_diag == 2
        assert soft_diag == 2

    def test_rigid_spring_mask(self):
        """rigid_spring_mask should be True only for RIGID/FIXED cell springs."""
        morph = np.array([[C.RIGID, C.SOFT]])
        data = voxel_to_dense_mass_spring(morph)

        rigid_mask = data["rigid_spring_mask"]
        spring_mask = data["spring_mask"]

        # Rigid cell: 4 main + 2 diag = 6 springs marked rigid
        rigid_active = rigid_mask & spring_mask
        assert np.sum(rigid_active) == 6

    def test_actuator_tracking_h_act(self):
        """H_ACT cell should produce 1 actuator cell driving 2 horizontal springs."""
        morph = np.array([[C.H_ACT]])
        data = voxel_to_dense_mass_spring(morph)
        springs = create_dense_spring_indices()

        cell_indices = data["actuator_cell_spring_indices"]
        assert cell_indices.shape == (1, 2)  # 1 cell, 2 springs

        # Both springs should be horizontal
        for idx in cell_indices[0]:
            assert springs[idx][2] == "horiz"

    def test_actuator_tracking_v_act(self):
        """V_ACT cell should produce 1 actuator cell driving 2 vertical springs."""
        morph = np.array([[C.V_ACT]])
        data = voxel_to_dense_mass_spring(morph)
        springs = create_dense_spring_indices()

        cell_indices = data["actuator_cell_spring_indices"]
        assert cell_indices.shape == (1, 2)

        for idx in cell_indices[0]:
            assert springs[idx][2] == "vert"

    def test_position_scaling(self):
        """Points should be at cell_size=0.1 intervals."""
        morph = np.array([[C.RIGID]])
        data = voxel_to_dense_mass_spring(morph)

        active = data["point_mask"]
        active_pos = data["positions"][active]

        # Active points should span 0.1 in each dimension
        x_range = active_pos[:, 0].max() - active_pos[:, 0].min()
        y_range = active_pos[:, 1].max() - active_pos[:, 1].min()
        assert x_range == pytest.approx(C.CELL_SIZE)
        assert y_range == pytest.approx(C.CELL_SIZE)

    def test_rest_lengths(self):
        """Main edges = cell_size, diagonals = sqrt(2)*cell_size."""
        morph = np.array([[C.RIGID]])
        data = voxel_to_dense_mass_spring(morph)

        active = data["spring_mask"]
        lengths = data["spring_rest_length"][active]

        diag_len = math.sqrt(2) * C.CELL_SIZE
        main_count = np.sum(np.isclose(lengths, C.CELL_SIZE))
        diag_count = np.sum(np.isclose(lengths, diag_len))
        assert main_count == 4
        assert diag_count == 2

    def test_fixed_voxel(self):
        """FIXED voxel corners should have fixed=True."""
        morph = np.array([[C.FIXED]])
        data = voxel_to_dense_mass_spring(morph)

        active = data["point_mask"]
        fixed = data["fixed"]

        # All active points should be fixed
        assert np.all(fixed[active])

    def test_fixed_adjacent_soft(self):
        """Shared point between FIXED and SOFT should be fixed."""
        morph = np.array([[C.FIXED, C.SOFT]])
        data = voxel_to_dense_mass_spring(morph)

        active = data["point_mask"]
        fixed = data["fixed"]

        # FIXED cell has 4 points fixed. Shared points (2) with SOFT should also be fixed.
        # SOFT cell's 2 non-shared points should NOT be fixed.
        fixed_count = np.sum(fixed[active, 0])
        assert fixed_count == 4  # The 4 FIXED corner points (2 shared)

    def test_mass_assignment(self):
        morph = np.array([[C.RIGID]])
        data = voxel_to_dense_mass_spring(morph)

        active = data["point_mask"]
        masses = data["masses"][active]
        assert np.all(masses == C.POINT_MASS)

    def test_ghost_points_properties(self):
        """Ghost points: mass=0, fixed=True, mask=False."""
        morph = np.array([[C.RIGID]])
        data = voxel_to_dense_mass_spring(morph)

        ghost = ~data["point_mask"]
        assert np.all(data["masses"][ghost] == 0.0)
        assert np.all(data["fixed"][ghost])

    def test_morphology_too_large(self):
        morph = np.ones((20, 20), dtype=int)
        with pytest.raises(ValueError, match="exceeds"):
            voxel_to_dense_mass_spring(morph)


# ---------------------------------------------------------------------------
# make_sim_state
# ---------------------------------------------------------------------------

class TestMakeSimState:

    def test_basic_construction(self):
        morph = np.array([[C.SOFT]])
        data = voxel_to_dense_mass_spring(morph)
        state, topo, act_info = make_sim_state(data)

        assert state.positions.shape == (225, 2)
        assert state.velocities.shape == (225, 2)
        assert state.spring_const.shape == (812,)
        assert topo.a_idx.shape == (812,)
        assert topo.b_idx.shape == (812,)

    def test_spawn_offset(self):
        morph = np.array([[C.RIGID]])
        data = voxel_to_dense_mass_spring(morph)

        state_no_offset, _, _ = make_sim_state(data, spawn_x=0, spawn_y=0)
        state_offset, _, _ = make_sim_state(data, spawn_x=10, spawn_y=5)

        # Active points should be shifted by (10*0.1, 5*0.1) = (1.0, 0.5)
        mask = state_no_offset.point_mask
        diff = state_offset.positions[mask] - state_no_offset.positions[mask]
        assert jnp.allclose(diff[:, 0], 1.0)
        assert jnp.allclose(diff[:, 1], 0.5)

    def test_ghost_points_not_shifted(self):
        morph = np.array([[C.RIGID]])
        data = voxel_to_dense_mass_spring(morph)

        state_no_offset, _, _ = make_sim_state(data, spawn_x=0, spawn_y=0)
        state_offset, _, _ = make_sim_state(data, spawn_x=10, spawn_y=5)

        ghost = ~state_no_offset.point_mask
        diff = state_offset.positions[ghost] - state_no_offset.positions[ghost]
        assert jnp.allclose(diff, 0.0)

    def test_actuator_info_empty_for_rigid(self):
        morph = np.array([[C.RIGID]])
        data = voxel_to_dense_mass_spring(morph)
        _, _, act_info = make_sim_state(data)
        assert act_info.cell_spring_indices.shape == (0, 2)

    def test_actuator_info_populated_for_h_act(self):
        morph = np.array([[C.H_ACT]])
        data = voxel_to_dense_mass_spring(morph)
        _, _, act_info = make_sim_state(data)
        assert act_info.cell_spring_indices.shape[0] > 0

    def test_jax_dtypes(self):
        morph = np.array([[C.RIGID]])
        data = voxel_to_dense_mass_spring(morph)
        state, topo, _ = make_sim_state(data)

        assert state.positions.dtype == jnp.float32
        assert state.fixed.dtype == jnp.bool_
        assert state.point_mask.dtype == jnp.bool_
        assert topo.a_idx.dtype == jnp.int32

    def test_rest_length_copies(self):
        """rest_length, goal, and init should all be equal at construction."""
        morph = np.array([[C.RIGID]])
        data = voxel_to_dense_mass_spring(morph)
        state, _, _ = make_sim_state(data)

        assert jnp.allclose(state.spring_rest_length, state.spring_rest_length_goal)
        assert jnp.allclose(state.spring_rest_length, state.spring_init_rest_length)


# ---------------------------------------------------------------------------
# Connectivity helpers
# ---------------------------------------------------------------------------

class TestGetFullConnectivity:

    def test_single_voxel(self):
        struct = np.array([[1]])
        conn = get_full_connectivity(struct)
        assert conn.shape == (2, 0)

    def test_two_horizontal(self):
        struct = np.array([[1, 1]])
        conn = get_full_connectivity(struct)
        assert conn.shape == (2, 1)
        assert set(conn[:, 0]) == {0, 1}

    def test_two_vertical(self):
        struct = np.array([[1], [1]])
        conn = get_full_connectivity(struct)
        assert conn.shape == (2, 1)
        assert set(conn[:, 0]) == {0, 1}

    def test_2x2_block(self):
        struct = np.ones((2, 2), dtype=int)
        conn = get_full_connectivity(struct)
        # 4 connections: (0,1), (2,3), (0,2), (1,3)
        assert conn.shape[1] == 4

    def test_empty_structure(self):
        struct = np.zeros((2, 2), dtype=int)
        conn = get_full_connectivity(struct)
        assert conn.shape == (2, 0)

    def test_l_shape(self):
        struct = np.array([
            [1, 0],
            [1, 1],
        ])
        conn = get_full_connectivity(struct)
        # Connections: (0,2), (2,3)
        assert conn.shape[1] == 2


class TestIsConnected:

    def test_single_voxel(self):
        assert is_connected(np.array([[1]]))

    def test_2x2_block(self):
        assert is_connected(np.ones((2, 2), dtype=int))

    def test_disconnected(self):
        struct = np.array([
            [1, 0, 1],
        ])
        assert not is_connected(struct)

    def test_l_shape(self):
        struct = np.array([
            [1, 0],
            [1, 1],
        ])
        assert is_connected(struct)

    def test_empty(self):
        # Empty morphology (no non-empty voxels) is not connected
        assert not is_connected(np.zeros((2, 2), dtype=int))


class TestHasActuator:

    def test_with_h_act(self):
        struct = np.array([[C.H_ACT]])
        assert has_actuator(struct)

    def test_with_v_act(self):
        struct = np.array([[C.V_ACT]])
        assert has_actuator(struct)

    def test_no_actuator(self):
        struct = np.array([[C.RIGID, C.SOFT]])
        assert not has_actuator(struct)


class TestSampleRobot:

    def test_basic(self):
        robot, conn = sample_robot((3, 3), rng=np.random.default_rng(42))
        assert robot.shape == (3, 3)
        assert is_connected(robot)
        assert has_actuator(robot)
        assert conn.shape[0] == 2

    def test_custom_pd(self):
        pd = np.array([0.0, 0.0, 0.0, 0.5, 0.5])
        robot, conn = sample_robot((2, 2), pd=pd, rng=np.random.default_rng(0))
        assert robot.shape == (2, 2)
        # Only H_ACT and V_ACT
        assert np.all((robot == C.H_ACT) | (robot == C.V_ACT))
