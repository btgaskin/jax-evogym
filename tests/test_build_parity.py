"""Object-separated world build parity checks."""

from __future__ import annotations

import numpy as np
import pytest

from jax_evogym import constants as C
from jax_evogym.build import compile_world_template, compile_world_templates, instantiate_world
from jax_evogym.spring_scaling import contractile_diag_ratio
from jax_evogym.world import EvoWorld


class TestObjectSeparatedContractileParity:
    def test_contractile_uses_actuator_main_and_diag_constants(self):
        """Compiled CONTRACTILE cells should use actuator spring materials."""
        world = EvoWorld()
        world.add_from_array("robot", np.array([[C.CONTRACTILE]], dtype=np.int32), 0, 0)

        template = compile_world_template(world, robot_name="robot")
        built = instantiate_world(template)

        spring_mask = np.asarray(built.sim_state.spring_mask, dtype=bool)
        spring_const = np.asarray(built.sim_state.spring_const, dtype=np.float32)[spring_mask]

        assert spring_const.shape[0] == 6  # 4 main + 2 diagonals
        np.testing.assert_equal(np.count_nonzero(np.isclose(spring_const, C.ACTUATOR_MAIN_K)), 4)
        np.testing.assert_equal(np.count_nonzero(np.isclose(spring_const, C.ACTUATOR_DIAG_K)), 2)

    def test_contractile_diagonal_springs_are_active(self):
        """Contractile diagonals should be present and use actuator diagonal stiffness."""
        world = EvoWorld()
        world.add_from_array("robot", np.array([[C.CONTRACTILE]], dtype=np.int32), 0, 0)

        template = compile_world_template(world, robot_name="robot")
        built = instantiate_world(template)

        spring_mask = np.asarray(built.sim_state.spring_mask, dtype=bool)
        spring_const = np.asarray(built.sim_state.spring_const, dtype=np.float32)[spring_mask]
        spring_rest = np.asarray(built.sim_state.spring_rest_length, dtype=np.float32)[spring_mask]

        diag_len = np.float32(np.sqrt(2.0) * C.CELL_SIZE)
        diagonal_active = np.isclose(spring_rest, diag_len, atol=1e-6)
        assert int(np.count_nonzero(diagonal_active)) == 2
        np.testing.assert_allclose(
            spring_const[diagonal_active],
            np.full((2,), C.ACTUATOR_DIAG_K, dtype=np.float32),
            atol=1e-3,
        )

    def test_contractile_scale_applies_with_robot_override(self):
        world = EvoWorld()
        world.add_from_array("robot", np.array([[C.CONTRACTILE]], dtype=np.int32), 0, 0)
        template = compile_world_template(world, robot_name="robot")
        built = instantiate_world(
            template,
            robot_override=np.array([[C.CONTRACTILE]], dtype=np.int32),
            spring_stiffness_scale=0.5,
        )
        spring_mask = np.asarray(built.sim_state.spring_mask, dtype=bool)
        spring_const = np.asarray(built.sim_state.spring_const, dtype=np.float32)[spring_mask]
        expected_main = C.ACTUATOR_MAIN_K * 0.5
        expected_diag = expected_main * contractile_diag_ratio(0.5)
        np.testing.assert_equal(np.count_nonzero(np.isclose(spring_const, expected_main)), 4)
        np.testing.assert_equal(np.count_nonzero(np.isclose(spring_const, expected_diag)), 2)

    def test_compile_rejects_slope_cells_by_default(self):
        world = EvoWorld()
        world.add_from_array(
            "ground",
            np.array([[C.SLOPE2_UP_RIGHT_LIGHT, C.SLOPE2_UP_RIGHT_HEAVY]], dtype=np.int32),
            0,
            0,
        )
        world.add_from_array("robot", np.array([[C.RIGID]], dtype=np.int32), 0, 2)

        with pytest.raises(ValueError, match="unsupported in the stable build path"):
            compile_world_template(world, robot_name="robot")

    def test_compile_accepts_valid_2x1_slope_pair(self):
        world = EvoWorld()
        world.add_from_array(
            "ground",
            np.array([[C.SLOPE2_UP_RIGHT_LIGHT, C.SLOPE2_UP_RIGHT_HEAVY]], dtype=np.int32),
            0,
            0,
        )
        world.add_from_array("robot", np.array([[C.RIGID]], dtype=np.int32), 0, 2)

        template = compile_world_template(world, robot_name="robot", allow_experimental_slopes=True)
        built = instantiate_world(template)
        static_types = np.asarray(built.static_collider_data.cell_types, dtype=np.int32)
        assert int(np.count_nonzero(static_types == C.SLOPE2_UP_RIGHT_HEAVY)) == 1
        assert int(np.count_nonzero(static_types == C.SLOPE2_UP_RIGHT_LIGHT)) == 1

    def test_compile_rejects_orphaned_2x1_slope_half(self):
        world = EvoWorld()
        world.add_from_array(
            "ground",
            np.array([[C.EMPTY, C.SLOPE2_UP_RIGHT_HEAVY]], dtype=np.int32),
            0,
            0,
        )
        world.add_from_array("robot", np.array([[C.RIGID]], dtype=np.int32), 0, 2)

        with pytest.raises(ValueError, match="invalid 2x1 slope pair"):
            compile_world_template(world, robot_name="robot", allow_experimental_slopes=True)

    def test_compile_rejects_wrong_2x1_slope_mate_type(self):
        world = EvoWorld()
        world.add_from_array(
            "ground",
            np.array([[C.FIXED, C.SLOPE2_UP_RIGHT_HEAVY]], dtype=np.int32),
            0,
            0,
        )
        world.add_from_array("robot", np.array([[C.RIGID]], dtype=np.int32), 0, 2)

        with pytest.raises(ValueError, match="invalid 2x1 slope pair"):
            compile_world_template(world, robot_name="robot", allow_experimental_slopes=True)

    def test_mirror_remaps_2x1_slope_orientation(self):
        world = EvoWorld()
        world.add_from_array(
            "ground",
            np.array([[C.SLOPE2_UP_RIGHT_LIGHT, C.SLOPE2_UP_RIGHT_HEAVY]], dtype=np.int32),
            0,
            0,
        )
        world.add_from_array("robot", np.array([[C.RIGID]], dtype=np.int32), 0, 2)

        templates = compile_world_templates(
            world,
            robot_name="robot",
            mirror_mode="paired",
            allow_experimental_slopes=True,
        )
        assert templates.mirror is not None
        built_mirror = instantiate_world(templates.mirror)
        static_types = np.asarray(built_mirror.static_collider_data.cell_types, dtype=np.int32)
        assert int(np.count_nonzero(static_types == C.SLOPE2_UP_LEFT_HEAVY)) == 1
        assert int(np.count_nonzero(static_types == C.SLOPE2_UP_LEFT_LIGHT)) == 1

    def test_compile_accepts_valid_3x1_slope_triplet(self):
        world = EvoWorld()
        world.add_from_array(
            "ground",
            np.array(
                [[C.SLOPE3_UP_RIGHT_LIGHT, C.SLOPE3_UP_RIGHT_MID, C.SLOPE3_UP_RIGHT_HEAVY]],
                dtype=np.int32,
            ),
            0,
            0,
        )
        world.add_from_array("robot", np.array([[C.RIGID]], dtype=np.int32), 0, 2)

        template = compile_world_template(world, robot_name="robot", allow_experimental_slopes=True)
        built = instantiate_world(template)
        static_types = np.asarray(built.static_collider_data.cell_types, dtype=np.int32)
        assert int(np.count_nonzero(static_types == C.SLOPE3_UP_RIGHT_LIGHT)) == 1
        assert int(np.count_nonzero(static_types == C.SLOPE3_UP_RIGHT_MID)) == 1
        assert int(np.count_nonzero(static_types == C.SLOPE3_UP_RIGHT_HEAVY)) == 1

    def test_compile_rejects_orphaned_3x1_slope_segment(self):
        world = EvoWorld()
        world.add_from_array(
            "ground",
            np.array([[C.EMPTY, C.EMPTY, C.SLOPE3_UP_RIGHT_HEAVY]], dtype=np.int32),
            0,
            0,
        )
        world.add_from_array("robot", np.array([[C.RIGID]], dtype=np.int32), 0, 2)

        with pytest.raises(ValueError, match="invalid 3x1 slope triplet"):
            compile_world_template(world, robot_name="robot", allow_experimental_slopes=True)

    def test_compile_rejects_wrong_3x1_slope_neighbors(self):
        world = EvoWorld()
        world.add_from_array(
            "ground",
            np.array([[C.FIXED, C.SLOPE3_UP_RIGHT_MID, C.SLOPE3_UP_RIGHT_HEAVY]], dtype=np.int32),
            0,
            0,
        )
        world.add_from_array("robot", np.array([[C.RIGID]], dtype=np.int32), 0, 2)

        with pytest.raises(ValueError, match="invalid 3x1 slope triplet"):
            compile_world_template(world, robot_name="robot", allow_experimental_slopes=True)

    def test_mirror_remaps_3x1_slope_orientation(self):
        world = EvoWorld()
        world.add_from_array(
            "ground",
            np.array(
                [[C.SLOPE3_UP_RIGHT_LIGHT, C.SLOPE3_UP_RIGHT_MID, C.SLOPE3_UP_RIGHT_HEAVY]],
                dtype=np.int32,
            ),
            0,
            0,
        )
        world.add_from_array("robot", np.array([[C.RIGID]], dtype=np.int32), 0, 2)

        templates = compile_world_templates(
            world,
            robot_name="robot",
            mirror_mode="paired",
            allow_experimental_slopes=True,
        )
        assert templates.mirror is not None
        built_mirror = instantiate_world(templates.mirror)
        static_types = np.asarray(built_mirror.static_collider_data.cell_types, dtype=np.int32)
        assert int(np.count_nonzero(static_types == C.SLOPE3_UP_LEFT_HEAVY)) == 1
        assert int(np.count_nonzero(static_types == C.SLOPE3_UP_LEFT_MID)) == 1
        assert int(np.count_nonzero(static_types == C.SLOPE3_UP_LEFT_LIGHT)) == 1
