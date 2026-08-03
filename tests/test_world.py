"""Tests for jax_evogym.world — EvoWorld and WorldObject."""

import os
import json

import numpy as np
import pytest

from jax_evogym import constants as C
from jax_evogym.world import EvoWorld, WorldObject


WALKER_JSON = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
    "jax_evogym",
    "data",
    "Walker-v0.json",
)


# ---------------------------------------------------------------------------
# WorldObject
# ---------------------------------------------------------------------------

class TestWorldObject:

    def test_from_array_basic(self):
        struct = np.array([
            [C.RIGID, C.SOFT],
            [C.H_ACT, C.V_ACT],
        ])
        obj = WorldObject.from_array("robot", struct)
        assert obj.name == "robot"
        assert obj.grid_size == (2, 2)

    def test_from_array_grid_is_flipped(self):
        """Internal grid stores row 0 = bottom (y-up)."""
        struct = np.array([
            [C.RIGID, C.SOFT],   # top row in user convention
            [C.H_ACT, C.V_ACT], # bottom row
        ])
        obj = WorldObject.from_array("robot", struct)
        # Internal grid[0] = bottom row of user convention
        assert obj.grid[0, 0] == C.H_ACT
        assert obj.grid[0, 1] == C.V_ACT
        assert obj.grid[1, 0] == C.RIGID
        assert obj.grid[1, 1] == C.SOFT

    def test_get_structure_roundtrip(self):
        """get_structure() should return the original user convention."""
        struct = np.array([
            [C.RIGID, C.SOFT],
            [C.H_ACT, C.V_ACT],
        ])
        obj = WorldObject.from_array("robot", struct)
        recovered = obj.get_structure()
        np.testing.assert_array_equal(recovered, struct)

    def test_set_pos(self):
        obj = WorldObject.from_array("test", np.array([[C.RIGID]]))
        obj.set_pos(5, 3)
        assert obj.get_pos() == (5, 3)

    def test_set_pos_negative_raises(self):
        obj = WorldObject.from_array("test", np.array([[C.RIGID]]))
        with pytest.raises(ValueError):
            obj.set_pos(-1, 0)

    def test_translate(self):
        obj = WorldObject.from_array("test", np.array([[C.RIGID]]))
        obj.set_pos(5, 3)
        obj.translate(2, -1)
        assert obj.get_pos() == (7, 2)

    def test_copy(self):
        obj = WorldObject.from_array("test", np.array([[C.RIGID, C.SOFT]]))
        obj.set_pos(1, 2)
        obj2 = obj.copy()
        assert obj2.name == "test"
        assert obj2.get_pos() == (1, 2)
        # Ensure deep copy
        obj2.set_pos(9, 9)
        assert obj.get_pos() == (1, 2)

    def test_connections_auto_computed(self):
        struct = np.array([[C.RIGID, C.SOFT]])
        obj = WorldObject.from_array("test", struct)
        conn = obj.get_connections()
        assert conn.shape[0] == 2
        assert conn.shape[1] > 0

    def test_str_repr(self):
        obj = WorldObject.from_array("bot", np.array([[C.RIGID]]))
        assert "bot" in str(obj)
        assert "WorldObject" in repr(obj)

    def test_to_world_wraps_single_object(self):
        obj = WorldObject.from_array("bot", np.array([[C.RIGID]]))
        obj.set_pos(2, 3)
        world = obj.to_world()
        assert set(world.objects) == {"bot"}
        assert world.objects["bot"].get_pos() == (2, 3)


# ---------------------------------------------------------------------------
# EvoWorld
# ---------------------------------------------------------------------------

class TestEvoWorld:

    def test_empty_world(self):
        world = EvoWorld()
        assert world.grid_size == (1, 1)
        assert len(world.objects) == 0

    def test_add_from_array(self):
        world = EvoWorld()
        struct = np.array([[C.RIGID]])
        world.add_from_array("block", struct, 0, 0)
        assert "block" in world.objects
        assert world.objects["block"].get_pos() == (0, 0)

    def test_add_from_array_extends_grid(self):
        world = EvoWorld()
        struct = np.array([[C.RIGID, C.RIGID]])
        world.add_from_array("wide", struct, 5, 3)
        w, h = world.grid_size
        assert w >= 7  # 5 + 2
        assert h >= 4  # 3 + 1

    def test_duplicate_name_raises(self):
        world = EvoWorld()
        struct = np.array([[C.RIGID]])
        world.add_from_array("block", struct, 0, 0)
        with pytest.raises(ValueError, match="duplicate"):
            world.add_from_array("block", struct, 2, 0)

    def test_overlap_raises(self):
        world = EvoWorld()
        struct = np.array([[C.RIGID]])
        world.add_from_array("a", struct, 0, 0)
        with pytest.raises(ValueError, match="overlaps"):
            world.add_from_array("b", struct, 0, 0)

    def test_remove_object(self):
        world = EvoWorld()
        struct = np.array([[C.RIGID]])
        world.add_from_array("block", struct, 0, 0)
        removed = world.remove_object("block")
        assert removed.name == "block"
        assert "block" not in world.objects

    def test_remove_nonexistent_raises(self):
        world = EvoWorld()
        with pytest.raises(ValueError):
            world.remove_object("nope")

    def test_get_robot_returns_actuated_object(self):
        world = EvoWorld()
        world.add_from_array("ground", np.array([[C.FIXED, C.FIXED]]), 0, 0)
        world.add_from_array("robot", np.array([[C.H_ACT]]), 5, 0)
        robot = world.get_robot()
        assert robot is not None
        assert robot.name == "robot"

    def test_get_robot_returns_none(self):
        world = EvoWorld()
        world.add_from_array("ground", np.array([[C.FIXED]]), 0, 0)
        assert world.get_robot() is None

    def test_grid_populated(self):
        world = EvoWorld()
        struct = np.array([[C.RIGID]])
        world.add_from_array("block", struct, 2, 3)
        # Grid should have RIGID at row 3, col 2
        # Internal grid is (y, x) — but row 3 after flip...
        # Actually the grid stores in internal y-up format from WorldObject,
        # but EvoWorld.grid just places obj.grid at (y_start:y_end, x_start:x_end)
        assert world.grid[3, 2] != 0


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

class TestJsonLoading:

    def test_load_walker_world(self):
        world = EvoWorld.from_json(WALKER_JSON)
        assert "ground" in world.objects

    def test_walker_ground_is_fixed(self):
        world = EvoWorld.from_json(WALKER_JSON)
        ground = world.objects["ground"]
        struct = ground.get_structure()
        # All voxels should be FIXED (type 5)
        assert np.all(struct[struct != 0] == C.FIXED)

    def test_walker_ground_size(self):
        world = EvoWorld.from_json(WALKER_JSON)
        ground = world.objects["ground"]
        w, h = ground.grid_size
        assert w == 100
        assert h == 1

    def test_walker_grid_size(self):
        world = EvoWorld.from_json(WALKER_JSON)
        w, h = world.grid_size
        assert w >= 100

    def test_world_object_from_json(self):
        obj = WorldObject.from_json(WALKER_JSON)
        assert obj.name == "ground"

    def test_world_to_json_dict_roundtrip(self):
        world = EvoWorld.from_json(WALKER_JSON)
        serialized = world.to_json_dict()
        assert serialized["grid_width"] >= 100
        assert "ground" in serialized["objects"]
        assert serialized["objects"]["ground"]["indices"][0] == 0

    def test_world_to_json_writes_file(self, tmp_path):
        world = EvoWorld.from_json(WALKER_JSON)
        target = tmp_path / "world.json"
        world.to_json(target)
        written = json.loads(target.read_text())
        assert written["grid_width"] >= 100
        assert "ground" in written["objects"]
