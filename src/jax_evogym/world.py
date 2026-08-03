"""World and object definitions. Mirrors original evogym.world API.

Uses numpy for one-time setup. All output converts to JAX arrays in utils.py.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import copy
import json
import numpy as np

from . import constants as C


def _world_type_array() -> np.ndarray:
    return np.array(sorted(C.WORLD_CELL_TYPES), dtype=np.int32)


def _validate_structure_types(structure: np.ndarray) -> None:
    """Reject unknown world cell ids at the API boundary."""
    invalid = np.unique(structure[~np.isin(structure, _world_type_array())])
    if invalid.size > 0:
        raise ValueError(f"unknown cell types: {invalid.tolist()}")


class WorldObject:
    """A voxel-based object in the world. Mirrors evogym.WorldObject."""

    def __init__(self) -> None:
        self.name: str = ""
        self.pos: Tuple[int, int] = (0, 0)
        self.grid: np.ndarray = np.zeros((0, 0), dtype=int)
        self.neighbors: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    @property
    def grid_size(self) -> Tuple[int, int]:
        """(width, height) of the object's grid."""
        if self.grid.size == 0:
            return (0, 0)
        return (self.grid.shape[1], self.grid.shape[0])

    @classmethod
    def from_array(
        cls,
        name: str,
        structure: np.ndarray,
        connections: Optional[np.ndarray] = None,
    ) -> WorldObject:
        """Create from numpy array. Matches evogym's add_from_array.

        Args:
            name: object name.
            structure: (H, W) array of voxel types (user convention: row 0 = top).
            connections: (2, k) array of pairwise connections into np.flatten(structure).
                If None, auto-compute full adjacency connectivity.
        """
        from .utils import get_full_connectivity

        obj = cls()
        obj.name = name
        structure = np.asarray(structure, dtype=np.int32)
        _validate_structure_types(structure)

        if connections is None:
            corrected = get_full_connectivity(structure).T.copy()
        else:
            corrected = connections.copy().T

        # Flip vertically: internal grid stores row 0 = bottom (y-up)
        obj.grid = np.flipud(structure).astype(int)
        w, h = obj.grid.shape[1], obj.grid.shape[0]

        # Remap connection indices through the y-flip
        for i in range(len(corrected)):
            for j in range(len(corrected[i])):
                x = corrected[i][j] % w
                y = corrected[i][j] // w
                y = (h - 1) - y
                corrected[i][j] = y * w + x

        # Build voxel list and neighbor map
        obj.neighbors = {}
        idx_to_voxel: Dict[int, Tuple[int, int]] = {}
        for y in range(h):
            for x in range(w):
                idx = y * w + x
                if obj.grid[y, x] != C.EMPTY:
                    idx_to_voxel[idx] = (x, y)

        for vx, vy in idx_to_voxel.values():
            obj.neighbors[(vx, vy)] = []

        for conn in corrected:
            a, b = int(conn[0]), int(conn[1])
            if a not in idx_to_voxel or b not in idx_to_voxel:
                raise ValueError(
                    f"connections for {name} contain invalid elements"
                )
            va, vb = idx_to_voxel[a], idx_to_voxel[b]
            obj.neighbors[va].append(vb)
            obj.neighbors[vb].append(va)

        return obj

    @classmethod
    def from_json(cls, file_path: str) -> WorldObject:
        """Load a single object from a JSON file."""
        world = EvoWorld.from_json(file_path)
        if len(world.objects) != 1:
            raise ValueError(
                f"file {file_path} contains {len(world.objects)} objects, expected 1"
            )
        return next(iter(world.objects.values()))

    def load_from_parsed_json(
        self,
        name: str,
        json_data: dict,
        grid_width: int,
        grid_height: int,
    ) -> None:
        """Load from parsed JSON data (called by EvoWorld.add_from_json)."""
        self.name = name

        indices = json_data["indices"]
        types = json_data["types"]
        neighbors_data = json_data["neighbors"]
        num_voxels = len(indices)
        _validate_structure_types(np.asarray(types, dtype=np.int32))

        # Convert flat indices to (x, y) coordinates
        voxels = []
        index_to_voxel: Dict[int, Tuple[int, int]] = {}
        for i in range(num_voxels):
            idx = indices[i]
            vx = idx % grid_width
            vy = idx // grid_width
            voxels.append((vx, vy))
            index_to_voxel[idx] = (vx, vy)

        if not voxels:
            raise ValueError(f"object {name} has no voxels")

        # Compute bounding box
        xs = [v[0] for v in voxels]
        ys = [v[1] for v in voxels]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        self.pos = (min_x, min_y)
        w = max_x - min_x + 1
        h = max_y - min_y + 1

        # Translate to local coordinates
        local_voxels = [(vx - min_x, vy - min_y) for vx, vy in voxels]
        for idx in index_to_voxel:
            vx, vy = index_to_voxel[idx]
            index_to_voxel[idx] = (vx - min_x, vy - min_y)

        # Build grid
        grid = np.zeros((h, w), dtype=int)
        self.neighbors = {}
        for vx, vy in local_voxels:
            self.neighbors[(vx, vy)] = []

        for i in range(num_voxels):
            idx = indices[i]
            vx, vy = index_to_voxel[idx]
            grid[vy, vx] = types[i]
            for nei_idx in neighbors_data[str(idx)]:
                if nei_idx not in index_to_voxel:
                    raise ValueError(
                        f"object {name} has voxels with invalid neighbors"
                    )
                self.neighbors[(vx, vy)].append(index_to_voxel[nei_idx])

        self.grid = grid

    def get_structure(self) -> np.ndarray:
        """Return structure in user convention (row 0 = top)."""
        return np.flipud(self.grid)

    def get_connections(self) -> np.ndarray:
        """Return (2, k) connections array in structure-space indices."""
        w, h = self.grid_size
        out = []
        for (px, py), neighs in self.neighbors.items():
            for (nx, ny) in neighs:
                out.append([
                    (h - py - 1) * w + px,
                    (h - ny - 1) * w + nx,
                ])
        if not out:
            return np.empty((2, 0), dtype=int)
        return np.array(out, dtype=int).T

    def get_pos(self) -> Tuple[int, int]:
        return self.pos

    def set_pos(self, x: int, y: int) -> None:
        if x < 0 or y < 0:
            raise ValueError(f"position ({x}, {y}) must be non-negative")
        self.pos = (x, y)

    def translate(self, dx: int, dy: int) -> None:
        new_x = self.pos[0] + dx
        new_y = self.pos[1] + dy
        if new_x < 0 or new_y < 0:
            raise ValueError(
                f"translated position ({new_x}, {new_y}) is invalid"
            )
        self.pos = (new_x, new_y)

    def copy(self) -> WorldObject:
        return copy.deepcopy(self)

    def to_world(self) -> EvoWorld:
        """Wrap this object in a single-object world for export."""
        world = EvoWorld()
        world.add_object(self.copy())
        return world

    def __str__(self) -> str:
        w, h = self.grid_size
        return f"Size ({w}, {h}) object named {self.name} at {self.pos}"

    def __repr__(self) -> str:
        return f"WorldObject({self})"


class EvoWorld:
    """World definition containing objects and terrain. Mirrors evogym.EvoWorld."""

    def __init__(self) -> None:
        self.objects: Dict[str, WorldObject] = {}
        self._grid: np.ndarray = np.array([[0]], dtype=int)
        self._grid_w: int = 1
        self._grid_h: int = 1

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    @property
    def grid_size(self) -> Tuple[int, int]:
        """(width, height)."""
        return (self._grid_w, self._grid_h)

    @classmethod
    def from_json(cls, file_path: str) -> EvoWorld:
        out = cls()
        out.add_from_json(file_path)
        return out

    def add_from_json(self, file_path: str) -> None:
        with open(file_path, "r") as f:
            state = json.load(f)

        gw = state["grid_width"]
        gh = state["grid_height"]

        if (
            not isinstance(gw, int)
            or isinstance(gw, bool)
            or not isinstance(gh, int)
            or isinstance(gh, bool)
            or gw <= 0
            or gh <= 0
        ):
            raise ValueError(
                f"corrupted world in {file_path}: grid dimensions must be positive integers"
            )

        for name, obj_data in state["objects"].items():
            if len(obj_data["indices"]) != len(obj_data["types"]):
                raise ValueError(f"corrupted object {name} in {file_path}")
            if len(set(obj_data["indices"])) != len(obj_data["indices"]):
                raise ValueError(
                    f"corrupted object {name} in {file_path}: duplicate indices"
                )
            for idx in obj_data["indices"]:
                if (
                    not isinstance(idx, int)
                    or isinstance(idx, bool)
                    or idx < 0
                    or idx >= gw * gh
                ):
                    raise ValueError(
                        f"corrupted object {name} in {file_path}: "
                        f"index {idx!r} is outside the declared {gw}x{gh} grid"
                    )
            if len(obj_data["indices"]) != len(obj_data["neighbors"]):
                raise ValueError(f"corrupted object {name} in {file_path}")
            if any(str(idx) not in obj_data["neighbors"] for idx in obj_data["indices"]):
                raise ValueError(
                    f"corrupted object {name} in {file_path}: missing neighbor entry"
                )
            if name in self.objects:
                raise ValueError(f"duplicate object name: {name}")

        # A declared JSON grid is part of the serialized coordinate system, even
        # when its occupied bounding box is much smaller. Reserve it before
        # adding objects so future flat indices use the same width and height.
        if gw > self._grid_w or gh > self._grid_h:
            new_w = max(self._grid_w, gw)
            new_h = max(self._grid_h, gh)
            new_grid = np.zeros((new_h, new_w), dtype=int)
            new_grid[: self._grid_h, : self._grid_w] = self._grid
            self._grid = new_grid
            self._grid_w = new_w
            self._grid_h = new_h

        for name, obj_data in state["objects"].items():
            obj = WorldObject()
            obj.load_from_parsed_json(name, obj_data, gw, gh)
            self.add_object(obj)

    def add_from_array(
        self,
        name: str,
        structure: np.ndarray,
        x: int,
        y: int,
        connections: Optional[np.ndarray] = None,
    ) -> None:
        new_obj = WorldObject.from_array(name, structure, connections)
        new_obj.set_pos(x, y)
        self.add_object(new_obj)

    def add_object(self, obj: WorldObject) -> None:
        if obj.name in self.objects:
            raise ValueError(f"duplicate object name: {obj.name}")
        _validate_structure_types(np.asarray(obj.grid, dtype=np.int32))

        obj_w, obj_h = obj.grid_size
        x_start, y_start = obj.pos
        x_end = x_start + obj_w
        y_end = y_start + obj_h

        # Extend grid if needed
        if x_end > self._grid_w or y_end > self._grid_h:
            new_w = max(self._grid_w, x_end)
            new_h = max(self._grid_h, y_end)
            new_grid = np.zeros((new_h, new_w), dtype=int)
            new_grid[: self._grid_h, : self._grid_w] = self._grid
            self._grid = new_grid
            self._grid_w = new_w
            self._grid_h = new_h

        if np.any(self._grid[y_start:y_end, x_start:x_end] != 0):
            raise ValueError(
                f"object {obj.name} overlaps with existing object"
            )

        self._grid[y_start:y_end, x_start:x_end] = obj.grid
        self.objects[obj.name] = obj

    def remove_object(self, obj_name: str) -> WorldObject:
        if obj_name not in self.objects:
            raise ValueError(f"no object named {obj_name}")
        obj = self.objects[obj_name].copy()

        obj_w, obj_h = obj.grid_size
        x_start, y_start = obj.pos
        self._grid[y_start : y_start + obj_h, x_start : x_start + obj_w] = 0
        del self.objects[obj_name]
        return obj

    def get_objects(self) -> Dict[str, WorldObject]:
        return self.objects

    def get_robot(self) -> Optional[WorldObject]:
        """Return the first object containing actuators, or None."""
        for obj in self.objects.values():
            struct = obj.get_structure()
            if np.any(np.isin(struct, np.array(sorted(C.ACTUATOR_TYPES), dtype=np.int32))):
                return obj
        return None

    def to_json_dict(self) -> dict:
        """Serialize the world to canonical EvoGym JSON."""
        objects_out = {}
        for name, obj in sorted(self.objects.items()):
            indices: list[int] = []
            types: list[int] = []
            index_to_local: dict[int, tuple[int, int]] = {}
            for local_y in range(obj.grid.shape[0]):
                for local_x in range(obj.grid.shape[1]):
                    vtype = int(obj.grid[local_y, local_x])
                    if vtype == C.EMPTY:
                        continue
                    world_x = obj.pos[0] + local_x
                    world_y = obj.pos[1] + local_y
                    flat_idx = world_y * self._grid_w + world_x
                    indices.append(flat_idx)
                    types.append(vtype)
                    index_to_local[flat_idx] = (local_x, local_y)

            order = np.argsort(np.array(indices, dtype=np.int64))
            indices = [indices[int(idx)] for idx in order]
            types = [types[int(idx)] for idx in order]

            neighbors: dict[str, list[int]] = {}
            for flat_idx in indices:
                local_x, local_y = index_to_local[flat_idx]
                neighbor_indices = [
                    (obj.pos[1] + neighbor_y) * self._grid_w
                    + obj.pos[0]
                    + neighbor_x
                    for neighbor_x, neighbor_y in obj.neighbors.get(
                        (local_x, local_y), []
                    )
                ]
                neighbors[str(flat_idx)] = sorted(neighbor_indices)

            objects_out[name] = {
                "indices": indices,
                "types": types,
                "neighbors": neighbors,
            }

        return {
            "grid_width": int(self._grid_w),
            "grid_height": int(self._grid_h),
            "objects": objects_out,
        }

    def to_json(self, path: str | Path) -> None:
        """Write the world to canonical EvoGym JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_json_dict(), handle, indent=4)
            handle.write("\n")

    def pretty_print(self, voxels_per_line: int = 50) -> None:
        labels = {
            C.EMPTY: ". ",
            C.RIGID: "R ",
            C.SOFT: "S ",
            C.H_ACT: "H ",
            C.V_ACT: "V ",
            C.FIXED: "F ",
            C.CONTRACTILE: "C ",
            C.SLOPE_UP_RIGHT: "UR",
            C.SLOPE_UP_LEFT: "UL",
            C.SLOPE_DOWN_RIGHT: "DR",
            C.SLOPE_DOWN_LEFT: "DL",
            C.SLOPE2_UP_RIGHT_LIGHT: "R1",
            C.SLOPE2_UP_RIGHT_HEAVY: "R2",
            C.SLOPE2_UP_LEFT_HEAVY: "L2",
            C.SLOPE2_UP_LEFT_LIGHT: "L1",
            C.SLOPE2_DOWN_RIGHT_LIGHT: "r1",
            C.SLOPE2_DOWN_RIGHT_HEAVY: "r2",
            C.SLOPE2_DOWN_LEFT_HEAVY: "l2",
            C.SLOPE2_DOWN_LEFT_LIGHT: "l1",
            C.SLOPE3_UP_RIGHT_LIGHT: "R1",
            C.SLOPE3_UP_RIGHT_MID: "R2",
            C.SLOPE3_UP_RIGHT_HEAVY: "R3",
            C.SLOPE3_UP_LEFT_HEAVY: "L3",
            C.SLOPE3_UP_LEFT_MID: "L2",
            C.SLOPE3_UP_LEFT_LIGHT: "L1",
            C.SLOPE3_DOWN_RIGHT_HEAVY: "r3",
            C.SLOPE3_DOWN_RIGHT_MID: "r2",
            C.SLOPE3_DOWN_RIGHT_LIGHT: "r1",
            C.SLOPE3_DOWN_LEFT_LIGHT: "l1",
            C.SLOPE3_DOWN_LEFT_MID: "l2",
            C.SLOPE3_DOWN_LEFT_HEAVY: "l3",
        }
        w, h = self.grid_size
        for start_x in range(0, w, voxels_per_line):
            for y in reversed(range(h)):
                print(f"\n{y % 10} | ", end="")
                for x in range(start_x, min(w, start_x + voxels_per_line)):
                    print(labels.get(int(self._grid[y, x]), "? "), end="")
            print(f"\n   -", end="")
            for x in range(start_x, min(w, start_x + voxels_per_line)):
                print("--", end="")
            print(f"\n    ", end="")
            for x in range(start_x, min(w, start_x + voxels_per_line)):
                print(f"{x % 10} ", end="")
            print()
