"""Object-separated world compilation and instantiation."""

from __future__ import annotations

from typing import Iterable

import jax.numpy as jnp
import numpy as np

from . import constants as C
from .cell_geometry import build_sparse_cell_geometry, grid_active_cells
from .types import (
    ActuatorInfo,
    BuiltWorld,
    CollisionData,
    DeformationInfo,
    ObjectTemplate,
    PerAxisActuatorInfo,
    RenderInfo,
    SimState,
    SpringTopology,
    StaticColliderData,
    WorldTemplate,
    WorldTemplateSet,
    default_physics_constants,
)
from .utils import _build_deformation_info, _build_mass_spring_from_padded
from .world import EvoWorld, WorldObject


def _ordered_objects(world: EvoWorld, robot_name: str) -> list[WorldObject]:
    if robot_name not in world.objects:
        raise ValueError(f"World is missing robot object {robot_name!r}")
    ordered: list[WorldObject] = [world.objects[robot_name]]
    for name in sorted(world.objects):
        if name == robot_name:
            continue
        ordered.append(world.objects[name])
    return ordered


def _is_robot_object(obj: WorldObject, robot_name: str) -> bool:
    return obj.name == robot_name


_STATIC_TERRAIN_ARRAY = np.array(sorted(C.STATIC_TERRAIN_TYPES), dtype=np.int32)
_SLOPE_TYPE_ARRAY = np.array(sorted(C.SLOPE_TYPES), dtype=np.int32)
_SLOPE_2X1_TYPE_ARRAY = np.array(sorted(C.SLOPE_2X1_TYPES), dtype=np.int32)
_SLOPE_3X1_TYPE_ARRAY = np.array(sorted(C.SLOPE_3X1_TYPES), dtype=np.int32)
_ACTUATOR_TYPE_ARRAY = np.array(sorted(C.ACTUATOR_TYPES), dtype=np.int32)
_H_RENDER_ACTUATION_TYPE_ARRAY = np.array([C.H_ACT, C.CONTRACTILE], dtype=np.int32)
_V_RENDER_ACTUATION_TYPE_ARRAY = np.array([C.V_ACT, C.CONTRACTILE], dtype=np.int32)


def _contains_slope(grid: np.ndarray) -> bool:
    return bool(np.any(np.isin(grid, _SLOPE_TYPE_ARRAY)))


def _validate_slope_2x1_pairing(grid: np.ndarray, *, obj_name: str) -> None:
    if not np.any(np.isin(grid, _SLOPE_2X1_TYPE_ARRAY)):
        return
    h, w = grid.shape
    for vy in range(h):
        for vx in range(w):
            vtype = int(grid[vy, vx])
            if vtype not in C.SLOPE_2X1_TYPES:
                continue
            mate_type = int(C.SLOPE_2X1_MATE_TYPE[vtype])
            dx, dy = C.SLOPE_2X1_MATE_OFFSET[vtype]
            mate_x = vx + int(dx)
            mate_y = vy + int(dy)
            if mate_x < 0 or mate_x >= w or mate_y < 0 or mate_y >= h:
                raise ValueError(
                    f"object {obj_name!r} has invalid 2x1 slope pair at local "
                    f"(x={vx}, y={vy}, type={vtype}): expected mate type {mate_type} "
                    f"at (x={mate_x}, y={mate_y}), but the mate is out of bounds"
                )
            found = int(grid[mate_y, mate_x])
            if found != mate_type:
                raise ValueError(
                    f"object {obj_name!r} has invalid 2x1 slope pair at local "
                    f"(x={vx}, y={vy}, type={vtype}): expected mate type {mate_type} "
                    f"at (x={mate_x}, y={mate_y}), found {found}"
                )


def _validate_slope_3x1_triplets(grid: np.ndarray, *, obj_name: str) -> None:
    if not np.any(np.isin(grid, _SLOPE_3X1_TYPE_ARRAY)):
        return
    h, w = grid.shape
    for vy in range(h):
        for vx in range(w):
            vtype = int(grid[vy, vx])
            if vtype not in C.SLOPE_3X1_TYPES:
                continue
            neighbor_rules = C.SLOPE_3X1_NEIGHBOR_RULES[vtype]
            for dx, dy, expected_type in neighbor_rules:
                mate_x = vx + int(dx)
                mate_y = vy + int(dy)
                if mate_x < 0 or mate_x >= w or mate_y < 0 or mate_y >= h:
                    raise ValueError(
                        f"object {obj_name!r} has invalid 3x1 slope triplet at local "
                        f"(x={vx}, y={vy}, type={vtype}): expected segment type {expected_type} "
                        f"at (x={mate_x}, y={mate_y}), but the segment is out of bounds"
                    )
                found = int(grid[mate_y, mate_x])
                if found != int(expected_type):
                    raise ValueError(
                        f"object {obj_name!r} has invalid 3x1 slope triplet at local "
                        f"(x={vx}, y={vy}, type={vtype}): expected segment type {expected_type} "
                        f"at (x={mate_x}, y={mate_y}), found {found}"
                    )


def _classify_object(obj: WorldObject, robot_name: str) -> bool:
    """Return True for dynamic objects, False for static-only terrain."""
    occupied = obj.grid[obj.grid != C.EMPTY]
    has_slope = occupied.size > 0 and bool(np.any(np.isin(occupied, _SLOPE_TYPE_ARRAY)))
    if _is_robot_object(obj, robot_name):
        if has_slope:
            raise ValueError(f"robot object {obj.name!r} cannot contain slope cells")
        return True
    if has_slope and np.any(~np.isin(occupied, _STATIC_TERRAIN_ARRAY)):
        raise ValueError(
            f"object {obj.name!r} mixes slope cells with dynamic materials; "
            "slope objects may only contain FIXED and slope cells"
        )
    return not (occupied.size > 0 and np.all(np.isin(occupied, _STATIC_TERRAIN_ARRAY)))


def _boxel_surface_edge_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    n_boxels = h * w
    surface = np.zeros((n_boxels, 4), dtype=bool)
    offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    for vy in range(h):
        for vx in range(w):
            idx = vy * w + vx
            if grid[vy, vx] == C.EMPTY:
                continue
            for e, (dy, dx) in enumerate(offsets):
                ny = vy + dy
                nx = vx + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    surface[idx, e] = True
                else:
                    surface[idx, e] = grid[ny, nx] == C.EMPTY
    return surface


def _grid_cell_corner_indices(h: int, w: int) -> np.ndarray:
    w1 = w + 1
    n_boxels = h * w
    corners = np.zeros((n_boxels, 4), dtype=np.int32)
    for vy in range(h):
        for vx in range(w):
            idx = vy * w + vx
            bl = vy * w1 + vx
            br = vy * w1 + (vx + 1)
            tr = (vy + 1) * w1 + (vx + 1)
            tl = (vy + 1) * w1 + vx
            corners[idx] = [bl, br, tr, tl]
    return corners


def _build_per_axis_info(grid: np.ndarray, n_springs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = grid.shape
    n_cells = h * w
    vert_offset = (h + 1) * w

    h_pairs = np.zeros((n_cells, 2), dtype=np.int32)
    v_pairs = np.zeros((n_cells, 2), dtype=np.int32)
    spring_act_count = np.zeros(n_springs, dtype=np.float32)
    actuated_mask = np.zeros(n_springs, dtype=bool)

    for vy in range(h):
        for vx in range(w):
            idx = vy * w + vx
            vtype = int(grid[vy, vx])
            h_enabled = vtype in (C.H_ACT, C.CONTRACTILE)
            v_enabled = vtype in (C.V_ACT, C.CONTRACTILE)
            if h_enabled:
                bot = vy * w + vx
                top = (vy + 1) * w + vx
                h_pairs[idx] = [bot, top]
                spring_act_count[bot] += 1.0
                spring_act_count[top] += 1.0
                actuated_mask[bot] = True
                actuated_mask[top] = True
            if v_enabled:
                left = vert_offset + vy * (w + 1) + vx
                right = vert_offset + vy * (w + 1) + (vx + 1)
                v_pairs[idx] = [left, right]
                spring_act_count[left] += 1.0
                spring_act_count[right] += 1.0
                actuated_mask[left] = True
                actuated_mask[right] = True

    return h_pairs, v_pairs, spring_act_count, actuated_mask


def _compile_object_template(
    obj: WorldObject,
    *,
    object_id: int,
    robot_name: str,
    spring_stiffness_scale: float = 1.0,
    allow_experimental_slopes: bool = False,
) -> ObjectTemplate:
    grid = np.asarray(obj.grid, dtype=np.int32)
    if _contains_slope(grid) and not allow_experimental_slopes:
        raise ValueError(
            "slope cells are unsupported in the stable build path; "
            "pass allow_experimental_slopes=True to compile experimental slope terrain"
        )
    _validate_slope_2x1_pairing(grid, obj_name=obj.name)
    _validate_slope_3x1_triplets(grid, obj_name=obj.name)
    is_dynamic = _classify_object(obj, robot_name)
    if is_dynamic:
        return _compile_dynamic_object_template(
            obj,
            object_id=object_id,
            robot_name=robot_name,
            spring_stiffness_scale=spring_stiffness_scale,
        )
    return _compile_static_object_template(
        obj,
        object_id=object_id,
        robot_name=robot_name,
    )


def _compile_dynamic_object_template(
    obj: WorldObject,
    *,
    object_id: int,
    robot_name: str,
    spring_stiffness_scale: float = 1.0,
) -> ObjectTemplate:
    grid = np.array(obj.grid, copy=True)
    if _contains_slope(grid):
        raise ValueError(f"dynamic object {obj.name!r} cannot contain slope cells")
    h, w = grid.shape
    dense = _build_mass_spring_from_padded(
        grid,
        h,
        w,
        composited_grid=False,
        spring_stiffness_scale=spring_stiffness_scale,
    )
    corners = _grid_cell_corner_indices(h, w)
    edge_a = corners[:, [1, 2, 3, 0]]
    edge_b = corners
    surface = _boxel_surface_edge_mask(grid)
    is_robot = _is_robot_object(obj, robot_name)
    world_offset = np.array([obj.pos[0] * C.CELL_SIZE, obj.pos[1] * C.CELL_SIZE], dtype=np.float32)
    positions = dense["positions"].astype(np.float32) + world_offset
    boxel_world_vy = np.repeat(np.arange(h, dtype=np.int32), w) + int(obj.pos[1])
    boxel_world_vx = np.tile(np.arange(w, dtype=np.int32), h) + int(obj.pos[0])
    if is_robot:
        robot_cell_positions = {
            (vx, vy)
            for vy in range(h)
            for vx in range(w)
            if grid[vy, vx] != C.EMPTY
        }
        deformation = _build_deformation_info(
            dense["_springs"],
            grid,
            h,
            w,
            robot_cell_positions=robot_cell_positions,
        )
        robot_point_indices = dense["robot_point_indices"]
    else:
        deformation = DeformationInfo(
            voxel_horiz_springs=jnp.zeros((0, 2), dtype=jnp.int32),
            voxel_vert_springs=jnp.zeros((0, 2), dtype=jnp.int32),
            n_robot_voxels=0,
        )
        robot_point_indices = np.zeros((0,), dtype=np.int32)

    per_axis_h_pairs, per_axis_v_pairs, per_axis_spring_act_count, per_axis_actuated_mask = (
        _build_per_axis_info(grid, int(dense["n_springs"]))
    )
    terrain_point_indices = dense["terrain_point_indices"].astype(np.int32)
    terrain_sample_positions = (
        positions[terrain_point_indices].astype(np.float32, copy=False)
        if terrain_point_indices.size > 0
        else np.zeros((0, 2), dtype=np.float32)
    )

    return ObjectTemplate(
        name=obj.name,
        object_id=object_id,
        is_robot=is_robot,
        is_dynamic=True,
        origin_x=int(obj.pos[0]),
        origin_y=int(obj.pos[1]),
        grid=grid,
        positions=positions,
        masses=dense["masses"].astype(np.float32),
        fixed=dense["fixed"].astype(bool),
        point_mask=dense["point_mask"].astype(bool),
        spring_a_idx=dense["spring_a_idx"].astype(np.int32),
        spring_b_idx=dense["spring_b_idx"].astype(np.int32),
        spring_const=dense["spring_const"].astype(np.float32),
        spring_rest_length=dense["spring_rest_length"].astype(np.float32),
        spring_mask=dense["spring_mask"].astype(bool),
        rigid_spring_mask=dense["rigid_spring_mask"].astype(bool),
        boxel_corners=corners,
        boxel_edge_a=edge_a,
        boxel_edge_b=edge_b,
        boxel_world_vy=boxel_world_vy,
        boxel_world_vx=boxel_world_vx,
        boxel_types=grid.flatten().astype(np.int32),
        boxel_mask=(grid.flatten() != C.EMPTY),
        boxel_surface_edge_mask=surface,
        cell_vertices=corners,
        cell_vertex_count=np.full((h * w,), 4, dtype=np.int32),
        cell_edge_a=edge_a,
        cell_edge_b=edge_b,
        cell_world_vy=boxel_world_vy,
        cell_world_vx=boxel_world_vx,
        cell_types=grid.flatten().astype(np.int32),
        cell_mask=(grid.flatten() != C.EMPTY),
        cell_surface_edge_mask=surface,
        robot_point_indices=robot_point_indices.astype(np.int32),
        terrain_point_indices=terrain_point_indices,
        terrain_sample_positions=terrain_sample_positions,
        actuator_cell_spring_indices=dense["actuator_cell_spring_indices"].astype(np.int32),
        actuator_spring_act_count=dense["actuator_spring_act_count"].astype(np.float32),
        actuated_spring_mask=dense["actuated_spring_mask"].astype(bool),
        per_axis_h_pairs=per_axis_h_pairs,
        per_axis_v_pairs=per_axis_v_pairs,
        per_axis_spring_act_count=per_axis_spring_act_count,
        per_axis_actuated_mask=per_axis_actuated_mask,
        deformation_horiz_springs=np.array(deformation.voxel_horiz_springs, dtype=np.int32),
        deformation_vert_springs=np.array(deformation.voxel_vert_springs, dtype=np.int32),
        n_robot_voxels=int(deformation.n_robot_voxels),
        point_count=int(dense["n_points"]),
        spring_count=int(dense["n_springs"]),
    )


def _compile_static_object_template(
    obj: WorldObject,
    *,
    object_id: int,
    robot_name: str,
) -> ObjectTemplate:
    del robot_name
    grid = np.array(obj.grid, copy=True)
    geometry = build_sparse_cell_geometry(grid_active_cells(grid, obj.pos[0], obj.pos[1]))
    point_positions = geometry.point_positions.astype(np.float32, copy=False)
    return ObjectTemplate(
        name=obj.name,
        object_id=object_id,
        is_robot=False,
        is_dynamic=False,
        origin_x=int(obj.pos[0]),
        origin_y=int(obj.pos[1]),
        grid=grid,
        positions=point_positions,
        masses=np.zeros((0, 2), dtype=np.float32),
        fixed=np.zeros((0, 2), dtype=bool),
        point_mask=np.zeros((0,), dtype=bool),
        spring_a_idx=np.zeros((0,), dtype=np.int32),
        spring_b_idx=np.zeros((0,), dtype=np.int32),
        spring_const=np.zeros((0,), dtype=np.float32),
        spring_rest_length=np.zeros((0,), dtype=np.float32),
        spring_mask=np.zeros((0,), dtype=bool),
        rigid_spring_mask=np.zeros((0,), dtype=bool),
        boxel_corners=np.zeros((0, 4), dtype=np.int32),
        boxel_edge_a=np.zeros((0, 4), dtype=np.int32),
        boxel_edge_b=np.zeros((0, 4), dtype=np.int32),
        boxel_world_vy=np.zeros((0,), dtype=np.int32),
        boxel_world_vx=np.zeros((0,), dtype=np.int32),
        boxel_types=np.zeros((0,), dtype=np.int32),
        boxel_mask=np.zeros((0,), dtype=bool),
        boxel_surface_edge_mask=np.zeros((0, 4), dtype=bool),
        cell_vertices=geometry.cell_vertices.astype(np.int32, copy=False),
        cell_vertex_count=geometry.cell_vertex_count.astype(np.int32, copy=False),
        cell_edge_a=geometry.cell_edge_a.astype(np.int32, copy=False),
        cell_edge_b=geometry.cell_edge_b.astype(np.int32, copy=False),
        cell_world_vy=geometry.cell_world_vy.astype(np.int32, copy=False),
        cell_world_vx=geometry.cell_world_vx.astype(np.int32, copy=False),
        cell_types=geometry.cell_types.astype(np.int32, copy=False),
        cell_mask=geometry.cell_mask.astype(bool, copy=False),
        cell_surface_edge_mask=geometry.surface_edge_mask.astype(bool, copy=False),
        robot_point_indices=np.zeros((0,), dtype=np.int32),
        terrain_point_indices=np.zeros((0,), dtype=np.int32),
        terrain_sample_positions=geometry.terrain_sample_positions.astype(np.float32, copy=False),
        actuator_cell_spring_indices=np.zeros((0, 2), dtype=np.int32),
        actuator_spring_act_count=np.zeros((0,), dtype=np.float32),
        actuated_spring_mask=np.zeros((0,), dtype=bool),
        per_axis_h_pairs=np.zeros((0, 2), dtype=np.int32),
        per_axis_v_pairs=np.zeros((0, 2), dtype=np.int32),
        per_axis_spring_act_count=np.zeros((0,), dtype=np.float32),
        per_axis_actuated_mask=np.zeros((0,), dtype=bool),
        deformation_horiz_springs=np.zeros((0, 2), dtype=np.int32),
        deformation_vert_springs=np.zeros((0, 2), dtype=np.int32),
        n_robot_voxels=0,
        point_count=0,
        spring_count=0,
    )


def _mirror_object(obj: WorldObject, world_width: int) -> WorldObject:
    mirrored = obj.copy()
    obj_w, _ = obj.grid_size
    flipped_grid = np.fliplr(obj.grid).copy()
    mirrored.grid = flipped_grid.copy()
    for source_type, mirrored_type in C.SLOPE_MIRROR_X_MAP.items():
        mirrored.grid[flipped_grid == source_type] = mirrored_type
    mirrored.pos = (int(world_width - (obj.pos[0] + obj_w)), int(obj.pos[1]))
    mirrored.neighbors = {
        (obj_w - 1 - x, y): [(obj_w - 1 - nx, ny) for (nx, ny) in neighs]
        for (x, y), neighs in obj.neighbors.items()
    }
    return mirrored


def compile_world_template(
    world: EvoWorld,
    robot_name: str = "robot",
    spring_stiffness_scale: float = 1.0,
    *,
    allow_experimental_slopes: bool = False,
) -> WorldTemplate:
    """Compile a world into a reusable template.

    Slope terrain is retained as experimental geometry-only implementation
    code, but is not part of the stable public build contract. Pass
    ``allow_experimental_slopes`` only for internal slope investigations and
    legacy fixtures.
    """
    if spring_stiffness_scale <= 0.0:
        raise ValueError(
            "spring_stiffness_scale must be > 0; "
            f"got {spring_stiffness_scale}."
        )
    ordered = _ordered_objects(world, robot_name)
    templates = tuple(
        _compile_object_template(
            obj,
            object_id=i,
            robot_name=robot_name,
            spring_stiffness_scale=spring_stiffness_scale,
            allow_experimental_slopes=allow_experimental_slopes,
        )
        for i, obj in enumerate(ordered)
    )
    grid_w, grid_h = world.grid_size
    return WorldTemplate(
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        robot_name=robot_name,
        object_templates=templates,
        robot_template_index=0,
    )


def compile_world_templates(
    world: EvoWorld,
    robot_name: str = "robot",
    mirror_mode: str = "none",
    spring_stiffness_scale: float = 1.0,
    *,
    allow_experimental_slopes: bool = False,
) -> WorldTemplateSet:
    if mirror_mode not in ("none", "paired"):
        raise ValueError(f"Unsupported mirror_mode={mirror_mode!r}")
    primary = compile_world_template(
        world,
        robot_name=robot_name,
        spring_stiffness_scale=spring_stiffness_scale,
        allow_experimental_slopes=allow_experimental_slopes,
    )
    mirror = None
    if mirror_mode == "paired":
        mirrored_world = EvoWorld()
        world_width, _ = world.grid_size
        mirrored_objects = [
            _mirror_object(obj, world_width)
            for obj in _ordered_objects(world, robot_name)
        ]
        for obj in sorted(
            mirrored_objects,
            key=lambda candidate: (_is_robot_object(candidate, robot_name), candidate.name),
        ):
            mirrored_world.add_object(obj)
        mirror = compile_world_template(
            mirrored_world,
            robot_name=robot_name,
            spring_stiffness_scale=spring_stiffness_scale,
            allow_experimental_slopes=allow_experimental_slopes,
        )
    return WorldTemplateSet(primary=primary, mirror=mirror, mirror_mode=mirror_mode)


def _fixed_sample_positions(template: ObjectTemplate) -> np.ndarray:
    if template.terrain_sample_positions.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return template.terrain_sample_positions


def _unique_positions(parts: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [p for p in parts if p.size > 0]
    if not arrays:
        return np.zeros((0, 2), dtype=np.float32)
    merged = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
    return np.unique(merged, axis=0)


def _concat_or_empty(
    parts: Iterable[np.ndarray],
    *,
    shape: tuple[int, ...],
    dtype: np.dtype | type[np.generic],
) -> np.ndarray:
    arrays = [np.asarray(part) for part in parts]
    if not arrays:
        return np.zeros(shape, dtype=dtype)
    return np.concatenate(arrays, axis=0).astype(dtype, copy=False)


def _robot_override_grid(template: ObjectTemplate, robot_override: np.ndarray | jnp.ndarray | None) -> np.ndarray:
    if robot_override is None:
        return np.array(template.grid, copy=True)
    override_np = np.asarray(robot_override, dtype=np.int32)
    if override_np.shape != tuple(template.grid.shape):
        raise ValueError(
            f"robot_override shape {override_np.shape} does not match robot frame {template.grid.shape}"
        )
    if np.any(np.isin(override_np, _SLOPE_TYPE_ARRAY)):
        raise ValueError("robot_override cannot contain slope cells")
    return np.flipud(override_np).astype(np.int32)


def _live_template(
    template: ObjectTemplate,
    robot_override: np.ndarray | jnp.ndarray | None,
    robot_name: str,
    spring_stiffness_scale: float = 1.0,
) -> ObjectTemplate:
    if not template.is_robot:
        return template
    if robot_override is None:
        return template
    obj = WorldObject()
    obj.name = template.name
    obj.pos = (template.origin_x, template.origin_y)
    obj.grid = _robot_override_grid(template, robot_override)
    obj.neighbors = {
        (x, y): list(neighs)
        for (x, y), neighs in {
            (vx, vy): [] for vy in range(obj.grid.shape[0]) for vx in range(obj.grid.shape[1]) if obj.grid[vy, vx] != C.EMPTY
        }.items()
    }
    return _compile_object_template(
        obj,
        object_id=template.object_id,
        robot_name=robot_name,
        spring_stiffness_scale=spring_stiffness_scale,
    )


def _make_collision_data(
    *,
    boxel_corners: np.ndarray,
    boxel_edge_a: np.ndarray,
    boxel_edge_b: np.ndarray,
    boxel_mask: np.ndarray,
    surface_edge_mask: np.ndarray,
    boxel_object_id: np.ndarray,
    boxel_is_robot: np.ndarray,
    boxel_world_vy: np.ndarray,
    boxel_world_vx: np.ndarray,
    max_triples: int = C.MAX_TRIPLES,
    max_self_triples: int = C.MAX_SELF_TRIPLES,
) -> CollisionData:
    n_boxels = int(boxel_mask.shape[0])
    if n_boxels == 0:
        zeros_triples = np.zeros((max_triples,), dtype=np.int32)
        zeros_self = np.zeros((max_self_triples,), dtype=np.int32)
        false_triples = np.zeros((max_triples,), dtype=bool)
        false_self = np.zeros((max_self_triples,), dtype=bool)
        return CollisionData(
            boxel_corners=jnp.zeros((0, 4), dtype=jnp.int32),
            boxel_edge_a=jnp.zeros((0, 4), dtype=jnp.int32),
            boxel_edge_b=jnp.zeros((0, 4), dtype=jnp.int32),
            boxel_mask=jnp.zeros((0,), dtype=jnp.bool_),
            surface_edge_mask=jnp.zeros((0, 4), dtype=jnp.bool_),
            boxel_object_id=jnp.zeros((0,), dtype=jnp.int32),
            boxel_is_robot=jnp.zeros((0,), dtype=jnp.bool_),
            boxel_world_vy=jnp.zeros((0,), dtype=jnp.int32),
            boxel_world_vx=jnp.zeros((0,), dtype=jnp.int32),
            corner_edge_indices=jnp.array([[0, 3], [0, 1], [1, 2], [2, 3]], dtype=jnp.int32),
            n_real_points=0,
            triple_i=jnp.array(zeros_triples, dtype=jnp.int32),
            triple_j=jnp.array(zeros_triples, dtype=jnp.int32),
            triple_k=jnp.array(zeros_triples, dtype=jnp.int32),
            triple_active=jnp.array(false_triples, dtype=jnp.bool_),
            triple_pair_slot=jnp.array(zeros_triples, dtype=jnp.int32),
            self_triple_i=jnp.array(zeros_self, dtype=jnp.int32),
            self_triple_j=jnp.array(zeros_self, dtype=jnp.int32),
            self_triple_k=jnp.array(zeros_self, dtype=jnp.int32),
            self_triple_active=jnp.array(false_self, dtype=jnp.bool_),
            candidate_pair_i=jnp.array(zeros_triples, dtype=jnp.int32),
            candidate_pair_j=jnp.array(zeros_triples, dtype=jnp.int32),
            candidate_pair_active=jnp.array(false_triples, dtype=jnp.bool_),
            n_triples=0,
            n_self_triples=0,
            n_candidate_pairs=0,
            n_robot_surface_boxels=0,
        )

    point_is_robot = np.zeros(int(boxel_corners.max()) + 1, dtype=bool)
    for idx in range(n_boxels):
        if not boxel_mask[idx] or not boxel_is_robot[idx]:
            continue
        point_is_robot[boxel_corners[idx]] = True

    corner_sets = [set(boxel_corners[i]) for i in range(n_boxels)]
    triples_i: list[int] = []
    triples_j: list[int] = []
    triples_k: list[int] = []
    self_i: list[int] = []
    self_j: list[int] = []
    self_k: list[int] = []
    candidate_pairs: list[tuple[int, int]] = []

    active_indices = np.where(boxel_mask)[0]
    for i in active_indices:
        for j in active_indices:
            if i == j:
                continue
            if boxel_object_id[i] == boxel_object_id[j]:
                dy = abs(int(boxel_world_vy[i]) - int(boxel_world_vy[j]))
                dx = abs(int(boxel_world_vx[i]) - int(boxel_world_vx[j]))
                if dy <= 1 and dx <= 1:
                    continue
            any_pair = False
            for k in range(4):
                pidx = int(boxel_corners[i, k])
                if not point_is_robot[pidx]:
                    continue
                if pidx in corner_sets[j]:
                    continue
                triples_i.append(int(i))
                triples_j.append(int(j))
                triples_k.append(int(k))
                any_pair = True
                if boxel_is_robot[i] and boxel_is_robot[j]:
                    self_i.append(int(i))
                    self_j.append(int(j))
                    self_k.append(int(k))
            if any_pair:
                candidate_pairs.append((int(i), int(j)))

    if len(triples_i) > max_triples:
        raise ValueError(f"Dynamic collision triple count {len(triples_i)} exceeds max_triples={max_triples}")
    if len(self_i) > max_self_triples:
        raise ValueError(f"Self collision triple count {len(self_i)} exceeds max_self_triples={max_self_triples}")

    unique_pairs = np.array(sorted(set(candidate_pairs)), dtype=np.int32) if candidate_pairs else np.zeros((0, 2), dtype=np.int32)
    pair_slot_lookup = {tuple(pair): slot for slot, pair in enumerate(unique_pairs.tolist())}

    triple_i_arr = np.zeros((max_triples,), dtype=np.int32)
    triple_j_arr = np.zeros((max_triples,), dtype=np.int32)
    triple_k_arr = np.zeros((max_triples,), dtype=np.int32)
    triple_active = np.zeros((max_triples,), dtype=bool)
    triple_pair_slot = np.zeros((max_triples,), dtype=np.int32)
    for idx, (i, j, k) in enumerate(zip(triples_i, triples_j, triples_k, strict=False)):
        triple_i_arr[idx] = i
        triple_j_arr[idx] = j
        triple_k_arr[idx] = k
        triple_active[idx] = True
        triple_pair_slot[idx] = pair_slot_lookup[(i, j)]

    self_i_arr = np.zeros((max_self_triples,), dtype=np.int32)
    self_j_arr = np.zeros((max_self_triples,), dtype=np.int32)
    self_k_arr = np.zeros((max_self_triples,), dtype=np.int32)
    self_active = np.zeros((max_self_triples,), dtype=bool)
    for idx, (i, j, k) in enumerate(zip(self_i, self_j, self_k, strict=False)):
        self_i_arr[idx] = i
        self_j_arr[idx] = j
        self_k_arr[idx] = k
        self_active[idx] = True

    candidate_pair_i = np.zeros((max_triples,), dtype=np.int32)
    candidate_pair_j = np.zeros((max_triples,), dtype=np.int32)
    candidate_pair_active = np.zeros((max_triples,), dtype=bool)
    n_pairs = unique_pairs.shape[0]
    if n_pairs > max_triples:
        raise ValueError(f"Dynamic candidate pair count {n_pairs} exceeds max_triples={max_triples}")
    if n_pairs > 0:
        candidate_pair_i[:n_pairs] = unique_pairs[:, 0]
        candidate_pair_j[:n_pairs] = unique_pairs[:, 1]
        candidate_pair_active[:n_pairs] = True

    return CollisionData(
        boxel_corners=jnp.array(boxel_corners, dtype=jnp.int32),
        boxel_edge_a=jnp.array(boxel_edge_a, dtype=jnp.int32),
        boxel_edge_b=jnp.array(boxel_edge_b, dtype=jnp.int32),
        boxel_mask=jnp.array(boxel_mask, dtype=jnp.bool_),
        surface_edge_mask=jnp.array(surface_edge_mask, dtype=jnp.bool_),
        boxel_object_id=jnp.array(boxel_object_id, dtype=jnp.int32),
        boxel_is_robot=jnp.array(boxel_is_robot, dtype=jnp.bool_),
        boxel_world_vy=jnp.array(boxel_world_vy, dtype=jnp.int32),
        boxel_world_vx=jnp.array(boxel_world_vx, dtype=jnp.int32),
        corner_edge_indices=jnp.array([[0, 3], [0, 1], [1, 2], [2, 3]], dtype=jnp.int32),
        n_real_points=int(boxel_corners.max()) + 1,
        triple_i=jnp.array(triple_i_arr, dtype=jnp.int32),
        triple_j=jnp.array(triple_j_arr, dtype=jnp.int32),
        triple_k=jnp.array(triple_k_arr, dtype=jnp.int32),
        triple_active=jnp.array(triple_active, dtype=jnp.bool_),
        triple_pair_slot=jnp.array(triple_pair_slot, dtype=jnp.int32),
        self_triple_i=jnp.array(self_i_arr, dtype=jnp.int32),
        self_triple_j=jnp.array(self_j_arr, dtype=jnp.int32),
        self_triple_k=jnp.array(self_k_arr, dtype=jnp.int32),
        self_triple_active=jnp.array(self_active, dtype=jnp.bool_),
        candidate_pair_i=jnp.array(candidate_pair_i, dtype=jnp.int32),
        candidate_pair_j=jnp.array(candidate_pair_j, dtype=jnp.int32),
        candidate_pair_active=jnp.array(candidate_pair_active, dtype=jnp.bool_),
        n_triples=len(triples_i),
        n_self_triples=len(self_i),
        n_candidate_pairs=n_pairs,
        n_robot_surface_boxels=int(
            np.sum(boxel_mask & boxel_is_robot & np.any(surface_edge_mask, axis=1))
        ),
    )


def _static_cell_aabbs(
    point_positions: np.ndarray,
    cell_vertices: np.ndarray,
    cell_vertex_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if cell_vertices.shape[0] == 0:
        zeros = np.zeros((0, 2), dtype=np.float32)
        return zeros, zeros

    cell_points = point_positions[cell_vertices]
    valid_vertex = (
        np.arange(cell_vertices.shape[1], dtype=np.int32)[None, :]
        < cell_vertex_count[:, None]
    )
    pos_inf = np.full_like(cell_points, np.inf, dtype=np.float32)
    neg_inf = np.full_like(cell_points, -np.inf, dtype=np.float32)
    cell_aabb_min = np.min(
        np.where(valid_vertex[:, :, None], cell_points, pos_inf),
        axis=1,
    ).astype(np.float32, copy=False)
    cell_aabb_max = np.max(
        np.where(valid_vertex[:, :, None], cell_points, neg_inf),
        axis=1,
    ).astype(np.float32, copy=False)
    return cell_aabb_min, cell_aabb_max


def _static_cell_column_index(
    cell_aabb_min: np.ndarray,
    cell_aabb_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if cell_aabb_min.shape[0] == 0:
        return (
            np.full((0, 0), -1, dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )

    min_columns = np.floor(cell_aabb_min[:, 0] / C.CELL_SIZE + 1e-6).astype(np.int32, copy=False)
    max_columns = np.floor((cell_aabb_max[:, 0] - 1e-6) / C.CELL_SIZE).astype(np.int32, copy=False)
    if np.any(max_columns < min_columns):
        raise ValueError("Static collider cell columns are malformed")

    grid_w = int(max(np.max(max_columns) + 1, 1))
    clipped_columns = np.clip(min_columns, 0, grid_w - 1)
    counts = np.bincount(clipped_columns, minlength=grid_w).astype(np.int32, copy=False)
    max_cells_per_column = int(np.max(counts)) if counts.size > 0 else 0
    column_cell_indices = np.full((grid_w, max_cells_per_column), -1, dtype=np.int32)
    write_offsets = np.zeros((grid_w,), dtype=np.int32)

    for cell_idx, column in enumerate(clipped_columns.tolist()):
        slot = int(write_offsets[column])
        column_cell_indices[column, slot] = int(cell_idx)
        write_offsets[column] = slot + 1

    return column_cell_indices, counts


def make_static_collider_data(
    *,
    point_positions: np.ndarray,
    cell_vertices: np.ndarray,
    cell_vertex_count: np.ndarray,
    cell_edge_a: np.ndarray,
    cell_edge_b: np.ndarray,
    surface_edge_mask: np.ndarray,
    cell_types: np.ndarray,
) -> StaticColliderData:
    cell_aabb_min, cell_aabb_max = _static_cell_aabbs(
        np.asarray(point_positions, dtype=np.float32),
        np.asarray(cell_vertices, dtype=np.int32),
        np.asarray(cell_vertex_count, dtype=np.int32),
    )
    column_cell_indices, column_cell_count = _static_cell_column_index(
        cell_aabb_min,
        cell_aabb_max,
    )
    return StaticColliderData(
        point_positions=jnp.array(point_positions, dtype=jnp.float32),
        cell_vertices=jnp.array(cell_vertices, dtype=jnp.int32),
        cell_vertex_count=jnp.array(cell_vertex_count, dtype=jnp.int32),
        cell_edge_a=jnp.array(cell_edge_a, dtype=jnp.int32),
        cell_edge_b=jnp.array(cell_edge_b, dtype=jnp.int32),
        surface_edge_mask=jnp.array(surface_edge_mask, dtype=jnp.bool_),
        cell_types=jnp.array(cell_types, dtype=jnp.int32),
        cell_aabb_min=jnp.array(cell_aabb_min, dtype=jnp.float32),
        cell_aabb_max=jnp.array(cell_aabb_max, dtype=jnp.float32),
        column_cell_indices=jnp.array(column_cell_indices, dtype=jnp.int32),
        column_cell_count=jnp.array(column_cell_count, dtype=jnp.int32),
    )


def instantiate_world(
    template: WorldTemplate,
    robot_override: np.ndarray | jnp.ndarray | None = None,
    *,
    max_triples: int = C.MAX_TRIPLES,
    max_self_triples: int = C.MAX_SELF_TRIPLES,
    spring_stiffness_scale: float = 1.0,
) -> BuiltWorld:
    if spring_stiffness_scale <= 0.0:
        raise ValueError(
            "spring_stiffness_scale must be > 0; "
            f"got {spring_stiffness_scale}."
        )
    dynamic_positions: list[np.ndarray] = []
    dynamic_masses: list[np.ndarray] = []
    dynamic_fixed: list[np.ndarray] = []
    dynamic_point_masks: list[np.ndarray] = []
    dynamic_spring_a: list[np.ndarray] = []
    dynamic_spring_b: list[np.ndarray] = []
    dynamic_spring_const: list[np.ndarray] = []
    dynamic_spring_rest: list[np.ndarray] = []
    dynamic_spring_mask: list[np.ndarray] = []
    dynamic_rigid_mask: list[np.ndarray] = []
    dynamic_boxel_corners: list[np.ndarray] = []
    dynamic_boxel_edge_a: list[np.ndarray] = []
    dynamic_boxel_edge_b: list[np.ndarray] = []
    dynamic_boxel_mask: list[np.ndarray] = []
    dynamic_boxel_surface: list[np.ndarray] = []
    dynamic_boxel_object_id: list[np.ndarray] = []
    dynamic_boxel_is_robot: list[np.ndarray] = []
    dynamic_boxel_world_vy: list[np.ndarray] = []
    dynamic_boxel_world_vx: list[np.ndarray] = []
    dynamic_robot_points: list[np.ndarray] = []
    dynamic_terrain_points: list[np.ndarray] = []
    render_cell_vertices_parts: list[np.ndarray] = []
    render_cell_vertex_count_parts: list[np.ndarray] = []
    render_cell_types_parts: list[np.ndarray] = []
    render_cell_mask_parts: list[np.ndarray] = []
    render_surface_edge_parts: list[np.ndarray] = []
    render_cell_is_robot_parts: list[np.ndarray] = []
    render_cell_is_dynamic_parts: list[np.ndarray] = []
    render_cell_object_id_parts: list[np.ndarray] = []
    render_cell_h_pairs_parts: list[np.ndarray] = []
    render_cell_v_pairs_parts: list[np.ndarray] = []
    static_point_positions_parts: list[np.ndarray] = []
    static_cell_vertices_parts: list[np.ndarray] = []
    static_cell_vertex_count_parts: list[np.ndarray] = []
    static_cell_edge_a_parts: list[np.ndarray] = []
    static_cell_edge_b_parts: list[np.ndarray] = []
    static_surface_edge_parts: list[np.ndarray] = []
    static_cell_types_parts: list[np.ndarray] = []
    static_cell_object_id_parts: list[np.ndarray] = []
    fixed_sample_parts: list[np.ndarray] = []

    live_templates: list[ObjectTemplate] = []
    for obj_template in template.object_templates:
        live_templates.append(
            _live_template(
                obj_template,
                robot_override,
                template.robot_name,
                spring_stiffness_scale=spring_stiffness_scale,
            )
        )

    robot_actuator_pairs = np.zeros((0, 2), dtype=np.int32)
    robot_actuator_count = np.zeros((0,), dtype=np.float32)
    robot_actuated_mask = np.zeros((0,), dtype=bool)
    robot_per_axis_h = np.zeros((0, 2), dtype=np.int32)
    robot_per_axis_v = np.zeros((0, 2), dtype=np.int32)
    robot_per_axis_count = np.zeros((0,), dtype=np.float32)
    robot_per_axis_mask = np.zeros((0,), dtype=bool)
    robot_cell_types = np.zeros((0,), dtype=np.int32)
    robot_cell_mask = np.zeros((0,), dtype=bool)
    robot_deform_h = np.zeros((0, 2), dtype=np.int32)
    robot_deform_v = np.zeros((0, 2), dtype=np.int32)
    robot_n_voxels = 0

    point_cursor = 0
    spring_cursor = 0
    static_point_cursor = 0
    for obj_template in live_templates:
        fixed_sample_parts.append(_fixed_sample_positions(obj_template))
        if obj_template.is_dynamic:
            dynamic_positions.append(obj_template.positions)
            dynamic_masses.append(obj_template.masses)
            dynamic_fixed.append(obj_template.fixed)
            dynamic_point_masks.append(obj_template.point_mask)
            dynamic_spring_a.append(obj_template.spring_a_idx + point_cursor)
            dynamic_spring_b.append(obj_template.spring_b_idx + point_cursor)
            dynamic_spring_const.append(obj_template.spring_const)
            dynamic_spring_rest.append(obj_template.spring_rest_length)
            dynamic_spring_mask.append(obj_template.spring_mask)
            dynamic_rigid_mask.append(obj_template.rigid_spring_mask)
            dynamic_boxel_corners.append(obj_template.boxel_corners + point_cursor)
            dynamic_boxel_edge_a.append(obj_template.boxel_edge_a + point_cursor)
            dynamic_boxel_edge_b.append(obj_template.boxel_edge_b + point_cursor)
            dynamic_boxel_mask.append(obj_template.boxel_mask)
            dynamic_boxel_surface.append(obj_template.boxel_surface_edge_mask)
            dynamic_boxel_object_id.append(
                np.full(obj_template.boxel_mask.shape, obj_template.object_id, dtype=np.int32)
            )
            dynamic_boxel_is_robot.append(
                np.full(obj_template.boxel_mask.shape, obj_template.is_robot, dtype=bool)
            )
            dynamic_boxel_world_vy.append(obj_template.boxel_world_vy)
            dynamic_boxel_world_vx.append(obj_template.boxel_world_vx)
            render_cell_vertices_parts.append(obj_template.cell_vertices + point_cursor)
            render_cell_vertex_count_parts.append(obj_template.cell_vertex_count)
            render_cell_types_parts.append(obj_template.cell_types)
            render_cell_mask_parts.append(obj_template.cell_mask)
            render_surface_edge_parts.append(obj_template.cell_surface_edge_mask)
            render_cell_is_robot_parts.append(
                np.full(obj_template.cell_mask.shape, obj_template.is_robot, dtype=bool)
            )
            render_cell_is_dynamic_parts.append(np.ones(obj_template.cell_mask.shape, dtype=bool))
            render_cell_object_id_parts.append(
                np.full(obj_template.cell_mask.shape, obj_template.object_id, dtype=np.int32)
            )
            n_obj_cells = int(obj_template.cell_mask.shape[0])
            if obj_template.is_robot:
                if (
                    obj_template.per_axis_h_pairs.shape[0] != n_obj_cells
                    or obj_template.per_axis_v_pairs.shape[0] != n_obj_cells
                ):
                    raise ValueError(
                        "robot per-axis spring mappings must match render cell count "
                        f"for object {obj_template.name!r}: "
                        f"h_pairs={obj_template.per_axis_h_pairs.shape[0]} "
                        f"v_pairs={obj_template.per_axis_v_pairs.shape[0]} "
                        f"cells={n_obj_cells}"
                    )
                local_types = np.asarray(obj_template.cell_types, dtype=np.int32)
                local_mask = np.asarray(obj_template.cell_mask, dtype=bool)
                local_h_pairs = np.full((n_obj_cells, 2), -1, dtype=np.int32)
                local_v_pairs = np.full((n_obj_cells, 2), -1, dtype=np.int32)
                h_enabled = local_mask & np.isin(local_types, _H_RENDER_ACTUATION_TYPE_ARRAY)
                v_enabled = local_mask & np.isin(local_types, _V_RENDER_ACTUATION_TYPE_ARRAY)
                local_h_pairs[h_enabled] = obj_template.per_axis_h_pairs[h_enabled] + spring_cursor
                local_v_pairs[v_enabled] = obj_template.per_axis_v_pairs[v_enabled] + spring_cursor
                render_cell_h_pairs_parts.append(local_h_pairs)
                render_cell_v_pairs_parts.append(local_v_pairs)
            else:
                render_cell_h_pairs_parts.append(np.full((n_obj_cells, 2), -1, dtype=np.int32))
                render_cell_v_pairs_parts.append(np.full((n_obj_cells, 2), -1, dtype=np.int32))
            if obj_template.is_robot:
                dynamic_robot_points.append(obj_template.robot_point_indices + point_cursor)
                robot_actuator_pairs = obj_template.actuator_cell_spring_indices + spring_cursor
                robot_actuator_count = obj_template.actuator_spring_act_count
                robot_actuated_mask = obj_template.actuated_spring_mask
                robot_per_axis_h = obj_template.per_axis_h_pairs + spring_cursor
                robot_per_axis_v = obj_template.per_axis_v_pairs + spring_cursor
                robot_per_axis_count = obj_template.per_axis_spring_act_count
                robot_per_axis_mask = obj_template.per_axis_actuated_mask
                robot_cell_types = np.asarray(obj_template.cell_types, dtype=np.int32)
                robot_cell_mask = np.asarray(obj_template.cell_mask, dtype=bool)
                robot_deform_h = obj_template.deformation_horiz_springs + spring_cursor
                robot_deform_v = obj_template.deformation_vert_springs + spring_cursor
                robot_n_voxels = obj_template.n_robot_voxels
            else:
                dynamic_terrain_points.append(np.flatnonzero(obj_template.point_mask).astype(np.int32) + point_cursor)
            point_cursor += obj_template.point_count
            spring_cursor += obj_template.spring_count
        else:
            active_cell_mask = np.asarray(obj_template.cell_mask, dtype=bool)
            if not np.any(active_cell_mask):
                continue
            active_cell_vertices_local = obj_template.cell_vertices[active_cell_mask]
            active_point_indices = np.unique(active_cell_vertices_local.reshape(-1)).astype(np.int32, copy=False)
            if active_point_indices.size == 0:
                continue
            static_positions = obj_template.positions[active_point_indices]
            local_to_static = {
                int(local): static_point_cursor + idx
                for idx, local in enumerate(active_point_indices.tolist())
            }
            static_point_positions_parts.append(static_positions)
            remapped_vertices = np.array(
                [[local_to_static[int(p)] for p in corners] for corners in active_cell_vertices_local],
                dtype=np.int32,
            )
            remapped_edge_a = np.array(
                [[local_to_static[int(p)] for p in edges] for edges in obj_template.cell_edge_a[active_cell_mask]],
                dtype=np.int32,
            )
            remapped_edge_b = np.array(
                [[local_to_static[int(p)] for p in edges] for edges in obj_template.cell_edge_b[active_cell_mask]],
                dtype=np.int32,
            )
            static_cell_vertices_parts.append(remapped_vertices)
            static_cell_vertex_count_parts.append(obj_template.cell_vertex_count[active_cell_mask])
            static_cell_edge_a_parts.append(remapped_edge_a)
            static_cell_edge_b_parts.append(remapped_edge_b)
            static_surface_edge_parts.append(obj_template.cell_surface_edge_mask[active_cell_mask])
            static_cell_types_parts.append(obj_template.cell_types[active_cell_mask])
            static_cell_object_id_parts.append(
                np.full((remapped_vertices.shape[0],), obj_template.object_id, dtype=np.int32)
            )
            static_point_cursor += static_positions.shape[0]

    positions = _concat_or_empty(dynamic_positions, shape=(0, 2), dtype=np.float32)
    masses = _concat_or_empty(dynamic_masses, shape=(0, 2), dtype=np.float32)
    fixed = _concat_or_empty(dynamic_fixed, shape=(0, 2), dtype=bool)
    point_mask = _concat_or_empty(dynamic_point_masks, shape=(0,), dtype=bool)
    spring_a_idx = _concat_or_empty(dynamic_spring_a, shape=(0,), dtype=np.int32)
    spring_b_idx = _concat_or_empty(dynamic_spring_b, shape=(0,), dtype=np.int32)
    spring_const = _concat_or_empty(dynamic_spring_const, shape=(0,), dtype=np.float32)
    spring_rest = _concat_or_empty(dynamic_spring_rest, shape=(0,), dtype=np.float32)
    spring_mask = _concat_or_empty(dynamic_spring_mask, shape=(0,), dtype=bool)
    rigid_mask = _concat_or_empty(dynamic_rigid_mask, shape=(0,), dtype=bool)
    robot_point_indices = (
        np.concatenate(dynamic_robot_points, axis=0).astype(np.int32, copy=False)
        if dynamic_robot_points
        else np.zeros((0,), dtype=np.int32)
    )
    dynamic_terrain_indices = (
        np.concatenate(dynamic_terrain_points, axis=0).astype(np.int32, copy=False)
        if dynamic_terrain_points
        else np.zeros((0,), dtype=np.int32)
    )

    sim_positions = jnp.array(positions, dtype=jnp.float32)
    zeros_pts = jnp.zeros_like(sim_positions)
    spring_rest_jax = jnp.array(spring_rest, dtype=jnp.float32)
    sim_state = SimState(
        positions=sim_positions,
        velocities=zeros_pts,
        positions_last=sim_positions,
        velocities_true=zeros_pts,
        masses=jnp.array(masses, dtype=jnp.float32),
        fixed=jnp.array(fixed, dtype=jnp.bool_),
        point_mask=jnp.array(point_mask, dtype=jnp.bool_),
        spring_rest_length=spring_rest_jax,
        spring_rest_length_goal=spring_rest_jax,
        spring_init_rest_length=spring_rest_jax,
        spring_const=jnp.array(spring_const, dtype=jnp.float32),
        spring_mask=jnp.array(spring_mask, dtype=jnp.bool_),
        rigid_spring_mask=jnp.array(rigid_mask, dtype=jnp.bool_),
        external_forces=zeros_pts,
        tangential_deformation=zeros_pts,
        friction_anchor=sim_positions,
        friction_anchor_active=jnp.zeros((sim_positions.shape[0],), dtype=jnp.bool_),
        friction_anchor_no_contact_count=jnp.zeros((sim_positions.shape[0],), dtype=jnp.int32),
        static_manifold_cell_idx=jnp.full((sim_positions.shape[0],), -1, dtype=jnp.int32),
        static_manifold_edge_idx=jnp.full((sim_positions.shape[0],), -1, dtype=jnp.int32),
        static_manifold_active=jnp.zeros((sim_positions.shape[0],), dtype=jnp.bool_),
        static_manifold_no_contact_count=jnp.zeros((sim_positions.shape[0],), dtype=jnp.int32),
        static_manifold_normal_force=jnp.zeros((sim_positions.shape[0],), dtype=jnp.float32),
        static_manifold_normal=jnp.zeros((sim_positions.shape[0], 2), dtype=jnp.float32),
    )
    topology = SpringTopology(
        a_idx=jnp.array(spring_a_idx, dtype=jnp.int32),
        b_idx=jnp.array(spring_b_idx, dtype=jnp.int32),
    )

    robot_spring_count = spring_rest.shape[0]
    global_act_count = np.zeros((robot_spring_count,), dtype=np.float32)
    global_actuated_mask = np.zeros((robot_spring_count,), dtype=bool)
    if robot_actuator_count.size:
        global_act_count[: robot_actuator_count.shape[0]] = robot_actuator_count
        global_actuated_mask[: robot_actuated_mask.shape[0]] = robot_actuated_mask
    actuator_info = ActuatorInfo(
        cell_spring_indices=jnp.array(robot_actuator_pairs, dtype=jnp.int32),
        spring_act_count=jnp.array(global_act_count, dtype=jnp.float32),
        actuated_spring_mask=jnp.array(global_actuated_mask, dtype=jnp.bool_),
    )

    global_per_axis_count = np.zeros((robot_spring_count,), dtype=np.float32)
    global_per_axis_mask = np.zeros((robot_spring_count,), dtype=bool)
    if robot_per_axis_count.size:
        global_per_axis_count[: robot_per_axis_count.shape[0]] = robot_per_axis_count
        global_per_axis_mask[: robot_per_axis_mask.shape[0]] = robot_per_axis_mask
    n_robot_cells = int(robot_per_axis_h.shape[0])
    h_compact_indices = np.zeros((n_robot_cells,), dtype=np.int32)
    v_compact_indices = np.zeros((n_robot_cells,), dtype=np.int32)
    h_compact_pairs = np.zeros((n_robot_cells, 2), dtype=np.int32)
    v_compact_pairs = np.zeros((n_robot_cells, 2), dtype=np.int32)
    h_compact_count = 0
    v_compact_count = 0
    if n_robot_cells > 0:
        h_enabled = robot_cell_mask & np.isin(robot_cell_types, _H_RENDER_ACTUATION_TYPE_ARRAY)
        v_enabled = robot_cell_mask & np.isin(robot_cell_types, _V_RENDER_ACTUATION_TYPE_ARRAY)
        h_src = np.flatnonzero(h_enabled).astype(np.int32, copy=False)
        v_src = np.flatnonzero(v_enabled).astype(np.int32, copy=False)
        h_compact_count = int(h_src.shape[0])
        v_compact_count = int(v_src.shape[0])
        if h_compact_count > 0:
            h_compact_indices[:h_compact_count] = h_src
            h_compact_pairs[:h_compact_count] = robot_per_axis_h[h_src]
        if v_compact_count > 0:
            v_compact_indices[:v_compact_count] = v_src
            v_compact_pairs[:v_compact_count] = robot_per_axis_v[v_src]
    per_axis_info = PerAxisActuatorInfo(
        h_spring_pairs=jnp.array(robot_per_axis_h, dtype=jnp.int32),
        v_spring_pairs=jnp.array(robot_per_axis_v, dtype=jnp.int32),
        h_compact_cell_indices=jnp.array(h_compact_indices, dtype=jnp.int32),
        v_compact_cell_indices=jnp.array(v_compact_indices, dtype=jnp.int32),
        h_compact_spring_pairs=jnp.array(h_compact_pairs, dtype=jnp.int32),
        v_compact_spring_pairs=jnp.array(v_compact_pairs, dtype=jnp.int32),
        h_compact_count=jnp.int32(h_compact_count),
        v_compact_count=jnp.int32(v_compact_count),
        actuated_spring_mask=jnp.array(global_per_axis_mask, dtype=jnp.bool_),
        spring_act_count=jnp.array(global_per_axis_count, dtype=jnp.float32),
    )

    deformation_info = DeformationInfo(
        voxel_horiz_springs=jnp.array(robot_deform_h, dtype=jnp.int32),
        voxel_vert_springs=jnp.array(robot_deform_v, dtype=jnp.int32),
        n_robot_voxels=int(robot_n_voxels),
    )

    dynamic_boxel_corners_arr = _concat_or_empty(dynamic_boxel_corners, shape=(0, 4), dtype=np.int32)
    dynamic_boxel_edge_a_arr = _concat_or_empty(dynamic_boxel_edge_a, shape=(0, 4), dtype=np.int32)
    dynamic_boxel_edge_b_arr = _concat_or_empty(dynamic_boxel_edge_b, shape=(0, 4), dtype=np.int32)
    dynamic_boxel_mask_arr = _concat_or_empty(dynamic_boxel_mask, shape=(0,), dtype=bool)
    dynamic_boxel_surface_arr = _concat_or_empty(dynamic_boxel_surface, shape=(0, 4), dtype=bool)
    dynamic_boxel_object_id_arr = _concat_or_empty(dynamic_boxel_object_id, shape=(0,), dtype=np.int32)
    dynamic_boxel_is_robot_arr = _concat_or_empty(dynamic_boxel_is_robot, shape=(0,), dtype=bool)
    dynamic_boxel_world_vy_arr = _concat_or_empty(dynamic_boxel_world_vy, shape=(0,), dtype=np.int32)
    dynamic_boxel_world_vx_arr = _concat_or_empty(dynamic_boxel_world_vx, shape=(0,), dtype=np.int32)

    dynamic_cd = _make_collision_data(
        boxel_corners=dynamic_boxel_corners_arr,
        boxel_edge_a=dynamic_boxel_edge_a_arr,
        boxel_edge_b=dynamic_boxel_edge_b_arr,
        boxel_mask=dynamic_boxel_mask_arr,
        surface_edge_mask=dynamic_boxel_surface_arr,
        boxel_object_id=dynamic_boxel_object_id_arr,
        boxel_is_robot=dynamic_boxel_is_robot_arr,
        boxel_world_vy=dynamic_boxel_world_vy_arr,
        boxel_world_vx=dynamic_boxel_world_vx_arr,
        max_triples=max_triples,
        max_self_triples=max_self_triples,
    )

    static_point_positions = _concat_or_empty(static_point_positions_parts, shape=(0, 2), dtype=np.float32)
    static_cell_vertices = _concat_or_empty(static_cell_vertices_parts, shape=(0, 4), dtype=np.int32)
    static_cell_vertex_count = _concat_or_empty(static_cell_vertex_count_parts, shape=(0,), dtype=np.int32)
    static_cell_edge_a = _concat_or_empty(static_cell_edge_a_parts, shape=(0, 4), dtype=np.int32)
    static_cell_edge_b = _concat_or_empty(static_cell_edge_b_parts, shape=(0, 4), dtype=np.int32)
    static_surface_edge_mask = _concat_or_empty(static_surface_edge_parts, shape=(0, 4), dtype=bool)
    static_cell_types = _concat_or_empty(static_cell_types_parts, shape=(0,), dtype=np.int32)
    static_collider_data = make_static_collider_data(
        point_positions=static_point_positions,
        cell_vertices=static_cell_vertices,
        cell_vertex_count=static_cell_vertex_count,
        cell_edge_a=static_cell_edge_a,
        cell_edge_b=static_cell_edge_b,
        surface_edge_mask=static_surface_edge_mask,
        cell_types=static_cell_types,
    )

    if static_cell_vertices.shape[0] > 0:
        render_cell_vertices_parts.append(static_cell_vertices + positions.shape[0])
        render_cell_vertex_count_parts.append(static_cell_vertex_count)
        render_cell_types_parts.append(static_cell_types)
        render_cell_mask_parts.append(np.ones((static_cell_vertices.shape[0],), dtype=bool))
        render_surface_edge_parts.append(static_surface_edge_mask)
        render_cell_is_robot_parts.append(np.zeros((static_cell_vertices.shape[0],), dtype=bool))
        render_cell_is_dynamic_parts.append(np.zeros((static_cell_vertices.shape[0],), dtype=bool))
        render_cell_object_id_parts.append(
            _concat_or_empty(static_cell_object_id_parts, shape=(0,), dtype=np.int32)
        )
        render_cell_h_pairs_parts.append(np.full((static_cell_vertices.shape[0], 2), -1, dtype=np.int32))
        render_cell_v_pairs_parts.append(np.full((static_cell_vertices.shape[0], 2), -1, dtype=np.int32))
    render_cell_vertices = _concat_or_empty(render_cell_vertices_parts, shape=(0, 4), dtype=np.int32)
    render_cell_vertex_count = _concat_or_empty(render_cell_vertex_count_parts, shape=(0,), dtype=np.int32)
    render_cell_types = _concat_or_empty(render_cell_types_parts, shape=(0,), dtype=np.int32)
    render_cell_mask = _concat_or_empty(render_cell_mask_parts, shape=(0,), dtype=bool)
    render_surface_edge_mask = _concat_or_empty(render_surface_edge_parts, shape=(0, 4), dtype=bool)
    render_cell_is_robot = _concat_or_empty(render_cell_is_robot_parts, shape=(0,), dtype=bool)
    render_cell_is_dynamic = _concat_or_empty(render_cell_is_dynamic_parts, shape=(0,), dtype=bool)
    render_cell_object_id = _concat_or_empty(render_cell_object_id_parts, shape=(0,), dtype=np.int32)
    render_cell_h_pairs = _concat_or_empty(render_cell_h_pairs_parts, shape=(0, 2), dtype=np.int32)
    render_cell_v_pairs = _concat_or_empty(render_cell_v_pairs_parts, shape=(0, 2), dtype=np.int32)
    if render_cell_h_pairs.shape != (render_cell_mask.shape[0], 2):
        raise ValueError(
            "render horizontal spring mapping shape mismatch: "
            f"expected {(render_cell_mask.shape[0], 2)} got {render_cell_h_pairs.shape}"
        )
    if render_cell_v_pairs.shape != (render_cell_mask.shape[0], 2):
        raise ValueError(
            "render vertical spring mapping shape mismatch: "
            f"expected {(render_cell_mask.shape[0], 2)} got {render_cell_v_pairs.shape}"
        )
    n_springs_total = int(spring_rest.shape[0])
    h_valid = render_cell_h_pairs >= 0
    v_valid = render_cell_v_pairs >= 0
    if np.any(h_valid & (render_cell_h_pairs >= n_springs_total)):
        raise ValueError("render horizontal spring mapping contains out-of-range spring indices")
    if np.any(v_valid & (render_cell_v_pairs >= n_springs_total)):
        raise ValueError("render vertical spring mapping contains out-of-range spring indices")
    actuator_cell_indices = np.where(
        render_cell_mask
        & render_cell_is_robot
        & np.isin(render_cell_types, _ACTUATOR_TYPE_ARRAY)
    )[0].astype(np.int32)
    if actuator_cell_indices.size > 0:
        act_types = render_cell_types[actuator_cell_indices]
        has_h_pairs = np.all(render_cell_h_pairs[actuator_cell_indices] >= 0, axis=1)
        has_v_pairs = np.all(render_cell_v_pairs[actuator_cell_indices] >= 0, axis=1)
        expected_h = np.isin(act_types, _H_RENDER_ACTUATION_TYPE_ARRAY)
        expected_v = np.isin(act_types, _V_RENDER_ACTUATION_TYPE_ARRAY)
        if np.any(expected_h != has_h_pairs):
            raise ValueError("render actuator horizontal spring mapping is inconsistent with actuator cell types")
        if np.any(expected_v != has_v_pairs):
            raise ValueError("render actuator vertical spring mapping is inconsistent with actuator cell types")
        if np.any(~(has_h_pairs | has_v_pairs)):
            raise ValueError("render actuator cells must map to at least one spring pair")
    render_info = RenderInfo(
        cell_vertices=render_cell_vertices,
        cell_vertex_count=render_cell_vertex_count,
        cell_types=render_cell_types,
        cell_mask=render_cell_mask,
        surface_edge_mask=render_surface_edge_mask,
        cell_is_robot=render_cell_is_robot,
        cell_is_dynamic=render_cell_is_dynamic,
        cell_object_id=render_cell_object_id,
        robot_point_indices=robot_point_indices.astype(np.int32, copy=False),
        grid_h=int(template.grid_h),
        grid_w=int(template.grid_w),
        actuator_cell_indices=actuator_cell_indices,
        cell_h_spring_pairs=render_cell_h_pairs,
        cell_v_spring_pairs=render_cell_v_pairs,
        spring_init_rest_length=spring_rest.astype(np.float32, copy=False),
        static_point_positions=static_point_positions,
    )

    robot_point_mask = np.zeros((positions.shape[0],), dtype=bool)
    robot_point_mask[robot_point_indices] = True
    return BuiltWorld(
        sim_state=sim_state,
        topology=topology,
        constants=default_physics_constants(),
        collision_data=dynamic_cd,
        static_collider_data=static_collider_data,
        actuator_info=actuator_info,
        per_axis_info=per_axis_info,
        deformation_info=deformation_info,
        robot_point_indices=jnp.array(robot_point_indices, dtype=jnp.int32),
        robot_point_mask=jnp.array(robot_point_mask, dtype=jnp.bool_),
        dynamic_terrain_point_indices=jnp.array(dynamic_terrain_indices, dtype=jnp.int32),
        fixed_terrain_sample_positions=jnp.array(_unique_positions(fixed_sample_parts), dtype=jnp.float32),
        render_info=render_info,
        grid_h=int(template.grid_h),
        grid_w=int(template.grid_w),
    )
