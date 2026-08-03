"""Shared cell geometry helpers for quads and slope polygons."""

from __future__ import annotations

from typing import Iterable, NamedTuple

import numpy as np

from . import constants as C

BL = 0
BR = 1
TR = 2
TL = 3
EDGE_SLOTS = 4
HALF = 0.5
THIRD = 1.0 / 3.0
TWO_THIRD = 2.0 / 3.0
GRID_KEY_SCALE = 6.0
CELL_TYPE_TO_VERTEX_OFFSETS = {
    C.RIGID: ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    C.SOFT: ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    C.H_ACT: ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    C.V_ACT: ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    C.FIXED: ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    C.CONTRACTILE: ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    C.SLOPE_UP_RIGHT: ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
    C.SLOPE_UP_LEFT: ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
    C.SLOPE_DOWN_RIGHT: ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    C.SLOPE_DOWN_LEFT: ((1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    C.SLOPE2_UP_RIGHT_LIGHT: ((0.0, 0.0), (1.0, 0.0), (1.0, HALF)),
    C.SLOPE2_UP_RIGHT_HEAVY: ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, HALF)),
    C.SLOPE2_UP_LEFT_HEAVY: ((0.0, 0.0), (1.0, 0.0), (1.0, HALF), (0.0, 1.0)),
    C.SLOPE2_UP_LEFT_LIGHT: ((0.0, 0.0), (1.0, 0.0), (0.0, HALF)),
    C.SLOPE2_DOWN_RIGHT_LIGHT: ((0.0, HALF), (1.0, 1.0), (0.0, 1.0)),
    C.SLOPE2_DOWN_RIGHT_HEAVY: ((0.0, 0.0), (1.0, HALF), (1.0, 1.0), (0.0, 1.0)),
    C.SLOPE2_DOWN_LEFT_HEAVY: ((0.0, HALF), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    C.SLOPE2_DOWN_LEFT_LIGHT: ((0.0, 1.0), (1.0, HALF), (1.0, 1.0)),
    C.SLOPE3_UP_RIGHT_LIGHT: ((0.0, 0.0), (1.0, 0.0), (1.0, THIRD)),
    C.SLOPE3_UP_RIGHT_MID: ((0.0, 0.0), (1.0, 0.0), (1.0, TWO_THIRD), (0.0, THIRD)),
    C.SLOPE3_UP_RIGHT_HEAVY: ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, TWO_THIRD)),
    C.SLOPE3_UP_LEFT_HEAVY: ((0.0, 0.0), (1.0, 0.0), (1.0, TWO_THIRD), (0.0, 1.0)),
    C.SLOPE3_UP_LEFT_MID: ((0.0, 0.0), (1.0, 0.0), (1.0, THIRD), (0.0, TWO_THIRD)),
    C.SLOPE3_UP_LEFT_LIGHT: ((0.0, 0.0), (1.0, 0.0), (0.0, THIRD)),
    C.SLOPE3_DOWN_RIGHT_HEAVY: ((0.0, 0.0), (1.0, THIRD), (1.0, 1.0), (0.0, 1.0)),
    C.SLOPE3_DOWN_RIGHT_MID: ((0.0, THIRD), (1.0, TWO_THIRD), (1.0, 1.0), (0.0, 1.0)),
    C.SLOPE3_DOWN_RIGHT_LIGHT: ((0.0, TWO_THIRD), (1.0, 1.0), (0.0, 1.0)),
    C.SLOPE3_DOWN_LEFT_LIGHT: ((0.0, 1.0), (1.0, TWO_THIRD), (1.0, 1.0)),
    C.SLOPE3_DOWN_LEFT_MID: ((0.0, TWO_THIRD), (1.0, THIRD), (1.0, 1.0), (0.0, 1.0)),
    C.SLOPE3_DOWN_LEFT_HEAVY: ((0.0, THIRD), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
}


class CellGeometry(NamedTuple):
    """Indexed polygon geometry for occupied cells."""

    point_positions: np.ndarray
    cell_vertices: np.ndarray
    cell_vertex_count: np.ndarray
    cell_edge_a: np.ndarray
    cell_edge_b: np.ndarray
    surface_edge_mask: np.ndarray
    cell_types: np.ndarray
    cell_mask: np.ndarray
    cell_world_vy: np.ndarray
    cell_world_vx: np.ndarray
    terrain_sample_positions: np.ndarray


def cell_corner_ids(vtype: int) -> tuple[int, ...]:
    """Return CCW vertex ids for an occupied cell type."""
    try:
        return tuple(range(len(CELL_TYPE_TO_VERTEX_OFFSETS[int(vtype)])))
    except KeyError as exc:  # pragma: no cover - defensive boundary
        raise ValueError(f"unsupported cell type {vtype}") from exc


def cell_corner_coords(cell_x: int, cell_y: int, vtype: int) -> tuple[tuple[float, float], ...]:
    """Return world-lattice polygon vertices for a cell."""
    try:
        offsets = CELL_TYPE_TO_VERTEX_OFFSETS[int(vtype)]
    except KeyError as exc:  # pragma: no cover - defensive boundary
        raise ValueError(f"unsupported cell type {vtype}") from exc
    return tuple((float(cell_x) + dx, float(cell_y) + dy) for dx, dy in offsets)


def grid_active_cells(
    grid: np.ndarray,
    origin_x: int = 0,
    origin_y: int = 0,
) -> list[tuple[int, int, int]]:
    """Return active cells from a y-up object grid in world coordinates."""
    cells: list[tuple[int, int, int]] = []
    h, w = grid.shape
    for vy in range(h):
        for vx in range(w):
            vtype = int(grid[vy, vx])
            if vtype == C.EMPTY:
                continue
            cells.append((int(origin_x) + vx, int(origin_y) + vy, vtype))
    return cells


def build_sparse_cell_geometry(cells: Iterable[tuple[int, int, int]]) -> CellGeometry:
    """Build indexed polygon geometry for active cells."""
    active_cells = [
        (int(cell_x), int(cell_y), int(vtype))
        for cell_x, cell_y, vtype in cells
        if int(vtype) != C.EMPTY
    ]
    n_cells = len(active_cells)
    point_lookup: dict[tuple[int, int], int] = {}
    lattice_points: list[tuple[float, float]] = []
    edge_occurrences: dict[
        tuple[tuple[int, int], tuple[int, int]],
        list[tuple[int, int, tuple[float, float], tuple[float, float]]],
    ] = {}

    cell_vertices = np.zeros((n_cells, EDGE_SLOTS), dtype=np.int32)
    cell_vertex_count = np.zeros((n_cells,), dtype=np.int32)
    cell_edge_a = np.zeros((n_cells, EDGE_SLOTS), dtype=np.int32)
    cell_edge_b = np.zeros((n_cells, EDGE_SLOTS), dtype=np.int32)
    surface_edge_mask = np.zeros((n_cells, EDGE_SLOTS), dtype=bool)
    cell_types = np.zeros((n_cells,), dtype=np.int32)
    cell_mask = np.ones((n_cells,), dtype=bool)
    cell_world_vy = np.zeros((n_cells,), dtype=np.int32)
    cell_world_vx = np.zeros((n_cells,), dtype=np.int32)

    for cell_idx, (cell_x, cell_y, vtype) in enumerate(active_cells):
        polygon = cell_corner_coords(cell_x, cell_y, vtype)
        vertex_indices: list[int] = []
        for lattice_point in polygon:
            point_key = _grid_key(lattice_point)
            point_idx = point_lookup.get(point_key)
            if point_idx is None:
                point_idx = len(lattice_points)
                point_lookup[point_key] = point_idx
                lattice_points.append(lattice_point)
            vertex_indices.append(point_idx)

        count = len(vertex_indices)
        padded_vertices = vertex_indices + [vertex_indices[-1]] * (EDGE_SLOTS - count)
        cell_vertices[cell_idx] = np.array(padded_vertices, dtype=np.int32)
        cell_vertex_count[cell_idx] = count
        cell_types[cell_idx] = vtype
        cell_world_vx[cell_idx] = cell_x
        cell_world_vy[cell_idx] = cell_y

        base_idx = vertex_indices[0]
        cell_edge_a[cell_idx, :] = base_idx
        cell_edge_b[cell_idx, :] = base_idx
        for edge_idx in range(count):
            next_idx = (edge_idx + 1) % count
            edge_a_idx = vertex_indices[next_idx]
            edge_b_idx = vertex_indices[edge_idx]
            cell_edge_a[cell_idx, edge_idx] = edge_a_idx
            cell_edge_b[cell_idx, edge_idx] = edge_b_idx
            lattice_a = polygon[next_idx]
            lattice_b = polygon[edge_idx]
            edge_key = _canonical_edge(_grid_key(lattice_a), _grid_key(lattice_b))
            edge_occurrences.setdefault(edge_key, []).append(
                (cell_idx, edge_idx, lattice_a, lattice_b)
            )

    sloped_edge_midpoints: list[tuple[float, float]] = []
    for occurrences in edge_occurrences.values():
        if len(occurrences) != 1:
            continue
        cell_idx, edge_idx, lattice_a, lattice_b = occurrences[0]
        surface_edge_mask[cell_idx, edge_idx] = True
        if _is_sloped_edge(lattice_a, lattice_b):
            sloped_edge_midpoints.append(
                (
                    (lattice_a[0] + lattice_b[0]) * 0.5 * C.CELL_SIZE,
                    (lattice_a[1] + lattice_b[1]) * 0.5 * C.CELL_SIZE,
                )
            )

    if lattice_points:
        point_positions = np.asarray(lattice_points, dtype=np.float32) * C.CELL_SIZE
        terrain_sample_positions = point_positions
    else:
        point_positions = np.zeros((0, 2), dtype=np.float32)
        terrain_sample_positions = np.zeros((0, 2), dtype=np.float32)

    if sloped_edge_midpoints:
        terrain_sample_positions = np.unique(
            np.concatenate(
                [
                    terrain_sample_positions,
                    np.asarray(sloped_edge_midpoints, dtype=np.float32),
                ],
                axis=0,
            ).astype(np.float32, copy=False),
            axis=0,
        )

    return CellGeometry(
        point_positions=point_positions,
        cell_vertices=cell_vertices,
        cell_vertex_count=cell_vertex_count,
        cell_edge_a=cell_edge_a,
        cell_edge_b=cell_edge_b,
        surface_edge_mask=surface_edge_mask,
        cell_types=cell_types,
        cell_mask=cell_mask,
        cell_world_vy=cell_world_vy,
        cell_world_vx=cell_world_vx,
        terrain_sample_positions=terrain_sample_positions.astype(np.float32, copy=False),
    )


def _canonical_edge(
    point_a: tuple[int, int],
    point_b: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    return (point_a, point_b) if point_a <= point_b else (point_b, point_a)


def _grid_key(point: tuple[float, float]) -> tuple[int, int]:
    return (
        int(round(point[0] * GRID_KEY_SCALE)),
        int(round(point[1] * GRID_KEY_SCALE)),
    )


def _is_sloped_edge(point_a: tuple[float, float], point_b: tuple[float, float]) -> bool:
    dx = abs(point_a[0] - point_b[0])
    dy = abs(point_a[1] - point_b[1])
    return dx > 1e-6 and dy > 1e-6
