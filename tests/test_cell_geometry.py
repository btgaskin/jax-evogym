"""Tests for cell polygon winding and convex collision assumptions."""

import numpy as np
import pytest
import jax.numpy as jnp

from jax_evogym import constants as C
from jax_evogym.cell_geometry import build_sparse_cell_geometry, cell_corner_coords
from jax_evogym.collision import point_in_convex_cell


def _signed_area(vertices: np.ndarray) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


@pytest.mark.parametrize("slope_type", sorted(C.SLOPE_TYPES))
def test_slope_corner_order_is_ccw(slope_type):
    verts = np.asarray(cell_corner_coords(0, 0, slope_type), dtype=np.float32)
    assert verts.shape[1] == 2
    assert verts.shape[0] in (3, 4)
    assert _signed_area(verts) > 0.0


def test_slope_down_right_convex_inside_test():
    verts = np.asarray(
        cell_corner_coords(0, 0, C.SLOPE_DOWN_RIGHT),
        dtype=np.float32,
    )
    edge_a = np.zeros((4, 2), dtype=np.float32)
    edge_b = np.zeros((4, 2), dtype=np.float32)
    edge_mask = np.zeros((4,), dtype=bool)
    for edge_idx in range(verts.shape[0]):
        next_idx = (edge_idx + 1) % verts.shape[0]
        edge_a[edge_idx] = verts[next_idx]
        edge_b[edge_idx] = verts[edge_idx]
        edge_mask[edge_idx] = True

    points = jnp.asarray(
        [
            [0.2, 0.8],  # inside (x <= y)
            [0.8, 0.2],  # outside (x > y)
        ],
        dtype=jnp.float32,
    )
    inside = point_in_convex_cell(
        points,
        jnp.asarray(edge_a, dtype=jnp.float32)[None, :, :],
        jnp.asarray(edge_b, dtype=jnp.float32)[None, :, :],
        jnp.asarray(edge_mask, dtype=jnp.bool_)[None, :],
    )
    assert bool(inside[0]) is True
    assert bool(inside[1]) is False


def test_2x1_shallow_slope_adds_surface_midpoint_samples():
    geometry = build_sparse_cell_geometry(
        [
            (0, 0, C.SLOPE2_UP_RIGHT_LIGHT),
            (1, 0, C.SLOPE2_UP_RIGHT_HEAVY),
        ]
    )
    samples = np.asarray(geometry.terrain_sample_positions, dtype=np.float32)
    expected_a = np.array([0.5 * C.CELL_SIZE, 0.25 * C.CELL_SIZE], dtype=np.float32)
    expected_b = np.array([1.5 * C.CELL_SIZE, 0.75 * C.CELL_SIZE], dtype=np.float32)
    assert np.any(np.all(np.isclose(samples, expected_a[None, :], atol=1e-6), axis=1))
    assert np.any(np.all(np.isclose(samples, expected_b[None, :], atol=1e-6), axis=1))


def test_3x1_shallow_slope_adds_surface_midpoint_samples():
    geometry = build_sparse_cell_geometry(
        [
            (0, 0, C.SLOPE3_UP_RIGHT_LIGHT),
            (1, 0, C.SLOPE3_UP_RIGHT_MID),
            (2, 0, C.SLOPE3_UP_RIGHT_HEAVY),
        ]
    )
    samples = np.asarray(geometry.terrain_sample_positions, dtype=np.float32)
    expected_a = np.array([0.5 * C.CELL_SIZE, (1.0 / 6.0) * C.CELL_SIZE], dtype=np.float32)
    expected_b = np.array([1.5 * C.CELL_SIZE, 0.5 * C.CELL_SIZE], dtype=np.float32)
    expected_c = np.array([2.5 * C.CELL_SIZE, (5.0 / 6.0) * C.CELL_SIZE], dtype=np.float32)
    assert np.any(np.all(np.isclose(samples, expected_a[None, :], atol=1e-6), axis=1))
    assert np.any(np.all(np.isclose(samples, expected_b[None, :], atol=1e-6), axis=1))
    assert np.any(np.all(np.isclose(samples, expected_c[None, :], atol=1e-6), axis=1))
