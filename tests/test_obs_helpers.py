"""Tests for observation helper functions in jax_evogym.environments._obs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_evogym.environments._obs import (
    get_vel_com_obs,
    get_relative_pos_obs,
    compute_orientation,
    get_floor_obs,
)


# ---------------------------------------------------------------------------
# get_vel_com_obs
# ---------------------------------------------------------------------------

class TestVelComObs:
    def test_shape(self):
        vel = jnp.ones((10, 2))
        idx = jnp.arange(5)
        result = get_vel_com_obs(vel, idx)
        assert result.shape == (2,)

    def test_mean_of_subset(self):
        vel = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        idx = jnp.array([0, 2])  # pick rows 0 and 2
        result = get_vel_com_obs(vel, idx)
        expected = jnp.array([3.0, 4.0])  # mean of [1,5] and [2,6]
        assert jnp.allclose(result, expected)

    def test_zero_velocity(self):
        vel = jnp.zeros((8, 2))
        idx = jnp.arange(8)
        result = get_vel_com_obs(vel, idx)
        assert jnp.allclose(result, 0.0)


# ---------------------------------------------------------------------------
# get_relative_pos_obs
# ---------------------------------------------------------------------------

class TestRelativePosObs:
    def test_shape(self):
        pos = jnp.ones((10, 2))
        idx = jnp.arange(4)
        result = get_relative_pos_obs(pos, idx)
        assert result.shape == (8,)  # 2*4

    def test_sums_to_zero(self):
        """Relative positions should sum to zero along each coordinate."""
        pos = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        idx = jnp.arange(4)
        result = get_relative_pos_obs(pos, idx)
        # First half is x coords, second half is y coords
        n = 4
        x_rel = result[:n]
        y_rel = result[n:]
        assert jnp.allclose(jnp.sum(x_rel), 0.0, atol=1e-6)
        assert jnp.allclose(jnp.sum(y_rel), 0.0, atol=1e-6)

    def test_transpose_flatten_order(self):
        """Should be (pos - com).T.flatten() giving [x0..xn, y0..yn]."""
        pos = jnp.array([[1.0, 10.0], [3.0, 30.0]])
        idx = jnp.arange(2)
        result = get_relative_pos_obs(pos, idx)
        com = jnp.array([2.0, 20.0])
        expected = jnp.array([-1.0, 1.0, -10.0, 10.0])
        assert jnp.allclose(result, expected)


# ---------------------------------------------------------------------------
# compute_orientation
# ---------------------------------------------------------------------------

class TestComputeOrientation:
    def test_no_rotation_is_near_zero(self):
        """Identical positions should give orientation ≈ 0."""
        pos = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        ort = compute_orientation(pos, pos)
        # Float32 precision: may be slightly nonzero
        assert float(ort) < 0.01

    def test_90_degrees_clockwise(self):
        """90° CW rotation should give ≈ π/2 (or 3π/2 depending on CW/CCW convention)."""
        initial = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        # 90° CW: (x, y) → (y, -x)
        rotated = jnp.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        ort = compute_orientation(rotated, initial)
        # Should be ~π/2 or ~3π/2 — the C++ convention uses cross product sign
        assert jnp.allclose(ort, jnp.pi / 2, atol=0.1) or jnp.allclose(ort, 3 * jnp.pi / 2, atol=0.1)

    def test_180_degrees(self):
        """180° rotation should give ≈ π."""
        initial = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        rotated = -initial
        ort = compute_orientation(rotated, initial)
        assert jnp.allclose(ort, jnp.pi, atol=0.1)

    def test_output_range(self):
        """Orientation should be in [0, 2π]."""
        rng = jax.random.PRNGKey(42)
        pos1 = jax.random.normal(rng, (8, 2))
        rng, key = jax.random.split(rng)
        pos2 = jax.random.normal(key, (8, 2))
        ort = compute_orientation(pos1, pos2)
        assert float(ort) >= 0.0
        assert float(ort) <= 2 * np.pi + 0.01

    def test_jit_compatible(self):
        pos = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        jitted = jax.jit(compute_orientation)
        ort = jitted(pos, pos)
        assert jnp.isfinite(ort)


# ---------------------------------------------------------------------------
# get_floor_obs
# ---------------------------------------------------------------------------

class TestFloorObs:
    def test_shape(self):
        com = jnp.array([0.5, 0.5])
        terrain = jnp.array([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])
        mask = jnp.ones(3, dtype=jnp.bool_)
        result = get_floor_obs(com, terrain, mask, sight_dist=2)
        assert result.shape == (5,)  # 2*2+1

    def test_flat_ground_uniform(self):
        """Flat ground should give uniform elevations."""
        # Robot COM at physical (0.5, 0.2). Ground points at y=0.
        com = jnp.array([0.5, 0.2])
        # Dense ground points every 0.05 across x
        xs = jnp.arange(0.0, 1.0, 0.05)
        terrain = jnp.stack([xs, jnp.zeros_like(xs)], axis=-1)
        mask = jnp.ones(len(xs), dtype=jnp.bool_)
        result = get_floor_obs(com, terrain, mask, sight_dist=1)
        # 3 columns, all should see ground at y=0
        # elevation = com_y_grid(2) - terrain_y_grid(0) = 2.0, clamped to [0, 5]
        # result = 2.0 * 0.1 = 0.2
        assert result.shape == (3,)
        assert jnp.all(result > 0)
        # All columns should be similar (flat ground)
        assert jnp.allclose(result, result[0], atol=0.05)

    def test_no_terrain_below(self):
        """When no terrain is below COM, should get max elevation."""
        com = jnp.array([0.5, 0.5])
        # Terrain above COM
        terrain = jnp.array([[0.5, 0.6]])
        mask = jnp.ones(1, dtype=jnp.bool_)
        result = get_floor_obs(com, terrain, mask, sight_dist=0)
        # Should be at max sight_range * 0.1
        assert result.shape == (1,)
        assert jnp.allclose(result, 0.5, atol=0.01)  # sight_range=5, *0.1=0.5

    def test_jit_compatible(self):
        com = jnp.array([0.5, 0.2])
        terrain = jnp.array([[0.5, 0.0]])
        mask = jnp.ones(1, dtype=jnp.bool_)
        jitted = jax.jit(lambda c, t, m: get_floor_obs(c, t, m, sight_dist=2))
        result = jitted(com, terrain, mask)
        assert jnp.all(jnp.isfinite(result))
