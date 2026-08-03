"""Tests for jax_evogym.constraints — PBD Phase 1 and Phase 2."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_evogym.types import SpringTopology
from jax_evogym.constraints import resolve_edge_constraints


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spring_system(pos_a, pos_b, init_rest_length, is_rigid=False, is_fixed_a=False):
    """Two points connected by one spring."""
    positions = jnp.array([list(pos_a), list(pos_b)], dtype=jnp.float32)
    fixed = jnp.array([
        [is_fixed_a, is_fixed_a],
        [False, False],
    ], dtype=bool)
    topology = SpringTopology(
        a_idx=jnp.array([0], dtype=jnp.int32),
        b_idx=jnp.array([1], dtype=jnp.int32),
    )
    init_rest = jnp.array([init_rest_length], dtype=jnp.float32)
    spring_mask = jnp.ones(1, dtype=bool)
    point_mask = jnp.ones(2, dtype=bool)
    rigid_mask = jnp.array([is_rigid], dtype=bool)
    return positions, fixed, topology, init_rest, spring_mask, point_mask, rigid_mask


# ---------------------------------------------------------------------------
# Phase 1 tests (all springs, ±25%, factor 0.8)
# ---------------------------------------------------------------------------

class TestPBDPhase1:
    """Test PBD Phase 1 — all springs, ±25% tolerance, factor 0.8."""

    def test_overshoot_correction(self):
        """Spring at 1.5x → corrected closer to 1.25x."""
        # rest=0.1, current dist=0.15, stress=1.5 > 1.25 → overshoot
        pos, fixed, topo, init_rest, smask, pmask, rmask = _make_spring_system(
            pos_a=(0.0, 0.0), pos_b=(0.15, 0.0), init_rest_length=0.1,
        )
        new_pos = resolve_edge_constraints(pos, fixed, topo, init_rest, smask, pmask, rmask)

        new_dist = float(jnp.sqrt(jnp.sum((new_pos[0] - new_pos[1]) ** 2)))
        old_dist = 0.15
        # Should be closer to 0.125 (1.25x) but not exactly there (factor 0.8)
        assert new_dist < old_dist, "Overshoot should reduce distance"
        assert new_dist > 0.1 * 1.25, "Single step shouldn't overshoot correction"

    def test_undershoot_correction(self):
        """Spring at 0.5x → corrected apart toward 0.75x."""
        # rest=0.1, current dist=0.05, stress=0.5 < 0.75 → undershoot
        pos, fixed, topo, init_rest, smask, pmask, rmask = _make_spring_system(
            pos_a=(0.0, 0.0), pos_b=(0.05, 0.0), init_rest_length=0.1,
        )
        new_pos = resolve_edge_constraints(pos, fixed, topo, init_rest, smask, pmask, rmask)

        new_dist = float(jnp.sqrt(jnp.sum((new_pos[0] - new_pos[1]) ** 2)))
        old_dist = 0.05
        assert new_dist > old_dist, "Undershoot should increase distance"

    def test_within_tolerance_no_change(self):
        """Spring at 1.1x → unchanged (within ±25%)."""
        # rest=0.1, current dist=0.11, stress=1.1 ∈ [0.75, 1.25] → no correction
        pos, fixed, topo, init_rest, smask, pmask, rmask = _make_spring_system(
            pos_a=(0.0, 0.0), pos_b=(0.11, 0.0), init_rest_length=0.1,
        )
        new_pos = resolve_edge_constraints(pos, fixed, topo, init_rest, smask, pmask, rmask)
        np.testing.assert_allclose(new_pos, pos, atol=1e-6)

    def test_fixed_point_doesnt_move(self):
        """Fixed endpoint doesn't move during Phase 1."""
        pos, fixed, topo, init_rest, smask, pmask, rmask = _make_spring_system(
            pos_a=(0.0, 0.0), pos_b=(0.15, 0.0), init_rest_length=0.1,
            is_fixed_a=True,
        )
        new_pos = resolve_edge_constraints(pos, fixed, topo, init_rest, smask, pmask, rmask)
        np.testing.assert_allclose(new_pos[0], pos[0], atol=1e-10)
        # Point b should still move
        new_dist = float(jnp.sqrt(jnp.sum((new_pos[0] - new_pos[1]) ** 2)))
        assert new_dist < 0.15


# ---------------------------------------------------------------------------
# Phase 2 tests (rigid springs, ±3%, factor 0.5)
# ---------------------------------------------------------------------------

class TestPBDPhase2:
    """Test PBD Phase 2 — rigid springs only, ±3% tolerance, factor 0.5."""

    def test_rigid_spring_corrected(self):
        """Spring at 1.05x with rigid_spring_mask=True → corrected by Phase 2."""
        # stress=1.05, within Phase 1 tolerance (±25%) but outside Phase 2 (±3%)
        pos, fixed, topo, init_rest, smask, pmask, rmask = _make_spring_system(
            pos_a=(0.0, 0.0), pos_b=(0.105, 0.0), init_rest_length=0.1,
            is_rigid=True,
        )
        new_pos = resolve_edge_constraints(pos, fixed, topo, init_rest, smask, pmask, rmask)

        new_dist = float(jnp.sqrt(jnp.sum((new_pos[0] - new_pos[1]) ** 2)))
        old_dist = 0.105
        assert new_dist < old_dist, "Phase 2 should correct rigid spring overshoot"

    def test_non_rigid_not_corrected_by_phase2(self):
        """Spring at 1.05x without rigid mask → only Phase 1 applies (which skips it)."""
        pos, fixed, topo, init_rest, smask, pmask, rmask = _make_spring_system(
            pos_a=(0.0, 0.0), pos_b=(0.105, 0.0), init_rest_length=0.1,
            is_rigid=False,
        )
        new_pos = resolve_edge_constraints(pos, fixed, topo, init_rest, smask, pmask, rmask)
        # stress=1.05 is within Phase 1 tolerance, and Phase 2 doesn't apply
        np.testing.assert_allclose(new_pos, pos, atol=1e-6)

    def test_mixed_springs_only_rigid_corrected(self):
        """Two springs: rigid at 1.05x corrected, non-rigid at 1.05x unchanged."""
        positions = jnp.array([
            [0.0, 0.0], [0.105, 0.0],  # spring 0 endpoints
            [0.0, 0.2], [0.105, 0.2],  # spring 1 endpoints
        ], dtype=jnp.float32)
        fixed = jnp.zeros((4, 2), dtype=bool)
        topology = SpringTopology(
            a_idx=jnp.array([0, 2], dtype=jnp.int32),
            b_idx=jnp.array([1, 3], dtype=jnp.int32),
        )
        init_rest = jnp.array([0.1, 0.1], dtype=jnp.float32)
        spring_mask = jnp.ones(2, dtype=bool)
        point_mask = jnp.ones(4, dtype=bool)
        rigid_mask = jnp.array([True, False], dtype=bool)  # only spring 0 is rigid

        new_pos = resolve_edge_constraints(
            positions, fixed, topology, init_rest, spring_mask, point_mask, rigid_mask,
        )

        # Spring 0 (rigid): should be corrected
        dist0 = float(jnp.sqrt(jnp.sum((new_pos[0] - new_pos[1]) ** 2)))
        assert dist0 < 0.105, "Rigid spring should be corrected"

        # Spring 1 (non-rigid): unchanged (1.05x within Phase 1 tolerance)
        dist1 = float(jnp.sqrt(jnp.sum((new_pos[2] - new_pos[3]) ** 2)))
        np.testing.assert_allclose(dist1, 0.105, atol=1e-5)


# ---------------------------------------------------------------------------
# Ghost spring tests
# ---------------------------------------------------------------------------

class TestGhostSprings:
    """Test ghost spring masking."""

    def test_ghost_spring_no_correction(self):
        """spring_mask=False → zero correction."""
        pos, fixed, topo, init_rest, _, pmask, rmask = _make_spring_system(
            pos_a=(0.0, 0.0), pos_b=(0.15, 0.0), init_rest_length=0.1,
        )
        smask = jnp.zeros(1, dtype=bool)  # ghost spring
        new_pos = resolve_edge_constraints(pos, fixed, topo, init_rest, smask, pmask, rmask)
        np.testing.assert_allclose(new_pos, pos, atol=1e-10)


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------

class TestJIT:
    """Test JIT compatibility."""

    def test_jit_resolve_edge_constraints(self):
        """jax.jit(resolve_edge_constraints) runs without error."""
        pos, fixed, topo, init_rest, smask, pmask, rmask = _make_spring_system(
            pos_a=(0.0, 0.0), pos_b=(0.15, 0.0), init_rest_length=0.1,
        )
        jit_fn = jax.jit(resolve_edge_constraints)
        new_pos = jit_fn(pos, fixed, topo, init_rest, smask, pmask, rmask)
        assert new_pos.shape == pos.shape
