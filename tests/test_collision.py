"""Tests for jax_evogym.collision — object-object collision detection and response."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_evogym.types import (
    SimState, SpringTopology, PhysicsConstants, ActuatorInfo, CollisionData,
    default_physics_constants,
)
from jax_evogym.build import compile_world_template, instantiate_world
from jax_evogym.collision import (
    make_collision_data, resolve_collisions, is_self_colliding,
    resolve_collisions_dynamic_broadphase,
    resolve_static_collisions,
    resolve_static_collisions_with_contacts,
    _compute_friction_with_stiction,
    point_in_quad, edge_intersection, dist_point_to_edge,
    check_triple_capacity,
)
from jax_evogym.sim import physics_substep, env_step
from jax_evogym.utils import voxel_to_dense_mass_spring, make_sim_state
from jax_evogym.world import EvoWorld
from jax_evogym import constants as C


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _padded_morphology(morph, H=5, W=5):
    """Flip + pad morphology like voxel_to_dense_mass_spring does."""
    H_morph, W_morph = morph.shape
    flipped = np.flipud(morph).astype(int)
    padded = np.zeros((H, W), dtype=int)
    padded[:H_morph, :W_morph] = flipped
    return padded


def _make_state_with_collision(morph, H=5, W=5, spawn_x=0, spawn_y=0):
    """Create state + collision data for a morphology."""
    dense = voxel_to_dense_mass_spring(morph, H=H, W=W)
    state, topology, actuator_info = make_sim_state(dense, spawn_x=spawn_x, spawn_y=spawn_y)
    padded = _padded_morphology(morph, H, W)
    coll_data = make_collision_data(padded, H, W)
    constants = default_physics_constants()
    return state, topology, actuator_info, constants, coll_data


# ---------------------------------------------------------------------------
# make_collision_data tests
# ---------------------------------------------------------------------------

class TestMakeCollisionData:

    def test_single_voxel_all_surface(self):
        """Single voxel: all 4 edges are surface."""
        morph = np.array([[C.SOFT]])
        padded = _padded_morphology(morph)
        cd = make_collision_data(padded, 5, 5)

        # Find active boxel
        active = np.array(cd.boxel_mask)
        assert np.sum(active) == 1
        idx = np.argmax(active)

        # All 4 edges should be surface
        assert np.all(np.array(cd.surface_edge_mask[idx]))

    def test_2x2_block_internal_edges(self):
        """2x2 block: internal edges are NOT surface."""
        morph = np.array([[C.SOFT, C.SOFT],
                          [C.SOFT, C.SOFT]])
        padded = _padded_morphology(morph, H=5, W=5)
        cd = make_collision_data(padded, 5, 5)

        active = np.array(cd.boxel_mask)
        assert np.sum(active) == 4

        # Each boxel has some non-surface edges (internal)
        surface = np.array(cd.surface_edge_mask)
        active_surface = surface[active]
        # Total surface edges: 4 boxels × 4 edges = 16 possible
        # Internal: 4 shared edges → 8 non-surface edge slots
        # Boundary: 8 edges → 8 surface edge slots
        total_surface = np.sum(active_surface)
        assert total_surface == 8, f"Expected 8 surface edges, got {total_surface}"

    def test_robot_terrain_boundary_is_surface(self):
        """Robot adjacent to terrain: shared boundary edges ARE surface."""
        # Robot on left, terrain on right (same row)
        morph = np.array([[C.SOFT, C.FIXED]])
        padded = _padded_morphology(morph, H=5, W=5)
        cd = make_collision_data(padded, 5, 5)

        active = np.array(cd.boxel_mask)
        assert np.sum(active) == 2

        # The shared edge between robot and terrain should be surface for both
        surface = np.array(cd.surface_edge_mask)
        active_surface = surface[active]
        # Each boxel: 3 external + 1 shared = all 4 are surface (different categories)
        total_surface = np.sum(active_surface)
        assert total_surface == 8, f"Expected 8 (all edges surface), got {total_surface}"

    def test_triple_same_boxel_not_indexed(self):
        """Same boxel → no indexed triples with i==j."""
        morph = np.array([[C.SOFT]])
        padded = _padded_morphology(morph)
        cd = make_collision_data(padded, 5, 5)

        ti = np.array(cd.triple_i)
        tj = np.array(cd.triple_j)
        active = np.array(cd.triple_active)
        assert not np.any((ti == tj) & active)

    def test_triple_adjacent_same_object_not_indexed(self):
        """Adjacent robot boxels → no indexed triples."""
        morph = np.array([[C.SOFT, C.SOFT]])
        padded = _padded_morphology(morph, H=5, W=5)
        cd = make_collision_data(padded, 5, 5)

        active = np.array(cd.triple_active)
        assert not np.any(active)

    def test_self_triple_robot_only(self):
        """Self-collision triples only have robot-robot pairs."""
        morph = np.array([[C.SOFT, C.FIXED]])
        padded = _padded_morphology(morph, H=5, W=5)
        cd = make_collision_data(padded, 5, 5)

        active = np.array(cd.self_triple_active)
        assert not np.any(active), "Adjacent robot+terrain should have no self-collision entries"

    def test_boxel_corners_order(self):
        """Corners are [BL, BR, TR, TL]."""
        morph = np.array([[C.SOFT]])
        padded = _padded_morphology(morph, H=3, W=3)
        cd = make_collision_data(padded, 3, 3)

        active = np.array(cd.boxel_mask)
        idx = np.argmax(active)
        corners = np.array(cd.boxel_corners[idx])

        W1 = 4  # W+1
        # The SOFT voxel is at (vy=0, vx=0) after flip+pad
        bl = 0 * W1 + 0
        br = 0 * W1 + 1
        tr = 1 * W1 + 1
        tl = 1 * W1 + 0
        np.testing.assert_array_equal(corners, [bl, br, tr, tl])

    def test_corner_edge_indices(self):
        """Static corner-to-edge mapping."""
        morph = np.array([[C.SOFT]])
        padded = _padded_morphology(morph)
        cd = make_collision_data(padded, 5, 5)

        expected = np.array([[0, 3], [0, 1], [1, 2], [2, 3]])
        np.testing.assert_array_equal(np.array(cd.corner_edge_indices), expected)


# ---------------------------------------------------------------------------
# Point-in-quad tests
# ---------------------------------------------------------------------------

class TestPointInQuad:

    def test_inside_axis_aligned(self):
        """Point inside axis-aligned unit quad → True."""
        quad = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        point = jnp.array([0.5, 0.5])
        assert bool(point_in_quad(point, quad))

    def test_outside_axis_aligned(self):
        """Point outside axis-aligned unit quad → False."""
        quad = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        point = jnp.array([1.5, 0.5])
        assert not bool(point_in_quad(point, quad))

    def test_inside_deformed_quad(self):
        """Point inside parallelogram → True."""
        # Parallelogram: BL=(0,0), BR=(1,0), TR=(1.5,1), TL=(0.5,1)
        quad = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.5, 1.0], [0.5, 1.0]])
        point = jnp.array([0.75, 0.5])
        assert bool(point_in_quad(point, quad))

    def test_batched(self):
        """Batched point-in-quad test."""
        quad = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        points = jnp.array([[0.5, 0.5], [1.5, 0.5]])
        # Broadcast: (2, 2) point with (4, 2) quad
        quads = jnp.broadcast_to(quad[None], (2, 4, 2))
        result = point_in_quad(points, quads)
        assert result[0] and not result[1]


# ---------------------------------------------------------------------------
# Edge intersection tests
# ---------------------------------------------------------------------------

class TestEdgeIntersection:

    def test_crossing(self):
        """Crossing segments → True."""
        a1 = jnp.array([0.0, 0.0])
        a2 = jnp.array([1.0, 1.0])
        b1 = jnp.array([0.0, 1.0])
        b2 = jnp.array([1.0, 0.0])
        assert bool(edge_intersection(a1, a2, b1, b2))

    def test_parallel(self):
        """Parallel segments → False."""
        a1 = jnp.array([0.0, 0.0])
        a2 = jnp.array([1.0, 0.0])
        b1 = jnp.array([0.0, 1.0])
        b2 = jnp.array([1.0, 1.0])
        assert not bool(edge_intersection(a1, a2, b1, b2))

    def test_non_intersecting(self):
        """Non-intersecting non-parallel segments → False."""
        a1 = jnp.array([0.0, 0.0])
        a2 = jnp.array([0.5, 0.5])
        b1 = jnp.array([0.7, 0.0])
        b2 = jnp.array([1.0, 0.5])
        assert not bool(edge_intersection(a1, a2, b1, b2))

    def test_t_intersection(self):
        """T-intersection at endpoint → True."""
        a1 = jnp.array([0.5, 0.0])
        a2 = jnp.array([0.5, 1.0])
        b1 = jnp.array([0.0, 0.5])
        b2 = jnp.array([1.0, 0.5])
        assert bool(edge_intersection(a1, a2, b1, b2))


# ---------------------------------------------------------------------------
# dist_point_to_edge tests
# ---------------------------------------------------------------------------

class TestDistPointToEdge:

    def test_perpendicular_distance(self):
        """Point directly above horizontal edge."""
        point = jnp.array([0.5, 1.0])
        edge_a = jnp.array([0.0, 0.0])
        edge_b = jnp.array([1.0, 0.0])
        d = float(dist_point_to_edge(point, edge_a, edge_b))
        np.testing.assert_allclose(d, 1.0, atol=1e-6)

    def test_zero_distance(self):
        """Point on the edge line."""
        point = jnp.array([0.5, 0.0])
        edge_a = jnp.array([0.0, 0.0])
        edge_b = jnp.array([1.0, 0.0])
        d = float(dist_point_to_edge(point, edge_a, edge_b))
        np.testing.assert_allclose(d, 0.0, atol=1e-6)

    def test_degenerate_edge(self):
        """Degenerate edge (zero length) → large distance."""
        point = jnp.array([0.5, 1.0])
        edge_a = jnp.array([0.0, 0.0])
        edge_b = jnp.array([0.0, 0.0])
        d = float(dist_point_to_edge(point, edge_a, edge_b))
        assert d > 100

    def test_beyond_endpoint(self):
        """Point beyond edge endpoint → distance to nearest endpoint, not line."""
        point = jnp.array([2.0, 0.0])
        edge_a = jnp.array([0.0, 0.0])
        edge_b = jnp.array([1.0, 0.0])
        d = float(dist_point_to_edge(point, edge_a, edge_b))
        np.testing.assert_allclose(d, 1.0, atol=1e-6)

    def test_beyond_endpoint_diagonal(self):
        """Point beyond diagonal edge endpoint → segment distance."""
        point = jnp.array([1.5, 1.0])
        edge_a = jnp.array([0.0, 0.0])
        edge_b = jnp.array([1.0, 0.5])
        d = float(dist_point_to_edge(point, edge_a, edge_b))
        expected = float(jnp.sqrt((1.5 - 1.0) ** 2 + (1.0 - 0.5) ** 2))
        np.testing.assert_allclose(d, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# resolve_collisions force tests
# ---------------------------------------------------------------------------

class TestResolveCollisions:

    def _make_robot_terrain_scenario(self, robot_y_offset=0.0):
        """Robot SOFT voxel above terrain FIXED voxel, with optional y offset.

        Grid layout (5x5):
          row 2: SOFT (robot) at (vy=2, vx=0)
          row 0: FIXED (terrain) at (vy=0, vx=0)
          (row 1 is empty gap — avoids shared grid points between objects)
        """
        H, W = 5, 5
        morph_padded = np.zeros((H, W), dtype=int)
        morph_padded[0, 0] = C.FIXED   # terrain at bottom
        morph_padded[2, 0] = C.SOFT    # robot 2 rows above (gap at row 1)

        cd = make_collision_data(morph_padded, H, W)
        constants = default_physics_constants()

        n_points = (H + 1) * (W + 1)
        positions = np.zeros((n_points, 2), dtype=np.float32)
        W1 = W + 1
        for gy in range(H + 1):
            for gx in range(W + 1):
                positions[gy * W1 + gx] = [gx * C.CELL_SIZE, gy * C.CELL_SIZE]

        # Shift robot points down by offset (simulating penetration)
        # Robot boxel is at (vy=2, vx=0), corners at grid rows 2 and 3
        for gy in [2, 3]:
            for gx in [0, 1]:
                pidx = gy * W1 + gx
                positions[pidx, 1] += robot_y_offset

        pos_jax = jnp.array(positions)
        vel_true = jnp.zeros((n_points, 2), dtype=jnp.float32)
        return pos_jax, vel_true, cd, constants

    def test_no_penetration_no_force(self):
        """Robot above terrain, no overlap → zero collision forces."""
        pos, vel, cd, consts = self._make_robot_terrain_scenario(robot_y_offset=0.0)
        forces = resolve_collisions(pos, vel, cd, consts)
        # No collision → all forces should be zero
        np.testing.assert_allclose(np.array(forces), 0.0, atol=1e-6)

    def test_penetration_produces_force(self):
        """Robot pushed into terrain → nonzero collision force."""
        # Push robot down so its bottom points overlap with terrain
        # Robot at vy=2 (y=0.2), terrain top at y=0.1 → offset -0.15 puts BL at y=0.05
        pos, vel, cd, consts = self._make_robot_terrain_scenario(robot_y_offset=-0.15)
        forces = resolve_collisions(pos, vel, cd, consts)
        total = float(jnp.sum(jnp.abs(forces)))
        assert total > 0, "Penetration should produce collision forces"

    def test_velocity_damping(self):
        """Moving point during collision gets extra damping."""
        pos, _, cd, consts = self._make_robot_terrain_scenario(robot_y_offset=-0.15)
        n_points = pos.shape[0]

        vel_zero = jnp.zeros((n_points, 2), dtype=jnp.float32)
        forces_static = resolve_collisions(pos, vel_zero, cd, consts)

        # Add downward velocity to robot points
        vel_moving = jnp.zeros((n_points, 2), dtype=jnp.float32)
        W1 = 6
        for gy in [2, 3]:  # robot grid rows (vy=2 boxel)
            for gx in [0, 1]:
                pidx = gy * W1 + gx
                vel_moving = vel_moving.at[pidx, 1].set(-1.0)

        forces_moving = resolve_collisions(pos, vel_moving, cd, consts)

        # Moving should produce different (larger) forces due to damping
        static_mag = float(jnp.sum(jnp.abs(forces_static)))
        moving_mag = float(jnp.sum(jnp.abs(forces_moving)))
        assert moving_mag > static_mag, "Velocity damping should increase forces"


# ---------------------------------------------------------------------------
# Static collider friction semantics
# ---------------------------------------------------------------------------

class TestStaticColliderFriction:
    @staticmethod
    def _make_overlap_fixture():
        world = EvoWorld()
        world.add_from_array(
            "terrain",
            np.array([[C.FIXED, C.FIXED, C.FIXED]], dtype=np.int32),
            0,
            0,
        )
        world.add_from_array("robot", np.array([[C.SOFT]], dtype=np.int32), 1, 1)

        template = compile_world_template(world, robot_name="robot")
        built = instantiate_world(template)

        positions = np.asarray(built.sim_state.positions, dtype=np.float32).copy()
        velocities = np.zeros_like(positions, dtype=np.float32)

        robot_points = np.asarray(built.robot_point_indices, dtype=np.int32)
        # Move the robot downward into the fixed terrain and add tangential velocity.
        positions[robot_points, 1] -= np.float32(0.11)
        velocities[robot_points, 0] = np.float32(0.5)
        return built, jnp.asarray(positions), jnp.asarray(velocities)

    def test_static_friction_uses_friction_const_not_ground_friction_const(self):
        """Static-object tangential friction should track friction_const only."""
        built, positions, velocities = self._make_overlap_fixture()
        constants = default_physics_constants()

        forces_base = resolve_static_collisions(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            constants,
        )
        fx_base = float(jnp.sum(jnp.abs(forces_base[:, 0])))
        assert fx_base > 0.0, "Expected non-zero tangential force in overlap fixture"

        forces_ground_scaled = resolve_static_collisions(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            constants._replace(ground_friction_const=constants.ground_friction_const * 100.0),
        )
        fx_ground_scaled = float(jnp.sum(jnp.abs(forces_ground_scaled[:, 0])))
        np.testing.assert_allclose(fx_ground_scaled, fx_base, rtol=1e-4, atol=1e-6)

        forces_friction_scaled = resolve_static_collisions(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            constants._replace(friction_const=constants.friction_const * 0.1),
        )
        fx_friction_scaled = float(jnp.sum(jnp.abs(forces_friction_scaled[:, 0])))
        assert fx_friction_scaled < fx_base * 0.25

    def test_static_collision_ignores_padded_cells(self):
        built, positions, velocities = self._make_overlap_fixture()
        constants = default_physics_constants()
        base_forces = resolve_static_collisions(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            constants,
        )

        scd = built.static_collider_data
        padded_static = scd._replace(
            point_positions=jnp.pad(scd.point_positions, ((0, 2), (0, 0))),
            cell_vertices=jnp.pad(scd.cell_vertices, ((0, 3), (0, 0))),
            cell_vertex_count=jnp.pad(scd.cell_vertex_count, ((0, 3),)),
            cell_edge_a=jnp.pad(scd.cell_edge_a, ((0, 3), (0, 0))),
            cell_edge_b=jnp.pad(scd.cell_edge_b, ((0, 3), (0, 0))),
            surface_edge_mask=jnp.pad(
                scd.surface_edge_mask,
                ((0, 3), (0, 0)),
                constant_values=False,
            ),
            cell_types=jnp.pad(scd.cell_types, ((0, 3),)),
            cell_aabb_min=jnp.pad(scd.cell_aabb_min, ((0, 3), (0, 0))),
            cell_aabb_max=jnp.pad(scd.cell_aabb_max, ((0, 3), (0, 0))),
        )
        padded_forces = resolve_static_collisions(
            positions,
            velocities,
            built.collision_data,
            padded_static,
            constants,
        )

        np.testing.assert_allclose(
            np.asarray(padded_forces),
            np.asarray(base_forces),
            atol=1e-6,
        )


class TestStaticTerrainStiction:
    def test_quasi_static_load_applies_force_at_rest(self):
        unit_tangent = jnp.array([[1.0, 0.0]], dtype=jnp.float32)
        force = _compute_friction_with_stiction(
            v_tang=jnp.array([0.0], dtype=jnp.float32),
            normal_force_mag=jnp.array([10.0], dtype=jnp.float32),
            unit_tangent=unit_tangent,
            friction_const=1200.0,
            mu_d=0.2,
            mu_s=0.5,
            k_stick=jnp.array([20_000.0], dtype=jnp.float32),
            switch_speed=jnp.array([0.12], dtype=jnp.float32),
            tangential_load=jnp.array([3.0], dtype=jnp.float32),
        )
        # At rest, quasi-static branch should counter tangential load directly.
        np.testing.assert_allclose(float(force[0, 0]), 3.0, atol=1e-5)

    def test_quasi_static_load_clamps_to_coulomb_cap(self):
        unit_tangent = jnp.array([[1.0, 0.0]], dtype=jnp.float32)
        force = _compute_friction_with_stiction(
            v_tang=jnp.array([0.0], dtype=jnp.float32),
            normal_force_mag=jnp.array([10.0], dtype=jnp.float32),
            unit_tangent=unit_tangent,
            friction_const=1200.0,
            mu_d=0.2,
            mu_s=0.5,
            k_stick=jnp.array([20_000.0], dtype=jnp.float32),
            switch_speed=jnp.array([0.12], dtype=jnp.float32),
            tangential_load=jnp.array([200.0], dtype=jnp.float32),
        )
        # mu_s * N = 5.0 cap.
        np.testing.assert_allclose(float(force[0, 0]), 5.0, atol=1e-5)

    def test_switch_speed_controls_blend_strength(self):
        unit_tangent = jnp.array([[1.0, 0.0]], dtype=jnp.float32)
        force_small_switch = _compute_friction_with_stiction(
            v_tang=jnp.array([1e-3], dtype=jnp.float32),
            normal_force_mag=jnp.array([10.0], dtype=jnp.float32),
            unit_tangent=unit_tangent,
            friction_const=1200.0,
            mu_d=0.2,
            mu_s=0.5,
            k_stick=jnp.array([10_000.0], dtype=jnp.float32),
            switch_speed=jnp.array([1e-4], dtype=jnp.float32),
            tangential_load=jnp.array([3.0], dtype=jnp.float32),
        )
        force_large_switch = _compute_friction_with_stiction(
            v_tang=jnp.array([1e-3], dtype=jnp.float32),
            normal_force_mag=jnp.array([10.0], dtype=jnp.float32),
            unit_tangent=unit_tangent,
            friction_const=1200.0,
            mu_d=0.2,
            mu_s=0.5,
            k_stick=jnp.array([10_000.0], dtype=jnp.float32),
            switch_speed=jnp.array([0.1], dtype=jnp.float32),
            tangential_load=jnp.array([3.0], dtype=jnp.float32),
        )
        assert abs(float(force_large_switch[0, 0])) > abs(float(force_small_switch[0, 0])) * 2.0

    def test_fixed_switch_decouples_blend_from_k(self):
        unit_tangent = jnp.array([[1.0, 0.0]], dtype=jnp.float32)
        force_small_k = _compute_friction_with_stiction(
            v_tang=jnp.array([5e-4], dtype=jnp.float32),
            normal_force_mag=jnp.array([10.0], dtype=jnp.float32),
            unit_tangent=unit_tangent,
            friction_const=1200.0,
            mu_d=0.2,
            mu_s=0.5,
            k_stick=jnp.array([1_000.0], dtype=jnp.float32),
            switch_speed=jnp.array([0.1], dtype=jnp.float32),
            tangential_load=jnp.array([0.0], dtype=jnp.float32),
        )
        force_large_k = _compute_friction_with_stiction(
            v_tang=jnp.array([5e-4], dtype=jnp.float32),
            normal_force_mag=jnp.array([10.0], dtype=jnp.float32),
            unit_tangent=unit_tangent,
            friction_const=1200.0,
            mu_d=0.2,
            mu_s=0.5,
            k_stick=jnp.array([20_000.0], dtype=jnp.float32),
            switch_speed=jnp.array([0.1], dtype=jnp.float32),
            tangential_load=jnp.array([0.0], dtype=jnp.float32),
        )
        assert abs(float(force_large_k[0, 0])) > abs(float(force_small_k[0, 0])) * 5.0

    @staticmethod
    def _make_overlap_fixture(tangential_velocity: float = 0.5):
        world = EvoWorld()
        world.add_from_array(
            "terrain",
            np.array([[C.FIXED, C.FIXED, C.FIXED]], dtype=np.int32),
            0,
            0,
        )
        world.add_from_array("robot", np.array([[C.SOFT]], dtype=np.int32), 1, 1)

        template = compile_world_template(world, robot_name="robot")
        built = instantiate_world(template)

        positions = np.asarray(built.sim_state.positions, dtype=np.float32).copy()
        velocities = np.zeros_like(positions, dtype=np.float32)
        robot_points = np.asarray(built.robot_point_indices, dtype=np.int32)
        positions[robot_points, 1] -= np.float32(0.11)
        velocities[robot_points, 0] = np.float32(tangential_velocity)
        return built, jnp.asarray(positions), jnp.asarray(velocities), robot_points

    @staticmethod
    def _make_surface_fixture(
        surface: str,
        tangential_velocity: float = 0.0,
        normal_gap: float = 0.0,
    ):
        world = EvoWorld()
        if surface == "floor":
            world.add_from_array(
                "terrain",
                np.array([[C.FIXED, C.FIXED, C.FIXED]], dtype=np.int32),
                0,
                0,
            )
            world.add_from_array("robot", np.array([[C.SOFT]], dtype=np.int32), 1, 1)
            tangent_axis = 0
        elif surface == "wall":
            world.add_from_array(
                "terrain",
                np.array(
                    [[C.FIXED], [C.FIXED], [C.FIXED], [C.FIXED], [C.FIXED]],
                    dtype=np.int32,
                ),
                0,
                0,
            )
            world.add_from_array("robot", np.array([[C.SOFT]], dtype=np.int32), 1, 2)
            tangent_axis = 1
        else:  # pragma: no cover - defensive boundary
            raise ValueError(f"Unsupported surface fixture {surface!r}")

        template = compile_world_template(world, robot_name="robot")
        built = instantiate_world(template)
        positions = np.asarray(built.sim_state.positions, dtype=np.float32).copy()
        velocities = np.zeros_like(positions, dtype=np.float32)
        robot_points = np.asarray(built.robot_point_indices, dtype=np.int32)
        if surface == "floor":
            positions[robot_points, 1] += np.float32(normal_gap)
        else:
            positions[robot_points, 0] += np.float32(normal_gap)
        velocities[robot_points, tangent_axis] = np.float32(tangential_velocity)
        return built, jnp.asarray(positions), jnp.asarray(velocities), robot_points

    @staticmethod
    def _make_sloped_overlap_fixture(
        tangential_velocity: float = 0.0,
        normal_gap: float = -0.05,
    ):
        world = EvoWorld()
        world.add_from_array(
            "terrain",
            np.array([[C.SLOPE2_UP_RIGHT_LIGHT, C.SLOPE2_UP_RIGHT_HEAVY]], dtype=np.int32),
            0,
            0,
        )
        world.add_from_array("robot", np.array([[C.SOFT]], dtype=np.int32), 0, 1)

        template = compile_world_template(world, robot_name="robot", allow_experimental_slopes=True)
        built = instantiate_world(template)
        positions = np.asarray(built.sim_state.positions, dtype=np.float32).copy()
        velocities = np.zeros_like(positions, dtype=np.float32)
        robot_points = np.asarray(built.robot_point_indices, dtype=np.int32)
        positions[robot_points, 1] += np.float32(normal_gap)
        velocities[robot_points, 0] = np.float32(tangential_velocity)
        return built, jnp.asarray(positions), jnp.asarray(velocities), robot_points

    def test_stiction_disabled_matches_current(self):
        built, positions, velocities, _ = self._make_overlap_fixture(tangential_velocity=0.02)
        constants = default_physics_constants()
        forces_base = resolve_static_collisions(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            constants,
        )
        forces_disabled = resolve_static_collisions(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            constants._replace(
                static_friction_const=0.0,
                terrain_stiction_stiffness=0.0,
                terrain_stiction_scope=C.STICTION_SCOPE_ALL_STATIC,
            ),
        )
        np.testing.assert_allclose(
            np.asarray(forces_base),
            np.asarray(forces_disabled),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_fixed_floor_contacts_are_classified_as_non_sloped(self):
        built, positions, velocities, _ = self._make_surface_fixture(
            "floor", tangential_velocity=0.0, normal_gap=-0.002
        )
        result = resolve_static_collisions_with_contacts(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            default_physics_constants(),
            tangential_deformation=jnp.zeros_like(positions),
        )
        diag = result.diagnostics
        assert int(diag.floor_contact_count) > 0
        assert int(diag.slope_contact_count) == 0

    def test_floor_stiction_activates_on_flat_contacts_without_terrain_stiction(self):
        built, positions, velocities, _ = self._make_surface_fixture(
            "floor", tangential_velocity=1e-3, normal_gap=-0.002
        )
        dynamic_constants = default_physics_constants()
        floor_constants = dynamic_constants._replace(
            floor_static_friction_const=0.4,
            floor_stiction_stiffness=20_000.0,
            floor_stiction_switch_speed=0.05,
        )

        result_dynamic = resolve_static_collisions_with_contacts(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            dynamic_constants,
            tangential_deformation=jnp.zeros_like(positions),
        )
        result_floor = resolve_static_collisions_with_contacts(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            floor_constants,
            tangential_deformation=jnp.zeros_like(positions),
        )

        fx_dynamic = float(jnp.sum(jnp.abs(result_dynamic.forces[:, 0])))
        fx_floor = float(jnp.sum(jnp.abs(result_floor.forces[:, 0])))
        diag = result_floor.diagnostics
        assert fx_floor > fx_dynamic * 2.0
        assert int(diag.floor_stiction_active_count) > 0
        assert int(diag.stiction_active_count) == 0
        assert float(diag.sum_stick_cap) > 0.0
        assert int(diag.friction_cap_count) > 0
        assert int(jnp.sum(result_floor.memory_contact_mask.astype(jnp.int32))) == 0
        assert int(diag.persistent_contact_count) == 0

    def test_floor_stiction_skips_sloped_contacts(self):
        built, positions, velocities, _ = self._make_sloped_overlap_fixture(tangential_velocity=1e-3)
        dynamic_constants = default_physics_constants()
        floor_constants = dynamic_constants._replace(
            floor_static_friction_const=0.4,
            floor_stiction_stiffness=20_000.0,
            floor_stiction_switch_speed=0.05,
        )

        result_dynamic = resolve_static_collisions_with_contacts(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            dynamic_constants,
            tangential_deformation=jnp.zeros_like(positions),
        )
        result_floor = resolve_static_collisions_with_contacts(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            floor_constants,
            tangential_deformation=jnp.zeros_like(positions),
        )

        diag = result_floor.diagnostics
        assert int(diag.slope_contact_count) > 0
        assert int(diag.floor_stiction_active_count) == 0
        np.testing.assert_allclose(
            np.asarray(result_floor.forces),
            np.asarray(result_dynamic.forces),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_sloped_contacts_are_normal_only_even_with_friction_knobs(self):
        built, positions, velocities, _ = self._make_sloped_overlap_fixture(tangential_velocity=0.2)
        constants = default_physics_constants()._replace(
            stiction_enabled=True,
            static_friction_const=0.8,
            terrain_stiction_stiffness=20_000.0,
            terrain_stiction_switch_speed=0.12,
            terrain_stiction_scope=C.STICTION_SCOPE_ALL_STATIC,
            friction_model=C.FRICTION_MODEL_PROJECTED_ANCHOR,
            anchor_stiffness=50_000.0,
            slope_contact_mode=C.SLOPE_CONTACT_MODE_GAP_SUPPORT,
            slope_grip_enabled=True,
            slope_grip_static_friction_const=0.5,
            slope_grip_support_stiffness=20_000.0,
            slope_grip_switch_speed=0.12,
            slope_grip_scope=C.STICTION_SCOPE_FLOOR_SLOPE_ONLY,
        )

        result = resolve_static_collisions_with_contacts(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            constants,
            tangential_deformation=jnp.zeros_like(positions),
        )
        diag = result.diagnostics
        contact_mask = np.asarray(result.contact_mask)
        tangential_projection = np.sum(
            np.asarray(result.forces) * np.asarray(result.contact_tangent),
            axis=1,
        )

        assert int(diag.slope_contact_count) > 0
        assert float(diag.sum_normal_force) > 0.0
        assert float(diag.sum_abs_v_tang) > 0.0
        assert float(diag.sum_abs_friction) == 0.0
        assert int(diag.stiction_active_count) == 0
        assert int(diag.floor_stiction_active_count) == 0
        assert int(diag.support_point_count) == 0
        assert int(diag.persistent_contact_count) == 0
        assert int(jnp.sum(result.memory_contact_mask.astype(jnp.int32))) == 0
        np.testing.assert_allclose(
            tangential_projection[contact_mask],
            np.zeros(np.count_nonzero(contact_mask), dtype=np.float32),
            rtol=1e-6,
            atol=1e-5,
        )

    def test_stiction_at_low_velocity_exceeds_dynamic(self):
        built, positions, velocities, _ = self._make_overlap_fixture(tangential_velocity=1e-3)
        constants = default_physics_constants()

        forces_dynamic = resolve_static_collisions(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            constants,
        )
        fx_dynamic = float(jnp.sum(jnp.abs(forces_dynamic[:, 0])))

        forces_stiction = resolve_static_collisions(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            constants._replace(
                stiction_enabled=True,
                static_friction_const=0.5,
                terrain_stiction_stiffness=10_000.0,
                terrain_stiction_switch_speed=0.12,
                terrain_stiction_scope=C.STICTION_SCOPE_ALL_STATIC,
            ),
        )
        fx_stiction = float(jnp.sum(jnp.abs(forces_stiction[:, 0])))
        assert fx_stiction > fx_dynamic * 2.0

    def test_stiction_clamps_to_mu_s_cone(self):
        built, positions, velocities, robot_points = self._make_overlap_fixture(tangential_velocity=0.2)
        constants = default_physics_constants()._replace(
            stiction_enabled=True,
            static_friction_const=0.5,
            terrain_stiction_stiffness=20_000.0,
            terrain_stiction_switch_speed=0.12,
            terrain_stiction_scope=C.STICTION_SCOPE_ALL_STATIC,
        )
        forces = resolve_static_collisions(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            constants,
        )
        robot_forces = np.asarray(forces)[robot_points]
        fy = np.abs(robot_forces[:, 1])
        fx = np.abs(robot_forces[:, 0])
        active = fy > 1e-6
        assert np.any(active), "Expected active contacts for Coulomb cap check"
        assert np.all(fx[active] <= constants.static_friction_const * fy[active] + 1e-3)

    def test_static_collision_diagnostics_populated_on_contact(self):
        built, positions, velocities, _ = self._make_overlap_fixture(tangential_velocity=0.0)
        constants = default_physics_constants()._replace(
            stiction_enabled=True,
            static_friction_const=0.8,
            terrain_stiction_stiffness=20_000.0,
            terrain_stiction_switch_speed=0.12,
            terrain_stiction_scope=C.STICTION_SCOPE_ALL_STATIC,
            friction_model=C.FRICTION_MODEL_PROJECTED_ANCHOR,
            anchor_stiffness=50_000.0,
        )
        result = resolve_static_collisions_with_contacts(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            constants,
            tangential_deformation=jnp.zeros_like(positions),
        )
        diag = result.diagnostics
        assert int(diag.active_triple_count) > 0
        assert int(diag.colliding_count) > 0
        assert int(diag.contact_point_count) > 0
        assert float(diag.sum_normal_force) > 0.0
        assert float(diag.sum_abs_v_tang) >= 0.0
        assert float(diag.sum_abs_friction) >= 0.0

    def test_static_collision_diagnostics_zero_without_static_collider(self):
        positions = jnp.zeros((4, 2), dtype=jnp.float32)
        velocities = jnp.zeros((4, 2), dtype=jnp.float32)
        collision_data = make_collision_data(np.zeros((2, 2), dtype=np.int32), 2, 2)
        result = resolve_static_collisions_with_contacts(
            positions,
            velocities,
            collision_data,
            static_collider_data=None,
            constants=default_physics_constants(),
            tangential_deformation=jnp.zeros((4, 2), dtype=jnp.float32),
        )
        diag = result.diagnostics
        assert int(diag.active_triple_count) == 0
        assert int(diag.colliding_count) == 0
        assert int(diag.contact_point_count) == 0
        assert float(diag.sum_normal_force) == 0.0

    def test_static_collision_diagnostics_can_be_disabled_without_affecting_forces(self):
        built, positions, velocities, _ = self._make_overlap_fixture(tangential_velocity=0.0)
        enabled = default_physics_constants()._replace(
            diagnostics_enabled=True,
            stiction_enabled=True,
            static_friction_const=0.8,
            terrain_stiction_stiffness=20_000.0,
            terrain_stiction_switch_speed=0.12,
            terrain_stiction_scope=C.STICTION_SCOPE_ALL_STATIC,
        )
        disabled = enabled._replace(diagnostics_enabled=False)

        result_enabled = resolve_static_collisions_with_contacts(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            enabled,
            tangential_deformation=jnp.zeros_like(positions),
        )
        result_disabled = resolve_static_collisions_with_contacts(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            disabled,
            tangential_deformation=jnp.zeros_like(positions),
        )

        assert np.allclose(np.asarray(result_enabled.forces), np.asarray(result_disabled.forces))
        assert int(result_disabled.diagnostics.active_triple_count) == 0
        assert int(result_disabled.diagnostics.allocated_worklist_count) == 0

    def test_slope_grip_without_gap_support_is_inert(self):
        built, positions, velocities, _ = self._make_surface_fixture(
            "floor", tangential_velocity=0.0, normal_gap=0.002
        )
        result = resolve_static_collisions_with_contacts(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            default_physics_constants()._replace(
                slope_grip_enabled=True,
                slope_grip_static_friction_const=0.5,
                slope_grip_support_stiffness=20_000.0,
                slope_grip_switch_speed=0.12,
                slope_grip_scope=C.STICTION_SCOPE_FLOOR_SLOPE_ONLY,
            ),
            tangential_deformation=jnp.zeros_like(positions),
        )
        diag = result.diagnostics
        assert int(diag.colliding_count) == 0
        assert int(diag.contact_point_count) == 0
        assert int(diag.support_point_count) == 0
        assert float(diag.sum_support_normal_force) == 0.0
        assert int(jnp.sum(result.support_mask.astype(jnp.int32))) == 0

    def test_slope_gap_support_is_noop_for_non_colliding_floor_contact(self):
        built, positions, velocities, _ = self._make_surface_fixture(
            "floor", tangential_velocity=0.0, normal_gap=0.002
        )
        result = resolve_static_collisions_with_contacts(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            default_physics_constants()._replace(
                slope_contact_mode=C.SLOPE_CONTACT_MODE_GAP_SUPPORT,
                slope_grip_enabled=True,
                slope_grip_static_friction_const=0.5,
                slope_grip_support_stiffness=20_000.0,
                slope_grip_switch_speed=0.12,
                slope_grip_scope=C.STICTION_SCOPE_FLOOR_SLOPE_ONLY,
            ),
            tangential_deformation=jnp.zeros_like(positions),
        )
        diag = result.diagnostics
        assert int(diag.colliding_count) == 0
        assert int(diag.contact_point_count) == 0
        assert int(diag.support_point_count) == 0
        assert int(diag.support_without_collision_count) == 0
        assert float(diag.sum_normal_force) == 0.0
        assert float(diag.sum_support_normal_force) == 0.0
        assert float(diag.sum_support_abs_friction) == 0.0
        assert int(jnp.sum(result.contact_mask.astype(jnp.int32))) == 0
        assert int(jnp.sum(result.support_mask.astype(jnp.int32))) == 0

    def test_slope_grip_is_noop_for_non_colliding_wall_contact(self):
        built, positions, velocities, _ = self._make_surface_fixture(
            "wall", tangential_velocity=0.0, normal_gap=0.002
        )
        base = default_physics_constants()._replace(
            slope_contact_mode=C.SLOPE_CONTACT_MODE_GAP_SUPPORT,
            slope_grip_enabled=True,
            slope_grip_static_friction_const=0.5,
            slope_grip_support_stiffness=20_000.0,
            slope_grip_switch_speed=0.12,
        )
        diag_floor = resolve_static_collisions_with_contacts(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            base._replace(slope_grip_scope=C.STICTION_SCOPE_FLOOR_SLOPE_ONLY),
            tangential_deformation=jnp.zeros_like(positions),
        ).diagnostics
        diag_all = resolve_static_collisions_with_contacts(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            base._replace(slope_grip_scope=C.STICTION_SCOPE_ALL_STATIC),
            tangential_deformation=jnp.zeros_like(positions),
        ).diagnostics
        assert int(diag_floor.support_point_count) == 0
        assert int(diag_floor.support_wall_count) == 0
        assert int(diag_all.support_point_count) == 0
        assert int(diag_all.support_wall_count) == 0
        assert float(diag_all.sum_support_normal_force) == 0.0

    def test_slope_grip_scope_is_noop_for_non_colliding_floor_contact(self):
        built, positions, velocities, _ = self._make_surface_fixture(
            "floor", tangential_velocity=0.0, normal_gap=0.002
        )
        base = default_physics_constants()._replace(
            slope_contact_mode=C.SLOPE_CONTACT_MODE_GAP_SUPPORT,
            slope_grip_enabled=True,
            slope_grip_static_friction_const=0.5,
            slope_grip_support_stiffness=20_000.0,
            slope_grip_switch_speed=0.12,
        )
        diag_floor = resolve_static_collisions_with_contacts(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            base._replace(slope_grip_scope=C.STICTION_SCOPE_FLOOR_SLOPE_ONLY),
            tangential_deformation=jnp.zeros_like(positions),
        ).diagnostics
        diag_slope_only = resolve_static_collisions_with_contacts(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            base._replace(slope_grip_scope=C.STICTION_SCOPE_SLOPE_ONLY),
            tangential_deformation=jnp.zeros_like(positions),
        ).diagnostics
        assert int(diag_floor.support_floor_count) == 0
        assert int(diag_floor.support_point_count) == 0
        assert int(diag_slope_only.support_point_count) == 0

    def test_floor_scope_does_not_stiction_vertical_wall(self):
        world = EvoWorld()
        world.add_from_array(
            "wall",
            np.array(
                [[C.FIXED], [C.FIXED], [C.FIXED], [C.FIXED], [C.FIXED]],
                dtype=np.int32,
            ),
            0,
            0,
        )
        world.add_from_array("robot", np.array([[C.SOFT]], dtype=np.int32), 1, 2)
        built = instantiate_world(compile_world_template(world, robot_name="robot"))

        positions = np.asarray(built.sim_state.positions, dtype=np.float32).copy()
        velocities = np.zeros_like(positions, dtype=np.float32)
        robot_points = np.asarray(built.robot_point_indices, dtype=np.int32)
        positions[robot_points, 0] -= np.float32(0.11)
        velocities[robot_points, 1] = np.float32(0.03)
        positions_j = jnp.asarray(positions)
        velocities_j = jnp.asarray(velocities)

        base_constants = default_physics_constants()._replace(
            stiction_enabled=True,
            static_friction_const=0.5,
            terrain_stiction_stiffness=20_000.0,
            terrain_stiction_switch_speed=0.12,
        )
        forces_floor_scope = resolve_static_collisions(
            positions_j,
            velocities_j,
            built.collision_data,
            built.static_collider_data,
            base_constants._replace(
                terrain_stiction_scope=C.STICTION_SCOPE_FLOOR_SLOPE_ONLY,
            ),
        )
        forces_all_static = resolve_static_collisions(
            positions_j,
            velocities_j,
            built.collision_data,
            built.static_collider_data,
            base_constants._replace(
                terrain_stiction_scope=C.STICTION_SCOPE_ALL_STATIC,
            ),
        )
        fy_floor = float(jnp.sum(jnp.abs(forces_floor_scope[:, 1])))
        fy_all = float(jnp.sum(jnp.abs(forces_all_static[:, 1])))
        assert fy_all > fy_floor * 2.0

        diag_floor = resolve_static_collisions_with_contacts(
            positions_j,
            velocities_j,
            built.collision_data,
            built.static_collider_data,
            base_constants._replace(
                terrain_stiction_scope=C.STICTION_SCOPE_FLOOR_SLOPE_ONLY,
            ),
            tangential_deformation=jnp.zeros_like(positions_j),
        ).diagnostics
        diag_all = resolve_static_collisions_with_contacts(
            positions_j,
            velocities_j,
            built.collision_data,
            built.static_collider_data,
            base_constants._replace(
                terrain_stiction_scope=C.STICTION_SCOPE_ALL_STATIC,
            ),
            tangential_deformation=jnp.zeros_like(positions_j),
        ).diagnostics
        assert int(diag_all.stiction_active_count) > int(diag_floor.stiction_active_count)

    def test_slope_only_skips_stiction_on_non_sloped_edges(self):
        built, positions, velocities, _ = self._make_overlap_fixture(tangential_velocity=1e-3)
        dynamic_constants = default_physics_constants()
        stiction_constants = dynamic_constants._replace(
            stiction_enabled=True,
            static_friction_const=0.5,
            terrain_stiction_stiffness=20_000.0,
            terrain_stiction_switch_speed=0.12,
        )

        forces_dynamic = resolve_static_collisions(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            dynamic_constants,
        )
        forces_all_static = resolve_static_collisions(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            stiction_constants._replace(
                terrain_stiction_scope=C.STICTION_SCOPE_ALL_STATIC,
            ),
        )
        forces_slope_only = resolve_static_collisions(
            positions,
            velocities,
            built.collision_data,
            built.static_collider_data,
            stiction_constants._replace(
                terrain_stiction_scope=C.STICTION_SCOPE_SLOPE_ONLY,
            ),
        )

        fx_dynamic = float(jnp.sum(jnp.abs(forces_dynamic[:, 0])))
        fx_all_static = float(jnp.sum(jnp.abs(forces_all_static[:, 0])))
        assert fx_all_static > fx_dynamic * 2.0
        np.testing.assert_allclose(
            np.asarray(forces_slope_only),
            np.asarray(forces_dynamic),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_projected_anchor_uses_tangential_memory(self):
        built, positions, _, robot_points = self._make_overlap_fixture(tangential_velocity=0.0)
        velocities = jnp.zeros_like(positions)
        n_points = int(positions.shape[0])
        m_zero = jnp.zeros((n_points, 2), dtype=jnp.float32)
        m_mem = np.zeros((n_points, 2), dtype=np.float32)
        m_mem[robot_points, 0] = 5e-4
        m_mem = jnp.asarray(m_mem, dtype=jnp.float32)
        constants = default_physics_constants()._replace(
            stiction_enabled=True,
            static_friction_const=0.8,
            terrain_stiction_stiffness=20_000.0,
            terrain_stiction_switch_speed=0.12,
            terrain_stiction_scope=C.STICTION_SCOPE_ALL_STATIC,
            friction_model=C.FRICTION_MODEL_PROJECTED_ANCHOR,
            anchor_stiffness=50_000.0,
        )

        fx_zero = float(
            jnp.sum(
                jnp.abs(
                    resolve_static_collisions_with_contacts(
                        positions,
                        velocities,
                        built.collision_data,
                        built.static_collider_data,
                        constants,
                        tangential_deformation=m_zero,
                    ).forces[:, 0]
                )
            )
        )
        fx_mem = float(
            jnp.sum(
                jnp.abs(
                    resolve_static_collisions_with_contacts(
                        positions,
                        velocities,
                        built.collision_data,
                        built.static_collider_data,
                        constants,
                        tangential_deformation=m_mem,
                    ).forces[:, 0]
                )
            )
        )
        assert fx_mem > fx_zero + 1e-4

    def test_projected_deformation_uses_tangential_memory(self):
        built, positions, _, robot_points = self._make_overlap_fixture(tangential_velocity=0.0)
        velocities = jnp.zeros_like(positions)
        n_points = int(positions.shape[0])
        m_zero = jnp.zeros((n_points, 2), dtype=jnp.float32)
        m_mem = np.zeros((n_points, 2), dtype=np.float32)
        m_mem[robot_points, 0] = 5e-4
        m_mem = jnp.asarray(m_mem, dtype=jnp.float32)
        constants = default_physics_constants()._replace(
            stiction_enabled=True,
            static_friction_const=0.8,
            terrain_stiction_stiffness=20_000.0,
            terrain_stiction_switch_speed=0.12,
            terrain_stiction_scope=C.STICTION_SCOPE_ALL_STATIC,
            friction_model=C.FRICTION_MODEL_PROJECTED_DEFORMATION,
            deformation_stiffness=50_000.0,
            deformation_damping=0.0,
            breakaway_threshold=0.0,
        )

        fx_zero = float(
            jnp.sum(
                jnp.abs(
                    resolve_static_collisions_with_contacts(
                        positions,
                        velocities,
                        built.collision_data,
                        built.static_collider_data,
                        constants,
                        tangential_deformation=m_zero,
                    ).forces[:, 0]
                )
            )
        )
        fx_mem = float(
            jnp.sum(
                jnp.abs(
                    resolve_static_collisions_with_contacts(
                        positions,
                        velocities,
                        built.collision_data,
                        built.static_collider_data,
                        constants,
                        tangential_deformation=m_mem,
                    ).forces[:, 0]
                )
            )
        )
        assert fx_mem > fx_zero + 1e-4


class TestExperimentalSlopeGeometry:
    @staticmethod
    def _build_slope_2x1() -> tuple[np.ndarray, int, int]:
        height, width = 8, 20
        terrain = np.zeros((height, width), dtype=np.int32)
        # Ascend right: each row adds one 2x1 slope pair.
        for step in range(5):
            y = 1 + step
            x = 2 + 2 * step
            terrain[y, x] = C.SLOPE2_UP_RIGHT_LIGHT
            terrain[y, x + 1] = C.SLOPE2_UP_RIGHT_HEAVY
        # Support fill so the terrain object remains connected and stable.
        for y in range(height):
            for x in range(width):
                if terrain[y, x] == C.EMPTY:
                    continue
                terrain[:y, x] = np.where(
                    terrain[:y, x] == C.EMPTY, C.FIXED, terrain[:y, x]
                )
        spawn_x, spawn_y = 7, 8
        return terrain, spawn_x, spawn_y

    @staticmethod
    def _build_slope_3x1() -> tuple[np.ndarray, int, int]:
        height, width = 8, 24
        terrain = np.zeros((height, width), dtype=np.int32)
        # Ascend right: each row adds one 3x1 triplet.
        for step in range(5):
            y = 1 + step
            x = 2 + 3 * step
            terrain[y, x] = C.SLOPE3_UP_RIGHT_LIGHT
            terrain[y, x + 1] = C.SLOPE3_UP_RIGHT_MID
            terrain[y, x + 2] = C.SLOPE3_UP_RIGHT_HEAVY
        for y in range(height):
            for x in range(width):
                if terrain[y, x] == C.EMPTY:
                    continue
                terrain[:y, x] = np.where(
                    terrain[:y, x] == C.EMPTY, C.FIXED, terrain[:y, x]
                )
        spawn_x, spawn_y = 8, 8
        return terrain, spawn_x, spawn_y

    @pytest.mark.parametrize("slope_kind", ["2x1", "3x1"])
    def test_experimental_slope_terrain_compiles_as_static_geometry(self, slope_kind: str):
        if slope_kind == "2x1":
            terrain, spawn_x, spawn_y = self._build_slope_2x1()
        else:
            terrain, spawn_x, spawn_y = self._build_slope_3x1()
        world = EvoWorld()
        world.add_from_array("terrain", terrain, 0, 0)
        world.add_from_array(
            "robot",
            np.array([[C.H_ACT, C.SOFT, C.V_ACT]], dtype=np.int32),
            spawn_x,
            spawn_y,
        )
        built = instantiate_world(
            compile_world_template(world, robot_name="robot", allow_experimental_slopes=True)
        )

        static_types = np.asarray(built.static_collider_data.cell_types)
        assert built.static_collider_data.cell_vertices.shape[0] > 0
        assert np.any(
            np.isin(static_types, np.array(sorted(C.SLOPE_TYPES), dtype=np.int32))
        )


# ---------------------------------------------------------------------------
# Self-collision tests
# ---------------------------------------------------------------------------

class TestSelfCollision:

    @staticmethod
    def _overlap_robot_boxels(cd: CollisionData, H: int, W: int) -> jnp.ndarray:
        """Translate each robot boxel to overlap at world origin."""
        n_points = (H + 1) * (W + 1)
        positions = np.zeros((n_points, 2), dtype=np.float32)
        W1 = W + 1
        for gy in range(H + 1):
            for gx in range(W + 1):
                positions[gy * W1 + gx] = [gx * C.CELL_SIZE, gy * C.CELL_SIZE]

        boxel_mask = np.asarray(cd.boxel_mask, dtype=bool)
        boxel_is_robot = np.asarray(cd.boxel_is_robot, dtype=bool)
        world_vx = np.asarray(cd.boxel_world_vx, dtype=np.int32)
        world_vy = np.asarray(cd.boxel_world_vy, dtype=np.int32)
        boxel_corners = np.asarray(cd.boxel_corners, dtype=np.int32)

        active_robot = np.where(boxel_mask & boxel_is_robot)[0]
        for boxel_idx in active_robot:
            corners = boxel_corners[boxel_idx]
            positions[corners, 0] -= float(world_vx[boxel_idx]) * C.CELL_SIZE
            positions[corners, 1] -= float(world_vy[boxel_idx]) * C.CELL_SIZE
        return jnp.asarray(positions, dtype=jnp.float32)

    @staticmethod
    def _make_rect_reset_state(
        morph: np.ndarray,
        H: int,
        W: int,
        spawn_x: int,
        spawn_y: int,
    ) -> tuple[SimState, CollisionData]:
        dense = voxel_to_dense_mass_spring(morph, H=H, W=W)
        state, _, _ = make_sim_state(dense, spawn_x=spawn_x, spawn_y=spawn_y)
        cd = make_collision_data(_padded_morphology(morph, H, W), H, W)
        return state, cd

    def test_adjacent_no_self_collision(self):
        """Two adjacent robot boxels → no self-collision."""
        morph = np.array([[C.SOFT, C.SOFT]])
        padded = _padded_morphology(morph, H=5, W=5)
        cd = make_collision_data(padded, 5, 5)

        dense = voxel_to_dense_mass_spring(morph, H=5, W=5)
        state, _, _ = make_sim_state(dense, spawn_x=0, spawn_y=0)

        assert not bool(is_self_colliding(state.positions, cd))

    def test_rect_reset_is_not_self_colliding_across_spawn_x(self):
        """Reset self-collision should not depend on absolute x translation."""
        morph = np.full((3, 5), C.CONTRACTILE, dtype=int)
        H = W = 20
        _, cd = self._make_rect_reset_state(morph, H, W, spawn_x=0, spawn_y=5)
        dense = voxel_to_dense_mass_spring(morph, H=H, W=W)

        for spawn_x in range(16):
            state, _, _ = make_sim_state(dense, spawn_x=spawn_x, spawn_y=5)
            assert not bool(is_self_colliding(state.positions, cd)), spawn_x

    def test_rect_reset_translation_preserves_non_collision(self):
        """Pure translation of an undeformed body must not flip self-collision."""
        morph = np.full((3, 5), C.CONTRACTILE, dtype=int)
        H = W = 20
        state_left, cd = self._make_rect_reset_state(morph, H, W, spawn_x=5, spawn_y=5)
        state_right, _ = self._make_rect_reset_state(morph, H, W, spawn_x=10, spawn_y=5)

        robot_points = np.asarray(state_left.point_mask) & np.asarray(state_right.point_mask)
        delta = np.asarray(state_right.positions - state_left.positions)[robot_points]
        np.testing.assert_allclose(delta[:, 0], 5 * C.CELL_SIZE, atol=1e-7)
        np.testing.assert_allclose(delta[:, 1], 0.0, atol=1e-7)
        assert not bool(is_self_colliding(state_left.positions, cd))
        assert not bool(is_self_colliding(state_right.positions, cd))

    def test_non_adjacent_overlap_below_surface_threshold(self):
        """Two overlapping non-adjacent boxels stay below C++ collision threshold."""
        H, W = 6, 6
        morph_padded = np.zeros((H, W), dtype=int)
        morph_padded[0, 0] = C.SOFT
        morph_padded[0, 2] = C.SOFT  # Non-adjacent (dx=2)
        cd = make_collision_data(morph_padded, H, W)
        pos = self._overlap_robot_boxels(cd, H, W)
        assert not bool(is_self_colliding(pos, cd))

    def test_non_adjacent_overlap_uses_undirected_pair_count(self):
        """Three overlapping boxels should not double-count mirrored pair ordering."""
        H, W = 6, 6
        morph_padded = np.zeros((H, W), dtype=int)
        morph_padded[0, 0] = C.SOFT
        morph_padded[0, 2] = C.SOFT
        morph_padded[2, 0] = C.SOFT
        cd = make_collision_data(morph_padded, H, W)
        pos = self._overlap_robot_boxels(cd, H, W)
        assert not bool(is_self_colliding(pos, cd))

    def test_non_adjacent_overlap_above_surface_threshold(self):
        """Four overlapping non-adjacent boxels exceed C++ self-collision threshold."""
        H, W = 6, 6
        morph_padded = np.zeros((H, W), dtype=int)
        morph_padded[0, 0] = C.SOFT
        morph_padded[0, 2] = C.SOFT
        morph_padded[2, 0] = C.SOFT
        morph_padded[2, 2] = C.SOFT
        cd = make_collision_data(morph_padded, H, W)
        pos = self._overlap_robot_boxels(cd, H, W)
        assert bool(is_self_colliding(pos, cd))


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_backward_compatibility_none(self):
        """collision_data=None matches existing behavior (no collision)."""
        morph = np.array([[C.SOFT]])
        dense = voxel_to_dense_mass_spring(morph, H=5, W=5)
        state, topo, act_info = make_sim_state(dense, spawn_x=0, spawn_y=2)
        consts = default_physics_constants()

        # Without collision_data
        state_no_coll = physics_substep(state, topo, consts, collision_data=None)
        # With explicit None
        state_explicit = physics_substep(state, topo, consts)

        np.testing.assert_allclose(
            state_no_coll.positions, state_explicit.positions, atol=1e-10
        )

    def test_env_step_with_collision(self):
        """env_step runs with collision_data without error."""
        morph = np.array([[C.SOFT]])
        H, W = 5, 5
        dense = voxel_to_dense_mass_spring(morph, H=H, W=W)
        state, topo, act_info = make_sim_state(dense, spawn_x=0, spawn_y=2)
        consts = default_physics_constants()
        padded = _padded_morphology(morph, H, W)
        cd = make_collision_data(padded, H, W)

        action = jnp.zeros(act_info.cell_spring_indices.shape[0], dtype=jnp.float32)
        new_state = env_step(state, topo, consts, action, act_info, collision_data=cd)
        assert new_state.positions.shape == state.positions.shape


# ---------------------------------------------------------------------------
# JIT compatibility tests
# ---------------------------------------------------------------------------

class TestJIT:

    def test_jit_resolve_collisions(self):
        """jax.jit(resolve_collisions) runs without error."""
        morph = np.array([[C.SOFT]])
        H, W = 5, 5
        padded = _padded_morphology(morph, H, W)
        cd = make_collision_data(padded, H, W)
        consts = default_physics_constants()

        n_points = (H + 1) * (W + 1)
        pos = jnp.zeros((n_points, 2), dtype=jnp.float32)
        vel = jnp.zeros((n_points, 2), dtype=jnp.float32)

        jit_fn = jax.jit(resolve_collisions)
        forces = jit_fn(pos, vel, cd, consts)
        assert forces.shape == (n_points, 2)

    def test_jit_is_self_colliding(self):
        """jax.jit(is_self_colliding) runs without error."""
        morph = np.array([[C.SOFT]])
        H, W = 5, 5
        padded = _padded_morphology(morph, H, W)
        cd = make_collision_data(padded, H, W)

        n_points = (H + 1) * (W + 1)
        pos = jnp.zeros((n_points, 2), dtype=jnp.float32)

        jit_fn = jax.jit(is_self_colliding)
        result = jit_fn(pos, cd)
        assert result.shape == ()

    def test_jit_env_step_with_collision(self):
        """jax.jit(env_step) with collision_data runs without error."""
        morph = np.array([[C.SOFT]])
        H, W = 5, 5
        dense = voxel_to_dense_mass_spring(morph, H=H, W=W)
        state, topo, act_info = make_sim_state(dense, spawn_x=0, spawn_y=2)
        consts = default_physics_constants()
        padded = _padded_morphology(morph, H, W)
        cd = make_collision_data(padded, H, W)

        action = jnp.zeros(act_info.cell_spring_indices.shape[0], dtype=jnp.float32)

        jit_fn = jax.jit(env_step)
        new_state = jit_fn(state, topo, consts, action, act_info, cd)
        assert new_state.positions.shape == state.positions.shape


# ---------------------------------------------------------------------------
# Indexed collision field tests
# ---------------------------------------------------------------------------

class TestIndexedFields:

    def test_indexed_triples_nonempty(self):
        """Non-adjacent robot+terrain should produce indexed triples."""
        H, W = 5, 5
        morph_padded = np.zeros((H, W), dtype=int)
        morph_padded[0, 0] = C.FIXED
        morph_padded[2, 0] = C.SOFT
        morph_padded[0, 3] = C.SOFT

        cd = make_collision_data(morph_padded, H, W)

        active = np.array(cd.triple_active)
        n_active = int(np.sum(active))
        assert n_active > 0, "Expected some active triples"
        assert n_active == cd.n_triples

    def test_self_triples_nonempty(self):
        """Non-adjacent robot boxels should produce self-collision triples."""
        H, W = 5, 5
        morph_padded = np.zeros((H, W), dtype=int)
        morph_padded[0, 0] = C.SOFT
        morph_padded[0, 3] = C.SOFT  # non-adjacent robot

        cd = make_collision_data(morph_padded, H, W)

        active = np.array(cd.self_triple_active)
        n_active = int(np.sum(active))
        assert n_active > 0, "Expected some active self-triples"
        assert n_active == cd.n_self_triples

    def test_inactive_entries_zero(self):
        """Inactive indexed entries are zero."""
        morph = np.array([[C.SOFT]])
        padded = _padded_morphology(morph)
        cd = make_collision_data(padded, 5, 5)

        active = np.array(cd.triple_active)
        ti = np.array(cd.triple_i)
        tj = np.array(cd.triple_j)
        tk = np.array(cd.triple_k)

        # All entries after last active should be 0
        if not np.all(active):
            first_inactive = np.argmin(active)
            assert np.all(ti[first_inactive:] == 0)
            assert np.all(tj[first_inactive:] == 0)
            assert np.all(tk[first_inactive:] == 0)

    def test_n_triples_diagnostic(self):
        """n_triples matches active count."""
        H, W = 5, 5
        morph_padded = np.zeros((H, W), dtype=int)
        morph_padded[0, 0] = C.FIXED
        morph_padded[2, 0] = C.SOFT

        cd = make_collision_data(morph_padded, H, W)
        expected = int(np.sum(np.array(cd.triple_active)))
        assert cd.n_triples == expected

        expected_self = int(np.sum(np.array(cd.self_triple_active)))
        assert cd.n_self_triples == expected_self

        expected_surface_robot = int(
            np.sum(
                np.array(cd.boxel_mask)
                & np.array(cd.boxel_is_robot)
                & np.any(np.array(cd.surface_edge_mask), axis=1)
            )
        )
        assert cd.n_robot_surface_boxels == expected_surface_robot


# ---------------------------------------------------------------------------
# Indexed vs dense parity tests
# ---------------------------------------------------------------------------

class TestIndexedCollisionBehavior:

    def _make_robot_terrain_scenario(self, robot_y_offset=0.0):
        """Robot SOFT voxel pushed into terrain FIXED voxel."""
        H, W = 5, 5
        morph_padded = np.zeros((H, W), dtype=int)
        morph_padded[0, 0] = C.FIXED
        morph_padded[2, 0] = C.SOFT

        cd = make_collision_data(morph_padded, H, W)
        constants = default_physics_constants()

        n_points = (H + 1) * (W + 1)
        positions = np.zeros((n_points, 2), dtype=np.float32)
        W1 = W + 1
        for gy in range(H + 1):
            for gx in range(W + 1):
                positions[gy * W1 + gx] = [gx * C.CELL_SIZE, gy * C.CELL_SIZE]

        for gy in [2, 3]:
            for gx in [0, 1]:
                pidx = gy * W1 + gx
                positions[pidx, 1] += robot_y_offset

        pos_jax = jnp.array(positions)
        vel_true = jnp.zeros((n_points, 2), dtype=jnp.float32)
        return pos_jax, vel_true, cd, constants

    def test_resolve_collisions_no_penetration(self):
        """No penetration: indexed solver produces zero forces."""
        pos, vel, cd, consts = self._make_robot_terrain_scenario(0.0)
        forces = resolve_collisions(pos, vel, cd, consts)
        np.testing.assert_allclose(np.array(forces), 0.0, atol=1e-5)

    def test_resolve_collisions_penetration(self):
        """Penetration: indexed solver produces nonzero forces."""
        pos, vel, cd, consts = self._make_robot_terrain_scenario(-0.15)
        forces = resolve_collisions(pos, vel, cd, consts)
        total = float(jnp.sum(jnp.abs(forces)))
        assert total > 0, "Penetration should produce collision forces"

    def test_resolve_collisions_with_velocity(self):
        """With velocity: damping increases forces."""
        pos, _, cd, consts = self._make_robot_terrain_scenario(-0.15)
        n_points = pos.shape[0]
        W1 = 6

        vel_zero = jnp.zeros((n_points, 2), dtype=jnp.float32)
        forces_static = resolve_collisions(pos, vel_zero, cd, consts)

        vel = jnp.zeros((n_points, 2), dtype=jnp.float32)
        for gy in [2, 3]:
            for gx in [0, 1]:
                pidx = gy * W1 + gx
                vel = vel.at[pidx, 1].set(-1.0)

        forces_moving = resolve_collisions(pos, vel, cd, consts)
        static_mag = float(jnp.sum(jnp.abs(forces_static)))
        moving_mag = float(jnp.sum(jnp.abs(forces_moving)))
        assert moving_mag > static_mag


# ---------------------------------------------------------------------------
# Capacity guard tests
# ---------------------------------------------------------------------------

class TestCapacityGuard:

    def test_small_grid_passes(self):
        """Small grid should not raise."""
        check_triple_capacity(5, 5, 7)

    def test_huge_grid_raises(self):
        """Very large grid should raise ValueError."""
        with pytest.raises(ValueError, match="max_triples"):
            check_triple_capacity(100, 100, 7)

    def test_returns_bounds(self):
        """check_triple_capacity returns (triple_bound, self_bound)."""
        triple_bound, self_bound = check_triple_capacity(5, 5, 7)
        assert triple_bound > 0
        assert self_bound > 0

    def test_custom_max_triples(self):
        """Custom max_triples is checked instead of global constant."""
        # Should pass with large custom max
        check_triple_capacity(10, 10, 7, max_triples=1_000_000, max_self_triples=100_000)
        # Should fail with tiny custom max
        with pytest.raises(ValueError, match="max_triples"):
            check_triple_capacity(10, 10, 7, max_triples=10, max_self_triples=100_000)


# ---------------------------------------------------------------------------
# Custom max_triples buffer tests
# ---------------------------------------------------------------------------

class TestCustomMaxTriples:

    def test_make_collision_data_custom_buffer(self):
        """make_collision_data works with custom max_triples."""
        H, W = 5, 5
        morph = np.zeros((H, W), dtype=int)
        morph[0, 0] = C.FIXED
        morph[2, 0] = C.SOFT

        # Default buffer
        cd_default = make_collision_data(morph, H, W)
        n_real = cd_default.n_triples

        # Custom tight buffer (4x headroom)
        tight = max(64, n_real * 4)
        cd_tight = make_collision_data(morph, H, W, max_triples=tight, max_self_triples=64)

        # Active triples should match
        assert cd_tight.n_triples == cd_default.n_triples
        assert cd_tight.n_self_triples == cd_default.n_self_triples

        # Buffer sizes differ
        assert cd_tight.triple_i.shape[0] == tight
        assert cd_default.triple_i.shape[0] == C.MAX_TRIPLES

    def test_tight_buffer_collision_parity(self):
        """Collision forces identical with tight vs large buffer."""
        from jax_evogym.collision import resolve_collisions
        from jax_evogym.types import default_physics_constants

        H, W = 5, 5
        morph = np.zeros((H, W), dtype=int)
        morph[0, 0] = C.FIXED
        morph[2, 0] = C.SOFT

        cd_default = make_collision_data(morph, H, W)
        n_real = cd_default.n_triples
        tight = max(64, n_real * 4)
        cd_tight = make_collision_data(morph, H, W, max_triples=tight, max_self_triples=64)
        constants = default_physics_constants()

        # Create positions with penetration
        n_points = (H + 1) * (W + 1)
        positions = np.zeros((n_points, 2), dtype=np.float32)
        W1 = W + 1
        for gy in range(H + 1):
            for gx in range(W + 1):
                positions[gy * W1 + gx] = [gx * C.CELL_SIZE, gy * C.CELL_SIZE]
        for gy in [2, 3]:
            for gx in [0, 1]:
                positions[gy * W1 + gx, 1] -= 0.15

        pos = jnp.array(positions)
        vel = jnp.zeros((n_points, 2), dtype=jnp.float32)

        forces_default = resolve_collisions(pos, vel, cd_default, constants)
        forces_tight = resolve_collisions(pos, vel, cd_tight, constants)

        np.testing.assert_allclose(
            np.array(forces_tight), np.array(forces_default), atol=1e-5,
            err_msg="Tight buffer should produce identical collision forces"
        )

    def test_buffer_too_small_raises(self):
        """max_triples smaller than actual count raises ValueError."""
        H, W = 5, 5
        morph = np.zeros((H, W), dtype=int)
        morph[0, 0] = C.FIXED
        morph[2, 0] = C.SOFT
        morph[0, 3] = C.SOFT

        with pytest.raises(ValueError, match="max_triples"):
            make_collision_data(morph, H, W, max_triples=1)


class TestDynamicBroadphaseRouting:

    @staticmethod
    def _penetrating_positions(H: int, W: int) -> jnp.ndarray:
        n_points = (H + 1) * (W + 1)
        positions = np.zeros((n_points, 2), dtype=np.float32)
        W1 = W + 1
        for gy in range(H + 1):
            for gx in range(W + 1):
                positions[gy * W1 + gx] = [gx * C.CELL_SIZE, gy * C.CELL_SIZE]
        for gy in [2, 3]:
            for gx in [0, 1]:
                positions[gy * W1 + gx, 1] -= 0.15
        return jnp.asarray(positions, dtype=jnp.float32)

    def test_resolve_collisions_strategy_flag_controls_solver(self):
        H, W = 5, 5
        morph = np.zeros((H, W), dtype=int)
        morph[0, 0] = C.FIXED
        morph[2, 0] = C.SOFT
        cd = make_collision_data(morph, H, W)
        constants = default_physics_constants()

        pos = self._penetrating_positions(H, W)
        vel = jnp.zeros_like(pos)

        forces_static_wide = resolve_collisions(
            pos,
            vel,
            cd,
            constants,
            collision_strategy=0,
            dynamic_bin_size=0.2,
        )
        forces_static_tight = resolve_collisions(
            pos,
            vel,
            cd,
            constants,
            collision_strategy=0,
            dynamic_bin_size=0.001,
        )
        np.testing.assert_allclose(
            np.asarray(forces_static_wide),
            np.asarray(forces_static_tight),
            atol=1e-6,
        )

        forces_dynamic_wide = resolve_collisions(
            pos,
            vel,
            cd,
            constants,
            collision_strategy=1,
            dynamic_bin_size=0.2,
            dynamic_impl="dense_mask",
        )
        forces_dynamic_tight = resolve_collisions(
            pos,
            vel,
            cd,
            constants,
            collision_strategy=1,
            dynamic_bin_size=0.001,
            dynamic_impl="dense_mask",
        )
        norm_wide = float(jnp.linalg.norm(forces_dynamic_wide))
        norm_tight = float(jnp.linalg.norm(forces_dynamic_tight))
        assert norm_wide > 0.0
        assert norm_tight < norm_wide

    def test_dynamic_broadphase_impl_modes_produce_finite_forces(self):
        H, W = 5, 5
        morph = np.zeros((H, W), dtype=int)
        morph[0, 0] = C.FIXED
        morph[2, 0] = C.SOFT
        cd = make_collision_data(morph, H, W)
        constants = default_physics_constants()

        pos = self._penetrating_positions(H, W)
        vel = jnp.zeros_like(pos)
        dense = resolve_collisions_dynamic_broadphase(
            pos,
            vel,
            cd,
            constants,
            max_candidates=256,
            bin_size=0.2,
            impl="dense_mask",
        )
        sparse = resolve_collisions_dynamic_broadphase(
            pos,
            vel,
            cd,
            constants,
            max_candidates=256,
            bin_size=0.2,
            impl="sparse_candidate_list",
        )
        assert dense.shape == sparse.shape
        assert np.all(np.isfinite(np.asarray(dense)))
        assert np.all(np.isfinite(np.asarray(sparse)))
