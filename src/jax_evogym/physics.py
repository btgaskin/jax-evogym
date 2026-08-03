"""Force computation and integration kernels.

The stable square-cell force order follows EvoGym's C++ ``get_pos_vel_slope``
and RK4 update structure. Constants may include documented JAX-side extensions,
such as ground normal damping, so this module should be treated as a parity
target with explicit deviations rather than a byte-for-byte C++ clone.
"""

import jax
import jax.numpy as jnp

from jax_evogym.friction import blended_friction_scalar
from jax_evogym.types import SimState, SpringTopology, PhysicsConstants


def compute_forces(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    masses: jnp.ndarray,
    point_mask: jnp.ndarray,
    topology: SpringTopology,
    spring_rest_length: jnp.ndarray,
    spring_const: jnp.ndarray,
    spring_mask: jnp.ndarray,
    external_forces: jnp.ndarray,
    constants: PhysicsConstants,
) -> jnp.ndarray:
    """Compute raw forces on all points. Returns (n_points, 2).

    Force order matches C++ PhysicsEngine.cpp:487-599 exactly:
    spring → viscous drag → gravity → external → ground collision → ground friction.
    """
    n_points = positions.shape[0]
    forces = jnp.zeros((n_points, 2))

    # 1. Spring forces (C++ lines 498-531)
    a_idx = topology.a_idx
    b_idx = topology.b_idx
    vec_q_to_p = positions[a_idx] - positions[b_idx]  # (n_springs, 2)
    dist = jnp.sqrt(jnp.sum(vec_q_to_p ** 2, axis=1))  # (n_springs,)
    dist_safe = jnp.maximum(dist, 1e-10)
    force_mag = (dist - spring_rest_length) * spring_const / dist_safe  # (n_springs,)
    force_mag = force_mag * spring_mask  # zero out ghost springs
    spring_forces = vec_q_to_p * force_mag[:, None]  # (n_springs, 2)
    # Deterministic endpoint accumulation (avoids repeated-index scatter-add on GPU).
    endpoint_weights = (
        jax.nn.one_hot(b_idx, n_points, dtype=spring_forces.dtype)
        - jax.nn.one_hot(a_idx, n_points, dtype=spring_forces.dtype)
    )  # (n_springs, n_points)
    forces = forces + endpoint_weights.T @ spring_forces

    # 2. Viscous drag (C++ line 540)
    forces = forces + (-constants.viscous_drag * velocities)

    # 3. Gravity (C++ line 549)
    forces = forces + jnp.stack(
        [jnp.zeros((n_points,), dtype=forces.dtype), -constants.gravity * masses[:, 0]],
        axis=1,
    )

    # 4. External forces (C++ line 558)
    forces = forces + external_forces

    # 5. Ground collision (C++ lines 569-573)
    pos_y = positions[:, 1]
    is_colliding = pos_y < 0
    normal_force = jnp.where(is_colliding, -pos_y * constants.collision_const_ground, 0.0)
    # Ground normal velocity damping (not in C++ — added for realistic settling).
    # See constants.py for scaling rules: ζ = c / (2√(k·m)).
    vel_y = velocities[:, 1]
    ground_damping = jnp.where(is_colliding, -vel_y * constants.ground_vel_damping, 0.0)
    forces = forces + jnp.stack(
        [jnp.zeros((n_points,), dtype=forces.dtype), normal_force + ground_damping],
        axis=1,
    )

    # 6. Ground friction (C++ line 577)
    # friction = -ground_friction_const * dynamic_friction_const * normal_force
    #            * tanh(vel_x * normal_force * dynamic_friction_const)
    vel_x = velocities[:, 0]
    f_ground = blended_friction_scalar(
        vel_x,
        normal_force,
        friction_const=constants.ground_friction_const,
        mu_d=constants.dynamic_friction_const,
        mu_s=constants.ground_static_friction_const,
        k_stick=constants.ground_stiction_stiffness,
        switch_speed=constants.ground_stiction_switch_speed,
        normalize_dynamic_by_load=False,
    )
    f_ground = jnp.where(is_colliding, f_ground, 0.0)
    forces = forces + jnp.stack(
        [f_ground, jnp.zeros((n_points,), dtype=forces.dtype)],
        axis=1,
    )

    # 7. Ghost masking
    forces = forces * point_mask[:, None]

    return forces


def rk4_step(
    state: SimState,
    topology: SpringTopology,
    constants: PhysicsConstants,
) -> SimState:
    """Single RK4 integration step. Matches C++ PhysicsEngine.cpp:427-485.

    Fixed-point masking applied ONLY to the final weighted update,
    NOT at intermediate k stages (C++ lines 479-480, intermediate masking commented out).
    """
    dt = constants.dt
    pos = state.positions
    vel = state.velocities
    masses = state.masses
    point_mask = state.point_mask
    fixed = state.fixed

    def get_derivatives(p, v):
        """Returns (d_pos, d_vel) = (velocity, force/mass)."""
        f = compute_forces(
            positions=p,
            velocities=v,
            masses=masses,
            point_mask=point_mask,
            topology=topology,
            spring_rest_length=state.spring_rest_length,
            spring_const=state.spring_const,
            spring_mask=state.spring_mask,
            external_forces=state.external_forces,
            constants=constants,
        )
        mass_safe = jnp.maximum(masses, 1e-10)
        accel = f / mass_safe
        return v, accel

    # k1
    dp1, dv1 = get_derivatives(pos, vel)
    k1_pos = dp1 * dt
    k1_vel = dv1 * dt

    # k2
    dp2, dv2 = get_derivatives(pos + k1_pos * 0.5, vel + k1_vel * 0.5)
    k2_pos = dp2 * dt
    k2_vel = dv2 * dt

    # k3
    dp3, dv3 = get_derivatives(pos + k2_pos * 0.5, vel + k2_vel * 0.5)
    k3_pos = dp3 * dt
    k3_vel = dv3 * dt

    # k4
    dp4, dv4 = get_derivatives(pos + k3_pos, vel + k3_vel)
    k4_pos = dp4 * dt
    k4_vel = dv4 * dt

    # Weighted average
    update_pos = (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6.0
    update_vel = (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) / 6.0

    # Fixed point masking AFTER weighted average (C++ lines 479-480)
    update_pos = jnp.where(fixed, 0.0, update_pos)
    update_vel = jnp.where(fixed, 0.0, update_vel)

    # Ghost point masking
    update_pos = update_pos * point_mask[:, None]
    update_vel = update_vel * point_mask[:, None]

    return state._replace(
        positions=pos + update_pos,
        velocities=vel + update_vel,
    )


def symplectic_euler_step(
    state: SimState,
    topology: SpringTopology,
    constants: PhysicsConstants,
) -> SimState:
    """Single kick-drift (velocity-first) symplectic Euler integration step.

    Update order is pinned deliberately:
      1) v_{t+1} = v_t + dt * a(x_t, v_t)
      2) x_{t+1} = x_t + dt * v_{t+1}
    """
    dt = constants.dt
    pos = state.positions
    vel = state.velocities
    masses = state.masses
    point_mask = state.point_mask
    fixed = state.fixed

    forces = compute_forces(
        positions=pos,
        velocities=vel,
        masses=masses,
        point_mask=point_mask,
        topology=topology,
        spring_rest_length=state.spring_rest_length,
        spring_const=state.spring_const,
        spring_mask=state.spring_mask,
        external_forces=state.external_forces,
        constants=constants,
    )
    mass_safe = jnp.maximum(masses, 1e-10)
    accel = forces / mass_safe

    update_vel = accel * dt
    update_vel = jnp.where(fixed, 0.0, update_vel)
    update_vel = update_vel * point_mask[:, None]
    vel_next = vel + update_vel

    update_pos = vel_next * dt
    update_pos = jnp.where(fixed, 0.0, update_pos)
    update_pos = update_pos * point_mask[:, None]

    return state._replace(
        positions=pos + update_pos,
        velocities=vel_next,
    )
