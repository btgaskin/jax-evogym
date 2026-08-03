"""Simulation loop for the object-separated core runtime."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import NamedTuple

from jax_evogym import constants as C
from jax_evogym.actuators import set_actuator_goals, update_actuators
from jax_evogym.collision import (
    StaticCollisionWorklist,
    resolve_collisions,
    resolve_collisions_dynamic_broadphase,
    resolve_collisions_dynamic_broadphase_with_probe,
    resolve_static_collisions_with_contacts,
)
from jax_evogym.constants import PHYSICS_UPDATES_PER_STEP
from jax_evogym.constraints import (
    apply_friction_position_constraint,
    resolve_edge_constraints,
)
from jax_evogym.physics import rk4_step, symplectic_euler_step
from jax_evogym.types import (
    ActuatorInfo,
    CollisionData,
    PhysicsConstants,
    SimState,
    SpringTopology,
    StaticColliderData,
)


class DynamicBroadphaseProbe(NamedTuple):
    """Probe-only dynamic broadphase diagnostics for one physics substep."""

    selected_pair_mask: jnp.ndarray
    overflow: jnp.ndarray
    selected_triple_active: jnp.ndarray
    triple_active: jnp.ndarray
    dynamic_forces: jnp.ndarray


class PhysicsSubstepProbe(NamedTuple):
    """Probe-only diagnostics for one dynamic-broadphase physics substep."""

    dynamic_selected_pair_mask: jnp.ndarray
    dynamic_overflow: jnp.ndarray
    dynamic_selected_triple_active: jnp.ndarray
    dynamic_triple_active: jnp.ndarray
    dynamic_forces: jnp.ndarray
    static_forces: jnp.ndarray
    static_contact_mask: jnp.ndarray
    static_memory_contact_mask: jnp.ndarray
    tangential_deformation: jnp.ndarray
    static_manifold_active: jnp.ndarray
    post_update_actuator_spring_rest_length: jnp.ndarray
    post_integrator_positions: jnp.ndarray
    post_integrator_velocities: jnp.ndarray
    post_constraint_positions: jnp.ndarray
    post_anchor_positions: jnp.ndarray
    final_positions: jnp.ndarray
    velocity_true_positions_last: jnp.ndarray
    velocity_true_displacement: jnp.ndarray
    final_velocities: jnp.ndarray
    final_velocities_true: jnp.ndarray


def _finish_substep(
    state: SimState,
    topology: SpringTopology,
    constants: PhysicsConstants,
    static_result=None,
) -> SimState:
    state = update_actuators(state, topology, constants)
    integrator_code = jnp.asarray(constants.integrator, dtype=jnp.int32)

    def _step_rk4(_: None) -> SimState:
        return rk4_step(state, topology, constants)

    def _step_symplectic(_: None) -> SimState:
        return symplectic_euler_step(state, topology, constants)

    state = jax.lax.cond(
        integrator_code == jnp.int32(C.INTEGRATOR_SYMPLECTIC_EULER),
        _step_symplectic,
        _step_rk4,
        operand=None,
    )

    pre_pbd_positions = state.positions
    new_positions = resolve_edge_constraints(
        positions=state.positions,
        fixed=state.fixed,
        topology=topology,
        init_rest_length=state.spring_init_rest_length,
        spring_mask=state.spring_mask,
        point_mask=state.point_mask,
        rigid_spring_mask=state.rigid_spring_mask,
    )
    # Phase 2 PBD does not gate on fixed endpoints, so restore anchored points
    # exactly before any post-PBD friction anchor logic runs.
    new_positions = jnp.where(state.fixed, pre_pbd_positions, new_positions)
    constraint_mode = jnp.asarray(constants.friction_constraint_mode, dtype=jnp.int32)
    use_pbd_anchor = (
        jnp.bool_(constants.stiction_enabled)
        & (constraint_mode == jnp.int32(C.FRICTION_CONSTRAINT_MODE_TERRAIN_PBD_ANCHOR))
    )

    if static_result is not None:
        def _apply_anchor(pos_in):
            return apply_friction_position_constraint(
                pos_in,
                fixed=state.fixed,
                point_mask=state.point_mask,
                friction_anchor=state.friction_anchor,
                friction_anchor_active=state.friction_anchor_active,
                friction_anchor_no_contact_count=state.friction_anchor_no_contact_count,
                contact_mask=static_result.memory_contact_mask,
                contact_normal=static_result.memory_contact_normal,
                normal_force_mag=static_result.memory_contact_normal_force,
                gravity=constants.gravity,
                point_mass=C.POINT_MASS,
                static_friction_const=constants.static_friction_const,
                anchor_stiffness=constants.friction_anchor_stiffness,
                correction=constants.friction_anchor_correction,
                persist_substeps=constants.friction_anchor_persist_substeps,
            )

        def _skip_anchor(pos_in):
            return (
                pos_in,
                state.friction_anchor,
                state.friction_anchor_active,
                state.friction_anchor_no_contact_count,
            )

        (
            new_positions,
            friction_anchor,
            friction_anchor_active,
            friction_anchor_no_contact_count,
        ) = jax.lax.cond(
            use_pbd_anchor,
            _apply_anchor,
            _skip_anchor,
            new_positions,
        )
    else:
        friction_anchor = state.friction_anchor
        friction_anchor_active = state.friction_anchor_active
        friction_anchor_no_contact_count = state.friction_anchor_no_contact_count

    dt = constants.dt
    velocities_true = (new_positions - state.positions_last) / dt
    return state._replace(
        positions=new_positions,
        velocities_true=velocities_true,
        positions_last=new_positions,
        external_forces=jnp.zeros_like(state.external_forces),
        friction_anchor=friction_anchor,
        friction_anchor_active=friction_anchor_active,
        friction_anchor_no_contact_count=friction_anchor_no_contact_count,
    )


def _finish_substep_with_probe(
    state: SimState,
    topology: SpringTopology,
    constants: PhysicsConstants,
    *,
    static_result=None,
) -> tuple[SimState, dict[str, jnp.ndarray]]:
    state = update_actuators(state, topology, constants)
    post_update_actuator_spring_rest_length = state.spring_rest_length
    integrator_code = jnp.asarray(constants.integrator, dtype=jnp.int32)

    def _step_rk4(_: None) -> SimState:
        return rk4_step(state, topology, constants)

    def _step_symplectic(_: None) -> SimState:
        return symplectic_euler_step(state, topology, constants)

    state = jax.lax.cond(
        integrator_code == jnp.int32(C.INTEGRATOR_SYMPLECTIC_EULER),
        _step_symplectic,
        _step_rk4,
        operand=None,
    )

    post_integrator_positions = state.positions
    post_integrator_velocities = state.velocities
    pre_pbd_positions = state.positions
    new_positions = resolve_edge_constraints(
        positions=state.positions,
        fixed=state.fixed,
        topology=topology,
        init_rest_length=state.spring_init_rest_length,
        spring_mask=state.spring_mask,
        point_mask=state.point_mask,
        rigid_spring_mask=state.rigid_spring_mask,
    )
    new_positions = jnp.where(state.fixed, pre_pbd_positions, new_positions)
    post_constraint_positions = new_positions
    constraint_mode = jnp.asarray(constants.friction_constraint_mode, dtype=jnp.int32)
    use_pbd_anchor = (
        jnp.bool_(constants.stiction_enabled)
        & (constraint_mode == jnp.int32(C.FRICTION_CONSTRAINT_MODE_TERRAIN_PBD_ANCHOR))
    )

    if static_result is not None:

        def _apply_anchor(pos_in):
            return apply_friction_position_constraint(
                pos_in,
                fixed=state.fixed,
                point_mask=state.point_mask,
                friction_anchor=state.friction_anchor,
                friction_anchor_active=state.friction_anchor_active,
                friction_anchor_no_contact_count=state.friction_anchor_no_contact_count,
                contact_mask=static_result.memory_contact_mask,
                contact_normal=static_result.memory_contact_normal,
                normal_force_mag=static_result.memory_contact_normal_force,
                gravity=constants.gravity,
                point_mass=C.POINT_MASS,
                static_friction_const=constants.static_friction_const,
                anchor_stiffness=constants.friction_anchor_stiffness,
                correction=constants.friction_anchor_correction,
                persist_substeps=constants.friction_anchor_persist_substeps,
            )

        def _skip_anchor(pos_in):
            return (
                pos_in,
                state.friction_anchor,
                state.friction_anchor_active,
                state.friction_anchor_no_contact_count,
            )

        (
            new_positions,
            friction_anchor,
            friction_anchor_active,
            friction_anchor_no_contact_count,
        ) = jax.lax.cond(
            use_pbd_anchor,
            _apply_anchor,
            _skip_anchor,
            new_positions,
        )
    else:
        friction_anchor = state.friction_anchor
        friction_anchor_active = state.friction_anchor_active
        friction_anchor_no_contact_count = state.friction_anchor_no_contact_count

    post_anchor_positions = new_positions
    dt = constants.dt
    velocity_true_positions_last = state.positions_last
    velocity_true_displacement = new_positions - velocity_true_positions_last
    velocities_true = velocity_true_displacement / dt
    final_state = state._replace(
        positions=new_positions,
        velocities_true=velocities_true,
        positions_last=new_positions,
        external_forces=jnp.zeros_like(state.external_forces),
        friction_anchor=friction_anchor,
        friction_anchor_active=friction_anchor_active,
        friction_anchor_no_contact_count=friction_anchor_no_contact_count,
    )
    return final_state, {
        "post_update_actuator_spring_rest_length": post_update_actuator_spring_rest_length,
        "post_integrator_positions": post_integrator_positions,
        "post_integrator_velocities": post_integrator_velocities,
        "post_constraint_positions": post_constraint_positions,
        "post_anchor_positions": post_anchor_positions,
        "final_positions": final_state.positions,
        "velocity_true_positions_last": velocity_true_positions_last,
        "velocity_true_displacement": velocity_true_displacement,
        "final_velocities": final_state.velocities,
        "final_velocities_true": final_state.velocities_true,
    }


def _update_tangential_deformation(
    state: SimState,
    constants: PhysicsConstants,
    contact_mask: jnp.ndarray,
    contact_tangent: jnp.ndarray,
    contact_normal_force: jnp.ndarray,
) -> jnp.ndarray:
    # Avoid bool->numeric casts in hot runtime math; gate values via predicates.
    model = jnp.asarray(constants.friction_model, dtype=jnp.int32)
    constraint_mode = jnp.asarray(constants.friction_constraint_mode, dtype=jnp.int32)
    use_memory_model = (
        (model != jnp.int32(C.FRICTION_MODEL_NONE))
        & (constraint_mode != jnp.int32(C.FRICTION_CONSTRAINT_MODE_TERRAIN_PBD_ANCHOR))
    )
    m_old = jnp.asarray(state.tangential_deformation, dtype=jnp.float32)
    dt = jnp.asarray(constants.dt, dtype=jnp.float32)
    point_mask = state.point_mask[:, None]
    in_contact = (contact_mask & state.point_mask)[:, None]
    velocity = jnp.asarray(state.velocities_true, dtype=jnp.float32)
    m_trial = m_old + jnp.where(in_contact, dt * velocity, 0.0)

    tangent = jnp.asarray(contact_tangent, dtype=jnp.float32)
    tangent_norm = jnp.linalg.norm(tangent, axis=-1, keepdims=True)
    tangent_safe = tangent / jnp.maximum(tangent_norm, 1e-10)
    m_t = jnp.sum(m_trial * tangent_safe, axis=-1, keepdims=True)
    m_t_vec = m_t * tangent_safe
    m_n_vec = m_trial - m_t_vec
    normal_bleed = jnp.asarray(constants.deformation_normal_bleed, dtype=jnp.float32)
    m_contact = m_t_vec + m_n_vec * normal_bleed

    k_anchor = jnp.asarray(constants.anchor_stiffness, dtype=jnp.float32)
    k_deform = jnp.asarray(constants.deformation_stiffness, dtype=jnp.float32)
    k_model = jnp.where(model == jnp.int32(C.FRICTION_MODEL_PROJECTED_ANCHOR), k_anchor, k_deform)
    k_safe = jnp.maximum(k_model, 1e-10)
    mu_s = jnp.asarray(constants.static_friction_const, dtype=jnp.float32)
    cap_disp = (mu_s * jnp.asarray(contact_normal_force, dtype=jnp.float32) / k_safe)[:, None]
    m_t_contact = jnp.sum(m_contact * tangent_safe, axis=-1, keepdims=True)
    m_t_clamped = jnp.clip(m_t_contact, -cap_disp, cap_disp)
    m_capped = m_contact + (m_t_clamped - m_t_contact) * tangent_safe

    breakaway = jnp.asarray(constants.breakaway_threshold, dtype=jnp.float32)
    zero_tang = jnp.abs(m_t_clamped) < breakaway
    m_capped = jnp.where(zero_tang, m_capped - m_t_clamped * tangent_safe, m_capped)

    decay = jnp.asarray(constants.deformation_decay, dtype=jnp.float32)
    m_next = jnp.where(in_contact, m_capped, m_old * decay)
    m_next = jnp.where(point_mask, m_next, 0.0)
    return jnp.where(use_memory_model, m_next, m_old)


def physics_substep(
    state: SimState,
    topology: SpringTopology,
    constants: PhysicsConstants,
    collision_data: CollisionData | None = None,
    static_collider_data: StaticColliderData | None = None,
    static_collision_worklist: StaticCollisionWorklist | None = None,
    collision_strategy: jnp.ndarray | int = 0,
    dynamic_max_candidates: jnp.ndarray | int = 4096,
    dynamic_bin_size: jnp.ndarray | float = 0.2,
    dynamic_impl: str = "direct",
    dynamic_max_narrow: int | None = None,
) -> SimState:
    if collision_data is not None:
        dynamic_forces = resolve_collisions(
            state.positions,
            state.velocities_true,
            collision_data,
            constants,
            collision_strategy=collision_strategy,
            dynamic_max_candidates=dynamic_max_candidates,
            dynamic_bin_size=dynamic_bin_size,
            dynamic_impl=dynamic_impl,
            dynamic_max_narrow=dynamic_max_narrow,
        )
        static_result = resolve_static_collisions_with_contacts(
            state.positions,
            state.velocities_true,
            collision_data,
            static_collider_data,
            constants,
            static_collision_worklist=static_collision_worklist,
            tangential_deformation=state.tangential_deformation,
            static_manifold_cell_idx=state.static_manifold_cell_idx,
            static_manifold_edge_idx=state.static_manifold_edge_idx,
            static_manifold_active=state.static_manifold_active,
            static_manifold_no_contact_count=state.static_manifold_no_contact_count,
            static_manifold_normal_force=state.static_manifold_normal_force,
            static_manifold_normal=state.static_manifold_normal,
        )
        m_next = _update_tangential_deformation(
            state,
            constants,
            # Keep tangential memory scoped to friction-eligible terrain contacts.
            # Using the raw colliding contact mask would leak memory state into
            # geometry-only slope contacts.
            static_result.memory_contact_mask,
            static_result.memory_contact_tangent,
            static_result.memory_contact_normal_force,
        )
        state = state._replace(
            external_forces=dynamic_forces + static_result.forces,
            tangential_deformation=m_next,
            static_manifold_cell_idx=static_result.static_manifold_cell_idx,
            static_manifold_edge_idx=static_result.static_manifold_edge_idx,
            static_manifold_active=static_result.static_manifold_active,
            static_manifold_no_contact_count=static_result.static_manifold_no_contact_count,
            static_manifold_normal_force=static_result.static_manifold_normal_force,
            static_manifold_normal=static_result.static_manifold_normal,
        )
        return _finish_substep(state, topology, constants, static_result=static_result)
    return _finish_substep(state, topology, constants)


def env_step(
    state: SimState,
    topology: SpringTopology,
    constants: PhysicsConstants,
    action: jnp.ndarray,
    actuator_info: ActuatorInfo,
    collision_data: CollisionData | None = None,
    static_collider_data: StaticColliderData | None = None,
    static_collision_worklist: StaticCollisionWorklist | None = None,
    collision_strategy: jnp.ndarray | int = 0,
    dynamic_max_candidates: jnp.ndarray | int = 4096,
    dynamic_bin_size: jnp.ndarray | float = 0.2,
    dynamic_impl: str = "direct",
    physics_substeps: int = PHYSICS_UPDATES_PER_STEP,
    dynamic_max_narrow: int | None = None,
) -> SimState:
    expected = int(actuator_info.cell_spring_indices.shape[0])
    actual_shape = tuple(action.shape)
    if actual_shape != (expected,):
        raise ValueError(
            f"env_step expected scalar actuator action shape {(expected,)} but got {actual_shape}."
        )

    state = set_actuator_goals(state, actuator_info, action)

    def scan_fn(carry, _):
        next_state = physics_substep(
            carry,
            topology,
            constants,
            collision_data=collision_data,
            static_collider_data=static_collider_data,
            static_collision_worklist=static_collision_worklist,
            collision_strategy=collision_strategy,
            dynamic_max_candidates=dynamic_max_candidates,
            dynamic_bin_size=dynamic_bin_size,
            dynamic_impl=dynamic_impl,
            dynamic_max_narrow=dynamic_max_narrow,
        )
        return next_state, None

    state, _ = jax.lax.scan(scan_fn, state, None, length=int(physics_substeps))
    return state


def make_episode_step(
    topology,
    constants,
    actuator_info,
    collision_data=None,
    static_collider_data=None,
    static_collision_worklist=None,
    collision_strategy: jnp.ndarray | int = 0,
    dynamic_max_candidates: jnp.ndarray | int = 4096,
    dynamic_bin_size: jnp.ndarray | float = 0.2,
    dynamic_impl: str = "direct",
    physics_substeps: int = PHYSICS_UPDATES_PER_STEP,
    dynamic_max_narrow: int | None = None,
):
    def episode_step(state, action):
        new_state = env_step(
            state,
            topology,
            constants,
            action,
            actuator_info,
            collision_data=collision_data,
            static_collider_data=static_collider_data,
            static_collision_worklist=static_collision_worklist,
            collision_strategy=collision_strategy,
            dynamic_max_candidates=dynamic_max_candidates,
            dynamic_bin_size=dynamic_bin_size,
            dynamic_impl=dynamic_impl,
            physics_substeps=physics_substeps,
            dynamic_max_narrow=dynamic_max_narrow,
        )
        return new_state, new_state.positions

    return episode_step


def physics_substep_dynamic_broadphase(
    state: SimState,
    topology: SpringTopology,
    constants: PhysicsConstants,
    collision_data: CollisionData | None = None,
    static_collider_data: StaticColliderData | None = None,
    static_collision_worklist: StaticCollisionWorklist | None = None,
    dynamic_max_candidates: jnp.ndarray | int = 4096,
    dynamic_bin_size: jnp.ndarray | float = 0.2,
    dynamic_impl: str = "direct",
    dynamic_max_narrow: int | None = None,
) -> SimState:
    next_state, _ = physics_substep_dynamic_broadphase_with_probe(
        state,
        topology,
        constants,
        collision_data=collision_data,
        static_collider_data=static_collider_data,
        static_collision_worklist=static_collision_worklist,
        dynamic_max_candidates=dynamic_max_candidates,
        dynamic_bin_size=dynamic_bin_size,
        dynamic_impl=dynamic_impl,
        dynamic_max_narrow=dynamic_max_narrow,
    )
    return next_state


def physics_substep_dynamic_broadphase_with_probe(
    state: SimState,
    topology: SpringTopology,
    constants: PhysicsConstants,
    collision_data: CollisionData | None = None,
    static_collider_data: StaticColliderData | None = None,
    static_collision_worklist: StaticCollisionWorklist | None = None,
    dynamic_max_candidates: jnp.ndarray | int = 4096,
    dynamic_bin_size: jnp.ndarray | float = 0.2,
    dynamic_impl: str = "direct",
    dynamic_max_narrow: int | None = None,
) -> tuple[SimState, PhysicsSubstepProbe]:
    if collision_data is not None:
        dynamic_forces, dynamic_probe = resolve_collisions_dynamic_broadphase_with_probe(
            state.positions,
            state.velocities_true,
            collision_data,
            constants,
            max_candidates=dynamic_max_candidates,
            bin_size=dynamic_bin_size,
            impl=dynamic_impl,
            max_narrow=dynamic_max_narrow,
        )
        static_result = resolve_static_collisions_with_contacts(
            state.positions,
            state.velocities_true,
            collision_data,
            static_collider_data,
            constants,
            static_collision_worklist=static_collision_worklist,
            tangential_deformation=state.tangential_deformation,
            static_manifold_cell_idx=state.static_manifold_cell_idx,
            static_manifold_edge_idx=state.static_manifold_edge_idx,
            static_manifold_active=state.static_manifold_active,
            static_manifold_no_contact_count=state.static_manifold_no_contact_count,
            static_manifold_normal_force=state.static_manifold_normal_force,
            static_manifold_normal=state.static_manifold_normal,
        )
        m_next = _update_tangential_deformation(
            state,
            constants,
            # Keep tangential memory scoped to friction-eligible terrain contacts.
            static_result.memory_contact_mask,
            static_result.memory_contact_tangent,
            static_result.memory_contact_normal_force,
        )
        state = state._replace(
            external_forces=dynamic_forces + static_result.forces,
            tangential_deformation=m_next,
            static_manifold_cell_idx=static_result.static_manifold_cell_idx,
            static_manifold_edge_idx=static_result.static_manifold_edge_idx,
            static_manifold_active=static_result.static_manifold_active,
            static_manifold_no_contact_count=static_result.static_manifold_no_contact_count,
            static_manifold_normal_force=static_result.static_manifold_normal_force,
            static_manifold_normal=static_result.static_manifold_normal,
        )
        final_state, finish_probe = _finish_substep_with_probe(
            state,
            topology,
            constants,
            static_result=static_result,
        )
        return final_state, PhysicsSubstepProbe(
            dynamic_selected_pair_mask=dynamic_probe.selected_pair_mask,
            dynamic_overflow=dynamic_probe.overflow,
            dynamic_selected_triple_active=dynamic_probe.selected_triple_active,
            dynamic_triple_active=dynamic_probe.triple_active,
            dynamic_forces=dynamic_forces,
            static_forces=static_result.forces,
            static_contact_mask=static_result.contact_mask,
            static_memory_contact_mask=static_result.memory_contact_mask,
            tangential_deformation=m_next,
            static_manifold_active=static_result.static_manifold_active,
            post_update_actuator_spring_rest_length=finish_probe[
                "post_update_actuator_spring_rest_length"
            ],
            post_integrator_positions=finish_probe["post_integrator_positions"],
            post_integrator_velocities=finish_probe["post_integrator_velocities"],
            post_constraint_positions=finish_probe["post_constraint_positions"],
            post_anchor_positions=finish_probe["post_anchor_positions"],
            final_positions=finish_probe["final_positions"],
            velocity_true_positions_last=finish_probe["velocity_true_positions_last"],
            velocity_true_displacement=finish_probe["velocity_true_displacement"],
            final_velocities=finish_probe["final_velocities"],
            final_velocities_true=finish_probe["final_velocities_true"],
        )
    final_state, finish_probe = _finish_substep_with_probe(state, topology, constants)
    n_points = state.positions.shape[0]
    zero_contact_mask = jnp.zeros((n_points,), dtype=jnp.bool_)
    zero_bool = jnp.zeros((0,), dtype=jnp.bool_)
    return final_state, PhysicsSubstepProbe(
        dynamic_selected_pair_mask=zero_bool,
        dynamic_overflow=jnp.bool_(False),
        dynamic_selected_triple_active=zero_bool,
        dynamic_triple_active=zero_bool,
        dynamic_forces=jnp.zeros_like(state.positions),
        static_forces=jnp.zeros_like(state.positions),
        static_contact_mask=zero_contact_mask,
        static_memory_contact_mask=zero_contact_mask,
        tangential_deformation=jnp.zeros_like(state.tangential_deformation),
        static_manifold_active=zero_contact_mask,
        post_update_actuator_spring_rest_length=finish_probe[
            "post_update_actuator_spring_rest_length"
        ],
        post_integrator_positions=finish_probe["post_integrator_positions"],
        post_integrator_velocities=finish_probe["post_integrator_velocities"],
        post_constraint_positions=finish_probe["post_constraint_positions"],
        post_anchor_positions=finish_probe["post_anchor_positions"],
        final_positions=finish_probe["final_positions"],
        velocity_true_positions_last=finish_probe["velocity_true_positions_last"],
        velocity_true_displacement=finish_probe["velocity_true_displacement"],
        final_velocities=finish_probe["final_velocities"],
        final_velocities_true=finish_probe["final_velocities_true"],
    )


def env_step_dynamic_broadphase(
    state: SimState,
    topology: SpringTopology,
    constants: PhysicsConstants,
    action: jnp.ndarray,
    actuator_info: ActuatorInfo,
    collision_data: CollisionData | None = None,
    static_collider_data: StaticColliderData | None = None,
    static_collision_worklist: StaticCollisionWorklist | None = None,
    dynamic_max_candidates: jnp.ndarray | int = 4096,
    dynamic_bin_size: jnp.ndarray | float = 0.2,
    dynamic_impl: str = "direct",
    physics_substeps: int = PHYSICS_UPDATES_PER_STEP,
    dynamic_max_narrow: int | None = None,
) -> SimState:
    expected = int(actuator_info.cell_spring_indices.shape[0])
    actual_shape = tuple(action.shape)
    if actual_shape != (expected,):
        raise ValueError(
            "env_step_dynamic_broadphase expected scalar actuator action shape "
            f"{(expected,)} but got {actual_shape}."
        )

    state = set_actuator_goals(state, actuator_info, action)

    def scan_fn(carry, _):
        next_state = physics_substep_dynamic_broadphase(
            carry,
            topology,
            constants,
            collision_data=collision_data,
            static_collider_data=static_collider_data,
            static_collision_worklist=static_collision_worklist,
            dynamic_max_candidates=dynamic_max_candidates,
            dynamic_bin_size=dynamic_bin_size,
            dynamic_impl=dynamic_impl,
            dynamic_max_narrow=dynamic_max_narrow,
        )
        return next_state, None

    state, _ = jax.lax.scan(scan_fn, state, None, length=int(physics_substeps))
    return state
