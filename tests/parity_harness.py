"""Shared parity harness for the reference-trace parity tests."""

from __future__ import annotations

from collections.abc import Callable
import importlib
import json
import os
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jax_evogym import constants as C
from jax_evogym import (
    CONTRACTILE,
    EMPTY,
    FIXED,
    H_ACT,
    RIGID,
    SOFT,
    V_ACT,
    BuiltWorld,
    BridgeWalkerV0,
    ClimberV0,
    EvoWorld,
    HeightMaximizerV0,
    JumperV0,
    WalkerV0,
    WingspanMaximizerV0,
    compile_world_template,
    compile_world_templates,
    instantiate_world,
)
from jax_evogym.data import data_path
from jax_evogym.environments import UpStepperV0
from jax_evogym.sim import env_step
from jax_evogym.types import PhysicsConstants, default_physics_constants

REFERENCE_DIR = Path(__file__).resolve().parent / "reference_data"
EXPERIMENTAL_SLOPE_FIXTURE_DIR = Path(__file__).resolve().parent / "experimental_slope_fixtures"
GRID_TO_PHYSICAL = 0.1
BODY_3X1 = np.array([[H_ACT, SOFT, V_ACT]], dtype=np.int32)
WALKER_OBS_STEP_TOLERANCES = {0: 0.25, 10: 1.0, 25: 2.0, 50: 5.0}
MULTI_ENV_CONFIGS = [
    (HeightMaximizerV0, "height_max", "free_body"),
    (WingspanMaximizerV0, "wingspan_max", "free_body"),
    (ClimberV0, "climber", "terrain_contact"),
    (BridgeWalkerV0, "bridge_walker", "terrain_contact"),
    (JumperV0, "jumper", "terrain_contact"),
    (UpStepperV0, "upstepper", "terrain_contact"),
]
STABILITY_ENV_CONFIGS = [
    ("WalkerV0", WalkerV0),
    ("HeightMaximizerV0", HeightMaximizerV0),
    ("WingspanMaximizerV0", WingspanMaximizerV0),
    ("ClimberV0", ClimberV0),
    ("BridgeWalkerV0", BridgeWalkerV0),
    ("JumperV0", JumperV0),
    ("UpStepperV0", UpStepperV0),
]
SLOPE_GUARD_CASES = (
    ("walker_flat", "Walker-v0.json", 1, 1),
    ("upstepper_static", "UpStepper-v0.json", 1, 1),
    ("bridge_dynamic", "BridgeWalker-v0.json", 2, 5),
    ("free_body", None, 0, 3),
)
SLOPE_GUARD_BLOCKING_CASES = ("walker_flat", "upstepper_static", "free_body")
SLOPE_GUARD_DIAGNOSTIC_CASES = ("bridge_dynamic",)


def sinusoidal_action(n_actuators: int, step: int, n_steps: int = 50) -> jnp.ndarray:
    """Deterministic action schedule shared by the checked-in reference traces."""
    t = step / n_steps
    i = jnp.arange(n_actuators, dtype=jnp.float32)
    action = 1.0 + 0.5 * jnp.sin(2.0 * jnp.pi * t + i * 0.5)
    return jnp.clip(action, 0.6, 1.6)


def run_trajectory(env: Any, n_steps: int = 50) -> dict[str, np.ndarray]:
    """Run an environment for n_steps and collect the parity diagnostics."""
    obs, state = env.reset()
    all_obs = [np.array(obs)]
    all_rewards: list[float] = []
    all_dones: list[bool] = []
    all_com_x: list[float] = []
    all_com_y: list[float] = []

    active_pos = state.sim_state.positions[env.robot_point_indices]
    all_com_x.append(float(jnp.mean(active_pos[:, 0])))
    all_com_y.append(float(jnp.mean(active_pos[:, 1])))

    for step in range(n_steps):
        action = sinusoidal_action(env.n_actuators, step, n_steps)
        obs, state, reward, done = env.step(state, action)
        all_obs.append(np.array(obs))
        all_rewards.append(float(reward))
        all_dones.append(bool(done))
        active_pos = state.sim_state.positions[env.robot_point_indices]
        all_com_x.append(float(jnp.mean(active_pos[:, 0])))
        all_com_y.append(float(jnp.mean(active_pos[:, 1])))

    return {
        "obs": np.array(all_obs),
        "rewards": np.array(all_rewards),
        "dones": np.array(all_dones),
        "com_x": np.array(all_com_x),
        "com_y": np.array(all_com_y),
    }


def load_walker_reference(name: str, *, skip_if_missing: bool = False) -> dict[str, np.ndarray] | None:
    """Load Walker reference data or return None when skipping missing traces."""
    path = REFERENCE_DIR / f"walker_v0_{name}.npz"
    if not path.exists():
        if skip_if_missing:
            return None
        raise FileNotFoundError(f"Missing Walker reference data: {path}")
    return dict(np.load(path))


def load_multi_env_reference(name: str, *, skip_if_missing: bool = False) -> dict[str, np.ndarray] | None:
    """Load non-Walker reference data or return None when skipping missing traces."""
    path = REFERENCE_DIR / f"{name}_simple_3x1.npz"
    if not path.exists():
        if skip_if_missing:
            return None
        raise FileNotFoundError(f"Missing reference data: {path}")
    return dict(np.load(path))


def _top_level_import_smoke() -> dict[str, Any]:
    module = importlib.import_module("jax_evogym")
    exported = [
        "compile_world_template",
        "compile_world_templates",
        "instantiate_world",
        "WalkerV0",
        "BridgeWalkerV0",
        "WorldTemplateSet",
        "BuiltWorld",
    ]
    missing = [name for name in exported if not hasattr(module, name)]
    return {
        "ok": len(missing) == 0,
        "missing_exports": missing,
    }


def _make_suite(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass",
        "metrics": {},
        "failures": [],
    }


def _record_failure(suite: dict[str, Any], message: str) -> None:
    suite["failures"].append(message)
    suite["status"] = "fail"


def _store_metric(suite: dict[str, Any], key: str, value: Any) -> None:
    suite["metrics"][key] = _to_jsonable(value)


def _check(suite: dict[str, Any], condition: bool, message: str) -> None:
    if not condition:
        _record_failure(suite, message)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, jnp.floating)):
        return float(value)
    if isinstance(value, (np.integer, jnp.integer)):
        return int(value)
    if isinstance(value, (np.bool_, jnp.bool_)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, jnp.ndarray):
        return np.array(value).tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _reward_ratio_ok(cpp_cum: float, jax_cum: float, near_zero_tol: float) -> tuple[bool, dict[str, float]]:
    if abs(cpp_cum) < 0.1 and abs(jax_cum) < 0.1:
        abs_diff = abs(cpp_cum - jax_cum)
        return abs_diff < near_zero_tol, {
            "mode": "near_zero_abs_diff",
            "cpp_cumulative": cpp_cum,
            "jax_cumulative": jax_cum,
            "abs_diff": abs_diff,
            "limit": near_zero_tol,
        }

    if abs(cpp_cum) < 1e-6 or abs(jax_cum) < 1e-6:
        ok = abs(cpp_cum) < 1.0 and abs(jax_cum) < 1.0
        return ok, {
            "mode": "near_zero_magnitude",
            "cpp_cumulative": cpp_cum,
            "jax_cumulative": jax_cum,
        }

    ratio = abs(jax_cum / cpp_cum)
    return 0.1 < ratio < 10.0, {
        "mode": "ratio",
        "cpp_cumulative": cpp_cum,
        "jax_cumulative": jax_cum,
        "ratio": ratio,
    }


def _com_bounds_ok(com: np.ndarray, x_bounds: tuple[float, float], y_bounds: tuple[float, float]) -> bool:
    return (
        x_bounds[0] < float(com[0]) < x_bounds[1]
        and y_bounds[0] < float(com[1]) < y_bounds[1]
    )


def _walker_reference_case(
    *,
    case_name: str,
    reference: dict[str, np.ndarray],
    env: WalkerV0,
    check_obs_steps: bool,
) -> dict[str, Any]:
    case = _make_suite(case_name)
    jax_data = run_trajectory(env)
    obs_reset, state_reset = env.reset()

    _check(
        case,
        env.obs_dim == int(reference["obs_dim"]),
        f"{case_name}: obs_dim mismatch jax={env.obs_dim} ref={int(reference['obs_dim'])}",
    )
    _check(
        case,
        env.n_actuators == int(reference["n_actuators"]),
        f"{case_name}: n_actuators mismatch jax={env.n_actuators} ref={int(reference['n_actuators'])}",
    )

    cpp_vel = np.array(reference["vel_com_obs"][0])
    jax_vel = np.array(obs_reset[:2])
    vel_max_abs = float(np.max(np.abs(jax_vel)))
    _store_metric(case, "initial_vel_max_abs", vel_max_abs)
    _check(case, np.allclose(cpp_vel, 0.0, atol=1e-6), f"{case_name}: reference initial vel_com is not zero")
    _check(case, np.allclose(jax_vel, 0.0, atol=1e-6), f"{case_name}: JAX initial vel_com is not zero")

    if case_name == "simple":
        cpp_rel = np.array(reference["rel_pos_obs"][0])
        jax_rel = np.array(obs_reset[2:])
        n = len(cpp_rel) // 2
        rel_x_max_err = float(np.max(np.abs(np.sort(np.abs(cpp_rel[:n])) - np.sort(np.abs(jax_rel[:n])))))
        rel_y_max_err = float(np.max(np.abs(np.sort(np.abs(cpp_rel[n:])) - np.sort(np.abs(jax_rel[n:])))))
        _store_metric(case, "initial_rel_pos_x_max_error", rel_x_max_err)
        _store_metric(case, "initial_rel_pos_y_max_error", rel_y_max_err)
        _check(case, rel_x_max_err < 0.02, f"{case_name}: initial rel_pos x magnitudes diverge by {rel_x_max_err:.6f}")
        _check(case, rel_y_max_err < 0.02, f"{case_name}: initial rel_pos y magnitudes diverge by {rel_y_max_err:.6f}")

    active_pos = state_reset.sim_state.positions[env.robot_point_indices]
    jax_initial_com_x = float(jnp.mean(active_pos[:, 0]))
    cpp_initial_com_x = float(reference["com_x"][0] * GRID_TO_PHYSICAL)
    initial_com_x_err = abs(jax_initial_com_x - cpp_initial_com_x)
    _store_metric(case, "initial_com_x_error", initial_com_x_err)
    _check(case, initial_com_x_err < 0.15, f"{case_name}: initial COM x error {initial_com_x_err:.6f} >= 0.15")

    cpp_com_x = np.array(reference["com_x"] * GRID_TO_PHYSICAL)
    cpp_com_y = np.array(reference["com_y"] * GRID_TO_PHYSICAL)
    jax_com_x = np.array(jax_data["com_x"])
    jax_com_y = np.array(jax_data["com_y"])
    com_x_max_error = float(np.max(np.abs(cpp_com_x - jax_com_x)))
    com_y_max_error = float(np.max(np.abs(cpp_com_y - jax_com_y)))
    _store_metric(case, "com_x_max_error", com_x_max_error)
    _store_metric(case, "com_y_max_error", com_y_max_error)
    _check(case, com_x_max_error < 0.5, f"{case_name}: COM x max error {com_x_max_error:.6f} >= 0.5")
    _check(case, com_y_max_error < 0.5, f"{case_name}: COM y max error {com_y_max_error:.6f} >= 0.5")

    if check_obs_steps:
        obs_step_errors: dict[str, float] = {}
        for step, limit in WALKER_OBS_STEP_TOLERANCES.items():
            cpp_obs = np.array(reference["obs"][step])
            jax_obs = np.array(jax_data["obs"][step])
            max_diff = float(np.max(np.abs(cpp_obs - jax_obs)))
            obs_step_errors[str(step)] = max_diff
            _check(
                case,
                max_diff < limit,
                f"{case_name}: obs max diff at step {step} is {max_diff:.6f} >= {limit:.6f}",
            )
        _store_metric(case, "obs_step_max_errors", obs_step_errors)

        for step in (0, 10, 25):
            jax_rel = np.array(jax_data["obs"][step, 2:])
            n = len(jax_rel) // 2
            x_sum = float(np.abs(np.sum(jax_rel[:n])))
            y_sum = float(np.abs(np.sum(jax_rel[n:])))
            _check(case, x_sum < 1e-4, f"{case_name}: JAX rel_pos x sum {x_sum:.6f} at step {step}")
            _check(case, y_sum < 1e-4, f"{case_name}: JAX rel_pos y sum {y_sum:.6f} at step {step}")

            cpp_rel = np.array(reference["rel_pos_obs"][step])
            x_ref_sum = float(np.abs(np.sum(cpp_rel[:n])))
            y_ref_sum = float(np.abs(np.sum(cpp_rel[n:])))
            _check(case, x_ref_sum < 1e-4, f"{case_name}: ref rel_pos x sum {x_ref_sum:.6f} at step {step}")
            _check(case, y_ref_sum < 1e-4, f"{case_name}: ref rel_pos y sum {y_ref_sum:.6f} at step {step}")

    cpp_cum = float(np.sum(reference["rewards"]))
    jax_cum = float(np.sum(jax_data["rewards"]))
    reward_ok, reward_metrics = _reward_ratio_ok(cpp_cum, jax_cum, near_zero_tol=0.1)
    _store_metric(case, "reward", reward_metrics)
    _check(case, reward_ok, f"{case_name}: cumulative reward mismatch {reward_metrics}")

    _check(case, np.all(np.isfinite(jax_data["obs"])), f"{case_name}: NaN in observations")
    _check(case, np.all(np.isfinite(jax_data["rewards"])), f"{case_name}: NaN in rewards")

    final_pos = state_reset.sim_state.positions
    _ = final_pos
    return case


def evaluate_walker_reference_suite(*, skip_if_missing: bool = False) -> dict[str, Any]:
    suite = _make_suite("walker_reference")
    simple_ref = load_walker_reference("simple_3x1", skip_if_missing=skip_if_missing)
    biped_ref = load_walker_reference("biped_3x4", skip_if_missing=skip_if_missing)
    if simple_ref is None or biped_ref is None:
        suite["status"] = "skipped"
        suite["metrics"]["reason"] = "reference data missing"
        return suite

    simple_env = WalkerV0(BODY_3X1, include_deformation=False)
    biped_body = np.array(
        [
            [0, 0, 0],
            [H_ACT, SOFT, V_ACT],
            [H_ACT, SOFT, V_ACT],
            [H_ACT, SOFT, V_ACT],
        ],
        dtype=np.int32,
    )
    biped_env = WalkerV0(biped_body, include_deformation=False)

    simple_case = _walker_reference_case(
        case_name="simple",
        reference=simple_ref,
        env=simple_env,
        check_obs_steps=True,
    )
    biped_case = _walker_reference_case(
        case_name="biped",
        reference=biped_ref,
        env=biped_env,
        check_obs_steps=False,
    )

    _, biped_state = biped_env.reset()
    for step in range(50):
        action = sinusoidal_action(biped_env.n_actuators, step)
        _, biped_state, _, _ = biped_env.step(biped_state, action)
    active_pos = biped_state.sim_state.positions[biped_env.robot_point_indices]
    com = np.array(jnp.mean(active_pos, axis=0))
    biped_com_ok = _com_bounds_ok(com, (-1.0, 5.0), (-0.5, 2.0))
    if not biped_com_ok:
        _record_failure(
            biped_case,
            f"biped: bounded COM check failed with final COM {com.tolist()}",
        )
    _store_metric(biped_case, "final_com", com)

    suite["metrics"]["cases"] = {
        "simple": simple_case["metrics"],
        "biped": biped_case["metrics"],
    }
    suite["failures"].extend([f"simple::{msg}" for msg in simple_case["failures"]])
    suite["failures"].extend([f"biped::{msg}" for msg in biped_case["failures"]])
    if suite["failures"]:
        suite["status"] = "fail"
    return suite


def evaluate_multi_env_reference_suite(*, skip_if_missing: bool = False) -> dict[str, Any]:
    suite = _make_suite("multi_env_reference")
    cases: dict[str, Any] = {}

    for env_cls, ref_name, category in MULTI_ENV_CONFIGS:
        ref = load_multi_env_reference(ref_name, skip_if_missing=skip_if_missing)
        if ref is None:
            suite["status"] = "skipped"
            suite["metrics"]["reason"] = "reference data missing"
            return suite

        env = env_cls(BODY_3X1, include_deformation=False)
        case = _make_suite(ref_name)
        _check(case, env.obs_dim == int(ref["obs_dim"]), f"{ref_name}: obs_dim mismatch")
        _check(case, env.n_actuators == int(ref["n_actuators"]), f"{ref_name}: n_actuators mismatch")

        jax_data = run_trajectory(env)
        _check(case, np.all(np.isfinite(jax_data["obs"])), f"{ref_name}: NaN in observations")
        _check(case, np.all(np.isfinite(jax_data["rewards"])), f"{ref_name}: NaN in rewards")

        cpp_com_x = np.array(ref["com_x"] * GRID_TO_PHYSICAL)
        cpp_com_y = np.array(ref["com_y"] * GRID_TO_PHYSICAL)
        jax_com_x = np.array(jax_data["com_x"])
        jax_com_y = np.array(jax_data["com_y"])
        com_x_max_error = float(np.max(np.abs(cpp_com_x - jax_com_x)))
        com_y_max_error = float(np.max(np.abs(cpp_com_y - jax_com_y)))
        limit = 1.5 if category == "terrain_contact" else 1.0
        _store_metric(case, "com_x_max_error", com_x_max_error)
        _store_metric(case, "com_y_max_error", com_y_max_error)
        _store_metric(case, "com_error_limit", limit)
        _check(case, com_x_max_error < limit, f"{ref_name}: COM x max error {com_x_max_error:.6f} >= {limit:.6f}")
        _check(case, com_y_max_error < limit, f"{ref_name}: COM y max error {com_y_max_error:.6f} >= {limit:.6f}")

        cpp_cum = float(np.sum(ref["rewards"]))
        jax_cum = float(np.sum(jax_data["rewards"]))
        reward_ok, reward_metrics = _reward_ratio_ok(cpp_cum, jax_cum, near_zero_tol=0.15)
        _store_metric(case, "reward", reward_metrics)
        _check(case, reward_ok, f"{ref_name}: cumulative reward mismatch {reward_metrics}")

        if ref_name == "bridge_walker":
            _store_metric(case, "n_robot_points", env.n_robot_points)
            _check(case, env.n_robot_points == 8, f"{ref_name}: expected 8 robot points, saw {env.n_robot_points}")

        cases[ref_name] = case["metrics"]
        suite["failures"].extend([f"{ref_name}::{msg}" for msg in case["failures"]])

    suite["metrics"]["cases"] = cases
    if suite["failures"]:
        suite["status"] = "fail"
    return suite


def _stability_action(env: Any, step: int, n_steps: int = 50) -> jnp.ndarray:
    phase = step / n_steps
    val = 0.6 + 1.0 * jnp.sin(2 * jnp.pi * phase)
    return jnp.full(env.n_actuators, val)


def _run_stability_steps(env: Any, n_steps: int = 50):
    _, state = env.reset()
    for i in range(n_steps):
        action = _stability_action(env, i, n_steps)
        _, state, _, _ = env.step(state, action)
    return state


def evaluate_terrain_stability_suite() -> dict[str, Any]:
    suite = _make_suite("terrain_stability")
    cases: dict[str, Any] = {}

    for env_name, env_cls in STABILITY_ENV_CONFIGS:
        env = env_cls(BODY_3X1)
        final_state = _run_stability_steps(env)
        case = _make_suite(env_name)

        static_points = np.array(env.static_collider_data.point_positions)
        dynamic_fixed_mask = np.array(env._init_sim_state.fixed[:, 0])
        init_positions = np.array(env._init_sim_state.positions)
        final_positions = np.array(final_state.sim_state.positions)

        if dynamic_fixed_mask.any():
            fixed_init = init_positions[dynamic_fixed_mask]
            fixed_final = final_positions[dynamic_fixed_mask]
            identical = np.array_equal(fixed_init, fixed_final)
            _store_metric(case, "dynamic_fixed_point_count", int(dynamic_fixed_mask.sum()))
            _check(case, identical, f"{env_name}: dynamic fixed points moved")
        else:
            _store_metric(case, "dynamic_fixed_point_count", 0)

        if static_points.size > 0:
            static_unchanged = np.array_equal(
                static_points,
                np.array(env.render_info.static_point_positions),
            )
            _store_metric(case, "static_point_count", int(static_points.shape[0]))
            _check(case, static_unchanged, f"{env_name}: static collider positions diverged from render static points")
        else:
            _store_metric(case, "static_point_count", 0)

        a_idx = np.array(env.topology.a_idx)
        b_idx = np.array(env.topology.b_idx)
        spring_mask = np.array(env._init_sim_state.spring_mask)
        if dynamic_fixed_mask.any():
            both_fixed = dynamic_fixed_mask[a_idx] & dynamic_fixed_mask[b_idx] & spring_mask
            if np.any(both_fixed):
                pos_a = final_positions[a_idx[both_fixed]]
                pos_b = final_positions[b_idx[both_fixed]]
                actual_lengths = np.sqrt(np.sum((pos_a - pos_b) ** 2, axis=1))
                expected_lengths = np.array(env._init_sim_state.spring_init_rest_length)[both_fixed]
                max_err = float(np.max(np.abs(actual_lengths - expected_lengths)))
                _store_metric(case, "both_fixed_spring_max_error", max_err)
                _check(case, np.allclose(actual_lengths, expected_lengths, atol=1e-6, rtol=0.0), f"{env_name}: fixed-endpoint spring lengths drifted")
            else:
                _store_metric(case, "both_fixed_spring_max_error", None)
        else:
            _store_metric(case, "both_fixed_spring_max_error", None)

        cases[env_name] = case["metrics"]
        suite["failures"].extend([f"{env_name}::{msg}" for msg in case["failures"]])

    suite["metrics"]["cases"] = cases
    if suite["failures"]:
        suite["status"] = "fail"
    return suite


def _make_template_world() -> EvoWorld:
    world = EvoWorld()
    floor = np.array([[FIXED, FIXED, FIXED, FIXED, FIXED]], dtype=np.int32)
    robot_frame = np.array(
        [
            [H_ACT, SOFT, V_ACT],
            [SOFT, CONTRACTILE, SOFT],
        ],
        dtype=np.int32,
    )
    world.add_from_array("floor", floor, 0, 0)
    world.add_from_array("robot", robot_frame, 1, 2)
    return world


def _step_built_world(built: BuiltWorld, n_steps: int = 10) -> BuiltWorld:
    state = built.sim_state
    action = jnp.ones((int(built.actuator_info.cell_spring_indices.shape[0]),), dtype=jnp.float32)
    for _ in range(n_steps):
        state = env_step(
            state,
            built.topology,
            built.constants,
            action,
            built.actuator_info,
            collision_data=built.collision_data,
            static_collider_data=built.static_collider_data,
        )
    jax.block_until_ready(state.positions)
    return built._replace(sim_state=state)


def _make_guard_world(json_name: str | None, spawn_x: int, spawn_y: int) -> BuiltWorld:
    if json_name is None:
        world = EvoWorld()
    else:
        world = EvoWorld.from_json(data_path(json_name))
    world.add_from_array("robot", BODY_3X1, spawn_x, spawn_y)
    template = compile_world_template(world, robot_name="robot")
    return instantiate_world(template)


def _compute_robot_com_from_built_world(state, robot_point_indices) -> tuple[float, float]:
    active_pos = state.positions[robot_point_indices]
    return (
        float(jnp.mean(active_pos[:, 0])),
        float(jnp.mean(active_pos[:, 1])),
    )


def _run_guard_trace(built: BuiltWorld, *, n_steps: int) -> dict[str, Any]:
    state = built.sim_state
    positions = [np.array(state.positions)]
    velocities = [np.array(state.velocities_true)]
    spring_rest = [np.array(state.spring_rest_length)]
    spring_goal = [np.array(state.spring_rest_length_goal)]
    tangential_deformation = [np.array(state.tangential_deformation)]
    friction_anchor = [np.array(state.friction_anchor)]
    friction_anchor_active = [np.array(state.friction_anchor_active)]
    friction_anchor_no_contact_count = [np.array(state.friction_anchor_no_contact_count)]
    com_x: list[float] = []
    com_y: list[float] = []

    init_com_x, init_com_y = _compute_robot_com_from_built_world(state, built.robot_point_indices)
    com_x.append(init_com_x)
    com_y.append(init_com_y)

    n_actuators = int(built.actuator_info.cell_spring_indices.shape[0])
    for step in range(n_steps):
        action = sinusoidal_action(n_actuators, step, n_steps)
        state = env_step(
            state,
            built.topology,
            built.constants,
            action,
            built.actuator_info,
            collision_data=built.collision_data,
            static_collider_data=built.static_collider_data,
        )
        positions.append(np.array(state.positions))
        velocities.append(np.array(state.velocities_true))
        spring_rest.append(np.array(state.spring_rest_length))
        spring_goal.append(np.array(state.spring_rest_length_goal))
        tangential_deformation.append(np.array(state.tangential_deformation))
        friction_anchor.append(np.array(state.friction_anchor))
        friction_anchor_active.append(np.array(state.friction_anchor_active))
        friction_anchor_no_contact_count.append(np.array(state.friction_anchor_no_contact_count))
        next_com_x, next_com_y = _compute_robot_com_from_built_world(state, built.robot_point_indices)
        com_x.append(next_com_x)
        com_y.append(next_com_y)

    return {
        "positions": np.stack(positions),
        "velocities_true": np.stack(velocities),
        "spring_rest_length": np.stack(spring_rest),
        "spring_rest_length_goal": np.stack(spring_goal),
        "tangential_deformation": np.stack(tangential_deformation),
        "friction_anchor": np.stack(friction_anchor),
        "friction_anchor_active": np.stack(friction_anchor_active),
        "friction_anchor_no_contact_count": np.stack(friction_anchor_no_contact_count),
        "com_x": np.array(com_x, dtype=np.float32),
        "com_y": np.array(com_y, dtype=np.float32),
        "finite_positions": bool(np.all(np.isfinite(np.stack(positions)))),
        "finite_velocities": bool(np.all(np.isfinite(np.stack(velocities)))),
    }


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def evaluate_slope_mode_invariance_suite(
    *,
    candidate_constants: PhysicsConstants,
    baseline_constants: PhysicsConstants | None = None,
    n_steps: int = 6,
    position_tol: float = 1e-6,
    velocity_tol: float = 1e-6,
    spring_tol: float = 1e-6,
    deformation_tol: float = 1e-6,
    com_tol: float = 1e-6,
    case_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    suite = _make_suite("slope_mode_invariance")
    baseline = baseline_constants or default_physics_constants()
    selected = set(case_names) if case_names is not None else None
    cases: dict[str, Any] = {}

    for case_name, json_name, spawn_x, spawn_y in SLOPE_GUARD_CASES:
        if selected is not None and case_name not in selected:
            continue

        base_world = _make_guard_world(json_name, spawn_x, spawn_y)._replace(constants=baseline)
        candidate_world = _make_guard_world(json_name, spawn_x, spawn_y)._replace(
            constants=candidate_constants
        )
        base_trace = _run_guard_trace(base_world, n_steps=int(n_steps))
        candidate_trace = _run_guard_trace(candidate_world, n_steps=int(n_steps))

        case = _make_suite(case_name)
        position_max_err = _max_abs_diff(base_trace["positions"], candidate_trace["positions"])
        velocity_max_err = _max_abs_diff(base_trace["velocities_true"], candidate_trace["velocities_true"])
        spring_rest_max_err = _max_abs_diff(
            base_trace["spring_rest_length"], candidate_trace["spring_rest_length"]
        )
        spring_goal_max_err = _max_abs_diff(
            base_trace["spring_rest_length_goal"], candidate_trace["spring_rest_length_goal"]
        )
        tangential_deformation_max_err = _max_abs_diff(
            base_trace["tangential_deformation"], candidate_trace["tangential_deformation"]
        )
        friction_anchor_max_err = _max_abs_diff(
            base_trace["friction_anchor"], candidate_trace["friction_anchor"]
        )
        com_x_max_err = _max_abs_diff(base_trace["com_x"], candidate_trace["com_x"])
        com_y_max_err = _max_abs_diff(base_trace["com_y"], candidate_trace["com_y"])

        _store_metric(
            case,
            "max_errors",
            {
                "positions": position_max_err,
                "velocities_true": velocity_max_err,
                "spring_rest_length": spring_rest_max_err,
                "spring_rest_length_goal": spring_goal_max_err,
                "tangential_deformation": tangential_deformation_max_err,
                "friction_anchor": friction_anchor_max_err,
                "com_x": com_x_max_err,
                "com_y": com_y_max_err,
            },
        )
        _store_metric(
            case,
            "finite",
            {
                "baseline_positions": base_trace["finite_positions"],
                "baseline_velocities": base_trace["finite_velocities"],
                "candidate_positions": candidate_trace["finite_positions"],
                "candidate_velocities": candidate_trace["finite_velocities"],
            },
        )

        _check(
            case,
            bool(np.array_equal(base_trace["friction_anchor_active"], candidate_trace["friction_anchor_active"])),
            f"{case_name}: friction_anchor_active diverged",
        )
        _check(
            case,
            bool(
                np.array_equal(
                    base_trace["friction_anchor_no_contact_count"],
                    candidate_trace["friction_anchor_no_contact_count"],
                )
            ),
            f"{case_name}: friction_anchor_no_contact_count diverged",
        )
        _check(
            case,
            base_trace["finite_positions"] == candidate_trace["finite_positions"],
            f"{case_name}: finite_positions diverged",
        )
        _check(
            case,
            base_trace["finite_velocities"] == candidate_trace["finite_velocities"],
            f"{case_name}: finite_velocities diverged",
        )
        _check(
            case,
            position_max_err <= position_tol,
            f"{case_name}: positions max error {position_max_err:.6e} > {position_tol:.6e}",
        )
        _check(
            case,
            velocity_max_err <= velocity_tol,
            f"{case_name}: velocities_true max error {velocity_max_err:.6e} > {velocity_tol:.6e}",
        )
        _check(
            case,
            spring_rest_max_err <= spring_tol,
            f"{case_name}: spring_rest_length max error {spring_rest_max_err:.6e} > {spring_tol:.6e}",
        )
        _check(
            case,
            spring_goal_max_err <= spring_tol,
            f"{case_name}: spring_rest_length_goal max error {spring_goal_max_err:.6e} > {spring_tol:.6e}",
        )
        _check(
            case,
            tangential_deformation_max_err <= deformation_tol,
            f"{case_name}: tangential_deformation max error {tangential_deformation_max_err:.6e} > {deformation_tol:.6e}",
        )
        _check(
            case,
            friction_anchor_max_err <= deformation_tol,
            f"{case_name}: friction_anchor max error {friction_anchor_max_err:.6e} > {deformation_tol:.6e}",
        )
        _check(
            case,
            com_x_max_err <= com_tol,
            f"{case_name}: com_x max error {com_x_max_err:.6e} > {com_tol:.6e}",
        )
        _check(
            case,
            com_y_max_err <= com_tol,
            f"{case_name}: com_y max error {com_y_max_err:.6e} > {com_tol:.6e}",
        )

        cases[case_name] = case["metrics"]
        suite["failures"].extend([f"{case_name}::{msg}" for msg in case["failures"]])

    _store_metric(
        suite,
        "tolerances",
        {
            "position_tol": position_tol,
            "velocity_tol": velocity_tol,
            "spring_tol": spring_tol,
            "deformation_tol": deformation_tol,
            "com_tol": com_tol,
        },
    )
    _store_metric(suite, "cases", cases)
    if suite["failures"]:
        suite["status"] = "fail"
    return suite


def evaluate_slope_mode_guard_suite(
    *,
    candidate_constants: PhysicsConstants,
    baseline_constants: PhysicsConstants | None = None,
    n_steps: int = 6,
    position_tol: float = 1e-6,
    velocity_tol: float = 1e-6,
    spring_tol: float = 1e-6,
    deformation_tol: float = 1e-6,
    com_tol: float = 1e-6,
) -> dict[str, Any]:
    blocking = evaluate_slope_mode_invariance_suite(
        candidate_constants=candidate_constants,
        baseline_constants=baseline_constants,
        n_steps=int(n_steps),
        position_tol=float(position_tol),
        velocity_tol=float(velocity_tol),
        spring_tol=float(spring_tol),
        deformation_tol=float(deformation_tol),
        com_tol=float(com_tol),
        case_names=SLOPE_GUARD_BLOCKING_CASES,
    )
    diagnostic = evaluate_slope_mode_invariance_suite(
        candidate_constants=candidate_constants,
        baseline_constants=baseline_constants,
        n_steps=int(n_steps),
        position_tol=float(position_tol),
        velocity_tol=float(velocity_tol),
        spring_tol=float(spring_tol),
        deformation_tol=float(deformation_tol),
        com_tol=float(com_tol),
        case_names=SLOPE_GUARD_DIAGNOSTIC_CASES,
    )
    suite = _make_suite("slope_mode_guard")
    suite["status"] = str(blocking["status"])
    suite["failures"] = list(blocking["failures"])
    _store_metric(
        suite,
        "blocking_case_names",
        list(SLOPE_GUARD_BLOCKING_CASES),
    )
    _store_metric(
        suite,
        "diagnostic_case_names",
        list(SLOPE_GUARD_DIAGNOSTIC_CASES),
    )
    _store_metric(suite, "blocking", blocking)
    _store_metric(suite, "diagnostic", diagnostic)
    _store_metric(suite, "diagnostic_failures", list(diagnostic["failures"]))
    return suite


def evaluate_structural_semantics_suite() -> dict[str, Any]:
    suite = _make_suite("structural_semantics")

    walker_world = EvoWorld.from_json(data_path("Walker-v0.json"))
    walker_world.add_from_array("robot", BODY_3X1, 1, 1)
    walker_template = compile_world_template(walker_world, robot_name="robot")
    walker_built = instantiate_world(walker_template)
    walker_robot_template = walker_template.object_templates[walker_template.robot_template_index]

    _store_metric(suite, "walker_dynamic_terrain_point_count", int(walker_built.dynamic_terrain_point_indices.shape[0]))
    _store_metric(suite, "walker_static_point_count", int(walker_built.static_collider_data.point_positions.shape[0]))
    _check(suite, walker_built.dynamic_terrain_point_indices.shape[0] == 0, "Walker: pure fixed terrain leaked into dynamic terrain points")
    _check(
        suite,
        int(walker_built.sim_state.positions.shape[0]) == int(walker_robot_template.point_count),
        "Walker: dynamic point count is not robot-only",
    )
    _check(
        suite,
        int(walker_built.sim_state.spring_mask.shape[0]) == int(walker_robot_template.spring_count),
        "Walker: dynamic spring count is not robot-only",
    )
    _check(
        suite,
        walker_built.static_collider_data.point_positions.shape[0] > 0,
        "Walker: static fixed terrain was not compiled into static collider data",
    )

    bridge_env = BridgeWalkerV0(BODY_3X1)
    _store_metric(suite, "bridge_dynamic_terrain_point_count", int(bridge_env.dynamic_terrain_point_indices.shape[0]))
    _check(
        suite,
        bridge_env.dynamic_terrain_point_indices.shape[0] > 0,
        "BridgeWalker: mixed soft terrain is not present in the dynamic terrain point set",
    )

    robot_points = set(np.asarray(walker_built.robot_point_indices).tolist())
    terrain_points = set(np.asarray(walker_built.dynamic_terrain_point_indices).tolist())
    _check(suite, robot_points.isdisjoint(terrain_points), "Walker: robot and terrain point sets are not disjoint")
    _check(
        suite,
        np.any(walker_built.render_info.cell_mask & walker_built.render_info.cell_is_dynamic),
        "Walker: render info is missing dynamic cells",
    )
    _check(
        suite,
        np.any(walker_built.render_info.cell_mask & ~walker_built.render_info.cell_is_dynamic),
        "Walker: render info is missing static cells",
    )

    contractile_world = EvoWorld()
    contractile_world.add_from_array("robot", np.array([[CONTRACTILE]], dtype=np.int32), 0, 0)
    contractile_template = compile_world_template(contractile_world, robot_name="robot")
    contractile_built = instantiate_world(contractile_template)
    contractile_mask = np.asarray(contractile_built.sim_state.spring_mask, dtype=bool)
    contractile_const = np.asarray(contractile_built.sim_state.spring_const, dtype=np.float32)[contractile_mask]
    contractile_rest = np.asarray(contractile_built.sim_state.spring_rest_length, dtype=np.float32)[contractile_mask]
    diag_len = np.float32(np.sqrt(2.0) * C.CELL_SIZE)
    diag_mask = np.isclose(contractile_rest, diag_len, atol=1e-6)

    _store_metric(
        suite,
        "contractile_material_counts",
        {
            "active_springs": int(contractile_const.shape[0]),
            "main_const_count": int(np.count_nonzero(np.isclose(contractile_const, C.ACTUATOR_MAIN_K))),
            "diag_const_count": int(np.count_nonzero(np.isclose(contractile_const, C.ACTUATOR_DIAG_K))),
            "diag_rest_count": int(np.count_nonzero(diag_mask)),
        },
    )
    _check(suite, contractile_const.shape[0] == 6, "Contractile template did not activate 6 springs")
    _check(
        suite,
        np.count_nonzero(np.isclose(contractile_const, C.ACTUATOR_MAIN_K)) == 4,
        "Contractile template main springs did not use ACTUATOR_MAIN_K",
    )
    _check(
        suite,
        np.count_nonzero(np.isclose(contractile_const, C.ACTUATOR_DIAG_K)) == 2,
        "Contractile template diagonal springs did not use ACTUATOR_DIAG_K",
    )
    _check(
        suite,
        np.count_nonzero(diag_mask) == 2,
        "Contractile template did not keep two active diagonal springs",
    )

    return suite


def evaluate_mirroring_suite() -> dict[str, Any]:
    suite = _make_suite("mirroring")
    world = _make_template_world()
    template_set = compile_world_templates(world, robot_name="robot", mirror_mode="paired")
    _check(suite, template_set.mirror is not None, "mirror_mode='paired' did not produce a mirror template")
    if template_set.mirror is None:
        return suite

    primary = template_set.primary
    mirror = template_set.mirror
    _check(
        suite,
        len(primary.object_templates) == len(mirror.object_templates),
        "Primary/mirror object counts differ",
    )
    for primary_obj, mirror_obj in zip(primary.object_templates, mirror.object_templates, strict=True):
        expected_origin_x = primary.grid_w - (primary_obj.origin_x + primary_obj.grid.shape[1])
        _check(
            suite,
            mirror_obj.origin_x == expected_origin_x,
            f"{primary_obj.name}: mirrored origin_x={mirror_obj.origin_x} expected {expected_origin_x}",
        )
        _check(
            suite,
            mirror_obj.grid.shape == primary_obj.grid.shape,
            f"{primary_obj.name}: mirrored grid shape {mirror_obj.grid.shape} != {primary_obj.grid.shape}",
        )

    override = np.array(
        [
            [H_ACT, SOFT, V_ACT],
            [EMPTY, CONTRACTILE, RIGID],
        ],
        dtype=np.int32,
    )
    built_primary = instantiate_world(primary, robot_override=override)
    built_mirror = instantiate_world(mirror, robot_override=override)
    stepped_primary = _step_built_world(built_primary)
    stepped_mirror = _step_built_world(built_mirror)

    def _active_cell_count(built: BuiltWorld, dynamic: bool) -> int:
        mask = built.render_info.cell_mask & (built.render_info.cell_is_dynamic == dynamic)
        return int(np.count_nonzero(mask))

    _store_metric(
        suite,
        "primary_counts",
        {
            "objects": len(primary.object_templates),
            "actuators": int(built_primary.actuator_info.cell_spring_indices.shape[0]),
            "dynamic_cells": _active_cell_count(built_primary, True),
            "static_cells": _active_cell_count(built_primary, False),
        },
    )
    _store_metric(
        suite,
        "mirror_counts",
        {
            "objects": len(mirror.object_templates),
            "actuators": int(built_mirror.actuator_info.cell_spring_indices.shape[0]),
            "dynamic_cells": _active_cell_count(built_mirror, True),
            "static_cells": _active_cell_count(built_mirror, False),
        },
    )
    _check(
        suite,
        int(built_primary.actuator_info.cell_spring_indices.shape[0]) == int(built_mirror.actuator_info.cell_spring_indices.shape[0]),
        "Primary/mirror actuator counts differ",
    )
    _check(
        suite,
        _active_cell_count(built_primary, True) == _active_cell_count(built_mirror, True),
        "Primary/mirror dynamic cell counts differ",
    )
    _check(
        suite,
        _active_cell_count(built_primary, False) == _active_cell_count(built_mirror, False),
        "Primary/mirror static cell counts differ",
    )
    _check(
        suite,
        bool(jnp.all(jnp.isfinite(stepped_primary.sim_state.positions))),
        "Primary mirrored world produced non-finite positions",
    )
    _check(
        suite,
        bool(jnp.all(jnp.isfinite(stepped_mirror.sim_state.positions))),
        "Mirror world produced non-finite positions",
    )

    return suite


def evaluate_robot_override_suite() -> dict[str, Any]:
    suite = _make_suite("robot_override")
    world = _make_template_world()
    template = compile_world_template(world, robot_name="robot")

    override_dense = np.array(
        [
            [H_ACT, SOFT, V_ACT],
            [SOFT, CONTRACTILE, RIGID],
        ],
        dtype=np.int32,
    )
    override_sparse = np.array(
        [
            [H_ACT, EMPTY, V_ACT],
            [EMPTY, CONTRACTILE, EMPTY],
        ],
        dtype=np.int32,
    )

    built_dense = instantiate_world(template, robot_override=override_dense)
    built_sparse = instantiate_world(template, robot_override=override_sparse)

    dense_robot_points = int(np.count_nonzero(np.asarray(built_dense.robot_point_mask)))
    sparse_robot_points = int(np.count_nonzero(np.asarray(built_sparse.robot_point_mask)))
    dense_springs = int(np.count_nonzero(np.asarray(built_dense.sim_state.spring_mask)))
    sparse_springs = int(np.count_nonzero(np.asarray(built_sparse.sim_state.spring_mask)))

    _store_metric(
        suite,
        "dense",
        {
            "robot_points": dense_robot_points,
            "active_springs": dense_springs,
            "actuators": int(built_dense.actuator_info.cell_spring_indices.shape[0]),
            "robot_voxels": int(built_dense.deformation_info.n_robot_voxels),
        },
    )
    _store_metric(
        suite,
        "sparse",
        {
            "robot_points": sparse_robot_points,
            "active_springs": sparse_springs,
            "actuators": int(built_sparse.actuator_info.cell_spring_indices.shape[0]),
            "robot_voxels": int(built_sparse.deformation_info.n_robot_voxels),
        },
    )
    _check(suite, dense_robot_points != sparse_robot_points, "robot_override did not change active robot point counts")
    _check(suite, dense_springs != sparse_springs, "robot_override did not change active spring counts")
    _check(suite, sparse_robot_points < dense_robot_points, "sparse override did not remove robot activity")
    _check(
        suite,
        int(built_dense.deformation_info.n_robot_voxels) == int(np.count_nonzero(override_dense != EMPTY)),
        "dense override deformation voxel count is inconsistent",
    )
    _check(
        suite,
        int(built_sparse.deformation_info.n_robot_voxels) == int(np.count_nonzero(override_sparse != EMPTY)),
        "sparse override deformation voxel count is inconsistent",
    )
    _check(
        suite,
        int(built_dense.actuator_info.cell_spring_indices.shape[0]) == int(np.count_nonzero(np.isin(override_dense, [H_ACT, V_ACT]))),
        "dense override single-axis actuator count is inconsistent",
    )
    _check(
        suite,
        int(built_sparse.actuator_info.cell_spring_indices.shape[0]) == int(np.count_nonzero(np.isin(override_sparse, [H_ACT, V_ACT]))),
        "sparse override single-axis actuator count is inconsistent",
    )

    dense_h_expected = int(np.count_nonzero(np.isin(override_dense, [H_ACT, CONTRACTILE])))
    dense_v_expected = int(np.count_nonzero(np.isin(override_dense, [V_ACT, CONTRACTILE])))
    sparse_h_expected = int(np.count_nonzero(np.isin(override_sparse, [H_ACT, CONTRACTILE])))
    sparse_v_expected = int(np.count_nonzero(np.isin(override_sparse, [V_ACT, CONTRACTILE])))

    dense_h_active = int(np.count_nonzero(np.asarray(built_dense.per_axis_info.h_spring_pairs).any(axis=1)))
    dense_v_active = int(np.count_nonzero(np.asarray(built_dense.per_axis_info.v_spring_pairs).any(axis=1)))
    sparse_h_active = int(np.count_nonzero(np.asarray(built_sparse.per_axis_info.h_spring_pairs).any(axis=1)))
    sparse_v_active = int(np.count_nonzero(np.asarray(built_sparse.per_axis_info.v_spring_pairs).any(axis=1)))

    _store_metric(
        suite,
        "per_axis_counts",
        {
            "dense_h_expected": dense_h_expected,
            "dense_h_active": dense_h_active,
            "dense_v_expected": dense_v_expected,
            "dense_v_active": dense_v_active,
            "sparse_h_expected": sparse_h_expected,
            "sparse_h_active": sparse_h_active,
            "sparse_v_expected": sparse_v_expected,
            "sparse_v_active": sparse_v_active,
        },
    )
    _check(suite, dense_h_active == dense_h_expected, "dense override horizontal per-axis actuator count is inconsistent")
    _check(suite, dense_v_active == dense_v_expected, "dense override vertical per-axis actuator count is inconsistent")
    _check(suite, sparse_h_active == sparse_h_expected, "sparse override horizontal per-axis actuator count is inconsistent")
    _check(suite, sparse_v_active == sparse_v_expected, "sparse override vertical per-axis actuator count is inconsistent")

    return suite


def evaluate_slope_semantics_suite() -> dict[str, Any]:
    suite = _make_suite("slope_semantics")
    ramp_world = EvoWorld.from_json(str(EXPERIMENTAL_SLOPE_FIXTURE_DIR / "ramp_demo.json"))
    ramp_world.add_from_array("robot", BODY_3X1, 1, 1)

    template_set = compile_world_templates(
        ramp_world,
        robot_name="robot",
        mirror_mode="paired",
        allow_experimental_slopes=True,
    )
    built_primary = instantiate_world(template_set.primary)
    _check(suite, template_set.mirror is not None, "ramp_demo did not produce a mirrored template")
    if template_set.mirror is None:
        return suite
    built_mirror = instantiate_world(template_set.mirror)

    primary_static_types = np.asarray(built_primary.static_collider_data.cell_types)
    mirror_static_types = np.asarray(built_mirror.static_collider_data.cell_types)
    primary_render_slope_mask = (
        np.asarray(built_primary.render_info.cell_mask)
        & np.isin(
            np.asarray(built_primary.render_info.cell_types),
            [C.SLOPE_UP_RIGHT, C.SLOPE_UP_LEFT],
        )
    )
    primary_render_slope_dynamic = primary_render_slope_mask & np.asarray(built_primary.render_info.cell_is_dynamic)

    _store_metric(
        suite,
        "ramp_demo",
        {
            "primary_static_points": int(built_primary.static_collider_data.point_positions.shape[0]),
            "primary_slope_cells": int(np.count_nonzero(primary_static_types == C.SLOPE_UP_RIGHT)),
            "mirror_slope_cells": int(np.count_nonzero(mirror_static_types == C.SLOPE_UP_LEFT)),
            "terrain_samples": int(built_primary.fixed_terrain_sample_positions.shape[0]),
        },
    )
    _check(suite, np.count_nonzero(primary_static_types == C.SLOPE_UP_RIGHT) > 0, "ramp_demo primary world is missing slope cells")
    _check(suite, np.count_nonzero(mirror_static_types == C.SLOPE_UP_LEFT) > 0, "ramp_demo mirror did not swap slope orientation")
    _check(suite, built_primary.dynamic_terrain_point_indices.shape[0] == 0, "ramp_demo leaked static terrain into dynamic terrain points")
    _check(suite, np.any(primary_render_slope_mask), "ramp_demo render info is missing slope cells")
    _check(suite, not np.any(primary_render_slope_dynamic), "ramp_demo slope cells were marked dynamic in render info")
    _check(
        suite,
        built_primary.fixed_terrain_sample_positions.shape[0] > built_primary.static_collider_data.point_positions.shape[0],
        "ramp_demo did not add extra terrain samples for exposed slope faces",
    )

    stepped_primary = _step_built_world(built_primary)
    stepped_mirror = _step_built_world(built_mirror)
    _check(
        suite,
        bool(jnp.all(jnp.isfinite(stepped_primary.sim_state.positions))),
        "ramp_demo primary rollout produced non-finite positions",
    )
    _check(
        suite,
        bool(jnp.all(jnp.isfinite(stepped_mirror.sim_state.positions))),
        "ramp_demo mirrored rollout produced non-finite positions",
    )

    return suite


def evaluate_smoke_suite() -> dict[str, Any]:
    suite = _make_suite("smoke")

    import_check = _top_level_import_smoke()
    _check(suite, import_check["ok"], f"Top-level import is missing exports: {import_check['missing_exports']}")

    walker = WalkerV0(BODY_3X1)
    bridge = BridgeWalkerV0(BODY_3X1)
    walker_state = walker.reset()[1]
    bridge_state = bridge.reset()[1]
    for step in range(10):
        action_w = sinusoidal_action(walker.n_actuators, step, 10)
        action_b = sinusoidal_action(bridge.n_actuators, step, 10)
        _, walker_state, _, _ = walker.step(walker_state, action_w)
        _, bridge_state, _, _ = bridge.step(bridge_state, action_b)

    _check(suite, bool(jnp.all(jnp.isfinite(walker_state.sim_state.positions))), "Walker smoke rollout produced non-finite positions")
    _check(suite, bool(jnp.all(jnp.isfinite(bridge_state.sim_state.positions))), "BridgeWalker smoke rollout produced non-finite positions")

    mirror_suite = evaluate_mirroring_suite()
    override_suite = evaluate_robot_override_suite()
    slope_suite = evaluate_slope_semantics_suite()
    _check(suite, mirror_suite["status"] == "pass", f"Mirroring smoke failed: {mirror_suite['failures']}")
    _check(suite, override_suite["status"] == "pass", f"robot_override smoke failed: {override_suite['failures']}")
    _check(suite, slope_suite["status"] == "pass", f"slope smoke failed: {slope_suite['failures']}")

    _check(suite, walker.dynamic_terrain_point_indices.shape[0] == 0, "Walker smoke: fixed terrain leaked into dynamic state")
    _check(suite, bridge.dynamic_terrain_point_indices.shape[0] > 0, "BridgeWalker smoke: mixed terrain is not dynamic")

    _store_metric(
        suite,
        "counts",
        {
            "walker_dynamic_terrain_points": int(walker.dynamic_terrain_point_indices.shape[0]),
            "bridge_dynamic_terrain_points": int(bridge.dynamic_terrain_point_indices.shape[0]),
        },
    )
    return suite


def _device_summary() -> tuple[str, str]:
    backend = jax.default_backend()
    devices = jax.devices()
    device_name = str(devices[0]) if devices else "no-devices"
    return backend, device_name


def run_probe(profile: str = "standard") -> dict[str, Any]:
    if profile not in ("smoke", "standard", "report_only", "slope_guard"):
        raise ValueError(f"Unsupported profile={profile!r}")

    summary: dict[str, Any] = {
        "status": "pass",
        "profile": profile,
        "backend": None,
        "device": None,
        "durations_s": {},
        "suites": {},
        "failures": [],
    }
    backend, device = _device_summary()
    summary["backend"] = backend
    summary["device"] = device

    suites: list[tuple[str, Callable[[], dict[str, Any]]]]
    if profile == "smoke":
        suites = [("smoke", evaluate_smoke_suite)]
    elif profile == "slope_guard":
        suites = [
            ("smoke", evaluate_smoke_suite),
            ("structural_semantics", evaluate_structural_semantics_suite),
        ]
    else:
        suites = [
            ("walker_reference", evaluate_walker_reference_suite),
            ("multi_env_reference", evaluate_multi_env_reference_suite),
            ("terrain_stability", evaluate_terrain_stability_suite),
            ("structural_semantics", evaluate_structural_semantics_suite),
            ("mirroring", evaluate_mirroring_suite),
            ("robot_override", evaluate_robot_override_suite),
            ("slope_semantics", evaluate_slope_semantics_suite),
        ]

    for suite_name, fn in suites:
        started = time.perf_counter()
        suite = fn()
        elapsed = time.perf_counter() - started
        summary["durations_s"][suite_name] = round(elapsed, 3)
        summary["suites"][suite_name] = _to_jsonable(suite)
        if suite["status"] == "fail":
            summary["status"] = "fail"
            summary["failures"].extend([f"{suite_name}::{msg}" for msg in suite["failures"]])

    return summary


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"profile={summary['profile']} status={summary['status']} backend={summary['backend']} device={summary['device']}",
    ]
    for suite_name, suite in summary["suites"].items():
        duration = summary["durations_s"].get(suite_name, 0.0)
        lines.append(
            f"- {suite_name}: status={suite['status']} duration={duration:.3f}s failures={len(suite['failures'])}"
        )
        for failure in suite["failures"][:5]:
            lines.append(f"  - {failure}")
        if len(suite["failures"]) > 5:
            lines.append(f"  - ... {len(suite['failures']) - 5} more")
    if summary["failures"]:
        lines.append("failures:")
        for failure in summary["failures"]:
            lines.append(f"- {failure}")
    return "\n".join(lines)


def write_summary(summary: dict[str, Any], path: str | os.PathLike[str]) -> str:
    path_obj = Path(path)
    path_obj.write_text(
        json.dumps(_to_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(path_obj)
