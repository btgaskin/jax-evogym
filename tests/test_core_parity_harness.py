"""Thin parity wrappers around the shared parity harness."""

from __future__ import annotations

import pytest

from jax_evogym import constants as C
from tests.parity_harness import (
    evaluate_slope_mode_guard_suite,
    evaluate_mirroring_suite,
    evaluate_robot_override_suite,
    evaluate_slope_mode_invariance_suite,
    evaluate_structural_semantics_suite,
    run_probe,
)
from jax_evogym.types import default_physics_constants


def _assert_suite_passes(suite):
    assert suite["status"] == "pass", "\n".join(suite["failures"])


class TestStructuralSemantics:
    def test_structural_semantics(self):
        _assert_suite_passes(evaluate_structural_semantics_suite())


class TestMirroring:
    def test_mirroring(self):
        _assert_suite_passes(evaluate_mirroring_suite())


class TestRobotOverride:
    def test_robot_override(self):
        _assert_suite_passes(evaluate_robot_override_suite())


class TestSlopeGuard:
    def test_slope_guard_profile(self):
        summary = run_probe("slope_guard")
        assert summary["status"] == "pass", "\n".join(summary["failures"])
        assert set(summary["suites"]) == {"smoke", "structural_semantics"}

    def test_slope_mode_invariance_default_vs_default_passes(self):
        suite = evaluate_slope_mode_invariance_suite(
            candidate_constants=default_physics_constants(),
            case_names=("free_body",),
            n_steps=2,
        )
        _assert_suite_passes(suite)

    def test_slope_mode_invariance_detects_non_slope_leakage(self):
        suite = evaluate_slope_mode_invariance_suite(
            candidate_constants=default_physics_constants()._replace(gravity=0.0),
            case_names=("free_body",),
            n_steps=2,
        )
        assert suite["status"] == "fail"

    def test_slope_mode_guard_ignores_diagnostic_case_failures(self):
        guard = evaluate_slope_mode_guard_suite(
            candidate_constants=default_physics_constants(),
            n_steps=2,
        )
        _assert_suite_passes(guard)
        assert guard["metrics"]["blocking"]["status"] == "pass"
        assert "diagnostic" in guard["metrics"]

    def test_slope_mode_guard_detects_blocking_leakage(self):
        guard = evaluate_slope_mode_guard_suite(
            candidate_constants=default_physics_constants()._replace(gravity=0.0),
            n_steps=2,
        )
        assert guard["status"] == "fail"

    def test_slope_mode_guard_projected_anchor_slope_only_does_not_leak_memory_state(self):
        guard = evaluate_slope_mode_guard_suite(
            candidate_constants=default_physics_constants()._replace(
                stiction_enabled=True,
                static_friction_const=0.75,
                terrain_stiction_stiffness=20_000.0,
                terrain_stiction_switch_speed=0.12,
                terrain_stiction_scope=int(C.STICTION_SCOPE_SLOPE_ONLY),
                friction_model=int(C.FRICTION_MODEL_PROJECTED_ANCHOR),
                anchor_stiffness=50_000.0,
            ),
            n_steps=2,
        )
        _assert_suite_passes(guard)
