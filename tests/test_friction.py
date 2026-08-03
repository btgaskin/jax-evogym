import numpy as np

from jax_evogym.friction import blended_friction_scalar, dynamic_friction_scalar


class TestDynamicFrictionScalar:
    def test_load_normalized_small_signal_slope_is_load_independent(self):
        velocity = 1e-5
        friction_const = 1200.0
        mu_d = 0.2

        force_low = float(
            dynamic_friction_scalar(
                velocity,
                0.5,
                friction_const=friction_const,
                mu_d=mu_d,
                normalize_by_load=True,
            )
        )
        force_high = float(
            dynamic_friction_scalar(
                velocity,
                5.0,
                friction_const=friction_const,
                mu_d=mu_d,
                normalize_by_load=True,
            )
        )

        expected = -friction_const * velocity
        np.testing.assert_allclose([force_low], [expected], rtol=2e-3, atol=1e-8)
        np.testing.assert_allclose([force_high], [expected], rtol=2e-3, atol=1e-8)

    def test_legacy_ground_small_signal_slope_grows_with_squared_load(self):
        velocity = 1e-5
        friction_const = 10.0
        mu_d = 0.2

        force_low = float(
            dynamic_friction_scalar(
                velocity,
                0.5,
                friction_const=friction_const,
                mu_d=mu_d,
                normalize_by_load=False,
            )
        )
        force_high = float(
            dynamic_friction_scalar(
                velocity,
                5.0,
                friction_const=friction_const,
                mu_d=mu_d,
                normalize_by_load=False,
            )
        )

        ratio = abs(force_high / force_low)
        np.testing.assert_allclose([ratio], [100.0], rtol=2e-3, atol=1e-8)


class TestBlendedFrictionScalar:
    def test_zero_velocity_uses_stick_branch_and_respects_tangential_load(self):
        supported = float(
            blended_friction_scalar(
                0.0,
                5.0,
                friction_const=1200.0,
                mu_d=0.2,
                mu_s=0.4,
                k_stick=20_000.0,
                switch_speed=0.05,
                tangential_load=1.5,
            )
        )
        capped = float(
            blended_friction_scalar(
                0.0,
                5.0,
                friction_const=1200.0,
                mu_d=0.2,
                mu_s=0.4,
                k_stick=20_000.0,
                switch_speed=0.05,
                tangential_load=10.0,
            )
        )

        np.testing.assert_allclose([supported], [1.5], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose([capped], [2.0], rtol=1e-6, atol=1e-8)

    def test_stiction_disabled_falls_back_to_dynamic_branch(self):
        dynamic = float(
            dynamic_friction_scalar(
                0.03,
                5.0,
                friction_const=1200.0,
                mu_d=0.2,
                normalize_by_load=True,
            )
        )
        blended = float(
            blended_friction_scalar(
                0.03,
                5.0,
                friction_const=1200.0,
                mu_d=0.2,
                mu_s=0.0,
                k_stick=20_000.0,
                switch_speed=0.05,
                tangential_load=3.0,
            )
        )

        np.testing.assert_allclose([blended], [dynamic], rtol=1e-6, atol=1e-8)
