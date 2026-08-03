"""Terrain stability checks routed through the shared parity harness."""

from tests.parity_harness import evaluate_terrain_stability_suite


def test_terrain_stability_suite():
    suite = evaluate_terrain_stability_suite()
    assert suite["status"] == "pass", "\n".join(suite["failures"])
