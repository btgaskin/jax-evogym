import pytest

from tests.parity_harness import evaluate_walker_reference_suite


@pytest.fixture(scope="module")
def walker_suite():
    suite = evaluate_walker_reference_suite(skip_if_missing=True)
    if suite["status"] == "skipped":
        pytest.skip(suite["metrics"]["reason"])
    return suite


def _assert_suite_passes(suite):
    assert suite["status"] == "pass", "\n".join(suite["failures"])


class TestWalkerReferenceParity:
    def test_suite_passes(self, walker_suite):
        _assert_suite_passes(walker_suite)

    def test_cases_present(self, walker_suite):
        cases = walker_suite["metrics"]["cases"]
        assert "simple" in cases
        assert "biped" in cases

    def test_thresholds_recorded(self, walker_suite):
        simple = walker_suite["metrics"]["cases"]["simple"]
        assert "com_x_max_error" in simple
        assert "obs_step_max_errors" in simple
