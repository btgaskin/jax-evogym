import pytest

from tests.parity_harness import evaluate_multi_env_reference_suite


@pytest.fixture(scope="module")
def multi_env_suite():
    suite = evaluate_multi_env_reference_suite(skip_if_missing=True)
    if suite["status"] == "skipped":
        pytest.skip(suite["metrics"]["reason"])
    return suite


def _assert_suite_passes(suite):
    assert suite["status"] == "pass", "\n".join(suite["failures"])


class TestMultiEnvParity:
    def test_suite_passes(self, multi_env_suite):
        _assert_suite_passes(multi_env_suite)

    def test_bridge_case_present(self, multi_env_suite):
        cases = multi_env_suite["metrics"]["cases"]
        assert "bridge_walker" in cases

    def test_cases_record_errors(self, multi_env_suite):
        for case in multi_env_suite["metrics"]["cases"].values():
            assert "com_x_max_error" in case
