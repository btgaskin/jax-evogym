"""Focused regression tests for the canonical world JSON contract."""

import json

import pytest

from jax_evogym.world import EvoWorld


def _write_world(tmp_path, payload, name="world.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_sparse_declared_grid_round_trips_without_renumbering(tmp_path):
    payload = {
        "grid_width": 24,
        "grid_height": 16,
        "objects": {
            "robot": {
                "indices": [25, 26],
                "types": [1, 2],
                "neighbors": {"25": [26], "26": [25]},
            }
        },
    }

    world = EvoWorld.from_json(_write_world(tmp_path, payload))
    serialized = world.to_json_dict()

    assert world.grid_size == (24, 16)
    assert json.dumps(serialized, separators=(",", ":")) == json.dumps(
        payload, separators=(",", ":")
    )


def test_custom_connectivity_survives_a_load_save_load_cycle(tmp_path):
    payload = {
        "grid_width": 24,
        "grid_height": 16,
        "objects": {
            "robot": {
                "indices": [25, 26, 27],
                "types": [1, 1, 1],
                "neighbors": {"25": [27], "26": [], "27": [25]},
            }
        },
    }

    world = EvoWorld.from_json(_write_world(tmp_path, payload, "source.json"))
    saved_path = tmp_path / "saved.json"
    world.to_json(saved_path)
    reloaded = EvoWorld.from_json(saved_path)

    assert world.to_json_dict()["objects"]["robot"]["neighbors"] == payload[
        "objects"
    ]["robot"]["neighbors"]
    assert reloaded.to_json_dict() == payload


@pytest.mark.parametrize("index", [-1, 24 * 16])
def test_out_of_range_indices_are_rejected(tmp_path, index):
    payload = {
        "grid_width": 24,
        "grid_height": 16,
        "objects": {
            "robot": {
                "indices": [index],
                "types": [1],
                "neighbors": {str(index): []},
            }
        },
    }

    with pytest.raises(ValueError, match=r"corrupted object robot.*outside the declared 24x16 grid"):
        EvoWorld.from_json(_write_world(tmp_path, payload))


def test_duplicate_indices_are_rejected(tmp_path):
    payload = {
        "grid_width": 24,
        "grid_height": 16,
        "objects": {
            "robot": {
                "indices": [25, 25],
                "types": [1, 2],
                "neighbors": {"25": []},
            }
        },
    }

    with pytest.raises(ValueError, match=r"corrupted object robot.*duplicate indices"):
        EvoWorld.from_json(_write_world(tmp_path, payload))
