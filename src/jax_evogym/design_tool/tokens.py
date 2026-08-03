"""Shared visual token helpers for the design tool and renderer."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np

from .. import constants as C


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a #RRGGBB string to an RGB tuple."""
    value = hex_color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"expected 6 hex characters, got {hex_color!r}")
    return tuple(int(value[idx : idx + 2], 16) for idx in (0, 2, 4))


def _build_gradient(start_hex: str, end_hex: str, steps: int = 100) -> np.ndarray:
    start = np.array(hex_to_rgb(start_hex), dtype=np.float32)
    end = np.array(hex_to_rgb(end_hex), dtype=np.float32)
    ratios = np.linspace(0.0, 1.0, num=steps, dtype=np.float32)[:, None]
    gradient = start + (end - start) * ratios
    return np.clip(gradient.round(), 0, 255).astype(np.uint8)


@lru_cache(maxsize=1)
def get_visual_tokens() -> dict[str, Any]:
    """Load the canonical visual token document."""
    token_path = Path(__file__).with_name("visual_tokens.json")
    return json.loads(token_path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_render_palette() -> dict[str, Any]:
    """Return renderer-friendly token values."""
    tokens = get_visual_tokens()
    render_tokens = tokens["render"]
    voxel_tokens = tokens["voxels"]
    return {
        "background": hex_to_rgb(tokens["ui"]["canvas_bg"]),
        "grid_minor": hex_to_rgb(render_tokens["grid_minor"]),
        "grid_major": hex_to_rgb(render_tokens["grid_major"]),
        "edge": hex_to_rgb(render_tokens["edge"]),
        "selection": hex_to_rgb(render_tokens["selection"]),
        "locked": hex_to_rgb(render_tokens["locked"]),
        "grid_major_every": int(render_tokens["grid_major_every"]),
        "edge_width_px": int(render_tokens["edge_width_px"]),
        "solid_voxels": {
            C.RIGID: hex_to_rgb(voxel_tokens["rigid"]["hex"]),
            C.SOFT: hex_to_rgb(voxel_tokens["soft"]["hex"]),
            C.FIXED: hex_to_rgb(voxel_tokens["fixed"]["hex"]),
            C.SLOPE_UP_RIGHT: hex_to_rgb(voxel_tokens["slope_up_right"]["hex"]),
            C.SLOPE_UP_LEFT: hex_to_rgb(voxel_tokens["slope_up_left"]["hex"]),
            C.SLOPE_DOWN_RIGHT: hex_to_rgb(voxel_tokens["slope_down_right"]["hex"]),
            C.SLOPE_DOWN_LEFT: hex_to_rgb(voxel_tokens["slope_down_left"]["hex"]),
            C.SLOPE2_UP_RIGHT_LIGHT: hex_to_rgb(voxel_tokens["slope2_up_right_light"]["hex"]),
            C.SLOPE2_UP_RIGHT_HEAVY: hex_to_rgb(voxel_tokens["slope2_up_right_heavy"]["hex"]),
            C.SLOPE2_UP_LEFT_HEAVY: hex_to_rgb(voxel_tokens["slope2_up_left_heavy"]["hex"]),
            C.SLOPE2_UP_LEFT_LIGHT: hex_to_rgb(voxel_tokens["slope2_up_left_light"]["hex"]),
            C.SLOPE2_DOWN_RIGHT_LIGHT: hex_to_rgb(voxel_tokens["slope2_down_right_light"]["hex"]),
            C.SLOPE2_DOWN_RIGHT_HEAVY: hex_to_rgb(voxel_tokens["slope2_down_right_heavy"]["hex"]),
            C.SLOPE2_DOWN_LEFT_HEAVY: hex_to_rgb(voxel_tokens["slope2_down_left_heavy"]["hex"]),
            C.SLOPE2_DOWN_LEFT_LIGHT: hex_to_rgb(voxel_tokens["slope2_down_left_light"]["hex"]),
            C.SLOPE3_UP_RIGHT_LIGHT: hex_to_rgb(voxel_tokens["slope3_up_right_light"]["hex"]),
            C.SLOPE3_UP_RIGHT_MID: hex_to_rgb(voxel_tokens["slope3_up_right_mid"]["hex"]),
            C.SLOPE3_UP_RIGHT_HEAVY: hex_to_rgb(voxel_tokens["slope3_up_right_heavy"]["hex"]),
            C.SLOPE3_UP_LEFT_HEAVY: hex_to_rgb(voxel_tokens["slope3_up_left_heavy"]["hex"]),
            C.SLOPE3_UP_LEFT_MID: hex_to_rgb(voxel_tokens["slope3_up_left_mid"]["hex"]),
            C.SLOPE3_UP_LEFT_LIGHT: hex_to_rgb(voxel_tokens["slope3_up_left_light"]["hex"]),
            C.SLOPE3_DOWN_RIGHT_HEAVY: hex_to_rgb(voxel_tokens["slope3_down_right_heavy"]["hex"]),
            C.SLOPE3_DOWN_RIGHT_MID: hex_to_rgb(voxel_tokens["slope3_down_right_mid"]["hex"]),
            C.SLOPE3_DOWN_RIGHT_LIGHT: hex_to_rgb(voxel_tokens["slope3_down_right_light"]["hex"]),
            C.SLOPE3_DOWN_LEFT_LIGHT: hex_to_rgb(voxel_tokens["slope3_down_left_light"]["hex"]),
            C.SLOPE3_DOWN_LEFT_MID: hex_to_rgb(voxel_tokens["slope3_down_left_mid"]["hex"]),
            C.SLOPE3_DOWN_LEFT_HEAVY: hex_to_rgb(voxel_tokens["slope3_down_left_heavy"]["hex"]),
        },
        "gradient_voxels": {
            C.H_ACT: _build_gradient(voxel_tokens["h_act"]["start"], voxel_tokens["h_act"]["end"]),
            C.V_ACT: _build_gradient(voxel_tokens["v_act"]["start"], voxel_tokens["v_act"]["end"]),
            C.CONTRACTILE: _build_gradient(
                voxel_tokens["contractile"]["start"],
                voxel_tokens["contractile"]["end"],
            ),
        },
    }
