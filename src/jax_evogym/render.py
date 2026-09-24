"""Post-hoc rendering for JAX EvoGym: PIL-based GIF generation.

Two-pass strategy:
1. Fast fitness pass: normal StepOutput (no extra render data)
2. Render pass: rerun top-K with RenderStepOutput capturing positions + spring state

Rendering is fully decoupled from JIT-compiled simulation. Pure numpy/PIL.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import constants as C
from .design_tool.tokens import get_render_palette
from .types import RenderInfo, RenderStepOutput

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default physics timing for render-time pacing. Callers can override this via
# RenderConfig.sim_time_per_step when the active sim preset differs.
SIM_TIME_PER_STEP = C.PHYSICS_UPDATES_PER_STEP * C.DT  # 0.003


@dataclass
class RenderConfig:
    enabled: bool = False    # Caller preference; direct render functions always render.
    width: int = 600
    height: int = 300
    fps: int = 50
    real_time: bool = True
    frame_skip: int = 7       # only used if real_time=False
    show_grid: bool = False
    show_edges: bool = True
    dynamic_actuator_colors: bool = True
    camera_padding: float = 0.3   # world units around robot COM
    viewport_width: float = 4.0   # world units visible
    camera_mode: str = "fit_robot"
    camera_smoothing: float = 0.18  # Fraction toward target: 0=hold, 1=instant.
    supersample: int = 2
    grid_major_every: int = 5
    show_minor_grid: bool = False
    edge_width_px: int = 2
    output_dir: str = "renders"
    target_frames: int | None = None
    sim_time_per_step: float | None = None
    max_colors: int = 256
    optimize: bool = False
    dither: bool = True

    @property
    def effective_frame_skip(self) -> int:
        if self.real_time:
            sim_time_per_step = self.sim_time_per_step
            if sim_time_per_step is None or float(sim_time_per_step) <= 0.0:
                sim_time_per_step = SIM_TIME_PER_STEP
            return max(1, round(1.0 / (self.fps * float(sim_time_per_step))))
        return self.frame_skip

    def frame_skip_for_steps(self, n_steps: int) -> int:
        """Frame skip that respects real-time pacing and optional frame budget."""
        base = int(self.effective_frame_skip)
        target = self.target_frames
        if target is None:
            return max(1, base)
        if int(target) <= 0:
            return max(1, base)
        by_budget = int(math.ceil(float(max(1, int(n_steps))) / float(int(target))))
        return max(1, base, by_budget)

    @property
    def frame_duration_ms(self) -> int:
        return max(20, 1000 // self.fps)  # GIF spec minimum ~10-20ms


# ---------------------------------------------------------------------------
# Color tables (matching C++ Interface.cpp / Colors.h)
# ---------------------------------------------------------------------------

_RENDER_PALETTE = get_render_palette()
BG_COLOR = _RENDER_PALETTE["background"]
GRID_MINOR_COLOR = _RENDER_PALETTE["grid_minor"]
GRID_MAJOR_COLOR = _RENDER_PALETTE["grid_major"]
EDGE_COLOR = _RENDER_PALETTE["edge"]

_H_ACT_GRADIENT = _RENDER_PALETTE["gradient_voxels"][C.H_ACT]
_V_ACT_GRADIENT = _RENDER_PALETTE["gradient_voxels"][C.V_ACT]
_CONTRACTILE_GRADIENT = _RENDER_PALETTE["gradient_voxels"][C.CONTRACTILE]
_SOLID_VOXELS = _RENDER_PALETTE["solid_voxels"]
_MID_ACTUATOR = 50

VOXEL_COLORS = {
    C.RIGID: _SOLID_VOXELS[C.RIGID],
    C.SOFT: _SOLID_VOXELS[C.SOFT],
    C.FIXED: _SOLID_VOXELS[C.FIXED],
    C.SLOPE_UP_RIGHT: _SOLID_VOXELS[C.SLOPE_UP_RIGHT],
    C.SLOPE_UP_LEFT: _SOLID_VOXELS[C.SLOPE_UP_LEFT],
    C.SLOPE_DOWN_RIGHT: _SOLID_VOXELS[C.SLOPE_DOWN_RIGHT],
    C.SLOPE_DOWN_LEFT: _SOLID_VOXELS[C.SLOPE_DOWN_LEFT],
    C.SLOPE2_UP_RIGHT_LIGHT: _SOLID_VOXELS[C.SLOPE2_UP_RIGHT_LIGHT],
    C.SLOPE2_UP_RIGHT_HEAVY: _SOLID_VOXELS[C.SLOPE2_UP_RIGHT_HEAVY],
    C.SLOPE2_UP_LEFT_HEAVY: _SOLID_VOXELS[C.SLOPE2_UP_LEFT_HEAVY],
    C.SLOPE2_UP_LEFT_LIGHT: _SOLID_VOXELS[C.SLOPE2_UP_LEFT_LIGHT],
    C.SLOPE2_DOWN_RIGHT_LIGHT: _SOLID_VOXELS[C.SLOPE2_DOWN_RIGHT_LIGHT],
    C.SLOPE2_DOWN_RIGHT_HEAVY: _SOLID_VOXELS[C.SLOPE2_DOWN_RIGHT_HEAVY],
    C.SLOPE2_DOWN_LEFT_HEAVY: _SOLID_VOXELS[C.SLOPE2_DOWN_LEFT_HEAVY],
    C.SLOPE2_DOWN_LEFT_LIGHT: _SOLID_VOXELS[C.SLOPE2_DOWN_LEFT_LIGHT],
    C.SLOPE3_UP_RIGHT_LIGHT: _SOLID_VOXELS[C.SLOPE3_UP_RIGHT_LIGHT],
    C.SLOPE3_UP_RIGHT_MID: _SOLID_VOXELS[C.SLOPE3_UP_RIGHT_MID],
    C.SLOPE3_UP_RIGHT_HEAVY: _SOLID_VOXELS[C.SLOPE3_UP_RIGHT_HEAVY],
    C.SLOPE3_UP_LEFT_HEAVY: _SOLID_VOXELS[C.SLOPE3_UP_LEFT_HEAVY],
    C.SLOPE3_UP_LEFT_MID: _SOLID_VOXELS[C.SLOPE3_UP_LEFT_MID],
    C.SLOPE3_UP_LEFT_LIGHT: _SOLID_VOXELS[C.SLOPE3_UP_LEFT_LIGHT],
    C.SLOPE3_DOWN_RIGHT_HEAVY: _SOLID_VOXELS[C.SLOPE3_DOWN_RIGHT_HEAVY],
    C.SLOPE3_DOWN_RIGHT_MID: _SOLID_VOXELS[C.SLOPE3_DOWN_RIGHT_MID],
    C.SLOPE3_DOWN_RIGHT_LIGHT: _SOLID_VOXELS[C.SLOPE3_DOWN_RIGHT_LIGHT],
    C.SLOPE3_DOWN_LEFT_LIGHT: _SOLID_VOXELS[C.SLOPE3_DOWN_LEFT_LIGHT],
    C.SLOPE3_DOWN_LEFT_MID: _SOLID_VOXELS[C.SLOPE3_DOWN_LEFT_MID],
    C.SLOPE3_DOWN_LEFT_HEAVY: _SOLID_VOXELS[C.SLOPE3_DOWN_LEFT_HEAVY],
    C.H_ACT: tuple(int(channel) for channel in _H_ACT_GRADIENT[_MID_ACTUATOR]),
    C.V_ACT: tuple(int(channel) for channel in _V_ACT_GRADIENT[_MID_ACTUATOR]),
    C.CONTRACTILE: tuple(int(channel) for channel in _CONTRACTILE_GRADIENT[_MID_ACTUATOR]),
}


def _actuation_color(vtype: int, actuation_level: float) -> tuple:
    """Map voxel type + actuation level to RGB color."""
    idx = int(np.clip((actuation_level - 0.6) * 100, 0, 99))
    if vtype == C.H_ACT:
        c = _H_ACT_GRADIENT[idx]
    elif vtype == C.CONTRACTILE:
        c = _CONTRACTILE_GRADIENT[idx]
    elif vtype == C.V_ACT:
        c = _V_ACT_GRADIENT[idx]
    else:
        return VOXEL_COLORS.get(vtype, (191, 191, 191))
    return (int(c[0]), int(c[1]), int(c[2]))


def _target_viewport(
    positions_np: np.ndarray,
    robot_point_indices: np.ndarray,
    config: RenderConfig,
    aspect_ratio: float,
) -> tuple[np.ndarray, float]:
    """Compute the target camera center and viewport width."""
    robot_pos = positions_np[robot_point_indices]
    min_xy = robot_pos.min(axis=0)
    max_xy = robot_pos.max(axis=0)
    target_center = (min_xy + max_xy) / 2.0

    if config.camera_mode == "fit_robot":
        padded_width = float(max_xy[0] - min_xy[0]) + (2.0 * float(config.camera_padding))
        padded_height = float(max_xy[1] - min_xy[1]) + (2.0 * float(config.camera_padding))
        target_width = max(float(config.viewport_width), padded_width, padded_height * aspect_ratio)
    else:
        target_width = float(config.viewport_width)

    return target_center, target_width


# ---------------------------------------------------------------------------
# Render step factory (works with any EvoGymBaseEnv subclass)
# ---------------------------------------------------------------------------

def make_render_episode_step(env):
    """Create a lax.scan-compatible step that captures render data.

    Works with any EvoGymBaseEnv subclass — no per-env changes needed.
    """
    def render_step(env_state, action):
        obs, new_env_state, reward, done = env.step(env_state, action)
        output = RenderStepOutput(
            positions=new_env_state.sim_state.positions,
            spring_rest_length=new_env_state.sim_state.spring_rest_length,
            reward=reward,
            done=done,
        )
        return new_env_state, output

    return render_step


# ---------------------------------------------------------------------------
# Core rendering functions
# ---------------------------------------------------------------------------

def render_frame(
    positions_np: np.ndarray,
    render_info: RenderInfo,
    config: RenderConfig,
    spring_rest_length_np: np.ndarray | None = None,
    camera_pos: np.ndarray | None = None,
    cell_actuation_np: np.ndarray | None = None,
    cell_color_overrides_np: np.ndarray | None = None,
) -> tuple:
    """Render a single frame as a PIL Image.

    Args:
        positions_np: (n_points, 2) numpy array of point positions.
        render_info: Static render data from env.render_info.
        config: Render configuration.
        spring_rest_length_np: (n_springs,) current spring rest lengths for
            dynamic actuator coloring. If None, uses static colors.
        camera_pos: (2,) camera center [x, y]. If None, computed from robot COM.

    Returns:
        (PIL.Image, camera_pos) tuple.
    """
    from PIL import Image, ImageDraw

    ri = render_info
    if ri.static_point_positions.size > 0:
        all_positions = np.concatenate(
            [positions_np, np.asarray(ri.static_point_positions, dtype=np.float32)],
            axis=0,
        )
    else:
        all_positions = positions_np
    w, h = config.width, config.height
    supersample = max(1, int(config.supersample))
    canvas_w = int(w * supersample)
    canvas_h = int(h * supersample)
    aspect_ratio = float(canvas_w) / float(max(1, canvas_h))

    target_center, viewport_width = _target_viewport(
        all_positions,
        ri.robot_point_indices,
        config,
        aspect_ratio,
    )
    if camera_pos is None:
        camera_pos = target_center.copy()
    else:
        smoothing = float(np.clip(config.camera_smoothing, 0.0, 1.0))
        camera_pos = (1.0 - smoothing) * camera_pos + smoothing * target_center

    # World-to-pixel transform
    scale = canvas_w / viewport_width
    viewport_height = canvas_h / scale

    def world_to_pixel(wx, wy):
        px = (wx - camera_pos[0]) * scale + canvas_w / 2
        py = canvas_h / 2 - (wy - camera_pos[1]) * scale
        return px, py

    # Compute actuation levels per cell (if dynamic coloring)
    cell_actuation = cell_actuation_np
    if (
        cell_actuation is None
        and config.dynamic_actuator_colors
        and spring_rest_length_np is not None
        and len(ri.actuator_cell_indices) > 0
    ):
        cell_actuation = _compute_cell_actuation(ri, spring_rest_length_np)

    # Create image
    img = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Optional grid
    if config.show_grid:
        # Draw grid lines across viewport
        cam_x, cam_y = camera_pos
        vw = viewport_width
        vh = viewport_height
        x_min = cam_x - vw / 2
        x_max = cam_x + vw / 2
        y_min = cam_y - vh / 2
        y_max = cam_y + vh / 2
        cell = C.CELL_SIZE
        gx = np.floor(x_min / cell) * cell
        while gx <= x_max:
            grid_idx = int(round(gx / cell))
            is_major = (grid_idx % max(1, int(config.grid_major_every))) == 0
            if not config.show_minor_grid and not is_major:
                gx += cell
                continue
            px, _ = world_to_pixel(gx, 0)
            color = GRID_MAJOR_COLOR if is_major else GRID_MINOR_COLOR
            width_px = max(1, supersample if is_major else 1)
            draw.line([(px, 0), (px, canvas_h)], fill=color, width=width_px)
            gx += cell
        gy = np.floor(y_min / cell) * cell
        while gy <= y_max:
            grid_idx = int(round(gy / cell))
            is_major = (grid_idx % max(1, int(config.grid_major_every))) == 0
            if not config.show_minor_grid and not is_major:
                gy += cell
                continue
            _, py = world_to_pixel(0, gy)
            color = GRID_MAJOR_COLOR if is_major else GRID_MINOR_COLOR
            width_px = max(1, supersample if is_major else 1)
            draw.line([(0, py), (canvas_w, py)], fill=color, width=width_px)
            gy += cell

    # Viewport culling bounds (world coords)
    cam_x, cam_y = camera_pos
    cull_x_min = cam_x - viewport_width / 2 - 0.2
    cull_x_max = cam_x + viewport_width / 2 + 0.2
    cull_y_min = cam_y - viewport_height / 2 - 0.2
    cull_y_max = cam_y + viewport_height / 2 + 0.2

    # Draw terrain first, then robot.
    for is_robot_pass in (False, True):
        for cell_idx in range(ri.cell_mask.shape[0]):
            if not ri.cell_mask[cell_idx]:
                continue
            if ri.cell_is_robot[cell_idx] != is_robot_pass:
                continue

            vertex_count = int(ri.cell_vertex_count[cell_idx])
            corner_pos = all_positions[ri.cell_vertices[cell_idx, :vertex_count]]

            # Viewport culling
            cx_min, cy_min = corner_pos.min(axis=0)
            cx_max, cy_max = corner_pos.max(axis=0)
            if cx_max < cull_x_min or cx_min > cull_x_max:
                continue
            if cy_max < cull_y_min or cy_min > cull_y_max:
                continue

            # Get pixel coordinates
            pixels = [world_to_pixel(point[0], point[1]) for point in corner_pos]

            # Color
            vtype = ri.cell_types[cell_idx]
            if (
                cell_color_overrides_np is not None
                and int(cell_color_overrides_np[cell_idx, 0]) >= 0
            ):
                override = cell_color_overrides_np[cell_idx]
                color = (int(override[0]), int(override[1]), int(override[2]))
            elif (
                not bool(ri.cell_is_robot[cell_idx])
                and int(ri.cell_object_id[cell_idx]) == 2
            ):
                color = (98, 179, 61)
            elif (config.dynamic_actuator_colors
                    and vtype in (C.H_ACT, C.V_ACT, C.CONTRACTILE)
                    and spring_rest_length_np is not None):
                act_level = 1.0  # default
                if cell_actuation is not None:
                    act_level = cell_actuation[cell_idx]
                color = _actuation_color(vtype, act_level)
            else:
                color = VOXEL_COLORS.get(int(vtype), (191, 191, 191))

            draw.polygon(pixels, fill=color)

            if config.show_edges:
                for edge_idx in range(vertex_count):
                    if ri.surface_edge_mask[cell_idx, edge_idx]:
                        next_idx = (edge_idx + 1) % vertex_count
                        draw.line(
                            [pixels[next_idx], pixels[edge_idx]],
                            fill=EDGE_COLOR,
                            width=max(1, int(config.edge_width_px) * supersample),
                        )

    if supersample > 1:
        img = img.resize((w, h), Image.Resampling.LANCZOS)
    return img, camera_pos


def _compute_cell_actuation(
    render_info: RenderInfo,
    spring_rest_length_np: np.ndarray,
) -> np.ndarray:
    """Compute per-cell actuation level from render-cell-indexed spring mappings."""
    n_cells = render_info.cell_mask.shape[0]
    actuation = np.ones(n_cells, dtype=np.float32)
    init_rl = render_info.spring_init_rest_length
    safe_init = np.where(init_rl > 1e-10, init_rl, 1.0)
    h_pairs = np.asarray(render_info.cell_h_spring_pairs, dtype=np.int32)
    v_pairs = np.asarray(render_info.cell_v_spring_pairs, dtype=np.int32)
    n_springs = int(spring_rest_length_np.shape[0])

    for cell_idx in np.asarray(render_info.actuator_cell_indices, dtype=np.int32):
        ratios: list[float] = []
        vtype = int(render_info.cell_types[cell_idx])

        if vtype in (C.H_ACT, C.CONTRACTILE):
            hs0, hs1 = (int(h_pairs[cell_idx, 0]), int(h_pairs[cell_idx, 1]))
            if 0 <= hs0 < n_springs:
                ratios.append(float(spring_rest_length_np[hs0] / safe_init[hs0]))
            if 0 <= hs1 < n_springs:
                ratios.append(float(spring_rest_length_np[hs1] / safe_init[hs1]))

        if vtype in (C.V_ACT, C.CONTRACTILE):
            vs0, vs1 = (int(v_pairs[cell_idx, 0]), int(v_pairs[cell_idx, 1]))
            if 0 <= vs0 < n_springs:
                ratios.append(float(spring_rest_length_np[vs0] / safe_init[vs0]))
            if 0 <= vs1 < n_springs:
                ratios.append(float(spring_rest_length_np[vs1] / safe_init[vs1]))

        if ratios:
            actuation[cell_idx] = np.clip(np.mean(ratios, dtype=np.float32), 0.6, 1.6)

    return actuation


def render_episode(
    render_outputs: RenderStepOutput,
    render_info: RenderInfo,
    config: RenderConfig,
    cell_spring_indices: np.ndarray | None = None,
    robot_cell_rgb_np: np.ndarray | None = None,
) -> list:
    """Render a full episode to a list of PIL Images.

    Args:
        render_outputs: RenderStepOutput from lax.scan, shape (n_steps, ...).
        render_info: Static render data from env.render_info.
        config: Render configuration.
        cell_spring_indices: legacy argument (ignored). Dynamic actuator colors use
            render_info per-cell spring mappings.

    Returns:
        List of PIL.Image frames.
    """
    # Convert JAX -> numpy once
    positions_all = np.array(render_outputs.positions)  # (n_steps, n_points, 2)
    spring_rl_all = np.array(render_outputs.spring_rest_length)  # (n_steps, n_springs)
    n_steps = positions_all.shape[0]

    frame_skip = config.frame_skip_for_steps(n_steps)
    del cell_spring_indices
    cell_color_overrides = None
    if robot_cell_rgb_np is not None:
        rgb_arr = np.asarray(robot_cell_rgb_np, dtype=np.float32)
        if rgb_arr.ndim == 3 and rgb_arr.shape[-1] == 3:
            rgb_flat = rgb_arr.reshape(-1, 3)
        elif rgb_arr.ndim == 2 and rgb_arr.shape[-1] == 3:
            rgb_flat = rgb_arr
        else:
            raise ValueError("robot_cell_rgb_np must have shape (H,W,3) or (N,3).")
        robot_render_indices = np.flatnonzero(
            np.asarray(render_info.cell_is_robot, dtype=bool)
            & np.asarray(render_info.cell_is_dynamic, dtype=bool)
        )
        cell_color_overrides = np.full(
            (int(render_info.cell_mask.shape[0]), 3),
            -1,
            dtype=np.int32,
        )
        n_assign = min(int(robot_render_indices.shape[0]), int(rgb_flat.shape[0]))
        if n_assign > 0:
            rgb_uint8 = np.clip(np.round(rgb_flat[:n_assign] * 255.0), 0.0, 255.0).astype(np.int32)
            cell_color_overrides[robot_render_indices[:n_assign]] = rgb_uint8

    frames = []
    camera_pos = None

    for t in range(0, n_steps, frame_skip):
        positions_np = positions_all[t]
        spring_rl_np = spring_rl_all[t]

        # Compute per-cell actuation from render-cell spring mappings.
        if config.dynamic_actuator_colors:
            cell_act = _compute_cell_actuation(render_info, spring_rl_np)
        else:
            cell_act = None

        img, camera_pos = render_frame(
            positions_np, render_info, config,
            spring_rest_length_np=spring_rl_np,
            camera_pos=camera_pos,
            cell_actuation_np=cell_act,
            cell_color_overrides_np=cell_color_overrides,
        )

        frames.append(img)

    return frames


def save_gif(frames: list, path: str | Path, config: RenderConfig) -> str:
    """Save frames as an animated GIF.

    Returns the path string.
    """
    if not frames:
        raise ValueError("save_gif requires at least one frame")

    from PIL import Image

    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    duration = config.frame_duration_ms

    # Quantize to a shared palette for materially smaller GIF size.
    colors = int(np.clip(config.max_colors, 2, 256))
    dither_mode = Image.Dither.FLOYDSTEINBERG if config.dither else Image.Dither.NONE
    first_rgb = frames[0].convert("RGB")
    palette_frame = first_rgb.quantize(colors=colors, dither=dither_mode)
    quantized_frames = [palette_frame]
    for frame in frames[1:]:
        quantized_frames.append(
            frame.convert("RGB").quantize(palette=palette_frame, dither=dither_mode)
        )

    quantized_frames[0].save(
        path,
        save_all=True,
        append_images=quantized_frames[1:],
        duration=duration,
        loop=0,
        optimize=bool(config.optimize),
        disposal=2,
    )
    return path


def render_top_k(
    batched_render_outputs: RenderStepOutput,
    rewards: np.ndarray,
    render_info: RenderInfo,
    config: RenderConfig,
    k: int,
    generation: int,
    cell_spring_indices: np.ndarray | None = None,
) -> list:
    """Render GIFs for top-K individuals from a batched evaluation.

    Args:
        batched_render_outputs: Time-major shape (n_steps, pop_size, ...).
        rewards: (pop_size,) total rewards per individual.
        render_info: Static render data.
        config: Render configuration.
        k: Number of top individuals to render.
        generation: Generation number for filenames.
        cell_spring_indices: (n_act_cells, 2) for dynamic actuator colors.

    Returns:
        List of GIF file paths.
    """
    rewards_np = np.array(rewards)
    top_indices = np.argsort(rewards_np)[-k:][::-1]

    paths = []
    for rank, idx in enumerate(top_indices):
        # Slice individual from time-major batch: x[:, idx, ...]
        individual_outputs = jax.tree.map(
            lambda x: x[:, idx], batched_render_outputs
        )
        frames = render_episode(individual_outputs, render_info, config, cell_spring_indices)

        out_dir = Path(config.output_dir)
        path = out_dir / f"gen{generation:04d}_rank{rank}.gif"
        save_gif(frames, path, config)
        paths.append(str(path))

    return paths


def rerun_top_k(
    env,
    init_state,
    all_actions: jnp.ndarray,
    rewards: jnp.ndarray,
    k: int,
    config: RenderConfig,
    generation: int = 0,
) -> list:
    """High-level: select top-K by reward, rerun with render step, save GIFs.

    Args:
        env: EvoGymBaseEnv subclass instance.
        init_state: Initial EnvState from env.reset().
        all_actions: (n_steps, pop_size, n_act) time-major actions.
        rewards: (pop_size,) total rewards.
        k: Number of top individuals.
        config: Render configuration.
        generation: Generation number for filenames.

    Returns:
        List of GIF file paths.
    """
    rewards_np = np.array(rewards)
    top_indices = np.argsort(rewards_np)[-k:][::-1]

    render_step = make_render_episode_step(env)
    cell_spring_indices = np.array(env.actuator_info.cell_spring_indices)

    paths = []
    for rank, idx in enumerate(top_indices):
        # Extract this individual's actions: (n_steps, n_act)
        individual_actions = all_actions[:, int(idx), :]

        # Run episode with render step
        _, outputs = jax.lax.scan(render_step, init_state, individual_actions)

        # Render and save
        frames = render_episode(outputs, env.render_info, config, cell_spring_indices)
        out_dir = Path(config.output_dir)
        path = out_dir / f"gen{generation:04d}_rank{rank}.gif"
        save_gif(frames, path, config)
        paths.append(str(path))

    return paths
