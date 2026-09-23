"""Render the README illustration without running a simulation or training."""

from pathlib import Path

import numpy as np

from jax_evogym import (
    EMPTY, FIXED, H_ACT, RIGID, SOFT, V_ACT,
    EvoWorld, RenderConfig, compile_world_template, instantiate_world, render_frame,
)


def main():
    world = EvoWorld()
    world.add_from_array("ground", np.full((1, 16), FIXED), 0, 0)
    body = np.array([
        [RIGID, RIGID, RIGID, RIGID, RIGID],
        [V_ACT, SOFT, SOFT, SOFT, V_ACT],
        [V_ACT, EMPTY, EMPTY, EMPTY, V_ACT],
        [H_ACT, EMPTY, EMPTY, EMPTY, H_ACT],
    ])
    world.add_from_array("robot", body, 5, 1)
    built = instantiate_world(compile_world_template(world))
    config = RenderConfig(
        width=1200, height=480, viewport_width=1.6,
        camera_padding=0.15, show_grid=True,
    )
    frame, _ = render_frame(np.asarray(built.sim_state.positions), built.render_info, config)
    output = Path(__file__).resolve().parents[1] / "docs/assets/soft-body-world.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.save(output)
    print(output)


if __name__ == "__main__":
    main()
