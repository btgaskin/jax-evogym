"""python -m examples.designer_world world.json --robot robot --output /tmp/world.gif"""
import argparse
import jax
import numpy as np
from jax_evogym import (
    CONTRACTILE, EvoWorld, RenderConfig, RenderStepOutput, compile_world_template,
    instantiate_world, env_step, render_episode, save_gif,
)
from examples.common import periodic_actions


def load_world(path, robot_name='robot'):
    world = EvoWorld.from_json(path)
    if robot_name not in world.objects:
        raise ValueError(f'No object named {robot_name!r}. Available: {list(world.objects)}')
    if np.any(world.objects[robot_name].grid == CONTRACTILE):
        raise ValueError('This scalar-action example supports H_ACT/V_ACT. CONTRACTILE requires per-axis control; see the Controllers guide.')
    built = instantiate_world(compile_world_template(world, robot_name=robot_name))
    if built.actuator_info.cell_spring_indices.shape[0] == 0:
        raise ValueError('The named robot needs at least one H_ACT or V_ACT cell.')
    return built


def run(path, robot_name='robot', steps=300, output='/tmp/world.gif'):
    built = load_world(path, robot_name)
    actions = periodic_actions(steps, built.actuator_info.cell_spring_indices.shape[0])

    def step(state, action):
        state = env_step(state, built.topology, built.constants, action, built.actuator_info,
                         collision_data=built.collision_data, static_collider_data=built.static_collider_data)
        return state, RenderStepOutput(state.positions, state.spring_rest_length, 0.0, False)

    final, outputs = jax.lax.scan(step, built.sim_state, actions)
    if not np.isfinite(np.asarray(outputs.positions)).all():
        raise RuntimeError('Rollout contains non-finite positions.')
    config = RenderConfig(width=480, height=240, viewport_width=1.2, target_frames=60)
    frames = render_episode(outputs, built.render_info, config)
    save_gif(frames, output, config)
    print(f'Saved {len(frames)} frames to {output}; physics-only rollout (no task reward).')
    return final, outputs, frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('world')
    parser.add_argument('--robot', default='robot')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--output', default='/tmp/world.gif')
    args = parser.parse_args()
    run(args.world, args.robot, args.steps, args.output)
