"""python -m examples.first_motion --steps 300 --output /tmp/first-motion.gif"""
import argparse
import numpy as np
from jax_evogym import WalkerV0, RenderConfig, render_episode, save_gif
from examples.common import BODY, periodic_actions, render_rollout


def run(steps=300, output='/tmp/first-motion.gif'):
    env = WalkerV0(BODY)
    actions = periodic_actions(steps, env.n_actuators)
    final, outputs = render_rollout(env, actions)
    if not np.isfinite(np.asarray(outputs.positions)).all():
        raise RuntimeError('Rollout contains non-finite positions.')
    config = RenderConfig(width=480, height=240, viewport_width=1.2, target_frames=60)
    frames = render_episode(outputs, env.render_info, config)
    save_gif(frames, output, config)
    print(f'Saved {len(frames)} frames to {output}; reward={float(outputs.reward.sum()):.6f}')
    return final, outputs, frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--output', default='/tmp/first-motion.gif')
    args = parser.parse_args()
    run(args.steps, args.output)
