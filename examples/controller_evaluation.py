"""Compare four fixed periodic controllers; this does not train a controller.

python -m examples.controller_evaluation --steps 100
Run batched evaluation on an appropriate compute host, not a restricted local workstation.
"""
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from jax_evogym import WalkerV0
from examples.common import BODY, periodic_actions, freeze_finished, render_rollout


def evaluate_one(env, parameters, steps=100):
    """Parameters are amplitude and phase; period is fixed at 80 environment steps."""
    actions = periodic_actions(steps, env.n_actuators, amplitude=parameters[0], phase=parameters[1])
    _, initial = env.reset()

    def step(state, action):
        _, proposed, reward, _ = env.step(state, action)
        next_state = freeze_finished(state, proposed)
        # Preserve the terminal transition's reward, mask only subsequent steps.
        return next_state, jnp.where(state.done, 0.0, reward)

    final, rewards = jax.lax.scan(step, initial, actions)
    return rewards.sum(), final


def evaluate_batch(env, parameters, steps=100):
    # Candidate-major outputs. Topology and body are shared, controller parameters vary.
    return jax.jit(jax.vmap(lambda candidate: evaluate_one(env, candidate, steps)))(parameters)


def run(steps=100, output=None):
    env = WalkerV0(BODY)
    candidates = jnp.array([[0.0, 0.0], [0.15, 0.0], [0.25, 0.0], [0.25, jnp.pi / 2]], dtype=jnp.float32)
    scores, _ = evaluate_batch(env, candidates, steps)
    scores_np = np.asarray(scores)  # waits for device computation before reporting
    if not np.isfinite(scores_np).all():
        raise RuntimeError('Non-finite candidate scores; inspect the rollout before selection.')
    best = int(np.argmax(scores_np))
    print(f'Scores: {scores_np.tolist()}\nBest candidate: {best} (this body, task and horizon only)')
    if output:
        from jax_evogym import RenderConfig, render_episode, save_gif
        actions = periodic_actions(steps, env.n_actuators, amplitude=candidates[best, 0], phase=candidates[best, 1])
        _, outputs = render_rollout(env, actions)
        config = RenderConfig(width=480, height=240, target_frames=60)
        save_gif(render_episode(outputs, env.render_info, config), output, config)
    return scores_np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--output', help='Optionally render the best candidate to a GIF')
    args = parser.parse_args()
    run(args.steps, args.output)
