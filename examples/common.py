"""Small deterministic building blocks shared by the introductory examples."""
import jax
import jax.numpy as jnp
import numpy as np
from jax_evogym import H_ACT, SOFT, V_ACT, RenderStepOutput

BODY = np.array([[H_ACT, SOFT, V_ACT]], dtype=np.int32)


def periodic_actions(steps, n_actuators, amplitude=0.25, period=80.0, phase=0.0):
    """An open-loop rhythm in simulator action units, not a learned gait."""
    if steps < 1 or n_actuators < 1:
        raise ValueError("Use positive steps and at least one H_ACT or V_ACT cell.")
    if period <= 0:
        raise ValueError("period must be positive")
    time = jnp.arange(steps, dtype=jnp.float32)[:, None]
    offsets = jnp.arange(n_actuators, dtype=jnp.float32)[None, :] * jnp.pi
    return jnp.clip(1.0 + amplitude * jnp.sin(2 * jnp.pi * time / period + offsets + phase), 0.6, 1.6)


def freeze_finished(previous, proposed):
    """Keep a completed episode's entire state, including its step counter."""
    return jax.tree.map(lambda old, new: jnp.where(previous.done, old, new), previous, proposed)


def render_rollout(env, actions):
    """Capture a single episode, freezing after the terminal transition."""
    _, initial = env.reset()

    def step(state, action):
        _, proposed, reward, _ = env.step(state, action)
        next_state = freeze_finished(state, proposed)
        output = RenderStepOutput(
            next_state.sim_state.positions,
            next_state.sim_state.spring_rest_length,
            jnp.where(state.done, 0.0, reward),
            next_state.done,
        )
        return next_state, output

    return jax.lax.scan(step, initial, actions)
