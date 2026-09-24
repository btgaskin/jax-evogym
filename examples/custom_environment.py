"""A minimal custom reward using the existing Walker physics and observations."""
import jax
import jax.numpy as jnp
from jax_evogym import WalkerV0


@jax.jit
def penalize_effort(reward, action, already_done, coefficient):
    clipped = jnp.clip(action, 0.6, 1.6)
    effort = jnp.sum(jnp.square(clipped - 1.0))
    return jnp.where(already_done, 0.0, reward - coefficient * effort)


class EffortWalker(WalkerV0):
    """Walker reward minus an actuation-effort penalty; no new physics."""
    def __init__(self, body, effort_coefficient=0.001, **kwargs):
        if effort_coefficient < 0:
            raise ValueError('effort_coefficient must be non-negative')
        super().__init__(body, **kwargs)
        self.effort_coefficient = effort_coefficient

    def step(self, state, action):
        obs, next_state, reward, done = super().step(state, action)
        reward = penalize_effort(reward, action, state.done, self.effort_coefficient)
        return obs, next_state, reward, done
