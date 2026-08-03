"""Backward-compatible re-exports from environments package.

The canonical location is now jax_evogym.environments.walker.
This module preserves the old import path for existing code.
"""

from jax_evogym.environments.walker import WalkerV0, make_walker_episode_step

__all__ = ["WalkerV0", "make_walker_episode_step"]
