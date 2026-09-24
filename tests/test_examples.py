"""Tiny deterministic documentation checks; population equivalence is CI-only."""
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_evogym import EvoWorld, WalkerV0, CONTRACTILE
from examples.common import BODY, freeze_finished, periodic_actions
from examples.custom_environment import EffortWalker, penalize_effort
from examples.designer_world import load_world


def test_animation_and_designer_export(tmp_path):
    pytest.importorskip('PIL')
    from PIL import Image
    from examples.first_motion import run as first_motion
    from examples.designer_world import run as designer_world
    for name, run, args in [
        ('first', first_motion, ()),
        ('designer', designer_world, (Path(__file__).parents[1] / 'examples/world.json',)),
    ]:
        path = tmp_path / f'{name}.gif'
        final, outputs, frames = run(*args, steps=2, output=str(path))
        assert np.isfinite(np.asarray(outputs.positions)).all()
        assert len(frames) > 0
        with Image.open(path) as image:
            assert image.format == 'GIF'
            assert image.size == (480, 240)


def test_designer_rejects_wrong_name_and_scalar_contractile(tmp_path):
    world = EvoWorld()
    world.add_from_array('robot', np.array([[CONTRACTILE]]), 0, 1)
    path = tmp_path / 'world.json'
    world.to_json(path)
    with pytest.raises(ValueError, match='Available'):
        load_world(path, 'missing')
    with pytest.raises(ValueError, match='per-axis'):
        load_world(path)


def test_terminal_transition_is_kept_then_frozen():
    from examples.common import render_rollout
    env = WalkerV0(BODY, max_steps=1)
    actions = periodic_actions(2, env.n_actuators)
    _, initial = env.reset()
    _, terminal, reward, done = env.step(initial, actions[0])
    assert bool(done)
    final, outputs = render_rollout(env, actions)
    # Compare within one compiled rollout: eager/scan fusion can round differently.
    np.testing.assert_array_equal(outputs.positions[0], outputs.positions[1])
    assert float(outputs.reward[1]) == 0.
    frozen = freeze_finished(terminal, initial)
    for actual, expected in zip(jax.tree.leaves(frozen), jax.tree.leaves(terminal)):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(outputs.reward[0], reward, atol=1e-6)


def test_custom_reward_uses_clipped_action_and_terminal_status():
    np.testing.assert_allclose(penalize_effort(2., jnp.array([100., -100.]), False, 0.1), 1.948)
    assert float(penalize_effort(2., jnp.ones(2), True, 0.1)) == 0.
    env = EffortWalker(BODY)
    obs, state = env.reset()
    next_obs, next_state, reward, done = env.step(state, jnp.ones(env.n_actuators))
    assert next_obs.shape == obs.shape
    assert np.isfinite(float(reward))


@pytest.mark.skipif(os.environ.get('JAX_EVOGYM_TEST_BATCH_EXAMPLES') != '1', reason='Population evaluation runs in CI, not on restricted local hosts')
def test_batch_matches_scalar_evaluations():
    from examples.controller_evaluation import evaluate_one, evaluate_batch
    env = WalkerV0(BODY, max_steps=1)
    candidates = jnp.array([[0., 0.], [0.2, 1.]])
    scores, states = evaluate_batch(env, candidates, steps=2)
    expected = jnp.stack([evaluate_one(env, p, steps=2)[0] for p in candidates])
    np.testing.assert_allclose(scores, expected, atol=1e-6)
    assert np.asarray(states.done).all()
