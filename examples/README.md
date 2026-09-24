# Runnable examples

From the repository root, install `python -m pip install -e ".[viz,dev]"`.
Run modules from that same directory; these examples are source-checkout tools,
not installed `jax_evogym` modules.

| Example | Command | Expected result |
|---|---|---|
| First motion | `python -m examples.first_motion --output /tmp/first-motion.gif` | A short GIF of a three-cell robot under periodic actuation; no learned gait is claimed. |
| Designer world | `python -m examples.designer_world evogym-world.json --robot robot --output /tmp/world.gif` | A physics-only GIF using the exported object positions and terrain. |
| Controller evaluation | `python -m examples.controller_evaluation --steps 100 --output /tmp/best.gif` | Four scores, a best-candidate index, and a GIF. Run on a host permitted to perform batched evaluation. |
| Custom reward | Import `EffortWalker` from `examples.custom_environment` | Walker behaviour with a configurable actuation-effort penalty. |

The first two defaults run one deterministic 300-step episode (0.9 simulated
seconds). The comparison runs four fixed candidates for 100 steps each; it is
not training. First calls include JAX compilation. No example allocates cloud
resources automatically. Use an appropriate remote/GPU environment for
population-scale evaluation or training.

The scalar examples use H_ACT and V_ACT. CONTRACTILE requires the per-axis
path described in the Controllers guide. All generated paths are explicit;
use `--steps 2` for a short execution check rather than a useful animation.

The guides link to these complete source files. `tests/test_examples.py`
checks small deterministic cases; batch equivalence runs in CI with
`JAX_EVOGYM_TEST_BATCH_EXAMPLES=1` and is skipped on restricted local hosts.
