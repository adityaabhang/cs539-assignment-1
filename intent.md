# Intent: Implement `student/algorithms.py` (BC + DAgger)
Author: Aditya Abhang. Status: draft.

## Problem
`student/algorithms.py` has 5 stub functions, each raising `NotImplementedError`. Nothing downstream (`ilr train`, `ilr dagger`, `ilr sweep`) can run until they're implemented. `tests/test_algorithms.py` currently fails 8/8.

## Proposed outcome
All 5 functions implemented to the exact contracts already specified by their docstrings and enforced by the public tests:
- `bc_loss` — MSE for continuous, mask-before-cross-entropy for discrete.
- `train_bc` — reproducible shuffled minibatch Adam training, returns per-epoch mean losses.
- `beta_schedule` — `0.5 ** (i+1)` for `i >= 0`, `ValueError` for `i < 0`.
- `collect_dagger_episode` — queries teacher at every visited state, executes teacher-vs-learner mixture per `beta`, stores teacher label as `action` and the executed action as `executed`, appends the row before stepping.
- `aggregate_dataset` — concatenates old + new `Batch` on axis 0 without aliasing, preserving all history.

`pytest tests/test_algorithms.py` passes 8/8 when done.

## Affected users and systems
- Only `student/algorithms.py` changes. No edits to `il_assignment/` (provided harness), action vocabulary, masks, prefix budgets, or model architecture.
- Downstream consumers: `il_assignment/cli.py` (`ilr train`, `ilr dagger`, `ilr sweep`), `il_assignment/experiments.py`.

## Constraints
- Do not reset policy weights or the observation normalizer inside `train_bc` (DAgger calls it again on the accumulated dataset).
- `collect_dagger_episode` must not call `env.finish()`/`reset()`, must not relabel after stepping, must not read `env.game.board`, and must query every visited state (including learner-executed ones) before stepping.
- `aggregate_dataset` must return a fresh `Batch` (no shared memory with inputs) and handle both discrete `[N]` and continuous `[N,2]` action shapes.
- No softmax before cross-entropy in `bc_loss`; mask applied to logits before the loss, not after.

## Open questions
None — contracts are fully specified by docstrings + `tests/test_algorithms.py`. Build order: one function at a time, red-green, in docstring order (`bc_loss` → `train_bc` → `beta_schedule` → `collect_dagger_episode` → `aggregate_dataset`).
