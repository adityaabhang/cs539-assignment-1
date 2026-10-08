"""Complete the five marked functions. Infrastructure and GUI are provided.

Use instructor/solutions.py only in the instructor release. Public tests specify
contracts without shipping complete answers. All observations are NumPy arrays;
Policy.forward expects float32 torch tensors. See assignment.pdf for equations.
"""
import numpy as np
import torch
from il_assignment.data import Batch, sample, rows_to_batch


def bc_loss(predictions, targets, masks, continuous):
    """TODO 1 (15 points): scalar mean loss.

    Continuous: predictions/targets [B,2], mean squared error over B and 2.
    Discrete: logits [B,A], int64 targets [B], Boolean masks [B,A]. Apply
    the mask BEFORE cross-entropy. No softmax before cross-entropy.
    """
    if continuous:
        # Step 1-4 from the MSE explanation: diff, square, sum, divide by count.
        return ((predictions - targets) ** 2).mean()

    # Step 3: blank out illegal actions BEFORE turning logits into probabilities.
    masked_logits = predictions.masked_fill(~masks, float("-inf"))

    # Step 1: logits -> log-probabilities (numerically-stable log-softmax),
    # done by hand instead of torch.nn.functional.log_softmax.
    log_probs = masked_logits - torch.logsumexp(masked_logits, dim=1, keepdim=True)

    # Step 2: penalty = -log(p) for the probability assigned to the correct class.
    correct_log_probs = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)

    # Step 4: average the per-example penalties over the batch.
    return -correct_log_probs.mean()


def train_bc(policy, data: Batch, *, epochs=12, batch_size=128, lr=1e-3, seed=0):
    """TODO 2 (15 points): shuffled minibatch Adam training; return epoch means.

    Use bc_loss, policy.spec.continuous, and every sample (including last small
    batch). Clear gradients before backward. Set training mode. Do not reset
    weights or normalizer: DAgger calls this again on the accumulated dataset.
    Use torch.Generator for reproducible shuffling. Return a list[float].
    """
    continuous = policy.spec.continuous
    obs = torch.as_tensor(data.obs, dtype=torch.float32)
    actions = torch.as_tensor(data.actions, dtype=torch.float32 if continuous else torch.long)
    masks = torch.as_tensor(data.masks, dtype=torch.bool)
    n = len(data)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)

    policy.train()
    epoch_losses = []
    for _ in range(epochs):
        # Shuffle the whole "flashcard stack" once per epoch, reproducibly.
        perm = torch.randperm(n, generator=generator)

        total_loss, total_count = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]  # last batch may be smaller; still used.

            optimizer.zero_grad()  # clear last batch's leftover gradients first.
            predictions = policy(obs[idx])
            loss = bc_loss(predictions, actions[idx], masks[idx], continuous)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)
            total_count += len(idx)

        epoch_losses.append(total_loss / total_count)

    return epoch_losses


def beta_schedule(round_index):
    """TODO 3 (5 points): probability of EXECUTING teacher action in round i.

    i is zero-based. Return 0.5 ** (i+1): .5, .25, .125, ... . Reject i<0.
    The stored label is always the teacher action, regardless of this choice.
    """
    if round_index < 0:
        raise ValueError("round_index must be >= 0")
    return 0.5 ** (round_index + 1)


def collect_dagger_episode(env, policy, beta, rng):
    """TODO 4 (20 points): collect labels on states actually visited by mixture.

    Environment has ALREADY been reset. Until env.prefix_done:
      obs, mask = env.observe(); label = env.expert_action()
      policy.act(obs, mask) gives learner action
      execute teacher with probability beta, otherwise learner
      append sample(obs, mask, label, executed, env.phase, 'dagger') BEFORE step
      env.step(executed)
    Return list of rows; do not call env.finish() or reset(), relabel after
    stepping, append bot suffixes, or read env.game.board. Query each visited
    state, including learner-executed states.
    """
    rows = []
    while not env.prefix_done:
        obs, mask = env.observe()
        label = env.expert_action()
        learner_action = policy.act(obs, mask)

        # Flip the coin: teacher drives with probability beta, else the learner.
        executed = label if rng.random() < beta else learner_action

        # Record the state actually visited and what the expert says was right,
        # BEFORE moving — moving would land us in a different state to label.
        rows.append(sample(obs, mask, label, executed, env.phase, "dagger"))

        env.step(executed)
    return rows


def aggregate_dataset(old: Batch, new: Batch):
    """TODO 5 (5 points): concatenate each field on axis 0; preserve ALL old rows.

    Return a fresh Batch without modifying old or new. Must handle action
    shapes [N] and [N,2]. DAgger is cumulative, not replacement learning.
    """
    return Batch(
        np.concatenate([old.obs, new.obs]),
        np.concatenate([old.actions, new.actions]),
        np.concatenate([old.masks, new.masks]),
    )
