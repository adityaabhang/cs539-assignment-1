"""Public contracts. Expected to fail until student/algorithms.py is completed.

Run ILR_IMPLEMENTATION=solution pytest tests/test_algorithms.py in instructor copy.
"""
import importlib
import os
import numpy as np
import pytest
import torch
from il_assignment.data import Batch
from il_assignment.model import Policy

alg = importlib.import_module("instructor.solutions" if os.environ.get("ILR_IMPLEMENTATION") == "solution" else "student.algorithms")


def test_continuous_loss_and_gradient():
    pred = torch.tensor([[1., 0.], [0., -1.]], requires_grad=True)
    target = torch.zeros_like(pred)
    loss = alg.bc_loss(pred, target, torch.ones_like(pred, dtype=torch.bool), True)
    assert loss.item() == pytest.approx(.5)
    loss.backward()
    torch.testing.assert_close(pred.grad, pred.detach()/2)


def test_discrete_mask_before_normalization():
    x = torch.tensor([[0., 1000., 0.]], requires_grad=True)
    loss = alg.bc_loss(x, torch.tensor([0]), torch.tensor([[True, False, True]]), False)
    assert loss.item() == pytest.approx(np.log(2))
    loss.backward()
    assert x.grad[0, 1].item() == 0
    assert torch.isfinite(x.grad).all()


def test_beta_decreases():
    assert [alg.beta_schedule(i) for i in range(3)] == [.5, .25, .125]
    with pytest.raises(ValueError):
        alg.beta_schedule(-1)


class ToyEnv:
    phase = "move"
    def __init__(self):
        self.state = 0
        self.prefix_done = False
        self.queries = []
    def observe(self):
        return np.array([self.state], np.float32), np.ones(2, bool)
    def expert_action(self):
        self.queries.append(self.state)
        return 1
    def step(self, action):
        self.state += 1 + int(action)
        self.prefix_done = self.state >= 3


class Learner:
    def act(self, obs, mask):
        return 0


@pytest.mark.parametrize("beta,states,executed", [(0, [0, 1, 2], 0), (1, [0, 2], 1)])
def test_dagger_labels_visited_states_not_learner_actions(beta, states, executed):
    env = ToyEnv()
    rows = alg.collect_dagger_episode(env, Learner(), beta, np.random.default_rng(1))
    assert env.queries == states
    assert [r["obs"].item() for r in rows] == states
    assert all(r["action"].item() == 1 for r in rows)
    assert all(r["executed"].item() == executed for r in rows)
    assert all(r["source"] == "dagger" for r in rows)


@pytest.mark.parametrize("continuous", [False, True])
def test_aggregation_preserves_history_without_aliasing(continuous):
    actions = np.zeros((2, 2) if continuous else (2,))
    a = Batch(np.ones((2, 8)), actions, np.ones((2, 2), bool))
    b = Batch(np.zeros((1, 8)), actions[:1], np.ones((1, 2), bool))
    result = alg.aggregate_dataset(a, b)
    assert len(result) == 3
    np.testing.assert_array_equal(result.obs[:2], a.obs)
    assert not np.shares_memory(result.obs, a.obs)
    assert result.actions.shape == ((3, 2) if continuous else (3,))


def test_train_overfits_tiny_supervised_problem():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    p = Policy("lander-discrete", hidden=16)
    data = Batch(np.zeros((17, 8), np.float32), np.full(17, 2, np.int64), np.ones((17, 4), bool))
    losses = alg.train_bc(p, data, epochs=30, batch_size=8, lr=.03, seed=0)
    assert len(losses) == 30
    assert losses[-1] < losses[0] * .1
    assert p.act(data.obs[0], data.masks[0]) == 2
