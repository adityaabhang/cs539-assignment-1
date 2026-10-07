"""Provided data/domain checks; stockfish tests skip if executable unavailable."""
import numpy as np
import pytest
import chess
from il_assignment.data import save_episode, sample, load_episode, load_bank
from il_assignment.domains import MOVES, MOVE_IDS, stockfish_path
from il_assignment.experiments import environment, protocol


def board_env(domain, **kwargs):
    try:
        stockfish_path()
    except RuntimeError:
        pytest.skip("Stockfish not installed")
    return environment(protocol(domain, **kwargs))


def test_vocabulary_and_promotions():
    assert len(MOVES) == len(MOVE_IDS)
    for uci in ("e2e4", "e1g1", "b1c3", "a7a8n", "h2h1r"):
        assert chess.Move.from_uci(uci) in MOVE_IDS


def test_episode_roundtrip_and_provenance(tmp_path):
    config = protocol("lander-discrete")
    row = sample(np.zeros(8), np.ones(4, bool), 2, source="teacher")
    path = tmp_path / "episode_00000.npz"
    save_episode(path, [row], {"seed": 0, "protocol": config})
    batch, meta = load_episode(path)
    assert len(batch) == 1 and meta["sources"] == ["teacher"]
    with pytest.raises(FileExistsError):
        save_episode(path, [row], {})
    with pytest.raises(ValueError, match="Teacher-generated"):
        load_bank(tmp_path, 1)
    assert len(load_bank(tmp_path, 1, allow_teacher=True)[0]) == 1


def test_nested_episode_subsets(tmp_path):
    for i in range(12):
        save_episode(tmp_path / f"episode_{i:05d}.npz", [sample(np.zeros(8)+i, np.ones(4, bool), 0)],
                     {"seed": i, "protocol": protocol("lander-discrete")})
    _, a = load_bank(tmp_path, 5)
    _, b = load_bank(tmp_path, 10)
    assert a["files"] == b["files"][:5]


@pytest.mark.parametrize("domain", ["lander-discrete", "lander-continuous"])
def test_lander_terminates_and_shapes(domain):
    env = environment(protocol(domain, max_steps=4))
    try:
        obs, mask = env.reset(3)
        assert obs.shape == (8,)
        while not env.prefix_done:
            a = env.expert_action()
            assert np.shape(a) == ((2,) if env.spec.continuous else ())
            env.step(a)
        result = env.finish()
        assert result["physics_steps"] <= 4 and result["truncated"]
    finally:
        env.close()


def test_chess_prefix_is_white_moves_not_plies():
    env = board_env("chess", board_decisions=2, nodes=64)
    try:
        env.reset(0)
        env.step(MOVE_IDS[chess.Move.from_uci("e2e4")])
        assert env.board.turn == chess.WHITE and env.moves == 1 and not env.prefix_done
        env.step(env.expert_action())
        assert env.prefix_done and env.moves == 2
    finally:
        env.close()
