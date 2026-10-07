"""Three variants, one decision interface. Chess actions are White moves."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil

import chess
import chess.engine
import numpy as np

DOMAINS = ("lander-discrete", "lander-continuous", "chess")

# A fixed geometric vocabulary, including all four promotion choices and pass.
def _move_vocabulary():
    result = []
    for src in chess.SQUARES:
        for dst in chess.SQUARES:
            dx = abs(chess.square_file(src) - chess.square_file(dst))
            dy = abs(chess.square_rank(src) - chess.square_rank(dst))
            if src != dst and (dx == 0 or dy == 0 or dx == dy or sorted((dx, dy)) == [1, 2]):
                result.append(chess.Move(src, dst))
            if (chess.square_rank(src), chess.square_rank(dst)) in ((6, 7), (1, 0)) and dx <= 1:
                for piece in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
                    result.append(chess.Move(src, dst, promotion=piece))
    return [None] + result

MOVES = _move_vocabulary()
MOVE_IDS = {m: i for i, m in enumerate(MOVES)}
BOARD_ACTIONS = len(MOVES) + 65  # Unused masked slots retain the existing Chess layout.
BOARD_OBS = 16 * 64 + 8

@dataclass(frozen=True)
class Spec:
    name: str
    obs_dim: int
    action_dim: int
    continuous: bool = False


def spec(name):
    if name not in DOMAINS:
        raise ValueError(f"Unknown domain {name}")
    if name.startswith("lander"):
        return Spec(name, 8, 2 if name.endswith("continuous") else 4, name.endswith("continuous"))
    return Spec(name, BOARD_OBS, BOARD_ACTIONS)


def stockfish_path():
    value = os.environ.get("STOCKFISH_EXECUTABLE") or shutil.which("stockfish")
    if not value and Path("/usr/games/stockfish").is_file():
        value = "/usr/games/stockfish"
    if not value or not Path(value).is_file():
        raise RuntimeError("Install Stockfish and set STOCKFISH_EXECUTABLE to its executable path. See README.md.")
    os.environ["STOCKFISH_EXECUTABLE"] = str(Path(value).resolve())
    return os.environ["STOCKFISH_EXECUTABLE"]


class Engine:
    """Fixed node budgets, one thread, no retained search hash between decisions."""
    def __init__(self, nodes=1024):
        self.nodes = nodes
        self.raw = chess.engine.SimpleEngine.popen_uci(stockfish_path())
        self.raw.configure({"Threads": 1, "Hash": 16})
        self.name = self.raw.id.get("name", "Stockfish")

    def play(self, board, limit=None, **kwargs):
        # Validate the position before sending it to Stockfish.
        board = board.copy(stack=False)
        board.castling_rights = board.clean_castling_rights()
        if board.status() & chess.STATUS_INVALID_EP_SQUARE:
            board.ep_square = None
        if not board.is_valid():
            raise chess.engine.EngineError("Invalid chess position")
        self.raw.configure({"Clear Hash": None})
        return self.raw.play(board, chess.engine.Limit(nodes=self.nodes), **kwargs)

    def quit(self):
        try:
            self.raw.quit()
        except chess.engine.EngineTerminatedError:
            pass


def encode_board(board, *, moves=0):
    """Chess features; two zero planes and one zero global retain the fixed layout."""
    p = np.zeros((16, 64), np.float32)
    for square, piece in board.piece_map().items():
        p[(0 if piece.color else 6) + piece.piece_type - 1, square] = 1
    occupied = p[:12].sum(axis=0) > 0
    p[12] = ~occupied
    if board.ep_square is not None:
        p[15, board.ep_square] = 1
    flags = [board.turn == chess.WHITE, False,
             board.has_kingside_castling_rights(chess.WHITE), board.has_queenside_castling_rights(chess.WHITE),
             board.has_kingside_castling_rights(chess.BLACK), board.has_queenside_castling_rights(chess.BLACK),
             min(board.halfmove_clock, 100) / 100, min(moves, 100) / 100]
    return np.concatenate([p.reshape(-1), np.asarray(flags, np.float32)])


class Lander:
    def __init__(self, name, prefix_moves=20, max_steps=600, nodes=1024, render=False):
        import gymnasium as gym
        from gymnasium.envs.box2d.lunar_lander import heuristic
        self.spec = spec(name)
        self.env = gym.make("LunarLander-v3", continuous=self.spec.continuous,
                            render_mode="rgb_array" if render else None, max_episode_steps=max_steps)
        self.heuristic = heuristic
        self.repeat = 2  # same 25 Hz decision interface for humans, teachers, learners
        self.engine_name = "Gymnasium 1.2.3 heuristic (not optimal)"
        self.phase = "control"

    def reset(self, seed):
        self.obs, _ = self.env.reset(seed=seed)
        self.done = self.prefix_done = False
        self.total_reward = 0.0
        self.physics_steps = self.decisions = 0
        self.terminated = self.truncated = False
        self.last_event = ""
        return self.observe()

    def observe(self):
        return self.obs.astype(np.float32).copy(), np.ones(self.spec.action_dim, bool)

    def expert_action(self):
        return self.heuristic(self.env.unwrapped, self.obs)

    def step(self, action):
        if self.done:
            raise RuntimeError("Step after episode end")
        if self.spec.continuous:
            action = np.asarray(action, np.float32)
            if action.shape != (2,) or not np.isfinite(action).all() or (np.abs(action) > 1).any():
                raise ValueError("Continuous action must have shape (2,) in [-1,1]")
        elif int(action) not in range(4):
            raise ValueError("Invalid discrete action")
        for _ in range(self.repeat):
            self.obs, reward, self.terminated, self.truncated, _ = self.env.step(action)
            self.total_reward += float(reward)
            self.physics_steps += 1
            if self.terminated or self.truncated:
                self.done = self.prefix_done = True
                break
        self.decisions += 1

    def finish(self):
        while not self.done:
            self.step(self.expert_action())
        return {"return": self.total_reward, "success": float(self.total_reward >= 200),
                "physics_steps": self.physics_steps, "decisions": self.decisions,
                "truncated": self.truncated, "teacher": self.engine_name}

    def close(self):
        self.env.close()


class ChessDomain:
    def __init__(self, name="chess", prefix_moves=20, max_steps=600, nodes=1024, render=False):
        self.spec = spec(name)
        self.prefix_moves = prefix_moves
        self.max_plies = 160
        self.engine = Engine(nodes)
        self.engine_name = self.engine.name
        self.phase = "move"

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.board = chess.Board()
        self.moves = 0
        self.done = self.prefix_done = False
        self.last_event = "You play White. Click a source and destination square."
        self.prefix_material = None
        return self.observe()

    def observe(self):
        mask = np.zeros(BOARD_ACTIONS, bool)
        for m in self.board.legal_moves:
            mask[MOVE_IDS[m]] = True
        return encode_board(self.board, moves=self.moves), mask

    def expert_action(self):
        if self.done:
            raise RuntimeError("Oracle queried after game end")
        return MOVE_IDS[self.engine.play(self.board).move]

    def _over(self):
        self.done = self.board.is_game_over(claim_draw=True) or self.board.ply() >= self.max_plies

    def step(self, action):
        if self.done or not self.observe()[1][int(action)]:
            raise ValueError("Invalid Chess action or finished game")
        move = MOVES[int(action)]
        self.last_event = "White: " + move.uci()
        self.board.push(move)
        self.moves += 1
        self._over()
        if not self.done:
            # Deliberately imperfect, seeded opponent gives diverse openings.
            if self.moves <= 4 and self.rng.random() < .25:
                legal = list(self.board.legal_moves)
                reply = legal[int(self.rng.integers(len(legal)))]
            else:
                reply = self.engine.play(self.board).move
            self.board.push(reply)
            self.last_event += " / Black: " + reply.uci()
            self._over()
        self.prefix_done = self.done or self.moves >= self.prefix_moves
        if self.prefix_done and self.prefix_material is None:
            self.prefix_material = material(self.board)

    def finish(self):
        # Bot takeover is deliberately NOT returned as training observations.
        while not self.done:
            self.step(self.expert_action())
        outcome = self.board.outcome(claim_draw=True)
        score = .5 if outcome is None or outcome.winner is None else float(outcome.winner == chess.WHITE)
        return {"score": score, "win": float(score == 1), "draw": float(score == .5),
                "loss": float(score == 0), "prefix_material": self.prefix_material,
                "white_moves": self.moves, "capped": self.board.ply() >= self.max_plies,
                "teacher": self.engine_name}

    def close(self):
        self.engine.quit()


def material(board):
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    return sum(v * (len(board.pieces(p, chess.WHITE)) - len(board.pieces(p, chess.BLACK))) for p, v in values.items())


def make_domain(name, **kwargs):
    spec(name)
    if name.startswith("lander"):
        return Lander(name, **kwargs)
    return ChessDomain(name, **kwargs)
