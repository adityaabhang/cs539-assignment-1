"""CLI: python -m il_assignment.cli --help, or `ilr` after editable install."""
import argparse
import json
from pathlib import Path
import sys
import torch
from .domains import DOMAINS
from .experiments import collect, protocol, train, dagger, sweep, save_evaluation
from .model import load_policy


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main(argv=None):
    p = argparse.ArgumentParser(description="Collect demonstrations, implement BC/DAgger, compare three variants.")
    commands = p.add_subparsers(dest="command", required=True)
    c = commands.add_parser("collect", help="Interactive human demonstrations; resumable, one file per episode")
    c.add_argument("--domain", choices=DOMAINS, required=True)
    c.add_argument("--data", required=True)
    c.add_argument("--episodes", type=positive, help="Default: 10 for Chess, 50 for Lander")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--mode", choices=["human", "teacher"], default="human")
    c.add_argument("--board-decisions", type=positive, default=20)
    c.add_argument("--max-steps", type=positive, default=600)
    c.add_argument("--nodes", type=positive, default=1024)
    for name in ("train", "dagger", "sweep"):
        q = commands.add_parser(name)
        q.add_argument("--data", required=True)
        q.add_argument("--out", required=True)
        q.add_argument("--implementation", choices=["student", "solution"], default="student")
        q.add_argument("--allow-teacher", action="store_true", help="Explicit opt-in for synthetic smoke-test data")
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--epochs", type=positive, default=12)
        if name != "dagger":
            q.add_argument("--domain", choices=DOMAINS, required=True)
        if name == "train":
            q.add_argument("--demos", type=positive, help="Default: 10 for Chess, 50 for Lander")
        else:
            q.add_argument("--rounds", type=positive, default=3)
            q.add_argument("--episodes-per-round", type=positive, default=5)
            q.add_argument("--eval-episodes", type=positive, default=20)
            q.add_argument("--split", choices=["validation", "test"], default="test")
        if name == "dagger":
            q.add_argument("--checkpoint", required=True)
        if name == "sweep":
            q.add_argument("--sizes", nargs="+", type=positive, help="Default: 5 10 for Chess; 5 10 50 for Lander. Override only for smoke tests")
    e = commands.add_parser("evaluate")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--episodes", type=positive, default=20)
    e.add_argument("--split", choices=["validation", "test"], default="test")
    commands.add_parser("doctor", help="Verify dependencies and all three environment resets")
    a = p.parse_args(argv)
    torch.set_num_threads(1)
    try:
        if a.command == "collect":
            collect(protocol(a.domain, a.board_decisions, a.max_steps, a.nodes), a.data, a.episodes or (10 if a.domain == "chess" else 50), a.seed, a.mode)
        elif a.command == "train":
            train(a.domain, a.data, a.demos or (10 if a.domain == "chess" else 50), a.out, implementation=a.implementation, seed=a.seed,
                  epochs=a.epochs, allow_teacher=a.allow_teacher)
        elif a.command == "dagger":
            dagger(a.checkpoint, a.data, a.out, rounds=a.rounds, episodes_per_round=a.episodes_per_round,
                   epochs=a.epochs, seed=a.seed, implementation=a.implementation, allow_teacher=a.allow_teacher,
                   eval_episodes=a.eval_episodes, eval_seed_base=10000 if a.split == "validation" else 20000)
        elif a.command == "sweep":
            if a.sizes is not None and a.sizes != sorted(set(a.sizes)):
                raise ValueError("--sizes must be increasing and unique")
            sweep(a.domain, a.data, a.out, sizes=a.sizes, seed=a.seed, epochs=a.epochs, rounds=a.rounds,
                  episodes_per_round=a.episodes_per_round, eval_episodes=a.eval_episodes,
                  implementation=a.implementation, allow_teacher=a.allow_teacher, split=a.split)
        elif a.command == "evaluate":
            info = json.loads((Path(a.checkpoint).parent / "training.json").read_text())
            save_evaluation(load_policy(a.checkpoint), info["manifest"]["protocol"], a.out,
                            a.episodes, 10000 if a.split == "validation" else 20000)
        else:
            from .experiments import environment
            for domain in DOMAINS:
                env = environment(protocol(domain))
                try:
                    obs, mask = env.reset(912)
                    act = env.expert_action()
                    env.step(act)
                    print(f"OK {domain}: observation={obs.shape}, actions={env.spec.action_dim}; {env.engine_name}")
                finally:
                    env.close()
    except (ValueError, RuntimeError, FileNotFoundError, NotImplementedError) as error:
        p.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
