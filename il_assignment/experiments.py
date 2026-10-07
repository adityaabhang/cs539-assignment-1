"""Provided orchestration. Student functions are loaded only when requested."""
import csv
import importlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import load_bank, rows_to_batch, save_episode, sample
from .domains import make_domain, spec
from .model import Policy, load_policy


def algorithms(implementation):
    try:
        return importlib.import_module("instructor.solutions" if implementation == "solution" else "student.algorithms")
    except ModuleNotFoundError as e:
        raise RuntimeError("Solutions are only in the instructor release. Use --implementation student.") from e


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def protocol(domain, board_decisions=20, max_steps=600, nodes=1024):
    if board_decisions < 2:
        raise ValueError("Use a board decision budget of at least 2")
    if max_steps < 1 or nodes < 1:
        raise ValueError("max_steps and nodes must be positive")
    return {"domain": domain, "board_decisions": board_decisions, "max_steps": max_steps,
            "nodes": nodes, "action_repeat": 2, "board_max_plies": 160, "version": 1}


def environment(config, render=False):
    return make_domain(config["domain"], prefix_moves=config["board_decisions"],
                       max_steps=config["max_steps"], nodes=config["nodes"], render=render)


def collect(config, directory, episodes, seed=0, mode="human"):
    from .ui import CollectorUI, CollectionAborted
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not 0 <= seed < 10000 or seed + episodes > 10000:
        raise ValueError("Collection seeds must be in [0, 10000); validation/test/DAgger seeds are reserved")
    env = environment(config, render=mode == "human")
    ui = None
    try:
        if mode == "human":
            ui = CollectorUI(env)
        for i in range(episodes):
            path = directory / f"episode_{seed+i:05d}.npz"
            if path.exists():
                print(f"Already saved: {path.name}; skipping.", flush=True)
                continue
            env.reset(seed + i)
            rows = []
            start = time.monotonic()
            if ui:
                ui.progress = f"Demonstration {i+1}/{episodes} | seed {seed+i} | 20 board decisions by default"
                ui.ready("Ready. Take a moment to plan; press Enter when ready.")
            while not env.prefix_done:
                obs, mask = env.observe()
                action, source = ui.action() if ui else (env.expert_action(), "teacher")
                rows.append(sample(obs, mask, action, phase=env.phase, source=source))
                env.step(action)
            # Save prefix first, so a slow/crashed engine cannot erase human work.
            metadata = {"seed": seed+i, "protocol": config, "teacher": env.engine_name,
                        "collection_seconds": time.monotonic()-start, "rows": len(rows),
                        "human_rows": sum(r["source"] == "human" for r in rows),
                        "assisted_rows": sum(r["source"] == "assisted" for r in rows), "mode": mode}
            save_episode(path, rows, metadata)
            if ui:
                ui.notice = "Demonstration saved. Bot is finishing the game; please wait."
                ui.draw()
            print(f"Saved {path}: {len(rows)} decisions. Finishing episode...", flush=True)
            result = env.finish()
            write_json(path.with_suffix(".json"), {**metadata, "outcome": result})
            print(json.dumps(result), flush=True)
    except CollectionAborted:
        print("Stopped. Completed demonstrations are saved; unfinished demonstration discarded.")
    finally:
        if ui:
            ui.close()
        env.close()


def train(domain, directory, n, out, *, implementation="student", seed=0, epochs=12, allow_teacher=False):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}")
    data, manifest = load_bank(directory, n, allow_teacher=allow_teacher)
    if manifest["protocol"]["domain"] != domain:
        raise ValueError("Data domain does not match requested policy")
    out.mkdir(parents=True, exist_ok=True)
    policy = Policy(domain)
    policy.set_normalizer(data.obs)
    start = time.monotonic()
    losses = algorithms(implementation).train_bc(policy, data, epochs=epochs, seed=seed)
    policy.save(out / "policy.pt")
    write_json(out / "training.json", {"method": "BC", "domain": domain, "seed": seed, "epochs": epochs,
                                      "loss": losses, "manifest": manifest, "seconds": time.monotonic()-start})
    return policy, data, manifest


def evaluate(policy, config, episodes=20, seed_base=20000):
    """Learned policy acts throughout prefix; teacher queried ONLY for diagnostics.

    Evaluation never adds training data.
    """
    env = environment(config)
    results = []
    try:
        for seed in range(seed_base, seed_base + episodes):
            env.reset(seed)
            errors = {}
            decisions = 0
            while not env.prefix_done:
                obs, mask = env.observe()
                action = policy.act(obs, mask)
                label = env.expert_action()
                err = float(np.mean((np.asarray(action) - label)**2)) if policy.spec.continuous else float(action != label)
                errors.setdefault(env.phase, []).append(err)
                env.step(action)
                decisions += 1
            outcome = env.finish()
            outcome.update({"seed": seed, "prefix_decisions": decisions})
            for phase, values in errors.items():
                outcome[f"{phase}_{'mse' if policy.spec.continuous else 'disagreement'}"] = float(np.mean(values))
            results.append(outcome)
    finally:
        env.close()
    return results


def summarize(results):
    """Percentile bootstrap across episodes, NOT across training seeds."""
    rng = np.random.default_rng(7)
    summary = {"episodes": len(results)}
    for key in results[0]:
        if key == "seed" or not isinstance(results[0][key], (int, float, bool)):
            continue
        values = np.array([r[key] for r in results], float)
        boot = values[rng.integers(0, len(values), (2000, len(values)))].mean(1)
        summary[key] = {"mean": float(values.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist()}
    return summary


def save_evaluation(policy, config, path, episodes=20, seed_base=20000):
    result = evaluate(policy, config, episodes, seed_base)
    output = {"protocol": config, "seed_base": seed_base, "episodes": result, "summary": summarize(result)}
    write_json(path, output)
    return output


def dagger(checkpoint, data_dir, out, *, rounds=3, episodes_per_round=5, epochs=12,
           seed=0, implementation="student", allow_teacher=False, eval_episodes=20, eval_seed_base=20000):
    checkpoint, out = Path(checkpoint), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}")
    info = json.loads((checkpoint.parent / "training.json").read_text())
    initial_n = info["manifest"]["episodes"]
    data, manifest = load_bank(data_dir, initial_n, allow_teacher=allow_teacher)
    # Ensure the same selected files and protocol; do not silently grow the bank.
    if manifest != info["manifest"]:
        raise ValueError("Bank changed or path differs from BC manifest; use the original frozen bank and same path")
    config = manifest["protocol"]
    policy = load_policy(checkpoint)
    alg = algorithms(implementation)
    out.mkdir(parents=True, exist_ok=True)
    env = environment(config)
    torch.set_num_threads(1)
    records = []
    budget = 0
    try:
        for i in range(rounds):
            beta = alg.beta_schedule(i)
            new_rows = []
            for j in range(episodes_per_round):
                episode_seed = 30000 + seed * 1000 + i * episodes_per_round + j
                env.reset(episode_seed)
                rows = alg.collect_dagger_episode(env, policy, beta, np.random.default_rng(episode_seed))
                budget += len(rows)
                new_rows.extend(rows)
                save_episode(out / "labels" / f"round_{i+1}_{j:03d}.npz", rows,
                             {"seed": episode_seed, "protocol": config, "beta": beta, "teacher": env.engine_name})
                # No need to finish suffix during DAgger training: no suffix labels or score used.
            data = alg.aggregate_dataset(data, rows_to_batch(new_rows))
            losses = alg.train_bc(policy, data, epochs=epochs, seed=seed+i+1)
            round_dir = out / f"round_{i+1}"
            round_dir.mkdir()
            policy.save(round_dir / "policy.pt")
            write_json(round_dir / "training.json", {"method": "DAgger", "manifest": manifest,
                       "domain": policy.spec.name, "seed": seed, "round": i+1, "beta": beta,
                       "teacher_labels_total": budget, "rows_total": len(data), "loss": losses})
            evaluation = save_evaluation(policy, config, round_dir / "evaluation.json", eval_episodes, eval_seed_base)
            records.append({"round": i+1, "beta": beta, "teacher_labels_total": budget,
                            "rows_total": len(data), "evaluation": evaluation})
            print(f"DAgger round {i+1}: beta={beta:.3f}, new labels={len(new_rows)}, total labels={budget}", flush=True)
    finally:
        env.close()
    write_json(out / "dagger.json", records)
    return records


def sweep(domain, data_dir, out, *, sizes=None, seed=0, epochs=12, rounds=3,
          episodes_per_round=5, eval_episodes=20, implementation="student", allow_teacher=False, split="test"):
    if sizes is None:
        sizes = (5, 10) if domain == "chess" else (5, 10, 50)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Choose a new, empty run directory: {out}")
    seed_base = 10000 if split == "validation" else 20000
    records = []
    bc_run = None
    for n in sizes:
        run = out / f"bc_{n}"
        policy, data, manifest = train(domain, data_dir, n, run, implementation=implementation, seed=seed,
                                      epochs=epochs, allow_teacher=allow_teacher)
        ev = save_evaluation(policy, manifest["protocol"], run / "evaluation.json", eval_episodes, seed_base)
        records.append({"method": "BC", "demos": n, "round": 0, "teacher_labels": 0, "evaluation": ev})
        print(f"Finished BC with {n} demonstrations", flush=True)
        bc_run = run
    # Original-distribution extra-training control. Match optimizer UPDATE count,
    # not epochs, because the aggregated DAgger dataset grows each round.
    if rounds:
        dg = dagger(bc_run / "policy.pt", data_dir, out / "dagger", rounds=rounds,
                    episodes_per_round=episodes_per_round, epochs=epochs, seed=seed,
                    implementation=implementation, allow_teacher=allow_teacher,
                    eval_episodes=eval_episodes, eval_seed_base=seed_base)
        for row in dg:
            records.append({"method": "DAgger", "demos": sizes[-1], "round": row["round"],
                            "teacher_labels": row["teacher_labels_total"], "evaluation": row["evaluation"]})
        from math import ceil
        # Full epochs on BC's data approximate the same update budget (within one BC epoch).
        extra_updates = sum(ceil(row["rows_total"] / 128) * epochs for row in dg)
        control_epochs = ceil(extra_updates / ceil(len(data) / 128))
        control = load_policy(bc_run / "policy.pt")
        alg = algorithms(implementation)
        loss = alg.train_bc(control, data, epochs=control_epochs, seed=seed+99)
        control_dir = out / "bc_extra_training"
        control_dir.mkdir()
        control.save(control_dir / "policy.pt")
        write_json(control_dir / "training.json", {"method": "BC-extra-training", "manifest": manifest,
                   "extra_epochs": control_epochs, "loss": loss, "target_updates": extra_updates,
                   "actual_updates": control_epochs * ceil(len(data)/128)})
        ev = save_evaluation(control, manifest["protocol"], control_dir / "evaluation.json", eval_episodes, seed_base)
        records.append({"method": "BC-extra-training", "demos": sizes[-1], "round": rounds,
                        "teacher_labels": 0, "evaluation": ev})
    write_json(out / "results.json", records)
    tabulate(records, out / "results.csv")
    plot_results(records, out / "learning_curves.png", domain)
    return records


def tabulate(records, path):
    rows = []
    for run in records:
        identity = {k: run[k] for k in ("method", "demos", "round", "teacher_labels")}
        for ep in run["evaluation"]["episodes"]:
            rows.append({**identity, **ep})
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(records, path, domain):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    key = "return" if domain.startswith("lander") else "score"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, selected, xkey, label in [
        (axes[0], [r for r in records if r["method"] == "BC"], "demos", "Human demonstrations"),
        (axes[1], [r for r in records if r["method"] == "DAgger"], "round", "DAgger round")]:
        if selected:
            x = [r[xkey] for r in selected]
            y = np.array([r["evaluation"]["summary"][key]["mean"] for r in selected])
            ci = np.array([r["evaluation"]["summary"][key]["ci95"] for r in selected]).T
            ax.errorbar(x, y, yerr=np.maximum(0, np.stack([y-ci[0], ci[1]-y])), fmt="o-", color="#31586a", capsize=4)
            ax.set_xticks(x)
        ax.set_xlabel(label)
        ax.set_ylabel("Episode return" if key == "return" else "Hybrid game score (prefix + bot)")
        ax.grid(alpha=.2)
    bc = [r for r in records if r["method"] == "BC"][-1]
    axes[1].axhline(bc["evaluation"]["summary"][key]["mean"], color="#b28c4c", linestyle="--", label=f"BC-{bc['demos']}")
    extra = [r for r in records if r["method"] == "BC-extra-training"]
    if extra:
        axes[1].axhline(extra[0]["evaluation"]["summary"][key]["mean"], color="#777777", linestyle=":", label="BC extra training")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"{domain} | episode-bootstrap 95% intervals; one training seed")
    fig.savefig(path, dpi=160)
    plt.close(fig)
