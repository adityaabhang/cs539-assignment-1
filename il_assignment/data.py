"""Episode-preserving NPZ storage; no pickle or bot suffix rows."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

@dataclass
class Batch:
    obs: np.ndarray
    actions: np.ndarray
    masks: np.ndarray

    def __len__(self):
        return len(self.obs)


def sample(obs, mask, label, executed=None, phase="control", source="human"):
    return {"obs": np.asarray(obs, np.float32).copy(), "action": np.asarray(label).copy(),
            "mask": np.asarray(mask, bool).copy(),
            "executed": np.asarray(label if executed is None else executed).copy(),
            "phase": phase, "source": source}


def rows_to_batch(rows):
    if not rows:
        raise ValueError("Empty demonstration")
    return Batch(np.stack([x["obs"] for x in rows]), np.stack([x["action"] for x in rows]),
                 np.stack([x["mask"] for x in rows]))


def concatenate(batches):
    return Batch(*(np.concatenate([getattr(b, field) for b in batches]) for field in ("obs", "actions", "masks")))


def save_episode(path, rows, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    b = rows_to_batch(rows)
    # Write-exclusive: accidental recollection cannot overwrite previous work.
    with path.open("xb") as f:
        np.savez_compressed(f, obs=b.obs, actions=b.actions, masks=b.masks,
                            executed=np.stack([r["executed"] for r in rows]),
                            phases=np.array([r["phase"] for r in rows]),
                            sources=np.array([r["source"] for r in rows]),
                            metadata=np.array(json.dumps(metadata, sort_keys=True)))


def load_episode(path):
    with np.load(path, allow_pickle=False) as z:
        b = Batch(z["obs"].copy(), z["actions"].copy(), z["masks"].copy())
        meta = json.loads(str(z["metadata"]))
        meta["sources"] = np.unique(z["sources"]).tolist()
    if not np.isfinite(b.obs).all() or not np.isfinite(b.actions).all():
        raise ValueError(f"Non-finite data in {path}")
    if b.actions.ndim == 1 and not b.masks[np.arange(len(b)), b.actions.astype(int)].all():
        raise ValueError(f"Label outside observation-time mask in {path}")
    return b, meta


def load_bank(directory, n, seed=2026, allow_teacher=False):
    files = sorted(Path(directory).glob("episode_*.npz"))
    if len(files) < n:
        raise ValueError(f"Need {n} episodes in {directory}; found {len(files)}")
    # Shuffle the complete fixed bank once; 5 and 10 are prefixes of the same 50.
    chosen = [files[i] for i in np.random.default_rng(seed).permutation(len(files))[:n]]
    batches, metas = zip(*(load_episode(f) for f in chosen))
    if not allow_teacher and any("teacher" in m["sources"] for m in metas):
        raise ValueError("Teacher-generated bank: use --allow-teacher only for smoke tests or an approved alternative.")
    configs = {json.dumps(m["protocol"], sort_keys=True) for m in metas}
    if len(configs) != 1:
        raise ValueError("Mixed collection protocols in one bank")
    seeds = [m["seed"] for m in metas]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate episode seeds")
    return concatenate(batches), {"files": [str(f) for f in chosen], "episodes": n,
                                 "rows": sum(map(len, batches)), "protocol": metas[0]["protocol"],
                                 "sources": sorted(set(s for m in metas for s in m["sources"]))}
