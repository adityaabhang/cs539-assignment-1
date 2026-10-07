# Assignment 1: From demonstrations to interactive imitation

Collect human data, implement behavior cloning (BC) and DAgger, and compare three environment variants. The handout is **`assignment.pdf`**. Plan for **approximately 3–6 hours of active work**, depending on chess familiarity and debugging. This is a workload estimate, not a student-piloted measurement. A CPU is sufficient; no online game account is needed.

## What you will do

- Collect **50 discrete Lunar Lander demonstrations, 50 continuous Lunar Lander demonstrations, and 10 Chess games**: 110 demonstrations total.
- Each Lander demonstration ends at termination or 600 physics steps. One control decision repeats for two physics steps (25 Hz in the GUI).
- Each Chess demonstration records **up to 20 White moves**; Black replies automatically. This is a cap of 20 student actions, not 20 total plies. Early game endings produce shorter demonstrations and still count.
- After each board-game prefix, a provided bot completes the game. Its continuation is **not** added to the demonstration data.
- Implement the **five functions** in `student/algorithms.py`. Run BC with nested subsets of 5/10/50 complete demonstrations for each Lander variant and 5/10 games for Chess, three DAgger rounds, and the supplied extra-training BC control. Write a 4–6 page report.

The code supports three variants of two domains. Chess's challenge is combinatorial structure, large action vocabulary, and sparse coverage; continuous Lander already has infinitely many mathematical states, so raw state-space cardinality is not a useful comparison.

## Setup

Use **Python 3.12** on a local desktop. Python 3.10–3.13 is accepted by package metadata but only Linux/Python 3.12 was exercised during development. Run commands from this folder.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# Linux CPU build (avoids downloading CUDA):
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

On macOS, install `torch==2.8.0` from the default pip index instead. On Windows, use `py -3.12 -m venv .venv` and `.venv\Scripts\Activate.ps1`; remaining Python commands are the same. Gymnasium's Box2D dependency may need a compiler and SWIG when a wheel is unavailable:

```bash
# Ubuntu/Debian, only if Box2D installation fails:
sudo apt-get install build-essential swig python3-dev
# macOS: install Xcode Command Line Tools and `brew install swig`
```

Install [Stockfish](https://stockfishchess.org/download/) separately. Ubuntu/Debian: `sudo apt-get install stockfish`; macOS: `brew install stockfish`. Windows: download and extract the official binary appropriate for your CPU. Set the full executable path if it is not on PATH:

```bash
export STOCKFISH_EXECUTABLE=/usr/games/stockfish
# macOS Apple Silicon commonly: /opt/homebrew/bin/stockfish
# Windows PowerShell:
# $env:STOCKFISH_EXECUTABLE = 'C:\tools\stockfish\stockfish.exe'
ilr doctor
```

Keep the same Stockfish build throughout the experiment and report its name/version. Engines use one thread, a cleared search hash, and 1,024 nodes per query. Different engine versions can give different labels and outcomes. The default node limit intentionally favors short classroom runs; it is not a perfect oracle.

`ilr` is equivalent to `python -m il_assignment.cli`. If a managed execution sandbox hangs at engine startup, run this local program in a regular terminal; the chess library uses asynchronous subprocess communication. A visible desktop is required for human collection. `SDL_VIDEODRIVER=dummy` is only for automated UI tests, not collecting human demonstrations.

## Collect the three banks

Practice a few episodes into `data/practice-*` first; do not count these toward the frozen banks. Do not discard poor demonstrations selectively. For the actual banks:

```bash
ilr collect --domain lander-discrete --data data/lander-discrete --episodes 50
ilr collect --domain lander-continuous --data data/lander-continuous --episodes 50
ilr collect --domain chess --data data/chess --episodes 10
```

**Enter** starts the next episode. **Esc** discards only the unfinished episode and exits. Rerun the same command to resume: existing seed-numbered files are skipped, never overwritten. The prefix is saved before the bot finishes. A JSON sidecar adds the final outcome if takeover completes.

| Variant | Controls |
|---|---|
| Discrete Lander | Up = main engine; Left = left jet; Right = right jet; no key = coast. Main has priority. |
| Continuous Lander | Hold the left mouse button in the control pad: vertical position controls main throttle, horizontal controls side throttle. Alternatively W/S ramp main, A/D ramp side; Space cuts main. |
| Chess | Click source then destination. Q/R/B/N selects a promotion (default queen). |

For continuous Lander, controls are in [-1,1]^2. Main values below zero switch the engine off; nonnegative values produce 50–100% thrust. Side thrust has a dead zone of (-0.5,0.5). This actuator nonlinearity makes action MSE different from physical control error. The UI provides genuinely graded actions, not only a binary keyboard mapping.

**H accepts one teacher action.** It is recorded as `assisted`, not `human`. Use it when needed, but report the assisted fraction. `--mode teacher` generates explicitly labeled synthetic data for smoke tests; it does not satisfy the human-collection task. Training rejects teacher-generated banks unless `--allow-teacher` is supplied. Do not use that flag for the required experiment.

Complete and freeze all 50 files per Lander bank and all 10 files in the Chess bank **before** comparing subsets: 5/10/50 for Lander, 5/10 for Chess. Selection uses one fixed episode permutation, so the smaller sets are nested. Adding more files afterward changes this permutation. Use the same data path when running standalone DAgger because the BC manifest is checked.

## Implement and test

| Function in `student/algorithms.py` | Points | Responsibility |
|---|---:|---|
| `bc_loss` | 15 | Continuous MSE; discrete cross-entropy with masks applied to logits first. |
| `train_bc` | 15 | Reproducible shuffled minibatch Adam; train all samples; return epoch losses. |
| `beta_schedule` | 5 | Teacher execution probability .5, .25, .125 for rounds 0, 1, 2. |
| `collect_dagger_episode` | 20 | Query teacher at visited observations, execute the mixture, store the teacher label before stepping. |
| `aggregate_dataset` | 5 | Preserve all earlier data and append the new labeled rows. |

```bash
python -m pytest tests/test_infrastructure.py -q
python -m pytest tests/test_algorithms.py -q
```

The algorithm tests should initially fail with `NotImplementedError`. Passing a tiny overfitting test is a debugging check, not evidence of a strong game-playing policy. Do not edit the supplied action vocabulary, masks, prefix budgets, or model architecture for the main comparison. Optional ablations belong in separately named runs.

## Run experiments

First debug with one domain and a small validation run:

```bash
ilr train --domain lander-discrete --data data/lander-discrete --demos 5 --out runs/debug-bc
ilr evaluate --checkpoint runs/debug-bc/policy.pt --split validation --episodes 5 --out runs/debug-validation.json
```

The complete required experiment is one command per variant:

```bash
ilr sweep --domain lander-discrete --data data/lander-discrete --out runs/lander-discrete
ilr sweep --domain lander-continuous --data data/lander-continuous --out runs/lander-continuous
ilr sweep --domain chess --data data/chess --out runs/chess
```

Defaults: collection and standalone BC training use 50 demonstrations for Lander and 10 games for Chess. Sweeps use 5/10/50 for Lander and 5/10 for Chess. Other defaults: one training seed (`--seed 0`), 12 BC epochs, batch size 128, Adam lr 0.001, two hidden layers of 64 units; 3 DAgger rounds × 5 new prefixes each; 12 warm-start training epochs per round; 20 evaluation episodes per checkpoint. The Lander normalizer is fitted only to the initial training subset and remains fixed during DAgger. Board planes have fixed scales.

Each sweep writes checkpoints, training curves, collection manifests, raw episode metrics, `results.csv`, `results.json`, and `learning_curves.png`. DAgger starts from BC-50 for Lander or BC-10 for Chess. The extra-training control continues that same initial BC policy for approximately the same additional optimizer-update count as DAgger, using only the original data. Counts are recorded; rounding is within one BC epoch. The optimizer state is reset on each `train_bc` call for both methods, while network weights persist. This control does not match the teacher's time or data distribution.

Collection seeds: 0–9999; validation: 10000–10019 by default; final test: 20000–20019; DAgger: 30000+. All compared policies use the same evaluation seeds. Do not tune against the final test results. If you tune, use `--split validation` first, freeze settings, then run final sweeps. Repeating final sweeps after looking at test outcomes compromises the holdout. Optional: three training seeds in different output folders; episode-bootstrap intervals from one run do not measure training-seed variability.

Standalone DAgger is also available:

```bash
ilr dagger --checkpoint runs/chess/bc_10/policy.pt --data data/chess --out runs/chess-dagger-new
```

DAgger records every oracle query, including learner-executed states. Stored `action` is the teacher's target; `executed` is the action actually taken by the mixture. Evaluation executes only the learner during its prefix; no expert mixing. Evaluation teacher queries are diagnostics and never become training rows.

## What is recorded?

Each `episode_*.npz` contains `obs`, `actions`, `masks`, `executed`, `phases`, `sources`, and a JSON `metadata` string. Load with `numpy.load(..., allow_pickle=False)`. Metadata identifies seed, protocol, teacher, collection duration, and label provenance. The `.json` sidecar contains the hybrid outcome. No RGB frames are stored as model inputs.

| Variant | Observation | Action | Maximum rows per demonstration |
|---|---|---|---:|
| Lander discrete | 8 float32 values | scalar class, 4 choices | 300 |
| Lander continuous | 8 float32 values | 2 float32 controls | 300 |
| Chess | 1,032 float32 features | masked move ID | 20 |

Board features: 12 piece planes, an empty-square plane, two unused zero planes, and an en-passant plane; plus side-to-move, an unused zero, four castling flags, halfmove counter, and White move index. Chess sees the actual board; full repetition history is not encoded. The fixed 2,034-slot output comprises 1,968 geometric moves/promotions and 66 unused slots. The action mask enables only legal Chess moves; unused slots are always masked out.

For one maximum-length board prefix, float32 observations alone require 20 × 1,032 × 4 = **82,560 bytes**. Boolean masks add 20 × 2,034 = **40,680 bytes** before compression. NPZ also stores labels, provenance, and metadata. Report actual compressed bank sizes and row counts; demonstrations differ in length, so 10 Chess games or 50 Lander demonstrations contain many more than 10 or 50 training examples.

## Report and submission

Use `report_template.md`. Submit a 4–6 page PDF plus completed `student/algorithms.py`, your three human data banks (50 discrete Lander, 50 continuous Lander, 10 Chess; NPZ and sidecars), run manifests, CSV/JSON results, and plots. Exclude `.venv`, engine binaries, instructor files, caches, and practice/smoke datasets. Keep raw demonstrations for reproducibility. See the handout for the rubric (60 code + 40 report).

Required figures: BC performance versus 5/10/50 demonstrations for each Lander variant and 5/10 games for Chess; DAgger performance by round or cumulative labels with the initial BC baseline (BC-50 for Lander, BC-10 for Chess) and extra-training controls; supporting prefix/teacher-disagreement metrics. The supplied charts provide the main curves; use the CSV for your diagnostic plot. Lander reports return and return >= 200 rate. Chess reports hybrid W/D/L, score = win + 0.5 draw, and prefix material. Truncations and turn-limit endings must be disclosed.

Interpret failures as results. Higher demonstration count need not improve every test estimate. DAgger combines new state coverage with a switch from human labels to provided teacher labels, so an improvement cannot be attributed to distribution correction alone. Chess agreement with Stockfish is not identical to move quality. A win after takeover is not evidence that the learner can play a full game unaided.

## Sources and layout

See `REFERENCES.md` and `THIRD_PARTY.md` for papers and library sources. `il_assignment/` contains provided environments, UI, storage, models, and orchestration; `student/` is your work; `tests/` contains public checks. `instructor/` exists only in the instructor release. `tools/build_handout.py` reproduces the handout using ReportLab; no external images or remote assets are required.
