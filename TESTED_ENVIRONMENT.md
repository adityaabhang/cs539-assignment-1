# Tested environment

## Original development validation

The following records the original development environment and checks relevant
to the remaining variants; it is not a claim that the revised assignment has
completed a new full experiment.

- Linux x86_64, Python 3.12.14.
- Stockfish 14.1; standalone external executable, one thread.
- CPU execution; no GPU or online service required.

| Distribution | Tested version |
|---|---|
| torch | 2.8.0+cpu |
| numpy | 2.5.3 |
| gymnasium | 1.2.3 |
| Box2D | 2.3.10 |
| pygame | 2.6.1 |
| python-chess | 1.999 |
| chess | 1.11.2 |
| matplotlib | 3.11.2 |
| pytest | 9.1.1 |
| reportlab | 5.0.1 |
| setuptools | 78.1.0 |

Validation performed:

- Public algorithm/data/domain checks with the instructor implementation.
- End-to-end synthetic smoke runs in both Lander variants and Chess: collection, bot takeover, BC subsets, one DAgger round, learner evaluation, extra-training baseline, CSV/JSON export, and plots.
- Pygame event-path checks for board clicks and Lander control, with headless screenshots visually inspected.
- Nine-page handout rendered and visually inspected page by page.
- Editable installation and command-line entry point tested.

Scope: smoke runs used two teacher-generated demonstrations per variant, two training epochs, and two evaluation episodes; these are infrastructure tests, not performance evidence. No human collection session or full required human-demonstration training study was performed during that original validation. Human workload remains an estimate. Cross-platform installation instructions have not been exercised on Windows or macOS.

## Revised assignment checks

The revised requirement is 50 demonstrations per Lander variant and 10 Chess
games (110 total), with BC subsets of 5/10/50 for Lander and 5/10 for Chess.
On Windows with Python 3.13.5 and Stockfish 19, all 14 remaining public tests
passed using the instructor solutions, and all three domain startup checks
passed. Domain-specific collection/training defaults, sweep subsets, and
explicit overrides were also verified. The rebuilt nine-page PDF was checked
for the revised requirements and identical copies; this was a text check, not
a new page-by-page visual review. These checks do not constitute a full
50/50/10 human-data experiment or a fresh installation test.
