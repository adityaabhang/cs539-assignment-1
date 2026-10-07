# Assignment 1 report

Name / collaborators / training seed(s) / Python and Stockfish versions / active time

## 1. Protocol and demonstrations

Describe controls, practice, collection quality, assisted-action usage, and game familiarity. Provide one table for all three variants: 50 demonstrations per Lander variant and 10 games for Chess, decision-row count, length distribution, compressed bytes, human/assisted fraction, and collection outcome summaries. Explain the board-prefix limits and early endings. Identify the frozen 5/10/50 Lander manifests and 5/10 Chess manifests and held-out evaluation seeds.

## 2. BC across representation and action choices

Explain masked cross-entropy versus continuous MSE, the provided architecture, training settings, and action restrictions. Include three-panel BC curves: 5/10/50 demonstrations for each Lander variant and 5/10 games for Chess with episode confidence intervals. Distinguish sample count from episode count. Discuss continuous-control dead zones and sparse combinatorial coverage in Chess. A larger feature vector is not proof of a larger mathematical state space.

## 3. DAgger

Explain when you query the teacher and why its label differs from the mixture's executed action. Report each round's beta, new and cumulative oracle labels, aggregate rows, and training update budget. Plot performance with the initial BC baseline (BC-50 for Lander, BC-10 for Chess) and extra-training controls. Discuss whether additional coverage, teacher quality, optimization, or sampling noise plausibly explains the outcome.

## 4. Diagnostics and interpretation

Include at least one diagnostic figure using prefix material or teacher disagreement. Describe one specific failure or surprising trend in each domain family (Lander, Chess). Explain why teacher disagreement can coexist with good return or vice versa. Disclose truncations and caps; distinguish prefix skill from hybrid final outcomes. Identify limitations of one training seed and a finite evaluation sample.

## 5. Conclusions and one follow-up

Which result changed your intuition? Propose one concrete, controlled experiment that could distinguish two competing explanations of a trend. Do not claim that BC or DAgger must improve monotonically. Cite at least Osa et al. (2018) and Ross et al. (2011).

## Reproduction appendix (outside page limit)

Exact commands, manifests/paths, runtime, dependency versions, test outcome, and any deviations. No test-set tuning or unreported teacher-generated “human” demonstrations.
