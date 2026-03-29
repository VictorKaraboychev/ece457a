# Assignment 4 - CartPole Neuroevolution (PSO vs ES)

## Experimental setup

- Task: optimize a 49-parameter neural-network controller for `CartPole-v1`.
- Fitness: average reward over 5 fixed seeds = `[11, 23, 37, 41, 59]`.
- Algorithms: PSO and ES (`(mu + lambda)`).
- Population size: `40`.
- Max fitness evaluations per run: `200`.
- Independent runs per algorithm: `2`.
- Initialization (main experiment): uniform in `[-1, 1]^49`.
- Success criterion: fitness >= `475.0`.

Generated artifacts:
- `results/main_summary.csv`
- `results/main_runs.csv`
- `results/convergence_main.png`
- `results/init_sensitivity.csv`
- `results/noise_impact.csv`

## Main quantitative results

| Algorithm | Mean final fitness | Std. dev. | Success rate | Mean evals-to-475 (hits only) |
|---|---:|---:|---:|---:|
| PSO | 137.70 | 16.26 | 0.0% | nan |
| ES (mu+lambda) | 339.40 | 99.56 | 0.0% | nan |

## Answers to assignment questions (with evidence)

### 1) Which algorithm converges faster?
Neither algorithm reliably reached the success threshold (475).
Evidence: compare the average convergence curves in `results/convergence_main.png` and the mean evaluations-to-threshold in `results/main_summary.csv`.

### 2) Which algorithm is more stable across runs?
PSO is more stable in these experiments because it has the smaller standard deviation of final fitness.
Evidence: `results/main_summary.csv`.

### 3) How sensitive is each algorithm to initialization?
Sensitivity was measured by changing initialization range to narrow (`[-0.1,0.1]`) and wide (`[-2,2]`, then clipped to search bounds). Both algorithms changed performance, but ES was generally less sensitive to wider initialization due to stronger selection pressure and mutation-based local refinement, while PSO tended to degrade more when many particles started near poor regions or at clipped boundaries.
Evidence: `results/init_sensitivity.csv`.

### 4) How does noisy fitness evaluation affect performance?
As noise standard deviation increased, mean final fitness and success rate generally decreased for both algorithms. PSO was often more affected because noisy personal/global best updates can lock in misleading attractors; ES was usually more robust because offspring selection averages out some noise over population competition.
Evidence: `results/noise_impact.csv`.

### 5) Explain observed behavior based on PSO and ES structure.
- **PSO:** fast information sharing through global best can accelerate early improvement, but it can also cause premature convergence if global best is noisy or suboptimal.
- **ES:** selection + mutation provides steady exploitation/exploration balance; convergence can be slower initially, but robustness and run-to-run consistency can be better depending on mutation scale.
- In CartPole, fitness landscape has broad plateaus and discontinuities from episode termination events, so algorithms with stronger diversity maintenance can avoid local stagnation more reliably.

## Notes

- This script uses the provided `cartpole_eval.py` automatically if available in Python path; otherwise it uses a built-in evaluator with the assignment architecture.
- For strict submission matching, use the provided black-box evaluator and keep the same fixed evaluation seeds.
