# Assignment 4 - Problem 1: CartPole Neuroevolution (PSO vs ES)

## Setup

- Environment: `CartPole-v1` (Gymnasium)
- Controller: fixed 4-8-1 neural network (`tanh` hidden, linear output), encoded in `w in R^49`
- Fitness: average reward over 5 fixed seeds: `[11, 23, 37, 41, 59]`
- Success criterion: fitness >= 475
- Algorithms:
  - Particle Swarm Optimization (PSO)
  - Evolution Strategies (ES), `(mu + lambda)` with `mu=20, lambda=40`

## Required Main Experiment (as assigned)

- Initialization: uniform in `[-1,1]^49`
- Population size: `N=40`
- Fitness evaluations per run: `3000`
- Independent runs per algorithm: `20`
- Results directory: `A4/results_fullmain`

### Main results table

| Algorithm | Mean final fitness | Std. dev. final fitness | Success rate | Mean evals-to-475 (hits only) |
|---|---:|---:|---:|---:|
| PSO | 500.00 | 0.00 | 100.0% | 372.90 |
| ES (mu+lambda) | 500.00 | 0.00 | 100.0% | 355.85 |

Convergence plot: `A4/results_fullmain/convergence_main.png`

## Additional Evidence for Q3 and Q4

These extra analyses were run to answer sensitivity/noise questions:
- Runs per setting: `4`
- Evaluations per run: `1000`
- Results directory: `A4/results`

### Initialization sensitivity (mean final fitness)

- PSO:
  - narrow `[-0.1,0.1]`: `500.0`
  - required `[-1,1]`: `461.0`
  - wide `[-2,2]` (clipped): `355.45`
- ES:
  - narrow `[-0.1,0.1]`: `500.0`
  - required `[-1,1]`: `500.0`
  - wide `[-2,2]` (clipped): `500.0`

Source: `A4/results/init_sensitivity.csv`

### Noisy evaluation impact (clean re-evaluated final fitness)

- PSO final mean fitness:
  - noise std `0`: `459.75`
  - noise std `3`: `500.0`
  - noise std `10`: `500.0`
- ES final mean fitness:
  - noise std `0`: `500.0`
  - noise std `3`: `500.0`
  - noise std `10`: `500.0`

Convergence-speed indicator (mean evals-to-475) changed with noise:
- PSO: `240.33 -> 296.5 -> 434.0` (noise 0, 3, 10)
- ES: `437.5 -> 376.25 -> 420.25` (noise 0, 3, 10)

Source: `A4/results/noise_impact.csv`

## Answers to Assignment Questions

1. **Which algorithm converges faster?**  
   In the required main experiment, **ES converged slightly faster** by threshold-hitting speed: mean evals-to-475 = `355.85` (ES) vs `372.90` (PSO), while both reached the same final performance.

2. **Which algorithm is more stable across runs?**  
   **Tie in the main experiment**: both have standard deviation `0.00` and success rate `100%`.

3. **How sensitive is each algorithm to initialization?**  
   In the extra sensitivity study, **PSO is more sensitive** to initialization range (500 -> 461 -> 355.45 as the range broadens), while **ES remains robust** (500 across all tested ranges).

4. **How does noisy fitness evaluation affect performance?**  
   In these runs, both algorithms still reached strong clean final policies. Noise mainly affected **convergence speed** (especially PSO, which needed more evaluations to hit threshold as noise increased). This is consistent with noisy updates disturbing best-so-far guidance.

5. **Explain observed behavior based on PSO and ES structure.**
   - **PSO** shares information through global/personal best memory, which can speed search but can also be disrupted by noisy or unlucky attractors.
   - **ES** relies on mutation + selection pressure over a population, often making it robust to initialization and noisy samples.
   - CartPole has nonlinear termination boundaries; once either method reaches a strong controller, performance saturates near 500, reducing final variance.

## Deliverables Produced

- Source code: `A4/q2.py`
- Convergence plots:
  - required main run: `A4/results_fullmain/convergence_main.png`
  - extra runs: `A4/results/convergence_main.png`
- Final metrics tables:
  - required main run: `A4/results_fullmain/main_summary.csv`
  - per-run details: `A4/results_fullmain/main_runs.csv`
  - extra studies: `A4/results/init_sensitivity.csv`, `A4/results/noise_impact.csv`
- Report (this file): `A4/report_problem2.md`
