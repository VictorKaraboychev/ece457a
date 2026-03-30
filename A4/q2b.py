"""
Assignment 4 - Q2B (ES only)

Standalone script for CartPole neuroevolution using Evolution Strategies (mu + lambda).
This file is intentionally independent from q2a.py.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("matplotlib is required. Install with: pip install matplotlib") from exc


DIM = 49
FITNESS_TARGET = 475.0

def load_black_box_evaluator() -> Callable[[np.ndarray], float]:
    """Load the assignment-provided evaluator from A4/cartpole_eval.py."""
    eval_path = Path(__file__).resolve().parent / "cartpole_eval.py"
    if not eval_path.exists():
        raise FileNotFoundError(f"Required evaluator not found: {eval_path}")

    spec = importlib.util.spec_from_file_location("cartpole_eval", eval_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {eval_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    evaluate_fn = getattr(module, "evaluate", None)
    if not callable(evaluate_fn):
        raise AttributeError("cartpole_eval.py must define a callable `evaluate(w)`.")
    return evaluate_fn


class FitnessFunction:
    def __init__(self):
        self.evaluate_fn = load_black_box_evaluator()

    def __call__(self, w: np.ndarray) -> float:
        return float(self.evaluate_fn(w))


def evaluate_population(pop: np.ndarray, fitness_fn: FitnessFunction) -> np.ndarray:
    return np.asarray([fitness_fn(ind) for ind in pop], dtype=np.float64)


def push_curve(curve: np.ndarray, start: int, values: np.ndarray, best_so_far: float) -> float:
    best = best_so_far
    for i, value in enumerate(values):
        best = max(best, float(value))
        curve[start + i] = best
    return best


def first_hit_eval(curve: np.ndarray, threshold: float = FITNESS_TARGET) -> int:
    mask = curve >= threshold
    return int(np.argmax(mask) + 1) if np.any(mask) else -1


@dataclass
class RunResult:
    final_best: float
    curve: np.ndarray
    evals_to_target: int


def run_es_mu_plus_lambda(
    fitness_fn: FitnessFunction,
    rng: np.random.Generator,
    max_evals: int,
    pop_size: int,
    mu: int,
    init_low: float = -1.0,
    init_high: float = 1.0,
) -> RunResult:
    lam = pop_size
    if mu >= lam:
        raise ValueError("mu must be < lambda.")

    initial_pop = rng.uniform(init_low, init_high, size=(lam, DIM))
    initial_fit = evaluate_population(initial_pop, fitness_fn)
    order = np.argsort(-initial_fit)
    parents = initial_pop[order[:mu]]
    parent_fit = initial_fit[order[:mu]]

    curve = np.empty(max_evals, dtype=np.float64)
    eval_idx = lam
    running_best = push_curve(curve, 0, initial_fit, -math.inf)
    sigma = 0.12
    tau = 1.0 / math.sqrt(2.0 * DIM)

    while eval_idx + lam <= max_evals:
        idx1 = rng.integers(0, mu, size=lam)
        idx2 = rng.integers(0, mu, size=lam)
        offspring = 0.5 * (parents[idx1] + parents[idx2])

        sigma = float(np.clip(sigma * math.exp(tau * rng.normal()), 0.01, 0.4))
        offspring = np.clip(offspring + rng.normal(0.0, sigma, size=(lam, DIM)), -1.0, 1.0)
        off_fit = evaluate_population(offspring, fitness_fn)
        running_best = push_curve(curve, eval_idx, off_fit, running_best)
        eval_idx += lam

        combined = np.vstack([parents, offspring])
        combined_fit = np.concatenate([parent_fit, off_fit])
        order = np.argsort(-combined_fit)
        parents = combined[order[:mu]]
        parent_fit = combined_fit[order[:mu]]

    return RunResult(
        final_best=float(np.max(curve)),
        curve=curve,
        evals_to_target=first_hit_eval(curve),
    )


def summarize_runs(runs: list[RunResult]) -> dict[str, float]:
    finals = np.asarray([r.final_best for r in runs], dtype=np.float64)
    hits = np.asarray([r.evals_to_target for r in runs if r.evals_to_target > 0], dtype=np.float64)
    return {
        "algorithm": "ES_mu+lambda",
        "mean_final_fitness": float(np.mean(finals)),
        "std_final_fitness": float(np.std(finals, ddof=1)) if finals.size > 1 else 0.0,
        "success_rate": float(np.mean(finals >= FITNESS_TARGET)),
        "mean_evals_to_target_if_hit": float(np.mean(hits)) if hits.size > 0 else float("nan"),
    }


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_convergence(path: Path, runs: list[RunResult]) -> None:
    curves = np.vstack([r.curve for r in runs])
    x = np.arange(1, curves.shape[1] + 1)
    mean = np.mean(curves, axis=0)
    std = np.std(curves, axis=0)

    plt.figure(figsize=(9, 5))
    plt.plot(x, mean, linewidth=2, label="ES (mu+lambda)")
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.axhline(FITNESS_TARGET, linestyle="--", linewidth=1, color="gray", label="Success threshold (475)")
    plt.xlabel("Fitness evaluations")
    plt.ylabel("Best fitness so far")
    plt.title("CartPole Neuroevolution: ES (mu+lambda)")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def run_experiment(
    runs: int,
    max_evals: int,
    pop_size: int,
    mu: int,
    out_dir: Path,
    init_low: float,
    init_high: float,
    base_seed: int,
) -> list[RunResult]:
    results: list[RunResult] = []
    t0 = time.time()
    for i in range(runs):
        rng = np.random.default_rng(base_seed + 1000 * i)
        fitness_fn = FitnessFunction()
        result = run_es_mu_plus_lambda(
            fitness_fn=fitness_fn,
            rng=rng,
            max_evals=max_evals,
            pop_size=pop_size,
            mu=mu,
            init_low=init_low,
            init_high=init_high,
        )
        results.append(result)
        print(f"Run {i + 1}/{runs} completed ({time.time() - t0:.1f}s elapsed)")

    summary = summarize_runs(results)
    per_run_rows = [
        {
            "algorithm": "ES_mu+lambda",
            "run": i + 1,
            "final_best_fitness": r.final_best,
            "evals_to_target": r.evals_to_target,
        }
        for i, r in enumerate(results)
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "main_summary.csv", [summary])
    write_csv(out_dir / "main_runs.csv", per_run_rows)
    plot_convergence(out_dir / "convergence_main.png", results)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--max-evals", type=int, default=1200)
    parser.add_argument("--pop-size", type=int, default=40)
    parser.add_argument("--mu", type=int, default=20)
    parser.add_argument("--init-low", type=float, default=-1.0)
    parser.add_argument("--init-high", type=float, default=1.0)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--out-dir", type=str, default="A4/results_q2b")
    parser.add_argument("--full", action="store_true", help="Use assignment settings: 20 runs, 3000 evals.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = 20 if args.full else args.runs
    max_evals = 3000 if args.full else args.max_evals
    out_dir = Path(args.out_dir)

    print(
        f"Starting ES experiment: runs={runs}, max_evals={max_evals}, pop_size={args.pop_size}, mu={args.mu}",
        flush=True,
    )
    run_experiment(
        runs=runs,
        max_evals=max_evals,
        pop_size=args.pop_size,
        mu=args.mu,
        out_dir=out_dir,
        init_low=args.init_low,
        init_high=args.init_high,
        base_seed=args.base_seed,
    )
    print(f"Done. Results saved under: {out_dir}")


if __name__ == "__main__":
    main()

