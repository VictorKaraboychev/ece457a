"""
Assignment 4 - Q2A (PSO only)

Standalone script for CartPole neuroevolution using Particle Swarm Optimization.
This file is intentionally independent from q2b.py.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    import gymnasium as gym
except ImportError as exc:
    raise ImportError("gymnasium is required. Install with: pip install gymnasium") from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("matplotlib is required. Install with: pip install matplotlib") from exc


DIM = 49
FITNESS_TARGET = 475.0
DEFAULT_EVAL_SEEDS = (11, 23, 37, 41, 59)


def decode_weights(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if w.size != DIM:
        raise ValueError(f"Expected {DIM} parameters, got {w.size}.")
    idx = 0
    w1 = w[idx : idx + 32].reshape(4, 8)
    idx += 32
    b1 = w[idx : idx + 8]
    idx += 8
    w2 = w[idx : idx + 8]
    idx += 8
    b2 = float(w[idx])
    return w1, b1, w2, b2


class InternalCartPoleEvaluator:
    def __init__(self, seeds: tuple[int, ...], max_steps: int = 500):
        self.seeds = tuple(int(s) for s in seeds)
        self.max_steps = int(max_steps)
        self.env = gym.make("CartPole-v1")

    def close(self) -> None:
        self.env.close()

    def evaluate(self, w: np.ndarray) -> float:
        w1, b1, w2, b2 = decode_weights(w)
        rewards = []
        for seed in self.seeds:
            obs, _ = self.env.reset(seed=seed)
            total = 0.0
            for _ in range(self.max_steps):
                h = np.tanh(obs @ w1 + b1)
                o = float(h @ w2 + b2)
                action = 1 if o >= 0.0 else 0
                obs, reward, terminated, truncated, _ = self.env.step(action)
                total += reward
                if terminated or truncated:
                    break
            rewards.append(total)
        return float(np.mean(rewards))


def try_external_evaluator(seeds: tuple[int, ...]) -> Optional[Callable[[np.ndarray], float]]:
    try:
        mod = importlib.import_module("cartpole_eval")
    except ModuleNotFoundError:
        return None

    for fn_name in ("evaluate", "evaluate_candidate", "fitness", "eval_candidate"):
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            def wrapped(w: np.ndarray, _fn=fn, _seeds=seeds) -> float:
                for call in (
                    lambda: _fn(w, seeds=_seeds),
                    lambda: _fn(w, _seeds),
                    lambda: _fn(w),
                ):
                    try:
                        return float(call())
                    except TypeError:
                        continue
                return float(_fn(w))

            return wrapped
    return None


class FitnessFunction:
    def __init__(self, seeds: tuple[int, ...]):
        self.internal = InternalCartPoleEvaluator(seeds)
        self.external = try_external_evaluator(seeds)

    def close(self) -> None:
        self.internal.close()

    def __call__(self, w: np.ndarray) -> float:
        if self.external is not None:
            return float(self.external(w))
        return float(self.internal.evaluate(w))


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


def run_pso(
    fitness_fn: FitnessFunction,
    rng: np.random.Generator,
    max_evals: int,
    pop_size: int,
    init_low: float = -1.0,
    init_high: float = 1.0,
) -> RunResult:
    x = rng.uniform(init_low, init_high, size=(pop_size, DIM))
    v = rng.uniform(-0.1, 0.1, size=(pop_size, DIM))
    pbest = x.copy()

    fx = evaluate_population(x, fitness_fn)
    pbest_f = fx.copy()
    g_idx = int(np.argmax(fx))
    gbest = x[g_idx].copy()
    gbest_f = float(fx[g_idx])

    curve = np.empty(max_evals, dtype=np.float64)
    eval_idx = pop_size
    running_best = push_curve(curve, 0, fx, -math.inf)

    vmax = 0.2
    c1 = 1.7
    c2 = 1.7
    while eval_idx + pop_size <= max_evals:
        inertia = 0.9 - 0.5 * (eval_idx / max_evals)
        r1 = rng.random(size=(pop_size, DIM))
        r2 = rng.random(size=(pop_size, DIM))
        v = inertia * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest[None, :] - x)
        v = np.clip(v, -vmax, vmax)
        x = np.clip(x + v, -1.0, 1.0)

        fx = evaluate_population(x, fitness_fn)
        running_best = push_curve(curve, eval_idx, fx, running_best)
        eval_idx += pop_size

        improved = fx > pbest_f
        pbest[improved] = x[improved]
        pbest_f[improved] = fx[improved]
        new_g_idx = int(np.argmax(pbest_f))
        if pbest_f[new_g_idx] > gbest_f:
            gbest_f = float(pbest_f[new_g_idx])
            gbest = pbest[new_g_idx].copy()

    return RunResult(
        final_best=float(np.max(curve)),
        curve=curve,
        evals_to_target=first_hit_eval(curve),
    )


def summarize_runs(runs: list[RunResult]) -> dict[str, float]:
    finals = np.asarray([r.final_best for r in runs], dtype=np.float64)
    hits = np.asarray([r.evals_to_target for r in runs if r.evals_to_target > 0], dtype=np.float64)
    return {
        "algorithm": "PSO",
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
    plt.plot(x, mean, linewidth=2, label="PSO")
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.axhline(FITNESS_TARGET, linestyle="--", linewidth=1, color="gray", label="Success threshold (475)")
    plt.xlabel("Fitness evaluations")
    plt.ylabel("Best fitness so far")
    plt.title("CartPole Neuroevolution: PSO")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def run_experiment(
    runs: int,
    max_evals: int,
    pop_size: int,
    out_dir: Path,
    init_low: float,
    init_high: float,
    base_seed: int,
) -> list[RunResult]:
    results: list[RunResult] = []
    t0 = time.time()
    for i in range(runs):
        rng = np.random.default_rng(base_seed + 1000 * i)
        fitness_fn = FitnessFunction(DEFAULT_EVAL_SEEDS)
        try:
            result = run_pso(
                fitness_fn=fitness_fn,
                rng=rng,
                max_evals=max_evals,
                pop_size=pop_size,
                init_low=init_low,
                init_high=init_high,
            )
            results.append(result)
        finally:
            fitness_fn.close()
        print(f"Run {i + 1}/{runs} completed ({time.time() - t0:.1f}s elapsed)")

    summary = summarize_runs(results)
    per_run_rows = [
        {
            "algorithm": "PSO",
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
    parser.add_argument("--init-low", type=float, default=-1.0)
    parser.add_argument("--init-high", type=float, default=1.0)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--out-dir", type=str, default="A4/results_q2a")
    parser.add_argument("--full", action="store_true", help="Use assignment settings: 20 runs, 3000 evals.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = 20 if args.full else args.runs
    max_evals = 3000 if args.full else args.max_evals
    out_dir = Path(args.out_dir)

    print(f"Starting PSO experiment: runs={runs}, max_evals={max_evals}, pop_size={args.pop_size}", flush=True)
    run_experiment(
        runs=runs,
        max_evals=max_evals,
        pop_size=args.pop_size,
        out_dir=out_dir,
        init_low=args.init_low,
        init_high=args.init_high,
        base_seed=args.base_seed,
    )
    print(f"Done. Results saved under: {out_dir}")


if __name__ == "__main__":
    main()

