
"""
cartpole_eval.py

Black-box evaluation module for the CartPole neuroevolution assignment.

Students MUST NOT modify this file.

Usage:
    from cartpole_eval import evaluate
    fitness = evaluate(w)   # w must be a 49-dimensional numpy array
"""

import numpy as np
import gymnasium as gym

# -------------------------------------------------
# Fixed neural network configuration
# -------------------------------------------------

INPUT_DIM = 4
HIDDEN_DIM = 8
OUTPUT_DIM = 1
PARAM_DIM = 49

EPISODES_PER_EVAL = 5
MAX_STEPS = 500

# Fixed seeds for reproducibility and fairness
EVAL_SEEDS = [101, 202, 303, 404, 505]


def decode(w):
    """
    Decode 49D vector into:
        W1 (4x8)
        b1 (8,)
        W2 (8,)
        b2 (scalar)
    """
    w = np.asarray(w).reshape(-1)

    if w.size != PARAM_DIM:
        raise ValueError(f"Expected vector of length {PARAM_DIM}, got {w.size}")

    idx = 0

    W1 = w[idx:idx+32].reshape((INPUT_DIM, HIDDEN_DIM))
    idx += 32

    b1 = w[idx:idx+8]
    idx += 8

    W2 = w[idx:idx+8]
    idx += 8

    b2 = w[idx]

    return W1, b1, W2, b2


def policy(w, state):
    """Forward pass: state (4D) -> action {0,1}"""
    W1, b1, W2, b2 = decode(w)

    h = np.tanh(np.dot(state, W1) + b1)
    o = np.dot(h, W2) + b2

    return 1 if o >= 0 else 0


def evaluate(w):
    """
    Returns average reward over 5 CartPole episodes.
    Maximization objective (max = 500).
    """

    total_reward = 0.0

    for seed in EVAL_SEEDS:
        env = gym.make("CartPole-v1")
        state, _ = env.reset(seed=seed)

        episode_reward = 0.0

        for _ in range(MAX_STEPS):
            action = policy(w, state)
            state, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward

            if terminated or truncated:
                break

        env.close()
        total_reward += episode_reward

    return total_reward / EPISODES_PER_EVAL
