"""Weighted Bradley-Terry MLE for global leaderboard ratings.

The MLE solver follows the am-ELO formulation (https://arxiv.org/abs/2505.03475):
maximise the log-likelihood of observed Bradley-Terry outcomes by minimising
the negative log-likelihood with L-BFGS-B, fixing one anchor model.
"""

from __future__ import annotations

import numpy as np
import scipy.optimize as optimize
import scipy.special


def fit_weighted_mle(
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    outcomes_a: np.ndarray,
    weights: np.ndarray,
    model_names: list[str],
    center_rating: float = 1500.0,
    l2_penalty: float = 1e-6,
) -> dict[str, float]:
    """Fit centered Bradley-Terry ratings from weighted fractional outcomes."""
    n_models = len(model_names)
    n_free = n_models - 1

    def objective(free_abilities: np.ndarray) -> tuple[float, np.ndarray]:
        abilities = np.append(free_abilities, 0.0)
        logits = abilities[pair_a] - abilities[pair_b]
        losses = np.logaddexp(0.0, logits) - outcomes_a * logits
        value = float(np.sum(weights * losses))
        value += 0.5 * l2_penalty * float(free_abilities @ free_abilities)

        residuals = weights * (scipy.special.expit(logits) - outcomes_a)
        gradient = np.zeros(n_models, dtype=np.float64)
        np.add.at(gradient, pair_a, residuals)
        np.add.at(gradient, pair_b, -residuals)
        gradient[:n_free] += l2_penalty * free_abilities
        return value, gradient[:n_free]

    result = optimize.minimize(
        objective,
        np.zeros(n_free, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
    )
    if not result.success:
        raise RuntimeError(result.message)

    abilities = np.append(result.x, 0.0)
    ratings = abilities * (400.0 / np.log(10.0))
    ratings += center_rating - float(ratings.mean())
    return dict(zip(model_names, ratings, strict=True))
