"""Pairwise outcome helpers and Bradley-Terry MLE for global leaderboard ratings.

The MLE solver follows the am-ELO formulation (https://arxiv.org/abs/2505.03475):
maximise the log-likelihood of observed Bradley-Terry outcomes by minimising
the negative log-likelihood with L-BFGS-B, fixing one anchor model.
"""

from __future__ import annotations

import numpy as np
import scipy.optimize as optimize
import scipy.special
import scipy.stats as stats

ModelName = str
MetricStats = tuple[float, float, float]
MatchOutcome = tuple[ModelName, ModelName, float]


def compute_bootstrap_ci(
    data: np.ndarray, confidence_level: float = 0.95
) -> MetricStats:
    """Mean and BCa CI of the sample mean via scipy.stats.bootstrap."""
    mean_val = float(np.mean(data))
    res = stats.bootstrap(
        (data,), np.mean, confidence_level=confidence_level, method="BCa"
    )
    return (
        mean_val,
        float(res.confidence_interval.low),
        float(res.confidence_interval.high),
    )


def determine_outcome_raw(
    baseline_data: np.ndarray,
    candidate_data: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Pairwise outcome from paired bootstrap arrays via the Wilcoxon signed-rank test.

    Returns 1.0 (candidate beats baseline), 0.0 (loses), or 0.5 (tie).
    """
    try:
        _, p_value = stats.wilcoxon(candidate_data, baseline_data, zero_method="pratt")
    except ValueError:
        return 0.5

    if not np.isfinite(p_value):
        # Modern scipy returns NaN when every paired difference is zero (degenerate
        # tie); older versions raised ValueError. Treat both as ties.
        return 0.5

    if p_value >= alpha:
        return 0.5
    return 1.0 if np.mean(candidate_data) > np.mean(baseline_data) else 0.0


def determine_outcome_aggregate(
    baseline_stats: MetricStats,
    candidate_stats: MetricStats,
    alpha: float = 0.05,
) -> float:
    """Pairwise outcome when only mean and 95% CI of each model are available."""
    mean_base, ci_low_base, ci_high_base = baseline_stats
    mean_cand, ci_low_cand, ci_high_cand = candidate_stats

    z_critical = stats.norm.ppf(0.975)
    se_base = (ci_high_base - ci_low_base) / (2 * z_critical)
    se_cand = (ci_high_cand - ci_low_cand) / (2 * z_critical)
    se_diff = float(np.sqrt(se_base**2 + se_cand**2))

    if se_diff == 0.0:
        if mean_cand == mean_base:
            return 0.5
        return 1.0 if mean_cand > mean_base else 0.0

    z_stat = (mean_cand - mean_base) / se_diff
    p_value = stats.norm.sf(np.abs(z_stat)) * 2

    if p_value >= alpha:
        return 0.5
    return 1.0 if mean_cand > mean_base else 0.0


def fit_batch_mle(
    match_outcomes: list[MatchOutcome],
    model_names: list[ModelName],
    anchor_model: str,
    anchor_rating: float = 1500.0,
) -> dict[str, float]:
    """Fit Bradley-Terry ratings by minimising the negative log-likelihood.

    The anchor model's rating is fixed (gauge-fixing); only the rest are optimised.
    """
    opt_models = [m for m in model_names if m != anchor_model]
    if not opt_models:
        return {anchor_model: anchor_rating}

    scale = 400.0 / np.log(10)

    def negative_log_likelihood(ratings_array: np.ndarray) -> float:
        current = {m: r for m, r in zip(opt_models, ratings_array)}
        current[anchor_model] = anchor_rating
        nll = 0.0
        for m_a, m_b, outcome in match_outcomes:
            r_a = current[m_a]
            r_b = current[m_b]
            p_a = 1.0 / (1.0 + np.exp(-(r_a - r_b) / scale))
            p_a = np.clip(p_a, 1e-15, 1 - 1e-15)
            nll -= outcome * np.log(p_a) + (1.0 - outcome) * np.log(1.0 - p_a)
        return nll

    initial_guess = np.full(len(opt_models), anchor_rating)
    result = optimize.minimize(
        negative_log_likelihood, initial_guess, method="L-BFGS-B"
    )

    final = {m: float(r) for m, r in zip(opt_models, result.x)}
    final[anchor_model] = anchor_rating
    return final


def fit_weighted_mle(
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    outcomes_a: np.ndarray,
    weights: np.ndarray,
    model_names: list[ModelName],
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
