#!/usr/bin/env python
"""Plot the top and bottom baselines from the combined real-case rating."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

_CENTER_RATING = 1500.0
_RATING_SCALE = 400.0 / math.log(10.0)
_TOP_COLOR = "#B9E3D2"
_BOTTOM_COLOR = "#EFCBE7"
_MODEL_LABELS = {
    "ankh3_large": "Ankh3 Large",
    "ankh3_xl": "Ankh3 XL",
    "ankh_base": "Ankh Base",
    "ankh_large": "Ankh Large",
    "esm2_3B": "ESM2 3B",
    "esm2_650M": "ESM2 650M",
    "esm3_open": "ESM3 Open",
    "esmc_300m": "ESMC 300M",
    "esmc_600m": "ESMC 600M",
    "profluent_e1_300m": "Profluent E1 300M",
    "profluent_e1_600m": "Profluent E1 600M",
    "prost5": "ProstT5",
    "prot5_xl_half_uniref50": "ProtT5 XL Half UniRef50",
    "prot5_xl_uniref50": "ProtT5 XL UniRef50",
}
_FAMILY_LABELS = {
    "linear": "Linear",
    "mlp": "MLP",
    "xgboost": "XGBoost",
}
_POOL_LABELS = {
    "attention": "Attention*",
    "no_attention": "No attention",
}


def _matches(report: dict) -> list[tuple[str, str, float]]:
    matches = []
    for task_report in report["per_task_report"].values():
        for pair in task_report["pairwise_outcomes"].values():
            matches.append(
                (
                    pair["baseline_a"],
                    pair["baseline_b"],
                    float(pair["weighted_outcome_a"]),
                )
            )
    return matches


def _centered_ratings_and_se(
    report: dict,
) -> tuple[dict[str, float], dict[str, float]]:
    ratings = {
        baseline: float(rating)
        for baseline, rating in report["ratings"].items()
    }
    baselines = list(ratings)
    anchor = report["anchor_baseline"]
    optimized = [baseline for baseline in baselines if baseline != anchor]
    opt_index = {
        baseline: index for index, baseline in enumerate(optimized)
    }
    hessian = np.zeros((len(optimized), len(optimized)), dtype=np.float64)

    for baseline_a, baseline_b, _ in _matches(report):
        delta = (ratings[baseline_a] - ratings[baseline_b]) / _RATING_SCALE
        probability_a = 1.0 / (1.0 + math.exp(-delta))
        curvature = probability_a * (1.0 - probability_a) / _RATING_SCALE**2
        if baseline_a != anchor:
            index_a = opt_index[baseline_a]
            hessian[index_a, index_a] += curvature
        if baseline_b != anchor:
            index_b = opt_index[baseline_b]
            hessian[index_b, index_b] += curvature
        if baseline_a != anchor and baseline_b != anchor:
            hessian[index_a, index_b] -= curvature
            hessian[index_b, index_a] -= curvature

    optimized_covariance = np.linalg.inv(hessian)
    baseline_index = {
        baseline: index for index, baseline in enumerate(baselines)
    }
    covariance = np.zeros((len(baselines), len(baselines)), dtype=np.float64)
    for baseline_a in optimized:
        for baseline_b in optimized:
            covariance[
                baseline_index[baseline_a],
                baseline_index[baseline_b],
            ] = optimized_covariance[
                opt_index[baseline_a],
                opt_index[baseline_b],
            ]

    centering = np.eye(len(baselines)) - np.ones(
        (len(baselines), len(baselines))
    ) / len(baselines)
    centered_covariance = centering @ covariance @ centering.T
    mean_rating = float(np.mean(list(ratings.values())))
    centered_ratings = {
        baseline: rating - mean_rating + _CENTER_RATING
        for baseline, rating in ratings.items()
    }
    standard_errors = {
        baseline: math.sqrt(max(centered_covariance[index, index], 0.0))
        for baseline, index in baseline_index.items()
    }
    return centered_ratings, standard_errors


def _extremes(ratings: dict[str, float], top_n: int) -> list[str]:
    ranked = sorted(ratings, key=ratings.get, reverse=True)
    return ranked[:top_n] + ranked[-top_n:]


def _baseline_label(report: dict, baseline: str) -> str:
    metadata = report["baseline_metadata"][baseline]
    return " / ".join(
        (
            _MODEL_LABELS[metadata["embedding_model"]],
            _FAMILY_LABELS[metadata["model_family"]],
            _POOL_LABELS[metadata["pool_type"]],
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rating",
        type=Path,
        default=Path("results/ratings/real_case__random__combined.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/real_case_bradley_terry_ratings.png"),
    )
    parser.add_argument("--top-n", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = json.loads(args.rating.read_text())
    ratings, standard_errors = _centered_ratings_and_se(report)
    baselines = _extremes(ratings, args.top_n)
    values = np.asarray([ratings[baseline] for baseline in baselines])
    errors = np.asarray([standard_errors[baseline] for baseline in baselines])
    positions = np.arange(len(baselines))
    colors = [_TOP_COLOR] * args.top_n + [_BOTTOM_COLOR] * args.top_n

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    )
    figure, ax = plt.subplots(figsize=(10.8, 6.2))
    ax.barh(
        positions,
        values - _CENTER_RATING,
        left=_CENTER_RATING,
        color=colors,
        edgecolor="none",
        height=0.76,
    )
    ax.errorbar(
        values,
        positions,
        xerr=errors,
        fmt="none",
        ecolor="#555555",
        elinewidth=1.0,
        capsize=2.5,
        capthick=1.0,
    )
    ax.axvline(
        _CENTER_RATING,
        color="#444444",
        linestyle="--",
        linewidth=1.0,
    )
    ax.axhline(
        args.top_n - 0.5,
        color="#BBBBBB",
        linestyle=":",
        linewidth=1.0,
    )
    ax.set_yticks(
        positions,
        [_baseline_label(report, baseline) for baseline in baselines],
    )
    ax.invert_yaxis()
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Bradley–Terry rating (±1 SE)")

    x_min = math.floor(float(np.min(values - errors) - 35.0) / 50.0) * 50.0
    x_max = math.ceil(float(np.max(values + errors) + 35.0) / 50.0) * 50.0
    ax.set_xlim(x_min, x_max)
    ax.set_title(
        "Combined (Kendall τ×2, MAE×1, RMSE×1, R²×1)",
        pad=12,
    )
    figure.suptitle(
        "Bradley–Terry Ratings — real_case / random "
        f"(n_baselines={report['n_baselines']}, "
        f"top {args.top_n} / bottom {args.top_n})",
        fontsize=14,
        y=0.98,
    )
    figure.text(
        0.5,
        0.02,
        f"All baselines compete directly on {report['n_tasks_used']} datasets; "
        "ratings are mean-centered at 1500. "
        "* Attention results are provisional pending the stabilization rerun.",
        ha="center",
        color="#555555",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.34, right=0.97, top=0.84, bottom=0.14)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(
        args.output.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    print(args.output)
    print(args.output.with_suffix(".svg"))


if __name__ == "__main__":
    main()
