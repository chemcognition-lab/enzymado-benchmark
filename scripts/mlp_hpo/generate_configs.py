"""Generate MLP HPO configs from the full search space.

Default behavior: create 400 sampled experiments.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

SEARCH_SPACE = {
    "activation": ["relu", "silu", "swiglu"],
    "num_layers": [1, 2, 3],
    "hidden_dim": [512, 1024, 2048, 4096],
    "dropout_prob": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5],
    "normalize": [False, True],
    "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6],
    "weight_decay": [0.3, 0.1, 0.03, 0.01, 0.0],
}


def _lhs_sample(space_values: list[list], n: int, seed: int) -> list[tuple]:
    rng = random.Random(seed)
    mapped_idx_per_dim: list[list[int]] = []
    for values in space_values:
        perm = list(range(n))
        rng.shuffle(perm)
        idxs: list[int] = []
        m = len(values)
        for cell in perm:
            point = (cell + rng.random()) / n
            idxs.append(min(int(point * m), m - 1))
        mapped_idx_per_dim.append(idxs)

    out: list[tuple] = []
    for row in range(n):
        out.append(
            tuple(values[mapped_idx_per_dim[col][row]] for col, values in enumerate(space_values))
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sampled MLP HPO configs.")
    parser.add_argument("--n-configs", type=int, default=400)
    parser.add_argument("--sampling", choices=["random", "lhs"], default="lhs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: scripts/mlp_hpo/configs_<n>_<sampling>.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_configs <= 0:
        raise ValueError("--n-configs must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.num_epochs <= 0:
        raise ValueError("--num-epochs must be > 0")

    keys = list(SEARCH_SPACE.keys())
    values = [SEARCH_SPACE[k] for k in keys]
    all_combinations = list(itertools.product(*values))
    total = len(all_combinations)  # 17,280
    if args.n_configs > total:
        raise ValueError(f"--n-configs={args.n_configs} exceeds full space ({total})")

    rng = random.Random(args.seed)
    if args.sampling == "random":
        sampled = rng.sample(all_combinations, args.n_configs)
    else:
        sampled = []
        seen: set[tuple] = set()
        for combo in _lhs_sample(values, args.n_configs, args.seed):
            if combo in seen:
                continue
            seen.add(combo)
            sampled.append(combo)
        if len(sampled) < args.n_configs:
            remaining = [combo for combo in all_combinations if combo not in seen]
            sampled.extend(rng.sample(remaining, args.n_configs - len(sampled)))

    configs = []
    for idx, combo in enumerate(sampled):
        cfg = dict(zip(keys, combo))
        cfg["config_index"] = idx
        cfg["optimizer"] = "adamw"
        cfg["batch_size"] = args.batch_size
        cfg["epochs"] = args.num_epochs
        configs.append(cfg)

    if args.output is None:
        out_path = Path(__file__).resolve().parent / f"configs_{args.n_configs}_{args.sampling}.json"
    else:
        out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(configs, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(configs)} configs to {out_path}")


if __name__ == "__main__":
    main()
