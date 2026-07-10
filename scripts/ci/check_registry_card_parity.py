#!/usr/bin/env python3
"""Fail if TASK_REGISTRY and benchmark_card.json disagree on which tasks exist.

TASK_REGISTRY (src/data_loading/data_registry.py) is the single runtime source
of truth for task configuration. benchmark_card.json is documentation only —
it must list the same task names, or a contributor adding/renaming a task in
one place and forgetting the other goes unnoticed until something downstream
breaks. This check only compares task-name sets; it does not compare per-task
fields (paths, schema, etc. are the card's own business, not TASK_REGISTRY's).

Usage:
    python scripts/ci/check_registry_card_parity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_loading.data_registry import TASK_REGISTRY  # noqa: E402


def main() -> int:
    card = json.loads((REPO_ROOT / "benchmark_card.json").read_text())

    registry_properties = {
        name for name, cfg in TASK_REGISTRY.items() if cfg.task_scope == "properties"
    }
    registry_real_case = {
        name for name, cfg in TASK_REGISTRY.items() if cfg.task_scope == "real_case"
    }
    card_properties = set(card.get("properties", {}))
    card_real_case = set(card.get("real_case", {}))

    problems = []
    if registry_properties != card_properties:
        problems.append(
            "properties mismatch: "
            f"in TASK_REGISTRY only={sorted(registry_properties - card_properties)}, "
            f"in card only={sorted(card_properties - registry_properties)}"
        )
    if registry_real_case != card_real_case:
        problems.append(
            "real_case mismatch: "
            f"in TASK_REGISTRY only={sorted(registry_real_case - card_real_case)}, "
            f"in card only={sorted(card_real_case - registry_real_case)}"
        )

    if problems:
        for p in problems:
            print(f"FAIL: {p}")
        return 1

    print(f"OK: {len(registry_properties)} properties + {len(registry_real_case)} real_case tasks match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
