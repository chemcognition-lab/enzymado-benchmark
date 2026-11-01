#!/usr/bin/env python
"""Distinct head settings of the published runs, to check the head-settings table.

Groups every hparams.json by scope, pooling, head and task group (scalar, EC
number, residue) and prints each distinct parameter set with its run count and
checkpoint-selection metric. Seeds, devices and thread counts are left out.

Example:
    python scripts/paper/head_settings.py
"""

from __future__ import annotations

import collections
import json

from enzymado import paths_and_constants

_P = paths_and_constants.Paths
GROUP = {"ec_number": "EC", "active_sites": "residue", "binding_site": "residue"}
IGNORED = {"random_state", "seed", "n_jobs", "device", "nthread"}


def main() -> None:
    root = _P.results_dir / "predictions"
    settings = collections.defaultdict(collections.Counter)
    for path in root.glob("*/*/*/*/*/*/hparams.json"):
        scope, pool, _split, head, task, _emb = path.parent.relative_to(root).parts
        hp = json.loads(path.read_text())
        params = {k: v for k, v in hp["params"].items() if k not in IGNORED}
        selection = hp.get("selection_protocol", {}).get("metric")
        key = (scope, pool, head, GROUP.get(task, "scalar"))
        settings[key][f"{json.dumps(params, sort_keys=True)} | select={selection}"] += 1
    for key in sorted(settings):
        print(key)
        for setting, count in settings[key].items():
            print(f"   {count:4d} x {setting}")


if __name__ == "__main__":
    main()
