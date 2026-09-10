"""Build isolated model configurations for individual benchmark runs."""

import collections.abc
import typing


def apply_random_seed(
    model_info: dict[str, object],
    random_seed: int,
) -> dict[str, object]:
    """Copy one registry entry and configure its loader and estimator seeds."""
    params = dict(typing.cast(dict[str, object], model_info["params"]))
    seeded_info = {
        **model_info,
        "params": params,
        "loader_seed": int(random_seed),
    }
    wrapper_name = getattr(model_info["wrap"], "__name__", "")
    if "random_state" in params or wrapper_name.startswith("XGB"):
        params["random_state"] = int(random_seed)
    return seeded_info


def apply_attention_overrides(
    model_info: dict[str, object],
    overrides: collections.abc.Mapping[str, object] | None,
) -> dict[str, object]:
    """Apply opt-in attention controls without mutating the registry entry."""
    if not overrides:
        return model_info
    params = typing.cast(dict[str, object], model_info["params"])
    return {
        **model_info,
        "params": {**params, **overrides},
    }
