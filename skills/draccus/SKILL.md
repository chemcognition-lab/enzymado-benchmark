---
name: draccus
description: How to use the draccus library (a Pyrallis fork) for dataclass-driven configuration — parsing from CLI/YAML, the @draccus.wrap() decorator, ChoiceRegistry subtyping, YAML !include directives, and dump/load serialization. Use whenever defining, parsing, or serializing a project config dataclass.
version: 1.0.0
---

# Draccus Configuration Library Reference

Draccus links a `dataclass` directly to CLI parsing and YAML config files. It is a fork of
[Pyrallis](https://github.com/eladrich/pyrallis) that adds config subtyping, `!include` directives,
and better support for containers of configs. For project-specific conventions (flag metadata,
Python 3.14 compatibility), see the `chemcoglab` skill's "Draccus & CLI Flag Conventions" section —
this skill covers the library API itself.

---

## 1. Basic Parsing

Define a config `dataclass` and parse it with `draccus.parse()` or the `@draccus.wrap()` decorator:

```python
from dataclasses import dataclass
import draccus


@dataclass
class TrainConfig:
    """Training Config for Machine Learning"""
    workers: int = 8
    exp_name: str = "default_exp"


@draccus.wrap()
def main(cfg: TrainConfig):
    print(f"Training {cfg.exp_name} with {cfg.workers} workers...")


if __name__ == "__main__":
    main()
```

Equivalent explicit form: `cfg = draccus.parse(config_class=TrainConfig)`.

Values can come from the CLI, a YAML file via `--config_path=some_config.yaml`, or both — CLI
arguments override the file. Nested dataclass fields are addressed with dotted CLI paths
(e.g. `--log.exp_name=...`, `--compute.eval_workers=2`).

`--help` is generated automatically from dataclass field comments/docstrings.

---

## 2. Config Subtyping via `ChoiceRegistry`

Use `ChoiceRegistry` for "pick one of several config types at runtime" (e.g. model/optimizer choice).
The `type` field in YAML/CLI selects the registered subclass:

```python
from dataclasses import dataclass
import draccus


@dataclass
class ModelConfig(draccus.ChoiceRegistry):
    pass


@ModelConfig.register_subclass("gpt")
@dataclass
class GPTConfig(ModelConfig):
    num_layers: int = 12
    num_heads: int = 12
    hidden_size: int = 768


@ModelConfig.register_subclass("bert")
@dataclass
class BERTConfig(ModelConfig):
    num_layers: int = 12
    num_heads: int = 12
    hidden_size: int = 768
    dropout: float = 0.1


@dataclass
class TrainConfig:
    workers: int = 8
    exp_name: str = "default_exp"
    model: ModelConfig = GPTConfig()
```

```yaml
exp_name: my_yaml_exp
workers: 42
model:
  type: bert
  num_layers: 24
  num_heads: 24
  hidden_size: 1024
  dropout: 0.2
```

---

## 3. Including Config Files

Split configs across files with the `!include` YAML tag:

```yaml
# train_config.yaml
exp_name: my_yaml_exp
workers: 42
model: !include model_config.yaml
```

The `include` keyword also works inline on the CLI for nested fields:
`--model="include model_config.yaml"`.

To merge keys at the top level, combine PyYAML's `<<` merge key with `!include`:

```yaml
<<: !include base_config.yaml
exp_name: my_yaml_exp
```

---

## 4. Nested Configs, `__post_init__`, and `@property`

Standard dataclass features work as expected — use `__post_init__` for derived defaults and
`@property` for computed fields:

```python
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ComputeConfig:
    workers: int = field(default=8)
    eval_workers: int | None = field(default=None)

    def __post_init__(self):
        self.eval_workers = self.eval_workers or self.workers


@dataclass
class LogConfig:
    exp_name: str = field(default="default_exp")
    exp_root: Path = field(default=Path("/share/experiments"))

    @property
    def exp_dir(self) -> Path:
        return self.exp_root / self.exp_name


@dataclass
class TrainConfig:
    log: LogConfig = field(default_factory=LogConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
```

---

## 5. Serialization: `draccus.dump` / `draccus.load`

```python
draccus.dump(cfg, open("run_config.yaml", "w"))

cfg = draccus.parse(config_class=TrainConfig, config_path="/share/configs/config.yaml")
# or: cfg = draccus.load(TrainConfig, "/share/configs/config.yaml")
```

CLI arguments always take priority over a loaded config file's values.

Register custom encode/decode for extra types:

```python
draccus.decode.register(np.ndarray, np.asarray)
draccus.encode.register(np.ndarray, lambda x: x.tolist())
```

---

## 6. Mutable Field Defaults

Plain `field(default=[...])` is disallowed for mutable defaults; the normal fix is
`field(default_factory=...)`, but for a non-empty default use `draccus.field` with `is_mutable=True`
instead of writing a factory lambda:

```python
from draccus import field

worker_inds: list[int] = field(default=[1, 2, 3], is_mutable=True)
```

---

## 7. Other Config Formats

Draccus/Pyrallis defaults to YAML but also supports `json` and `toml` via
`draccus.set_config_type("json")` or `with draccus.config_type("json"):`.
