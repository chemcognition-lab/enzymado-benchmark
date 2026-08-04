---
name: chemcoglab
description: ChemCognition Lab project coding philosophy, architectural guidelines, code organization standards, syntactic rules, import conventions, PyTorch DL standards, and scientific code practices.
---

# ChemCognition Lab Project Coding Philosophy & Style Guide

*Last Updated: 2026-08-03*

This skill defines the coding philosophy, architectural mindset, data organization, syntactic formatting, import rules, and deep learning / scientific computing standards for all ChemCognition Lab (`chemcoglab`) repositories.

---

## 0. Philosophy & Architectural Mindset

- **The Google Foundation (Safety & Scale):** Your baseline architectural philosophy prioritizes long-term maintainability, traceability, and explicit namespace management. Favor explicit architectural design over dynamic metaprogramming, reflection, or overly clever abstractions. Reference: [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- **The Modern Adaptation (Ergonomics & Brevity):** While maintaining semantic safety, abandon legacy corporate syntactic rigidities. Embrace the open-source ecosystem's modern tooling (Ruff/Black standards) and Python 3.10+ syntax to reduce verbosity and eliminate unnecessary boilerplate.
- **Flat is Better than Nested:** Avoid deep nesting (the Arrow Anti-Pattern, max 2 levels). Use guard clauses to return or raise errors early. Extract complex logic into flat helper functions. Keep script execution logic and training loops modular. If a control flow exceeds ~100 lines, extract distinct sub-steps (like visualization plotting, evaluation loops, or metric formatting) into specialized helper functions.
- **Contextual Conciseness:** When variable intent or scope is self-evident from local context, prefer clean, concise names (`in_dim`, `out_dim`, `mu`, `std`, `n_params`, `metric_fn`, `_EPS`, `k, v`) over overly verbose or repetitive names (`in_features`, `secondary_metric_function`, `target_stats_fitted`).
- **No Defensive Programming Noise:** Do NOT write defensive checks (`if not variable: raise ValueError(...)`, `if generator is None: raise...`, `np.nan_to_num(...)`, or silent `try/except` fallbacks). Trust system contracts, Pydantic type invariants, and standard Python exceptions. If an invalid state occurs, let Python fail directly and raise native errors (`AttributeError`, `KeyError`, `IndexError`, `TypeError`) naturally.

---

## 1. Code Organization & Data Philosophy (CCDS)

Adhere to the structural philosophy of [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) to ensure reproducibility. Treat your data analysis pipeline as a Directed Acyclic Graph (DAG) where raw data is strictly immutable.

When generating multi-file architectures or referencing paths, use this standard structure:

- `data/raw/`, `data/interim/`, `data/processed/`, and `models/`.
- `path_and_constants.py`: Centralize project-specific paths and constants here using `class Paths:` and `class Constants:`. In scripts/notebooks, instantiate and use them as `_C = local_lib.Constants()` and `_P = local_lib.Paths()`.
- `types.py`: Hold complicated, project-specific type definitions here.
- `utils.py`: Hold generic, reusable utility functions here.
- `vis.py`: Hold all plotting and visualization functions here.
- **Pathlib Over Built-ins:** When using `pathlib.Path` objects, fully leverage their object-oriented API. Use `Path.open()` instead of the built-in `open(Path)` function.

---

## 2. Global Syntactic and Formatting Rules

- **Module Documentation:** Every script MUST begin with a module-level docstring containing at least a single-sentence summary. Avoid including execution examples at the top of python modules (with `paths_and_constants.py` being the sole exception).
- **Line Length:** Maximum line length is 88 characters. Use implied line continuation inside parentheses, brackets, and braces over backslashes.
- **Concise Function Naming:** Keep function names brief, action-oriented, and focused on intent rather than exhaustive mechanical descriptions. Avoid overly verbose phrasing. For example, prefer `extract_results_from_md` over `parse_markdown_to_csv_data`.
- **Standardized Labels & Predictions:** Standardize target arrays/tensors to `y_train`, `y_val`, `y_test` and predictions to `pred_train`, `pred_val`. Overwrite variable names inline when casting types (`y_true = y_true.long()`).
- **Pythonic Idioms & Comprehensions:** Favor list and dict comprehensions for data transformation, provided they are not nested more than two levels deep. You can create small helper functions (e.g., `_helper`) if it helps enable this. If a comprehension requires a line break to read comfortably, convert it into a standard `for` loop.
- **Single-Line Statements:** Keep simple assignments, list comprehensions, `dataclass` default factories (`field(default_factory=...)`), logger calls, and `raise` statements on a single unified line whenever under 88 characters.
- **Type Hinting:** Implement comprehensive, strict static typing for all function signatures and class attributes. Utilize modern Python syntax (PEP 585/604). Do not use legacy `typing` imports like `List`, `Dict`, `Union`, or `Optional`.
- **True/False Evaluations:** Use the "implicit" false for empty sequences and zeros. Always use `is None` or `is not None`.
- **Unused Arguments:** Prefix unused arguments with an underscore (e.g., `_event`). Do not use the `del` keyword for unused variables.
- **Default Parameter Resolution:** Prefer the pythonic `or` pattern (`logp = logp or compute_logp(mol)`) over explicit `if x is None:` checks unless falsy non-default values (`0`, `False`) are valid inputs.
- **Exhaustive Dispatchers:** Use Python `match` statements over `if-elif-else` chains for enum or string dispatching. Always include a wildcard `case _:` block raising `ValueError`.
- **Comments & Formatting Invariants:** Keep comments minimal. Do not write comments describing what the code is doing unless the logic is exceptionally complex. Non-obvious operations get end-of-line comments. Strictly ban "boxed" comments or heavy character separators (such as `# ====...` banners).
- **Exceptions vs. Asserts:** Raise built-in exceptions (e.g., `ValueError`) for preconditions. Do not use `assert` statements for critical application logic; reserve them strictly for testing.
- **Less Defensive Programming:** Avoid unnecessary checks, bounds, and excessively protective preconditions in code. Do not write defensive existence checks (like `if not path.exists():` or `if key not in dict:`) purely to throw custom error messages. Let Python's native exceptions (`FileNotFoundError`, `KeyError`) propagate naturally.
- **Type Checker Appeasement:** Never introduce runtime overhead (such as calling `.tolist()` or copy operations on NumPy arrays/PyTorch Tensors) solely to satisfy static type checkers (like Mypy or Pyright). If the code runs correctly, suppress warnings using `# type: ignore` instead of writing inefficient runtime code.
- **Notebook Validation:** Avoid running automated notebook execution/validation (e.g., `nbconvert --execute` or `papermill`) on build/commit, as it can take too long.

---

## 3. Import Architecture

- **Namespace Safety:** Use `import` statements for packages and modules only, not for individual types, classes, or functions. (An exception is allowed for `from pathlib import Path` to avoid overly verbose code).
- **Aliases:** Use `import y as z` only when `z` is a standard abbreviation. Strictly use standard industry aliases (e.g., `import numpy as np`, `import pandas as pd`, `import torch` — do NOT use `th` for torch).
- **Sorting:** Assume `isort` or `Ruff` will handle the exact grouping and sorting. Focus purely on using the correct syntax and aliases.
- **No Local Imports in Libraries:** Strictly ban inline or local imports within library/package modules (e.g., under `mlpbo/`). All imports must reside at the module level.
- **Direct Imports over Re-exports:** Import functions directly from their defining module rather than creating duplicate re-export layers or import namespaces.

---

## 4. Deep Learning & ML Philosophy (PyTorch Ecosystem)

When generating ML models, training loops, or data pipelines, strictly enforce the following patterns:

- **Tensor Typing (`jaxtyping`):** You MUST use [jaxtyping](https://github.com/patrick-kidger/jaxtyping) for all tensor type hints and shape documentation (e.g., `Float[Tensor, "batch channels height width"]`).
- **Reproducibility & Device:** Always include a block to set seeds across all relevant libraries (`torch`, `numpy`, `random`). Include robust device mapping: `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- **Model Architecture Parameters:** Avoid hardcoding dimensions. Pass them explicitly as arguments to the model's `__init__` constructor.
- **Evaluation Framework:** Decouple inference from metric calculation. Define evaluation functions with the signature `def evaluate(y_true, y_pred, **kwargs) -> dict:`.
  - Inputs (`y_true`, `y_pred`) should strictly be `torch.Tensor` objects or dictionaries of tensors.
  - **Regression:** Calculate $R^2$, MAE, and Kendall Tau.
  - **Classification:** Calculate AUROC, AUPRC, and F1 Score.
  - Prefer `torchmetrics`. Inject tracking metadata using dictionary union syntax (e.g., `{'split': 'val'} | evaluate(...)`).
- **Training Loop & Learning Rate Strategy:**
  - **Exploratory & Quick Baseline Loops:** Use **EarlyStopping** monitoring a single validation performance metric (e.g. validation AUROC/metric score `val_score`, **never** loss).
  - **Production Models:** Do not rely on EarlyStopping. Instead, train using a learning rate scheduler (e.g., `OneCycleLR`) over a fixed, learned number of total epochs to guarantee full convergence.
  - **Single Evaluation Pass:** Compute both loss and evaluation metrics in a single pass over dataloaders to avoid redundant forward passes.
- **Training Loop Metrics Logging:** Use `tqdm` for progress bars. Model training histories should include learning rate (`lr`), and represent records with one row per split type (e.g., storing both `loss` and `auroc` for train and val splits). Accumulate metrics into a list of dictionaries, update the progress bar by passing the latest history record directly (e.g., `pbar.set_postfix(history[-1])`) without verbose dictionary construction, and convert the final accumulated list to a `pandas.DataFrame` at the end for logging/visualization.
- **Training Execution:** Avoid executing training scripts or training loops unless explicitly requested by the user, as they can take a long time.
- **Inference Outputs:** Utility functions that run inference (e.g. `make_inferences` or model forward passes) should accept the loss function as a parameter, returning `(y_probs, y_true, average_loss)` with sigmoid-activated probabilities rather than raw logits.
- **Architecture Validation:** Always instantiate the model and run `torchinfo.summary()` with dummy input tensors to explicitly validate the parameter count before training.
- **Execution Profiling & Logging:** Always capture and output execution timing (using `time.time()`) for major time-intensive operations (such as model training, data featurization, database queries, or hyperparameter sweeps) to assist in bottleneck detection.

---

## 5. Scientific, Environment & Execution Specifics

- **Environment Execution:** Always check `environment.yml` / `environment.yaml` first to identify the project-specific environment name (e.g. `name: mirra` or `name: pom`). Run execution commands (such as `python`, `ruff`, `pytest`, and `pre-commit`) wrapped in `conda run -n <env_name>` (e.g. `conda run -n pom`) to ensure commands execute within the project environment.
- **Package Installation Preference:** When installing or adding new dependencies, prefer `mamba install -n <env_name> <package>` over `pip` whenever possible to preserve solver consistency and binary compatibility.
- **Configuration:** Strictly avoid global variables. Rely on dataclass configuration libraries like `pyrallis` or `draccus` (using `parse(config_class=Config)` or `@wrap`) or `pydantic-settings` to manage CLI arguments and hyperparameters.
- **Draccus & CLI Flag Conventions:**
  - Define boolean flags in dataclasses with `metadata={"action": "store_true"}` to allow command-line flag toggles (e.g. `--test True`).
  - In Python 3.14+, type unions like `Path | None` are not callable in `argparse`; define path defaults as direct class-types (e.g. `results_dir: Path = _P.results_dir / "mlp"`).
  - Provide both standard (`Example: python scripts/...`) and test (`Test run: python scripts/... --test True`) invocation commands in script docstrings.
- **Python 3.14 Dataclass Parser Compatibility:** In CLI entrypoint scripts using `pyrallis` or `draccus` under Python 3.14+, omit `from __future__ import annotations` (or import concrete type annotations from `typing`) so parser field inspection works directly without stringification errors.
- **Compressed Molecular Caches:** Large molecular coordinate structures (like SDF files) must always be stored and handled as gzipped files (`.sdf.gz`) to optimize disk space and I/O. Use streaming suppliers (like RDKit's `ForwardSDMolSupplier`) directly on `gzip.open(path, "rb")` without intermediate uncompressed disk dumps.
- **DataFrame Defragmentation:** When reading CSV files or adding/updating multiple columns on pandas DataFrames, call `.copy()` immediately to defragment the internal block manager and prevent `PerformanceWarning` warnings.
- **Naming Conventions:** For mathematically-heavy code, short variable names that would otherwise violate the style guide (e.g., `x`, `y`, `w`) are permitted when they match established notation in a reference paper or algorithm.
- **Parameter Naming:** Standardize naming hyperparameter dictionary/object instances as `hp` (instead of `hparams`) for simplicity and brevity.
- **No Automatic Production Model Promotion:** Training scripts should write outputs to experiment directories (`results/<task>/`). Automatically writing or copying checkpoints to `models/` is prohibited, as promoting weights to production models is a manual operational step.

---

## 6. Tracking Style Guide Deviations & General Observations

- **Local Style Tracking:** Any project-specific style adjustments, deviations, or general observations that differ from this guide must be logged in `.agents/AGENTS.md` (and excluded via `.gitignore`). The assistant must proactively check and append any newly discovered style preferences (e.g., parameter renaming preferences, comments preferences, early stopping configurations, etc.) to this file to preserve continuity.
