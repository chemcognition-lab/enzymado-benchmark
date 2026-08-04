---
name: code-reviewer
description: Strict code review, design spec verification, maintainability audit, coding philosophy compliance, and automated testing workflow for Python repositories. Trigger whenever the user asks for code review, audit, maintainability check, or cleanup.
version: 2.1.0
---

# Code Reviewer Skill & Maintainability Audit Protocol

Act as an expert Python code reviewer and research software engineer. Perform a rigorous, systematic review of the codebase or working directory changes against our **ChemCognition Lab Coding Philosophy (`chemcoglab` skill)** and **Mamba Environment Execution Protocol (`mamba-environment` skill)**.

---

## 1. Core General Coding Standards (`chemcoglab` Reference)

Always verify that modified code adheres strictly to the standards defined in the `chemcoglab` skill ([SKILL.md](file://.agents/skills/chemcoglab/SKILL.md)) and `.agents/AGENTS.md`:

### A. Architectural Mindset & Structure
- **Google Foundation & Modern Adaptation**: Prioritize long-term maintainability, traceability, and explicit namespace management over dynamic metaprogramming or overly clever abstractions. Embrace modern Python 3.10+ syntax and Ruff standards.
- **Flat is Better than Nested**: Ban deep nesting (the Arrow Anti-Pattern, max depth of 2 levels) using top-of-function guard clauses and early returns.
- **Function & File Invariants**: Functions must be brief and action-oriented (< 100 lines of code). Break multi-step routines into flat helper functions. Line length must strictly be $\le 88$ characters.
- **Pathlib Over Built-ins**: Fully leverage `pathlib.Path` objects and the object-oriented `Path.open()` method instead of `open(Path)`.

### B. Defensive Programming & Error Handling
- **Zero Defensive Noise**: Strictly BAN unnecessary defensive checks (such as `if not var: raise...`, `if generator is None: raise...`, `np.nan_to_num(...)` fallbacks). Trust system contracts, dataclass type invariants, and standard Python exceptions.
- **Native Exceptions**: Let Python's native exceptions (`FileNotFoundError`, `KeyError`, `IndexError`, `TypeError`, `AttributeError`) propagate naturally when invalid states or missing keys occur.
- **Type Checker Appeasement**: Never introduce runtime overhead (e.g. calling `.tolist()` or copy operations on NumPy arrays/Tensors) solely to satisfy static type checkers (Mypy, Pyright). Use `# type: ignore` if runtime logic is sound.

### C. Import Architecture & Formatting
- **Module-Level Imports Only**: All imports must reside at the very top of module files. Strictly BAN inline, local, or function-nested `import` statements within library/package modules.
- **Namespace Safety & Industry Aliases**: Use standard aliases only (`import numpy as np`, `import pandas as pd`, `import torch`). Avoid legacy `typing` imports (`List`, `Dict`, `Union`); use PEP 585/604 hints (`list`, `dict`, `int | float`).
- **Direct Imports over Re-exports**: Import functions directly from defining modules rather than creating duplicate re-export layers.
- **Comments & Formatting**: Keep comments minimal. Strictly BAN boxed comments, divider banners (`# ===...`), or mechanical inline comments that restate obvious code logic.

### D. Deep Learning & ML Ecosystem (PyTorch)
- **Jaxtyping**: Require `jaxtyping` annotations for all tensor shapes (e.g., `Float[Tensor, "batch channels height width"]`).
- **Model Architecture Validation**: Pass model dimensions explicitly in `__init__`. Instantiate models and run `torchinfo.summary()` with dummy tensors before training.
- **Learning Rate & Training Strategy**: Verify exploratory loops use EarlyStopping monitoring validation metrics (never loss), while production models use scheduled training (e.g., `OneCycleLR`) over fixed/learned epochs.
- **Execution Profiling**: Always track and log execution timing using `time.time()` for major time-intensive operations (training, data featurization, sweeps).
- **Standard Variable Naming**: Enforce standard variable naming conventions:
  - Target labels: `y_train`, `y_val`, `y_test`, `y_true`
  - Predictions: `pred_train`, `pred_val`, `y_pred`
  - Split indices: `split` (deserialized from `deserialize_splits`)
  - Hyperparameter dataclass instances: `hp`

---

## 2. Dynamic Mamba/Conda Environment Execution Protocol

Always execute terminal verification commands within the project's dedicated conda/mamba environment following the `mamba-environment` skill:

1. **Deduce Environment Name**: Check `environment.yml` / `environment.yaml` in the root directory for `name: <env_name>` (e.g. `name: pom`).
2. **Prefix All Commands**: Never run `python`, `ruff`, `pyrefly`, or `pytest` globally. Always execute via:
   ```bash
   conda run -n <env_name> <command>
   ```

---

## 3. Systematic Review Workflow

When triggered:
1. **Identify Changes**: Run `git status --short` and `git diff` to locate active changes.
2. **Audit Against `chemcoglab` Philosophy**: Verify every modified line against the rules in Section 1 and the `chemcoglab` skill (imports, line length, nesting depth, variable names, zero defensive noise).
3. **Automated Verification**:
   - Run linter: `conda run -n <env_name> ruff check .`
   - Run line-length check: `git diff --name-only | xargs -I {} sh -c 'awk "length(\$0) > 88" {}'`
   - Run test suite: `conda run -n <env_name> pytest`
4. **Actionable Refactoring**: Apply clean refactorings directly to fix any identified violations.
5. **Synthesis & Report**: Provide a concise, professional report of findings, refactoring changes, and test results.
