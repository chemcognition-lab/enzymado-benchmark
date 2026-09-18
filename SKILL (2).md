---
name: mamba-environment
description: Dynamic conda/mamba environment detection and execution rules. Automatically identifies and executes commands (python, ruff, pytest, pip, etc.) within the repository's dedicated conda/mamba environment.
version: 2.0.1
---

# Dynamic Conda/Mamba Environment Execution Protocol

Always execute terminal commands (such as running python scripts, formatting/checking code, running tests, type checking, or installing packages) inside the repository's specific conda/mamba environment. Do not use global python or base environments.

## 1. Deducing the Environment Name

When you enter a workspace, determine the correct conda/mamba environment name by following these discovery steps in order:

1. **Check Environment Files**: Look in the root directory for configuration files:
   - `environment.yml` / `environment.yaml`: Read the `name` field (e.g. `name: pom`).
   - `pixi.toml` / `pyproject.toml`: Check if a custom environment name is defined.
2. **Check Project Rules**: Read `.agents/AGENTS.md` or other skill files in the workspace to see if an environment name is specified.
3. **Scan Active Conda Envs**: Run `conda env list` (or `mamba env list`) and look for an environment that matches the repository directory name or project name.
4. **Fallback to Active Env**: Use the environment name from the `CONDA_DEFAULT_ENV` environment variable if it matches a custom project environment.

---

## 2. Rules for Command Execution

Once the environment name (`<env_name>`) is identified:
1. **Never run commands directly**: Do not invoke `python`, `ruff`, `pytest`, or `pre-commit` globally or without environment prefixing.
2. **Always prefix commands**: Wrap execution with:
   ```bash
   conda run -n <env_name> <command>

