"""Smoke-test that every source module is syntactically importable."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


def _module_name(path: Path) -> str:
    relative = path.relative_to(SRC_ROOT).with_suffix("")
    parts = ("src", *relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


MODULE_NAMES = tuple(
    sorted({_module_name(path) for path in SRC_ROOT.rglob("*.py")})
)


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_source_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)
