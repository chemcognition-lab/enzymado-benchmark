"""Compatibility facade for model blocks, wrappers, and registries.

This module used to contain every model definition in one file. The code now
lives in focused modules:

- `common.py`: shared utilities
- `nn_blocks.py`: reusable neural network blocks
- `torch_wrappers.py`: Torch training wrappers
- `xgb_wrappers.py`: XGBoost wrappers
- `registry.py`: MODEL_MAP and STRUCTURED_MODEL_MAP

Imports from `models.models` continue to work so the rest of the training
pipeline and older notebooks do not need to change all at once.
"""

from models import *  # noqa: F401,F403
