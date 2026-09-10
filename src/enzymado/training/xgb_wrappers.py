"""Compatibility facade for native and custom XGBoost wrappers."""

from . import xgb_block_ec, xgb_residue, xgb_support

xgb = xgb_support.xgb
XGBClassifier = xgb_support.XGBClassifier
XGBRegressor = xgb_support.XGBRegressor
XGBoostError = xgb_support.XGBoostError
_progress_log = xgb_support._progress_log
_should_log_progress = xgb_support._should_log_progress
_estimate_gib = xgb_support._estimate_gib
_as_xgb_residue_features = xgb_support._as_xgb_residue_features
_is_xgb_cuda_oom = xgb_support._is_xgb_cuda_oom
_is_disk_write_error = xgb_support._is_disk_write_error
_is_xgb_cuda_failure = xgb_support._is_xgb_cuda_failure
_atomic_joblib_dump = xgb_support._atomic_joblib_dump
_save_xgb_checkpoint = xgb_support._save_xgb_checkpoint
_make_xgb_checkpoint_callback = xgb_support._make_xgb_checkpoint_callback
_maybe_resume_xgb_wrapper = xgb_support._maybe_resume_xgb_wrapper
XGBFlatECBlockClassifier = xgb_block_ec.XGBFlatECBlockClassifier
XGBResidueBinaryClassifier = xgb_residue.XGBResidueBinaryClassifier
