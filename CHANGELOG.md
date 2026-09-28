# Changelog

## [1.1.0] - Stable Release
### Added
- Native optional support for modern gradient boosters: XGBoost, LightGBM, and CatBoost.
- Genuine Split-Conformal Prediction for mathematically valid prediction intervals.
- Genuine Probability Calibration (Platt Scaling and Isotonic Regression) with holdout sets.
- Semantic Artifact Auditing via optional local LLMs (e.g., Ollama/llama.cpp) with structured JSON extraction.
- Interactive offline-capable Plotly visualizations replacing static base64 matplotlib PNGs.
- Advanced evaluation metrics: Matthews Correlation Coefficient (MCC), PR-AUC, and Brier Score.
- Memory safety guard for sparse-to-dense conversions preventing OOM crashes.

### Changed
- Re-architected preprocessing.py to preserve sparse matrices from NLP/categorical encoding through to estimators.
- Rewrote Auto-Balancing logic to safely inspect and target estimator configurations instead of using blind injection.
- Re-architected feature selection to ensure it strictly respects cross-validation folds and optimization boundaries.
- Adjusted report terminology to remove scientifically unverifiable claims (e.g., replacing "SOTA" and "best" with objective measurement language).

### Fixed
- Fixed critical data leakage in the probability calibration pipeline.
- Fixed severe concurrency deadlock risks related to uncontrolled nested parallelism across GridSearchCV, joblib, and boosting estimators.

### Deprecated
- Deprecated the naive Regex/OCR methodology.pdf artifact checks in favor of semantic LLM structured extraction or deterministic textual evidence.