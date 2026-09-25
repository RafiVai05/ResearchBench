import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error

def _get_metric(y_true, y_pred, task):
    if task == "classification":
        return accuracy_score(y_true, y_pred)
    else:
        return -mean_squared_error(y_true, y_pred) # Negative MSE so higher is better

def evaluate_robustness(model, X_test, y_test, task, config):
    """
    Tests model degradation under continuous noise and missing value injection.
    """
    robustness_config = config.get("robustness", {})
    if not robustness_config.get("enabled", False):
        return None
        
    if not hasattr(X_test, "select_dtypes"):
        return None
        
    np.random.seed(config.get("random_seed", 42))
    
    # Baseline
    base_preds = model.predict(X_test)
    base_score = _get_metric(y_test, base_preds, task)
    
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()
    
    noise_results = []
    if numeric_cols:
        noise_levels = [0.1, 0.5, 1.0, 2.0] # std dev multipliers
        stds = X_test[numeric_cols].std()
        
        for level in noise_levels:
            X_noisy = X_test.copy()
            for col in numeric_cols:
                noise = np.random.normal(0, stds[col] * level, len(X_noisy))
                X_noisy[col] += noise
            preds = model.predict(X_noisy)
            score = _get_metric(y_test, preds, task)
            noise_results.append({
                "level": level,
                "score": float(score),
                "degradation": float(base_score - score)
            })
            
    missing_results = []
    drop_fractions = [0.1, 0.2, 0.5]
    for frac in drop_fractions:
        X_missing = X_test.copy()
        for col in X_missing.columns:
            mask = np.random.rand(len(X_missing)) < frac
            X_missing.loc[mask, col] = np.nan
            
        try:
            preds = model.predict(X_missing)
            score = _get_metric(y_test, preds, task)
            missing_results.append({
                "fraction": frac,
                "score": float(score),
                "degradation": float(base_score - score)
            })
        except Exception:
            # Model cannot handle NaNs natively and no imputer in pipeline
            missing_results.append({
                "fraction": frac,
                "score": None,
                "degradation": None,
                "error": "Pipeline failed to handle NaNs"
            })
            
    # Robustness Score (Area under degradation curve roughly)
    rob_score = 1.0
    if noise_results:
        avg_deg = np.mean([r["degradation"] for r in noise_results])
        if base_score != 0:
            rob_score = max(0.0, 1.0 - (avg_deg / abs(base_score)))
            
    return {
        "baseline_score": float(base_score),
        "noise_robustness": noise_results,
        "missing_robustness": missing_results,
        "robustness_score": float(rob_score)
    }
