import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import clone

def calculate_conformal_bounds(model_pipeline, X, y, confidence_level=0.90):
    """
    Implements true split-conformal prediction for regression.
    The calibration set is strictly isolated from model fitting.
    """
    try:
        # 1. Train / Calibration Split (e.g., 70/30)
        X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 2. Fit clone on training subset
        model_clone = clone(model_pipeline)
        model_clone.fit(X_train, y_train)
        
        # 3. Generate calibration predictions
        calib_preds = model_clone.predict(X_calib)
        
        # 4. Calculate absolute nonconformity scores
        y_calib_arr = np.array(y_calib)
        calib_preds_arr = np.array(calib_preds)
        nonconformity_scores = np.abs(y_calib_arr - calib_preds_arr)
        
        # 5. Calculate conformal finite-sample quantile
        n = len(nonconformity_scores)
        if n == 0:
            return None
            
        alpha = confidence_level
        q = min((n + 1.0) * alpha / n, 1.0)
        
        # 6. Empirical bound
        error_bound = float(np.quantile(nonconformity_scores, q))
        
        return {
            "method": "split-conformal",
            "confidence_level": float(confidence_level),
            "calibration_samples": n,
            "error_bound": error_bound
        }
    except Exception:
        return None