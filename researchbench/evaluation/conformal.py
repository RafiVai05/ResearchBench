import numpy as np

def calculate_conformal_bounds(oof_y, oof_preds, confidence_level=0.90):
    """
    Calculates the conformal prediction empirical error bound.
    """
    if not oof_y or not oof_preds:
        return None
        
    y_true = np.array(oof_y)
    y_pred = np.array(oof_preds)
    
    # Calculate absolute residuals
    residuals = np.abs(y_true - y_pred)
    
    n = len(residuals)
    # The quantile to extract is ceil((n+1)(alpha)) / n, simplified below
    alpha = confidence_level
    q = min((n + 1.0) * alpha / n, 1.0)
    
    bound = np.quantile(residuals, q)
    
    return {
        "confidence_level": float(confidence_level),
        "error_bound": float(bound)
    }
