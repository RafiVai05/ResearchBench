import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_ood_correlation(X_train, X_val, validation_errors):
    """
    Trains an IsolationForest on X_train and scores X_val.
    Checks if validation errors correlate with anomaly scores.
    """
    if len(X_train) < 50 or len(X_val) < 20:
        return None
        
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return None
        
    X_t_num = X_train[numeric_cols].fillna(0)
    X_v_num = X_val[numeric_cols].fillna(0)
    
    iso = IsolationForest(random_state=42, contamination=0.1)
    iso.fit(X_t_num)
    
    # Anomaly scores: lower is more anomalous. We invert it so higher = anomalous.
    scores = -iso.score_samples(X_v_num)
    
    # Correlation between anomaly score and model error
    corr = np.corrcoef(scores, validation_errors)[0, 1] if len(scores) > 1 else 0.0
    
    # Calculate average error for Top 10% most anomalous vs Bottom 90%
    threshold = np.percentile(scores, 90)
    anomalous_mask = scores >= threshold
    
    err_anomalous = np.mean(validation_errors[anomalous_mask]) if np.any(anomalous_mask) else 0.0
    err_normal = np.mean(validation_errors[~anomalous_mask]) if np.any(~anomalous_mask) else 0.0
    
    return {
        "anomaly_error_correlation": float(corr),
        "mean_error_anomalous": float(err_anomalous),
        "mean_error_normal": float(err_normal),
        "ratio": float(err_anomalous / (err_normal + 1e-9))
    }
