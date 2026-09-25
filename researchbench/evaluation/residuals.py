import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .preprocessing import build_model_pipeline
from .regression import get_regression_models

def analyze_residuals(X: pd.DataFrame, y: pd.Series, model_name: str, config: dict = None, preprocess: str = "auto"):
    available = get_regression_models()
    if model_name not in available:
        raise ValueError(f"Unknown regression model: {model_name}")
        
    model = available[model_name]
    pipeline = build_model_pipeline(model, X, config, preprocess)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    
    preds = pipeline.predict(X_test)
    residuals = y_test - preds
    
    # Calculate some basic statistics
    res_mean = np.mean(residuals)
    res_std = np.std(residuals)
    
    concerns = []
    # Simple heteroscedasticity check: 
    # split predictions into low and high halves, compare variance
    med_pred = np.median(preds)
    low_res = residuals[preds < med_pred]
    high_res = residuals[preds >= med_pred]
    
    if len(low_res) > 5 and len(high_res) > 5:
        var_low = np.var(low_res)
        var_high = np.var(high_res)
        if var_low > 0 and var_high / var_low > 2:
            concerns.append("Potential pattern detected: Residual variance appears to increase with predicted values.")
        elif var_high > 0 and var_low / var_high > 2:
            concerns.append("Potential pattern detected: Residual variance appears to decrease with predicted values.")
            
    # Check for large residuals concentrated
    # Are top 5% of absolute residuals larger than 3 standard deviations?
    abs_res = np.abs(residuals)
    if len(abs_res) > 20:
        threshold = 3 * res_std
        large_res = abs_res[abs_res > threshold]
        if len(large_res) > len(abs_res) * 0.05:
            concerns.append("Potential pattern detected: Large residuals are concentrated in a specific subset.")
            
    return {
        "preds": preds.tolist(),
        "residuals": residuals.tolist(),
        "mean": float(res_mean),
        "std": float(res_std),
        "concerns": concerns
    }