import numpy as np
from .classification import evaluate_classification_metrics
from .regression import evaluate_regression_metrics

def analyze_subgroups(model_results, processed_models, X, y, task, config):
    subgroups_config = config.get("subgroups", [])
    if not subgroups_config:
        return None
        
    disparity_report = {}
    
    for subgroup in subgroups_config:
        if subgroup not in X.columns:
            continue
            
        unique_vals = X[subgroup].unique()
        disparity_report[subgroup] = {}
        
        for name, pipeline in processed_models.items():
            preds = pipeline.predict(X)
            probs = pipeline.predict_proba(X) if hasattr(pipeline, "predict_proba") else None
            
            disparity_report[subgroup][name] = {}
            
            for val in unique_vals:
                idx = (X[subgroup] == val).values
                X_slice = X[idx]
                y_slice = y.iloc[idx] if hasattr(y, 'iloc') else y[idx]
                preds_slice = preds[idx]
                probs_slice = probs[idx] if probs is not None else None
                
                n_samples = len(y_slice)
                
                if n_samples == 0:
                    continue
                    
                if task == "classification":
                    try:
                        metrics, _ = evaluate_classification_metrics(y_slice, preds_slice, probs_slice, config)
                    except ValueError:
                        # Could fail if only 1 class is present in the slice
                        from sklearn.metrics import accuracy_score
                        metrics = {"Accuracy": accuracy_score(y_slice, preds_slice)}
                else:
                    metrics = evaluate_regression_metrics(y_slice, preds_slice, config)
                    
                disparity_report[subgroup][name][str(val)] = {
                    "n_samples": n_samples,
                    "metrics": metrics,
                    "warning": "Small subgroup size may cause metric instability" if n_samples < 30 else None
                }
                
    return disparity_report