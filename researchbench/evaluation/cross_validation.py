from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.base import clone
import numpy as np
from .classification import evaluate_classification_metrics
from .regression import evaluate_regression_metrics

def run_cross_validation(model, X, y, task: str, folds: int = 5):
    if task == "classification":
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=folds, shuffle=True, random_state=42)
        
    X_arr = X
    y_arr = y
    
    fold_scores = []
    main_metric_name = "Macro F1" if task == "classification" else "MAE"
    
    for train_idx, test_idx in cv.split(X_arr, y_arr):
        X_train, X_test = X_arr.iloc[train_idx] if hasattr(X_arr, 'iloc') else X_arr[train_idx], X_arr.iloc[test_idx] if hasattr(X_arr, 'iloc') else X_arr[test_idx]
        y_train, y_test = y_arr.iloc[train_idx] if hasattr(y_arr, 'iloc') else y_arr[train_idx], y_arr.iloc[test_idx] if hasattr(y_arr, 'iloc') else y_arr[test_idx]
        
        m = clone(model)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        
        if task == "classification":
            probs = m.predict_proba(X_test) if hasattr(m, "predict_proba") else None
            metrics, _ = evaluate_classification_metrics(y_test, preds, probs)
            fold_scores.append(metrics[main_metric_name])
        else:
            metrics = evaluate_regression_metrics(y_test, preds)
            fold_scores.append(metrics[main_metric_name])
            
    return {
        "metric": main_metric_name,
        "folds": fold_scores,
        "mean": float(np.mean(fold_scores)),
        "std": float(np.std(fold_scores)),
        "min": float(np.min(fold_scores)),
        "max": float(np.max(fold_scores))
    }