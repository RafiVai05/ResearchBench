import os

content = '''from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.base import clone
import numpy as np
from joblib import Parallel, delayed
from .classification import evaluate_classification_metrics
from .regression import evaluate_regression_metrics

def _eval_seed(model, X, y, task, seed, main_metric_name):
    if task == "classification":
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=3, shuffle=True, random_state=seed)
        
    X_arr = X
    y_arr = y
    
    seed_folds = []
    for train_idx, test_idx in cv.split(X_arr, y_arr):
        X_train = X_arr.iloc[train_idx] if hasattr(X_arr, 'iloc') else X_arr[train_idx]
        X_test = X_arr.iloc[test_idx] if hasattr(X_arr, 'iloc') else X_arr[test_idx]
        y_train = y_arr.iloc[train_idx] if hasattr(y_arr, 'iloc') else y_arr[train_idx]
        y_test = y_arr.iloc[test_idx] if hasattr(y_arr, 'iloc') else y_arr[test_idx]
        
        m = clone(model)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        
        if task == "classification":
            probs = m.predict_proba(X_test) if hasattr(m, "predict_proba") else None
            metrics, _ = evaluate_classification_metrics(y_test, preds, probs)
            seed_folds.append(metrics[main_metric_name])
        else:
            metrics = evaluate_regression_metrics(y_test, preds)
            seed_folds.append(metrics[main_metric_name])
            
    return np.mean(seed_folds)

def run_stability_analysis(model, X, y, task: str, seeds: list, config: dict = None, n_jobs: int = 1):
    main_metric_name = "Macro F1" if task == "classification" else "MAE"
    
    seed_scores = Parallel(n_jobs=n_jobs)(
        delayed(_eval_seed)(model, X, y, task, s, main_metric_name) for s in seeds
    )
        
    return {
        "metric": main_metric_name,
        "seeds": seed_scores,
        "mean": float(np.mean(seed_scores)),
        "std": float(np.std(seed_scores)),
        "min": float(np.min(seed_scores)),
        "max": float(np.max(seed_scores))
    }
'''

with open('researchbench/evaluation/stability.py', 'w', encoding='utf-8') as f:
    f.write(content)