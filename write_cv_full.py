import os

content = '''from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.base import clone
import numpy as np
from joblib import Parallel, delayed
from .classification import evaluate_classification_metrics
from .regression import evaluate_regression_metrics

def _eval_fold(model, X_train, X_test, y_train, y_test, task, main_metric_name):
    m = clone(model)
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    
    if task == "classification":
        probs = m.predict_proba(X_test) if hasattr(m, "predict_proba") else None
        metrics, _ = evaluate_classification_metrics(y_test, preds, probs)
        return metrics[main_metric_name]
    else:
        metrics = evaluate_regression_metrics(y_test, preds)
        return metrics[main_metric_name]

def run_cross_validation(model, X, y, task: str, folds: int = 5, config: dict = None, n_jobs: int = 1):
    if task == "classification":
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=folds, shuffle=True, random_state=42)
        
    X_arr = X
    y_arr = y
    
    main_metric_name = "Macro F1" if task == "classification" else "MAE"
    
    def get_split(train_idx, test_idx):
        X_train = X_arr.iloc[train_idx] if hasattr(X_arr, 'iloc') else X_arr[train_idx]
        X_test = X_arr.iloc[test_idx] if hasattr(X_arr, 'iloc') else X_arr[test_idx]
        y_train = y_arr.iloc[train_idx] if hasattr(y_arr, 'iloc') else y_arr[train_idx]
        y_test = y_arr.iloc[test_idx] if hasattr(y_arr, 'iloc') else y_arr[test_idx]
        return X_train, X_test, y_train, y_test
        
    fold_scores = Parallel(n_jobs=n_jobs)(
        delayed(_eval_fold)(model, *get_split(train_idx, test_idx), task, main_metric_name)
        for train_idx, test_idx in cv.split(X_arr, y_arr)
    )
            
    mean_val = float(np.mean(fold_scores))
    std_val = float(np.std(fold_scores))
    
    ci_lower = None
    ci_upper = None
    
    stats_config = config.get("statistics", {}).get("confidence_intervals", {}) if config else {}
    if stats_config.get("enabled", False):
        try:
            B = stats_config.get("bootstrap_samples", 1000)
            level = stats_config.get("level", 0.95)
            boot_means = []
            for _ in range(B):
                sample = np.random.choice(fold_scores, size=len(fold_scores), replace=True)
                boot_means.append(np.mean(sample))
            alpha = 1.0 - level
            ci_lower = float(np.percentile(boot_means, alpha/2 * 100))
            ci_upper = float(np.percentile(boot_means, (1 - alpha/2) * 100))
        except:
            pass

    return {
        "metric": main_metric_name,
        "folds": fold_scores,
        "mean": mean_val,
        "std": std_val,
        "min": float(np.min(fold_scores)),
        "max": float(np.max(fold_scores)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }
'''

with open('researchbench/evaluation/cross_validation.py', 'w', encoding='utf-8') as f:
    f.write(content)