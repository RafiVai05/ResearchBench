import os

with open('researchbench/evaluation/cross_validation.py', 'r', encoding='utf-8') as f:
    cv_code = f.read()

replacement = '''from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit
from sklearn.base import clone
import numpy as np
from joblib import Parallel, delayed
from .classification import evaluate_classification_metrics
from .regression import evaluate_regression_metrics

def _eval_fold(model, X_train, X_test, y_train, y_test, task, main_metric_name, config=None):
    m = clone(model)
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    
    if task == "classification":
        probs = m.predict_proba(X_test) if hasattr(m, "predict_proba") else None
        metrics, _ = evaluate_classification_metrics(y_test, preds, probs, config)
        return metrics[main_metric_name]
    else:
        metrics = evaluate_regression_metrics(y_test, preds, config)
        return metrics[main_metric_name]

def run_cross_validation(model, X, y, task: str, folds: int = 5, config: dict = None, n_jobs: int = 1):
    strategy = config.get("evaluation", {}).get("strategy", "stratified") if config else "stratified"
    groups = None
    
    if strategy == "group_kfold":
        cv = GroupKFold(n_splits=folds)
        group_col = config.get("evaluation", {}).get("group_column")
        if group_col and group_col in X.columns:
            groups = X[group_col]
        else:
            raise ValueError(f"Group column '{group_col}' not found for GroupKFold")
    elif strategy == "time_series":
        cv = TimeSeriesSplit(n_splits=folds)
        time_col = config.get("evaluation", {}).get("time_column")
        if time_col and time_col in X.columns:
            # Sort by time_col
            sort_idx = np.argsort(X[time_col].values)
            X = X.iloc[sort_idx]
            y = y.iloc[sort_idx]
        else:
            raise ValueError(f"Time column '{time_col}' not found for TimeSeriesSplit")
    elif task == "classification":
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
        delayed(_eval_fold)(model, *get_split(train_idx, test_idx), task, main_metric_name, config)
        for train_idx, test_idx in cv.split(X_arr, y_arr, groups=groups)
    )
'''

import re
cv_code = re.sub(r'from sklearn\.model_selection.*?for train_idx, test_idx in cv\.split\(X_arr, y_arr\)\n    \)', replacement.strip(), cv_code, flags=re.DOTALL)

with open('researchbench/evaluation/cross_validation.py', 'w', encoding='utf-8') as f:
    f.write(cv_code)