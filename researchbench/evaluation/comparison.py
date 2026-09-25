from .baselines import get_baseline_models
from .classification import get_classification_models, evaluate_classification_metrics
from .regression import get_regression_models, evaluate_regression_metrics
from .cross_validation import run_cross_validation
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def evaluate_models(X, y, task: str, model_names: list, folds: int = 5):
    if task == "classification":
        available = get_classification_models()
        baselines = get_baseline_models(task)
    else:
        available = get_regression_models()
        baselines = get_baseline_models(task)
        
    models_to_run = {}
    for name in model_names:
        if name in available:
            models_to_run[name] = available[name]
        else:
            raise ValueError(f"Unknown model: {name}")
            
    # Add baselines
    models_to_run.update(baselines)
    
    # Standard split for basic metrics and confusion matrix
    if task == "classification":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    results = {}
    
    for name, model in models_to_run.items():
        # Fit and evaluate
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        if task == "classification":
            probs = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            metrics, cm = evaluate_classification_metrics(y_test, preds, probs)
            results[name] = {"metrics": metrics, "confusion_matrix": cm.tolist()}
        else:
            metrics = evaluate_regression_metrics(y_test, preds)
            results[name] = {"metrics": metrics}
            
        # Run CV
        cv_res = run_cross_validation(model, X, y, task, folds=folds)
        results[name]["cv"] = cv_res
        
    return results