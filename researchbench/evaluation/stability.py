from sklearn.model_selection import train_test_split
from sklearn.base import clone
import numpy as np
from .classification import evaluate_classification_metrics
from .regression import evaluate_regression_metrics

def run_stability_analysis(model, X, y, task: str, seeds: list):
    X_arr = X
    y_arr = y
    
    seed_scores = []
    main_metric = "Macro F1" if task == "classification" else "MAE"
    
    for seed in seeds:
        # Use seed for both split and model random_state if applicable
        if task == "classification":
            X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=0.2, random_state=seed, stratify=y_arr)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=0.2, random_state=seed)
            
        m = clone(model)
        if hasattr(m, "random_state"):
            m.random_state = seed
            
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        
        if task == "classification":
            probs = m.predict_proba(X_test) if hasattr(m, "predict_proba") else None
            metrics, _ = evaluate_classification_metrics(y_test, preds, probs)
            seed_scores.append(metrics[main_metric])
        else:
            metrics = evaluate_regression_metrics(y_test, preds)
            seed_scores.append(metrics[main_metric])
            
    return {
        "metric": main_metric,
        "seeds": seeds,
        "scores": seed_scores,
        "mean": float(np.mean(seed_scores)),
        "std": float(np.std(seed_scores)),
        "min": float(np.min(seed_scores)),
        "max": float(np.max(seed_scores))
    }