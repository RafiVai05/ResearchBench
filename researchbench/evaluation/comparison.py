from .baselines import get_baseline_models
from .classification import get_classification_models, evaluate_classification_metrics
from .regression import get_regression_models, evaluate_regression_metrics
from .cross_validation import run_cross_validation
from .preprocessing import build_model_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np

def evaluate_models(X, y, task: str, model_names: list, config: dict = None, folds: int = 5, preprocess_mode: str = "auto", n_jobs: int = 1):
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
    for b_name, b_model in baselines.items():
        models_to_run[b_name] = b_model
    
    processed_models = {}
    for name, model in models_to_run.items():
        # Build preprocessing pipeline
        pipeline = build_model_pipeline(model, X, config, preprocess_mode)
        grid = config.get("models", {}).get(name, {})
        
        if grid:
            if not hasattr(pipeline, "steps"):
                # If it's just the model, strip 'model__' prefix from grid
                new_grid = {}
                for k, v in grid.items():
                    new_grid[k.replace("model__", "")] = v
                grid = new_grid
                
            pipeline = GridSearchCV(pipeline, param_grid=grid, cv=folds, n_jobs=n_jobs)
        
        processed_models[name] = pipeline

    if task == "classification":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    results = {}
    
    for name, pipeline in processed_models.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        
        if task == "classification":
            probs = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
            metrics, cm = evaluate_classification_metrics(y_test, preds, probs, config)
            results[name] = {"metrics": metrics, "confusion_matrix": cm.tolist()}
        else:
            metrics = evaluate_regression_metrics(y_test, preds, config)
            results[name] = {"metrics": metrics}
            
        if isinstance(pipeline, GridSearchCV):
            # Convert non-serializable objects if necessary
            results[name]["best_params"] = {k: str(v) for k, v in pipeline.best_params_.items()}
            

        from sklearn.model_selection import RandomizedSearchCV
        
        cv_res = None
        if config and config.get("optimization", {}).get("enabled", False):
            opt_conf = config["optimization"]
            n_trials = opt_conf.get("n_trials", 10)
            
            # Simple generic grid based on model string representation
            param_grid = {}
            m_str = str(model).lower()
            if "logisticregression" in m_str:
                param_grid = {'model__C': [0.01, 0.1, 1.0, 10.0], 'model__penalty': ['l2']}
            elif "randomforest" in m_str:
                param_grid = {'model__n_estimators': [50, 100, 200], 'model__max_depth': [None, 5, 10, 20]}
            elif "decisiontree" in m_str:
                param_grid = {'model__max_depth': [None, 3, 5, 10]}
                
            if param_grid:
                try:
                    search = RandomizedSearchCV(pipeline, param_grid, n_iter=n_trials, cv=3, random_state=42, n_jobs=n_jobs)
                    search.fit(X, y)
                    pipeline = search.best_estimator_
                except Exception:
                    pass
                    
        cv_res = run_cross_validation(pipeline, X, y, task, folds=folds, config=config, n_jobs=n_jobs)



        results[name]["cv"] = cv_res
        
        evaluate_models.last_processed = processed_models
    return results