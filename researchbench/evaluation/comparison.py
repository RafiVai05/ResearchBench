from .baselines import get_baseline_models
from .classification import get_classification_models, evaluate_classification_metrics
from .regression import get_regression_models, evaluate_regression_metrics
from .cross_validation import run_cross_validation
from .preprocessing import build_model_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import pandas as pd
import numpy as np

def _apply_class_weight(estimator):
    applied = False
    if hasattr(estimator, 'class_weight'):
        try:
            setattr(estimator, 'class_weight', 'balanced')
            applied = True
        except Exception:
            pass
    elif hasattr(estimator, 'scale_pos_weight'):
        # For XGBoost binary classification, we can't easily auto-set scale_pos_weight without knowing counts here,
        # but LightGBM supports class_weight="balanced".
        pass
    return applied

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
            print(f"Unknown model: {name}")
            
    for b_name, b_model in baselines.items():
        models_to_run[b_name] = b_model
        
    balancing_info = {}
    if task == "classification":
        class_counts = pd.Series(y).value_counts(normalize=True)
        if len(class_counts) > 0 and class_counts.min() < 0.15:
            for name, model in models_to_run.items():
                balancing_info[name] = "Applied" if _apply_class_weight(model) else "Unsupported"
                
    if len(models_to_run) > 1 and config.get("ensembling", {}).get("enabled", True):
        estimators = [(name, model) for name, model in models_to_run.items() if "Baseline" not in name]
        if estimators:
            if task == "classification":
                from sklearn.ensemble import VotingClassifier
                # Use n_jobs=1 internally to prevent nested joblib deadlocks
                models_to_run["ensemble_voting"] = VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)
            else:
                from sklearn.ensemble import VotingRegressor
                models_to_run["ensemble_voting"] = VotingRegressor(estimators=estimators, n_jobs=1)
                
    processed_models = {}
    for name, model in models_to_run.items():
        # Pipeline handles dense conversion safely now
        pipeline = build_model_pipeline(model, X, config, preprocess_mode)
        processed_models[name] = pipeline

    if task == "classification":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    results = {}
    
    for name, pipeline in processed_models.items():
        opt_conf = config.get("optimization", {})
        if opt_conf.get("enabled", False) and "Baseline" not in name and "ensemble" not in name:
            n_trials = opt_conf.get("n_trials", 10)
            param_grid = {}
            m_str = str(pipeline.named_steps.get('model', '')).lower()
            if "logisticregression" in m_str:
                param_grid = {'model__C': [0.01, 0.1, 1.0, 10.0]}
            elif "randomforest" in m_str:
                param_grid = {'model__n_estimators': [50, 100, 200], 'model__max_depth': [None, 5, 10]}
            elif "xgboost" in m_str:
                param_grid = {'model__learning_rate': [0.01, 0.1, 0.2], 'model__max_depth': [3, 5, 7]}
            elif "lightgbm" in m_str:
                param_grid = {'model__learning_rate': [0.01, 0.1, 0.2], 'model__num_leaves': [31, 50, 100]}
                
            if param_grid:
                try:
                    search = RandomizedSearchCV(pipeline, param_grid, n_iter=n_trials, cv=3, random_state=42, n_jobs=n_jobs)
                    search.fit(X_train, y_train)
                    pipeline = search.best_estimator_
                except Exception:
                    pass

        try:
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
        except MemoryError as me:
            print(f"MemoryError on {name}: {me}")
            continue
        except Exception as e:
            print(f"Error on {name}: {e}")
            continue
            
        results[name] = {}
        if name in balancing_info:
            results[name]["balancing"] = balancing_info[name]
            
        if task == "classification":
            probs = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
            metrics, cm = evaluate_classification_metrics(y_test, preds, probs, config)
            results[name]["metrics"] = metrics
            results[name]["confusion_matrix"] = cm.tolist()
        else:
            metrics = evaluate_regression_metrics(y_test, preds, config)
            results[name]["metrics"] = metrics
            
        # Cross Validation inside pipeline
        cv_res = run_cross_validation(pipeline, X, y, task, folds=folds, config=config, n_jobs=n_jobs)
        results[name]["cv"] = cv_res

        evaluate_models.last_processed = processed_models
                
    return results