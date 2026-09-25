from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit
from sklearn.base import clone
import numpy as np
from joblib import Parallel, delayed
from .classification import evaluate_classification_metrics
from .regression import evaluate_regression_metrics

def _eval_fold(model, X_train, X_test, y_train, y_test, task, main_metric_name, config=None):
    m = clone(model)
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    
    probs = None
    if task == "classification":
        probs = m.predict_proba(X_test) if hasattr(m, "predict_proba") else None
        metrics, _ = evaluate_classification_metrics(y_test, preds, probs, config)
        score = metrics[main_metric_name]
    else:
        metrics = evaluate_regression_metrics(y_test, preds, config)
        score = metrics[main_metric_name]
        
    # Feature attribution (Permutation Importance) on this fold
    attribution = None
    if config and config.get("feature_attribution", {}).get("enabled", False):
        try:
            from sklearn.inspection import permutation_importance
            n_repeats = config.get("feature_attribution", {}).get("n_repeats", 5)
            # Use appropriate scoring
            scoring = 'f1_macro' if task == 'classification' else 'neg_mean_absolute_error'
            result = permutation_importance(m, X_test, y_test, n_repeats=n_repeats, random_state=42, scoring=scoring, n_jobs=1)
            attribution = {
                "importances_mean": result.importances_mean,
                "importances_std": result.importances_std,
                "features": X_test.columns.tolist() if hasattr(X_test, "columns") else [f"feature_{i}" for i in range(X_test.shape[1])]
            }
        except Exception as e:
            attribution = {"error": str(e)}

    return {
        "score": score,
        "y_test": y_test.tolist() if hasattr(y_test, "tolist") else list(y_test),
        "probs": probs.tolist() if hasattr(probs, "tolist") and probs is not None else probs,
        "attribution": attribution
    }

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
        
    fold_results = Parallel(n_jobs=n_jobs)(
        delayed(_eval_fold)(model, *get_split(train_idx, test_idx), task, main_metric_name, config)
        for train_idx, test_idx in cv.split(X_arr, y_arr, groups=groups)
    )
            
    fold_scores = [r["score"] for r in fold_results]
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

    # Aggregate out-of-fold data for Calibration
    oof_y = []
    oof_probs = []
    has_probs = True
    for r in fold_results:
        oof_y.extend(r["y_test"])
        if r["probs"] is None:
            has_probs = False
        else:
            oof_probs.extend(r["probs"])
            
    # Aggregate feature attribution
    attribution_results = None
    if config and config.get("feature_attribution", {}).get("enabled", False):
        attr_means = []
        features = None
        for r in fold_results:
            if r["attribution"] and "importances_mean" in r["attribution"]:
                attr_means.append(r["attribution"]["importances_mean"])
                features = r["attribution"]["features"]
        if attr_means and features:
            avg_importance = np.mean(attr_means, axis=0)
            std_importance = np.std(attr_means, axis=0)
            attribution_results = {
                "features": features,
                "importances_mean": avg_importance.tolist(),
                "importances_std": std_importance.tolist()
            }

    return {
        "metric": main_metric_name,
        "folds": fold_scores,
        "mean": mean_val,
        "std": std_val,
        "min": float(np.min(fold_scores)),
        "max": float(np.max(fold_scores)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "oof_y": oof_y,
        "oof_probs": oof_probs if has_probs else None,
        "attribution": attribution_results
    }
