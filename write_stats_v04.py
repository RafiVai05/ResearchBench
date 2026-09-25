import os

with open('researchbench/audit/statistics.py', 'r', encoding='utf-8') as f:
    stats_code = f.read()

replacement = '''
import numpy as np
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.base import clone
from researchbench.evaluation.classification import evaluate_classification_metrics
from researchbench.evaluation.regression import evaluate_regression_metrics

def _eval_5x2cv_fold(model, X_train, X_test, y_train, y_test, task):
    m = clone(model)
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    if task == "classification":
        probs = m.predict_proba(X_test) if hasattr(m, "predict_proba") else None
        metrics, _ = evaluate_classification_metrics(y_test, preds, probs)
        return metrics["Macro F1"]
    else:
        metrics = evaluate_regression_metrics(y_test, preds)
        return metrics["MAE"]

def paired_t_test_5x2cv(model_a, model_b, X, y, task):
    X_arr = X
    y_arr = y
    
    diffs = np.zeros((5, 2))
    
    for i in range(5):
        cv = KFold(n_splits=2, shuffle=True, random_state=i*42)
        train_idx, test_idx = next(cv.split(X_arr))
        
        # Split 1
        X_train1, X_test1 = X_arr.iloc[train_idx] if hasattr(X_arr, 'iloc') else X_arr[train_idx], X_arr.iloc[test_idx] if hasattr(X_arr, 'iloc') else X_arr[test_idx]
        y_train1, y_test1 = y_arr.iloc[train_idx] if hasattr(y_arr, 'iloc') else y_arr[train_idx], y_arr.iloc[test_idx] if hasattr(y_arr, 'iloc') else y_arr[test_idx]
        
        s_a_1 = _eval_5x2cv_fold(model_a, X_train1, X_test1, y_train1, y_test1, task)
        s_b_1 = _eval_5x2cv_fold(model_b, X_train1, X_test1, y_train1, y_test1, task)
        diffs[i, 0] = s_a_1 - s_b_1
        
        # Split 2 (Reverse)
        X_train2, X_test2 = X_test1, X_train1
        y_train2, y_test2 = y_test1, y_train1
        
        s_a_2 = _eval_5x2cv_fold(model_a, X_train2, X_test2, y_train2, y_test2, task)
        s_b_2 = _eval_5x2cv_fold(model_b, X_train2, X_test2, y_train2, y_test2, task)
        diffs[i, 1] = s_a_2 - s_b_2
        
    # Variance of diffs
    variances = np.var(diffs, axis=1, ddof=1)
    
    # 5x2cv t-statistic
    numerator = diffs[0, 0]
    denominator = np.sqrt(np.sum(variances) / 5)
    
    if denominator == 0:
        t_stat = 0.0
        p_val = 1.0
    else:
        t_stat = numerator / denominator
        p_val = stats.t.sf(np.abs(t_stat), 5) * 2
        
    return float(t_stat), float(p_val), float(np.mean(diffs))

def mcnemar_test(model_a, model_b, X, y):
    m_a = clone(model_a)
    m_b = clone(model_b)
    
    # Normally we do this on a test set, but we will just use 20% holdout for the test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    m_a.fit(X_train, y_train)
    m_b.fit(X_train, y_train)
    
    preds_a = m_a.predict(X_test)
    preds_b = m_b.predict(X_test)
    
    y_test_arr = np.array(y_test)
    
    n00 = n01 = n10 = n11 = 0
    for i in range(len(y_test_arr)):
        correct_a = (preds_a[i] == y_test_arr[i])
        correct_b = (preds_b[i] == y_test_arr[i])
        
        if correct_a and correct_b: n11 += 1
        elif correct_a and not correct_b: n10 += 1
        elif not correct_a and correct_b: n01 += 1
        else: n00 += 1
        
    # McNemar stat
    b = n10
    c = n01
    
    if b + c == 0:
        stat = 0.0
        p_val = 1.0
    else:
        stat = ((np.abs(b - c) - 1.0)**2) / (b + c)
        p_val = stats.chi2.sf(stat, 1)
        
    return float(stat), float(p_val), {"both_correct": n11, "a_correct_b_wrong": n10, "a_wrong_b_correct": n01, "both_wrong": n00}

def perform_statistical_comparison(model_results: dict, config: dict, processed_models=None, X=None, y=None, task=None):
    stats_config = config.get("statistics", {})
    if not stats_config.get("enabled", False):
        return None
        
    method = stats_config.get("method", "wilcoxon").lower()
    correction = stats_config.get("correction", "holm").lower()
    
    models = list(model_results.keys())
    if len(models) < 2:
        return None
        
    reference_model = models[0]
    comparisons = []
    raw_p_values = []
    
    if method == "5x2cv" and processed_models and X is not None and y is not None:
        model_a = processed_models[reference_model]
        for m in models[1:]:
            model_b = processed_models[m]
            t_stat, p_val, diff_mean = paired_t_test_5x2cv(model_b, model_a, X, y, task)
            comparisons.append({
                "model_a": m,
                "model_b": reference_model,
                "stat": t_stat,
                "p_value_raw": p_val,
                "diff_mean": diff_mean
            })
            raw_p_values.append(p_val)
    elif method == "mcnemar" and task == "classification" and processed_models and X is not None and y is not None:
        model_a = processed_models[reference_model]
        for m in models[1:]:
            model_b = processed_models[m]
            stat, p_val, counts = mcnemar_test(model_b, model_a, X, y)
            comparisons.append({
                "model_a": m,
                "model_b": reference_model,
                "stat": stat,
                "p_value_raw": p_val,
                "diff_mean": 0.0,
                "contingency": counts
            })
            raw_p_values.append(p_val)
    elif method in ["wilcoxon", "paired_t_test"]:
        ref_folds = model_results[reference_model].get("cv", {}).get("folds", [])
        for m in models[1:]:
            m_folds = model_results[m].get("cv", {}).get("folds", [])
            if not ref_folds or not m_folds or len(ref_folds) != len(m_folds):
                continue
            try:
                if method == "paired_t_test":
                    stat, p_val = stats.ttest_rel(m_folds, ref_folds)
                else:
                    stat, p_val = stats.wilcoxon(m_folds, ref_folds)
                comparisons.append({
                    "model_a": m,
                    "model_b": reference_model,
                    "stat": float(stat),
                    "p_value_raw": float(p_val),
                    "diff_mean": float(np.mean(m_folds) - np.mean(ref_folds))
                })
                raw_p_values.append(p_val)
            except Exception:
                pass
    else:
        # Invalid / skipped
        pass
'''

stats_code = replacement + '''
    # Holm correction
    if correction == "holm" and raw_p_values:
        sorted_indices = np.argsort(raw_p_values)
        m_tests = len(raw_p_values)
        adj_p = np.zeros(m_tests)
        
        for i, idx in enumerate(sorted_indices):
            adj_val = raw_p_values[idx] * (m_tests - i)
            adj_p[idx] = min(1.0, adj_val)
            
        for i in range(1, m_tests):
            idx_prev = sorted_indices[i-1]
            idx_curr = sorted_indices[i]
            adj_p[idx_curr] = max(adj_p[idx_prev], adj_p[idx_curr])
            
        for i, comp in enumerate(comparisons):
            comp["p_value_adj"] = float(adj_p[i])
            
    return {
        "method": method,
        "correction": correction,
        "comparisons": comparisons,
        "warning": "5x2cv is the preferred valid methodological comparison." if method == "5x2cv" else "Exploratory statistical comparison. Does not establish generalized scientific supremacy."
    }
'''

with open('researchbench/audit/statistics.py', 'w', encoding='utf-8') as f:
    f.write(stats_code)