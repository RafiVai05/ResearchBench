import numpy as np
from scipy import stats

def perform_statistical_comparison(model_results: dict, config: dict):
    stats_config = config.get("statistics", {})
    if not stats_config.get("enabled", False):
        return None
        
    method = stats_config.get("method", "wilcoxon").lower()
    correction = stats_config.get("correction", "holm").lower()
    
    # We need to compare models. Let's compare all models to a baseline, or pairwise?
    # The instruction says "Exploratory statistical comparison"
    # We'll compare each model to the first model in the dictionary (or 'Baseline')
    models = list(model_results.keys())
    if len(models) < 2:
        return None
        
    reference_model = models[0]
    ref_folds = model_results[reference_model].get("cv", {}).get("folds", [])
    
    comparisons = []
    raw_p_values = []
    
    for m in models[1:]:
        m_folds = model_results[m].get("cv", {}).get("folds", [])
        if not ref_folds or not m_folds or len(ref_folds) != len(m_folds):
            continue
            
        try:
            if method == "paired_t_test":
                stat, p_val = stats.ttest_rel(m_folds, ref_folds)
            else: # wilcoxon default
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
            
    # Holm correction
    if correction == "holm" and raw_p_values:
        sorted_indices = np.argsort(raw_p_values)
        m_tests = len(raw_p_values)
        adj_p = np.zeros(m_tests)
        
        for i, idx in enumerate(sorted_indices):
            adj_val = raw_p_values[idx] * (m_tests - i)
            adj_p[idx] = min(1.0, adj_val)
            
        # Ensure monotonic
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
        "warning": "Exploratory statistical comparison on CV folds. Does not establish generalized scientific supremacy."
    }