import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chisquare
import logging

def detect_drift(X_train, config):
    drift_config = config.get("drift_detection", {})
    if not drift_config.get("enabled", False):
        return None
        
    ext_path = drift_config.get("external_dataset")
    if not ext_path:
        return None
        
    try:
        X_ext = pd.read_csv(ext_path)
    except Exception as e:
        logging.warning(f"Could not load external dataset for drift detection: {e}")
        return None
        
    results = []
    p_values = []
    
    # Only test common columns
    common_cols = [c for c in X_train.columns if c in X_ext.columns]
    
    for col in common_cols:
        col_train = X_train[col].dropna()
        col_ext = X_ext[col].dropna()
        
        if pd.api.types.is_numeric_dtype(col_train):
            # KS Test
            stat, p_val = ks_2samp(col_train, col_ext)
            results.append({
                "feature": col,
                "type": "numerical",
                "test": "Kolmogorov-Smirnov",
                "statistic": float(stat),
                "p_value_raw": float(p_val)
            })
            p_values.append(p_val)
        else:
            # Chi-square test
            # Needs frequencies for common categories
            categories = set(col_train.unique()).union(set(col_ext.unique()))
            train_counts = col_train.value_counts().reindex(categories, fill_value=0)
            ext_counts = col_ext.value_counts().reindex(categories, fill_value=0)
            
            # Normalize to probabilities to compare distributions properly, or scale ext to train size
            # Standard chisquare expects expected frequencies. 
            # We treat train as expected distribution (scaled to ext size).
            expected = train_counts / train_counts.sum() * ext_counts.sum()
            
            # Add small epsilon to expected to prevent division by zero
            expected = expected + 1e-8
            
            stat, p_val = chisquare(f_obs=ext_counts, f_exp=expected)
            results.append({
                "feature": col,
                "type": "categorical",
                "test": "Chi-Square",
                "statistic": float(stat),
                "p_value_raw": float(p_val)
            })
            p_values.append(p_val)
            
    # Benjamini-Hochberg Correction
    if p_values:
        try:
            from statsmodels.stats.multitest import multipletests
            _, p_adj, _, _ = multipletests(p_values, method='fdr_bh')
            for i, r in enumerate(results):
                r["p_value_adj"] = float(p_adj[i])
                r["shifted"] = bool(p_adj[i] < 0.05)
        except ImportError:
            # Fallback if statsmodels not installed
            for i, r in enumerate(results):
                r["p_value_adj"] = r["p_value_raw"]
                r["shifted"] = bool(r["p_value_raw"] < 0.05)
                
    return {
        "external_dataset": ext_path,
        "n_features_tested": len(results),
        "n_shifted_features": sum(1 for r in results if r.get("shifted", False)),
        "details": results
    }