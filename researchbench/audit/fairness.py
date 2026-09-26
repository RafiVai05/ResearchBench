import numpy as np
import pandas as pd

def audit_fairness(df: pd.DataFrame, target: str, config: dict, oof_preds: dict = None) -> dict:
    """
    Calculates fairness metrics based on protected attributes.
    """
    fairness_res = {}
    if not config or "fairness" not in config:
        return fairness_res
        
    fair_conf = config["fairness"]
    protected_attrs = fair_conf.get("protected_attributes", [])
    privileged_classes = fair_conf.get("privileged_classes", {})
    
    if not protected_attrs:
        return fairness_res
        
    y_true = df[target].values
    
    for attr in protected_attrs:
        if attr not in df.columns:
            continue
            
        attr_data = df[attr].values
        priv_val = privileged_classes.get(attr)
        
        # If privileged class not specified, just pick the most frequent
        if priv_val is None:
            priv_val = df[attr].mode()[0]
            
        priv_mask = (attr_data == priv_val)
        unpriv_mask = ~priv_mask
        
        attr_res = {
            "privileged_class": str(priv_val),
            "base_rate_privileged": float(np.mean(y_true[priv_mask])) if np.sum(priv_mask) > 0 else 0,
            "base_rate_unprivileged": float(np.mean(y_true[unpriv_mask])) if np.sum(unpriv_mask) > 0 else 0,
            "models": {}
        }
        
        # Model specific metrics
        if oof_preds:
            for m_name, preds in oof_preds.items():
                if len(preds) != len(y_true):
                    continue
                
                preds = np.array(preds)
                
                # Disparate Impact (Ratio of Positive Predictions)
                sr_priv = np.mean(preds[priv_mask] == 1) if np.sum(priv_mask) > 0 else 0
                sr_unpriv = np.mean(preds[unpriv_mask] == 1) if np.sum(unpriv_mask) > 0 else 0
                
                di = (sr_unpriv / sr_priv) if sr_priv > 0 else 1.0
                
                # Equal Opportunity Difference (Difference in TPR)
                priv_pos = priv_mask & (y_true == 1)
                unpriv_pos = unpriv_mask & (y_true == 1)
                
                tpr_priv = np.mean(preds[priv_pos] == 1) if np.sum(priv_pos) > 0 else 0
                tpr_unpriv = np.mean(preds[unpriv_pos] == 1) if np.sum(unpriv_pos) > 0 else 0
                
                eod = tpr_unpriv - tpr_priv
                
                attr_res["models"][m_name] = {
                    "selection_rate_privileged": float(sr_priv),
                    "selection_rate_unprivileged": float(sr_unpriv),
                    "disparate_impact": float(di),
                    "equal_opportunity_difference": float(eod)
                }
                
        fairness_res[attr] = attr_res
        
    return fairness_res
