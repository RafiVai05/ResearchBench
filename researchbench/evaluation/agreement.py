import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

def evaluate_agreement(model_preds_dict, y_true, task):
    """
    Evaluates pairwise agreement between models and identifies failure overlaps.
    """
    models = list(model_preds_dict.keys())
    if len(models) < 2:
        return None
        
    agreements = []
    
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            m1 = models[i]
            m2 = models[j]
            preds1 = np.array(model_preds_dict[m1])
            preds2 = np.array(model_preds_dict[m2])
            y = np.array(y_true)
            
            if task == "classification":
                kappa = cohen_kappa_score(preds1, preds2)
                
                fails1 = (preds1 != y)
                fails2 = (preds2 != y)
                
                both_fail = np.sum(fails1 & fails2)
                only_m1_fails = np.sum(fails1 & ~fails2)
                only_m2_fails = np.sum(~fails1 & fails2)
                
                agreements.append({
                    "model_a": m1,
                    "model_b": m2,
                    "metric": "cohen_kappa",
                    "score": float(kappa),
                    "both_fail_count": int(both_fail),
                    "only_a_fails": int(only_m1_fails),
                    "only_b_fails": int(only_m2_fails)
                })
            else:
                # Regression correlation
                corr = np.corrcoef(preds1, preds2)[0, 1] if len(preds1) > 1 else 0
                
                err1 = np.abs(preds1 - y)
                err2 = np.abs(preds2 - y)
                
                mean_err1 = np.mean(err1)
                mean_err2 = np.mean(err2)
                
                fails1 = err1 > mean_err1
                fails2 = err2 > mean_err2
                
                both_fail = np.sum(fails1 & fails2)
                only_m1_fails = np.sum(fails1 & ~fails2)
                only_m2_fails = np.sum(~fails1 & fails2)
                
                agreements.append({
                    "model_a": m1,
                    "model_b": m2,
                    "metric": "pearson_correlation",
                    "score": float(corr),
                    "both_high_error": int(both_fail),
                    "only_a_high_error": int(only_m1_fails),
                    "only_b_high_error": int(only_m2_fails)
                })
                
    return agreements
