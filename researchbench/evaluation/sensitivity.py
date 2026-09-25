import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from researchbench.evaluation.cross_validation import run_cross_validation

def run_sensitivity_analysis(base_model, X, y, task, config):
    sensitivity_config = config.get("sensitivity_analysis", {})
    if not sensitivity_config.get("enabled", False):
        return None
        
    parameters = sensitivity_config.get("parameters", {})
    if not parameters:
        return None
        
    grid = list(ParameterGrid(parameters))
    results = []
    
    cv_folds = config.get("evaluation", {}).get("cv", {}).get("folds", 5)
    n_jobs = config.get("evaluation", {}).get("n_jobs", 1)
    
    for params in grid:
        model = clone(base_model)
        
        try:
            # Auto-prefix for pipelines
            if hasattr(model, "steps"):
                prefixed = {}
                for k, v in params.items():
                    if not k.startswith("model__") and not k.startswith("preprocessor__"):
                        prefixed[f"model__{k}"] = v
                    else:
                        prefixed[k] = v
                model.set_params(**prefixed)
            else:
                model.set_params(**params)
        except ValueError:
            continue
        
        # Run evaluation (without heavy attribution/calibration to save time)
        import copy
        fast_config = copy.deepcopy(config)
        fast_config["feature_attribution"] = {"enabled": False}
        fast_config["calibration"] = {"enabled": False}
        fast_config["statistics"] = {"enabled": False}
        
        cv_res = run_cross_validation(model, X, y, task, folds=cv_folds, config=fast_config, n_jobs=n_jobs)
        
        results.append({
            "parameters": params,
            "mean_score": cv_res["mean"],
            "std_score": cv_res["std"]
        })
        
    if not results:
        return None
        
    scores = [r["mean_score"] for r in results]
    mean_all = np.mean(scores)
    std_all = np.std(scores)
    
    # Stability Score: bounded 0 to 1 based on coefficient of variation
    if mean_all == 0:
        stability_score = 0.0
    else:
        cv = std_all / np.abs(mean_all)
        stability_score = float(np.exp(-cv))
        
    return {
        "stability_score": stability_score,
        "mean_performance": float(mean_all),
        "std_performance": float(std_all),
        "tested_configurations": len(results),
        "results": results
    }