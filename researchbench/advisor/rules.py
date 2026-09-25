def apply_rules(audit_results: dict, model_results: dict, task: str) -> dict:
    observations = []
    concerns = audit_results.get("all_concerns", [])
    suggestions = []
    
    # 1. Observations from Model Results
    # Find best model for the main metric
    main_metric = "Macro F1" if task == "classification" else "MAE"
    best_model = None
    best_val = -float('inf') if task == "classification" else float('inf')
    
    for m_name, res in model_results.items():
        if "metrics" in res and main_metric in res["metrics"]:
            val = res["metrics"][main_metric]
            if task == "classification" and val > best_val:
                best_val = val
                best_model = m_name
            elif task == "regression" and val < best_val:
                best_val = val
                best_model = m_name
                
    if best_model:
        if task == "classification":
            observations.append(f"{best_model} achieved the highest {main_metric} ({best_val:.3f}) among the evaluated models.")
        else:
            observations.append(f"{best_model} achieved the lowest {main_metric} ({best_val:.3f}) among the evaluated models.")
            
    # CV stability observation
    lowest_std_model = None
    lowest_std = float('inf')
    for m_name, res in model_results.items():
        if "cv" in res and res["cv"] is not None:
            if res["cv"]["std"] < lowest_std:
                lowest_std = res["cv"]["std"]
                lowest_std_model = m_name
                
    if lowest_std_model and lowest_std_model != best_model:
        observations.append(f"{lowest_std_model} showed lower fold-to-fold variability (std: {lowest_std:.3f}).")
        
    # 2. Suggestions based on concerns
    if any("imbalance" in c.lower() for c in concerns):
        suggestions.append("Compare per-class recall and balanced accuracy before drawing conclusions from overall accuracy.")
        
    if any("leakage" in c.lower() for c in concerns):
        suggestions.append("Investigate potential leakage indicators before final modeling to ensure validity.")
        
    if any("missing" in c.lower() for c in concerns):
        suggestions.append("Check if missing values are structurally related to the target, which could introduce bias.")

    return {
        "observations": observations,
        "concerns": concerns,
        "suggestions": suggestions
    }