def generate_executive_summary(audit_results: dict, model_results: dict) -> str:
    """
    Generates a heuristic-based human-readable paragraph summarizing the experiment.
    """
    paragraphs = []
    
    # 1. Dataset Shape and Imbalance
    repro = audit_results.get("reproducibility", {})
    shape = repro.get("dataset_shape", [0, 0])
    dist = audit_results.get("distribution", {})
    is_imbalanced = dist.get("is_imbalanced", False)
    
    p1 = f"The dataset contains {shape[0]} rows and {shape[1]} columns."
    if is_imbalanced:
        p1 += " The target variable exhibits severe class imbalance, which may bias models towards the majority class."
    paragraphs.append(p1)
    
    # 2. Critical Audit Concerns (Leakage & Collinearity)
    leakage = audit_results.get("leakage_concerns", [])
    collin = audit_results.get("collinearity_concerns", [])
    schema = audit_results.get("schema_concerns", [])
    
    if schema:
        paragraphs.append(f"CRITICAL: {len(schema)} data integrity schema violations were detected. The dataset structurally violates predefined constraints.")
    if leakage:
        paragraphs.append(f"WARNING: {len(leakage)} potential target leakage concerns were identified. Features may be artificially inflating model performance by peering into the future.")
    if collin:
        paragraphs.append("Multicollinearity was detected. Several features are highly correlated, reducing the reliability of feature importance metrics.")
        
    # 3. Model Performance
    if model_results:
        best_model = None
        best_score = -9999
        main_metric = ""
        
        for name, metrics in model_results.items():
            cv = metrics.get("cv", {})
            if cv and "score" in cv:
                if cv["score"] > best_score:
                    best_score = cv["score"]
                    best_model = name
                    
        if best_model:
            task = repro.get("task", "classification")
            metric_name = "F1-Macro" if task == "classification" else "Negative MAE"
            paragraphs.append(f"The best performing baseline model was {best_model}, achieving a cross-validated {metric_name} score of {best_score:.4f}.")
            
    return " ".join(paragraphs)
