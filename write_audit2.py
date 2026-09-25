import os

content = '''from .leakage import check_leakage
from .imbalance import check_imbalance
from .metrics import check_metrics_suitability
from .sample_size import check_sample_size
from .reproducibility import get_reproducibility_info
from researchbench.dataset.health import audit_dataset_health
from researchbench.dataset.distribution import check_target_distribution

def perform_research_audit(df, target, task, config=None):
    health = audit_dataset_health(df, target, config)
    dist = check_target_distribution(df, target, task)
    
    has_imbalance = dist.get("is_imbalanced", False)
    has_outliers = dist.get("has_outliers", False)
    
    leakage_concerns = check_leakage(df, target)
    imbalance_concerns = check_imbalance(df, target, task, config)
    metric_concerns = check_metrics_suitability(task, has_imbalance, has_outliers)
    sample_concerns = check_sample_size(health["profile"]["num_rows"], health["profile"]["num_cols"])
    
    preproc_concerns = []
    if config and "preprocessing" in config:
        pass
    else:
        preproc_concerns.append("Preprocessing configuration not specified explicitly.")
        
    repro_info = get_reproducibility_info()
    repro_info["dataset_shape"] = [health["profile"]["num_rows"], health["profile"]["num_cols"]]
    repro_info["target"] = target
    repro_info["task"] = task
    
    # Compile findings
    all_concerns = []
    all_concerns.extend(health["concerns"])
    all_concerns.extend(leakage_concerns)
    all_concerns.extend(imbalance_concerns)
    all_concerns.extend(metric_concerns)
    all_concerns.extend(sample_concerns)
    all_concerns.extend(preproc_concerns)
    
    return {
        "health": health,
        "distribution": dist,
        "leakage_concerns": leakage_concerns,
        "imbalance_concerns": imbalance_concerns,
        "metric_concerns": metric_concerns,
        "sample_concerns": sample_concerns,
        "preproc_concerns": preproc_concerns,
        "all_concerns": list(set(all_concerns)),
        "reproducibility": repro_info
    }
'''
with open('researchbench/audit/audit.py', 'w', encoding='utf-8') as f:
    f.write(content)