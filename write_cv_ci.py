import os
import re

with open('researchbench/evaluation/cross_validation.py', 'r', encoding='utf-8') as f:
    cv_code = f.read()

replacement = '''
    mean_val = float(np.mean(fold_scores))
    std_val = float(np.std(fold_scores))
    
    ci_lower = None
    ci_upper = None
    
    stats_config = config.get("statistics", {}).get("confidence_intervals", {}) if config else {}
    if stats_config.get("enabled", False):
        try:
            B = stats_config.get("bootstrap_samples", 1000)
            level = stats_config.get("level", 0.95)
            # Bootstrap on the folds
            boot_means = []
            for _ in range(B):
                sample = np.random.choice(fold_scores, size=len(fold_scores), replace=True)
                boot_means.append(np.mean(sample))
            alpha = 1.0 - level
            ci_lower = float(np.percentile(boot_means, alpha/2 * 100))
            ci_upper = float(np.percentile(boot_means, (1 - alpha/2) * 100))
        except:
            pass

    return {
        "metric": main_metric_name,
        "folds": fold_scores,
        "mean": mean_val,
        "std": std_val,
        "min": float(np.min(fold_scores)),
        "max": float(np.max(fold_scores)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }
'''

cv_code = cv_code.replace("def run_cross_validation(model, X, y, task: str, folds: int = 5):", "def run_cross_validation(model, X, y, task: str, folds: int = 5, config: dict = None, n_jobs: int = 1):")

# I need to apply n_jobs to the cv loop! 
# Let's completely rewrite cross_validation.py instead to use Parallel.