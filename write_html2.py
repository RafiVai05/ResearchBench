import os

with open('researchbench/reporting/html.py', 'r', encoding='utf-8') as f:
    html = f.read()

replacement = '''
    plots = {
        "target_dist": plot_target_distribution(audit_results["health"]["profile"].get("target_distribution", {}), target),
        "cv_boxplot": plot_cv_boxplot(model_results)
    }
    
    from .visualizer import generate_residual_plots
    if audit_results["reproducibility"]["task"] == "regression":
        # Find if any model has residuals available
        # But wait, in report we didn't calculate residuals to pass in.
        # We can just leave it out for now, the user can run esearchbench residuals independently.
        pass
'''
html = html.replace('''    plots = {
        "target_dist": plot_target_distribution(audit_results["health"]["profile"].get("target_distribution", {}), target),
        "cv_boxplot": plot_cv_boxplot(model_results)
    }
    
    # Try to generate residual plots if regression
    from .visualizer import generate_residual_plots
    if audit_results["reproducibility"]["task"] == "regression":
        # Find best model
        # Just grab the first model's residuals if available
        # Wait, we don't pass residuals in model_results currently!
        # That's fine, we will just pass empty for now unless it's explicitly run.
        pass''', replacement)

with open('researchbench/reporting/html.py', 'w', encoding='utf-8') as f:
    f.write(html)