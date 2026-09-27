import os

with open('researchbench/evaluation/comparison.py', 'r', encoding='utf-8') as f:
    text = f.read()

replacement = '''
        evaluate_models.last_processed = processed_models
        
    # Model Stability Comparison (v1.0.8)
    try:
        from researchbench.evaluation.stability_plot import generate_stability_plot
        stability_plot = generate_stability_plot(results)
        if stability_plot:
            results["_global_stability_plot"] = stability_plot
    except Exception:
        pass
        
    return results
'''

text = text.replace(
    '        evaluate_models.last_processed = processed_models\n    return results',
    replacement.strip('\\n')
)

with open('researchbench/evaluation/comparison.py', 'w', encoding='utf-8') as f:
    f.write(text)