import os

with open('researchbench/evaluation/comparison.py', 'r', encoding='utf-8') as f:
    text = f.read()

replacement = '''
    # Model Stability Comparison (v1.0.8)
    try:
        from researchbench.evaluation.stability_plot import generate_stability_plot
        stability_plot = generate_stability_plot(results)
        if stability_plot:
            results["_global_stability_plot"] = stability_plot
    except Exception:
        pass
'''

text = text.replace(replacement.strip('\\n'), '')

with open('researchbench/evaluation/comparison.py', 'w', encoding='utf-8') as f:
    f.write(text)