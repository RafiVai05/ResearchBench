import os

with open('researchbench/evaluation/stability_plot.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace(
    "ax.boxplot(model_folds, labels=model_names, vert=False, patch_artist=True, \\n                   boxprops=dict(facecolor='lightblue', color='blue'),\\n                   medianprops=dict(color='red'))",
    "ax.boxplot(model_folds, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'), medianprops=dict(color='red'))\\n        ax.set_yticklabels(model_names)"
)

text = text.replace('import traceback\\n        traceback.print_exc()\\n        ', '')

with open('researchbench/evaluation/stability_plot.py', 'w', encoding='utf-8') as f:
    f.write(text)