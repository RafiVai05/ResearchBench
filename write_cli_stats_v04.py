import os
import re

with open('researchbench/cli.py', 'r', encoding='utf-8') as f:
    cli = f.read()

cli = cli.replace('stats_results = perform_statistical_comparison(model_results, config)', 'stats_results = perform_statistical_comparison(model_results, config, getattr(evaluate_models, "last_processed", None), X, y, task)')

with open('researchbench/cli.py', 'w', encoding='utf-8') as f:
    f.write(cli)

# Update comparison.py to attach last_processed models
with open('researchbench/evaluation/comparison.py', 'r', encoding='utf-8') as f:
    comp = f.read()

comp = comp.replace('return results', 'evaluate_models.last_processed = processed_models\\n    return results')

with open('researchbench/evaluation/comparison.py', 'w', encoding='utf-8') as f:
    f.write(comp)