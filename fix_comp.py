import os

with open('researchbench/evaluation/comparison.py', 'r', encoding='utf-8') as f:
    comp = f.read()

comp = comp.replace('evaluate_models.last_processed = processed_models\\n    return results', '    evaluate_models.last_processed = processed_models\n    return results')

with open('researchbench/evaluation/comparison.py', 'w', encoding='utf-8') as f:
    f.write(comp)