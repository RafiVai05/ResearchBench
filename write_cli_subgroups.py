import os
import re

with open('researchbench/cli.py', 'r', encoding='utf-8') as f:
    cli = f.read()

replacement = '''
        stats_results = perform_statistical_comparison(model_results, config, getattr(evaluate_models, "last_processed", None), X, y, task)
        
        # Subgroup analysis
        from researchbench.evaluation.subgroups import analyze_subgroups
        subgroup_results = analyze_subgroups(model_results, getattr(evaluate_models, "last_processed", None), X, y, task, config)
        
        # Dataset Fingerprint
'''

cli = re.sub(r'stats_results = perform_statistical_comparison.*?# Dataset Fingerprint', replacement.strip() + '\\n        # Dataset Fingerprint', cli, flags=re.DOTALL)

cli = cli.replace('"statistics": stats_results,', '"statistics": stats_results,\n            "subgroups": subgroup_results,')

with open('researchbench/cli.py', 'w', encoding='utf-8') as f:
    f.write(cli)