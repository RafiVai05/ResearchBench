import os

with open('researchbench/cli.py', 'r', encoding='utf-8') as f:
    cli = f.read()

replacement = '''
        folds = config.get("evaluation", {}).get("cv", {}).get("folds", 5)
        n_jobs = config.get("evaluation", {}).get("n_jobs", 1)
        
        from researchbench.evaluation.comparison import evaluate_models
        model_results = evaluate_models(
            X=X, 
            y=y, 
            task=task, 
            model_names=models_to_run,
            config=config,
            folds=folds,
            preprocess_mode=preprocess_mode,
            n_jobs=n_jobs
        )
'''

import re
cli = re.sub(r'folds = config.get.*?model_results = evaluate_models\([^)]*\)', replacement.strip(), cli, flags=re.DOTALL)

with open('researchbench/cli.py', 'w', encoding='utf-8') as f:
    f.write(cli)