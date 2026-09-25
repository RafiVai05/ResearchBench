import os

with open('researchbench/evaluation/comparison.py', 'r', encoding='utf-8') as f:
    comp = f.read()

replacement = '''
    processed_models = {}
    for name, model in models_to_run.items():
        # Build preprocessing pipeline
        pipeline = build_model_pipeline(model, X, config, preprocess_mode)
        grid = config.get("models", {}).get(name, {})
        
        if grid:
            if not hasattr(pipeline, "steps"):
                # If it's just the model, strip 'model__' prefix from grid
                new_grid = {}
                for k, v in grid.items():
                    new_grid[k.replace("model__", "")] = v
                grid = new_grid
                
            pipeline = GridSearchCV(pipeline, param_grid=grid, cv=folds, n_jobs=n_jobs)
        
        processed_models[name] = pipeline
'''

import re
comp = re.sub(r'    processed_models = \{\}.*?processed_models\[name\] = pipeline', replacement.strip(), comp, flags=re.DOTALL)

with open('researchbench/evaluation/comparison.py', 'w', encoding='utf-8') as f:
    f.write(comp)