import os

with open('researchbench/evaluation/comparison.py', 'r', encoding='utf-8') as f:
    comp = f.read()

comp = comp.replace('def evaluate_models(X, y, task: str, model_names: list, config: dict = None, folds: int = 5, preprocess_mode: str = "auto"):', 'def evaluate_models(X, y, task: str, model_names: list, config: dict = None, folds: int = 5, preprocess_mode: str = "auto", n_jobs: int = 1):')
comp = comp.replace("cv_res = run_cross_validation(pipeline, X, y, task, folds=folds)", "cv_res = run_cross_validation(pipeline, X, y, task, folds=folds, config=config, n_jobs=n_jobs)")
comp = comp.replace("pipeline = GridSearchCV(pipeline, param_grid=grid, cv=folds, n_jobs=1)", "pipeline = GridSearchCV(pipeline, param_grid=grid, cv=folds, n_jobs=n_jobs)")

with open('researchbench/evaluation/comparison.py', 'w', encoding='utf-8') as f:
    f.write(comp)