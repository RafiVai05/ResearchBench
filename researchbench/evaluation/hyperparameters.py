import pandas as pd
from sklearn.model_selection import GridSearchCV

def apply_hyperparameter_grid(model, model_name: str, config: dict, task: str):
    if not config or 'models' not in config:
        return model
        
    model_config = config['models'].get(model_name)
    if not model_config or not isinstance(model_config, dict):
        return model
        
    # Treat the dict as the param grid. If empty, no tuning.
    if not model_config:
        return model
        
    # We must format keys for pipeline if model is a pipeline
    # Wait, the model might be a Pipeline. We will apply grid search over the pipeline!
    # If it's a pipeline, we prepend 'model__' to the grid keys.
    # To keep it simple, we do that inside the evaluation loop where we know if it's a pipeline.
    pass
