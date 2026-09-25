import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessor(X: pd.DataFrame, config: dict = None) -> ColumnTransformer:
    if config is None:
        from researchbench.config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG
    
    config = config.get("preprocessing", {})
        
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    transformers = []
    
    if num_cols:
        num_steps = []
        if config.get('numerical', {}).get('imputation') == 'median':
            num_steps.append(('imputer', SimpleImputer(strategy='median')))
        elif config.get('numerical', {}).get('imputation') == 'mean':
            num_steps.append(('imputer', SimpleImputer(strategy='mean')))
            
        if config.get('numerical', {}).get('scaling') == 'standard':
            num_steps.append(('scaler', StandardScaler()))
            
        if num_steps:
            transformers.append(('num', Pipeline(num_steps), num_cols))
        else:
            transformers.append(('num', 'passthrough', num_cols))
            
    if cat_cols:
        cat_steps = []
        if config.get('categorical', {}).get('imputation') == 'most_frequent':
            cat_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
            
        if config.get('categorical', {}).get('encoding') == 'onehot':
            cat_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
            
        if cat_steps:
            transformers.append(('cat', Pipeline(cat_steps), cat_cols))
        else:
            transformers.append(('cat', 'drop', cat_cols))
            
    if not transformers:
        return None
        
    return ColumnTransformer(transformers, remainder='drop')

def build_model_pipeline(model, X: pd.DataFrame, config: dict = None, preprocess: str = "auto") -> Pipeline:
    if preprocess == "none":
        return model
        
    preprocessor = build_preprocessor(X, config)
    if preprocessor is None:
        return model
        
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])