import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, PolynomialFeatures, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import logging

class SparseToDenseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, memory_limit_gb=2.0):
        self.memory_limit_gb = memory_limit_gb
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if sp.issparse(X):
            # Estimate memory
            estimated_bytes = X.shape[0] * X.shape[1] * 8 # assuming float64
            estimated_gb = estimated_bytes / (1024 ** 3)
            if estimated_gb > self.memory_limit_gb:
                raise MemoryError(
                    f"Sparse-to-dense conversion refused. Estimated memory requirement: {estimated_gb:.2f} GB. "
                    f"Configured safety threshold: {self.memory_limit_gb:.2f} GB."
                )
            return X.toarray()
        return X

def get_transformer(name: str):
    name = name.lower()
    if name == 'standard':
        return StandardScaler(with_mean=False) # sparse safe
    elif name == 'minmax':
        return MinMaxScaler() # sparse safe
    elif name == 'robust':
        return RobustScaler(with_centering=False) # sparse safe
    elif name == 'power':
        return PowerTransformer()
    elif name == 'polynomial':
        return PolynomialFeatures()
    elif name == 'log1p':
        return FunctionTransformer(np.log1p, validate=False, accept_sparse=True)
    elif name == 'median':
        return SimpleImputer(strategy='median')
    elif name == 'mean':
        return SimpleImputer(strategy='mean')
    elif name == 'most_frequent':
        return SimpleImputer(strategy='most_frequent')
    elif name == 'onehot':
        return OneHotEncoder(handle_unknown='ignore', sparse_output=True) # CHANGED
    elif name == 'tfidf':
        return TfidfVectorizer()
    elif name == 'count':
        return CountVectorizer()
    else:
        raise ValueError(f"Unknown transformer: {name}")

def build_preprocessor(X: pd.DataFrame, config: dict = None) -> ColumnTransformer:
    if config is None:
        from researchbench.config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG
    
    preproc_config = config.get("preprocessing", {})
    if not preproc_config:
        preproc_config = {}
        
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    transformers = []
    explicit_cols = set()
    
    text_conf = preproc_config.get('text', {})
    if isinstance(text_conf, dict) and 'columns' in text_conf:
        text_cols = text_conf['columns']
        vectorizer_name = text_conf.get('vectorizer', 'tfidf')
        
        for c in text_cols:
            if c in X.columns:
                transformers.append((f"text_{c}", get_transformer(vectorizer_name), c))
                explicit_cols.add(c)
                
    num_conf = preproc_config.get("numerical", {})
    if isinstance(num_conf, dict) and 'columns' in num_conf:
        for col_def in num_conf['columns']:
            names = col_def.get('names', [])
            transforms = col_def.get('transforms', [])
            if not names or not transforms: continue
            
            steps = []
            for t in transforms:
                steps.append((t, get_transformer(t)))
                
            transformers.append((f"num_explicit_{len(transformers)}", Pipeline(steps), names))
            explicit_cols.update(names)
            
    # Default numerical
    default_num_cols = [c for c in num_cols if c not in explicit_cols]
    if default_num_cols:
        default_num_conf = num_conf.get('default', num_conf) if isinstance(num_conf, dict) else num_conf
        if not default_num_conf:
            default_num_conf = {'imputation': 'median', 'scaling': 'standard'}
            
        if isinstance(default_num_conf, dict):
            num_steps = []
            if default_num_conf.get('imputation') in ['median', 'mean']:
                num_steps.append(('imputer', get_transformer(default_num_conf['imputation'])))
            if default_num_conf.get('scaling') in ['standard', 'minmax', 'robust', 'power']:
                num_steps.append(('scaler', get_transformer(default_num_conf['scaling'])))
                
            if num_steps:
                transformers.append(('num_default', Pipeline(num_steps), default_num_cols))
            else:
                transformers.append(('num_default', 'passthrough', default_num_cols))
        else:
            transformers.append(('num_default', 'passthrough', default_num_cols))

    # Auto-detect text columns if not explicitly defined
    auto_text_cols = []
    if config.get("auto_nlp", {}).get("enabled", True):
        for col in cat_cols:
            if col not in explicit_cols:
                # heuristic: high cardinality and string length
                unique_ratio = X[col].nunique() / max(1, len(X))
                if unique_ratio > 0.5:
                    sample = X[col].dropna().astype(str)
                    if len(sample) > 0 and sample.str.len().mean() > 20:
                        auto_text_cols.append(col)
                        
    for c in auto_text_cols:
        transformers.append((f"auto_text_{c}", TfidfVectorizer(max_features=50), c))
        explicit_cols.add(c)
        
    cat_conf = preproc_config.get('categorical', {})

    default_cat_cols = [c for c in cat_cols if c not in explicit_cols]
    if default_cat_cols:
        default_cat_conf = cat_conf.get('default', cat_conf) if isinstance(cat_conf, dict) else cat_conf
        if not default_cat_conf:
            default_cat_conf = {'imputation': 'most_frequent', 'encoding': 'onehot'}
            
        if isinstance(default_cat_conf, dict):
            cat_steps = []
            if default_cat_conf.get('imputation') == 'most_frequent':
                cat_steps.append(('imputer', get_transformer('most_frequent')))
            if default_cat_conf.get('encoding') == 'onehot':
                cat_steps.append(('encoder', get_transformer('onehot')))
                
            if cat_steps:
                transformers.append(('cat_default', Pipeline(cat_steps), default_cat_cols))
            else:
                transformers.append(('cat_default', 'drop', default_cat_cols))
        else:
            transformers.append(('cat_default', 'drop', default_cat_cols))
            
    if not transformers:
        return None
        
    # ALWAYS preserve sparse matrices through ColumnTransformer if possible
    return ColumnTransformer(transformers, remainder='drop', sparse_threshold=1.0)

def build_model_pipeline(model, X: pd.DataFrame, config: dict = None, preprocess: str = "auto") -> Pipeline:
    if preprocess == "none":
        return model
        
    preprocessor = build_preprocessor(X, config)
    if preprocessor is None:
        steps = []
    else:
        steps = [('preprocessor', preprocessor)]
        
    # Feature Selection (v1.1.0)
    if config and config.get("feature_selection", {}).get("enabled", False):
        from researchbench.evaluation.feature_selection import ResearchBenchFeatureSelector
        fs_conf = config.get("feature_selection", {})
        k = fs_conf.get("k", 10)
        task = config.get("task", "classification")
        steps.append(('feature_selection', ResearchBenchFeatureSelector(task=task, k=k)))
        
    # Check if model supports sparse. If not, densify safely
    # Most tree models support sparse (RF, DT, XGBoost, LightGBM, CatBoost)
    # Linear models support sparse (LogisticRegression, Ridge, Lasso)
    # We will assume modern estimators support sparse unless proven otherwise.
    # We can add SparseToDenseTransformer for specific models if needed.
    model_str = str(model).lower()
    needs_dense = any(x in model_str for x in ['gaussiannb', 'kneighbors'])
    if needs_dense:
        steps.append(('to_dense', SparseToDenseTransformer(memory_limit_gb=2.0)))
        
    steps.append(('model', model))
    
    return Pipeline(steps)