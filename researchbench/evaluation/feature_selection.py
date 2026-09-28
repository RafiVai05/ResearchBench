import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.base import BaseEstimator, TransformerMixin

class ResearchBenchFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, task='classification', strategy='mutual_info', k=10, variance_threshold=0.0):
        self.task = task
        self.strategy = strategy
        self.k = k
        self.variance_threshold = variance_threshold
        self.selected_indices_ = None
        self.feature_names_in_ = None
        self.dropped_features_ = []

    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
            
        if isinstance(X, pd.DataFrame):
            X_val = X.values
        else:
            X_val = np.array(X)
            
        # 1. Variance Threshold
        variances = np.var(X_val, axis=0)
        var_mask = variances > self.variance_threshold
        
        # 2. SelectKBest
        if self.strategy == 'mutual_info' and y is not None:
            if self.task == 'classification':
                score_func = mutual_info_classif
            else:
                score_func = mutual_info_regression
                
            selector = SelectKBest(score_func=score_func, k=min(self.k, sum(var_mask)))
            # Only fit on features that passed variance threshold
            if sum(var_mask) > 0:
                X_var = X_val[:, var_mask]
                selector.fit(X_var, y)
                mi_mask = selector.get_support()
                
                # Combine masks
                final_mask = np.zeros(X_val.shape[1], dtype=bool)
                final_mask[var_mask] = mi_mask
            else:
                final_mask = var_mask
        else:
            final_mask = var_mask
            
        self.selected_indices_ = np.where(final_mask)[0]
        self.dropped_features_ = [self.feature_names_in_[i] for i in range(len(final_mask)) if not final_mask[i]]
        
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_indices_]
        return X[:, self.selected_indices_]
        
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return np.array(input_features)[self.selected_indices_]