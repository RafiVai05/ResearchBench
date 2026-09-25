import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

def _get_tree_rules(tree_model, feature_names, max_rules=3):
    tree_ = tree_model.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]

    rules = []
    
    def recurse(node, current_rule):
        if tree_.feature[node] != -2:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # Left child
            left_rule = current_rule + [f"{name} <= {threshold:.3f}"]
            recurse(tree_.children_left[node], left_rule)
            
            # Right child
            right_rule = current_rule + [f"{name} > {threshold:.3f}"]
            recurse(tree_.children_right[node], right_rule)
        else:
            # Leaf node
            val = tree_.value[node][0]
            if len(val) == 1:
                pred = val[0] # Regressor
            else:
                pred = np.argmax(val) # Classifier
                
            samples = tree_.n_node_samples[node]
            rules.append({
                "rule": " AND ".join(current_rule) if current_rule else "All data",
                "prediction": float(pred),
                "samples": int(samples)
            })

    recurse(0, [])
    
    # Sort rules by number of samples to find the most general ones
    rules.sort(key=lambda x: x["samples"], reverse=True)
    return rules[:max_rules]

def find_error_slices(X_val, errors):
    """
    Fits a shallow decision tree on the residuals/errors to find where the model fails.
    """
    if len(X_val) < 20:
        return []
        
    numeric_cols = X_val.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return []
        
    X_num = X_val[numeric_cols].fillna(0)
    
    # We want to find regions of HIGH error. 
    dt = DecisionTreeRegressor(max_depth=3, min_samples_leaf=max(5, int(len(X_val)*0.05)), random_state=42)
    dt.fit(X_num, errors)
    
    rules = _get_tree_rules(dt, numeric_cols, max_rules=10)
    
    # Filter for rules that have higher than average error
    mean_err = np.mean(errors)
    bad_slices = [r for r in rules if r["prediction"] > mean_err * 1.2]
    
    # Sort by worst prediction error
    bad_slices.sort(key=lambda x: x["prediction"], reverse=True)
    return bad_slices[:3]

def extract_surrogate_rules(X_train, model_preds, task):
    """
    Fits a shallow decision tree to mimic the model's predictions globally.
    """
    if len(X_train) < 20:
        return []
        
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return []
        
    X_num = X_train[numeric_cols].fillna(0)
    
    if task == "classification":
        dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(5, int(len(X_train)*0.05)), random_state=42)
    else:
        dt = DecisionTreeRegressor(max_depth=3, min_samples_leaf=max(5, int(len(X_train)*0.05)), random_state=42)
        
    dt.fit(X_num, model_preds)
    
    return _get_tree_rules(dt, numeric_cols, max_rules=5)
