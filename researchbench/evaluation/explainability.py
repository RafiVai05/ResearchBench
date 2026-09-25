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

from sklearn.linear_model import Ridge, LogisticRegression

def explain_local_prediction(model_pipeline, x_instance, X_background, task):
    """
    Very lightweight local surrogate (LIME-style) to explain a single prediction.
    """
    numeric_cols = X_background.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return {}
        
    x_inst_num = x_instance[numeric_cols].fillna(0).values.reshape(1, -1)
    X_bg_num = X_background[numeric_cols].fillna(0).values
    
    # Generate perturbed samples (just Gaussian noise around the instance)
    np.random.seed(42)
    stds = np.std(X_bg_num, axis=0) + 1e-6
    n_samples = 100
    
    noise = np.random.normal(0, stds, size=(n_samples, len(numeric_cols)))
    perturbed_X = x_inst_num + noise
    
    # We need to construct DataFrame to pass to pipeline
    perturbed_df = pd.DataFrame(perturbed_X, columns=numeric_cols)
    for col in X_background.columns:
        if col not in numeric_cols:
            perturbed_df[col] = x_instance[col] # just copy categorical
            
    # Get model predictions on perturbed data
    try:
        preds = model_pipeline.predict(perturbed_df)
    except Exception:
        return {}
        
    # Fit a linear model on perturbed data to see which features drove the prediction
    if task == "classification":
        # Fit Ridge against probabilities if available, else Logistic Regression
        if hasattr(model_pipeline, "predict_proba"):
            probs = model_pipeline.predict_proba(perturbed_df)
            preds_continuous = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
            surrogate = Ridge(alpha=1.0)
            surrogate.fit(perturbed_X, preds_continuous)
            coefs = surrogate.coef_
        else:
            surrogate = LogisticRegression()
            surrogate.fit(perturbed_X, preds)
            coefs = surrogate.coef_[0]
    else:
        surrogate = Ridge(alpha=1.0)
        surrogate.fit(perturbed_X, preds)
        coefs = surrogate.coef_
        
    # Zip with feature names and sort by absolute magnitude
    feature_importance = list(zip(numeric_cols, coefs))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Return top 3
    return {k: float(v) for k, v in feature_importance[:3]}
