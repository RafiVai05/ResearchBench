import pandas as pd
import numpy as np

def check_leakage(df: pd.DataFrame, target: str) -> list:
    concerns = []
    if target not in df.columns:
        return concerns
        
    y = df[target]
    X = df.drop(columns=[target])
    
    # 1. Obvious target copies (exact match)
    for col in X.columns:
        if X[col].equals(y):
            concerns.append(f"Potential leakage indicator: '{col}' is an exact copy of the target.")
            
    # 2. High correlation with target (if numeric)
    if pd.api.types.is_numeric_dtype(y):
        for col in X.select_dtypes(include=[np.number]).columns:
            corr = X[col].corr(y)
            if abs(corr) > 0.99:
                concerns.append(f"Potential leakage indicator: '{col}' has suspiciously high correlation ({corr:.3f}) with target.")
                
    return concerns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split

def profile_predictive_leakage(df: pd.DataFrame, target: str, task: str) -> list:
    concerns = []
    if target not in df.columns:
        return concerns
        
    y = df[target]
    X = df.drop(columns=[target])
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return concerns
        
    try:
        X_train, X_test, y_train, y_test = train_test_split(X[numeric_cols].fillna(0), y, test_size=0.3, random_state=42)
        
        for col in numeric_cols:
            if task == "classification" and len(np.unique(y)) == 2:
                dt = DecisionTreeClassifier(max_depth=1, random_state=42)
                dt.fit(X_train[[col]], y_train)
                preds = dt.predict_proba(X_test[[col]])[:, 1]
                score = roc_auc_score(y_test, preds)
                if score > 0.95:
                    concerns.append(f"Suspiciously high predictive power: '{col}' achieves AUC of {score:.3f} all by itself.")
            elif task == "regression":
                dt = DecisionTreeRegressor(max_depth=1, random_state=42)
                dt.fit(X_train[[col]], y_train)
                preds = dt.predict(X_test[[col]])
                score = r2_score(y_test, preds)
                if score > 0.90:
                    concerns.append(f"Suspiciously high predictive power: '{col}' achieves R2 of {score:.3f} all by itself.")
    except Exception:
        pass
        
    return concerns
