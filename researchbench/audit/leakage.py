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